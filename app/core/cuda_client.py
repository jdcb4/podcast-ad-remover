"""Isolate native GPU failures from the scheduler and HTTP server."""
import atexit
import json
import os
import queue
import subprocess
import signal
import sys
import threading
import time

from filelock import Timeout
from app.core import cuda_runtime as cuda

OPERATION_TIMEOUT = 7200

class GpuFailure(RuntimeError):
    pass


class GpuWorker:
    def __init__(self):
        self.process = None
        self.guard = threading.RLock()
        self.gpu_lock = None
        self.last_progress = (0, 0)
        self.busy = False
        self.stopped = threading.Event()
        self.watcher = threading.Thread(target=self._idle_watch, daemon=True)
        self.watcher.start()
        atexit.register(self.close)

    def _idle_watch(self):
        while not self.stopped.wait(1):
            try:
                if self.process and not self.busy:
                    from app.core.cuda_setup import preferences
                    if cuda.state().get('probe_requested') or preferences().get('whisper_device') != 'cuda':
                        self.close()
            except Exception:
                pass  # Database may be closing during application shutdown.

    def close(self):
        with self.guard:
            if self.process:
                if sys.platform == 'linux':
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self.process.kill()
                self.process.wait(timeout=10)
                for stream in (self.process.stdin, self.process.stdout):
                    try:
                        stream.close()
                    except OSError:
                        pass
                self.process = None
            if self.gpu_lock:
                self.gpu_lock.release()
                self.gpu_lock = None

    def dispose(self):
        self.close()
        self.stopped.set()
        atexit.unregister(self.close)

    def _start(self, callback):
        self.gpu_lock = cuda.lock('gpu')
        deadline = time.monotonic() + OPERATION_TIMEOUT
        while True:
            try:
                self.gpu_lock.acquire()
                break
            except Timeout:
                if time.monotonic() > deadline:
                    raise GpuFailure('Timed out waiting for the GPU worker')
                if callback:
                    callback(0, 0)
                time.sleep(1)
        try:
            self.process = subprocess.Popen([sys.executable, '-m', 'app.core.cuda_worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', env=cuda.worker_environment(), bufsize=1,
                start_new_session=sys.platform == 'linux')
        except OSError as exc:
            raise GpuFailure('Could not start GPU worker') from exc
        self.messages = queue.Queue()
        process, messages = self.process, self.messages
        def read():
            try:
                for line in process.stdout:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        messages.put({'error': 'Invalid response from GPU worker'})
            finally:
                messages.put({'error': 'GPU worker exited unexpectedly'})
        threading.Thread(target=read, daemon=True).start()

    def request(self, command, callback=None):
        with self.guard:
            self.busy = True
            self.last_progress = (0, 0)
            try:
                if not self.process:
                    self._start(callback)
                try:
                    self.process.stdin.write(json.dumps(command) + '\n')
                    self.process.stdin.flush()
                except (OSError, ValueError) as exc:
                    raise GpuFailure('GPU worker connection failed') from exc
                deadline = time.monotonic() + OPERATION_TIMEOUT
                while time.monotonic() < deadline:
                    try:
                        response = self.messages.get(timeout=1)
                    except queue.Empty:
                        response = {}
                    if 'error' in response:
                        raise GpuFailure(response['error'])
                    if 'result' in response:
                        return response['result']
                    if 'progress' in response:
                        self.last_progress = response['progress']
                    if callback:
                        callback(*self.last_progress)  # Cancellation remains responsive during native inference.
                raise GpuFailure('GPU operation exceeded the two-hour limit')
            except BaseException:
                self.close()
                raise
            finally:
                self.busy = False


def probe(runtime):
    worker = GpuWorker()
    try:
        return worker.request({'action': 'probe', 'runtime': dict(runtime)})
    finally:
        worker.dispose()
