import subprocess
import sys

import pytest

from app.core import cuda_client as client
from app.infra.database import init_db


@pytest.fixture
def worker(isolated_data_dir, monkeypatch):
    init_db()
    # Exercise real pipe/process/lock cleanup without requiring an NVIDIA device.
    real_popen = subprocess.Popen
    code = """import sys,json,time,os
for line in sys.stdin:
 request=json.loads(line)
 if request['action']=='crash': os._exit(2)
 if request['action']=='hang': time.sleep(60)
 print(json.dumps({'result': {'pid': os.getpid()}}),flush=True)
"""
    monkeypatch.setattr(client.subprocess, 'Popen', lambda args, **kwargs: real_popen([sys.executable, '-u', '-c', code], **kwargs))
    monkeypatch.setattr(client.GpuWorker, '_idle_watch', lambda self: None)
    obj = client.GpuWorker()
    yield obj
    obj.dispose()


def test_worker_reused_then_crash_releases_gpu(worker):
    a = worker.request({'action': 'ok'})
    assert worker.request({'action': 'ok'}) == a
    with pytest.raises(client.GpuFailure, match='exited'):
        worker.request({'action': 'crash'})
    assert worker.process is None
    assert worker.gpu_lock is None
    assert worker.request({'action': 'ok'}) != a


def test_native_hang_can_be_cancelled(worker):
    def cancel(*args): raise RuntimeError('CancelledByUser')
    with pytest.raises(RuntimeError, match='CancelledByUser'):
        worker.request({'action': 'hang'}, cancel)
    assert worker.process is None
    assert worker.gpu_lock is None


def test_worker_timeout_is_bounded(worker, monkeypatch):
    monkeypatch.setattr(client, 'OPERATION_TIMEOUT', 0.1)
    with pytest.raises(client.GpuFailure, match='limit'):
        worker.request({'action': 'hang'})
    assert worker.process is None
def test_container_pid_one_is_a_valid_worker_parent(monkeypatch, capsys):
    import io
    import json
    from types import SimpleNamespace
    from app.core import cuda_worker, transcription_settings
    monkeypatch.setattr(cuda_worker, 'sys', SimpleNamespace(platform='linux', stdin=io.StringIO(json.dumps({
        'action': 'invalid', 'runtime': {'whisper_cuda_compute_type': 'float16'}}) + '\n')))
    monkeypatch.setattr(cuda_worker, 'signal', SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(cuda_worker.os, 'getppid', lambda: 1)
    monkeypatch.setattr(cuda_worker.ctypes, 'CDLL', lambda *args: SimpleNamespace(prctl=lambda *args: 0))
    monkeypatch.setattr(transcription_settings, 'supported_compute_types', lambda device: ['float16'])
    cuda_worker.main()
    assert json.loads(capsys.readouterr().out)['error'] == 'Unknown GPU worker action'
