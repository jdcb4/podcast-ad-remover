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
