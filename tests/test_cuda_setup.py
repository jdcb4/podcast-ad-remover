from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.core import cuda_runtime as cuda, cuda_setup as setup
from app.core.config import settings
from app.infra.database import init_db, get_db_connection


@pytest.fixture
def configured(isolated_data_dir, monkeypatch):
    init_db()
    tasks = []
    class Thread:
        def __init__(self, target, args, **kwargs): self.target, self.args = target, args
        def start(self): tasks.append(lambda: self.target(*self.args))
    monkeypatch.setattr(setup.threading, 'Thread', Thread)
    monkeypatch.setattr(cuda, 'install_bundle', lambda: None)
    monkeypatch.setattr(cuda, 'installed', lambda: True)
    from app.core import cuda_client
    monkeypatch.setattr(cuda_client, 'probe', lambda runtime: {'supported': ['float32', 'float16'], 'precision': runtime.get('whisper_cuda_compute_type') or 'float16'})
    return tasks


def test_cpu_startup_never_downloads(configured, monkeypatch):
    monkeypatch.setattr(settings, 'CUDA_SETUP', False)
    setup.startup()
    assert not configured
    assert setup.preferences()['whisper_device'] == 'cpu'


def test_bootstrap_success_and_manual_cpu_precedence(configured, monkeypatch):
    monkeypatch.setattr(settings, 'CUDA_SETUP', True)
    setup.startup(); configured.pop()()
    assert setup.preferences()['whisper_device'] == 'cuda'
    assert setup.ready(setup.preferences())
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET whisper_device='cpu', whisper_choice_version=1")
        conn.commit()
    setup.startup()
    assert not configured
    assert setup.preferences()['whisper_device'] == 'cpu'


def test_newer_manual_choice_wins_setup_race(configured):
    assert setup.start_setup()
    assert not setup.start_setup()  # Serialized across requests/processes.
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET whisper_choice_version=whisper_choice_version+1, whisper_model='small'")
        conn.commit()
    configured.pop()()
    assert setup.preferences()['whisper_device'] == 'cpu'
    assert setup.preferences()['whisper_model'] == 'small'


def test_failed_setup_preserves_preferences_and_retry_unlocks(configured, monkeypatch):
    def fail(): raise RuntimeError('No NVIDIA device exposed')
    monkeypatch.setattr(cuda, 'install_bundle', fail)
    assert setup.start_setup(); configured.pop()()
    assert setup.preferences()['whisper_device'] == 'cpu'
    assert cuda.state()['phase'] == 'failed'
    assert not cuda.state()['probe_requested']
    assert setup.start_setup(); configured.pop()()


def test_requested_gpu_revalidated_with_flag_off(configured, monkeypatch):
    assert setup.start_setup(); configured.pop()()
    monkeypatch.setattr(settings, 'CUDA_SETUP', False)
    setup.startup()
    assert not setup.ready(setup.preferences())
    configured.pop()()
    assert setup.ready(setup.preferences())


def test_admin_settings_validation_and_other_page_preservation(configured):
    from app.web import router as web
    app = FastAPI(); app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.include_router(web.router)
    with TestClient(app) as client:
        assert client.post('/admin/ai/update', data={'section': 'ai_transcription', 'whisper_device': 'cuda'}).status_code == 400
        assert client.post('/admin/ai/update', data={'section': 'ai_transcription', 'whisper_compute_type': 'made-up'}).status_code == 400
        response = client.post('/admin/ai/cuda/setup', headers={'Origin': 'https://evil.test'}, follow_redirects=False)
        assert response.status_code == 403
        response = client.post('/admin/ai/cuda/setup', follow_redirects=False)
        assert response.status_code == 303
        configured.pop()()
        before = setup.preferences()
        client.post('/admin/ai/update', data={'section': 'ai_voice'})
        after = setup.preferences()
        assert all(before[key] == after[key] for key in ('whisper_device', 'whisper_compute_type', 'whisper_cuda_compute_type'))
        response = client.get('/admin/ai/transcription')
        assert response.status_code == 200
        assert 'Set up / retest GPU acceleration' in response.text
        assert 'cuda-setup.js' in response.text
        with get_db_connection() as conn:
            conn.execute('UPDATE app_settings SET auth_enabled=1'); conn.commit()
        assert client.post('/admin/ai/cuda/setup').status_code == 401


def test_gpu_failure_retries_cpu_once_and_quarantines(configured, monkeypatch):
    from app.core.ai_services import Transcriber
    from app.core import cuda_client
    assert setup.start_setup(); configured.pop()()
    calls = []
    class Worker:
        def request(self, *args):
            calls.append('gpu'); raise cuda_client.GpuFailure('out of GPU memory')
        def dispose(self): pass
    monkeypatch.setattr(cuda_client, 'GpuWorker', Worker)
    t = Transcriber()
    def cpu(audio, callback, runtime):
        calls.append(runtime['whisper_compute_type']); return {'segments': [], 'text': ''}
    monkeypatch.setattr(t, '_transcribe_local', cpu)
    assert t.transcribe('fixture')['_execution']['device'] == 'cpu'
    t.transcribe('fixture')
    assert calls == ['gpu', 'float32', 'float32']
    assert cuda.state()['disabled']


def test_cancellation_does_not_trigger_cpu_retry(configured, monkeypatch):
    from app.core.ai_services import Transcriber
    from app.core import cuda_client
    assert setup.start_setup(); configured.pop()()
    class Worker:
        def request(self, *args): raise RuntimeError('CancelledByUser')
        def dispose(self): pass
    monkeypatch.setattr(cuda_client, 'GpuWorker', Worker)
    t = Transcriber()
    monkeypatch.setattr(t, '_transcribe_local', lambda *args: pytest.fail('CPU retry after cancellation'))
    with pytest.raises(RuntimeError, match='CancelledByUser'):
        t.transcribe('fixture')
