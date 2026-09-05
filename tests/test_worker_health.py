import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.worker_health import worker_status, record_heartbeat, record_feed_check
from app.core.provider_readiness import provider_configuration_error
from app.infra.database import get_db_connection, init_db


def test_health_distinguishes_disabled_starting_healthy_stale(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, 'PROCESSOR_ENABLED', False)
    assert worker_status()['state'] == 'disabled'
    monkeypatch.setattr(settings, 'PROCESSOR_ENABLED', True)
    assert worker_status()['state'] == 'starting'
    record_heartbeat('test-worker')
    record_feed_check(60)
    first = worker_status()
    assert first['state'] == 'healthy'
    assert worker_status()['next_feed_check'] == first['next_feed_check']
    with get_db_connection() as conn:
        conn.execute("UPDATE worker_status SET heartbeat_at=datetime(CURRENT_TIMESTAMP,'-5 minutes')")
        conn.commit()
    assert worker_status()['state'] == 'stale'
    from app.main import app
    assert TestClient(app, raise_server_exceptions=False).get('/health').status_code == 503
    monkeypatch.setattr(settings, 'PROCESSOR_ENABLED', False)
    # A stopped child from an earlier lifespan must not degrade a disabled worker.
    monkeypatch.setattr(app.state, 'processor_process', SimpleNamespace(is_alive=lambda: False), raising=False)
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET auth_enabled=1')
        conn.commit()
    assert TestClient(app, raise_server_exceptions=False).get('/health').status_code == 200


@pytest.mark.asyncio
async def test_supervisor_restarts_dead_child_with_backoff(monkeypatch):
    from app.main import supervise_processor
    delays, replacements = [], []
    async def sleep(delay):
        delays.append(delay)
        if len(delays) == 3:
            raise asyncio.CancelledError()
    child = SimpleNamespace(is_alive=lambda: False, join=lambda **kw: None, exitcode=1)
    app = SimpleNamespace(state=SimpleNamespace(processor_process=child))
    def factory():
        replacements.append(True)
        return SimpleNamespace(is_alive=lambda: True, start=lambda: None)
    monkeypatch.setattr('app.main.asyncio.sleep', sleep)
    with pytest.raises(asyncio.CancelledError):
        await supervise_processor(app, factory)
    assert len(replacements) == 1
    assert delays == [5, 10, 5]


def test_custom_keyless_readiness_and_initial_whisper_setting(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(settings, 'WHISPER_MODEL', 'tiny')
    init_db()
    with get_db_connection() as conn:
        assert conn.execute('SELECT whisper_model FROM app_settings').fetchone()[0] == 'tiny'
    assert provider_configuration_error({'active_ai_provider':'custom', 'custom_llm_base_url':'http://localhost:1234/v1', 'custom_llm_model':'local-model'}) is None
    assert provider_configuration_error({'active_ai_provider':'custom', 'custom_llm_model':'local-model'})
