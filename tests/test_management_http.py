from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.api import subscriptions, audio_routes
from app.infra.database import init_db, get_db_connection
from app.core.config import settings
from app.web import router as web
from app.web.auth import require_auth, auth_middleware


@pytest.fixture
def populated_client(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (id,username,password_hash) VALUES (91,'owner','unused'), (92,'reader','unused')")
        conn.execute("INSERT INTO subscriptions (id,feed_url,title,slug,owner_user_id) VALUES (90,'https://example.com/feed','Show','show',91)")
        conn.execute("INSERT INTO episodes (id,subscription_id,guid,title,original_url,status) VALUES (90,90,'one','One','https://example.com/one','completed'), (91,90,'two','Two','https://example.com/two','completed')")
        conn.commit()
    app = FastAPI()
    app.include_router(web.router)
    app.include_router(subscriptions.router, prefix='/api')
    app.include_router(audio_routes.router)
    app.dependency_overrides[require_auth] = lambda: SimpleNamespace(id=92, is_admin=False)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize('method,path', [
    ('DELETE', '/api/episodes/90'), ('POST', '/api/episodes/90/cancel'),
    ('POST', '/api/episodes/90/process'), ('POST', '/api/episodes/90/reprocess'),
    ('POST', '/api/episodes/90/ignore'), ('POST', '/episodes/90/download'),
])
def test_non_owner_cannot_mutate_shared_episode(populated_client, method, path):
    assert populated_client.request(method, path).status_code == 403
    with get_db_connection() as conn:
        assert conn.execute('SELECT status FROM episodes WHERE id=90').fetchone()[0] == 'completed'


def test_membership_json_and_form_use_real_repository(populated_client):
    for action, member in [('add', True), ('add', True), ('remove', False), ('remove', False)]:
        result = populated_client.post('/subscriptions/90/library', data={'action': action}, headers={'Accept': 'application/json'})
        assert result.status_code == 200
        assert result.json()['in_user_library'] is member
        assert result.json()['user_library_count'] == int(member)
    result = populated_client.post('/subscriptions/90/library', data={'action': 'add'}, follow_redirects=False)
    assert result.status_code == 303


def test_audio_tracking_uses_complete_path_and_invalid_range_is_client_error(populated_client):
    from pathlib import Path
    for episode, name in [(90, 'one'), (91, 'two')]:
        path = Path(settings.PODCASTS_DIR) / 'show' / name / 'processed.mp3'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'audio fixture')
        with get_db_connection() as conn:
            conn.execute('UPDATE episodes SET local_filename=? WHERE id=?', (str(path), episode))
            conn.commit()
    response = populated_client.get('/audio/show/two/processed.mp3', headers={'Range': 'bytes=garbage-'})
    assert response.status_code in (400, 416)
    assert populated_client.get('/audio/show/two/processed.mp3').status_code == 200
    with get_db_connection() as conn:
        assert [r[0] for r in conn.execute('SELECT listen_count FROM episodes WHERE id IN (90,91) ORDER BY id')] == [0, 1]


def test_ip_denial_crosses_real_middleware_stack_as_403(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET ip_allowlist='192.0.2.1' WHERE id=1")
        conn.commit()
    app = FastAPI()
    app.middleware('http')(auth_middleware)
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.get('/')(lambda: {'ok': True})
    assert TestClient(app, raise_server_exceptions=False).get('/').status_code == 403
