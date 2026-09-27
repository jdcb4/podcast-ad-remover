from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image

from app.infra.database import init_db, get_db_connection
from app.web import router as web


@pytest.fixture
def client(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(web, '_reconcile_artwork_and_feeds', lambda *a: None)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='fixture-only-secret')
    app.include_router(web.router)
    return TestClient(app)


@pytest.mark.parametrize('path', ['/', '/?view=library', '/admin/ai/transcription', '/admin/ai/voice', '/admin/ai/text-analysis', '/admin/global-subscription-settings', '/admin/unified-feed', '/admin/prompts', '/admin/system', '/admin/users', '/admin/notifications', '/admin/setup'])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200, response.text
    assert 'app-sidebar' in response.text
    assert 'name="whitelist_mode"' not in response.text
    assert '>Legacy (existing behaviour)<' not in response.text


def test_scoped_system_save_preserves_access_and_api(client):
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET ai_api_enabled=1,ai_api_default_requests_per_day=1234,ip_allowlist=?', ('127.0.0.1',))
        conn.commit()
    response = client.post('/admin/system/update', data={'section':'system','concurrent_downloads':3}, follow_redirects=False)
    assert response.status_code == 303
    values = web.get_global_settings()
    assert values['ai_api_enabled'] == 1
    assert values['ai_api_default_requests_per_day'] == 1234
    assert values['ip_allowlist'] == '127.0.0.1'


def test_setup_dismiss_resume_and_finish(client):
    assert web.get_global_settings()['onboarding_status'] == 'not_started'
    client.post('/admin/setup', data={'action':'dismiss'})
    assert web.get_global_settings()['onboarding_status'] == 'dismissed'
    assert client.get('/admin/setup').status_code == 200
    client.post('/admin/setup', data={'step':1,'app_external_url':'http://podcast.local:8000'})
    assert web.get_global_settings()['onboarding_status'] == 'in_progress'
    client.post('/admin/setup', data={'step':3,'remove_ads':'true'})
    assert web.get_global_settings()['default_remove_ads'] == 1
    assert web.get_global_settings()['default_remove_intros'] == 0
    client.post('/admin/setup', data={'action':'finish'})
    assert web.get_global_settings()['onboarding_status'] == 'completed'


def test_upload_artwork_validates_and_serves_encoded_raster(client):
    image = BytesIO()
    Image.new('RGB', (128,128), 'red').save(image, 'PNG')
    response = client.post('/admin/unified-feed/update', data={'unified_feed_title':'Mine','artwork_source':'upload'}, files={'artwork_upload':('image.png',image.getvalue(),'image/png')}, follow_redirects=False)
    assert response.status_code == 303 and 'error=' not in response.headers['location']
    values = web.get_global_settings()
    filename = values['unified_feed_artwork_upload']
    assert values['unified_feed_artwork_source'] == 'upload'
    assert client.get('/artwork/unified/'+filename).headers['content-type'] == 'image/jpeg'
    response = client.post('/admin/unified-feed/update', data={'unified_feed_title':'Wrong','artwork_source':'upload'}, files={'artwork_upload':('image.svg',b'<svg/>','image/svg+xml')}, follow_redirects=False)
    assert 'error=' in response.headers['location']
    assert web.get_global_settings()['unified_feed_title'] == 'Mine'


def test_users_and_legacy_links_converge(client):
    assert client.get('/admin/feed-access', follow_redirects=False).headers['location'] == '/admin/users#feed-tokens'
    assert client.get('/admin/access-requests', follow_redirects=False).headers['location'] == '/admin/users#access-requests'
    page=client.get('/admin/users').text
    assert 'name="confirm_password"' in page
    assert 'action="/admin/api-tokens"' in page
    assert 'action="/admin/api-tokens"' not in client.get('/admin/system').text
