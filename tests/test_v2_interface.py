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
    assert web.get_global_settings()['app_external_url'] != 'http://podcast.local:8000'
    client.post('/admin/setup', data={'step':4,'remove_ads':'true'})
    assert 'Review changes' in client.get('/admin/setup?step=5').text
    client.post('/admin/setup', data={'action':'finish'})
    assert web.get_global_settings()['onboarding_status'] == 'completed'
    assert web.get_global_settings()['app_external_url'] == 'http://podcast.local:8000'
    assert web.get_global_settings()['default_remove_intros'] == 0


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


def test_optional_podcast_guidance_can_be_cleared(client):
    from app.core.models import SubscriptionCreate
    sub = web.sub_repo.create(SubscriptionCreate(feed_url='https://example.com/optional.xml'), 'Optional', 'optional')
    response = client.post(f'/subscriptions/{sub.id}/settings', data={'remove_ads':'true', 'custom_instructions':''}, follow_redirects=False)
    assert response.status_code == 303
    assert 'error=' not in response.headers['location']
    assert web.sub_repo.get_by_id(sub.id).custom_instructions is None
    response = client.post(f'/subscriptions/{sub.id}/settings', data={'append_summary':'true'}, follow_redirects=False)
    assert response.status_code == 422


def test_preview_provider_transport_failure_is_actionable(client, monkeypatch):
    import httpx
    import app.core.speech as speech
    async def unavailable(*args, **kwargs):
        raise httpx.ConnectError('private endpoint details')
    monkeypatch.setattr(speech, 'generate_speech', unavailable)
    response = client.post('/admin/ai/voice/preview', headers={'Origin':'http://testserver'})
    assert response.status_code == 502
    assert 'Check the endpoint' in response.text
    assert 'private endpoint' not in response.text


def test_setup_cancel_discards_and_apply_detects_concurrent_edits(client):
    original = web.get_global_settings()['app_external_url']
    client.post('/admin/setup', data={'step':1,'app_external_url':'http://draft.local'})
    client.post('/admin/setup', data={'action':'dismiss'})
    assert web.get_global_settings()['app_external_url'] == original
    client.post('/admin/setup', data={'step':1,'app_external_url':'http://draft.local'})
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET app_external_url='http://changed.local' WHERE id=1")
        conn.commit()
    result = client.post('/admin/setup', data={'action':'finish'})
    assert result.status_code == 400 and 'changed elsewhere' in result.text
    assert web.get_global_settings()['app_external_url'] == 'http://changed.local'


def test_setup_error_retains_values_and_ai_can_be_skipped(client):
    result = client.post('/admin/setup', data={'step':2,'provider':'custom','model':'my-model','base_url':'invalid'})
    assert result.status_code == 400 and 'value="my-model"' in result.text
    result = client.post('/admin/setup', data={'step':2,'action':'skip'}, follow_redirects=False)
    assert result.headers['location'].endswith('step=3')
    client.post('/admin/setup', data={'step':3,'whisper_model':'small','whisper_device':'cpu'})
    client.post('/admin/setup', data={'step':4,'retention':'standard'})
    client.post('/admin/setup', data={'action':'finish'})
    assert web.get_global_settings()['whisper_model'] == 'small'
    assert web.get_global_settings()['default_retention_limit'] == 1
