import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.core import timeline
from app.core.processor import Processor
from app.infra.database import get_db_connection, init_db
from app.infra.repository import JobRepository, SubscriptionRepository
from app.web import router as web


@pytest.fixture
def client(isolated_data_dir, monkeypatch):
    init_db()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug,custom_instructions) VALUES(1,'https://example.com/feed','Show','show','Preserve musical demonstrations')")
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status) VALUES(1,1,'guid','Episode','https://example.com/source','pending')")
        conn.execute("UPDATE app_settings SET ad_prompt_base='Saved legacy {targets} {custom_instr}', openai_api_key='test-secret-must-not-render', whitelist_mode=1")
        conn.commit()
    async def noop(*args, **kwargs):
        pass
    for name in ('cleanup_old_episodes', 'check_feeds', 'process_queue'):
        monkeypatch.setattr(Processor, name, noop)
    monkeypatch.setattr(web, '_reconcile_artwork_and_feeds', lambda *args, **kwargs: None)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.include_router(web.router)
    return TestClient(app)


def test_rules_page_preview_and_save_are_separate_from_opt_in(client):
    page = client.get('/admin/prompts')
    assert page.status_code == 200
    assert 'Saved legacy {targets} {custom_instr}' in page.text
    assert 'test-secret-must-not-render' not in page.text
    for label in timeline.DEFINITIONS:
        assert f'name="definition_{label}"' in page.text
    data = {'definition_Outro': 'Only generic closing housekeeping.', 'timeline_output_mode': 'strict', 'preview_subscription_id': '1'}
    preview = client.post('/admin/prompts/timeline/preview', data=data)
    assert preview.status_code == 200
    assert 'Preserve musical demonstrations' in preview.json()['system_prompt']
    assert 'Only generic closing housekeeping.' in preview.json()['system_prompt']
    assert 'test-secret-must-not-render' not in preview.text
    with get_db_connection() as conn:
        assert conn.execute('SELECT timeline_definitions FROM app_settings').fetchone()[0] is None
    result = client.post('/admin/prompts/timeline', data=data)
    assert result.status_code == 200
    with get_db_connection() as conn:
        saved = dict(conn.execute('SELECT * FROM app_settings').fetchone())
        assert saved['default_processing_workflow'] == 'legacy'
        assert saved['whitelist_mode'] == 1
        assert saved['ad_prompt_base'] == 'Saved legacy {targets} {custom_instr}'
        assert json.loads(saved['timeline_definitions']) == {'Outro': 'Only generic closing housekeeping.'}
        assert saved['timeline_output_mode'] == 'strict'
    assert SubscriptionRepository().get_by_id(1).processing_workflow == 'legacy'


def test_default_reset_is_scoped_and_invalid_output_mode_does_not_write(client):
    client.post('/admin/prompts/timeline', data={'definition_Intro': 'Opening housekeeping only.', 'definition_Ad': 'Paid messages.'})
    assert client.post('/admin/prompts/timeline', data={'definition_Intro': '', 'timeline_output_mode': 'invalid'}).status_code == 400
    client.post('/admin/prompts/timeline', data={'definition_Intro': ''})
    with get_db_connection() as conn:
        assert json.loads(conn.execute('SELECT timeline_definitions FROM app_settings').fetchone()[0]) == {'Ad': 'Paid messages.'}


def test_podcast_opt_in_zero_threshold_and_existing_queued_job_are_preserved(client):
    jobs = JobRepository()
    jobs.enqueue(1)
    data = {'timeline_settings_present': 'true', 'processing_workflow': 'complete_timeline',
            'remove_ads': 'true', 'remove_intros': 'true', 'remove_outros': 'true',
            'minimum_retained_seconds': '0', 'inherit_custom_instructions': 'true'}
    result = client.post('/subscriptions/1/settings', data=data, follow_redirects=False)
    assert result.status_code == 303
    sub = SubscriptionRepository().get_by_id(1)
    assert sub.processing_workflow == 'complete_timeline' and not sub.inherit_processing_workflow
    assert sub.minimum_retained_seconds == 0 and not sub.remove_non_editorial_non_speech
    with get_db_connection() as conn:
        assert json.loads(conn.execute('SELECT processing_snapshot FROM jobs').fetchone()[0])['workflow'] == 'legacy'
    assert client.get('/subscriptions/1').status_code == 200
    # An older form omits new fields and must not reset them.
    assert client.post('/subscriptions/1/settings', data={'remove_ads': 'true', 'inherit_custom_instructions': 'true'}, follow_redirects=False).status_code == 303
    after = SubscriptionRepository().get_by_id(1)
    assert after.processing_workflow == 'complete_timeline' and after.minimum_retained_seconds == 0


def test_global_default_does_not_convert_existing_podcasts_and_old_form_preserves_it(client):
    result = client.post('/admin/global-subscription-settings/update', data={
        'default_processing_workflow': 'complete_timeline', 'timeline_settings_present': 'true',
        'default_minimum_retained_seconds': '0', 'whitelist_mode': 'true'}, follow_redirects=False)
    assert result.status_code == 303
    assert SubscriptionRepository().get_by_id(1).processing_workflow == 'legacy'
    assert client.get('/admin/global-subscription-settings').status_code == 200
    result = client.post('/admin/global-subscription-settings/update', data={'default_remove_ads': 'true', 'whitelist_mode': 'true'}, follow_redirects=False)
    assert result.status_code == 303
    with get_db_connection() as conn:
        row = conn.execute('SELECT default_processing_workflow,default_minimum_retained_seconds,whitelist_mode FROM app_settings').fetchone()
        assert tuple(row) == ('complete_timeline', 0, 1)


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf', '601'])
def test_invalid_thresholds_are_rejected_without_changes(client, value):
    result = client.post('/admin/global-subscription-settings/update', data={'default_minimum_retained_seconds': value})
    assert result.status_code == 400
    with get_db_connection() as conn:
        assert conn.execute('SELECT default_minimum_retained_seconds FROM app_settings').fetchone()[0] == 10


def test_prompt_endpoints_require_admin(client, monkeypatch):
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET auth_enabled=1')
        conn.commit()
    monkeypatch.setattr('app.web.auth.get_current_user', lambda request: SimpleNamespace(id=99, is_admin=False))
    for path in ('/admin/prompts/timeline', '/admin/prompts/timeline/preview'):
        assert client.post(path, data={}).status_code == 403


def test_report_escapes_model_text_and_keeps_extra_cuts_distinct():
    from app.core.reports import render_ad_report
    rows = [{'first_id': 1, 'last_id': 1, 'start': 0, 'end': 5, 'label': 'Ad', 'reason': '<script>bad()</script>'},
            {'first_id': 2, 'last_id': 2, 'start': 5, 'end': 8, 'label': 'Content', 'reason': 'Substantive'},
            {'first_id': 3, 'last_id': 3, 'start': 8, 'end': 12, 'label': 'Ad', 'reason': 'Advertisement'}]
    policy = timeline.apply_preferences(rows, {'remove_ads': True})
    analysis = {'segments': rows, 'timeline': [{'text': '<img onerror="bad()">'}] * 3,
                'summary': '<script>summary()</script>', 'provider': 'custom', 'model': 'fixture'}
    page = render_ad_report(SimpleNamespace(id=1, title='<unsafe>', guid='guid'), policy['segments'], analysis=analysis, edit_policy=policy)
    assert '<script>' not in page and '<img onerror=' not in page
    assert '&lt;script&gt;' in page and 'Extra cuts from the short-island rule: 3.00s' in page
    assert 'Content' in page and 'short_island' in page


@pytest.mark.asyncio
async def test_api_new_fields_preserve_omitted_or_null_options_and_accept_zero(isolated_data_dir, monkeypatch):
    from fastapi import BackgroundTasks
    from app.api.v1 import router as api
    from app.api.v1.schemas import SubscriptionSettingsUpdate
    from app.core.models import SubscriptionCreate
    init_db()
    repo = SubscriptionRepository()
    sub = repo.create(SubscriptionCreate(feed_url='https://example.com/feed'), 'Show', 'show')
    monkeypatch.setattr(api, '_processor', lambda: SimpleNamespace())
    async def update(**fields):
        return await api.update_subscription_settings(
            sub.id, SubscriptionSettingsUpdate(**fields), BackgroundTasks(), SimpleNamespace(is_admin=True, user_id=1))
    await update(minimum_retained_seconds=None, processing_workflow=None)
    same = repo.get_by_id(sub.id)
    assert same.inherit_content_removal and same.inherit_processing_workflow
    await update(processing_workflow='complete_timeline', minimum_retained_seconds=0, remove_editorial_non_speech=False)
    changed = repo.get_by_id(sub.id)
    assert changed.processing_workflow == 'complete_timeline' and not changed.inherit_processing_workflow
    assert changed.minimum_retained_seconds == 0 and not changed.inherit_content_removal
    await update(remove_ads=False)
    assert repo.get_by_id(sub.id).minimum_retained_seconds == 0
