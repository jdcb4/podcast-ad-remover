"""Ordered model and API key fallback for text analysis."""
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.core import gemini_quota as quota
from app.core import timeline
from app.core.ai_services import (AdDetector, AnalysisError, AnthropicProvider, OpenAIProvider,
                                  PermanentProviderError, RateLimitError)
from app.core.config import settings
from app.core.provider_budget import ProviderBudgetExceeded, current_claim
from app.core.provider_settings import all_values, credential, credentials, first_value, store_values
from app.infra.database import get_db_connection, init_db
from app.web import router as web
from test_processing_recovery import episodes


class ProviderFailure(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status


def failure(status, message='failed'):
    return ProviderFailure(status, message)


def completion(text='ok'):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=text, refusal=None))])


def provider(monkeypatch, outcomes, models=('one', 'two'), keys=('key-a', 'key-b'), **kwargs):
    """Record (model, key) for each request and play back the scripted outcomes."""
    calls = []

    def init_client(self):
        def create(**request):
            calls.append((request['model'], self.api_keys[self.current_key_idx]))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return completion(outcome)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(OpenAIProvider, '_init_client', init_client)
    return OpenAIProvider(list(keys), list(models), **kwargs), calls


@pytest.mark.parametrize('value,expected', [
    ('one', ['one']),
    ('["one", "two"]', ['one', 'two']),
    ('one, two\nthree\r\nfour', ['one', 'two', 'three', 'four']),
    ('one,one, ,two', ['one', 'two']),
    (['one', ' two ', 3, ''], ['one', 'two']),
    ('', []), (None, []), ('[]', []), ('{}', []), (7, []),
])
def test_all_values_reads_every_stored_and_entered_format(value, expected):
    assert all_values(value) == expected
    assert first_value(value, 'fallback') == (expected[0] if expected else 'fallback')


def test_store_values_keeps_a_single_value_as_plain_text():
    assert store_values(['one']) == 'one'
    assert store_values(['one', 'two']) == '["one", "two"]'
    assert store_values([]) == ''
    assert all_values(store_values(['one', 'two'])) == ['one', 'two']


def test_environment_keys_replace_saved_keys(monkeypatch):
    saved = {'gemini_api_keys': '["saved-a", "saved-b"]', 'gemini_api_key': 'legacy'}
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', None)
    assert credentials('gemini', saved) == ['saved-a', 'saved-b']
    assert credential('gemini', saved) == 'saved-a'
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'env-a, env-b')
    assert credentials('gemini', saved) == ['env-a', 'env-b']
    monkeypatch.setattr(settings, 'OPENAI_API_KEY', None)
    assert credentials('openai', {'openai_api_key': 'one,two'}) == ['one', 'two']
    assert credentials('custom', {'custom_llm_api_key': '["one", "two"]'}) == ['one', 'two']


def test_detector_builds_full_chain_but_defaults_to_one_model(monkeypatch):
    monkeypatch.setattr(settings, 'OPENAI_API_KEY', None)
    detector = AdDetector()
    detector.settings = {'openai_api_key': '["a", "b"]', 'openai_model': '["one", "two"]'}
    built = detector.create_provider('openai')
    assert built.api_keys == ['a', 'b'] and built.models == ['one', 'two']
    detector.settings = {'openai_api_key': 'a'}
    assert len(detector.create_provider('openai').models) == 1
    built = detector.create_provider('openai', api_key='x,y', model='three\nfour')
    assert built.api_keys == ['x', 'y'] and built.models == ['three', 'four']


def test_snapshot_freezes_the_whole_chain():
    snapshot = timeline.make_snapshot({}, {'active_ai_provider': 'openai', 'openai_model': '["one", "two"]'})
    assert json.loads(snapshot['settings']['openai_model']) == ['one', 'two']
    assert not any('api_key' in field for field in snapshot['settings'])


@pytest.mark.parametrize('status', [500, 503, 504, 404, 400])
def test_unavailable_model_falls_back_without_changing_key(monkeypatch, status):
    instance, calls = provider(monkeypatch, [failure(status), 'answer'])
    assert instance.generate('test') == 'answer'
    assert calls == [('one', 'key-a'), ('two', 'key-a')]
    assert instance.last_model == 'two'


def test_connection_failure_and_truncated_output_fall_back(monkeypatch):
    instance, calls = provider(monkeypatch, [ConnectionError('timed out'), 'answer'], keys=['key-a'])
    assert instance.generate('test') == 'answer'
    assert [model for model, _ in calls] == ['one', 'two']
    instance, calls = provider(monkeypatch, [], keys=['key-a'])
    replies = [SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='', refusal=None))]), completion('answer')]
    instance.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    assert instance.generate('test') == 'answer' and instance.last_model == 'two'


def test_rate_limit_tries_other_keys_before_next_model(monkeypatch):
    instance, calls = provider(monkeypatch, [failure(429), failure(429), 'answer'])
    assert instance.generate('test') == 'answer'
    assert calls == [('one', 'key-a'), ('one', 'key-b'), ('two', 'key-b')]


def test_working_key_is_kept_for_later_requests(monkeypatch):
    instance, calls = provider(monkeypatch, [failure(429), 'first', 'second'])
    assert instance.generate('test') == 'first'
    assert instance.generate('test') == 'second'
    assert calls == [('one', 'key-a'), ('one', 'key-b'), ('one', 'key-b')]
    assert instance.current_key_idx == 1


def test_all_keys_and_models_rate_limited_schedules_a_wait(monkeypatch):
    instance, calls = provider(monkeypatch, [failure(429) for _ in range(4)])
    with pytest.raises(RateLimitError):
        instance.generate('test')
    assert len(calls) == 4


def test_rejected_key_is_skipped_and_not_reused(monkeypatch):
    instance, calls = provider(monkeypatch, [failure(401), failure(503), 'answer'])
    assert instance.generate('test') == 'answer'
    assert calls == [('one', 'key-a'), ('one', 'key-b'), ('two', 'key-b')]


@pytest.mark.parametrize('error', [failure(401), failure(400, 'API key not valid. Please pass a valid API key.'),
                                   failure(429, 'insufficient_quota')])
def test_every_key_rejected_is_a_permanent_configuration_error(monkeypatch, error):
    instance, calls = provider(monkeypatch, [error, error])
    with pytest.raises(PermanentProviderError):
        instance.generate('test')
    assert calls == [('one', 'key-a'), ('one', 'key-b')]


def test_forbidden_model_does_not_disable_the_key_for_other_models(monkeypatch):
    instance, calls = provider(monkeypatch, [failure(403), 'answer'], keys=['key-a'])
    assert instance.generate('test') == 'answer'
    assert calls == [('one', 'key-a'), ('two', 'key-a')]


def test_forbidden_everywhere_is_permanent(monkeypatch):
    instance, _ = provider(monkeypatch, [failure(403), failure(403)], keys=['key-a'])
    with pytest.raises(PermanentProviderError):
        instance.generate('test')


def test_mixed_failures_raise_last_usable_error_for_normal_retry(monkeypatch):
    last = failure(503, 'overloaded')
    instance, _ = provider(monkeypatch, [failure(429), last], keys=['key-a'])
    with pytest.raises(type(last), match='overloaded'):
        instance.generate('test')
    bad_key_last = failure(404, 'no such model')
    instance, _ = provider(monkeypatch, [bad_key_last, failure(401)], keys=['key-a'])
    with pytest.raises(type(bad_key_last), match='no such model'):
        instance.generate('test')


def test_single_model_and_key_keep_original_error(monkeypatch):
    original = failure(503, 'overloaded')
    instance, calls = provider(monkeypatch, [original], models=['one'], keys=['key-a'])
    with pytest.raises(type(original)) as caught:
        instance.generate('test')
    assert caught.value is original and len(calls) == 1


def test_key_values_are_never_logged(monkeypatch, caplog):
    instance, _ = provider(monkeypatch, [failure(429), 'answer'], keys=['secret-a', 'secret-b'])
    with caplog.at_level('INFO'):
        instance.generate('test')
    assert 'secret-' not in caplog.text and 'key #1' in caplog.text


def test_anthropic_falls_back_across_keys_and_models(monkeypatch):
    calls, outcomes = [], [failure(429), failure(529), SimpleNamespace(content=[SimpleNamespace(type='text', text='answer')], stop_reason='end_turn', usage=None)]

    def init_client(self):
        def create(**request):
            calls.append((request['model'], self.api_keys[self.current_key_idx]))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(AnthropicProvider, '_init_client', init_client)
    instance = AnthropicProvider(['key-a', 'key-b'], ['one', 'two'])
    assert instance.generate('test') == 'answer'
    assert calls == [('one', 'key-a'), ('one', 'key-b'), ('two', 'key-b')]


@pytest.fixture
def clock(isolated_data_dir, monkeypatch):
    init_db()
    now = [1_800_000_000.0]
    monkeypatch.setattr(quota.time, 'time', lambda: now[0])
    return now


def test_free_tier_moves_to_next_model_without_rotating_keys(monkeypatch, clock):
    instance, calls = provider(monkeypatch, [failure(429, 'daily quota exceeded'), 'answer'],
                               models=['free-one', 'free-two'], rate_limit_provider='gemini', gemini_free_tier=True)
    assert instance.generate('test') == 'answer'
    assert calls == [('free-one', 'key-a'), ('free-two', 'key-a')]


def test_free_tier_cooldown_skips_requests_and_waits_for_earliest_model(monkeypatch, clock):
    quota.record_error('free-one', failure(429, 'daily quota exceeded'))
    quota.record_error('free-two', failure(503))
    instance, calls = provider(monkeypatch, [], models=['free-one', 'free-two'], rate_limit_provider='gemini', gemini_free_tier=True)
    with pytest.raises(RateLimitError) as caught:
        instance.generate('test')
    assert not calls
    assert caught.value.retry_at == datetime.fromtimestamp(clock[0] + 30, timezone.utc).replace(tzinfo=None)


def chain_timeline():
    units, _ = timeline.prepare_timeline({'segments': [{'start': 0, 'end': 5, 'text': 'Speech.'}]}, 5)
    return units


SUMMARY = 'This episode includes a test. It has two sentences.'
GOOD = json.dumps({'segments': [{'first_id': 1, 'last_id': 1, 'label': 'Content', 'reason': 'Show'}], 'summary': SUMMARY})


def classify(monkeypatch, instance):
    detector = AdDetector()
    monkeypatch.setattr(detector, '_load_settings', lambda: {})
    monkeypatch.setattr(AdDetector, '_get_provider', lambda self: instance)
    return detector.classify_timeline(chain_timeline(), 5, {}, timeline.make_snapshot({}, {}))


def test_model_failing_validation_twice_hands_over_to_next_model(monkeypatch):
    instance, calls = provider(monkeypatch, ['{"segments":[]}', '{"segments":[]}', GOOD], keys=['key-a'])
    result = classify(monkeypatch, instance)
    assert [model for model, _ in calls] == ['one', 'one', 'two']
    assert result['model'] == 'two' and result['summary'] == SUMMARY


def test_validation_failure_on_last_model_fails_closed(monkeypatch):
    instance, calls = provider(monkeypatch, ['{"segments":[]}'] * 4, keys=['key-a'])
    with pytest.raises(AnalysisError):
        classify(monkeypatch, instance)
    assert len(calls) == 4


def test_connection_test_checks_every_model_and_key_and_reports_bad_entries(monkeypatch):
    instance, calls = provider(monkeypatch, ['hello', 'hello', 'hello'])
    result = instance.test_connection()
    assert result['status'] == 'ok' and result['checked'] == 3
    assert calls == [('one', 'key-a'), ('two', 'key-a'), ('one', 'key-b')]
    instance, _ = provider(monkeypatch, ['hello', failure(404, 'no such model'), failure(401, 'bad key')])
    result = instance.test_connection()
    assert result['status'] == 'error'
    assert 'two with key #1: no such model' in result['error'] and 'one with key #2: bad key' in result['error']


@pytest.fixture
def claimed_job(episodes):
    claim = episodes[1].claim_due(1)[0]
    token = current_claim.set((claim['job_id'], claim['claim_token']))
    yield claim
    current_claim.reset(token)


def budget_used(claim):
    with get_db_connection() as conn:
        return conn.execute('SELECT provider_call_count FROM jobs WHERE id=?', (claim['job_id'],)).fetchone()[0]


def test_fallback_after_rejection_charges_one_request(claimed_job, monkeypatch):
    instance, calls = provider(monkeypatch, [failure(503), failure(429), failure(429), 'answer'], models=['one', 'two', 'three'])
    assert instance.generate('test') == 'answer'
    assert len(calls) == 4 and budget_used(claimed_job) == 1
    with get_db_connection() as conn:
        rows = conn.execute('SELECT outcome FROM provider_calls WHERE job_id=? ORDER BY id', (claimed_job['job_id'],)).fetchall()
    assert [row['outcome'] for row in rows] == ['failed', 'failed', 'failed', 'succeeded']


def test_failed_pass_charges_same_as_one_model_so_retries_survive(claimed_job, monkeypatch):
    monkeypatch.setattr(settings, 'MAX_PROVIDER_CALLS_PER_JOB', 7)
    for attempt in range(7):
        instance, _ = provider(monkeypatch, [failure(503), failure(503), failure(503)], models=['one', 'two', 'three'], keys=['key-a'])
        with pytest.raises(Exception, match='failed'):
            instance.generate('test')
        assert budget_used(claimed_job) == attempt + 1
    instance, calls = provider(monkeypatch, [], models=['one', 'two', 'three'], keys=['key-a'])
    with pytest.raises(ProviderBudgetExceeded):
        instance.generate('test')
    assert not calls


def test_possibly_billed_failures_stay_charged(claimed_job, monkeypatch):
    instance, _ = provider(monkeypatch, [ConnectionError('timed out'), 'answer'], keys=['key-a'])
    assert instance.generate('test') == 'answer'
    assert budget_used(claimed_job) == 2


def test_budget_stop_is_never_treated_as_a_model_failure(claimed_job, monkeypatch):
    monkeypatch.setattr(settings, 'MAX_PROVIDER_CALLS_PER_JOB', 1)
    instance, calls = provider(monkeypatch, [ConnectionError('timed out'), 'unused'], keys=['key-a'])
    with pytest.raises(ProviderBudgetExceeded):
        instance.generate('test')
    assert len(calls) == 1


@pytest.fixture
def client(isolated_data_dir, monkeypatch):
    for name in ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY'):
        monkeypatch.setattr(settings, name, None)
    init_db()
    monkeypatch.setattr(web, '_reconcile_artwork_and_feeds', lambda *a: None)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='fixture-only-secret')
    app.include_router(web.router)
    return TestClient(app, headers={'Origin': 'http://testserver'})


def saved():
    with get_db_connection() as conn:
        return dict(conn.execute('SELECT * FROM app_settings WHERE id=1').fetchone())


def test_settings_save_multiple_models_and_keys_without_rendering_keys(client):
    response = client.post('/admin/ai/update', data={
        'section': 'ai_text', 'active_ai_provider': 'openai',
        'openai_model': 'one\ntwo\none', 'openai_api_key': 'secret-a, secret-b',
        'gemini_api_keys': 'secret-c,secret-d',
    }, follow_redirects=False)
    assert response.status_code == 303
    row = saved()
    assert json.loads(row['openai_model']) == ['one', 'two']
    assert json.loads(row['openai_api_key']) == ['secret-a', 'secret-b']
    assert json.loads(row['gemini_api_keys']) == ['secret-c', 'secret-d']
    page = client.get('/admin/ai/text-analysis').text
    assert 'secret-' not in page
    assert '2 configured' in page
    field = re.search(r'<textarea[^>]*name="openai_model"[^>]*>(.*?)</textarea>', page, re.S)
    assert field.group(1).split() == ['one', 'two']


def test_blank_key_submission_keeps_saved_keys_and_order_change_is_saved(client):
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one\ntwo', 'openai_api_key': 'a,b'})
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'two\none', 'openai_api_key': ''})
    row = saved()
    assert json.loads(row['openai_api_key']) == ['a', 'b']
    assert json.loads(row['openai_model']) == ['two', 'one']


def test_single_values_stay_plain_text_for_compatibility(client):
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one', 'openai_api_key': 'a'})
    row = saved()
    assert row['openai_model'] == 'one' and row['openai_api_key'] == 'a'


def test_disabled_provider_panels_keep_their_saved_chains(client):
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one\ntwo'})
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'gemini', 'ai_model_cascade': 'g-one\ng-two'})
    row = saved()
    assert json.loads(row['openai_model']) == ['one', 'two']
    assert json.loads(row['ai_model_cascade']) == ['g-one', 'g-two']


@pytest.mark.parametrize('field,value', [('openai_model', '\n'.join(f'm{i}' for i in range(21))), ('openai_api_key', ','.join(f'k{i}' for i in range(21)))])
def test_oversized_chains_are_rejected(client, field, value):
    data = {'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one', field: value}
    assert client.post('/admin/ai/update', data=data).status_code == 400


def test_remove_saved_keys_clears_the_whole_chain(client):
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one', 'openai_api_key': 'a,b'})
    assert client.delete('/admin/ai/credentials/openai').status_code == 200
    assert saved()['openai_api_key'] is None


@pytest.mark.parametrize('provider_name,column', [('openai', 'openai_api_key'), ('gemini', 'gemini_api_keys')])
def test_voice_key_replaces_only_the_first_shared_key(client, provider_name, column):
    field = 'gemini_api_keys' if provider_name == 'gemini' else 'openai_api_key'
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': provider_name, field: 'old-a,keep-b'})
    response = client.post('/admin/ai/update', data={
        'section': 'ai_voice', 'tts_provider': provider_name, 'tts_model': 'voice-model', 'tts_voice': 'voice',
        'speech_credential': 'new-a',
    }, follow_redirects=False)
    assert response.status_code == 303
    assert json.loads(saved()[column]) == ['new-a', 'keep-b']


def test_setup_wizard_shows_and_accepts_a_chain(client):
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one\ntwo'})
    page = client.get('/admin/setup?step=2').text
    assert 'value="one, two"' in page and '[&#34;' not in page
    client.post('/admin/setup', data={'step': 2, 'provider': 'gemini', 'model': 'g-one, g-two', 'api_key': 'secret-a, secret-b'})
    review = client.get('/admin/setup?step=5').text
    assert 'secret-' not in review
    client.post('/admin/setup', data={'action': 'finish'})
    row = saved()
    assert json.loads(row['ai_model_cascade']) == ['g-one', 'g-two']
    assert json.loads(row['gemini_api_keys']) == ['secret-a', 'secret-b']


def test_model_list_add_control_and_hidden_panels_in_browser(client):
    import shutil, subprocess
    client.post('/admin/ai/update', data={'section': 'ai_text', 'active_ai_provider': 'openai', 'openai_model': 'one\ntwo'})
    html = client.get('/admin/ai/text-analysis').text
    script = r'''
const {JSDOM}=require('jsdom'),fs=require('node:fs'),assert=require('node:assert/strict');
const html=JSON.parse(fs.readFileSync(0,'utf8'));
const dom=new JSDOM(html,{url:'http://testserver/admin/ai/text-analysis',runScripts:'outside-only'}),w=dom.window,d=w.document;
w.matchMedia=()=>({addEventListener(){}});
const sent=[];
w.fetch=async(url,options)=>{sent.push(options.body);return {ok:true,json:async()=>({status:'ok',response:'hello',checked:3})};};
w.eval(fs.readFileSync('app/web/static/js/v2.js','utf8'));
(async()=>{
 const panel=d.querySelector('[data-provider-panel="openai"]'),list=panel.querySelector('[data-model-list]'),picker=panel.querySelector('[data-model-add]');
 assert.equal(panel.hidden,false);assert.equal(list.value,'one\ntwo');assert.equal(list.disabled,false);
 const hidden=d.querySelector('[data-model-list][name="ai_model_cascade"]');assert.equal(hidden.disabled,true);
 picker.value='three';panel.querySelector('[data-add-model]').click();assert.equal(list.value,'one\ntwo\nthree');assert.equal(picker.value,'');
 picker.value='two';picker.dispatchEvent(new w.KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}));assert.equal(list.value,'one\ntwo\nthree');
 assert.equal(picker.name,'');
 panel.querySelector('[data-test-provider]').click();await new Promise(resolve=>setImmediate(resolve));await new Promise(resolve=>setImmediate(resolve));
 assert.equal(sent[0].get('model'),'one\ntwo\nthree');
 assert.match(panel.querySelector('[data-provider-status]').textContent,/3 model and key combinations/);
 const select=d.querySelector('[data-provider-select]');select.value='gemini';select.dispatchEvent(new w.Event('change'));
 assert.equal(list.disabled,true);assert.equal(hidden.disabled,false);
 w.close();
})().catch(error=>{console.error(error);process.exit(1);});
'''
    result = subprocess.run([shutil.which('node'), '--input-type=commonjs', '-e', script], input=json.dumps(html), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
