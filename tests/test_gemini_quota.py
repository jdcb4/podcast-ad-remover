import json
import multiprocessing
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core import gemini_quota as quota
from app.core.ai_services import OpenAIProvider, AdDetector, RateLimitError, PermanentProviderError
from app.core.config import settings
from app.core.model_defaults import MODEL_DEFAULTS
from app.core.provider_budget import provider_request, current_claim, ProviderBudgetExceeded
from app.infra.database import get_db_connection, init_db
from tests.test_timeline_http import client
from test_processing_recovery import episodes

MODEL = 'gemini-2.5-flash'
LITE = 'gemini-3.1-flash-lite'


@pytest.fixture
def clock(isolated_data_dir, monkeypatch):
    init_db()
    now = [datetime(2026, 9, 16, 12, tzinfo=timezone.utc).timestamp()]
    monkeypatch.setattr(quota.time, 'time', lambda: now[0])
    return now


def call(model=MODEL, tokens=100, actual=None):
    with provider_request('gemini', model, gemini_free_tier=True, input_tokens=tokens) as metrics:
        metrics['input_tokens'] = actual


def test_rolling_minute_daily_and_restart(clock):
    for minute in range(4):
        for _ in range(5):
            call()
        with pytest.raises(quota.GeminiCooldown) as caught:
            call()
        if minute < 3:
            assert caught.value.until == clock[0] + 60
        clock[0] += 60
    midnight = quota.day_window(clock[0])[1]
    init_db()  # Restart/migration must preserve usage.
    with pytest.raises(quota.GeminiCooldown) as caught:
        call()
    assert caught.value.until == midnight
    assert quota.usage([MODEL])[0]['day_requests'] == 20
    clock[0] = midnight
    call()
    assert quota.usage([MODEL])[0]['day_requests'] == 1


def test_tokens_reserve_reconcile_and_failed_calls_count(clock):
    call(tokens=249_999, actual=10)
    call(tokens=249_990)
    with pytest.raises(quota.GeminiCooldown, match='token'):
        call(tokens=1)
    assert quota.usage([MODEL])[0]['minute_tokens'] == 250_000
    clock[0] += 60
    with pytest.raises(RuntimeError, match='transport'):
        with provider_request('gemini', MODEL, gemini_free_tier=True, input_tokens=42):
            raise RuntimeError('transport')
    assert quota.usage([MODEL])[0]['day_requests'] == 3
    assert quota.usage([MODEL])[0]['minute_tokens'] == 42
    with pytest.raises(ValueError, match='exceeds'):
        call(tokens=250_001)


def test_cross_worker_atomic_reservations(clock):
    def attempt(_):
        try:
            call(tokens=100_000)
            return True
        except quota.GeminiCooldown:
            return False
    with ThreadPoolExecutor(8) as workers:
        assert sum(workers.map(attempt, range(8))) == 2
    assert quota.usage([MODEL])[0]['day_requests'] == 2


def test_reservations_survive_an_unfinished_request_and_restart(clock):
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quota.reserve(conn, MODEL, 250_000)
        conn.commit()  # Simulate a worker stopping before usage reconciliation.
    init_db()
    with pytest.raises(quota.GeminiCooldown, match='token'):
        call(tokens=1)
    clock[0] += 60
    call()
    assert quota.usage([MODEL])[0]['day_requests'] == 2


def _process_calls(data_dir, barrier, results):
    settings.DATA_DIR = data_dir
    barrier.wait(timeout=20)
    accepted = 0
    for _ in range(5):
        try:
            call()
            accepted += 1
        except quota.GeminiCooldown:
            pass
    results.put(accepted)


def test_separate_processes_share_model_allowance(isolated_data_dir):
    init_db()
    context = multiprocessing.get_context('spawn')
    barrier, results = context.Barrier(2), context.Queue()
    workers = [context.Process(target=_process_calls, args=(settings.DATA_DIR, barrier, results)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        assert sum(results.get(timeout=40) for _ in workers) == 5
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()


def error(status=429, message='quota exceeded', body=None, headers=None):
    response = httpx.Response(status, request=httpx.Request('POST', 'https://example.test'), headers=headers)
    return openai.APIStatusError(message, response=response, body=body)


@pytest.mark.parametrize('when,hours', [('2026-03-08T08:00:00+00:00', 23), ('2026-11-01T07:00:00+00:00', 25)])
def test_pacific_reset_observes_dst(when, hours):
    now = datetime.fromisoformat(when).timestamp()
    assert quota.day_window(now)[1] - now == hours * 3600


def test_provider_daily_beats_short_retry_and_persists(clock):
    body = {'error': {'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure',
            'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]},
            {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '2s'}]}}
    blocked = quota.record_error(MODEL, error(body=body, headers={'retry-after': '3'}))
    assert blocked.until == quota.day_window(clock[0])[1]
    init_db()
    with pytest.raises(quota.GeminiCooldown, match='daily'):
        call()
    # A concurrent shorter minute error cannot shorten a daily suspension.
    assert quota.record_error(MODEL, error()).until == blocked.until
    clock[0] = blocked.until
    call()


@pytest.mark.parametrize('status,message,body,seconds,reason', [
    (429, 'quota exceeded', {'details': [{'retryDelay': '90.5s'}]}, 90.5, 'minute'),
    (429, 'quota exceeded', {'details': [{'retryDelay': {'seconds': '70', 'nanos': 500000000}}]}, 70.5, 'minute'),
    (503, 'model overloaded', None, 30, 'overloaded'),
    (503, 'RESOURCE_EXHAUSTED', None, 30, 'overloaded'),
    (429, 'temporary capacity unavailable', None, 30, 'overloaded'),
    (429, 'GenerateRequestsPerMinutePerProject quota exceeded', None, 60, 'minute'),
])
def test_error_classification(clock, status, message, body, seconds, reason):
    until, why = quota.error_cooldown(error(status, message, body))
    assert until == clock[0] + seconds
    assert reason in why


def test_retry_after_http_date_and_long_daily_delay(clock):
    from email.utils import format_datetime
    future = datetime.fromtimestamp(clock[0] + 180, timezone.utc)
    assert quota.error_cooldown(error(headers={'retry-after': format_datetime(future)}))[0] == clock[0] + 180
    assert quota.error_cooldown(error(message='daily quota exceeded', headers={'retry-after': '172800'}))[0] == clock[0] + 172800
    assert quota.error_cooldown(error(404, 'not found')) is None


def provider(monkeypatch, outcomes, models=None, enabled=True, name='gemini'):
    calls = []
    def send(**kwargs):
        calls.append(kwargs['model'])
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3), choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='ok', refusal=None))])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    monkeypatch.setattr(OpenAIProvider, '_init_client', lambda self: setattr(self, 'client', fake))
    return OpenAIProvider(['key-one', 'key-two'], models or [MODEL, LITE],
                          rate_limit_provider=name, gemini_free_tier=enabled), calls


def test_fallback_excludes_shared_quota_without_editing_cascade(clock, monkeypatch):
    instance, calls = provider(monkeypatch, [error(message='daily quota exceeded'), 'ok', 'ok'])
    assert instance.generate('test') == 'ok'
    assert instance.generate('test again') == 'ok'
    assert calls == [MODEL, LITE, LITE]
    assert instance.models == [MODEL, LITE]
    assert instance.current_key_idx == 0
    assert quota.usage([LITE])[0]['minute_tokens'] == 14


def test_mixed_errors_schedule_earliest_eligible_model(clock, monkeypatch):
    instance, calls = provider(monkeypatch, [error(message='daily quota exceeded'), error(503, 'overloaded'), error(404, 'not found')],
                               models=[MODEL, LITE, 'unknown'])
    with pytest.raises(RateLimitError) as caught:
        instance.generate('test')
    assert caught.value.get_next_retry_time() == datetime.fromtimestamp(clock[0] + 30, timezone.utc).replace(tzinfo=None)
    assert calls == [MODEL, LITE, 'unknown']
    assert instance.current_key_idx == 0


def test_all_excluded_make_no_requests_or_job_budget_charges(episodes, monkeypatch):
    quota.record_error(MODEL, error(message='daily quota exceeded'))
    quota.record_error(LITE, error())
    instance, calls = provider(monkeypatch, [])
    claim = episodes[1].claim_due(1)[0]
    token = current_claim.set((claim['job_id'], claim['claim_token']))
    try:
        with pytest.raises(RateLimitError):
            instance.generate('test')
    finally:
        current_claim.reset(token)
    with get_db_connection() as conn:
        assert conn.execute('SELECT provider_call_count FROM jobs WHERE id=?', (claim['job_id'],)).fetchone()[0] == 0
    assert not calls


def test_job_budget_rejection_rolls_back_quota_reservation(episodes, monkeypatch):
    claim = episodes[1].claim_due(1)[0]
    token = current_claim.set((claim['job_id'], claim['claim_token']))
    monkeypatch.setattr(settings, 'MAX_PROVIDER_CALLS_PER_JOB', 0)
    try:
        with pytest.raises(ProviderBudgetExceeded):
            call()
    finally:
        current_claim.reset(token)
    assert quota.usage([MODEL])[0]['day_requests'] == 0


@pytest.mark.parametrize('enabled,name', [(False, 'gemini'), (True, 'openai'), (True, 'openrouter'), (True, 'custom')])
def test_opt_out_and_other_providers_keep_existing_key_rotation(clock, monkeypatch, enabled, name):
    instance, calls = provider(monkeypatch, [error(message='429 rate limit'), 'ok'], models=[MODEL], enabled=enabled, name=name)
    assert instance.generate('test') == 'ok'
    assert len(calls) == 2 and instance.current_key_idx == 1
    assert quota.usage([MODEL])[0]['day_requests'] == 0


def test_authentication_stays_actionable(clock, monkeypatch):
    instance, _ = provider(monkeypatch, [error(401, 'unauthorized')])
    with pytest.raises(PermanentProviderError):
        instance.generate('test')


def test_checkbox_save_reload_scoping_and_usage(client):
    page = client.get('/admin/ai/text-analysis')
    assert page.status_code == 200
    assert 'enable gemini free tier rate limit handling' in page.text
    def checkbox(html):
        return re.search(r'<input\b[^>]*name="gemini_free_tier_enabled"[^>]*>', html).group()
    assert 'checked' not in checkbox(page.text)
    data = {'section': 'ai_text', 'gemini_free_tier_present': 'true', 'gemini_free_tier_enabled': 'true'}
    assert client.post('/admin/ai/update', data=data, follow_redirects=False).status_code == 303
    call()
    quota.record_error(MODEL, error(message='daily quota exceeded'))
    quota.record_error('gemini-3.8-flash', error(message='daily quota exceeded'))
    # Other AI pages and older forms must not unset this checkbox.
    client.post('/admin/ai/update', data={'section': 'ai_transcription', 'whisper_model': 'base'})
    client.post('/admin/ai/update', data={'section': 'ai_text'})
    page = client.get('/admin/ai/text-analysis')
    assert 'checked' in checkbox(page.text)
    assert 'Requests/day (Pacific)' in page.text
    assert 'provider daily quota exhausted; resets at midnight Pacific' in page.text
    with get_db_connection() as conn:
        saved = dict(conn.execute('SELECT * FROM app_settings').fetchone())
    assert saved['gemini_free_tier_enabled'] == 1
    detector = AdDetector()
    detector.settings = saved
    assert detector.create_provider('gemini', api_key='test').gemini_free_tier
    assert 'gemini-3-flash' not in json.loads(saved['ai_model_cascade'])
    assert 'gemini-3-flash' not in MODEL_DEFAULTS['gemini']
    client.post('/admin/ai/update', data={'section': 'ai_text', 'gemini_free_tier_present': 'true'})
    with get_db_connection() as conn:
        assert conn.execute('SELECT gemini_free_tier_enabled FROM app_settings').fetchone()[0] == 0


def test_upgrade_is_additive_backed_up_and_idempotent(isolated_data_dir):
    import sqlite3
    from pathlib import Path
    from app.infra import database
    migration = database.FORMAL_MIGRATIONS.pop()
    try:
        init_db()
        with get_db_connection() as conn:
            conn.execute("UPDATE app_settings SET ai_model_cascade='[\"custom-gemini-model\"]'")
            conn.commit()
    finally:
        database.FORMAL_MIGRATIONS.append(migration)
    backups_before = set(Path(settings.DATA_DIR).glob('backups/*.db'))
    init_db()
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM app_settings').fetchone()
        assert row['gemini_free_tier_enabled'] == 0
        assert json.loads(row['ai_model_cascade']) == ['custom-gemini-model']
    backups = set(Path(settings.DATA_DIR).glob('backups/*.db')) - backups_before
    assert len(backups) == 1
    with sqlite3.connect(str(backups.pop())) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert 'gemini_free_tier_enabled' not in {row[1] for row in conn.execute('PRAGMA table_info(app_settings)')}
    call()
    init_db()
    assert quota.usage([MODEL])[0]['day_requests'] == 1


@pytest.mark.asyncio
async def test_speech_quota_does_not_rotate_keys(clock, monkeypatch, tmp_path):
    from test_ai_provider_architecture import FakeAsyncClient, FakeGeminiTtsResponse
    detector = AdDetector()
    detector.settings = {'gemini_free_tier_enabled': 1, 'gemini_api_keys': '["one", "two"]',
                         'gemini_tts_model_cascade': '["speech-model"]'}
    FakeAsyncClient.calls = []
    FakeAsyncClient.responses = [FakeGeminiTtsResponse(status_code=429, payload={
        'error': {'message': 'daily quota exceeded', 'details': [{'retryDelay': '2s'}]}})]
    monkeypatch.setattr('app.core.ai_services.httpx.AsyncClient', FakeAsyncClient)
    with pytest.raises(RateLimitError) as caught:
        await detector._generate_gemini_tts('speech', str(tmp_path / 'audio.wav'))
    assert caught.value.get_next_retry_time() == datetime.fromtimestamp(quota.day_window(clock[0])[1], timezone.utc).replace(tzinfo=None)
    with pytest.raises(RateLimitError):
        await detector._generate_gemini_tts('speech', str(tmp_path / 'audio.wav'))
    assert len(FakeAsyncClient.calls) == 1
    assert quota.usage(['speech-model'])[0]['day_requests'] == 1


@pytest.mark.parametrize('model,rpm,rpd', [
    ('gemini-3.5-flash', 5, 20), ('gemini-3.5-flash-lite', 15, 500),
    ('gemini-3.6-flash', 5, 20), ('gemini-3.8-flash', 5, 20),
    ('gemini-2.5-flash', 5, 20), ('gemini-3.1-flash-lite', 15, 500),
    ('gemini-3.7-flash', 5, 20),
])
def test_requested_model_limits(clock, model, rpm, rpd):
    for _ in range(rpm):
        call(model)
    with pytest.raises(quota.GeminiCooldown, match='minute request'):
        call(model)
    clock[0] += 60
    with get_db_connection() as conn:
        conn.execute('UPDATE gemini_quota_state SET requests=? WHERE model=?', (rpd - 1, model))
        conn.commit()
    call(model)
    with pytest.raises(quota.GeminiCooldown, match='daily'):
        call(model)


def test_legacy_summary_propagates_enabled_quota_only(clock, monkeypatch):
    detector = AdDetector()
    retry = datetime.fromtimestamp(clock[0] + 60, timezone.utc).replace(tzinfo=None)
    def fail(prompt):
        raise RateLimitError('wait', retry_at=retry)
    monkeypatch.setattr(detector, '_get_provider', lambda: SimpleNamespace(generate=fail))
    with pytest.raises(RateLimitError):
        detector.generate_summary({'segments': [{'text': 'text'}]}, 'Show', 'Title', 'today')


@pytest.mark.asyncio
async def test_timeline_summary_wait_preserves_classification_cache(clock, monkeypatch, tmp_path):
    from app.core import timeline
    from app.core.processor import Processor
    worker = Processor()
    worker._attempt_dir = tmp_path / 'attempt'
    worker._attempt_dir.mkdir()
    monkeypatch.setattr('app.core.processor.AudioProcessor.get_duration', lambda _: 5.0)
    calls = []
    retry = datetime.fromtimestamp(clock[0] + 60, timezone.utc).replace(tzinfo=None)
    def generate(messages, schema, mode):
        calls.append(schema)
        if len(calls) == 1:
            return json.dumps({'segments': [{'first_id': 1, 'last_id': 1, 'label': 'Content', 'reason': 'Discussion'}], 'summary': 'Invalid'})
        if len(calls) in (2, 3):
            raise RateLimitError('quota wait', retry_at=retry)
        return json.dumps({'summary': 'This episode includes discussion. It reviews a topic.'})
    monkeypatch.setattr(AdDetector, '_get_provider', lambda self: SimpleNamespace(generate_structured=generate))
    snapshot = timeline.make_snapshot({'processing_workflow': 'complete_timeline'}, {})
    args = (SimpleNamespace(title='Episode', pub_date=None, ad_report_path=None), SimpleNamespace(title='Show'),
            {'segments': [{'start': 0, 'end': 5, 'text': 'Discussion'}]}, 'source', 'fingerprint', snapshot, tmp_path)
    first = await worker._classify_complete_timeline(*args)
    assert first['summary_retry_at'] == retry.isoformat()
    with pytest.raises(RateLimitError):
        await worker._classify_complete_timeline(*args)
    result = await worker._classify_complete_timeline(*args)
    assert result['summary_retry_at'] is None and result['summary']
    assert calls == [timeline.SCHEMA] + [timeline.SUMMARY_SCHEMA] * 3
