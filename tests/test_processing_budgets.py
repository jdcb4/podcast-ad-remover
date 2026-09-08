from datetime import datetime
from types import SimpleNamespace

import pytest

from app.core.ai_services import rate_limit_error, raise_permanent_provider_error, PermanentProviderError
from app.core.config import settings
from app.core.provider_budget import current_claim, provider_request, ProviderBudgetExceeded
from app.core.resource_budget import require_scratch
from app.infra.database import get_db_connection
from app.infra.repository import JobRepository
from test_processing_recovery import episodes


def test_provider_budget_persists_across_attempts_and_records_usage(episodes, monkeypatch):
    repo, jobs = episodes
    monkeypatch.setattr(settings, 'MAX_PROVIDER_CALLS_PER_JOB', 1)
    claim = jobs.claim_due(1)[0]
    token = current_claim.set((claim['job_id'], claim['claim_token']))
    try:
        with provider_request('test-provider', 'test-model') as usage:
            usage.update(input_tokens=10, output_tokens=5)
        with pytest.raises(ProviderBudgetExceeded):
            with provider_request('test-provider', 'second-model'):
                pytest.fail('Budget allowed an extra request')
    finally:
        current_claim.reset(token)
    with get_db_connection() as conn:
        count = conn.execute('SELECT provider_call_count FROM jobs WHERE id=?', (claim['job_id'],)).fetchone()[0]
        recorded = conn.execute('SELECT * FROM provider_calls').fetchone()
    assert count == 1
    assert recorded['outcome'] == 'succeeded' and recorded['input_tokens'] == 10
    jobs.acknowledge(claim['job_id'], claim['claim_token'])
    replacement = next(c for c in jobs.claim_due(10) if c['id'] == claim['id'])
    assert replacement['provider_call_count'] == 1


def test_retry_after_and_daily_limits_are_provider_specific():
    error = RuntimeError('rate limited')
    error.response = SimpleNamespace(headers={'retry-after': '45'})
    retry = rate_limit_error(error, 'openai')
    assert not retry.is_daily_limit
    assert 44 <= (retry.get_next_retry_time() - datetime.utcnow()).total_seconds() <= 46
    assert not rate_limit_error(RuntimeError('429 quota'), 'openai').is_daily_limit
    assert rate_limit_error(RuntimeError('daily quota'), 'gemini').is_daily_limit
    error.status_code = 401
    with pytest.raises(PermanentProviderError):
        raise_permanent_provider_error(error)


def test_scratch_reservations_stop_claims_and_stage_before_disk_exhaustion(episodes, monkeypatch):
    monkeypatch.setattr('shutil.disk_usage', lambda _: SimpleNamespace(free=settings.MIN_FREE_SPACE_BYTES + 10))
    assert JobRepository().claim_due(3, max_running=3) == []
    with pytest.raises(RuntimeError, match='scratch space'):
        require_scratch(3600, 100_000_000)
