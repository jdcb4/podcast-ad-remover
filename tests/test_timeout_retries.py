from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.ai_services import PermanentProviderError
from app.core.audio import AudioProcessor
from app.core.processor import Processor
from app.core.provider_budget import ProviderBudgetExceeded
from app.core.time_utils import now_utc
from app.infra.database import get_db_connection
from app.infra.repository import EpisodeRepository, SubscriptionRepository
from test_processing_recovery import episodes


@pytest.mark.asyncio
@pytest.mark.parametrize('previous_retries,delay', list(enumerate((5, 10, 20, 40, 80, 180, None))))
async def test_analysis_timeout_schedules_durable_retries_then_notifies(episodes, monkeypatch, previous_retries, delay):
    repo, jobs = episodes
    with get_db_connection() as conn:
        conn.execute('UPDATE episodes SET retry_count=? WHERE id=90', (previous_retries,))
        conn.commit()
    claim = jobs.claim_due(1)[0]
    assert claim['id'] == 90
    clock = now_utc()
    monkeypatch.setattr('app.core.processor.now_utc', lambda: clock)

    async def download(url, directory, **kwargs):
        source = Path(directory) / 'original.mp3'
        source.write_bytes(b'fixture-source')
        return str(source)

    notifications = []

    async def notify(*args, **kwargs):
        notifications.append((args, kwargs))

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout('analysis request timed out')

    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    monkeypatch.setattr(AudioProcessor, 'get_duration', lambda _: 4)
    worker = Processor()
    worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    worker.transcriber = SimpleNamespace(transcribe=lambda *a, **k: {'segments': [{'start': 0, 'end': 4, 'text': 'Text'}]})
    monkeypatch.setattr(worker.ad_detector, 'classify_timeline', timeout)
    await worker._process_episode_inner(repo.get_by_id(90), SubscriptionRepository().get_by_id(90), claim)

    episode = repo.get_by_id(90)
    assert episode.status == 'failed'
    assert episode.local_filename is None
    with get_db_connection() as conn:
        job = dict(conn.execute('SELECT * FROM jobs WHERE id=?', (claim['job_id'],)).fetchone())
    if delay is None:
        assert episode.next_retry_at is None
        assert job['status'] == 'failed'
        assert len(notifications) == 1
        args, kwargs = notifications[0]
        assert args[0:2] == ('breaking_error', 'Episode processing failed')
        assert 'failed after all retry attempts: analysis request timed out' in args[2]
        assert kwargs['severity'] == 'error'
    else:
        assert episode.retry_count == previous_retries + 1
        assert episode.next_retry_at == clock + timedelta(minutes=delay)
        assert job['status'] == 'retry_scheduled'
        assert notifications == []
        assert all(c['id'] != 90 for c in jobs.claim_due(10))
        # A fresh repository can claim the durable retry once its due time arrives.
        with get_db_connection() as conn:
            conn.execute("UPDATE jobs SET next_run_at=datetime('now','-1 second') WHERE id=?", (claim['job_id'],))
            conn.commit()
        replacement = type(jobs)().claim_due(1)[0]
        assert replacement['id'] == 90
        assert replacement['retry_count'] == previous_retries + 1


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [PermanentProviderError('authentication failed'), ProviderBudgetExceeded('request budget exhausted')])
async def test_long_retry_does_not_override_terminal_guards(episodes, monkeypatch, error):
    repo, jobs = episodes
    claim = jobs.claim_due(1)[0]

    async def fail_download(*args, **kwargs):
        raise error

    async def notify(*args, **kwargs):
        pytest.fail('Terminal provider guards unexpectedly notified')

    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=fail_download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    worker = Processor()
    worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    await worker._process_episode_inner(repo.get_by_id(claim['id']), SubscriptionRepository().get_by_id(90), claim)
    episode = repo.get_by_id(claim['id'])
    assert episode.status == 'failed'
    assert episode.next_retry_at is None
    assert all(c['id'] != claim['id'] for c in jobs.claim_due(10))


def test_timeout_default_and_explicit_override(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv('PROVIDER_TIMEOUT_SECONDS', raising=False)
    assert Settings(_env_file=None).PROVIDER_TIMEOUT_SECONDS == 300
    monkeypatch.setenv('PROVIDER_TIMEOUT_SECONDS', '480')
    assert Settings(_env_file=None).PROVIDER_TIMEOUT_SECONDS == 480
