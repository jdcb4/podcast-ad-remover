import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from app.core.audio import AudioProcessor, EmptyAudioResult
from app.core.config import settings
from app.core.processor import Processor
from app.core.rss_gen import RSSGenerator
from app.core import gemini_quota as quota
from app.core.ai_services import OpenAIProvider
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository, SubscriptionRepository, StaleAttempt
from test_processing_recovery import episodes
from test_gemini_quota import provider, error, MODEL


def test_no_retained_segments_never_invokes_ffmpeg(monkeypatch):
    monkeypatch.setattr(AudioProcessor, 'get_duration', lambda _: 10)
    monkeypatch.setattr(AudioProcessor, '_run_ffmpeg', lambda *a, **k: pytest.fail('FFmpeg called with no retained audio'))
    with pytest.raises(EmptyAudioResult, match='Skipped/non-episode'):
        AudioProcessor.remove_segments('source', 'output', [{'start': 0, 'end': 5}, {'start': 4, 'end': 12}],
                                       warning_tones={'start': True, 'middle': True, 'end': True})


def test_failed_source_probe_is_not_a_non_episode(monkeypatch):
    monkeypatch.setattr(AudioProcessor, 'get_duration', lambda _: 0)
    with pytest.raises(ValueError, match='Source audio') as caught:
        AudioProcessor.remove_segments('source', 'output', [{'start': 0, 'end': 10}])
    assert not isinstance(caught.value, EmptyAudioResult)


@pytest.mark.asyncio
@pytest.mark.parametrize('workflow', ['legacy', 'complete_timeline'])
@pytest.mark.parametrize('previous_publication', [False, True])
async def test_empty_removal_is_terminal_and_excluded_from_both_feeds(isolated_data_dir, monkeypatch, workflow, previous_publication):
    init_db()
    old_audio = Path(settings.PODCASTS_DIR) / 'show' / 'prior.mp3'
    old_audio.parent.mkdir()
    if previous_publication:
        old_audio.write_bytes(b'previous-publication')
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug,processing_workflow) VALUES(1,'https://example.com/feed','Show','show',?)", (workflow,))
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status,duration,local_filename) VALUES(1,1,'guid','Promo','https://example.com/audio','pending',10,?)",
                     (str(old_audio) if previous_publication else None,))
        conn.commit()
    rss = RSSGenerator()
    rss.generate_feed(1)
    rss.generate_unified_feed()
    if previous_publication:
        assert len(ElementTree.parse(Path(settings.FEEDS_DIR) / 'unified.xml').findall('.//item')) == 1
    async def download(url, directory, **kwargs):
        source = Path(directory) / 'original.mp3'
        source.write_bytes(b'fixture-source')
        return str(source)
    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr(AudioProcessor, 'get_duration', lambda _: 10)
    monkeypatch.setattr(AudioProcessor, '_run_ffmpeg', lambda *a, **k: pytest.fail('FFmpeg invoked for empty audio'))
    worker = Processor()
    worker.transcriber = SimpleNamespace(transcribe=lambda *a, **k: {'segments': [{'start': 0, 'end': 10, 'text': 'Advertising only.'}]})
    monkeypatch.setattr(worker.ad_detector, 'detect_ads', lambda *a, **k: [{'start': 0, 'end': 10, 'label': 'Ad', 'reason': 'Promotion'}])
    monkeypatch.setattr(worker.ad_detector, '_get_provider', lambda: SimpleNamespace(generate_structured=lambda *a: json.dumps({
        'segments': [{'first_id': 1, 'last_id': 1, 'label': 'Ad', 'reason': 'Promotion'}],
        'summary': 'This episode includes a promotion. It contains no editorial audio.'})))
    jobs, repo = JobRepository(), EpisodeRepository()
    jobs.enqueue(1)
    claim = jobs.claim_due(1)[0]
    worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    await worker._process_episode_inner(repo.get_by_id(1), SubscriptionRepository().get_by_id(1), claim)
    # The outer task's deletion finalizer must not change the skip reason or delete old media.
    worker._finalize_episode_deletion(1)
    result = repo.get_by_id(1)
    assert result.status == 'ignored'
    assert result.processing_step == 'skipped/non-episode'
    assert 'no retained audio' in result.error_message
    assert result.next_retry_at is None
    with get_db_connection() as conn:
        assert conn.execute('SELECT status FROM jobs WHERE id=?', (claim['job_id'],)).fetchone()[0] == 'completed'
    assert jobs.claim_due(1) == []
    for filename in ('show.xml', 'unified.xml'):
        assert ElementTree.parse(Path(settings.FEEDS_DIR) / filename).findall('.//item') == []
    if previous_publication:
        assert old_audio.read_bytes() == b'previous-publication'


def test_stale_worker_cannot_skip_replacement_job(episodes):
    repo, jobs = episodes
    claim = jobs.claim_due(1)[0]
    worker = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    repo.reset_status(claim['id'])
    with pytest.raises(StaleAttempt):
        worker.mark_non_episode(claim['id'], 'stale')


@pytest.mark.asyncio
async def test_exhausted_cascade_schedules_job_without_permanent_failure(episodes, monkeypatch):
    repo, jobs = episodes
    claim = jobs.claim_due(1)[0]
    quota.record_error(MODEL, error(message='daily quota exceeded'))
    instance, calls = provider(monkeypatch, [], models=[MODEL])
    async def download(url, directory, **kwargs):
        source = Path(directory) / 'original.mp3'
        source.write_bytes(b'fixture-source')
        return str(source)
    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    worker = Processor()
    worker.transcriber = SimpleNamespace(transcribe=lambda *a, **k: {'segments': [{'start': 0, 'end': 4, 'text': 'Text'}]})
    monkeypatch.setattr(worker.ad_detector, 'detect_ads', lambda *a, **k: instance.generate('test'))
    worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    await worker._process_episode_inner(repo.get_by_id(claim['id']), SubscriptionRepository().get_by_id(90), claim)
    result = repo.get_by_id(claim['id'])
    assert result.status == 'rate_limited'
    assert result.next_retry_at is not None and result.retry_count == 0
    with get_db_connection() as conn:
        job = conn.execute('SELECT * FROM jobs WHERE id=?', (claim['job_id'],)).fetchone()
    assert job['status'] == 'rate_limited' and job['provider_call_count'] == 0
    assert not calls
