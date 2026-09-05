import asyncio
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from threading import Barrier

import pytest

from app.core.config import settings
from app.core.processor import Processor
from app.core.publication import atomic_write
from app.core.artifacts import episode_directory, legacy_directory
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository, SubscriptionRepository, StaleAttempt


@pytest.fixture
def episodes(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug) VALUES(90,'https://example.com/feed','Show','show')")
        for i, guid in [(90, '..'), (91, 'a/b'), (92, 'a_b')]:
            conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status,duration) VALUES(?,90,?,'Episode','https://example.com/source','pending',4)", (i, guid))
        conn.commit()
    for i in (90, 91, 92):
        JobRepository().enqueue(i)
    return EpisodeRepository(), JobRepository()


def test_claim_capacity_is_transactional_between_independent_connections(episodes):
    barrier = Barrier(2)
    def claim():
        barrier.wait()
        return JobRepository().claim_due(2, max_running=2)
    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(lambda _: claim(), range(2)))
    assert sum(map(len, results)) == 2
    assert JobRepository().count_running() == 2


def test_cancel_retry_retains_lease_and_fences_late_updates(episodes):
    repo, jobs = episodes
    claim = jobs.claim_due(1)[0]
    worker = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    repo.reset_status(claim['id'])
    repo.update_status(claim['id'], 'pending')
    assert jobs.is_running_for_episode(claim['id'])
    assert all(c['id'] != claim['id'] for c in jobs.claim_due(10))
    with pytest.raises(StaleAttempt):
        worker.update_progress(claim['id'], 'late transcription', 70)
    jobs.acknowledge(claim['job_id'], claim['claim_token'])
    replacement = jobs.claim_due(1)[0]
    assert replacement['id'] == claim['id']
    with pytest.raises(StaleAttempt):
        worker.update_status(claim['id'], 'completed', filename='wrong.mp3')
    jobs.acknowledge(claim['job_id'], claim['claim_token'])
    assert jobs.is_running_for_episode(claim['id'])


def test_atomic_feed_failure_retains_previous_complete_file(tmp_path, monkeypatch):
    output = tmp_path / 'feed.xml'
    output.write_text('<rss>previous</rss>')
    def fail(*args):
        raise OSError('injected replacement failure')
    monkeypatch.setattr('app.core.publication.os.replace', fail)
    with pytest.raises(OSError):
        atomic_write(str(output), '<rss>replacement</rss>')
    assert output.read_text() == '<rss>previous</rss>'
    assert list(tmp_path.iterdir()) == [output]


def test_storage_identity_does_not_depend_on_feed_guid(episodes):
    assert legacy_directory('show', '..') is None
    assert episode_directory('show', 91) != episode_directory('show', 92)
    with pytest.raises(ValueError):
        settings.get_episode_dir('..', 'episode')


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
async def test_pipeline_preserves_publication_on_feed_failure_and_failed_reprocess(episodes, tmp_path, monkeypatch):
    repo, jobs = episodes
    source = tmp_path / 'fixture.mp3'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=4',str(source)], check=True, timeout=30)
    async def download(url, directory, **kwargs):
        target = Path(directory) / 'original.mp3'
        shutil.copyfile(source, target)
        return str(target)
    async def notify(*args, **kwargs):
        pass
    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    claim = jobs.claim_due(1)[0]
    processor = Processor()
    processor.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    processor.transcriber = SimpleNamespace(transcribe=lambda *a, **k: {'segments': [{'start': 0, 'end': 4, 'text': 'hello'}]})
    processor.ad_detector = SimpleNamespace(detect_ads=lambda *a, **k: [{'start': 1, 'end': 3, 'label': 'Ad'}])
    def fail_feed(*args):
        raise OSError('injected feed write failure')
    monkeypatch.setattr(processor.rss_gen, 'generate_feed', fail_feed)
    sub = SubscriptionRepository().get_by_id(90)
    await processor._process_episode_inner(repo.get_by_id(claim['id']), sub, claim)
    first = repo.get_by_id(claim['id'])
    assert first.status == 'completed'
    assert first.publication_pending
    assert Path(first.local_filename).is_file()
    assert 1.8 < first.output_duration < 2.2
    old_audio, old_guid = first.local_filename, first.guid
    monkeypatch.undo()
    await Processor().publish_pending_feeds()
    assert not repo.get_by_id(first.id).publication_pending
    assert '2</itunes:duration>' in (Path(settings.FEEDS_DIR) / 'show.xml').read_text()
    await Processor().version_episode(first.id)
    repo.reset_status(first.id)
    repo.update_status(first.id, 'pending')
    assert repo.get_by_id(first.id).local_filename == old_audio
    assert repo.get_by_id(first.id).guid == old_guid
    assert any(ep['id'] == first.id for ep in repo.get_completed_by_subscription(90))
    replacement = next(c for c in jobs.claim_due(10) if c['id'] == first.id)
    worker = Processor()
    worker.ep_repo = EpisodeRepository(attempt=(replacement['job_id'], replacement['claim_token']))
    worker.transcriber = processor.transcriber
    def fail_analysis(*args, **kwargs):
        raise RuntimeError('injected provider failure')
    worker.ad_detector = SimpleNamespace(detect_ads=fail_analysis)
    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    await worker._process_episode_inner(repo.get_by_id(first.id), sub, replacement)
    failed = repo.get_by_id(first.id)
    assert failed.status == 'failed'
    assert failed.local_filename == old_audio and Path(old_audio).exists()
    assert failed.guid == old_guid
    assert any(ep['id'] == first.id for ep in repo.get_completed_by_subscription(90))


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
async def test_retry_reuses_verified_transcription_and_analysis(episodes, tmp_path, monkeypatch):
    repo, jobs = episodes
    from app.core.audio import AudioProcessor
    source = tmp_path / 'source.mp3'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=4',str(source)], check=True, timeout=30)
    calls = {'transcribe': 0, 'detect': 0, 'download': 0}
    async def download(url, directory, **kwargs):
        calls['download'] += 1
        path = Path(directory) / 'original.mp3'
        shutil.copyfile(source, path)
        return str(path)
    def transcribe(*args, **kwargs):
        calls['transcribe'] += 1
        return {'segments': [{'start': 0, 'end': 4, 'text': 'hello'}]}
    def detect(*args, **kwargs):
        calls['detect'] += 1
        return []
    async def notify(*args, **kwargs): pass
    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    original_cut = AudioProcessor.remove_segments
    def fail_cut(*args, **kwargs): raise OSError('injected FFmpeg failure')
    monkeypatch.setattr(AudioProcessor, 'remove_segments', fail_cut)
    first = jobs.claim_due(1)[0]
    sub = SubscriptionRepository().get_by_id(90)
    for claim in (first,):
        worker = Processor()
        worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
        worker.transcriber = SimpleNamespace(transcribe=transcribe)
        worker.ad_detector = SimpleNamespace(detect_ads=detect)
        await worker._process_episode_inner(repo.get_by_id(claim['id']), sub, claim)
    assert repo.get_by_id(first['id']).status == 'failed'
    repo.update_status(first['id'], 'pending')
    second = next(c for c in jobs.claim_due(10) if c['id'] == first['id'])
    monkeypatch.setattr(AudioProcessor, 'remove_segments', original_cut)
    worker.ep_repo = EpisodeRepository(attempt=(second['job_id'], second['claim_token']))
    await worker._process_episode_inner(repo.get_by_id(second['id']), sub, second)
    assert repo.get_by_id(second['id']).status == 'completed'
    assert calls == {'transcribe': 1, 'detect': 1, 'download': 1}
