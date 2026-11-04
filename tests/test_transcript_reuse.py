"""A transcribed episode is not transcribed again unless it is manually requeued."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core import artifacts
from app.core.audio import AudioProcessor
from app.core.processor import Processor
from app.infra.database import get_db_connection
from app.infra.repository import EpisodeRepository, SubscriptionRepository
from test_processing_recovery import episodes


class Killed(BaseException):
    """Stands in for the process dying: nothing in the worker can catch or tidy up."""


class ReachedAnalysis(BaseException):
    """Ends the attempt once analysis is reached; later stages need FFmpeg."""


@pytest.fixture
def harness(episodes, monkeypatch):
    repo, jobs = episodes
    with get_db_connection() as conn:  # one episode, so every claim below is the same one
        conn.execute("UPDATE jobs SET status='cancelled' WHERE episode_id != 90")
        conn.commit()
    calls = SimpleNamespace(transcribe=0, download=0, detect=[])

    async def download(url, directory, **kwargs):
        calls.download += 1
        path = Path(directory) / 'original.mp3'
        path.write_bytes(b'fixture-source')
        return str(path)

    def transcribe(*args, **kwargs):
        calls.transcribe += 1
        return {'segments': [{'start': 0, 'end': 4, 'text': 'hello'}]}

    def detect(*args, **kwargs):
        raise calls.detect.pop(0) if calls.detect else ReachedAnalysis()

    async def notify(*args, **kwargs):
        pass

    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    monkeypatch.setattr(AudioProcessor, 'get_duration', lambda _: 4)

    async def run():
        """Claim the episode's job and run one attempt, as a fresh worker would."""
        with get_db_connection() as conn:
            conn.execute("UPDATE jobs SET next_run_at=CURRENT_TIMESTAMP WHERE status IN ('queued','retry_scheduled','rate_limited')")
            conn.commit()
        claim = jobs.claim_due(10)[0]
        worker = Processor()
        worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
        worker.transcriber = SimpleNamespace(transcribe=transcribe)
        monkeypatch.setattr(worker.ad_detector, 'classify_timeline', detect)
        try:
            await worker._process_episode_inner(repo.get_by_id(claim['id']), SubscriptionRepository().get_by_id(90), claim)
        except ReachedAnalysis:
            pass
        return claim

    return SimpleNamespace(run=run, calls=calls, repo=repo, jobs=jobs)


def requeue(harness, claim, flags=None):
    with get_db_connection() as conn:
        conn.execute("UPDATE jobs SET status='failed', locked_by=NULL WHERE id=?", (claim['job_id'],))
        conn.commit()
    harness.repo.reset_status(claim['id'], processing_flags=flags)
    harness.repo.update_status(claim['id'], 'pending')
    harness.jobs.enqueue(claim['id'])


@pytest.mark.asyncio
async def test_killed_during_analysis_resumes_after_restart(harness):
    harness.calls.detect = [Killed()]
    with pytest.raises(Killed):
        await harness.run()
    harness.repo.requeue_stuck()
    await harness.run()
    assert (harness.calls.transcribe, harness.calls.download) == (1, 1)


@pytest.mark.asyncio
async def test_killed_before_retry_records_its_manifest(harness, monkeypatch):
    harness.calls.detect = [RuntimeError('503 overloaded')]
    await harness.run()
    real, state = artifacts.source_fingerprint, {'calls': 0}

    def dies_after_copy(path):
        state['calls'] += 1
        if state['calls'] == 2:  # the first verifies the resume copy; the second precedes the new manifest
            raise Killed()
        return real(path)

    monkeypatch.setattr('app.core.processor.source_fingerprint', dies_after_copy)
    with pytest.raises(Killed):
        await harness.run()
    monkeypatch.setattr('app.core.processor.source_fingerprint', real)
    harness.repo.requeue_stuck()
    await harness.run()
    assert (harness.calls.transcribe, harness.calls.download) == (1, 1)


@pytest.mark.asyncio
async def test_replacement_job_and_retry_button_reuse_transcript(harness):
    harness.calls.detect = [RuntimeError('503 overloaded')]
    claim = await harness.run()
    requeue(harness, claim)
    await harness.run()
    assert (harness.calls.transcribe, harness.calls.download) == (1, 1)


@pytest.mark.asyncio
async def test_manual_requeue_transcribes_again_exactly_once(harness):
    harness.calls.detect = [RuntimeError('503 overloaded')]
    claim = await harness.run()
    requeue(harness, claim, json.dumps({'skip_transcription': False}))
    harness.calls.detect = [Killed()]
    with pytest.raises(Killed):
        await harness.run()
    assert harness.calls.transcribe == 2
    harness.repo.requeue_stuck()
    await harness.run()
    assert harness.calls.transcribe == 2


@pytest.mark.asyncio
async def test_manual_requeue_killed_while_transcribing_does_not_use_older_transcript(harness):
    harness.calls.detect = [RuntimeError('503 overloaded')]
    claim = await harness.run()
    requeue(harness, claim, json.dumps({'skip_transcription': False}))
    worker_transcribe = harness.calls.transcribe
    harness.calls.detect = [Killed()]
    with pytest.raises(Killed):
        await harness.run()
    assert harness.calls.transcribe == worker_transcribe + 1


@pytest.mark.asyncio
async def test_changed_saved_audio_is_not_trusted(harness):
    harness.calls.detect = [RuntimeError('503 overloaded')]
    claim = await harness.run()
    root = artifacts.episode_directory(SubscriptionRepository().get_by_id(90).slug, claim['id'])
    for attempt in root.glob('attempt-*'):
        (attempt / 'original.mp3').write_bytes(b'changed on disk')
    await harness.run()
    assert (harness.calls.transcribe, harness.calls.download) == (2, 2)


@pytest.mark.asyncio
async def test_saved_audio_without_transcript_avoids_a_second_download(harness):
    harness.calls.detect = [Killed()]
    with pytest.raises(Killed):
        await harness.run()
    harness.repo.requeue_stuck()
    with get_db_connection() as conn:
        work = conn.execute('SELECT work_directory FROM jobs WHERE episode_id=90').fetchone()[0]
    (Path(work) / 'transcript.json').unlink()
    await harness.run()
    assert (harness.calls.transcribe, harness.calls.download) == (2, 1)


def test_current_model_settings_replace_those_saved_with_the_job(monkeypatch):
    from app.core import timeline
    from app.core.ai_services import AdDetector
    snapshot = timeline.make_snapshot({}, {'active_ai_provider': 'gemini', 'ai_model_cascade': 'old-model'})
    used = {}

    class Stop(Exception):
        pass

    def provider(self):
        used.update(self.settings)
        raise Stop()

    monkeypatch.setattr(AdDetector, '_get_provider', provider)
    detector = AdDetector()
    current = {'active_ai_provider': 'openai', 'openai_model': '["new-one", "new-two"]'}
    monkeypatch.setattr(detector, '_load_settings', lambda: current)
    with pytest.raises(Stop):
        detector.classify_timeline([], 4, {}, snapshot)
    assert used['active_ai_provider'] == 'openai' and used['openai_model'] == current['openai_model']
    assert 'ai_model_cascade' not in used
    used.clear()
    assert detector.repair_cached_timeline_summary([], 4, {}, snapshot)[0] is None
    assert used['active_ai_provider'] == 'openai'
