import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.ai_services import AdDetector
from app.core.config import settings
from app.core.processor import Processor
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository, SubscriptionRepository


SUMMARY = 'This episode includes a musical demonstration. It closes with substantive analysis.'


@pytest.mark.asyncio
async def test_cached_classification_can_repair_a_previously_failed_summary(isolated_data_dir, tmp_path, monkeypatch):
    from app.core import timeline
    init_db()
    worker = Processor()
    worker._attempt_dir = tmp_path / 'attempt'
    worker._attempt_dir.mkdir()
    monkeypatch.setattr('app.core.processor.AudioProcessor.get_duration', lambda _: 5.0)
    calls = []
    def generate(messages, schema, mode):
        calls.append(schema)
        if len(calls) == 1:
            return json.dumps({'segments': [{'first_id': 1, 'last_id': 1, 'label': 'Content', 'reason': 'Discussion'}], 'summary': 'Bad format'})
        if len(calls) == 2:
            raise RuntimeError('Temporary summary failure')
        return json.dumps({'summary': SUMMARY})
    monkeypatch.setattr(AdDetector, '_get_provider', lambda self: SimpleNamespace(generate_structured=generate))
    snapshot = timeline.make_snapshot({'processing_workflow': 'complete_timeline'}, {})
    args = (SimpleNamespace(title='Episode', pub_date=None, ad_report_path=None), SimpleNamespace(title='Show'),
            {'segments': [{'start': 0, 'end': 5, 'text': 'Substantive discussion'}]},
            'synthetic-source', 'same-fingerprint', snapshot, tmp_path)
    first = await worker._classify_complete_timeline(*args)
    assert first['summary'] is None and first['summary_error'] == 'Temporary summary failure'
    second = await worker._classify_complete_timeline(*args)
    assert second['summary'] == SUMMARY and second['summary_error'] is None
    assert calls == [timeline.SCHEMA, timeline.SUMMARY_SCHEMA, timeline.SUMMARY_SCHEMA]
    assert first['segments'] == second['segments']


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
@pytest.mark.parametrize('rewrite,spoken', [(False, False), (True, False), (False, True)])
async def test_complete_pipeline_preserves_editorial_audio_and_summary_preferences(isolated_data_dir, tmp_path, monkeypatch, rewrite, spoken):
    init_db()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug,processing_workflow,minimum_retained_seconds,remove_ads,remove_intros,remove_outros,ai_rewrite_description,ai_audio_summary) VALUES(1,'https://example.com/feed','Show','show','complete_timeline',0,1,1,1,?,?)", (rewrite, spoken))
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status,duration,description) VALUES(1,1,'guid','Episode','https://example.com/source','pending',6,'Original description')")
        conn.commit()
    source = tmp_path / 'fixture.mp3'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6', str(source)], check=True, timeout=30)
    async def download(url, directory, **kwargs):
        destination = Path(directory) / 'original.mp3'
        shutil.copyfile(source, destination)
        return str(destination)
    async def notify(*args, **kwargs):
        pass
    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _: SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async', notify)
    calls = []
    def generate(messages, schema, mode):
        calls.append(messages)
        data = json.loads(messages[1]['content'])
        # A synthetic reference classification; the prompt itself receives no cut flags.
        assert 'remove_ads' not in data
        categories = {0: 'Intro', 1: 'Content', 2: 'EditorialNonSpeech', 3: 'Ad', 4: 'Content', 5: 'Outro'}
        return json.dumps({'segments': [{'first_id': u['id'], 'last_id': u['id'], 'label': categories.get(u['start'], 'Content'), 'reason': 'Synthetic reference'} for u in data['timeline']], 'summary': SUMMARY})
    fake = SimpleNamespace(generate_structured=generate, last_model='fixture-model', last_output_mode='json_schema')
    monkeypatch.setattr(AdDetector, '_get_provider', lambda self: fake)
    spoken_texts = []
    async def speech(self, text, output_path):
        spoken_texts.append(text)
        shutil.copyfile(source, output_path)
    monkeypatch.setattr(AdDetector, 'validate_tts', notify)
    monkeypatch.setattr(AdDetector, 'generate_audio', speech)
    def unexpected_legacy_call(*args, **kwargs):
        raise AssertionError('Complete Timeline must not invoke legacy classification or summary generation')
    monkeypatch.setattr(AdDetector, 'detect_ads', unexpected_legacy_call)
    monkeypatch.setattr(AdDetector, 'generate_summary', unexpected_legacy_call)
    jobs, repo = JobRepository(), EpisodeRepository()
    jobs.enqueue(1)
    claim = jobs.claim_due(1)[0]
    worker = Processor()
    worker.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    worker.transcriber = SimpleNamespace(transcribe=lambda *a, **k: {'segments': [
        {'start': 0, 'end': 1, 'text': 'Welcome to the show.'},
        {'start': 1, 'end': 2, 'text': 'Listen to this sample to understand the rhythm.'},
        {'start': 3, 'end': 4, 'text': 'Buy a pillow at example.com.'},
        {'start': 4, 'end': 5, 'text': 'In conclusion, the rhythm explains the genre.'},
        {'start': 5, 'end': 6, 'text': 'Like and subscribe. Next week we have another episode.'}]})
    await worker._process_episode_inner(repo.get_by_id(1), SubscriptionRepository().get_by_id(1), claim)
    completed = repo.get_by_id(1)
    assert completed.status == 'completed', completed.error_message
    assert len(calls) == 1
    assert completed.ai_summary == (SUMMARY if rewrite else None)
    assert spoken_texts == ([SUMMARY] if spoken else [])
    expected_duration = 9 if spoken else 3
    assert abs(completed.output_duration - expected_duration) < .25
    report = json.loads(Path(completed.ad_report_path).read_text(encoding='utf-8'))
    assert [(r['start'], r['end']) for r in report['segments']] == [(0, 1), (3, 4), (5, 6)]
    assert report['analysis']['summary'] == SUMMARY
    assert report['analysis']['timeline'][-1]['end'] == report['analysis']['duration']
    assert any(u['kind'] == 'GAP' and u['start'] == 2 and u['end'] == 3 for u in report['analysis']['timeline'])
    assert report['edit_policy']['island_seconds'] == 0
    html = Path(completed.report_path).read_text(encoding='utf-8')
    assert 'Editorial non-speech' in html and 'Complete timeline classification' in html
    rss = (Path(settings.FEEDS_DIR) / 'show.xml').read_text(encoding='utf-8')
    assert (SUMMARY in rss) is rewrite
    assert ('Original description' in rss) is (not rewrite)

    # Reprocessing with the same source can change cuts without a second LLM call.
    with get_db_connection() as conn:
        conn.execute('UPDATE subscriptions SET remove_intros=0 WHERE id=1')
        conn.commit()
    repo.reset_status(1)
    repo.update_status(1, 'pending')
    next_claim = jobs.claim_due(1)[0]
    worker.ep_repo = EpisodeRepository(attempt=(next_claim['job_id'], next_claim['claim_token']))
    await worker._process_episode_inner(repo.get_by_id(1), SubscriptionRepository().get_by_id(1), next_claim)
    replacement = repo.get_by_id(1)
    assert replacement.status == 'completed', replacement.error_message
    assert len(calls) == 1
    assert Path(completed.local_filename).is_file()
    new_report = json.loads(Path(replacement.ad_report_path).read_text(encoding='utf-8'))
    assert [(r['start'], r['end']) for r in new_report['segments']] == [(3, 4), (5, 6)]
