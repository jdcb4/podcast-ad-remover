from pathlib import Path
import pytest
from app.core import media_storage as media
from app.core.config import settings
from app.infra.database import init_db, get_db_connection
from app.infra.repository import JobRepository
from app.api.audio_routes import _resolve_audio_file_path
from fastapi import HTTPException


@pytest.fixture
def storage(isolated_data_dir, monkeypatch, tmp_path):
    init_db()
    destination = tmp_path / 'media'
    destination.mkdir()
    monkeypatch.setattr(settings, 'MEDIA_DIR', str(destination))
    monkeypatch.setattr(settings, 'MIN_FREE_SPACE_BYTES', 0)
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug) VALUES(90,'https://example.org/feed','Show','show')")
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status) VALUES(90,90,'guid','Episode','https://example.org/audio','completed')")
        conn.commit()
    return destination


def audio(attempt='old', content=b'published audio'):
    path = Path(settings.PODCASTS_DIR) / 'show' / 'episode-90' / ('attempt-' + attempt) / 'processed.mp3'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    (path.parent / 'published.json').write_text('{"episode_id":90}')
    (path.parent / 'transcript.json').write_text('{}')
    return path


def run():
    for _ in range(20):
        media.step()
        if media.state()['status'] not in {'waiting','copying','cleaning'}:
            return
    raise AssertionError('Migration did not finish')


def test_copy_old_revisions_stable_urls_then_explicit_cleanup(storage):
    old, current = audio('old'), audio('new', b'new audio')
    with get_db_connection() as conn:
        conn.execute('UPDATE episodes SET local_filename=? WHERE id=90', (str(current),)); conn.commit()
    media.enable()
    assert media.preview()['count'] == 2
    media.start(); run()
    assert media.state()['status'] == 'complete'
    assert Path(media.state()['backup_path']).is_file()
    for path in (old,current):
        key = media.key_for(path)
        assert _resolve_audio_file_path(key) == storage / 'audio' / key
        assert path.exists()
    with get_db_connection() as conn:
        assert conn.execute('SELECT local_filename FROM episodes WHERE id=90').fetchone()[0] == str(current)
    media.start('cleanup'); run()
    assert not old.exists() and not current.exists()
    assert (current.parent / 'transcript.json').exists()
    assert media.status()['retained'] == 0


def test_missing_mount_never_falls_back_to_original(storage):
    path = audio(); media.enable(); media.start(); run()
    (storage / '.par-media-volume').unlink()
    assert path.exists()
    with pytest.raises(HTTPException) as exc:
        _resolve_audio_file_path(media.key_for(path))
    assert exc.value.status_code == 503
    with get_db_connection() as conn:
        assert media.processing_blocked(conn)
    with pytest.raises(media.StorageUnavailable):
        media.remove_original(media.key_for(path))
    assert path.exists()


def test_conflicting_destination_never_overwritten_resume_after_repair(storage):
    path = audio(); media.enable()
    target = storage / 'audio' / media.key_for(path)
    target.parent.mkdir(parents=True); target.write_bytes(b'other')
    media.start(); run()
    assert media.state()['status'] == 'error'
    assert target.read_bytes() == b'other' and path.exists()
    target.unlink(); media.control('resume'); run()
    assert media.state()['status'] == 'complete'
    assert media.status()['completed'] == 1


def test_cleanup_rechecks_destination_and_source(storage):
    path = audio(); media.enable(); media.start(); run()
    target = storage / 'audio' / media.key_for(path)
    target.write_bytes(b'corruption')
    media.start('cleanup'); run()
    assert media.state()['status'] == 'error' and path.exists()
    target.write_bytes(path.read_bytes()); path.write_bytes(b'changed original')
    media.control('resume'); run()
    assert media.state()['status'] == 'error' and path.exists()


def test_pause_cancel_and_restart_resume_manifest(storage):
    audio('one'); audio('two'); media.enable(); media.start(); media.step(); media.step()
    assert media.status()['completed'] == 1
    media.control('pause'); media.step()
    assert media.status()['completed'] == 1
    with get_db_connection() as conn:
        assert media.processing_blocked(conn)
    media.control('resume'); run()
    assert media.status()['completed'] == 2
    media.start('cleanup'); media.control('cancel'); media.step()
    assert media.state()['status'] == 'cancelled'
    assert media.status()['retained'] == 2


def test_claims_pause_and_running_job_drains(storage):
    audio(); media.enable()
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET status='pending' WHERE id=90"); conn.commit()
    jobs = JobRepository(); jobs.enqueue(90)
    claimed = jobs.claim_due(1)
    assert len(claimed) == 1
    media.start(); media.step()
    assert media.state()['status'] == 'waiting'
    assert jobs.claim_due(1) == []
    with get_db_connection() as conn:
        conn.execute("UPDATE jobs SET status='completed' WHERE episode_id=90"); conn.commit()
    run()
    assert media.state()['status'] == 'complete'


def test_low_space_changes_no_mapping(storage, monkeypatch):
    audio(); media.enable()
    monkeypatch.setattr(media.shutil,'disk_usage',lambda _: type('Usage',(),{'free':0})())
    assert not media.preview()['enough']
    media.start(); run()
    assert media.state()['status'] == 'error'
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM media_files').fetchone()[0] == 0


def test_new_publication_has_relative_mapping_and_explicit_enable(storage):
    path = audio()
    with pytest.raises(media.StorageUnavailable): media.stage_publication(path)
    media.enable()
    assert media.stage_publication(path)
    assert path.exists()  # until fenced episode commit
    media.remove_original(media.key_for(path))
    assert not path.exists()
    assert media.resolve(media.key_for(path)).read_bytes() == b'published audio'


def test_symlink_escape_rejected(storage, tmp_path):
    media.enable(); path = audio()
    outside = tmp_path / 'outside'; outside.mkdir()
    (storage / 'audio').mkdir()
    try: (storage / 'audio' / 'show').symlink_to(outside, target_is_directory=True)
    except OSError: pytest.skip('Symlinks unavailable')
    with pytest.raises(ValueError): media.stage_publication(path)
    assert not list(outside.iterdir())


def test_retention_deletes_media_but_not_other_episode(storage):
    path = audio(); media.enable(); media.start(); run()
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        media.delete_media('show/episode-9', conn)
        assert conn.execute('SELECT COUNT(*) FROM media_files').fetchone()[0] == 1
        media.delete_media('show/episode-90', conn); conn.commit()
    assert not (storage / 'audio' / media.key_for(path)).exists()


def test_admin_storage_auth_and_confirmation(storage):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware
    from app.web.router import router
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='storage-test-only')
    app.include_router(router)
    client = TestClient(app)
    assert client.get('/admin/system/storage').status_code == 200
    media.enable()
    assert client.post('/admin/system/storage', data={'action':'copy'}).status_code == 400
    assert media.state()['status'] == 'idle'
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET auth_enabled=1 WHERE id=1'); conn.commit()
    assert client.post('/admin/system/storage', data={'action':'copy','confirmed':'true'},follow_redirects=False).status_code in {302,303,401,403}
import asyncio
import shutil
import subprocess
from types import SimpleNamespace
from app.core.processor import Processor
from app.infra.repository import EpisodeRepository, SubscriptionRepository
from tests.test_processing_recovery import fixture_analysis


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
@pytest.mark.parametrize('failure', [None, 'stage', 'commit'])
def test_pipeline_publishes_to_media_and_keeps_metadata_local(storage, tmp_path, monkeypatch, failure):
    media.enable()
    source = tmp_path / 'fixture.mp3'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=4',str(source)],check=True,timeout=30)
    async def download(url, directory, **kwargs):
        target = Path(directory) / 'original.mp3'; shutil.copyfile(source,target); return str(target)
    async def notify(*args,**kwargs): pass
    monkeypatch.setattr('app.core.processor.get_source_adapter',lambda _:SimpleNamespace(download=download))
    monkeypatch.setattr('app.core.processor.send_notification_async',notify)
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET status='pending',duration=4 WHERE id=90")
        conn.execute('UPDATE app_settings SET cut_tone_enabled=0 WHERE id=1'); conn.commit()
    jobs=JobRepository(); jobs.enqueue(90); claim=jobs.claim_due(1)[0]
    processor=Processor()
    processor.ep_repo=EpisodeRepository(attempt=(claim['job_id'],claim['claim_token']))
    processor.transcriber=SimpleNamespace(transcribe=lambda *a,**k:{'segments':[{'start':0,'end':1,'text':'hello'},{'start':1,'end':3,'text':'Ad'},{'start':3,'end':4,'text':'Closing'}]})
    monkeypatch.setattr(processor.ad_detector,'classify_timeline',lambda units,*a:fixture_analysis(units,2))
    if failure == 'stage':
        def fail_stage(*args):
            raise OSError('simulated media outage')
        monkeypatch.setattr(media, 'stage_publication', fail_stage)
    elif failure == 'commit':
        original_update = processor.ep_repo.update_status
        def fail_commit(id, status, **kwargs):
            if status == 'completed':
                raise OSError('simulated commit failure')
            return original_update(id, status, **kwargs)
        monkeypatch.setattr(processor.ep_repo, 'update_status', fail_commit)
    asyncio.run(processor._process_episode_inner(EpisodeRepository().get_by_id(90),SubscriptionRepository().get_by_id(90),claim))
    ep=EpisodeRepository().get_by_id(90)
    if failure:
        assert ep.status != 'completed'
        assert not list(Path(settings.PODCASTS_DIR).rglob('published.json'))
        assert media.preview()['count'] == 0
        return
    assert ep.status == 'completed'
    assert not Path(ep.local_filename).exists()
    output=media.resolve(media.key_for(ep.local_filename))
    assert output.is_file() and output.is_relative_to(storage)
    assert Path(ep.transcript_path).is_relative_to(Path(settings.DATA_DIR)) and Path(ep.transcript_path).is_file()
    assert not (Path(ep.local_filename).parent/'original.mp3').exists()
    feed=(Path(settings.FEEDS_DIR)/'show.xml').read_text()
    assert '/audio/show/episode-90/attempt-' in feed and '/media/' not in feed
    asyncio.run(processor.delete_episode(90))
    assert not output.exists()


def test_interrupted_copy_does_not_switch_playback(storage, monkeypatch):
    path=audio();media.enable();media.start();media.step()
    real_copy=media.shutil.copyfileobj
    def fail_copy(source,destination,*args):
        destination.write(b'partial');raise OSError('simulated disconnected NAS')
    monkeypatch.setattr(media.shutil,'copyfileobj',fail_copy)
    media.step()
    assert media.state()['status']=='error'
    assert media.resolve(media.key_for(path))==path
    assert not list(storage.rglob('*.par-part'))
    monkeypatch.setattr(media.shutil,'copyfileobj',real_copy)
    media.control('resume');run()
    assert media.resolve(media.key_for(path)).read_bytes()==path.read_bytes()

def test_cancelled_attempt_removes_unpublished_media_copy(storage):
    path=audio();media.enable();media.stage_publication(path)
    target=media.resolve(media.key_for(path));assert target.exists()
    processor=Processor();processor._attempt_dir=path.parent
    processor._cleanup_artifacts(EpisodeRepository().get_by_id(90))
    assert not target.exists() and not path.parent.exists()


def test_cancel_cleanup_never_removes_committed_publication(storage):
    path=audio();media.enable();media.stage_publication(path)
    with get_db_connection() as conn:
        conn.execute('UPDATE episodes SET local_filename=? WHERE id=90',(str(path),));conn.commit()
    processor=Processor();processor._attempt_dir=path.parent
    processor._cleanup_artifacts(EpisodeRepository().get_by_id(90))
    assert media.resolve(media.key_for(path)).exists() and path.exists()

def test_cancel_during_copy_keeps_processing_and_deletion_paused_until_file_finishes(storage, monkeypatch):
    audio();media.enable();media.start();media.step()
    original=media.shutil.copyfileobj
    def cancel_during_copy(src,dst,*args):
        media.control('cancel')
        with get_db_connection() as conn:
            assert media.processing_blocked(conn)
            with pytest.raises(media.StorageUnavailable): media.delete_media('show',conn)
        return original(src,dst,*args)
    monkeypatch.setattr(media.shutil,'copyfileobj',cancel_during_copy)
    media.step()
    assert media.state()['status']=='cancelling'
    media.step()
    assert media.state()['status']=='cancelled'
    assert media.status()['completed']==1


def test_episode_deletion_defers_cleanly_during_migration(storage):
    path = audio()
    with get_db_connection() as conn:
        conn.execute('UPDATE episodes SET local_filename=? WHERE id=90', (str(path),)); conn.commit()
    media.enable(); media.start()
    processor = Processor()
    assert asyncio.run(processor.delete_episode(90))
    assert path.exists()
    with get_db_connection() as conn:
        row = conn.execute('SELECT status,processing_step FROM episodes WHERE id=90').fetchone()
        assert row['status'] == 'ignored' and row['processing_step'] != 'deleted'
    media.control('cancel'); media.step()
    processor._finalize_episode_deletion(90)
    assert not path.exists()
    with get_db_connection() as conn:
        assert conn.execute('SELECT processing_step FROM episodes WHERE id=90').fetchone()[0] == 'deleted'
