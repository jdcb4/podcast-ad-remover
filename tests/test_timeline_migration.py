import json
import sqlite3
from pathlib import Path

from app.core.models import SubscriptionCreate
from app.infra import database
from app.infra.database import get_db_connection, init_db
from app.infra.repository import JobRepository, SubscriptionRepository


def test_upgrade_keeps_legacy_workflows_jobs_prompts_and_publications(isolated_data_dir, monkeypatch):
    migrations = database.FORMAL_MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(database, 'FORMAL_MIGRATIONS', migrations[:-1])
        init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET ad_prompt_base='Saved base', ad_target_sponsor='Saved ads', summary_prompt_template='Saved summary', whitelist_mode=1")
        conn.execute("INSERT INTO subscriptions(id,feed_url,slug,custom_instructions,inherit_content_removal) VALUES(1,'https://example.com/feed','show','Keep these instructions',1)")
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status,local_filename,published_guid,output_duration) VALUES(1,1,'source-guid','Published','https://example.com/source','completed','published.mp3','stable-published-guid',123.4)")
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status) VALUES(2,1,'queued-guid','Queued','https://example.com/source','pending')")
        conn.execute("INSERT INTO jobs(episode_id,status) VALUES(2,'queued')")
        conn.commit()
    media = isolated_data_dir / 'published.mp3'
    feed = isolated_data_dir / 'feed.xml'
    media.write_bytes(b'Existing media sentinel')
    feed.write_text('<rss>Existing feed sentinel</rss>', encoding='utf-8')

    init_db()
    backups = list((isolated_data_dir / 'backups').glob('*.db'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert 'processing_workflow' not in {r[1] for r in backup.execute('PRAGMA table_info(subscriptions)')}
        assert backup.execute('SELECT ad_prompt_base FROM app_settings').fetchone()[0] == 'Saved base'
    with get_db_connection() as conn:
        before = dict(conn.execute('SELECT * FROM app_settings').fetchone())
        assert before['default_processing_workflow'] == 'legacy'
        assert before['ad_prompt_base'] == 'Saved base' and before['ad_target_sponsor'] == 'Saved ads'
        assert before['summary_prompt_template'] == 'Saved summary' and before['whitelist_mode'] == 1
        assert conn.execute('SELECT processing_snapshot FROM jobs').fetchone()[0] is None
        conn.execute("UPDATE app_settings SET default_processing_workflow='complete_timeline', openrouter_api_key='never-snapshot-this-key'")
        conn.commit()
    repo = SubscriptionRepository()
    old = repo.get_by_id(1)
    assert old.processing_workflow == 'legacy' and not old.inherit_processing_workflow
    assert old.custom_instructions == 'Keep these instructions'
    new = repo.create(SubscriptionCreate(feed_url='https://example.com/new'), 'New', 'new')
    assert new.processing_workflow == 'complete_timeline' and new.inherit_processing_workflow
    with get_db_connection() as conn:
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status) VALUES(3,?,'new-guid','New job','https://example.com/new-source','pending')", (new.id,))
        conn.commit()
    JobRepository().enqueue(3)
    JobRepository().enqueue(2)  # Updating an existing queued job must not opt it in.
    with get_db_connection() as conn:
        snapshot = conn.execute('SELECT processing_snapshot FROM jobs WHERE episode_id=3').fetchone()[0]
        assert 'never-snapshot-this-key' not in snapshot
        assert json.loads(snapshot)['workflow'] == 'complete_timeline'
        assert conn.execute('SELECT processing_snapshot FROM jobs WHERE episode_id=2').fetchone()[0] is None
        conn.execute("UPDATE app_settings SET timeline_summary_instructions='New future summary rule', default_minimum_retained_seconds=0")
        conn.commit()
    JobRepository().enqueue(3)
    with get_db_connection() as conn:
        assert conn.execute('SELECT processing_snapshot FROM jobs WHERE episode_id=3').fetchone()[0] == snapshot
        published = conn.execute('SELECT published_guid,output_duration,local_filename FROM episodes WHERE id=1').fetchone()
        assert tuple(published) == ('stable-published-guid', 123.4, 'published.mp3')
    init_db()
    assert len(list((isolated_data_dir / 'backups').glob('*.db'))) == 1
    assert media.read_bytes() == b'Existing media sentinel'
    assert feed.read_text(encoding='utf-8') == '<rss>Existing feed sentinel</rss>'


def test_legacy_api_repository_updates_preserve_new_options(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    sub = repo.create(SubscriptionCreate(feed_url='https://example.com/feed'), 'Show', 'show')
    with get_db_connection() as conn:
        conn.execute("UPDATE subscriptions SET processing_workflow='complete_timeline',inherit_processing_workflow=0, minimum_retained_seconds=0, remove_non_editorial_non_speech=0 WHERE id=?", (sub.id,))
        conn.commit()
    repo.update_settings(sub.id, True, False, False, True, 'Saved custom instructions', False, False, False, False)
    after = repo.get_by_id(sub.id)
    assert after.processing_workflow == 'complete_timeline'
    assert after.setting_overrides['minimum_retained_seconds'] == 0
    assert after.setting_overrides['remove_non_editorial_non_speech'] == 0
