import json
import sqlite3

import pytest

from app.infra import database
from app.infra.database import get_db_connection, init_db


def old_database(monkeypatch):
    migrations = database.FORMAL_MIGRATIONS
    with monkeypatch.context() as patch:
        patch.setattr(database, 'FORMAL_MIGRATIONS', [m for m in migrations if m[0] != database.V2_MIGRATION])
        init_db()


def test_v2_upgrade_preserves_owners_media_and_selects_single_settings(isolated_data_dir, monkeypatch):
    old_database(monkeypatch)
    with get_db_connection() as conn:
        conn.execute('''UPDATE app_settings SET ai_model_cascade='["first", "second"]',
            gemini_api_keys='["secret-first", "secret-second"]', warning_tone_middle=1,
            unified_feed_artwork_url='https://example.com/cover.jpg' ''')
        conn.execute("INSERT INTO subscriptions(id,feed_url,slug,owner_user_id) VALUES(1,'https://example.com/feed','show',NULL)")
        conn.execute("INSERT INTO episodes(id,subscription_id,guid,title,original_url,status) VALUES(1,1,'guid','Queued','https://example.com/audio','pending')")
        conn.execute("INSERT INTO jobs(episode_id,status) VALUES(1,'queued')")
        conn.commit()
    sentinel = isolated_data_dir / 'published.mp3'
    sentinel.write_bytes(b'preserved')
    init_db()
    with get_db_connection() as conn:
        values = dict(conn.execute('SELECT * FROM app_settings').fetchone())
        assert values['ai_model_cascade'] == 'first'
        assert values['gemini_api_key'] == 'secret-first'
        assert values['gemini_api_keys'] is None
        assert values['tts_provider'] == 'unconfigured'
        assert values['cut_tone_enabled'] == 1
        assert values['onboarding_status'] == 'dismissed'
        assert values['unified_feed_artwork_source'] == 'url'
        assert 'secret' not in values['v2_migration_report']
        assert conn.execute('SELECT owner_user_id FROM subscriptions').fetchone()[0] is None
        snapshot = conn.execute('SELECT processing_snapshot FROM jobs').fetchone()[0]
        assert json.loads(snapshot)['workflow'] == 'complete_timeline'
        assert 'secret' not in snapshot
    backups = list((isolated_data_dir / 'backups').glob('*.db'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert json.loads(conn.execute('SELECT ai_model_cascade FROM app_settings').fetchone()[0]) == ['first', 'second']
    init_db()
    assert len(list((isolated_data_dir / 'backups').glob('*.db'))) == 1
    assert sentinel.read_bytes() == b'preserved'


def test_fresh_database_offers_setup(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        assert conn.execute('SELECT onboarding_status FROM app_settings').fetchone()[0] == 'not_started'


def test_v2_refuses_to_rewrite_leased_job(isolated_data_dir, monkeypatch):
    old_database(monkeypatch)
    with get_db_connection() as conn:
        conn.execute("INSERT INTO jobs(episode_id,status) VALUES(123,'running')")
        conn.commit()
    with pytest.raises(RuntimeError, match='drain running jobs'):
        init_db()
