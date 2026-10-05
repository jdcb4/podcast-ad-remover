import sqlite3
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.core.models import SubscriptionCreate
from app.core.podcast_titles import podcast_feed_title, validate_title_affix
from app.core.rss_gen import RSSGenerator
from app.infra import database
from app.infra.database import get_db_connection, init_db
from app.web import router as web


@pytest.fixture
def client(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(web, '_reconcile_artwork_and_feeds', lambda: None)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.include_router(web.router)
    return TestClient(app)


def test_defaults_roundtrip_and_disabled_text_preserved(client):
    page = client.get('/admin/global-subscription-settings')
    assert 'The Rest is History (ad free)' in page.text
    values = web.get_global_settings()
    assert values['podcast_title_prefix'] == 'PAR - '
    assert not values['podcast_title_prefix_enabled']
    assert values['podcast_title_suffix_enabled']
    response = client.post('/admin/global-subscription-settings/update', data={
        'podcast_titles_present': 'true', 'podcast_title_prefix_enabled': 'true',
        'podcast_title_prefix': 'Mine & Yours - ', 'podcast_title_suffix_enabled': 'true',
        'podcast_title_suffix': '(clean)'}, follow_redirects=False)
    assert 'error=' not in response.headers['location']
    assert podcast_feed_title('Show', web.get_global_settings()) == 'Mine & Yours - Show (clean)'
    client.post('/admin/global-subscription-settings/update', data={'podcast_titles_present': 'true'})
    values = web.get_global_settings()
    assert podcast_feed_title('Show', values) == 'Show'
    assert values['podcast_title_prefix'] == 'Mine & Yours - '
    assert values['podcast_title_suffix'] == '(clean)'
    client.post('/admin/global-subscription-settings/update', data={'default_retention_limit': 3})
    assert web.get_global_settings()['podcast_title_suffix_enabled'] == 0


@pytest.mark.parametrize('bad', ['', '   ', 'x'*41, '<b>test</b>', 'a\nb', 'a\tb', 'a\x00b', 'a\u2028b', 'a\u202eb'])
def test_invalid_title_rejected_without_partial_save(client, bad):
    before = web.get_global_settings()
    result = client.post('/admin/global-subscription-settings/update', data={
        'podcast_titles_present': 'true', 'podcast_title_prefix_enabled': 'true',
        'podcast_title_prefix': bad, 'default_retention_limit': 40}, follow_redirects=False)
    assert 'error=' in result.headers['location']
    after = web.get_global_settings()
    assert after['default_retention_limit'] == before['default_retention_limit']
    assert after['podcast_title_prefix'] == before['podcast_title_prefix']


def test_plain_text_and_spacing():
    assert validate_title_affix('A & B — 🎧', 'Prefix', True) == 'A & B — 🎧'
    assert validate_title_affix('x'*40, 'Suffix', True) == 'x'*40
    assert podcast_feed_title('Show', {'podcast_title_prefix_enabled': True}) == 'PAR - Show (ad free)'
    assert podcast_feed_title('Show', {'podcast_title_prefix_enabled': True,
        'podcast_title_prefix': 'PAR -', 'podcast_title_suffix': ' (clean)'}) == 'PAR - Show (clean)'


def test_only_individual_rss_channel_changes(client):
    sub = web.sub_repo.create(SubscriptionCreate(feed_url='https://example.com/feed'),
        'The Rest is History', 'history')
    generator = RSSGenerator()
    original_unified = Path(generator.generate_unified_feed()).read_text(encoding='utf-8')
    client.post('/admin/global-subscription-settings/update', data={
        'podcast_titles_present': 'true', 'podcast_title_prefix_enabled': 'true',
        'podcast_title_prefix': 'PAR & Friends - '})
    rss = fromstring(Path(generator.generate_feed(sub.id)).read_text(encoding='utf-8'))
    assert rss.findtext('channel/title') == 'PAR & Friends - The Rest is History'
    assert web.sub_repo.get_by_id(sub.id).title == 'The Rest is History'
    assert Path(generator.generate_unified_feed()).read_text(encoding='utf-8') == original_unified


def test_upgrade_backup_defaults_and_idempotency(isolated_data_dir):
    init_db()
    columns = ['podcast_title_prefix_enabled', 'podcast_title_prefix',
               'podcast_title_suffix_enabled', 'podcast_title_suffix']
    with get_db_connection() as conn:
        for column in columns:
            conn.execute(f'ALTER TABLE app_settings DROP COLUMN {column}')
        conn.execute('DELETE FROM schema_migrations WHERE version=?', (database.PODCAST_TITLES_MIGRATION,))
        conn.commit()
    before = set((isolated_data_dir / 'backups').glob('*.db'))
    init_db()
    backups = set((isolated_data_dir / 'backups').glob('*.db')) - before
    assert backups
    with sqlite3.connect(next(iter(backups))) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert 'podcast_title_prefix' not in {row[1] for row in conn.execute('PRAGMA table_info(app_settings)')}
    values = web.get_global_settings()
    assert [values[c] for c in columns] == [0, 'PAR - ', 1, '(ad free)']
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET podcast_title_prefix='Custom',podcast_title_suffix_enabled=0")
        conn.commit()
    init_db()
    values = web.get_global_settings()
    assert values['podcast_title_prefix'] == 'Custom'
    assert values['podcast_title_suffix_enabled'] == 0
