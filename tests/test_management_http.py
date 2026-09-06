from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.api import subscriptions, audio_routes
from app.infra.database import init_db, get_db_connection
from app.core.config import settings
from app.web import router as web
from app.web.auth import require_auth, auth_middleware


@pytest.fixture
def populated_client(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (id,username,password_hash) VALUES (91,'owner','unused'), (92,'reader','unused')")
        conn.execute("INSERT INTO subscriptions (id,feed_url,title,slug,owner_user_id) VALUES (90,'https://example.com/feed','Show','show',91)")
        conn.execute("INSERT INTO episodes (id,subscription_id,guid,title,original_url,status) VALUES (90,90,'one','One','https://example.com/one','completed'), (91,90,'two','Two','https://example.com/two','completed')")
        conn.commit()
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.include_router(web.router)
    app.include_router(subscriptions.router, prefix='/api')
    app.include_router(audio_routes.router)
    app.dependency_overrides[require_auth] = lambda: SimpleNamespace(id=92, is_admin=False)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize('method,path', [
    ('DELETE', '/api/episodes/90'), ('POST', '/api/episodes/90/cancel'),
    ('POST', '/api/episodes/90/process'), ('POST', '/api/episodes/90/reprocess'),
    ('POST', '/api/episodes/90/ignore'), ('POST', '/episodes/90/download'),
])
def test_non_owner_cannot_mutate_shared_episode(populated_client, method, path):
    assert populated_client.request(method, path).status_code == 403
    with get_db_connection() as conn:
        assert conn.execute('SELECT status FROM episodes WHERE id=90').fetchone()[0] == 'completed'


def test_membership_json_and_form_use_real_repository(populated_client):
    for action, member in [('add', True), ('add', True), ('remove', False), ('remove', False)]:
        result = populated_client.post('/subscriptions/90/library', data={'action': action}, headers={'Accept': 'application/json'})
        assert result.status_code == 200
        assert result.json()['in_user_library'] is member
        assert result.json()['user_library_count'] == int(member)
    result = populated_client.post('/subscriptions/90/library', data={'action': 'add'}, follow_redirects=False)
    assert result.status_code == 303


@pytest.mark.parametrize('initial_count', [0, 2, 5])
def test_rss_api_initial_count_controls_actual_queue(populated_client, monkeypatch, initial_count):
    from app.core import processor
    source = SimpleNamespace(source_type='rss', external_id=None,
                             canonical_url='https://example.com/new-feed', title='New feed',
                             slug='new-feed', description='', image_url=None)
    monkeypatch.setattr(subscriptions, 'resolve_source', lambda url: source)

    async def discover(source_type, url):
        return SimpleNamespace(entries=[{
            'guid': f'new-{i}', 'title': f'Episode {i}',
            'original_url': f'https://example.com/{i}.mp3',
            'pub_date': '2026-09-06T00:00:00', 'duration': 120,
            'description': '', 'file_size': 1000,
        } for i in range(5)])

    monkeypatch.setattr(processor, 'get_source_adapter',
                        lambda source_type: SimpleNamespace(discover=discover))
    response = populated_client.post(f'/api/subscriptions?initial_count={initial_count}',
                                     json={'feed_url': source.canonical_url})
    assert response.status_code == 200
    subscription_id = response.json()['id']
    with get_db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM episodes WHERE subscription_id=? AND status='pending'",
                            (subscription_id,)).fetchone()[0] == initial_count
        assert conn.execute('SELECT COUNT(*) FROM episodes WHERE subscription_id=?',
                            (subscription_id,)).fetchone()[0] == 5


def test_rss_api_rejects_negative_initial_count_without_creating_subscription(populated_client):
    response = populated_client.post('/api/subscriptions?initial_count=-1',
                                     json={'feed_url': 'https://example.com/new-feed'})
    assert response.status_code == 400
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM subscriptions').fetchone()[0] == 1


def test_owner_cancellation_preserves_audio_and_running_lease(populated_client, tmp_path):
    from app.infra.repository import EpisodeRepository, JobRepository
    audio = tmp_path / 'published.mp3'
    audio.write_bytes(b'previous publication')
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET local_filename=?,status='pending' WHERE id=90", (str(audio),))
        conn.commit()
    jobs = JobRepository()
    jobs.enqueue(90)
    claim = jobs.claim_due(1)[0]
    populated_client.app.dependency_overrides[require_auth] = lambda: SimpleNamespace(id=91, is_admin=False)
    response = populated_client.post('/api/episodes/90/cancel')
    assert response.status_code == 200
    episode = EpisodeRepository().get_by_id(90)
    assert episode.status == 'unprocessed' and episode.local_filename == str(audio)
    assert audio.read_bytes() == b'previous publication'
    assert jobs.is_running_for_episode(90)
    jobs.acknowledge(claim['job_id'], claim['claim_token'])
    assert not jobs.is_running_for_episode(90)


@pytest.mark.parametrize('status', ['completed', 'pending', 'processing', 'failed', 'unprocessed', 'ignored'])
def test_published_episode_counts_and_listens_survive_reprocessing(populated_client, tmp_path, status):
    from app.infra.repository import EpisodeRepository
    audio = tmp_path / 'published.mp3'
    audio.write_bytes(b'previous publication')
    with get_db_connection() as conn:
        conn.execute('UPDATE episodes SET local_filename=?,status=?,listen_count=7 WHERE id=90',
                     (str(audio), status))
        conn.commit()
    public = populated_client.get('/subscribe')
    assert public.status_code == 200
    entry = public.context['subscriptions'][0]
    expected = 0 if status == 'ignored' else 1
    assert entry['episode_count'] == expected
    assert bool(entry['latest_episode']) is bool(expected)
    dashboard = populated_client.get('/')
    assert dashboard.status_code == 200
    assert len(dashboard.context['subscriptions'][0]['episodes']) == expected
    assert EpisodeRepository().get_subscription_listen_count(90) == 7 * expected


def test_rendered_episode_page_and_shipped_javascript(populated_client):
    import subprocess
    import shutil
    response = populated_client.get('/subscriptions/90')
    assert response.status_code == 200
    result = subprocess.run([shutil.which('node'), 'tests/episode_dom.cjs'], input=response.text,
                            text=True, encoding='utf-8', capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_paginated_cards_share_safe_markup_and_read_only_controls(populated_client):
    with get_db_connection() as conn:
        conn.execute('UPDATE episodes SET title=?,description=? WHERE id=90',
                     ('<img src=x onerror=alert(1)>', '<script>bad()</script>'))
        conn.commit()
    html = populated_client.get('/api/subscriptions/90/episodes').json()['html']
    assert '&lt;img src=x onerror=alert(1)&gt;' in html
    assert '<img src=x' not in html and '<script>bad()' not in html
    assert 'disabled hidden' in html
    assert 'onclick="downloadEpisode' not in html and 'onclick="deleteEpisode' not in html


def test_filter_and_search_apply_before_pagination(populated_client):
    with get_db_connection() as conn:
        for i in range(100, 650):
            conn.execute("INSERT INTO episodes(id, subscription_id, guid, title, original_url, status, listen_count) VALUES(?,90,?,'Match','https://example.com','unprocessed',?)", (i, str(i), int(i == 100)))
        conn.commit()
    page = populated_client.get('/api/subscriptions/90/episodes?filter=played').json()
    assert [ep['id'] for ep in page['episodes']] == [100]
    page = populated_client.get('/api/subscriptions/90/episodes?search=Match&offset=500').json()
    assert page['total'] == 550 and page['has_more']
    assert len(page['episodes']) == 20
    assert populated_client.get('/api/subscriptions/90/episodes?limit=500').status_code == 422


def test_audio_tracking_uses_complete_path_and_invalid_range_is_client_error(populated_client):
    from pathlib import Path
    for episode, name in [(90, 'one'), (91, 'two')]:
        path = Path(settings.PODCASTS_DIR) / 'show' / name / 'processed.mp3'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'audio fixture')
        with get_db_connection() as conn:
            conn.execute('UPDATE episodes SET local_filename=? WHERE id=?', (str(path), episode))
            conn.commit()
    response = populated_client.get('/audio/show/two/processed.mp3', headers={'Range': 'bytes=garbage-'})
    assert response.status_code in (400, 416)
    assert populated_client.get('/audio/show/two/processed.mp3').status_code == 200
    with get_db_connection() as conn:
        assert [r[0] for r in conn.execute('SELECT listen_count FROM episodes WHERE id IN (90,91) ORDER BY id')] == [0, 1]


def test_ip_denial_crosses_real_middleware_stack_as_403(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET ip_allowlist='192.0.2.1' WHERE id=1")
        conn.commit()
    app = FastAPI()
    app.middleware('http')(auth_middleware)
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.get('/')(lambda: {'ok': True})
    assert TestClient(app, raise_server_exceptions=False).get('/').status_code == 403
