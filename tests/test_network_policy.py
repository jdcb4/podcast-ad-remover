import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.api import subscriptions
from app.core.artwork import ArtworkWatermarker
from app.core.config import settings
from app.core.feed import FeedManager
from app.core.network_policy import private_targets_allowed, user_network_scope
from app.core.podcast_import import import_feed
from app.core.processor import Processor
from app.core import podcast_operations
from app.infra.database import get_db_connection, init_db
from app.web import router as web
from app.web.auth import require_auth

LOOPBACK = 'http://127.0.0.1:9/internal'
ADMIN, MEMBER = 1, 2


@pytest.fixture
def users(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, 'ALLOW_PRIVATE_FEEDS', True)
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (id,username,password_hash,is_admin) VALUES (1,'admin','unused',1), (2,'member','unused',0)")
        conn.commit()


def add_podcast(owner):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id,feed_url,title,slug,owner_user_id,watermark_artwork,image_url) VALUES (1,?,'Show','show',?,1,?)",
            (LOOPBACK, owner, LOOPBACK),
        )
        conn.commit()


def client_for(user):
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.include_router(web.router)
    app.include_router(subscriptions.router, prefix='/api')
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize('user_id,allowed', [
    (ADMIN, True), (MEMBER, False), (None, True), (999, False),
])
def test_scope_follows_the_account(users, user_id, allowed):
    with user_network_scope(user_id):
        assert private_targets_allowed() is allowed
    assert private_targets_allowed() is True


def test_scope_cannot_widen_the_global_restriction(users, monkeypatch):
    monkeypatch.setattr(settings, 'ALLOW_PRIVATE_FEEDS', False)
    with user_network_scope(ADMIN):
        assert private_targets_allowed() is False


def test_scope_resets_after_an_error(users):
    with pytest.raises(RuntimeError):
        with user_network_scope(MEMBER):
            raise RuntimeError
    assert private_targets_allowed() is True


def test_scope_reaches_worker_threads(users):
    async def run():
        with user_network_scope(MEMBER):
            return await asyncio.to_thread(private_targets_allowed)
    assert asyncio.run(run()) is False


def test_member_fetch_of_loopback_never_connects(users, monkeypatch):
    def fail_if_connected(*_args, **_kwargs):
        raise AssertionError('connected to a private address')

    monkeypatch.setattr(httpx.Client, 'stream', fail_if_connected)
    with user_network_scope(MEMBER):
        with pytest.raises(ValueError, match='not allowed'):
            FeedManager._fetch_feed(LOOPBACK)


def test_member_redirect_to_loopback_is_refused(users):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={'Location': LOOPBACK})

    from app.core.http_downloads import stream_get
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with user_network_scope(MEMBER):
            with pytest.raises(ValueError, match='not allowed'):
                with stream_get(client, 'https://93.184.216.34/feed'):
                    pass
    assert LOOPBACK not in requests


@pytest.mark.parametrize('path,payload_key', [('/add', 'data'), ('/api/subscriptions', 'json')])
def test_member_cannot_add_a_loopback_feed(users, monkeypatch, path, payload_key):
    def fail_if_connected(*_args, **_kwargs):
        raise AssertionError('connected to a private address')

    monkeypatch.setattr(httpx.Client, 'stream', fail_if_connected)
    client = client_for(SimpleNamespace(id=MEMBER, is_admin=False))
    response = client.post(path, **{payload_key: {'feed_url': LOOPBACK}}, follow_redirects=False)
    assert response.status_code in {200, 400}
    with get_db_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM subscriptions').fetchone()[0] == 0


def test_admin_add_keeps_private_access(users, monkeypatch):
    seen = {}

    def resolve(_url):
        seen['allowed'] = private_targets_allowed()
        raise ValueError('stop here')

    monkeypatch.setattr(web, 'resolve_source', resolve)
    client_for(SimpleNamespace(id=ADMIN, is_admin=True)).post('/add', data={'feed_url': LOOPBACK})
    assert seen['allowed'] is True


def test_import_runs_as_the_importing_member(users, monkeypatch):
    seen = {}

    def resolve(_url):
        seen['allowed'] = private_targets_allowed()
        raise ValueError('stop here')

    monkeypatch.setattr('app.core.podcast_import.resolve_source', resolve)
    with pytest.raises(ValueError):
        import_feed('https://example.com/feed', MEMBER)
    assert seen['allowed'] is False


@pytest.mark.parametrize('owner,allowed', [(MEMBER, False), (ADMIN, True), (None, True)])
def test_worker_feed_check_uses_the_owner_policy(users, monkeypatch, owner, allowed):
    add_podcast(owner)
    seen = {}

    class Adapter:
        async def discover(self, _source_type, _url):
            seen['allowed'] = private_targets_allowed()
            return SimpleNamespace(entries=[])

    monkeypatch.setattr('app.core.processor.get_source_adapter', lambda _source_type: Adapter())
    asyncio.run(Processor().check_feeds(subscription_id=1))
    assert seen['allowed'] is allowed


@pytest.mark.parametrize('owner,allowed', [(MEMBER, False), (ADMIN, True)])
def test_artwork_fetch_uses_the_owner_policy(users, monkeypatch, owner, allowed):
    add_podcast(owner)
    seen = {}

    def download(_url):
        seen['allowed'] = private_targets_allowed()
        raise ValueError('stop here')

    monkeypatch.setattr(ArtworkWatermarker, '_download', staticmethod(download))
    with pytest.raises(ValueError):
        ArtworkWatermarker().reconcile(1)
    assert seen['allowed'] is allowed


def test_feed_replacement_preview_uses_the_owner_policy(users, monkeypatch):
    add_podcast(MEMBER)
    seen = {}

    def discover(_url):
        seen['allowed'] = private_targets_allowed()
        raise ValueError('stop here')

    monkeypatch.setattr(podcast_operations, 'discover_history', discover)
    with pytest.raises(ValueError):
        podcast_operations.preview(1, 'feed', 'http://192.168.1.5/feed')
    assert seen['allowed'] is False
