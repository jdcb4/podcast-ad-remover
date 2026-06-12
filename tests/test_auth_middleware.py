import pytest
import base64
from starlette.requests import Request
from starlette.responses import Response

from app.infra.database import get_db_connection, init_db
from app.infra.repository import FeedTokenRepository
from app.web.auth import auth_middleware
from app.web.auth_utils import hash_password
from app.web.middleware import feed_auth_middleware


def make_request(
    path: str,
    headers: dict[str, str] | None = None,
    session: dict | None = None,
    query_string: str = "",
) -> Request:
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string.encode("ascii"),
        "headers": raw_headers,
        "client": ("203.0.113.10", 12345),
        "scheme": "http",
        "server": ("testserver", 80),
        "session": session or {},
    }
    return Request(scope)


def enable_dashboard_auth() -> None:
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET auth_enabled = 1 WHERE id = 1")
        conn.commit()


def set_feed_auth(enabled: bool, username: str | None = None, password_hash: str | None = None) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET enable_feed_auth = ?,
                feed_auth_username = ?,
                feed_auth_password = ?
            WHERE id = 1
            """,
            (1 if enabled else 0, username, password_hash),
        )
        conn.commit()


@pytest.mark.asyncio
async def test_dashboard_auth_redirects_management_but_allows_public_subscribe(isolated_data_dir):
    init_db()
    enable_dashboard_auth()
    calls = []

    async def call_next(request):
        calls.append(request.url.path)
        return Response("ok")

    management_response = await auth_middleware(make_request("/admin/system"), call_next)
    subscribe_response = await auth_middleware(make_request("/subscribe"), call_next)

    assert management_response.status_code == 302
    assert management_response.headers["location"] == "/login"
    assert subscribe_response.status_code == 200
    assert calls == ["/subscribe"]


@pytest.mark.asyncio
async def test_dashboard_auth_bypasses_feed_paths_for_feed_auth_middleware(isolated_data_dir):
    init_db()
    enable_dashboard_auth()
    calls = []

    async def call_next(request):
        calls.append(request.url.path)
        return Response("ok")

    response = await auth_middleware(make_request("/feeds/example.xml"), call_next)

    assert response.status_code == 200
    assert calls == ["/feeds/example.xml"]


@pytest.mark.asyncio
async def test_feed_auth_mode_is_independent_from_dashboard_auth(isolated_data_dir):
    init_db()
    set_feed_auth(enabled=False)
    calls = []

    async def call_next(request):
        calls.append(request.url.path)
        return Response("ok")

    public_response = await feed_auth_middleware(make_request("/feeds/example.xml"), call_next)

    set_feed_auth(enabled=True)
    protected_response = await feed_auth_middleware(make_request("/feeds/example.xml"), call_next)

    assert public_response.status_code == 200
    assert protected_response.status_code == 401
    assert calls == ["/feeds/example.xml"]


@pytest.mark.asyncio
async def test_feed_auth_accepts_valid_feed_token_and_rejects_revoked_token(isolated_data_dir):
    init_db()
    set_feed_auth(enabled=True)
    token_repo = FeedTokenRepository()
    token = token_repo.create(name="pytest")
    calls = []

    async def call_next(request):
        calls.append(request.url.path)
        return Response("ok")

    allowed_response = await feed_auth_middleware(
        make_request("/audio/show/episode/audio.mp3", query_string=f"token={token}"),
        call_next,
    )
    token_repo.revoke(token)
    revoked_response = await feed_auth_middleware(
        make_request("/audio/show/episode/audio.mp3", query_string=f"token={token}"),
        call_next,
    )

    assert allowed_response.status_code == 200
    assert revoked_response.status_code == 401
    assert calls == ["/audio/show/episode/audio.mp3"]
    assert token_repo.list_active() == []


@pytest.mark.asyncio
async def test_feed_auth_accepts_hashed_standalone_basic_auth(isolated_data_dir):
    init_db()
    set_feed_auth(enabled=True, username="feeduser", password_hash=hash_password("feedpass"))
    credentials = base64.b64encode(b"feeduser:feedpass").decode("ascii")
    calls = []

    async def call_next(request):
        calls.append(request.url.path)
        return Response("ok")

    response = await feed_auth_middleware(
        make_request("/feeds/example.xml", headers={"Authorization": f"Basic {credentials}"}),
        call_next,
    )

    assert response.status_code == 200
    assert calls == ["/feeds/example.xml"]
