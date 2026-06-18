from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.router import router as ai_api_router
from app.infra.database import get_db_connection, init_db
from app.infra.repository import ApiRateLimitRepository, ApiTokenRepository, SubscriptionRepository
from app.core.models import SubscriptionCreate


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(ai_api_router, prefix="/api/v1")
    return TestClient(app)


def enable_ai_api(*, per_minute: int = 60, per_day: int = 1000, unauth_per_minute: int = 10) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET ai_api_enabled = 1,
                ai_api_default_requests_per_minute = ?,
                ai_api_default_requests_per_day = ?,
                ai_api_unauth_requests_per_minute = ?
            WHERE id = 1
            """,
            (per_minute, per_day, unauth_per_minute),
        )
        conn.commit()


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_user(username: str, *, is_admin: bool = False) -> int:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, "hash", 1 if is_admin else 0),
        )
        conn.commit()
        return cursor.lastrowid


def create_subscription(feed_url: str, title: str, slug: str, *, owner_user_id: int | None = None):
    return SubscriptionRepository().create(
        sub=SubscriptionCreate(feed_url=feed_url),
        title=title,
        slug=slug,
        owner_user_id=owner_user_id,
    )


def test_ai_api_migration_creates_defaults_and_tables(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        migrations = {
            row["version"]
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        settings = conn.execute(
            """
            SELECT ai_api_enabled,
                   ai_api_default_requests_per_minute,
                   ai_api_default_requests_per_day,
                   ai_api_unauth_requests_per_minute
            FROM app_settings WHERE id = 1
            """
        ).fetchone()

    assert "api_tokens" in tables
    assert "api_rate_limits" in tables
    assert "20260617_0007_ai_api" in migrations
    assert settings["ai_api_enabled"] == 0
    assert settings["ai_api_default_requests_per_minute"] == 60
    assert settings["ai_api_default_requests_per_day"] == 1000
    assert settings["ai_api_unauth_requests_per_minute"] == 10


def test_api_tokens_validate_list_and_revoke_without_exposing_hash(isolated_data_dir):
    init_db()
    repo = ApiTokenRepository()

    user_id = create_user("assistant")
    token = repo.create(
        "Assistant",
        scopes=["read", "process"],
        user_id=user_id,
        requests_per_minute=5,
        requests_per_day=50,
    )
    token_row = repo.validate(token)
    listed = repo.list_active()

    assert token.startswith("par_")
    assert token_row is not None
    assert token_row["name"] == "Assistant"
    assert token_row["scopes"] == "process,read"
    assert token_row["user_id"] == user_id
    assert token_row["username"] == "assistant"
    assert token_row["is_admin"] == 0
    assert listed[0]["token_prefix"] == token[:12]
    assert listed[0]["username"] == "assistant"
    assert "token_hash" not in listed[0]

    assert repo.revoke_by_id(listed[0]["id"]) is True
    assert repo.validate(token) is None


def test_ai_api_disabled_blocks_protected_routes(isolated_data_dir):
    init_db()
    client = make_client()

    response = client.get("/api/v1/subscriptions")

    assert response.status_code == 404
    assert response.json()["detail"] == "API not enabled"


def test_ai_api_requires_valid_token_and_scope(isolated_data_dir):
    init_db()
    enable_ai_api()
    token = ApiTokenRepository().create("Reader", scopes=["read"])
    client = make_client()

    missing = client.get("/api/v1/subscriptions")
    invalid = client.get("/api/v1/subscriptions", headers=auth_header("par_invalid"))
    allowed = client.get("/api/v1/subscriptions", headers=auth_header(token))
    denied = client.post("/api/v1/subscriptions/1/check", headers=auth_header(token))

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json() == []
    assert denied.status_code == 403


def test_ai_api_rate_limits_authenticated_tokens(isolated_data_dir):
    init_db()
    enable_ai_api(per_minute=1, per_day=100)
    token = ApiTokenRepository().create("Limited", scopes=["read"])
    client = make_client()

    first = client.get("/api/v1/subscriptions", headers=auth_header(token))
    second = client.get("/api/v1/subscriptions", headers=auth_header(token))

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"]


def test_ai_api_rate_limiter_recovers_after_window_changes(isolated_data_dir, monkeypatch):
    init_db()
    repo = ApiRateLimitRepository()
    times = iter([1000, 1001, 1061])
    monkeypatch.setattr("app.infra.repository.time.time", lambda: next(times))

    assert repo.check_and_increment("token:1", 1, 60, "minute")[0] is True
    assert repo.check_and_increment("token:1", 1, 60, "minute")[0] is False
    assert repo.check_and_increment("token:1", 1, 60, "minute")[0] is True


def test_ai_api_openapi_schema_includes_v1_paths(isolated_data_dir):
    init_db()
    enable_ai_api()
    client = make_client()

    response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/subscriptions" in paths
    assert "/api/v1/subscriptions/{subscription_id}/episodes" in paths
    assert "/api/v1/episodes/{episode_id}/reprocess" in paths


def test_create_subscription_accepts_valid_http_url_and_returns_existing_on_duplicate(
    isolated_data_dir,
    monkeypatch,
):
    init_db()
    enable_ai_api()
    token = ApiTokenRepository().create("Writer", scopes=["read", "write"])
    client = make_client()
    checked = []

    class FakeProcessor:
        async def check_feeds(self, subscription_id=None, limit=5):
            checked.append({"subscription_id": subscription_id, "limit": limit})

    async def fake_send_notification_async(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "app.api.v1.router.FeedManager.parse_feed",
        staticmethod(lambda url: ("NASA Podcast", "nasa-podcast", "https://example.com/art.jpg", "Space notes")),
    )
    monkeypatch.setattr("app.api.v1.router._processor", lambda: FakeProcessor())
    monkeypatch.setattr("app.api.v1.router.send_notification_async", fake_send_notification_async)

    payload = {
        "feed_url": "https://feeds.megaphone.fm/NATIONALAERONAUTICSANDSPACEADMINISTRATION8162188566",
        "initial_count": 0,
    }

    created = client.post("/api/v1/subscriptions", headers=auth_header(token), json=payload)
    duplicate = client.post("/api/v1/subscriptions", headers=auth_header(token), json=payload)
    listed = client.get("/api/v1/subscriptions", headers=auth_header(token))

    assert created.status_code == 200
    assert created.json()["title"] == "NASA Podcast"
    assert created.json()["feed_url"] == payload["feed_url"]
    assert created.json()["slug"] == "nasa-podcast"
    assert created.json()["is_active"] is True
    assert checked == [{"subscription_id": created.json()["id"], "limit": 0}]

    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == created.json()["id"]
    assert listed.status_code == 200
    assert [sub["feed_url"] for sub in listed.json()] == [payload["feed_url"]]


def test_create_subscription_rejects_invalid_url(isolated_data_dir):
    init_db()
    enable_ai_api()
    token = ApiTokenRepository().create("Writer", scopes=["write"])
    client = make_client()

    response = client.post(
        "/api/v1/subscriptions",
        headers=auth_header(token),
        json={"feed_url": "file:///etc/passwd", "initial_count": 0},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only HTTP and HTTPS URLs are supported"


def test_linked_user_token_reads_entire_library(isolated_data_dir):
    init_db()
    enable_ai_api()
    first_user_id = create_user("first")
    second_user_id = create_user("second")
    first_sub = create_subscription(
        "https://example.com/first.xml",
        "First Show",
        "first-show",
        owner_user_id=first_user_id,
    )
    second_sub = create_subscription(
        "https://example.com/second.xml",
        "Second Show",
        "second-show",
        owner_user_id=second_user_id,
    )
    token = ApiTokenRepository().create("First assistant", scopes=["read"], user_id=first_user_id)
    client = make_client()

    listed = client.get("/api/v1/subscriptions", headers=auth_header(token))
    allowed = client.get(f"/api/v1/subscriptions/{first_sub.id}", headers=auth_header(token))
    other = client.get(f"/api/v1/subscriptions/{second_sub.id}", headers=auth_header(token))

    assert listed.status_code == 200
    assert {sub["id"] for sub in listed.json()} == {first_sub.id, second_sub.id}
    assert allowed.status_code == 200
    assert other.status_code == 200


def test_linked_user_token_can_update_owned_subscription_only(isolated_data_dir):
    init_db()
    enable_ai_api()
    first_user_id = create_user("first")
    second_user_id = create_user("second")
    first_sub = create_subscription(
        "https://example.com/first.xml",
        "First Show",
        "first-show",
        owner_user_id=first_user_id,
    )
    second_sub = create_subscription(
        "https://example.com/second.xml",
        "Second Show",
        "second-show",
        owner_user_id=second_user_id,
    )
    token = ApiTokenRepository().create("First writer", scopes=["write"], user_id=first_user_id)
    client = make_client()

    owned = client.patch(
        f"/api/v1/subscriptions/{first_sub.id}/settings",
        headers=auth_header(token),
        json={"remove_ads": False},
    )
    other = client.patch(
        f"/api/v1/subscriptions/{second_sub.id}/settings",
        headers=auth_header(token),
        json={"remove_ads": False},
    )

    assert owned.status_code == 200
    assert owned.json()["status"] == "updated"
    assert other.status_code == 403
    assert other.json()["detail"] == "Only admins and the podcast owner can change podcast settings"


def test_linked_user_token_can_process_owned_subscription_only(isolated_data_dir, monkeypatch):
    init_db()
    enable_ai_api()
    checked = []

    class FakeProcessor:
        async def check_feeds(self, subscription_id=None, limit=5):
            checked.append(subscription_id)

    monkeypatch.setattr("app.api.v1.router._processor", lambda: FakeProcessor())

    first_user_id = create_user("first")
    second_user_id = create_user("second")
    first_sub = create_subscription(
        "https://example.com/first.xml",
        "First Show",
        "first-show",
        owner_user_id=first_user_id,
    )
    second_sub = create_subscription(
        "https://example.com/second.xml",
        "Second Show",
        "second-show",
        owner_user_id=second_user_id,
    )
    token = ApiTokenRepository().create("First processor", scopes=["process"], user_id=first_user_id)
    client = make_client()

    owned = client.post(f"/api/v1/subscriptions/{first_sub.id}/check", headers=auth_header(token))
    other = client.post(f"/api/v1/subscriptions/{second_sub.id}/check", headers=auth_header(token))

    assert owned.status_code == 200
    assert owned.json()["status"] == "check_triggered"
    assert checked == [first_sub.id]
    assert other.status_code == 403
    assert other.json()["detail"] == "Only admins and the podcast owner can manage this podcast"


def test_admin_linked_token_can_read_all_and_use_admin_scope(isolated_data_dir):
    init_db()
    enable_ai_api()
    admin_user_id = create_user("site-admin", is_admin=True)
    first_user_id = create_user("first")
    second_user_id = create_user("second")
    create_subscription("https://example.com/first.xml", "First Show", "first-show", owner_user_id=first_user_id)
    create_subscription("https://example.com/second.xml", "Second Show", "second-show", owner_user_id=second_user_id)
    token = ApiTokenRepository().create("Admin assistant", scopes=["read", "admin"], user_id=admin_user_id)
    client = make_client()

    listed = client.get("/api/v1/subscriptions", headers=auth_header(token))
    status_response = client.get("/api/v1/system/status", headers=auth_header(token))

    assert listed.status_code == 200
    assert {sub["title"] for sub in listed.json()} == {"First Show", "Second Show"}
    assert status_response.status_code == 200


def test_non_admin_linked_token_with_admin_scope_is_denied_system_status(isolated_data_dir):
    init_db()
    enable_ai_api()
    user_id = create_user("regular")
    token = ApiTokenRepository().create("Regular admin-scoped", scopes=["admin"], user_id=user_id)
    client = make_client()

    response = client.get("/api/v1/system/status", headers=auth_header(token))

    assert response.status_code == 403
    assert response.json()["detail"] == "API token is not linked to an admin user"
