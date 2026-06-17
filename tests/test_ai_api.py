from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.router import router as ai_api_router
from app.infra.database import get_db_connection, init_db
from app.infra.repository import ApiRateLimitRepository, ApiTokenRepository


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

    token = repo.create("Assistant", scopes=["read", "process"], requests_per_minute=5, requests_per_day=50)
    token_row = repo.validate(token)
    listed = repo.list_active()

    assert token.startswith("par_")
    assert token_row is not None
    assert token_row["name"] == "Assistant"
    assert token_row["scopes"] == "process,read"
    assert listed[0]["token_prefix"] == token[:12]
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
