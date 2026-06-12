import pytest

from app.core.config import DEFAULT_SESSION_SECRET_KEY, settings
from app.infra.database import get_db_connection, init_db
from app.web import router as web_router


def set_auth_flags(*, dashboard: bool = False, feed: bool = False) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET auth_enabled = ?, enable_feed_auth = ? WHERE id = 1",
            (1 if dashboard else 0, 1 if feed else 0),
        )
        conn.commit()


def test_default_session_secret_is_allowed_when_auth_is_off(isolated_data_dir, monkeypatch):
    init_db()
    from app.main import validate_startup_security_settings

    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", DEFAULT_SESSION_SECRET_KEY)
    set_auth_flags(dashboard=False, feed=False)

    validate_startup_security_settings()


@pytest.mark.parametrize(
    ("dashboard", "feed"),
    [
        (True, False),
        (False, True),
    ],
)
def test_default_session_secret_fails_closed_when_auth_is_enabled(
    isolated_data_dir,
    monkeypatch,
    dashboard,
    feed,
):
    init_db()
    from app.main import validate_startup_security_settings

    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", DEFAULT_SESSION_SECRET_KEY)
    set_auth_flags(dashboard=dashboard, feed=feed)

    with pytest.raises(RuntimeError, match="Set SESSION_SECRET_KEY"):
        validate_startup_security_settings()


def test_custom_session_secret_allows_auth(isolated_data_dir, monkeypatch):
    init_db()
    from app.main import validate_startup_security_settings

    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", "test-secret-that-is-not-the-default")
    set_auth_flags(dashboard=True, feed=True)

    validate_startup_security_settings()


@pytest.mark.asyncio
async def test_system_settings_refuse_auth_with_default_session_secret(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", DEFAULT_SESSION_SECRET_KEY)

    response = await web_router.update_system_settings(
        request=object(),
        auth_enabled=True,
        enable_feed_auth=False,
        redirect_to=None,
        admin_user=object(),
    )

    assert response.status_code == 303
    assert "Set+SESSION_SECRET_KEY+before+enabling+dashboard+or+feed+authentication" in response.headers["location"]

    with get_db_connection() as conn:
        row = conn.execute("SELECT auth_enabled, enable_feed_auth FROM app_settings WHERE id = 1").fetchone()

    assert row["auth_enabled"] == 0
    assert row["enable_feed_auth"] == 0


@pytest.mark.asyncio
async def test_setup_admin_refuses_auth_with_default_session_secret(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", DEFAULT_SESSION_SECRET_KEY)

    response = await web_router.create_setup_admin_user(
        username="admin",
        password="long-enough",
        confirm_password="long-enough",
        user=object(),
    )

    assert response.status_code == 303
    assert "Set+SESSION_SECRET_KEY+before+enabling+login" in response.headers["location"]

    with get_db_connection() as conn:
        user_count = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
        auth_enabled = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()["auth_enabled"]

    assert user_count == 0
    assert auth_enabled == 0


@pytest.mark.asyncio
async def test_system_settings_refuse_standalone_feed_auth_without_credentials(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", "test-secret-that-is-not-the-default")

    response = await web_router.update_system_settings(
        request=object(),
        auth_enabled=False,
        enable_feed_auth=True,
        feed_auth_username=None,
        feed_auth_password=None,
        redirect_to=None,
        admin_user=object(),
    )

    assert response.status_code == 303
    assert "Standalone+feed+authentication+requires+a+username+and+password" in response.headers["location"]

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT auth_enabled, enable_feed_auth, feed_auth_username, feed_auth_password FROM app_settings WHERE id = 1"
        ).fetchone()

    assert row["auth_enabled"] == 0
    assert row["enable_feed_auth"] == 0
    assert row["feed_auth_username"] is None
    assert row["feed_auth_password"] is None
