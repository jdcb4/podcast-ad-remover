from app.core.config import settings
from app.infra.database import get_db_connection, init_db
from app.web.router import get_setup_status


class DummyRequest:
    base_url = "http://localhost:8000/"


def set_auth_state(*, dashboard: bool, feed: bool, app_external_url: str | None = None) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET auth_enabled = ?,
                enable_feed_auth = ?,
                app_external_url = ?
            WHERE id = 1
            """,
            (1 if dashboard else 0, 1 if feed else 0, app_external_url),
        )
        conn.commit()


def test_setup_status_warns_when_https_auth_uses_insecure_cookie(isolated_data_dir, monkeypatch):
    init_db()
    set_auth_state(dashboard=True, feed=False, app_external_url="https://podcasts.example.com")
    monkeypatch.setattr(settings, "COOKIE_SECURE", False)
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", "test-secret-that-is-not-default")

    setup_status = get_setup_status(
        DummyRequest(),
        {
            "auth_enabled": 1,
            "enable_feed_auth": 0,
            "public_subscribe_page_enabled": 1,
            "app_external_url": "https://podcasts.example.com",
        },
    )

    assert setup_status["security_warnings"] == [
        "COOKIE_SECURE is false while authentication is enabled on an HTTPS base URL. "
        "Set COOKIE_SECURE=true when users access this app through HTTPS."
    ]


def test_setup_status_does_not_warn_for_lan_http_auth(isolated_data_dir, monkeypatch):
    init_db()
    set_auth_state(dashboard=True, feed=False, app_external_url="http://podcasts.local:8000")
    monkeypatch.setattr(settings, "COOKIE_SECURE", False)
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", "test-secret-that-is-not-default")

    setup_status = get_setup_status(
        DummyRequest(),
        {
            "auth_enabled": 1,
            "enable_feed_auth": 0,
            "public_subscribe_page_enabled": 1,
            "app_external_url": "http://podcasts.local:8000",
        },
    )

    assert setup_status["security_warnings"] == []


def test_setup_status_warns_when_session_secret_is_default(isolated_data_dir, monkeypatch):
    init_db()
    set_auth_state(dashboard=False, feed=False, app_external_url="http://podcasts.local:8000")
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", "super-secret-session-key-change-me")

    setup_status = get_setup_status(
        DummyRequest(),
        {
            "auth_enabled": 0,
            "enable_feed_auth": 0,
            "public_subscribe_page_enabled": 1,
            "app_external_url": "http://podcasts.local:8000",
        },
    )

    assert setup_status["security_warnings"] == [
        "SESSION_SECRET_KEY is still using the default value. "
        "Set a unique SESSION_SECRET_KEY before enabling dashboard or feed authentication."
    ]


def test_system_template_renders_setup_security_warnings():
    template_source = open("app/web/templates/admin/system.html", encoding="utf-8").read()

    assert "setup_status.security_warnings" in template_source
    assert "Security setting needs attention" in template_source
