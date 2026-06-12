from app.core import notifications
from app.infra.database import get_db_connection, init_db


def test_notifications_are_off_by_default(isolated_data_dir, monkeypatch):
    init_db()
    sent = []
    monkeypatch.setattr(
        notifications,
        "_send_with_apprise",
        lambda urls, title, body, severity: sent.append((urls, title, body, severity)) or True,
    )

    result = notifications.send_notification(
        notifications.EVENT_ACCESS_REQUEST,
        "Access request",
        "Someone requested access",
    )

    assert result is False
    assert sent == []


def test_notification_event_toggle_controls_send(isolated_data_dir, monkeypatch):
    init_db()
    sent = []
    monkeypatch.setattr(
        notifications,
        "_send_with_apprise",
        lambda urls, title, body, severity: sent.append((urls, title, body, severity)) or True,
    )

    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET notifications_enabled = 1,
                notification_urls = ?,
                notify_access_requests = 0,
                notify_new_podcasts = 1
            WHERE id = 1
            """,
            ("ntfy://example-topic\n# ignored\n",),
        )
        conn.commit()

    disabled_result = notifications.send_notification(
        notifications.EVENT_ACCESS_REQUEST,
        "Access request",
        "Someone requested access",
    )
    enabled_result = notifications.send_notification(
        notifications.EVENT_NEW_PODCAST,
        "Podcast added",
        "A podcast was added",
        severity="success",
    )

    assert disabled_result is False
    assert enabled_result is True
    assert sent == [(["ntfy://example-topic"], "Podcast added", "A podcast was added", "success")]
