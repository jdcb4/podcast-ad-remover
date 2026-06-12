import asyncio
import logging
from dataclasses import dataclass

from app.infra.database import get_db_connection

logger = logging.getLogger(__name__)


EVENT_ACCESS_REQUEST = "access_request"
EVENT_NEW_PODCAST = "new_podcast"
EVENT_EPISODE_DOWNLOAD = "episode_download"
EVENT_BREAKING_ERROR = "breaking_error"

EVENT_COLUMNS = {
    EVENT_ACCESS_REQUEST: "notify_access_requests",
    EVENT_NEW_PODCAST: "notify_new_podcasts",
    EVENT_EPISODE_DOWNLOAD: "notify_episode_downloads",
    EVENT_BREAKING_ERROR: "notify_breaking_errors",
}

SEVERITY_TO_NOTIFY_TYPE = {
    "info": "info",
    "success": "success",
    "warning": "warning",
    "error": "failure",
}


@dataclass(frozen=True)
class NotificationSettings:
    enabled: bool
    urls: list[str]
    events: dict[str, bool]


def _parse_urls(raw_urls: str | None) -> list[str]:
    if not raw_urls:
        return []
    return [
        line.strip()
        for line in raw_urls.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def get_notification_settings() -> NotificationSettings:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT notifications_enabled,
                   notification_urls,
                   notify_access_requests,
                   notify_new_podcasts,
                   notify_episode_downloads,
                   notify_breaking_errors
            FROM app_settings
            WHERE id = 1
            """
        ).fetchone()

    if not row:
        return NotificationSettings(enabled=False, urls=[], events={})

    return NotificationSettings(
        enabled=bool(row["notifications_enabled"]),
        urls=_parse_urls(row["notification_urls"]),
        events={
            event: bool(row[column])
            for event, column in EVENT_COLUMNS.items()
        },
    )


def event_enabled(event: str, settings: NotificationSettings | None = None) -> bool:
    settings = settings or get_notification_settings()
    return settings.enabled and bool(settings.urls) and settings.events.get(event, False)


def _send_with_apprise(urls: list[str], title: str, body: str, severity: str) -> bool:
    try:
        import apprise
    except ImportError:
        logger.warning("Notifications are enabled but the apprise package is not installed")
        return False

    notifier = apprise.Apprise()
    valid_targets = 0
    for url in urls:
        try:
            if notifier.add(url):
                valid_targets += 1
        except Exception:
            logger.warning("Ignoring invalid Apprise notification URL")

    if valid_targets == 0:
        logger.warning("Notifications are enabled but no valid Apprise URLs are configured")
        return False

    notify_type = getattr(
        apprise.NotifyType,
        SEVERITY_TO_NOTIFY_TYPE.get(severity, "info").upper(),
        apprise.NotifyType.INFO,
    )
    return bool(notifier.notify(title=title, body=body, notify_type=notify_type))


def send_notification(event: str, title: str, body: str, severity: str = "info") -> bool:
    try:
        settings = get_notification_settings()
        if not event_enabled(event, settings):
            return False
        return _send_with_apprise(settings.urls, title, body, severity)
    except Exception as exc:
        logger.warning("Failed to send notification for %s: %s", event, exc)
        return False


async def send_notification_async(event: str, title: str, body: str, severity: str = "info") -> bool:
    return await asyncio.to_thread(send_notification, event, title, body, severity)


def send_test_notification() -> bool:
    try:
        settings = get_notification_settings()
        if not settings.enabled or not settings.urls:
            return False
        return _send_with_apprise(
            settings.urls,
            "Podcast Ad Remover test notification",
            "Notifications are configured and reachable.",
            "success",
        )
    except Exception as exc:
        logger.warning("Failed to send test notification: %s", exc)
        return False
