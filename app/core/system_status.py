import os
import shutil
from datetime import datetime, timedelta

from app.core.config import settings
from app.infra.database import get_db_connection


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "Unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _read_load_average():
    if hasattr(os, "getloadavg"):
        try:
            one, five, fifteen = os.getloadavg()
            return {
                "one_minute": round(one, 2),
                "five_minutes": round(five, 2),
                "fifteen_minutes": round(fifteen, 2),
            }
        except OSError:
            return None
    return None


def _read_memory():
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        return None

    values = {}
    with open(meminfo_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = int(parts[1]) * 1024

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None

    used = total - available
    return {
        "total": total,
        "used": used,
        "available": available,
        "used_percent": round((used / total) * 100, 1),
        "total_display": _format_bytes(total),
        "used_display": _format_bytes(used),
        "available_display": _format_bytes(available),
    }


def get_operation_status() -> dict:
    disk = shutil.disk_usage(settings.DATA_DIR)
    now = datetime.now()

    with get_db_connection() as conn:
        active_job = conn.execute("""
            SELECT j.*, e.title AS episode_title, s.title AS podcast_title, e.progress, e.processing_step
            FROM jobs j
            JOIN episodes e ON e.id = j.episode_id
            JOIN subscriptions s ON s.id = e.subscription_id
            WHERE j.status = 'running'
            ORDER BY j.locked_at ASC
            LIMIT 1
        """).fetchone()

        next_retry = conn.execute("""
            SELECT j.next_run_at, j.status, e.title AS episode_title, s.title AS podcast_title, j.error
            FROM jobs j
            JOIN episodes e ON e.id = j.episode_id
            JOIN subscriptions s ON s.id = e.subscription_id
            WHERE j.status IN ('retry_scheduled', 'rate_limited')
              AND j.next_run_at IS NOT NULL
            ORDER BY j.next_run_at ASC
            LIMIT 1
        """).fetchone()

        settings_row = conn.execute(
            "SELECT check_interval_minutes FROM app_settings WHERE id = 1"
        ).fetchone()

        queue_counts = {
            row["status"]: row["count"]
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        }

    interval_minutes = settings_row["check_interval_minutes"] if settings_row else settings.CHECK_INTERVAL_MINUTES
    next_feed_check = now + timedelta(minutes=interval_minutes or settings.CHECK_INTERVAL_MINUTES)

    return {
        "active_job": dict(active_job) if active_job else None,
        "next_retry": dict(next_retry) if next_retry else None,
        "next_feed_check": next_feed_check.strftime("%Y-%m-%d %H:%M:%S"),
        "queue_counts": queue_counts,
        "load_average": _read_load_average(),
        "memory": _read_memory(),
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "used_percent": round((disk.used / disk.total) * 100, 1) if disk.total else 0,
            "total_display": _format_bytes(disk.total),
            "used_display": _format_bytes(disk.used),
            "free_display": _format_bytes(disk.free),
        },
    }
