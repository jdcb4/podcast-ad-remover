from datetime import datetime, timedelta, timezone

import pytest

from app.core.processor import Processor
from app.core.system_status import get_operation_status
from app.infra.database import get_db_connection, init_db
from app.core.worker_health import record_feed_check, record_heartbeat
from app.infra.repository import EpisodeRepository
from tests.conftest import assert_local_clock_skewed


def seed_processing_episode():
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id, feed_url, title, slug) "
            "VALUES (1, 'https://example.com/feed.xml', 'Show', 'show')"
        )
        conn.execute(
            """
            INSERT INTO episodes (id, subscription_id, guid, title, pub_date, original_url, duration, status)
            VALUES (7, 1, 'g1', 'Ep', '2026-01-01T10:00:00', 'https://cdn.example.com/1.mp3', 60, 'processing')
            """
        )
        conn.commit()


def seed_retry_job(next_run_at: str):
    seed_processing_episode()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (episode_id, type, status, next_run_at) "
            "VALUES (7, 'process_episode', 'retry_scheduled', ?)",
            (next_run_at,),
        )
        conn.commit()


def seed_running_job(timestamp: str):
    seed_processing_episode()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (episode_id, type, status, locked_at, next_run_at, created_at, updated_at) "
            "VALUES (7, 'process_episode', 'running', ?, ?, ?, ?)",
            (timestamp, timestamp, timestamp, timestamp),
        )
        conn.commit()


def test_update_status_writes_processed_at_in_utc(isolated_data_dir, non_utc_timezone):
    assert_local_clock_skewed()
    init_db()
    seed_processing_episode()

    EpisodeRepository().update_status(7, "completed", filename="/tmp/e.mp3", file_size=10)

    with get_db_connection() as conn:
        row = conn.execute("SELECT processed_at FROM episodes WHERE id = 7").fetchone()
    stored = datetime.fromisoformat(str(row["processed_at"]).replace(" ", "T"))
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((utc_now - stored).total_seconds()) < 60


def test_next_feed_check_is_utc_iso_z(isolated_data_dir, non_utc_timezone):
    assert_local_clock_skewed()
    init_db()
    # The scheduler now supplies this value; without a heartbeat and a
    # recorded check, get_operation_status() reports a label instead.
    record_heartbeat("test-worker")
    record_feed_check(30)

    status = get_operation_status()

    value = status["next_feed_check"]
    assert value.endswith("Z"), f"expected Z-suffixed ISO, got {value!r}"
    parsed = datetime.fromisoformat(value.replace("Z", ""))
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    # The next check is utc_now + check interval (default well under 12h).
    # Computed from the +14h local clock this lands >13h out and fails.
    assert timedelta(0) <= (parsed - utc_now) <= timedelta(hours=12)



def test_next_feed_check_keeps_its_label_when_nothing_is_scheduled(isolated_data_dir):
    """utc_iso() cannot parse these labels and returns None for them.

    Normalizing unconditionally would replace the operator-facing status text
    with None, so the raw value has to survive when it is not a timestamp.
    """
    init_db()

    assert get_operation_status()["next_feed_check"] == "Not yet scheduled"

def test_next_retry_next_run_at_is_utc_iso_z(isolated_data_dir):
    init_db()
    seed_retry_job("2026-03-04 05:06:07")

    status = get_operation_status()

    assert status["next_retry"]["next_run_at"] == "2026-03-04T05:06:07Z"
    # The rest of the row must survive the rewrite untouched.
    assert status["next_retry"]["episode_title"] == "Ep"
    assert status["next_retry"]["status"] == "retry_scheduled"


def test_active_job_timestamps_are_utc_iso_z(isolated_data_dir):
    init_db()
    seed_running_job("2026-03-04 05:06:07")

    status = get_operation_status()

    active_job = status["active_job"]
    assert active_job["locked_at"] == "2026-03-04T05:06:07Z"
    assert active_job["next_run_at"] == "2026-03-04T05:06:07Z"
    assert active_job["created_at"] == "2026-03-04T05:06:07Z"
    assert active_job["updated_at"] == "2026-03-04T05:06:07Z"
    # The rest of the row must survive the rewrite untouched.
    assert active_job["episode_title"] == "Ep"
    assert active_job["status"] == "running"


def test_next_retry_keeps_uninterpretable_next_run_at(isolated_data_dir):
    """The `or` fallback: an unparseable stored value passes through unchanged."""
    init_db()
    seed_retry_job("not-a-timestamp")

    status = get_operation_status()

    assert status["next_retry"]["next_run_at"] == "not-a-timestamp"


@pytest.mark.asyncio
async def test_cleanup_old_logs_uses_utc_cutoff(isolated_data_dir, non_utc_timezone):
    """Rows inside the 30-day window must survive even on a +14h local clock.

    The 30-day window is 720h. Pre-fix the cutoff was `local_now - 720h`,
    which at UTC+14 resolves to `utc_now - 706h`, so the 716h-old row fell
    outside it and was wrongly deleted.
    """
    assert_local_clock_skewed()
    init_db()
    # Ages in whole hours: SQLite rejects a combined "-29 days -20 hours"
    # modifier and silently yields NULL, which would never match the DELETE.
    ages = {"recent": "-24 hours", "boundary": "-716 hours", "ancient": "-744 hours"}
    with get_db_connection() as conn:
        for username, age in ages.items():
            conn.execute(
                "INSERT INTO login_attempts (username, ip_address, success, timestamp) "
                "VALUES (?, '127.0.0.1', 0, datetime('now', ?))",
                (username, age),
            )
        conn.commit()
        seeded = conn.execute(
            "SELECT COUNT(*) AS n FROM login_attempts WHERE timestamp IS NOT NULL"
        ).fetchone()["n"]
    assert seeded == len(ages), "seeded timestamps must be non-NULL to be comparable"

    await Processor().cleanup_old_logs()

    with get_db_connection() as conn:
        survivors = {
            row["username"]
            for row in conn.execute("SELECT username FROM login_attempts").fetchall()
        }
    assert survivors == {"recent", "boundary"}
