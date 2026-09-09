import re
from pathlib import Path

from app.core.worker_health import record_feed_check, record_heartbeat
from app.infra.database import get_db_connection, init_db

# A stored SQLite/Python timestamp before normalization: 'YYYY-MM-DD HH:MM:SS'
# or 'YYYY-MM-DDTHH:MM:SS.ffffff'. Anything matching this that does not end in
# 'Z' escaped the serializer.
_RAW_DB_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def unnormalized_timestamps(row: dict) -> dict:
    """Timestamp-shaped fields in `row` that are not Z-suffixed UTC."""
    return {
        key: value
        for key, value in row.items()
        if isinstance(value, str)
        and _RAW_DB_TIMESTAMP.match(value)
        and not value.endswith("Z")
    }


def seed_queue_and_history():
    """One episode in the queue with a live job, one recently completed."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id, feed_url, title, slug) "
            "VALUES (1, 'https://example.com/feed.xml', 'Show', 'show')"
        )
        conn.execute(
            """
            INSERT INTO episodes (id, subscription_id, guid, title, pub_date, original_url,
                                  duration, status, next_retry_at, discovered_at)
            VALUES (7, 1, 'g1', 'Queued Ep', '2026-01-01T10:00:00',
                    'https://cdn.example.com/1.mp3', 60, 'processing',
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
        )
        conn.execute(
            """
            INSERT INTO episodes (id, subscription_id, guid, title, pub_date, original_url,
                                  duration, status, processed_at, discovered_at, processed_at_is_utc)
            VALUES (8, 1, 'g2', 'Done Ep', '2026-01-01T10:00:00',
                    'https://cdn.example.com/2.mp3', 60, 'completed',
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            """
        )
        conn.execute(
            "INSERT INTO jobs (episode_id, type, status, locked_at, next_run_at) "
            "VALUES (7, 'process_episode', 'running', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        conn.commit()


def test_queue_template_uses_local_time_filter():
    source = Path("app/web/templates/admin/queue.html").read_text(encoding="utf-8")

    assert "operation_status.next_feed_check|local_time('datetime')" in source
    assert "operation_status.next_retry.next_run_at|local_time('datetime')" in source
    assert "item.next_retry_at|local_time('datetime')" in source
    assert "item.processed_at|local_time('datetime', item.processed_at_is_utc)" in source
    assert "item.processed_at[:16]" not in source


def test_queue_refresh_js_localizes_timestamps():
    source = Path("app/web/templates/admin/queue.html").read_text(encoding="utf-8")

    # Both refreshed timestamps go through the one localizer...
    assert "setLocalTime(feedCheck, status.next_feed_check, '')" in source
    assert "setLocalTime(retry, status.next_retry.next_run_at, 'Retry: ')" in source
    # ...which formats via AppLocalTime, read lazily and guarded so a missing
    # local-time.js degrades to the raw UTC instant instead of throwing and
    # stranding the handler updates that follow it.
    assert "const lt = window.AppLocalTime;" in source
    assert "lt ? lt.formatIso(iso, 'datetime') : ''" in source
    # Assigning textContent discards the hydrated <time>, so the UTC instant
    # has to survive on the wrapper's tooltip.
    assert "(UTC: ${iso})" in source


def test_api_queue_status_serializes_timestamps_with_z(isolated_data_dir):
    from app.web.router import api_queue_status
    import asyncio

    init_db()
    record_heartbeat("test-worker")
    record_feed_check(30)
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id, feed_url, title, slug) "
            "VALUES (1, 'https://example.com/feed.xml', 'Show', 'show')"
        )
        conn.execute(
            """
            INSERT INTO episodes (id, subscription_id, guid, title, pub_date, original_url, duration,
                                  status, processed_at, processed_at_is_utc)
            VALUES (7, 1, 'g1', 'Ep', '2026-01-01T10:00:00', 'https://cdn.example.com/1.mp3', 60,
                    'completed', CURRENT_TIMESTAMP, 1)
            """
        )
        conn.commit()

    payload = asyncio.run(api_queue_status(user=None))

    assert payload["operation_status"]["next_feed_check"].endswith("Z")
    assert payload["recently_processed"][0]["processed_at"].endswith("Z")


def test_api_queue_status_leaves_no_raw_timestamps_in_any_row(isolated_data_dir):
    """Every timestamp-shaped field, not just the ones a template renders.

    discovered_at and the job_* aliases from get_queue()'s join were shipping
    raw 'YYYY-MM-DD HH:MM:SS' strings alongside Z-suffixed siblings in the
    same row.
    """
    from app.web.router import api_queue_status
    import asyncio

    init_db()
    seed_queue_and_history()

    payload = asyncio.run(api_queue_status(user=None))

    assert payload["queue"], "seed did not produce a queue row"
    assert payload["recently_processed"], "seed did not produce a history row"
    for row in payload["queue"] + payload["recently_processed"]:
        assert unnormalized_timestamps(row) == {}


def test_v1_queue_endpoint_matches_the_web_queue_payload(isolated_data_dir):
    """The two endpoints build one payload, so they cannot drift again."""
    from app.web.router import api_queue_status
    import asyncio

    from tests.conftest import auth_header, enable_ai_api, make_client
    from app.infra.repository import ApiTokenRepository

    init_db()
    enable_ai_api()
    seed_queue_and_history()
    token = ApiTokenRepository().create("Reader", scopes=["read"])

    response = make_client().get("/api/v1/queue", headers=auth_header(token))
    assert response.status_code == 200
    v1_payload = response.json()

    assert v1_payload["queue"], "seed did not produce a queue row"
    assert v1_payload["recently_processed"], "seed did not produce a history row"
    for row in v1_payload["queue"] + v1_payload["recently_processed"]:
        assert unnormalized_timestamps(row) == {}

    web_payload = asyncio.run(api_queue_status(user=None))
    assert v1_payload["queue"] == web_payload["queue"]
    assert v1_payload["recently_processed"] == web_payload["recently_processed"]


def test_queue_tooltips_use_the_attribute_safe_filter():
    """The two title="" slots must stay on utc_isoformat, not local_time."""
    source = Path("app/web/templates/admin/queue.html").read_text(encoding="utf-8")

    assert 'title="Waiting for API quota reset: {{ item.next_retry_at|utc_isoformat }}"' in source
    assert 'title="Retry scheduled: {{ item.next_retry_at|utc_isoformat }}"' in source
