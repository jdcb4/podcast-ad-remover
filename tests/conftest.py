import calendar
import os
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.router import router as ai_api_router
from app.core.config import settings
from app.infra.database import get_db_connection


def make_client() -> TestClient:
    """A bare FastAPI app mounting only the v1 API router, for v1 endpoint tests."""
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


@pytest.fixture()
def isolated_data_dir(tmp_path):
    original_data_dir = settings.DATA_DIR
    settings.DATA_DIR = str(tmp_path)
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    os.makedirs(settings.PODCASTS_DIR, exist_ok=True)
    os.makedirs(settings.FEEDS_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    yield tmp_path
    settings.DATA_DIR = original_data_dir


def assert_local_clock_skewed():
    """Guard that the ambient timezone really is far from UTC.

    Without this, a test that relies on a local/UTC divergence passes
    vacuously on a UTC host or if the timezone fixture silently no-ops.
    """
    skew = abs(
        (datetime.now() - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
    )
    assert skew > 3600, (
        f"local clock is only {skew:.0f}s from UTC; "
        "the non_utc_timezone fixture is not in effect"
    )


# The UTC instant exercised by the DST regression test in
# tests/test_feed_pub_date_utc.py. Defined once here and imported by that
# test module, so the RSS pubDate string, the expected parsed datetime, and
# the DST-active check in assert_dst_active() below can never drift apart.
DST_TEST_INSTANT = datetime(2026, 8, 27, 5, 5, 0, tzinfo=timezone.utc)


def assert_dst_active():
    """Guard that DST is genuinely in effect for DST_TEST_INSTANT.

    Without this, a test that relies on the mktime()/fromtimestamp() DST
    offset mismatch passes vacuously if the tz database, or DST_TEST_INSTANT
    itself, ever changes so DST no longer applies on that date.
    """
    epoch = calendar.timegm(DST_TEST_INSTANT.utctimetuple())
    assert time.localtime(epoch).tm_isdst == 1, (
        "DST is not in effect for DST_TEST_INSTANT in this timezone; "
        "the dst_timezone fixture is not exercising a DST period"
    )


def _tz_fixture(tz_name, guard):
    """Shared machinery behind non_utc_timezone and dst_timezone below.

    Skips on non-POSIX platforms, switches to tz_name via a private
    MonkeyPatch (undoing the shared, function-scoped `monkeypatch` here
    would also revert patches made by the test itself), runs `guard()` to
    prove the change actually took effect, then restores the original zone.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("TZ manipulation requires a POSIX platform")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TZ", tz_name)
        time.tzset()
        guard()
        yield
    # The env var is restored by the context manager above; tzset() must run
    # afterwards so the process picks the original zone back up.
    time.tzset()


@pytest.fixture()
def non_utc_timezone():
    """Run the test at UTC+14 so any local-clock write diverges from UTC by hours."""
    yield from _tz_fixture("Pacific/Kiritimati", assert_local_clock_skewed)


@pytest.fixture()
def dst_timezone():
    """Run the test in America/New_York while DST is active.

    feedparser always sets tm_isdst=0 on published_parsed, so mktime()
    encodes it using the zone's standard offset while fromtimestamp()
    decodes using the zone's actual (DST) offset for that date — a
    mismatch that only shows up in a DST-observing zone during its DST
    period.
    """
    yield from _tz_fixture("America/New_York", assert_dst_active)
