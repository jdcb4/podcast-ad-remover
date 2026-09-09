"""Naive-UTC time helpers.

New timestamps use naive UTC, matching SQLite's CURRENT_TIMESTAMP.
Historical last_login and processed_at values can have an unknown timezone;
their provenance flags prevent serializers from treating them as UTC.
"""
from collections.abc import Iterable
from datetime import datetime, timezone


def now_utc() -> datetime:
    """Current UTC time as a naive datetime, matching CURRENT_TIMESTAMP."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_iso(value: str | datetime | None, *, assume_utc: bool = True) -> str | None:
    """Normalize a stored timestamp to 'YYYY-MM-DDTHH:MM:SSZ'.

    Accepts the two naive string shapes found in the database
    ('YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS.ffffff'), Z-suffixed and
    offset-suffixed ISO strings, and datetime objects (naive or aware).
    Timezone-aware input is converted to UTC before formatting. Returns None
    when the value cannot be interpreted. Set assume_utc=False for legacy
    values whose timezone is unknown; only an explicit offset can identify
    their UTC instant.
    """
    if value is None or value == "":
        return None
    dt = value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    elif not assume_utc:
        return None
    return dt.replace(microsecond=0).isoformat() + "Z"


# Timestamp columns that reach a browser or an API response body: the
# `jobs` columns (created_at, locked_at, next_run_at, updated_at) and the
# `episodes` columns (discovered_at, next_retry_at, processed_at, pub_date).
#
# This is the known-serialized set, NOT every timestamp column in the
# schema. Around a dozen others (revoked_at, last_used_at, last_seen_at,
# requested_at, reviewed_at, last_checked_at, last_check_error_at,
# deletion_started_at, deletion_updated_at, login_attempts.timestamp, ...)
# are deliberately absent because nothing serializes them today. Add a key
# here when one starts being rendered or returned.
TIMESTAMP_KEYS: tuple[str, ...] = (
    "created_at",
    "discovered_at",
    "locked_at",
    "next_retry_at",
    "next_run_at",
    "processed_at",
    "pub_date",
    "updated_at",
)

# EpisodeRepository.get_queue() exposes its joined `jobs` row under `job_*`
# aliases, which the unprefixed names above cannot match.
QUEUE_ROW_TIMESTAMP_KEYS: tuple[str, ...] = TIMESTAMP_KEYS + (
    "job_locked_at",
    "job_next_run_at",
)

TIMESTAMP_PROVENANCE_KEYS = {
    "processed_at": "processed_at_is_utc",
    "last_login": "last_login_is_utc",
}


def with_utc_timestamps(row: dict, keys: Iterable[str] = TIMESTAMP_KEYS) -> dict:
    """Return a copy of `row` with its timestamp fields as Z-suffixed ISO.

    `row` itself is never modified. Only keys that are present and truthy
    are rewritten; every other field is carried across untouched, and a
    value utc_iso() cannot interpret is left exactly as it was found —
    never dropped and never replaced with None. Ambiguous historical fields
    retain their raw value and provenance flag instead of acquiring a false Z.
    """
    out = dict(row)
    for key in keys:
        value = out.get(key)
        if value:
            provenance_key = TIMESTAMP_PROVENANCE_KEYS.get(key)
            known_utc = bool(out.get(provenance_key)) if provenance_key else True
            out[key] = utc_iso(value, assume_utc=known_utc) or value
    return out
