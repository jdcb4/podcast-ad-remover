from datetime import datetime, timedelta, timezone

from app.core.time_utils import now_utc, utc_iso, with_utc_timestamps


def test_now_utc_is_naive():
    assert now_utc().tzinfo is None


def test_now_utc_tracks_the_utc_clock():
    reference = datetime.now(timezone.utc).replace(tzinfo=None)
    assert abs((now_utc() - reference).total_seconds()) < 5


def test_utc_iso_from_sqlite_current_timestamp_string():
    assert utc_iso("2026-08-27 17:47:12") == "2026-08-27T17:47:12Z"


def test_utc_iso_from_python_isoformat_string_with_microseconds():
    assert utc_iso("2026-08-27T17:47:12.345678") == "2026-08-27T17:47:12Z"


def test_utc_iso_from_datetime_object():
    assert utc_iso(datetime(2026, 8, 27, 17, 47, 12, 345678)) == "2026-08-27T17:47:12Z"


def test_utc_iso_passes_through_z_suffixed_strings():
    assert utc_iso("2026-08-27T17:47:12Z") == "2026-08-27T17:47:12Z"


def test_utc_iso_returns_none_for_empty_and_garbage():
    assert utc_iso(None) is None
    assert utc_iso("") is None
    assert utc_iso("not a date") is None
    assert utc_iso(12345) is None


def test_utc_iso_converts_aware_offset_string_to_utc():
    assert utc_iso("2026-08-27T17:47:12+05:00") == "2026-08-27T12:47:12Z"


def test_utc_iso_converts_aware_offset_string_with_microseconds_to_utc():
    assert utc_iso("2026-08-27T17:47:12.345678+05:00") == "2026-08-27T12:47:12Z"


def test_utc_iso_converts_aware_datetime_object_to_utc():
    aware = datetime(2026, 8, 27, 17, 47, 12, 345678, tzinfo=timezone(timedelta(hours=5)))
    assert utc_iso(aware) == "2026-08-27T12:47:12Z"


def test_with_utc_timestamps_returns_a_copy_and_leaves_the_input_alone():
    row = {"processed_at": "2026-08-27 17:47:12", "title": "Ep"}

    out = with_utc_timestamps(row)

    assert out["processed_at"] == "2026-08-27T17:47:12Z"
    assert out["title"] == "Ep"
    assert out is not row
    assert row == {"processed_at": "2026-08-27 17:47:12", "title": "Ep"}


def test_with_utc_timestamps_passes_through_values_utc_iso_cannot_read():
    row = {"next_run_at": "not-a-timestamp", "processed_at": None, "pub_date": ""}

    out = with_utc_timestamps(row)

    assert out["next_run_at"] == "not-a-timestamp"
    assert out["processed_at"] is None
    assert out["pub_date"] == ""


def test_with_utc_timestamps_honours_an_explicit_key_list():
    row = {"processed_at": "2026-08-27 17:47:12", "job_locked_at": "2026-08-27 17:47:12"}

    out = with_utc_timestamps(row, ("job_locked_at",))

    assert out["job_locked_at"] == "2026-08-27T17:47:12Z"
    assert out["processed_at"] == "2026-08-27 17:47:12"
