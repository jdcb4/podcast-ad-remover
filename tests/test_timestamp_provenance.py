import sqlite3
from datetime import datetime

from fastapi.testclient import TestClient

from app.core.models import User
from app.core.time_utils import utc_iso, with_utc_timestamps
from app.infra import database
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository
from app.main import app
from app.web.auth_utils import hash_password
from app.web.template_filters import local_time


MIGRATION = "20260910_0015_timestamp_provenance"
OLD_LOGIN = "2026-09-01 12:00:00"


def seed_history(processed_at):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO users(id,username,password_hash,is_admin,last_login) VALUES(1,'reader',?,1,?)",
            (hash_password("test-password"), OLD_LOGIN),
        )
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug) VALUES(1,'https://example.com/feed','Show','show')")
        conn.execute(
            """INSERT INTO episodes(id,subscription_id,guid,title,original_url,status,processed_at,
                                    local_filename,published_guid,pub_date)
               VALUES(7,1,'episode','Episode','https://example.com/audio','completed',?,
                      'existing.mp3','stable-guid','2026-09-01 02:00:00')""",
            (processed_at,),
        )
        conn.commit()


def test_upgrade_preserves_historical_times_and_creates_a_usable_backup(isolated_data_dir, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(database, "FORMAL_MIGRATIONS", [m for m in database.FORMAL_MIGRATIONS if m[0] != MIGRATION])
        init_db()
        seed_history("2026-01-02T12:34:56.123456")

    init_db()
    init_db()

    with get_db_connection() as conn:
        user = dict(conn.execute("SELECT * FROM users WHERE id=1").fetchone())
        episode = dict(conn.execute("SELECT * FROM episodes WHERE id=7").fetchone())
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE version=?", (MIGRATION,)).fetchone()[0] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert user["last_login"] == OLD_LOGIN
    assert user["last_login_is_utc"] == 0
    assert episode["processed_at"] == "2026-01-02T12:34:56.123456"
    assert episode["processed_at_is_utc"] == 0
    assert episode["local_filename"] == "existing.mp3"
    assert episode["published_guid"] == "stable-guid"
    assert not User.model_validate(user).last_login_is_utc
    assert not EpisodeRepository().get_by_id(7).processed_at_is_utc

    backups = list((isolated_data_dir / "backups").glob("*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert "last_login_is_utc" not in {r[1] for r in conn.execute("PRAGMA table_info(users)")}
        assert conn.execute("SELECT last_login FROM users WHERE id=1").fetchone()[0] == OLD_LOGIN
        assert conn.execute("SELECT processed_at FROM episodes WHERE id=7").fetchone()[0] == episode["processed_at"]


def test_legacy_history_is_labelled_and_keeps_its_raw_api_value(isolated_data_dir):
    init_db()
    # A recent value ensures the real queue/history endpoint includes this row.
    with get_db_connection() as conn:
        recent = conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
    seed_history(recent)
    client = TestClient(app)

    users = client.get("/admin/users")
    history = client.get("/admin/queue")
    assert users.status_code == history.status_code == 200
    assert 'data-timezone="unknown"' in users.text
    assert "01 Sep 12:00 (timezone unknown)" in users.text
    assert 'datetime="2026-09-01T12:00:00Z"' not in users.text
    assert 'data-timezone="unknown"' in history.text

    response = client.get("/api/queue/status")
    assert response.status_code == 200
    row = response.json()["recently_processed"][0]
    assert row["processed_at"] == recent
    assert row["processed_at_is_utc"] == 0
    assert row["pub_date"] == "2026-09-01T02:00:00Z"


def test_new_login_and_completion_atomically_replace_legacy_provenance(isolated_data_dir, monkeypatch):
    init_db()
    seed_history("2026-09-01 12:00:00")
    fixed = datetime(2026, 9, 10, 2, 3, 4)
    monkeypatch.setattr("app.web.router.now_utc", lambda: fixed)
    monkeypatch.setattr("app.infra.repository.now_utc", lambda: fixed)
    client = TestClient(app)
    response = client.post("/login", data={"username": "reader", "password": "test-password"}, follow_redirects=False)
    assert response.status_code == 302

    repo = EpisodeRepository()
    # A rejected completion must not relabel the old timestamp as UTC.
    repo.update_status(7, "completed")
    assert not repo.get_by_id(7).processed_at_is_utc
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET status='processing' WHERE id=7")
        conn.commit()
    repo.update_status(7, "completed")
    with get_db_connection() as conn:
        user = dict(conn.execute("SELECT * FROM users WHERE id=1").fetchone())
        episode = dict(conn.execute("SELECT * FROM episodes WHERE id=7").fetchone())
    assert user["last_login"] == episode["processed_at"] == "2026-09-10 02:03:04"
    assert user["last_login_is_utc"] == episode["processed_at_is_utc"] == 1
    assert with_utc_timestamps(episode)["processed_at"] == "2026-09-10T02:03:04Z"
    assert 'datetime="2026-09-10T02:03:04Z"' in client.get("/admin/users").text

    repo.update_status(7, "pending")
    assert repo.get_by_id(7).processed_at_is_utc
    assert repo.get_by_id(7).processed_at == fixed
    init_db()
    assert repo.get_by_id(7).processed_at_is_utc


def test_timezone_unknown_values_are_never_guessed_or_hydrated():
    assert utc_iso(OLD_LOGIN, assume_utc=False) is None
    assert with_utc_timestamps({"processed_at": OLD_LOGIN})["processed_at"] == OLD_LOGIN
    html = str(local_time(OLD_LOGIN, "datetime", False))
    assert "01 Sep 2026 12:00 (timezone unknown)" in html
    assert "<time " not in html
    assert "<script>" not in str(local_time("<script>x</script>", known_utc=False))

    # Explicit offsets provide enough information even without a provenance flag.
    aware = "2026-09-01T12:00:00+10:00"
    assert utc_iso(aware, assume_utc=False) == "2026-09-01T02:00:00Z"
    html = str(local_time(aware, "datetime", False))
    assert 'datetime="2026-09-01T02:00:00Z"' in html
    assert ">01 Sep 2026 02:00</time>" in html
    assert "timezone unknown" not in html
