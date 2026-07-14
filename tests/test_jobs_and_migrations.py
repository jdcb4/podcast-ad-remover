import sqlite3

from app.infra.database import get_db_connection, init_db
from app.core.models import SubscriptionCreate
from app.infra.repository import EpisodeRepository, JobRepository, SubscriptionRepository
from app.core.config import settings


def test_fresh_init_does_not_create_migration_backup(isolated_data_dir):
    init_db()

    backup_dir = isolated_data_dir / "backups"

    assert not backup_dir.exists()


def test_existing_database_is_backed_up_before_formal_migrations(isolated_data_dir):
    db_path = isolated_data_dir / "db" / "podcasts.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")

    init_db()

    backup_dir = isolated_data_dir / "backups"
    backups = list(backup_dir.glob("podcasts-before-migration-*.db"))

    assert settings.DB_PATH == str(db_path)
    assert len(backups) == 1


def test_init_db_creates_formal_migration_tables(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        migrations = {
            row["version"]
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }

    assert "jobs" in tables
    assert "feed_tokens" in tables
    assert "api_tokens" in tables
    assert "api_rate_limits" in tables
    assert "user_subscriptions" in tables
    assert "schema_migrations" in tables
    assert "20260609_0001_jobs" in migrations
    assert "20260609_0002_feed_tokens" in migrations
    assert "20260612_0003_user_podcast_library" in migrations
    assert "20260612_0004_access_request_password_hash" in migrations
    assert "20260612_0005_notifications" in migrations
    assert "20260612_0006_tts_provider_settings" in migrations
    assert "20260617_0007_ai_api" in migrations

    with get_db_connection() as conn:
        access_request_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(access_requests)").fetchall()
        }
        settings_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(app_settings)").fetchall()
        }

    assert "password_hash" in access_request_columns
    assert {
        "notifications_enabled",
        "notification_urls",
        "notify_access_requests",
        "notify_new_podcasts",
        "notify_episode_downloads",
        "notify_breaking_errors",
        "tts_provider",
        "gemini_tts_voice",
        "gemini_tts_model_cascade",
        "ai_api_enabled",
        "ai_api_default_requests_per_minute",
        "ai_api_default_requests_per_day",
        "ai_api_unauth_requests_per_minute",
        "whisper_compute_type",
    }.issubset(settings_columns)


def test_init_db_creates_resource_tuning_defaults(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        row = conn.execute("""
            SELECT whisper_cpu_threads, ffmpeg_threads, unload_whisper_after_job,
                   whisper_compute_type,
                   ai_model_cascade, openrouter_model,
                   notifications_enabled, notification_urls,
                   notify_access_requests, notify_new_podcasts,
                   notify_episode_downloads, notify_breaking_errors,
                   tts_provider, gemini_tts_voice, gemini_tts_model_cascade,
                   ai_api_enabled, ai_api_default_requests_per_minute,
                   ai_api_default_requests_per_day, ai_api_unauth_requests_per_minute
            FROM app_settings WHERE id = 1
        """).fetchone()

    assert row["whisper_cpu_threads"] == 0
    assert row["ffmpeg_threads"] == 0
    assert row["unload_whisper_after_job"] == 0
    assert row["whisper_compute_type"] == "float32"
    assert "gemini-3.5-flash" in row["ai_model_cascade"]
    assert "gemini-3-flash" in row["ai_model_cascade"]
    assert "gemini-3.1-flash-lite" in row["ai_model_cascade"]
    assert "google/gemini-3.5-flash" in row["openrouter_model"]
    assert "google/gemini-3-flash" in row["openrouter_model"]
    assert "google/gemini-3.1-flash-lite" in row["openrouter_model"]
    assert row["notifications_enabled"] == 0
    assert row["notification_urls"] is None
    assert row["notify_access_requests"] == 1
    assert row["notify_new_podcasts"] == 1
    assert row["notify_episode_downloads"] == 1
    assert row["notify_breaking_errors"] == 1
    assert row["tts_provider"] == "piper"
    assert row["gemini_tts_voice"] == "Orus"
    assert "gemini-3.1-flash-tts-preview" in row["gemini_tts_model_cascade"]
    assert "gemini-2.5-flash-preview-tts" in row["gemini_tts_model_cascade"]
    assert row["ai_api_enabled"] == 0
    assert row["ai_api_default_requests_per_minute"] == 60
    assert row["ai_api_default_requests_per_day"] == 1000
    assert row["ai_api_unauth_requests_per_minute"] == 10


def test_database_connections_use_wal_and_busy_timeout(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

    assert journal_mode == "wal"
    assert busy_timeout == 30000


def test_legacy_database_rows_survive_and_pending_work_is_backfilled(isolated_data_dir):
    db_path = isolated_data_dir / "db" / "podcasts.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_url TEXT UNIQUE NOT NULL,
            title TEXT,
            slug TEXT UNIQUE,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_checked_at TIMESTAMP
        );
        CREATE TABLE app_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            whisper_model TEXT DEFAULT 'base',
            concurrent_downloads INTEGER DEFAULT 2,
            retention_days INTEGER DEFAULT 30
        );
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER NOT NULL,
            guid TEXT NOT NULL,
            title TEXT NOT NULL,
            pub_date TIMESTAMP,
            original_url TEXT NOT NULL,
            duration INTEGER,
            status TEXT DEFAULT 'pending',
            processed_at TIMESTAMP,
            error_message TEXT,
            local_filename TEXT,
            retry_count INTEGER DEFAULT 0,
            next_retry_at TIMESTAMP,
            FOREIGN KEY (subscription_id) REFERENCES subscriptions (id),
            UNIQUE(subscription_id, guid)
        );
        INSERT INTO app_settings (id, whisper_model, concurrent_downloads, retention_days)
        VALUES (1, 'tiny', 1, 7);
        INSERT INTO subscriptions (id, feed_url, title, slug)
        VALUES (1, 'https://example.com/feed.xml', 'Legacy Show', 'legacy-show');
        INSERT INTO episodes (subscription_id, guid, title, original_url, status, error_message, retry_count, next_retry_at)
        VALUES
            (1, 'pending-guid', 'Pending Episode', 'https://example.com/pending.mp3', 'pending', NULL, 0, NULL),
            (1, 'completed-guid', 'Completed Episode', 'https://example.com/completed.mp3', 'completed', NULL, 0, NULL),
            (1, 'failed-guid', 'Failed Episode', 'https://example.com/failed.mp3', 'failed', 'temporary failure', 2, CURRENT_TIMESTAMP);
        """
    )
    conn.commit()
    conn.close()

    init_db()

    with get_db_connection() as migrated:
        subscription = migrated.execute("SELECT title, slug FROM subscriptions WHERE id = 1").fetchone()
        app_settings = migrated.execute(
            "SELECT whisper_model, concurrent_downloads, retention_days FROM app_settings WHERE id = 1"
        ).fetchone()
        episodes = {
            row["guid"]: row
            for row in migrated.execute("SELECT guid, title, status FROM episodes").fetchall()
        }
        jobs = {
            row["guid"]: row
            for row in migrated.execute(
                """
                SELECT e.guid, j.status, j.attempts, j.error
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                """
            ).fetchall()
        }
        backups = list((isolated_data_dir / "backups").glob("podcasts-before-migration-*.db"))

    assert subscription["title"] == "Legacy Show"
    assert subscription["slug"] == "legacy-show"
    assert app_settings["whisper_model"] == "tiny"
    assert app_settings["concurrent_downloads"] == 1
    assert app_settings["retention_days"] == 7
    assert episodes["pending-guid"]["title"] == "Pending Episode"
    assert episodes["completed-guid"]["status"] == "completed"
    assert jobs["pending-guid"]["status"] == "queued"
    assert jobs["pending-guid"]["attempts"] == 0
    assert jobs["failed-guid"]["status"] == "retry_scheduled"
    assert jobs["failed-guid"]["error"] == "temporary failure"
    assert "completed-guid" not in jobs
    assert len(backups) == 1


def test_user_podcast_library_migration_backfills_existing_users(isolated_data_dir):
    db_path = isolated_data_dir / "db" / "podcasts.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_url TEXT UNIQUE NOT NULL,
            title TEXT,
            slug TEXT UNIQUE,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_checked_at TIMESTAMP
        );
        CREATE TABLE app_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            whisper_model TEXT DEFAULT 'base',
            concurrent_downloads INTEGER DEFAULT 2,
            retention_days INTEGER DEFAULT 30
        );
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER NOT NULL,
            guid TEXT NOT NULL,
            title TEXT NOT NULL,
            original_url TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            next_retry_at TIMESTAMP,
            UNIQUE(subscription_id, guid)
        );
        INSERT INTO app_settings (id) VALUES (1);
        INSERT INTO users (id, username, password_hash, is_admin)
        VALUES (1, 'admin', 'hash', 1), (2, 'viewer', 'hash', 0);
        INSERT INTO subscriptions (id, feed_url, title, slug)
        VALUES
            (1, 'https://example.com/one.xml', 'One', 'one'),
            (2, 'https://example.com/two.xml', 'Two', 'two');
        """
    )
    conn.commit()
    conn.close()

    init_db()

    with get_db_connection() as migrated:
        subscription_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(subscriptions)").fetchall()
        }
        rows = migrated.execute(
            "SELECT user_id, subscription_id FROM user_subscriptions ORDER BY user_id, subscription_id"
        ).fetchall()

    assert "owner_user_id" in subscription_columns
    assert [(row["user_id"], row["subscription_id"]) for row in rows] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_pending_episode_creates_and_claims_job(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        conn.commit()

    episode_repo = EpisodeRepository()
    created = episode_repo.create_or_ignore(
        {
            "subscription_id": subscription_id,
            "guid": "episode-one",
            "title": "Episode One",
            "pub_date": None,
            "original_url": "https://example.com/episode.mp3",
            "duration": 100,
            "description": "",
            "status": "pending",
            "file_size": 0,
        }
    )

    job_repo = JobRepository()

    assert job_repo.count_claimable() == 1

    claimed = job_repo.claim_due(1, worker_id="pytest-worker")

    assert created is True
    assert len(claimed) == 1
    assert claimed[0]["title"] == "Episode One"
    assert episode_repo.get_status(claimed[0]["id"]) == "processing"
    assert job_repo.count_claimable() == 0

    with get_db_connection() as conn:
        job = conn.execute("SELECT status, attempts, locked_by FROM jobs").fetchone()

    assert job["status"] == "running"
    assert job["attempts"] == 1
    assert job["locked_by"] == "pytest-worker"


def test_subscription_owner_is_member_and_removal_releases_ownership(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()

    with get_db_connection() as conn:
        user_id = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("owner", "hash"),
        ).lastrowid
        conn.commit()

    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/feed.xml"),
        "Owned Show",
        "owned-show",
        owner_user_id=user_id,
    )

    assert sub.owner_user_id == user_id
    assert repo.is_in_user_library(user_id, sub.id) is True

    removed = repo.remove_from_user_library(user_id, sub.id)
    updated = repo.get_by_id(sub.id)

    assert removed is True
    assert updated.owner_user_id is None
    assert repo.is_in_user_library(user_id, sub.id) is False


def test_admin_owner_reassignment_adds_new_owner_to_library(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()

    with get_db_connection() as conn:
        first_user_id = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("first-owner", "hash"),
        ).lastrowid
        second_user_id = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("second-owner", "hash"),
        ).lastrowid
        conn.commit()

    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/reassign.xml"),
        "Reassigned Show",
        "reassigned-show",
        owner_user_id=first_user_id,
    )

    updated = repo.set_owner(sub.id, second_user_id)
    reassigned = repo.get_by_id(sub.id)

    assert updated is True
    assert reassigned.owner_user_id == second_user_id
    assert repo.is_in_user_library(second_user_id, sub.id) is True
    assert repo.is_in_user_library(first_user_id, sub.id) is True


def test_admin_can_clear_subscription_owner(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()

    with get_db_connection() as conn:
        user_id = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("owner", "hash"),
        ).lastrowid
        conn.commit()

    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/clear-owner.xml"),
        "Clear Owner Show",
        "clear-owner-show",
        owner_user_id=user_id,
    )

    updated = repo.set_owner(sub.id, None)
    cleared = repo.get_by_id(sub.id)

    assert updated is True
    assert cleared.owner_user_id is None
    assert repo.is_in_user_library(user_id, sub.id) is True


def test_owner_reassignment_rejects_unknown_user(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()

    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/unknown-owner.xml"),
        "Unknown Owner Show",
        "unknown-owner-show",
    )

    updated = repo.set_owner(sub.id, 999)

    assert updated is False
    assert repo.get_by_id(sub.id).owner_user_id is None


def test_stale_running_job_is_recovered_and_claimable(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status, processing_step, progress)
            VALUES (?, ?, ?, ?, 'processing', 'transcribing', 42)
            """,
            (subscription_id, "stale-episode", "Stale Episode", "https://example.com/stale.mp3"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO jobs (episode_id, type, status, attempts, locked_at, locked_by, updated_at)
            VALUES (
                ?,
                'process_episode',
                'running',
                1,
                datetime(CURRENT_TIMESTAMP, '-2 days'),
                'dead-worker',
                datetime(CURRENT_TIMESTAMP, '-2 days')
            )
            """,
            (episode_id,),
        )
        conn.commit()

    job_repo = JobRepository()
    recovered = job_repo.recover_stale_running(max_age_minutes=60)

    assert recovered == 1
    assert job_repo.count_claimable() == 1

    with get_db_connection() as conn:
        job = conn.execute("SELECT status, locked_at, locked_by, error FROM jobs").fetchone()
        episode = conn.execute("SELECT status, processing_step, progress FROM episodes").fetchone()

    assert job["status"] == "queued"
    assert job["locked_at"] is None
    assert job["locked_by"] is None
    assert "worker interruption" in job["error"]
    assert episode["status"] == "pending"
    assert episode["processing_step"] == "retry scheduled after worker interruption"
    assert episode["progress"] == 0


def test_recent_running_job_is_not_recovered(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status, processing_step, progress)
            VALUES (?, ?, ?, ?, 'processing', 'transcribing', 42)
            """,
            (subscription_id, "recent-episode", "Recent Episode", "https://example.com/recent.mp3"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO jobs (episode_id, type, status, attempts, locked_at, locked_by)
            VALUES (?, 'process_episode', 'running', 1, CURRENT_TIMESTAMP, 'active-worker')
            """,
            (episode_id,),
        )
        conn.commit()

    job_repo = JobRepository()

    assert job_repo.recover_stale_running(max_age_minutes=60) == 0
    assert job_repo.count_running() == 1
    assert job_repo.count_claimable() == 0


def test_running_job_with_recent_heartbeat_is_not_recovered(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status, processing_step, progress)
            VALUES (?, ?, ?, ?, 'processing', 'transcribing', 42)
            """,
            (subscription_id, "heartbeat-episode", "Heartbeat Episode", "https://example.com/heartbeat.mp3"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO jobs (episode_id, type, status, attempts, locked_at, locked_by, updated_at)
            VALUES (?, 'process_episode', 'running', 1, datetime(CURRENT_TIMESTAMP, '-2 days'), 'active-worker', CURRENT_TIMESTAMP)
            """,
            (episode_id,),
        )
        conn.commit()

    job_repo = JobRepository()

    assert job_repo.recover_stale_running(max_age_minutes=60) == 0
    assert job_repo.count_running() == 1
    assert job_repo.count_claimable() == 0


def test_running_job_progress_updates_heartbeat(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status)
            VALUES (?, ?, ?, ?, 'processing')
            """,
            (subscription_id, "progress-episode", "Progress Episode", "https://example.com/progress.mp3"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO jobs (episode_id, type, status, attempts, locked_at, locked_by, updated_at)
            VALUES (?, 'process_episode', 'running', 1, datetime(CURRENT_TIMESTAMP, '-2 days'), 'active-worker', datetime(CURRENT_TIMESTAMP, '-2 days'))
            """,
            (episode_id,),
        )
        conn.commit()

    EpisodeRepository().update_progress(episode_id, "transcribing", 25)

    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT updated_at > datetime(CURRENT_TIMESTAMP, '-1 minute') AS heartbeat_is_recent
            FROM jobs
            WHERE episode_id = ?
            """,
            (episode_id,),
        ).fetchone()

    assert row["heartbeat_is_recent"] == 1


def test_inconsistent_running_job_does_not_block_capacity(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status, processing_step, progress)
            VALUES (?, ?, ?, ?, 'completed', 'completed', 100)
            """,
            (subscription_id, "orphan-episode", "Orphan Episode", "https://example.com/orphan.mp3"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO jobs (episode_id, type, status, attempts, locked_at, locked_by)
            VALUES (?, 'process_episode', 'running', 1, datetime(CURRENT_TIMESTAMP, '-1 minute'), 'dead-worker')
            """,
            (episode_id,),
        )
        conn.commit()

    job_repo = JobRepository()

    assert job_repo.count_running() == 0
    assert job_repo.recover_stale_running(max_age_minutes=60) == 1

    with get_db_connection() as conn:
        job = conn.execute("SELECT status, locked_at, locked_by, error FROM jobs").fetchone()

    assert job["status"] == "completed"
    assert job["locked_at"] is None
    assert job["locked_by"] is None
    assert "inconsistent" in job["error"]


def test_pending_episode_without_active_job_is_repaired(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            (subscription_id, "missing-job", "Missing Job Episode", "https://example.com/missing.mp3"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO jobs (episode_id, type, status)
            VALUES (?, 'process_episode', 'cancelled')
            """,
            (episode_id,),
        )
        conn.commit()

    job_repo = JobRepository()

    assert job_repo.count_claimable() == 0
    assert job_repo.repair_missing_active_jobs() == 1
    assert job_repo.count_claimable() == 1

    with get_db_connection() as conn:
        active_job = conn.execute(
            """
            SELECT status
            FROM jobs
            WHERE episode_id = ?
              AND type = 'process_episode'
              AND status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
            """,
            (episode_id,),
        ).fetchone()

    assert active_job["status"] == "queued"


def test_failed_episode_without_retry_job_is_repaired(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status, next_retry_at)
            VALUES (?, ?, ?, ?, 'failed', CURRENT_TIMESTAMP)
            """,
            (subscription_id, "missing-retry", "Missing Retry Episode", "https://example.com/retry.mp3"),
        ).lastrowid
        conn.commit()

    job_repo = JobRepository()

    assert job_repo.count_claimable() == 0
    assert job_repo.repair_missing_active_jobs() == 1
    assert job_repo.count_claimable() == 1

    with get_db_connection() as conn:
        active_job = conn.execute(
            """
            SELECT status, error
            FROM jobs
            WHERE episode_id = ?
              AND type = 'process_episode'
              AND status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
            """,
            (episode_id,),
        ).fetchone()

    assert active_job["status"] == "retry_scheduled"
    assert "Recreated missing" in active_job["error"]


def test_pending_status_creates_job_and_reset_cancels_without_deleting_episode(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        episode_id = conn.execute(
            """
            INSERT INTO episodes (subscription_id, guid, title, original_url, status)
            VALUES (?, ?, ?, ?, 'unprocessed')
            """,
            (subscription_id, "manual-episode", "Manual Episode", "https://example.com/manual.mp3"),
        ).lastrowid
        conn.commit()

    episode_repo = EpisodeRepository()
    episode_repo.update_status(episode_id, "pending")

    with get_db_connection() as conn:
        queued = conn.execute(
            "SELECT status FROM jobs WHERE episode_id = ? AND type = 'process_episode'",
            (episode_id,),
        ).fetchone()

    assert queued["status"] == "queued"

    episode_repo.reset_status(episode_id)

    with get_db_connection() as conn:
        episode = conn.execute("SELECT status FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        job = conn.execute(
            "SELECT status FROM jobs WHERE episode_id = ? AND type = 'process_episode'",
            (episode_id,),
        ).fetchone()

    assert episode["status"] == "unprocessed"
    assert job["status"] == "cancelled"
