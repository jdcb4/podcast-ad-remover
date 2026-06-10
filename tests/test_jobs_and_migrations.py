from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository
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
    assert "schema_migrations" in tables
    assert "20260609_0001_jobs" in migrations
    assert "20260609_0002_feed_tokens" in migrations


def test_init_db_creates_resource_tuning_defaults(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        row = conn.execute("""
            SELECT whisper_cpu_threads, ffmpeg_threads, unload_whisper_after_job, openrouter_model
            FROM app_settings WHERE id = 1
        """).fetchone()

    assert row["whisper_cpu_threads"] == 0
    assert row["ffmpeg_threads"] == 0
    assert row["unload_whisper_after_job"] == 0
    assert "google/gemini-3.1-flash-lite" in row["openrouter_model"]
    assert "google/gemini-3-flash-preview" in row["openrouter_model"]


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
