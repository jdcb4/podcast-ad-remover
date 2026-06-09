from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository


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

    claimed = JobRepository().claim_due(1, worker_id="pytest-worker")

    assert created is True
    assert len(claimed) == 1
    assert claimed[0]["title"] == "Episode One"
    assert episode_repo.get_status(claimed[0]["id"]) == "processing"

    with get_db_connection() as conn:
        job = conn.execute("SELECT status, attempts, locked_by FROM jobs").fetchone()

    assert job["status"] == "running"
    assert job["attempts"] == 1
    assert job["locked_by"] == "pytest-worker"
