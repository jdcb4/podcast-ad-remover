import sqlite3

from scripts.migration_dry_run import run_migration_dry_run


def create_legacy_database(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
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
            INSERT INTO episodes (subscription_id, guid, title, original_url, status)
            VALUES (1, 'pending-guid', 'Pending Episode', 'https://example.com/pending.mp3', 'pending');
            """
        )


def table_names(db_path):
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }


def test_migration_dry_run_migrates_copy_without_touching_source(tmp_path):
    source_db = tmp_path / "source" / "podcasts.db"
    dry_run_data = tmp_path / "dry-run-data"
    create_legacy_database(source_db)

    result = run_migration_dry_run(source_db, dry_run_data)

    assert result.source_db == source_db.resolve()
    assert result.copied_db == (dry_run_data / "db" / "podcasts.db").resolve()
    assert "schema_migrations" in result.table_names
    assert "jobs" in result.table_names
    assert "feed_tokens" in result.table_names
    assert "20260609_0001_jobs" in result.schema_versions
    assert "schema_migrations" not in table_names(source_db)

    with sqlite3.connect(result.copied_db) as migrated:
        migrated.row_factory = sqlite3.Row
        subscription = migrated.execute("SELECT title, slug FROM subscriptions WHERE id = 1").fetchone()
        job_count = migrated.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    assert dict(subscription) == {"title": "Legacy Show", "slug": "legacy-show"}
    assert job_count == 1
