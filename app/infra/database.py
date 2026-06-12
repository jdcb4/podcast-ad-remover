import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from app.core.config import settings


DEFAULT_GEMINI_MODEL_CASCADE = '["gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]'
DEFAULT_OPENROUTER_MODEL_CASCADE = '["google/gemini-3.5-flash", "google/gemini-3-flash", "google/gemini-3.1-flash-lite", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]'
DEFAULT_GEMINI_TTS_MODEL_CASCADE = '["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]'


FORMAL_MIGRATIONS = [
    (
        "20260609_0001_jobs",
        [
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                type TEXT NOT NULL DEFAULT 'process_episode',
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL DEFAULT 100,
                attempts INTEGER NOT NULL DEFAULT 0,
                locked_at TIMESTAMP,
                locked_by TEXT,
                next_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (episode_id) REFERENCES episodes (id)
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_per_episode_type
            ON jobs(episode_id, type)
            WHERE status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_claim
            ON jobs(status, next_run_at, priority, created_at)
            """,
            """
            INSERT INTO jobs (episode_id, type, status, priority, attempts, next_run_at, error)
            SELECT id,
                   'process_episode',
                   CASE
                       WHEN status = 'rate_limited' THEN 'rate_limited'
                       WHEN status = 'failed' THEN 'retry_scheduled'
                       ELSE 'queued'
                   END,
                   100,
                   COALESCE(retry_count, 0),
                   COALESCE(next_retry_at, CURRENT_TIMESTAMP),
                   error_message
            FROM episodes
            WHERE status IN ('pending', 'rate_limited')
               OR (status = 'failed' AND next_retry_at IS NOT NULL)
            ON CONFLICT DO NOTHING
            """,
        ],
    ),
    (
        "20260609_0002_feed_tokens",
        [
            """
            CREATE TABLE IF NOT EXISTS feed_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL DEFAULT 'Podcast app',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                revoked_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_feed_tokens_user_active
            ON feed_tokens(user_id, revoked_at)
            """,
        ],
    ),
    (
        "20260612_0003_user_podcast_library",
        [
            "ALTER TABLE subscriptions ADD COLUMN owner_user_id INTEGER",
            """
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                user_id INTEGER NOT NULL,
                subscription_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, subscription_id),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (subscription_id) REFERENCES subscriptions (id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_user_subscriptions_subscription
            ON user_subscriptions(subscription_id)
            """,
            """
            INSERT OR IGNORE INTO user_subscriptions (user_id, subscription_id)
            SELECT u.id, s.id
            FROM users u
            CROSS JOIN subscriptions s
            """,
        ],
    ),
    (
        "20260612_0004_access_request_password_hash",
        [
            "ALTER TABLE access_requests ADD COLUMN password_hash TEXT",
        ],
    ),
    (
        "20260612_0005_notifications",
        [
            "ALTER TABLE app_settings ADD COLUMN notifications_enabled INTEGER DEFAULT 0",
            "ALTER TABLE app_settings ADD COLUMN notification_urls TEXT",
            "ALTER TABLE app_settings ADD COLUMN notify_access_requests INTEGER DEFAULT 1",
            "ALTER TABLE app_settings ADD COLUMN notify_new_podcasts INTEGER DEFAULT 1",
            "ALTER TABLE app_settings ADD COLUMN notify_episode_downloads INTEGER DEFAULT 1",
            "ALTER TABLE app_settings ADD COLUMN notify_breaking_errors INTEGER DEFAULT 1",
        ],
    ),
    (
        "20260612_0006_tts_provider_settings",
        [
            "ALTER TABLE app_settings ADD COLUMN tts_provider TEXT DEFAULT 'piper'",
            "ALTER TABLE app_settings ADD COLUMN gemini_tts_voice TEXT DEFAULT 'Orus'",
            "ALTER TABLE app_settings ADD COLUMN gemini_tts_model_cascade TEXT DEFAULT '[\"gemini-3.1-flash-tts-preview\", \"gemini-2.5-flash-preview-tts\"]'",
        ],
    ),
]

SQLITE_BUSY_TIMEOUT_MS = 30000


def _connect_db() -> sqlite3.Connection:
    """Open SQLite with the same lock-tolerant settings for startup and runtime."""
    conn = sqlite3.connect(settings.DB_PATH, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    return conn


def _backup_database_if_needed(migration_ids: list[str]):
    """Create a timestamped DB backup before applying formal migrations."""
    if not migration_ids or not os.path.exists(settings.DB_PATH):
        return

    backup_dir = os.path.join(settings.DATA_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(backup_dir, f"podcasts-before-migration-{timestamp}.db")
    shutil.copy2(settings.DB_PATH, backup_path)


def _apply_formal_migrations(conn: sqlite3.Connection, create_backup: bool):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """)

    applied = {
        row[0]
        for row in cursor.execute("SELECT version FROM schema_migrations").fetchall()
    }
    pending = [(version, statements) for version, statements in FORMAL_MIGRATIONS if version not in applied]
    if create_backup:
        _backup_database_if_needed([version for version, _ in pending])

    for version, statements in pending:
        for sql in statements:
            cursor.execute(sql)
        cursor.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))


def init_db():
    """Initialize the database with the schema."""
    db_existed = os.path.exists(settings.DB_PATH)
    conn = _connect_db()
    cursor = conn.cursor()
    
    # Subscriptions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feed_url TEXT UNIQUE NOT NULL,
        title TEXT,
        description TEXT,
        slug TEXT UNIQUE,
        image_url TEXT,
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_checked_at TIMESTAMP,
        remove_ads BOOLEAN DEFAULT 1,
        remove_promos BOOLEAN DEFAULT 1,
        remove_intros BOOLEAN DEFAULT 0,
        remove_outros BOOLEAN DEFAULT 0,
        custom_instructions TEXT,
        append_summary BOOLEAN DEFAULT 0,
        append_title_intro BOOLEAN DEFAULT 0,
        ai_rewrite_description BOOLEAN DEFAULT 0,
        ai_audio_summary BOOLEAN DEFAULT 0
    )
    """)
    
    # Simple migration attempts (ignore if exists)
    try:
        cursor.execute("ALTER TABLE subscriptions ADD COLUMN ai_rewrite_description BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE subscriptions ADD COLUMN ai_audio_summary BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass


    # App Settings Singleton Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS app_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        whisper_model TEXT DEFAULT 'base',
        ai_model_cascade TEXT DEFAULT '["gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]',
        piper_model TEXT DEFAULT 'en_GB-cori-high.onnx',
        concurrent_downloads INTEGER DEFAULT 2,
        retention_days INTEGER DEFAULT 30,
        check_interval_minutes INTEGER DEFAULT 60,
        daily_download_limit INTEGER DEFAULT 0,
        
        ad_prompt_base TEXT,
        ad_target_sponsor TEXT,
        ad_target_promo TEXT,
        ad_target_intro TEXT,
        ad_target_outro TEXT,
        summary_prompt_template TEXT,
        
        active_ai_provider TEXT DEFAULT 'gemini',
        openai_api_key TEXT,
        anthropic_api_key TEXT,
        openrouter_api_key TEXT,
        openai_model TEXT DEFAULT 'gpt-4o',
        anthropic_model TEXT DEFAULT 'claude-3-5-sonnet',
        openrouter_model TEXT DEFAULT '["google/gemini-3.5-flash", "google/gemini-3-flash", "google/gemini-3.1-flash-lite", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]',
        app_external_url TEXT,
        
        enable_feed_auth INTEGER DEFAULT 0,
        feed_auth_username TEXT,
        feed_auth_password TEXT,
        public_subscribe_page_enabled INTEGER DEFAULT 1,
        whisper_cpu_threads INTEGER DEFAULT 0,
        ffmpeg_threads INTEGER DEFAULT 0,
        unload_whisper_after_job INTEGER DEFAULT 0,
        
        auth_enabled INTEGER DEFAULT 0,
        require_password_change INTEGER DEFAULT 0,
        initial_password TEXT,
        ip_allowlist TEXT,
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Ensure default settings exist
    cursor.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")

    try:
        cursor.execute("ALTER TABLE app_settings ADD COLUMN summary_prompt_template TEXT")
    except sqlite3.OperationalError:
        pass
    
    # Set default summary prompt template if not set
    cursor.execute("""
        UPDATE app_settings 
        SET summary_prompt_template = ?
        WHERE id = 1 AND (summary_prompt_template IS NULL OR summary_prompt_template = '')
    """, ("""You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
The summary must:
1. NOT mention the podcast name, episode title, or date.
2. Start immediately with "This episode includes".
3. Briefly summarize key topics.
Transcript Context: {transcript_context}""",))

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    """)
    
    # Access Requests Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS access_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT,
        reason TEXT,
        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending',
        ip_address TEXT,
        reviewed_by TEXT,
        reviewed_at TIMESTAMP
    )
    """)
    
    # Login Attempts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS login_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        ip_address TEXT,
        success INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_agent TEXT
    )
    """)

    # Episodes Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS episodes (
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
        transcript_path TEXT,
        ad_report_path TEXT,
        processing_step TEXT,
        progress INTEGER DEFAULT 0,
        description TEXT,
        ai_summary TEXT,
        report_path TEXT,
        file_size INTEGER,
        FOREIGN KEY (subscription_id) REFERENCES subscriptions (id),
        UNIQUE(subscription_id, guid)
    )
    """)
    
    # Migrations for existing databases
    migrations = [
        "ALTER TABLE episodes ADD COLUMN transcript_path TEXT",
        "ALTER TABLE episodes ADD COLUMN ad_report_path TEXT",
        "ALTER TABLE episodes ADD COLUMN processing_step TEXT",
        "ALTER TABLE episodes ADD COLUMN progress INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN description TEXT",
        "ALTER TABLE episodes ADD COLUMN report_path TEXT",
        "ALTER TABLE subscriptions ADD COLUMN image_url TEXT",
        "ALTER TABLE episodes ADD COLUMN file_size INTEGER",
        "ALTER TABLE episodes ADD COLUMN retry_count INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN next_retry_at TIMESTAMP",
        "ALTER TABLE subscriptions ADD COLUMN remove_ads BOOLEAN DEFAULT 1",
        "ALTER TABLE subscriptions ADD COLUMN remove_promos BOOLEAN DEFAULT 1",
        "ALTER TABLE subscriptions ADD COLUMN remove_intros BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN remove_outros BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN custom_instructions TEXT",
        "ALTER TABLE subscriptions ADD COLUMN append_summary BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN append_title_intro BOOLEAN DEFAULT 0",
        
        # New prompt migrations
        "ALTER TABLE app_settings ADD COLUMN ad_prompt_base TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_sponsor TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_promo TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_intro TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_outro TEXT",
        "ALTER TABLE app_settings ADD COLUMN summary_prompt_template TEXT",
        
        # Multi-Provider AI migrations
        "ALTER TABLE app_settings ADD COLUMN ai_model_cascade TEXT DEFAULT '[\"gemini-3.5-flash\", \"gemini-3-flash\", \"gemini-3.1-flash-lite\", \"gemini-2.5-flash\", \"gemini-2.5-flash-lite\"]'",
        "ALTER TABLE app_settings ADD COLUMN active_ai_provider TEXT DEFAULT 'gemini'",
        "ALTER TABLE app_settings ADD COLUMN openai_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN anthropic_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN openrouter_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN openai_model TEXT DEFAULT 'gpt-4o'",
        "ALTER TABLE app_settings ADD COLUMN anthropic_model TEXT DEFAULT 'claude-3-5-sonnet'",
        "ALTER TABLE app_settings ADD COLUMN openrouter_model TEXT DEFAULT '[\"google/gemini-3.5-flash\", \"google/gemini-3-flash\", \"google/gemini-3.1-flash-lite\", \"google/gemini-2.5-flash\", \"google/gemini-2.5-flash-lite\"]'",
        "ALTER TABLE episodes ADD COLUMN processing_flags TEXT",
        "ALTER TABLE app_settings ADD COLUMN gemini_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN app_external_url TEXT",
        "ALTER TABLE app_settings ADD COLUMN enable_feed_auth INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_username TEXT",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_password TEXT",
        "ALTER TABLE app_settings ADD COLUMN public_subscribe_page_enabled INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN whisper_cpu_threads INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN ffmpeg_threads INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN unload_whisper_after_job INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN auth_enabled INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN require_password_change INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN initial_password TEXT",
        "ALTER TABLE app_settings ADD COLUMN ip_allowlist TEXT",
        "ALTER TABLE subscriptions ADD COLUMN description TEXT",
        "ALTER TABLE episodes ADD COLUMN ai_summary TEXT",
        "ALTER TABLE episodes ADD COLUMN is_manual_download BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN retention_days INTEGER DEFAULT 30",
        "ALTER TABLE subscriptions ADD COLUMN manual_retention_days INTEGER DEFAULT 14",
        "ALTER TABLE subscriptions ADD COLUMN retention_limit INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN check_interval_minutes INTEGER DEFAULT 60",
        
        # Global Subscription Defaults
        "ALTER TABLE app_settings ADD COLUMN default_remove_ads INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN default_remove_promos INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN default_remove_intros INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_remove_outros INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_ai_rewrite_description INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_ai_audio_summary INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_append_title_intro INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_retention_limit INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN default_retention_days INTEGER DEFAULT 30",
        "ALTER TABLE app_settings ADD COLUMN default_manual_retention_days INTEGER DEFAULT 14",
        "ALTER TABLE app_settings ADD COLUMN default_custom_instructions TEXT",
        "ALTER TABLE episodes ADD COLUMN listen_count INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN gemini_api_keys TEXT",

        # Whitelist mode: inverts filtering to keep only Content segments
        "ALTER TABLE app_settings ADD COLUMN whitelist_mode INTEGER DEFAULT 0"
    ]
    
    for sql in migrations:
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError:
            pass # Column likely exists

    cursor.execute("""
        UPDATE app_settings
        SET ai_model_cascade = ?
        WHERE id = 1
          AND (
              ai_model_cascade IS NULL
              OR ai_model_cascade = ''
              OR ai_model_cascade = '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]'
              OR ai_model_cascade = '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]'
          )
    """, (DEFAULT_GEMINI_MODEL_CASCADE,))

    cursor.execute("""
        UPDATE app_settings
        SET openrouter_model = ?
        WHERE id = 1
          AND (
              openrouter_model IS NULL
              OR openrouter_model = ''
              OR openrouter_model = 'google/gemini-2.0-flash-001'
              OR openrouter_model = '["google/gemini-3.1-flash-lite", "google/gemini-3-flash-preview", "google/gemini-2.5-flash-lite"]'
          )
    """, (DEFAULT_OPENROUTER_MODEL_CASCADE,))

    _apply_formal_migrations(conn, create_backup=db_existed)

    cursor.execute("""
        UPDATE app_settings
        SET gemini_tts_model_cascade = ?
        WHERE id = 1
          AND (
              gemini_tts_model_cascade IS NULL
              OR gemini_tts_model_cascade = ''
          )
    """, (DEFAULT_GEMINI_TTS_MODEL_CASCADE,))

    conn.commit()
    conn.close()

@contextmanager
def get_db_connection():
    """Get a database connection."""
    conn = _connect_db()
    try:
        yield conn
    finally:
        conn.close()
