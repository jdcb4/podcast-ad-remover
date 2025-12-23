import sqlite3
from contextlib import contextmanager
from app.core.config import settings

def init_db():
    """Initialize the database with the schema."""
    conn = sqlite3.connect(settings.DB_PATH)
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
        ai_model_cascade TEXT DEFAULT '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]',
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
        openrouter_model TEXT DEFAULT 'google/gemini-2.0-flash-001',
        app_external_url TEXT,
        
        enable_feed_auth INTEGER DEFAULT 0,
        feed_auth_username TEXT,
        feed_auth_password TEXT,
        
        auth_enabled INTEGER DEFAULT 0,
        require_password_change INTEGER DEFAULT 0,
        initial_password TEXT,
        ip_allowlist TEXT,
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Ensure default settings exist
    cursor.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
    
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
        "ALTER TABLE app_settings ADD COLUMN active_ai_provider TEXT DEFAULT 'gemini'",
        "ALTER TABLE app_settings ADD COLUMN openai_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN anthropic_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN openrouter_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN openai_model TEXT DEFAULT 'gpt-4o'",
        "ALTER TABLE app_settings ADD COLUMN anthropic_model TEXT DEFAULT 'claude-3-5-sonnet'",
        "ALTER TABLE app_settings ADD COLUMN openrouter_model TEXT DEFAULT 'google/gemini-2.0-flash-001'",
        "ALTER TABLE episodes ADD COLUMN processing_flags TEXT",
        "ALTER TABLE app_settings ADD COLUMN gemini_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN app_external_url TEXT",
        "ALTER TABLE app_settings ADD COLUMN enable_feed_auth INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_username TEXT",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_password TEXT",
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
        "ALTER TABLE episodes ADD COLUMN listen_count INTEGER DEFAULT 0"
    ]
    
    for sql in migrations:
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError:
            pass # Column likely exists

    conn.commit()
    conn.close()

@contextmanager
def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    # Set a busy timeout to avoid 'database is locked' errors during heavy processing
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()
