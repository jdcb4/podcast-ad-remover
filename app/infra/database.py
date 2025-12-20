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
        append_title_intro BOOLEAN DEFAULT 0
    )
    """)



    # App Settings Singleton Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS app_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        whisper_model TEXT DEFAULT 'base',
        ai_model_cascade TEXT DEFAULT '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]',
        piper_model TEXT DEFAULT 'en_GB-cori-high.onnx',
        concurrent_downloads INTEGER DEFAULT 3,
        retention_days INTEGER DEFAULT 30,
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
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Ensure default settings exist
    cursor.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")

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
        "ALTER TABLE app_settings ADD COLUMN gemini_api_key TEXT"
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
    try:
        yield conn
    finally:
        conn.close()
