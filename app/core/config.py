import os
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic import Field

DEFAULT_SESSION_SECRET_KEY = "super-secret-session-key-change-me"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Core
    ENVIRONMENT: str = Field("production", description="Environment: development or production")
    GEMINI_API_KEY: str | None = Field(None, description="Google Gemini API Key (comma-separated for multiple keys)")
    OPENAI_API_KEY: str | None = Field(None, description="OpenAI API Key")
    ANTHROPIC_API_KEY: str | None = Field(None, description="Anthropic API Key")
    OPENROUTER_API_KEY: str | None = Field(None, description="OpenRouter API Key")
    LOG_LEVEL: str = "INFO"
    SESSION_SECRET_KEY: str = Field(DEFAULT_SESSION_SECRET_KEY, description="Secret key for session encryption")
    
    # Paths
    DATA_DIR: str = "/data"

    
    # Web
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BASE_URL: str = "http://localhost:8000"
    COOKIE_SECURE: bool = False
    TRUST_PROXY_HEADERS: bool = False
    
    # Processing
    PROCESSOR_ENABLED: bool = True
    SPONSORBLOCK_ENABLED: bool = False
    CHECK_INTERVAL_MINUTES: int = 60
    WHISPER_MODEL: str = "base"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    MAX_FEED_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_DOWNLOAD_BYTES: int = 1500 * 1024 * 1024  # 1.5 GB
    MIN_FREE_SPACE_BYTES: int = 1024 * 1024 * 1024  # 1 GB
    FFMPEG_TIMEOUT_SECONDS: int = 7200  # 2 hours per FFmpeg operation
    ALLOW_PRIVATE_FEEDS: bool = True
    MAX_PROVIDER_CALLS_PER_JOB: int = Field(12, ge=1, le=100)
    PROVIDER_TIMEOUT_SECONDS: int = Field(120, ge=5, le=600)
    
    @property
    def DB_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, "db", "podcasts.db")
    
    @property
    def PODCASTS_DIR(self) -> str:
        """Base directory for all podcast data organized by podcast/episode"""
        return os.path.join(self.DATA_DIR, "podcasts")
        
    @property
    def FEEDS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "feeds")

    @property
    def ARTWORK_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "artwork")
        
    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "models")
    
    def get_episode_dir(self, podcast_slug: str, episode_slug: str) -> str:
        """Require two literal components; never allow aliases or root traversal."""
        from pathlib import Path
        for part in (podcast_slug, episode_slug):
            if not part or part in {'.', '..'} or any(c in part for c in '/\\\x00:') or part.endswith((' ', '.')):
                raise ValueError('Invalid episode storage component')
        root = Path(self.PODCASTS_DIR).resolve()
        target = root / podcast_slug / episode_slug
        if target.resolve() != target or not target.is_relative_to(root):
            raise ValueError('Episode storage must not traverse filesystem aliases')
        return str(target)

settings = Settings()


def is_default_session_secret() -> bool:
    return settings.SESSION_SECRET_KEY.strip() in {
        "", DEFAULT_SESSION_SECRET_KEY, "replace-with-a-long-random-secret",
        "change-me", "changeme",
    }

# Ensure directories exist
for path in [
    os.path.dirname(settings.DB_PATH),
    settings.PODCASTS_DIR,
    settings.FEEDS_DIR,
    settings.ARTWORK_DIR,
    settings.MODELS_DIR
]:
    os.makedirs(path, exist_ok=True)
