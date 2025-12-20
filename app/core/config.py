import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Core
    GEMINI_API_KEY: str | None = Field(None, description="Google Gemini API Key")
    OPENAI_API_KEY: str | None = Field(None, description="OpenAI API Key")
    ANTHROPIC_API_KEY: str | None = Field(None, description="Anthropic API Key")
    OPENROUTER_API_KEY: str | None = Field(None, description="OpenRouter API Key")
    LOG_LEVEL: str = "INFO"
    
    # Paths
    DATA_DIR: str = "/data"

    
    # Web
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BASE_URL: str = "http://localhost:8000"
    
    # Processing
    CHECK_INTERVAL_MINUTES: int = 60
    WHISPER_MODEL: str = "base"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    
    @property
    def DB_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, "db", "podcasts.db")
        
    @property
    def DOWNLOADS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "downloads")
        
    @property
    def TRANSCRIPTS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "transcripts")
        
    @property
    def FEEDS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "feeds")
        
    @property
    def AUDIO_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "audio")

    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "models")

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
for path in [
    os.path.dirname(settings.DB_PATH),
    settings.DOWNLOADS_DIR,
    settings.TRANSCRIPTS_DIR,
    settings.FEEDS_DIR,
    settings.AUDIO_DIR,
    settings.MODELS_DIR
]:
    os.makedirs(path, exist_ok=True)
