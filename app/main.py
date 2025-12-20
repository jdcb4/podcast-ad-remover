from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging

from app.core.config import settings
from app.infra.database import init_db
from app.core.processor import Processor

# Configure logging
from logging.handlers import RotatingFileHandler
import os

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            os.path.join(settings.DATA_DIR, "app.log"),
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Podcast Ad Remover...")
    init_db()
    logger.info(f"Database initialized at {settings.DB_PATH}")
    import os
    if os.path.exists(settings.DB_PATH):
        size = os.path.getsize(settings.DB_PATH)
        logger.info(f"Database size: {size} bytes")
    else:
        logger.warning("Database file not found!")
    
    # Start background scheduler
    processor = Processor()
    asyncio.create_task(processor.run_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

from app.api import subscriptions
from app.web import router as web_router

app = FastAPI(
    title="Podcast Ad Remover",
    lifespan=lifespan
)

app.include_router(subscriptions.router, prefix="/api")
app.include_router(web_router.router)

# Mount static files
app.mount("/feeds", StaticFiles(directory=settings.FEEDS_DIR), name="feeds")
app.mount("/audio", StaticFiles(directory=settings.AUDIO_DIR), name="audio")
# Mount general static files (css, js, images)
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

@app.get("/")
async def root():
    return {"message": "Podcast Ad Remover is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
