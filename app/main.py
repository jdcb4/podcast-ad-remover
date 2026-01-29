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

log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = os.path.join(settings.DATA_DIR, "app.log")

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=settings.LOG_MAX_BYTES,
    backupCount=settings.LOG_BACKUP_COUNT
)
file_handler.setFormatter(log_formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

# Root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Capture uvicorn logs
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    l = logging.getLogger(logger_name)
    l.handlers = [file_handler, stream_handler]
    l.propagate = False

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
    
    # Auto-populate Public Application URL with detected IP if not set
    try:
        from app.infra.database import get_db_connection
        import socket
        
        with get_db_connection() as conn:
            row = conn.execute("SELECT app_external_url FROM app_settings WHERE id = 1").fetchone()
            current_url = row['app_external_url'] if row else None
            
            if not current_url:
                # Detect IP
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    lan_ip = s.getsockname()[0]
                    s.close()
                    
                    if lan_ip:
                        final_url = f"http://{lan_ip}:{settings.PORT}"
                        logger.info(f"Auto-configuring Public Application URL to: {final_url}")
                        conn.execute("UPDATE app_settings SET app_external_url = ? WHERE id = 1", (final_url,))
                        conn.commit()
                except Exception as e:
                    logger.warning(f"Could not auto-detect LAN IP: {e}")
    except Exception as e:
        logger.error(f"Error checking/updating app settings on startup: {e}")
    
    # Start background scheduler in a separate process
    from app.core.processor import start_processor_process
    import multiprocessing
    
    # Use spawn start method for consistency across platforms (especially Mac)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    p = multiprocessing.Process(target=start_processor_process, name="PodcastProcessor", daemon=True)
    p.start()
    app.state.processor_process = p
    logger.info(f"Background processor started in separate process (PID: {p.pid})")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if hasattr(app.state, "processor_process"):
        logger.info("Stopping background processor...")
        app.state.processor_process.terminate()
        app.state.processor_process.join(timeout=5)

from app.api import subscriptions
from app.web import router as web_router
from app.web.middleware import feed_auth_middleware
from app.web.auth import auth_middleware
from starlette.middleware.sessions import SessionMiddleware
import secrets

app = FastAPI(
    title="Podcast Ad Remover",
    lifespan=lifespan
)

# Add middleware (order matters - added in reverse of execution order)
# Execution order: SessionMiddleware -> auth_middleware -> feed_auth_middleware
app.middleware("http")(feed_auth_middleware)
app.middleware("http")(auth_middleware)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32))

app.include_router(subscriptions.router, prefix="/api")
app.include_router(web_router.router)

# Mount static files
app.mount("/feeds", StaticFiles(directory=settings.FEEDS_DIR), name="feeds")
app.mount("/audio", StaticFiles(directory=settings.PODCASTS_DIR), name="audio")
# Mount general static files (css, js, images)
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

@app.get("/")
async def root():
    return {"message": "Podcast Ad Remover is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
