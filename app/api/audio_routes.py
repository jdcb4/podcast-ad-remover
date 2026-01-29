"""
Dynamic audio serving routes with listen tracking.
Replaces static file serving to track downloads from podcast apps.
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, Response
from pathlib import Path
import os
import time
import hashlib
import logging

from app.core.config import settings
from app.infra.repository import EpisodeRepository, SubscriptionRepository

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory cache for deduplication (IP+episode -> last_access_time)
# Key: hash(IP + episode_id), Value: timestamp
_listen_cache: dict[str, float] = {}
DEDUPE_WINDOW_SECONDS = 2 * 60 * 60  # 2 hours


def _get_cache_key(ip: str, episode_id: int) -> str:
    """Generate a cache key for deduplication."""
    return hashlib.md5(f"{ip}:{episode_id}".encode()).hexdigest()


def _should_count_listen(ip: str, episode_id: int) -> bool:
    """Check if this request should count as a new listen (deduplication)."""
    global _listen_cache
    
    cache_key = _get_cache_key(ip, episode_id)
    now = time.time()
    
    # Clean old entries (prevent memory leak)
    expired = [k for k, v in _listen_cache.items() if now - v > DEDUPE_WINDOW_SECONDS]
    for k in expired:
        del _listen_cache[k]
    
    # Check if already counted recently
    if cache_key in _listen_cache:
        last_access = _listen_cache[cache_key]
        if now - last_access < DEDUPE_WINDOW_SECONDS:
            return False
    
    _listen_cache[cache_key] = now
    return True


def _is_first_byte_request(request: Request) -> bool:
    """Check if this is the first request for the file (not a mid-stream Range request)."""
    range_header = request.headers.get("Range", "")
    if not range_header:
        return True  # No Range header = full file request
    
    # Parse Range: bytes=START-END
    if range_header.startswith("bytes="):
        range_spec = range_header[6:]
        if "-" in range_spec:
            start = range_spec.split("-")[0]
            # Count as first request if starting at 0 or very beginning
            if start == "" or start == "0" or int(start) < 1024:
                return True
    return False


@router.get("/audio/{path:path}")
async def serve_audio(path: str, request: Request):
    """
    Serve audio files dynamically with listen tracking.
    
    Path format: {subscription_slug}/{episode_guid}/{filename}
    or: {subscription_slug}/{filename}
    """
    # Build full file path
    file_path = Path(settings.PODCASTS_DIR) / path
    
    if not file_path.exists():
        logger.warning(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not a file")
    
    # Security: ensure file is within PODCASTS_DIR
    try:
        file_path.resolve().relative_to(Path(settings.PODCASTS_DIR).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Track listen if this is a first-byte request
    if _is_first_byte_request(request):
        try:
            # Try to find episode by path
            ep_repo = EpisodeRepository()
            sub_repo = SubscriptionRepository()
            
            # Parse path to extract subscription slug and filename
            path_parts = path.split("/")
            if len(path_parts) >= 1:
                subscription_slug = path_parts[0]
                filename = path_parts[-1]
                
                # Find subscription
                sub = sub_repo.get_by_slug(subscription_slug)
                if sub:
                    # Find episode by filename
                    episode = ep_repo.get_by_subscription_and_filename(sub.id, filename)
                    if episode:
                        # Get client IP
                        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
                        if "," in client_ip:
                            client_ip = client_ip.split(",")[0].strip()
                        
                        # Deduplicated listen count
                        if _should_count_listen(client_ip, episode.id):
                            ep_repo.increment_listen_count(episode.id)
                            logger.info(f"Tracked listen: episode={episode.id} ({filename}), IP={client_ip}")
                        else:
                            logger.debug(f"Deduplicated listen: episode={episode.id}, IP={client_ip}")
        except Exception as e:
            logger.error(f"Error tracking listen: {e}")
            # Don't fail the request if tracking fails
    
    # Determine media type
    media_type = "audio/mpeg"
    if file_path.suffix.lower() == ".m4a":
        media_type = "audio/mp4"
    elif file_path.suffix.lower() == ".ogg":
        media_type = "audio/ogg"
    elif file_path.suffix.lower() == ".wav":
        media_type = "audio/wav"
    
    # Use FileResponse which handles Range requests automatically
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )
