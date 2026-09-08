"""Conservative scratch estimates and stage boundary free-space checks."""
import shutil

from app.core.config import settings


def estimated_scratch(duration: float | None, source_bytes: int = 0) -> int:
    # Two 16kHz mono PCM copies plus source/output space, bounded below for short clips.
    return max(64 * 1024 * 1024, int(max(0, duration or 3600) * 128000) + source_bytes * 2)


def require_scratch(duration: float | None, source_bytes: int = 0):
    required = estimated_scratch(duration, source_bytes)
    if shutil.disk_usage(settings.DATA_DIR).free - required < settings.MIN_FREE_SPACE_BYTES:
        raise RuntimeError('Insufficient scratch space for processing; free disk space before retrying')
