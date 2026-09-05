"""Serialize feed snapshots across processes and replace complete XML atomically."""
import os
import tempfile
from functools import wraps
from pathlib import Path

from filelock import FileLock

from app.core.config import settings


def serialized_feed(function):
    @wraps(function)
    def generate(*args, **kwargs):
        with FileLock(str(Path(settings.FEEDS_DIR) / '.publication.lock'), timeout=30):
            return function(*args, **kwargs)
    return generate


def atomic_write(path: str, content: str):
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.feed-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
