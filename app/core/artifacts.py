"""Stable episode storage with bounded, read-compatible legacy lookup."""
import hashlib
from pathlib import Path

from app.core.config import settings


def episode_directory(subscription_slug: str, episode_id: int) -> Path:
    if int(episode_id) <= 0:
        raise ValueError('Episode storage requires a persisted positive ID')
    return Path(settings.get_episode_dir(subscription_slug, f'episode-{episode_id}'))


def legacy_directory(subscription_slug: str, guid: str) -> Path | None:
    slug = str(guid).replace('/', '_').replace(' ', '_')
    try:
        return Path(settings.get_episode_dir(subscription_slug, slug))
    except (ValueError, OSError):
        return None


def artifact_path(row: dict, column: str, filename: str) -> str | None:
    root = Path(settings.DATA_DIR).resolve()
    recorded = row.get(column)
    if recorded:
        path = Path(recorded).resolve()
        if path.is_relative_to(root) and path.is_file():
            return str(path)
    slug = row.get('subscription_slug') or row.get('slug')
    legacy = legacy_directory(slug, row['guid']) if slug else None
    if legacy:
        candidate = legacy / filename
        if candidate.is_file() and candidate.resolve().is_relative_to(root):
            return str(candidate)
    return None


def source_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
