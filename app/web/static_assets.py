"""Content-derived versioning for static assets served through external caches."""

import hashlib
from functools import lru_cache
from pathlib import Path


STATIC_DIR = Path(__file__).with_name("static").resolve()


@lru_cache(maxsize=None)
def static_asset_version(relative_path: str) -> str:
    """Return a content-derived version for an asset contained by the static directory."""
    asset_path = (STATIC_DIR / relative_path).resolve()
    try:
        asset_path.relative_to(STATIC_DIR)
    except ValueError as exc:
        raise ValueError("Static asset path escapes the static directory") from exc
    return hashlib.sha256(asset_path.read_bytes()).hexdigest()[:12]


def configure_static_asset_versioning(templates) -> None:
    """Expose static asset versioning to one Jinja environment."""
    templates.env.globals["static_asset_version"] = static_asset_version
