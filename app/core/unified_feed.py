"""Defaults, validation, and resolution for unified RSS feed preferences."""

from __future__ import annotations

from typing import Any, Mapping

from app.core.url_utils import validate_http_url


DEFAULT_UNIFIED_FEED_TITLE = "Unified Feed (Ad-Free)"
DEFAULT_UNIFIED_FEED_DESCRIPTION = "All your ad-free podcasts in one place."
DEFAULT_UNIFIED_FEED_INCLUDE_PODCAST_NAME = True
DEFAULT_UNIFIED_FEED_ARTWORK_PATH = "/static/unified_feed_cover.png"

MAX_UNIFIED_FEED_TITLE_LENGTH = 200
MAX_UNIFIED_FEED_DESCRIPTION_LENGTH = 4000
MAX_UNIFIED_FEED_ARTWORK_URL_LENGTH = 2048


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_unified_feed_settings(
    title: str,
    description: str,
    include_podcast_name: bool,
    artwork_url: str | None,
) -> dict[str, Any]:
    """Validate form values and return normalized database values."""
    normalized_title = (title or "").strip()
    normalized_description = (description or "").strip()
    normalized_artwork_url = (artwork_url or "").strip()

    if not normalized_title:
        raise ValueError("Feed name is required")
    if len(normalized_title) > MAX_UNIFIED_FEED_TITLE_LENGTH:
        raise ValueError(
            f"Feed name must be {MAX_UNIFIED_FEED_TITLE_LENGTH} characters or fewer"
        )
    if not normalized_description:
        raise ValueError("Feed description is required")
    if len(normalized_description) > MAX_UNIFIED_FEED_DESCRIPTION_LENGTH:
        raise ValueError(
            "Feed description must be "
            f"{MAX_UNIFIED_FEED_DESCRIPTION_LENGTH} characters or fewer"
        )
    if len(normalized_artwork_url) > MAX_UNIFIED_FEED_ARTWORK_URL_LENGTH:
        raise ValueError(
            "Artwork URL must be "
            f"{MAX_UNIFIED_FEED_ARTWORK_URL_LENGTH} characters or fewer"
        )
    if normalized_artwork_url:
        validate_http_url(normalized_artwork_url, allow_private=True)

    return {
        "title": normalized_title,
        "description": normalized_description,
        "include_podcast_name": bool(include_podcast_name),
        "custom_artwork_url": normalized_artwork_url or None,
    }


def resolve_unified_feed_settings(
    global_settings: Mapping[str, Any],
    base_url: str,
) -> dict[str, Any]:
    """Resolve stored settings with backward-compatible defaults."""
    title = str(
        global_settings.get("unified_feed_title") or DEFAULT_UNIFIED_FEED_TITLE
    ).strip()
    description = str(
        global_settings.get("unified_feed_description")
        or DEFAULT_UNIFIED_FEED_DESCRIPTION
    ).strip()
    custom_artwork_url = str(
        global_settings.get("unified_feed_artwork_url") or ""
    ).strip()

    return {
        "title": title or DEFAULT_UNIFIED_FEED_TITLE,
        "description": description or DEFAULT_UNIFIED_FEED_DESCRIPTION,
        "include_podcast_name": _as_bool(
            global_settings.get("unified_feed_include_podcast_name"),
            DEFAULT_UNIFIED_FEED_INCLUDE_PODCAST_NAME,
        ),
        "custom_artwork_url": custom_artwork_url or None,
        "artwork_url": custom_artwork_url
        or f"{base_url.rstrip('/')}{DEFAULT_UNIFIED_FEED_ARTWORK_PATH}",
    }
