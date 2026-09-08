"""Defaults, validation, and resolution for unified RSS feed preferences."""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from app.core.url_utils import validate_http_url


DEFAULT_UNIFIED_FEED_TITLE = "Unified Feed (Ad-Free)"
DEFAULT_UNIFIED_FEED_DESCRIPTION = "All your ad-free podcasts in one place."
DEFAULT_UNIFIED_FEED_INCLUDE_PODCAST_NAME = True
DEFAULT_UNIFIED_FEED_ARTWORK_PATH = "/static/unified_feed_cover.png"

MAX_UNIFIED_FEED_TITLE_LENGTH = 200
MAX_UNIFIED_FEED_DESCRIPTION_LENGTH = 4000
MAX_UNIFIED_FEED_ARTWORK_URL_LENGTH = 2048

# XML 1.0 allows tabs, line breaks, and Unicode outside these ranges.
_INVALID_XML_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


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

    for label, value in (
        ("Feed name", normalized_title),
        ("Feed description", normalized_description),
        ("Artwork URL", normalized_artwork_url),
    ):
        if _INVALID_XML_CHARACTERS.search(value):
            raise ValueError(f"{label} contains characters that cannot be used in RSS")

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
    title = _INVALID_XML_CHARACTERS.sub("", str(
        global_settings.get("unified_feed_title") or DEFAULT_UNIFIED_FEED_TITLE
    )).strip()
    description = _INVALID_XML_CHARACTERS.sub("", str(
        global_settings.get("unified_feed_description")
        or DEFAULT_UNIFIED_FEED_DESCRIPTION
    )).strip()
    custom_artwork_url = _INVALID_XML_CHARACTERS.sub("", str(
        global_settings.get("unified_feed_artwork_url") or ""
    )).strip()

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


def resolve_unified_feed_artwork_preview(
    custom_artwork_url: str | None, page_origin: str | None,
) -> str | None:
    """Preview HTTPS or same-origin HTTP artwork without relaxing the page CSP."""
    if not custom_artwork_url:
        return DEFAULT_UNIFIED_FEED_ARTWORK_PATH
    try:
        artwork = urlsplit(custom_artwork_url)
        page = urlsplit(page_origin or "")
        if artwork.scheme == "https":
            return custom_artwork_url
        if artwork.scheme == page.scheme == "http" and (
            artwork.hostname, artwork.port or 80
        ) == (page.hostname, page.port or 80):
            return custom_artwork_url
    except ValueError:
        pass
    return None
