"""Strict, public-only YouTube discovery and audio download support."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

from app.core.config import settings
from app.core.feed import slugify


logger = logging.getLogger(__name__)

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
CHANNEL_SCAN_LIMIT = 50
PLAYLIST_SCAN_LIMIT = 500
YOUTUBE_INITIAL_LIMIT = 5


class YouTubeUrlError(ValueError):
    pass


class YouTubeDownloadCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class YouTubeReference:
    source_type: str
    url: str


@dataclass(frozen=True)
class YouTubeResolvedSource:
    source_type: str
    external_id: str
    canonical_url: str
    title: str
    slug: str
    image_url: str | None
    description: str


@dataclass(frozen=True)
class YouTubeDiscovery:
    entries: list[dict]
    truncated: bool


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def looks_like_youtube_url(value: str) -> bool:
    try:
        return _hostname(value.strip()) in YOUTUBE_HOSTS | {"youtu.be"}
    except Exception:
        return False


def video_id_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    if (parsed.hostname or "").lower().rstrip(".") not in YOUTUBE_HOSTS:
        return None
    ids = parse_qs(parsed.query).get("v", [])
    return ids[0] if len(ids) == 1 and ids[0] else None


def classify_youtube_url(value: str) -> YouTubeReference | None:
    """Classify only the URL shapes supported as subscription sources."""
    value = value.strip()
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in YOUTUBE_HOSTS:
        if host == "youtu.be":
            raise YouTubeUrlError("Individual YouTube video URLs are not supported")
        return None
    if parsed.scheme not in {"http", "https"}:
        raise YouTubeUrlError("YouTube URLs must use http or https")

    path = parsed.path.rstrip("/") or "/"
    query = parse_qs(parsed.query)
    if path == "/playlist":
        playlist_ids = query.get("list", [])
        if len(playlist_ids) != 1 or not playlist_ids[0].strip():
            raise YouTubeUrlError("An explicit YouTube playlist URL with a list ID is required")
        return YouTubeReference("youtube_playlist", value)

    if path in {"/watch", "/results", "/feed", "/shorts"} or path.startswith("/shorts/"):
        if path == "/watch" and "list" in query:
            raise YouTubeUrlError("Use the explicit YouTube playlist URL, not a watch URL containing a playlist")
        raise YouTubeUrlError("Only YouTube channel and explicit playlist URLs are supported")

    parts = [part for part in path.split("/") if part]
    if not parts:
        raise YouTubeUrlError("A direct YouTube channel or playlist URL is required")
    if parts[-1].lower() in {"shorts", "streams", "live"}:
        raise YouTubeUrlError("Channel Shorts and streams tabs are not supported")

    first = parts[0]
    if first.startswith("@"):
        valid_channel = len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "videos")
    elif first in {"channel", "c", "user"}:
        valid_channel = len(parts) == 2 or (len(parts) == 3 and parts[2].lower() == "videos")
    else:
        valid_channel = False
    if not valid_channel:
        raise YouTubeUrlError("Only YouTube channel and explicit playlist URLs are supported")
    return YouTubeReference("youtube_channel", value)


def _youtube_dl(options: dict):
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("YouTube support requires the pinned yt-dlp dependency") from exc
    return yt_dlp.YoutubeDL(options)


def _base_options() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "ignoreconfig": True,
        "cookiefile": None,
        "remote_components": set(),
        "js_runtimes": {"deno": {}},
    }


def _source_thumbnail(info: dict) -> str | None:
    if info.get("thumbnail"):
        return str(info["thumbnail"])
    thumbnails = info.get("thumbnails") or []
    return str(thumbnails[-1]["url"]) if thumbnails and thumbnails[-1].get("url") else None


def resolve_youtube_source(value: str) -> YouTubeResolvedSource:
    reference = classify_youtube_url(value)
    if reference is None:
        raise YouTubeUrlError("Not a YouTube URL")

    extract_url = reference.url
    if reference.source_type == "youtube_channel" and not urlparse(extract_url).path.rstrip("/").endswith("/videos"):
        extract_url = extract_url.rstrip("/") + "/videos"

    options = _base_options() | {
        "extract_flat": "in_playlist",
        "playlistend": 1,
        "skip_download": True,
    }
    with _youtube_dl(options) as ydl:
        info = ydl.extract_info(extract_url, download=False)
    if not info:
        raise YouTubeUrlError("YouTube did not return source metadata")

    if reference.source_type == "youtube_playlist":
        external_id = str(info.get("id") or parse_qs(urlparse(value).query)["list"][0])
        canonical_url = f"https://www.youtube.com/playlist?list={external_id}"
    else:
        external_id = str(info.get("channel_id") or info.get("uploader_id") or info.get("id") or "").strip()
        if not external_id:
            raise YouTubeUrlError("Could not resolve the YouTube channel ID")
        canonical_url = f"https://www.youtube.com/channel/{external_id}/videos"

    title = str(info.get("channel") or info.get("uploader") or info.get("title") or "YouTube source").strip()
    if reference.source_type == "youtube_playlist":
        title = str(info.get("title") or title).strip()
    return YouTubeResolvedSource(
        source_type=reference.source_type,
        external_id=external_id,
        canonical_url=canonical_url,
        title=title,
        slug=slugify(title) or f"youtube-{external_id[-8:].lower()}",
        image_url=_source_thumbnail(info),
        description=str(info.get("description") or ""),
    )


def discover_youtube_entries(source_type: str, canonical_url: str) -> YouTubeDiscovery:
    if source_type not in {"youtube_channel", "youtube_playlist"}:
        raise ValueError(f"Unsupported YouTube source type: {source_type}")
    limit = CHANNEL_SCAN_LIMIT if source_type == "youtube_channel" else PLAYLIST_SCAN_LIMIT
    options = _base_options() | {
        "extract_flat": "in_playlist",
        "playlistend": limit + 1,
        "skip_download": True,
    }
    with _youtube_dl(options) as ydl:
        info = ydl.extract_info(canonical_url, download=False)
    entries = [dict(entry) for entry in (info or {}).get("entries") or [] if entry and entry.get("id")]
    return YouTubeDiscovery(entries=entries[:limit], truncated=len(entries) > limit)


def _parse_upload_date(info: dict) -> datetime | None:
    timestamp = info.get("timestamp") or info.get("release_timestamp")
    if timestamp is not None:
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError, OSError):
            pass
    upload_date = str(info.get("upload_date") or "")
    if len(upload_date) == 8 and upload_date.isdigit():
        return datetime.strptime(upload_date, "%Y%m%d")
    return None


def hydrate_youtube_entry(video_id: str) -> tuple[dict | None, str | None, bool]:
    """Return episode metadata, exclusion reason, and whether exclusion is transient."""
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    options = _base_options() | {"noplaylist": True, "skip_download": True}
    try:
        with _youtube_dl(options) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
    except Exception as exc:
        message = str(exc).lower()
        transient = any(term in message for term in ("upcoming", "premiere", "live", "temporarily"))
        return None, "temporarily_unavailable" if transient else "unavailable", transient
    if not info:
        return None, "unavailable", False

    availability = str(info.get("availability") or "public").lower()
    if availability not in {"public", "unlisted"}:
        return None, availability or "unavailable", False

    live_status = str(info.get("live_status") or "not_live").lower()
    if live_status in {"is_upcoming", "is_live", "post_live"}:
        return None, live_status, live_status != "post_live"
    if info.get("was_live") or live_status == "was_live":
        return None, "completed_livestream", False

    webpage_url = str(info.get("webpage_url") or canonical_url)
    duration_value = float(info.get("duration") or 0)
    aspect_ratio = float(info.get("aspect_ratio") or 0)
    if (
        "/shorts/" in webpage_url
        or str(info.get("original_url") or "").find("/shorts/") >= 0
        or (0 < duration_value <= 180 and 0 < aspect_ratio <= 1.0)
    ):
        return None, "short", False

    duration = int(duration_value)
    return {
        "guid": video_id,
        "title": str(info.get("title") or "Untitled YouTube video"),
        "pub_date": _parse_upload_date(info),
        "original_url": canonical_url,
        "duration": duration,
        "description": str(info.get("description") or ""),
        "file_size": int(info.get("filesize_approx") or info.get("filesize") or 0),
    }, None, False


def download_youtube_audio(
    video_url: str,
    episode_dir: str,
    *,
    progress_callback: Callable[[int], None] | None = None,
    cancellation_callback: Callable[[], bool] | None = None,
) -> str:
    """Download best audio-only into the episode directory and return its stable path."""
    free_space = shutil.disk_usage(settings.DATA_DIR).free
    if free_space < settings.MIN_FREE_SPACE_BYTES:
        raise RuntimeError("Not enough free disk space to download episode")

    directory = Path(episode_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("youtube-download.*"):
        if stale.is_file():
            stale.unlink()

    def hook(status: dict) -> None:
        if cancellation_callback and cancellation_callback():
            raise YouTubeDownloadCancelled("YouTube download cancelled")
        downloaded = int(status.get("downloaded_bytes") or 0)
        if downloaded > settings.MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Episode download exceeds configured maximum size")
        total = int(status.get("total_bytes") or status.get("total_bytes_estimate") or 0)
        if progress_callback and total > 0:
            progress_callback(min(100, int(downloaded * 100 / total)))

    output_template = str(directory / "youtube-download.%(ext)s")
    options = _base_options() | {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": output_template,
        "max_filesize": settings.MAX_DOWNLOAD_BYTES,
        "overwrites": True,
        "continuedl": False,
        "progress_hooks": [hook],
    }
    try:
        with _youtube_dl(options) as ydl:
            info = ydl.extract_info(video_url, download=True)
        requested = (info or {}).get("requested_downloads") or []
        candidate = Path(requested[0].get("filepath")) if requested and requested[0].get("filepath") else None
        if not candidate or not candidate.exists():
            matches = [path for path in directory.glob("youtube-download.*") if not path.name.endswith(".part")]
            candidate = matches[0] if len(matches) == 1 else None
        if not candidate or not candidate.exists():
            raise RuntimeError("yt-dlp completed without producing an audio file")
        if candidate.stat().st_size > settings.MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Episode download exceeds configured maximum size")
        extension = candidate.suffix.lower() or ".media"
        final_path = directory / f"original{extension}"
        os.replace(candidate, final_path)
        return str(final_path)
    except Exception:
        for stale in directory.glob("youtube-download.*"):
            if stale.is_file():
                stale.unlink()
        raise
