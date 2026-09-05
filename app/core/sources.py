"""Subscription source adapters shared by web, API, and processing flows."""

from __future__ import annotations

from app.core.http_downloads import async_stream_get

import asyncio
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import aiofiles
import httpx

from app.core.config import settings
from app.core.feed import FeedManager
from app.core.youtube import (
    YouTubeResolvedSource,
    classify_youtube_url,
    discover_youtube_entries,
    download_youtube_audio,
    looks_like_youtube_url,
    resolve_youtube_source,
)
from app.core.url_utils import is_audio_content_type, validate_http_url, validate_redirect_target


@dataclass(frozen=True)
class ResolvedSource:
    source_type: str
    external_id: str | None
    canonical_url: str
    title: str
    slug: str
    image_url: str | None
    description: str

    def search_result(self) -> dict:
        return {
            "title": self.title,
            "feed_url": self.canonical_url,
            "image": self.image_url,
            "description": self.description,
            "source_type": self.source_type,
            "source_external_id": self.external_id,
        }


@dataclass(frozen=True)
class SourceDiscovery:
    entries: list[dict]
    truncated: bool = False


class SourceAdapter(Protocol):
    """Provider boundary for metadata, discovery, and source-media retrieval."""

    source_types: frozenset[str]

    def resolve(self, url: str) -> ResolvedSource: ...

    async def discover(self, source_type: str, canonical_url: str) -> SourceDiscovery: ...

    async def download(
        self,
        media_url: str,
        episode_dir: str,
        *,
        progress_callback: Callable[[int], None] | None = None,
        cancellation_callback: Callable[[], bool] | None = None,
    ) -> str: ...


def validate_rss_download_response(
    original_url: str,
    final_url: str,
    headers,
    free_space: int,
) -> int:
    validate_redirect_target(original_url, final_url, allow_private=settings.ALLOW_PRIVATE_FEEDS)
    try:
        total = int(headers.get("Content-Length", 0) or 0)
    except (TypeError, ValueError):
        total = 0
    if total and total > settings.MAX_DOWNLOAD_BYTES:
        raise RuntimeError("Episode download exceeds configured maximum size")
    if total and free_space - total < settings.MIN_FREE_SPACE_BYTES:
        raise RuntimeError(
            "Episode download would leave less than the configured minimum free disk space"
        )
    if not is_audio_content_type(headers.get("Content-Type")):
        raise RuntimeError(f"Episode URL did not return audio content: {headers.get('Content-Type')}")
    return total


class RssSourceAdapter:
    source_types = frozenset({"rss"})

    def resolve(self, url: str) -> ResolvedSource:
        title, slug, image_url, description = FeedManager.parse_feed(url)
        return ResolvedSource(
            source_type="rss",
            external_id=None,
            canonical_url=url,
            title=title or "Untitled podcast",
            slug=slug or "podcast",
            image_url=image_url,
            description=description or "",
        )

    async def discover(self, source_type: str, canonical_url: str) -> SourceDiscovery:
        if source_type != "rss":
            raise ValueError(f"RSS adapter cannot discover source type {source_type}")
        entries = await asyncio.to_thread(FeedManager.parse_episodes, canonical_url)
        return SourceDiscovery(entries=entries)

    async def download(
        self,
        media_url: str,
        episode_dir: str,
        *,
        progress_callback: Callable[[int], None] | None = None,
        cancellation_callback: Callable[[], bool] | None = None,
    ) -> str:
        validate_http_url(media_url, allow_private=settings.ALLOW_PRIVATE_FEEDS)
        free_space = shutil.disk_usage(settings.DATA_DIR).free
        if free_space < settings.MIN_FREE_SPACE_BYTES:
            raise RuntimeError("Not enough free disk space to download episode")

        directory = Path(episode_dir)
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / "original.mp3"
        partial_path = directory / "original.mp3.part"
        partial_path.unlink(missing_ok=True)
        last_cancel_check = datetime.now()
        try:
            async with httpx.AsyncClient(trust_env=settings.ALLOW_PRIVATE_FEEDS) as client:
                async with async_stream_get(client, media_url, timeout=300.0) as response:
                    response.raise_for_status()
                    total = validate_rss_download_response(
                        media_url, str(response.url), response.headers, free_space
                    )
                    downloaded = 0
                    last_percent = -1
                    last_space_check = 0
                    async with aiofiles.open(partial_path, "wb") as handle:
                        async for chunk in response.aiter_bytes():
                            await handle.write(chunk)
                            downloaded += len(chunk)
                            if downloaded - last_space_check >= 8 * 1024 * 1024:
                                if shutil.disk_usage(settings.DATA_DIR).free < settings.MIN_FREE_SPACE_BYTES:
                                    raise RuntimeError("Download stopped at minimum free disk space")
                                last_space_check = downloaded
                            if downloaded > settings.MAX_DOWNLOAD_BYTES:
                                raise RuntimeError("Episode download exceeds configured maximum size")
                            if (datetime.now() - last_cancel_check).total_seconds() > 2.0:
                                if cancellation_callback and cancellation_callback():
                                    raise RuntimeError("CancelledByUser")
                                last_cancel_check = datetime.now()
                            if total > 0 and progress_callback:
                                percent = int(downloaded * 100 / total)
                                if percent % 5 == 0 and percent != last_percent:
                                    progress_callback(percent)
                                    last_percent = percent
            if cancellation_callback and cancellation_callback():
                raise RuntimeError("CancelledByUser")
            os.replace(partial_path, final_path)
            return str(final_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise


class YouTubeSourceAdapter:
    source_types = frozenset({"youtube_channel", "youtube_playlist"})

    def resolve(self, url: str) -> ResolvedSource:
        resolved: YouTubeResolvedSource = resolve_youtube_source(url)
        return ResolvedSource(
            source_type=resolved.source_type,
            external_id=resolved.external_id,
            canonical_url=resolved.canonical_url,
            title=resolved.title,
            slug=resolved.slug,
            image_url=resolved.image_url,
            description=resolved.description,
        )

    async def discover(self, source_type: str, canonical_url: str) -> SourceDiscovery:
        result = await asyncio.to_thread(discover_youtube_entries, source_type, canonical_url)
        return SourceDiscovery(entries=result.entries, truncated=result.truncated)

    async def download(
        self,
        media_url: str,
        episode_dir: str,
        *,
        progress_callback: Callable[[int], None] | None = None,
        cancellation_callback: Callable[[], bool] | None = None,
    ) -> str:
        return await asyncio.to_thread(
            download_youtube_audio,
            media_url,
            episode_dir,
            progress_callback=progress_callback,
            cancellation_callback=cancellation_callback,
        )


RSS_SOURCE_ADAPTER = RssSourceAdapter()
YOUTUBE_SOURCE_ADAPTER = YouTubeSourceAdapter()


def get_source_adapter(source_type: str) -> SourceAdapter:
    if source_type in RSS_SOURCE_ADAPTER.source_types:
        return RSS_SOURCE_ADAPTER
    if source_type in YOUTUBE_SOURCE_ADAPTER.source_types:
        return YOUTUBE_SOURCE_ADAPTER
    raise ValueError(f"Unsupported subscription source type: {source_type}")


def resolve_source(url: str) -> ResolvedSource:
    reference = classify_youtube_url(url)
    adapter = YOUTUBE_SOURCE_ADAPTER if reference is not None else RSS_SOURCE_ADAPTER
    return adapter.resolve(url)


def is_youtube_input(value: str) -> bool:
    return looks_like_youtube_url(value)
