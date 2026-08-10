"""Subscription source adapters shared by web, API, and processing flows."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.feed import FeedManager
from app.core.youtube import (
    YouTubeResolvedSource,
    classify_youtube_url,
    looks_like_youtube_url,
    resolve_youtube_source,
)


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


def resolve_source(url: str) -> ResolvedSource:
    reference = classify_youtube_url(url)
    if reference is not None:
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


def is_youtube_input(value: str) -> bool:
    return looks_like_youtube_url(value)
