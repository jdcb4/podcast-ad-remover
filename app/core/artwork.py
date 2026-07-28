"""Safe generation and serving helpers for optional ad-free podcast artwork."""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.url_utils import validate_http_url, validate_redirect_target
from app.infra.database import get_db_connection
from app.infra.repository import SubscriptionRepository


logger = logging.getLogger(__name__)

MAX_ARTWORK_BYTES = 12 * 1024 * 1024
MAX_ARTWORK_EDGE = 8000
OUTPUT_MAX_EDGE = 3000
BADGE_SCALE = 0.30
BADGE_MARGIN = 0.035
BADGE_PATH = Path(__file__).resolve().parents[1] / "web" / "static" / "ad_free_badge.png"


def effective_artwork_url(subscription, base_url: str) -> str | None:
    path = getattr(subscription, "watermarked_image_path", None)
    if getattr(subscription, "watermark_artwork", False) and path and Path(path).is_file():
        version = getattr(subscription, "watermarked_image_hash", None) or "current"
        return f"{base_url.rstrip('/')}/artwork/{subscription.id}.png?v={version[:12]}"
    return getattr(subscription, "image_url", None)


class ArtworkWatermarker:
    def __init__(self):
        self.sub_repo = SubscriptionRepository()

    @staticmethod
    def _download(url: str) -> bytes:
        validate_http_url(url, allow_private=settings.ALLOW_PRIVATE_FEEDS)
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                validate_redirect_target(
                    url,
                    str(response.url),
                    allow_private=settings.ALLOW_PRIVATE_FEEDS,
                )
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type and not content_type.startswith("image/"):
                    raise ValueError("Podcast artwork response is not an image")
                content_length = int(response.headers.get("content-length") or 0)
                if content_length > MAX_ARTWORK_BYTES:
                    raise ValueError("Podcast artwork exceeds the size limit")

                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_ARTWORK_BYTES:
                        raise ValueError("Podcast artwork exceeds the size limit")
                    chunks.append(chunk)
                return b"".join(chunks)

    @staticmethod
    def _output_path(subscription_id: int) -> Path:
        return Path(settings.ARTWORK_DIR) / f"{int(subscription_id)}.png"

    def clear(self, subscription_id: int) -> None:
        output = self._output_path(subscription_id).resolve()
        root = Path(settings.ARTWORK_DIR).resolve()
        if output.parent == root and output.exists():
            output.unlink()
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE subscriptions
                SET watermarked_image_path = NULL, watermarked_image_hash = NULL
                WHERE id = ?
                """,
                (subscription_id,),
            )
            conn.commit()

    def reconcile(self, subscription_id: int) -> str | None:
        sub = self.sub_repo.get_by_id(subscription_id)
        if not sub or not sub.watermark_artwork or not sub.image_url:
            self.clear(subscription_id)
            return None

        source = self._download(sub.image_url)
        badge_bytes = BADGE_PATH.read_bytes()
        digest = hashlib.sha256(source + badge_bytes + b"artwork-v1").hexdigest()
        output = self._output_path(subscription_id)
        if (
            sub.watermarked_image_hash == digest
            and sub.watermarked_image_path
            and Path(sub.watermarked_image_path).is_file()
        ):
            return sub.watermarked_image_path

        try:
            with Image.open(io.BytesIO(source)) as opened:
                opened.load()
                if opened.width > MAX_ARTWORK_EDGE or opened.height > MAX_ARTWORK_EDGE:
                    raise ValueError("Podcast artwork dimensions exceed the limit")
                artwork = opened.convert("RGBA")
        except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("Podcast artwork could not be decoded safely") from exc

        if max(artwork.size) > OUTPUT_MAX_EDGE:
            artwork.thumbnail((OUTPUT_MAX_EDGE, OUTPUT_MAX_EDGE), Image.Resampling.LANCZOS)

        with Image.open(BADGE_PATH) as badge_opened:
            badge = badge_opened.convert("RGBA")
            bounds = badge.getbbox()
            if bounds:
                badge = badge.crop(bounds)

        badge_edge = max(48, round(min(artwork.size) * BADGE_SCALE))
        badge.thumbnail((badge_edge, badge_edge), Image.Resampling.LANCZOS)
        margin = max(8, round(min(artwork.size) * BADGE_MARGIN))
        position = (
            max(0, artwork.width - badge.width - margin),
            max(0, artwork.height - badge.height - margin),
        )
        artwork.alpha_composite(badge, position)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.png")
        artwork.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, output)

        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE subscriptions
                SET watermarked_image_path = ?, watermarked_image_hash = ?
                WHERE id = ?
                """,
                (str(output), digest, subscription_id),
            )
            conn.commit()
        logger.info("Generated ad-free artwork for subscription %s", subscription_id)
        return str(output)
