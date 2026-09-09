"""Safe generation and serving helpers for optional ad-free podcast artwork."""

from __future__ import annotations

from app.core.http_downloads import stream_get

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
# Apple's podcast artwork spec allows 1400-3000px square. 1400 is the smallest
# compliant size, and clients only ever draw a thumbnail from it.
OUTPUT_MAX_EDGE = 1400
JPEG_QUALITY = 82
BADGE_SCALE = 0.30
BADGE_MARGIN = 0.035
BADGE_PATH = Path(__file__).resolve().parents[1] / "web" / "static" / "ad_free_badge.png"
LEGACY_OUTPUT_SUFFIX = ".png"
# Part of the cache digest, so changing the encoding regenerates every file and
# publishes a new ?v= for URLs that clients cache as immutable for a year.
ENCODE_SALT = f"artwork-v2:jpeg:{OUTPUT_MAX_EDGE}:q{JPEG_QUALITY}".encode()


def effective_artwork_url(subscription, base_url: str) -> str | None:
    path = getattr(subscription, "watermarked_image_path", None)
    if getattr(subscription, "watermark_artwork", False) and path and Path(path).is_file():
        version = getattr(subscription, "watermarked_image_hash", None) or "current"
        suffix = Path(path).suffix or ".jpg"
        return f"{base_url.rstrip('/')}/artwork/{subscription.id}{suffix}?v={version[:12]}"
    return getattr(subscription, "image_url", None)


class ArtworkWatermarker:
    def __init__(self):
        self.sub_repo = SubscriptionRepository()

    @staticmethod
    def _download(url: str) -> bytes:
        validate_http_url(url, allow_private=settings.ALLOW_PRIVATE_FEEDS)
        with httpx.Client(trust_env=settings.ALLOW_PRIVATE_FEEDS, timeout=30.0) as client:
            with stream_get(client, url) as response:
                response.raise_for_status()
                validate_redirect_target(
                    url,
                    str(response.url),
                    allow_private=settings.ALLOW_PRIVATE_FEEDS,
                )
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                # Some podcast CDNs serve valid images with a generic binary MIME
                # type. Reconcile still decodes and validates the bounded bytes
                # before publishing any replacement artwork.
                if content_type and content_type != "application/octet-stream" and not content_type.startswith("image/"):
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
        return Path(settings.ARTWORK_DIR) / f"{int(subscription_id)}.jpg"

    @staticmethod
    def _remove_legacy_output(subscription_id: int) -> None:
        """Delete the PNG output written by releases before the JPEG switch."""
        legacy = Path(settings.ARTWORK_DIR) / f"{int(subscription_id)}{LEGACY_OUTPUT_SUFFIX}"
        root = Path(settings.ARTWORK_DIR).resolve()
        try:
            resolved = legacy.resolve()
            if resolved.parent == root and resolved.is_file():
                resolved.unlink()
        except OSError as exc:
            logger.warning(
                "Could not remove legacy artwork for subscription %s: %s", subscription_id, exc
            )

    def clear(self, subscription_id: int) -> None:
        output = self._output_path(subscription_id).resolve()
        root = Path(settings.ARTWORK_DIR).resolve()
        if output.parent == root and output.exists():
            output.unlink()
        self._remove_legacy_output(subscription_id)
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
        digest = hashlib.sha256(source + badge_bytes + ENCODE_SALT).hexdigest()
        output = self._output_path(subscription_id)
        if (
            sub.watermarked_image_hash == digest
            and sub.watermarked_image_path
            and Path(sub.watermarked_image_path).is_file()
        ):
            if Path(sub.watermarked_image_path).resolve() == output.resolve():
                self._remove_legacy_output(subscription_id)
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

        # JPEG has no alpha channel, so composite the result onto white.
        flattened = Image.new("RGB", artwork.size, (255, 255, 255))
        flattened.paste(artwork, mask=artwork.getchannel("A"))

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.jpg")
        flattened.save(
            temporary,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
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
        # The database must point to the JPEG before its predecessor is removed.
        # If publication fails, existing feed URLs can still serve the PNG.
        self._remove_legacy_output(subscription_id)
        logger.info("Generated ad-free artwork for subscription %s", subscription_id)
        return str(output)
