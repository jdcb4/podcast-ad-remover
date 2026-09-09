import hashlib
import io
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest
import httpx
from PIL import Image

from app.core.artwork import BADGE_PATH, ArtworkWatermarker, effective_artwork_url
from app.core.config import settings
from app.core.models import SubscriptionCreate
from app.infra.database import get_db_connection, init_db
from app.infra.repository import SubscriptionRepository
from app.web.router import router as web_router
from app.web.router import serve_watermarked_artwork


CACHE_CONTROL = "public, max-age=31536000, immutable"


@pytest.mark.parametrize("content_type,valid_image,accepted", [
    ("Application/Octet-Stream ; charset=binary", True, True),
    ("application/octet-stream", False, False),
    ("text/html", True, False),
])
def test_artwork_download_validates_generic_binary_responses(
    isolated_data_dir, monkeypatch, content_type, valid_image, accepted,
):
    init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET default_watermark_artwork=1 WHERE id=1")
        conn.commit()
    repo = SubscriptionRepository()
    sub = repo.create(SubscriptionCreate(feed_url="https://example.com/feed.xml"),
                      "Binary artwork", "binary-artwork", image_url="https://example.com/cover")
    legacy = Path(settings.ARTWORK_DIR) / f"{sub.id}.png"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(_image_bytes())
    with get_db_connection() as conn:
        conn.execute("UPDATE subscriptions SET watermarked_image_path=?,watermarked_image_hash=? WHERE id=?",
                     (str(legacy), "legacy", sub.id))
        conn.commit()
    source = io.BytesIO()
    Image.new("RGB", (1534, 1534), (180, 40, 80)).save(source, format="JPEG")
    body = source.getvalue() if valid_image else b"<html>Temporarily unavailable</html>"
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": content_type, "content-length": str(len(body))}, content=body,
    ))
    client_type = httpx.Client
    monkeypatch.setattr(settings, "ALLOW_PRIVATE_FEEDS", True)
    monkeypatch.setattr("app.core.artwork.httpx.Client", lambda **kwargs: client_type(transport=transport, **kwargs))
    if accepted:
        output = ArtworkWatermarker().reconcile(sub.id)
        with Image.open(output) as image:
            assert image.format == "JPEG" and image.size == (1400, 1400)
        assert not legacy.exists()
        assert repo.get_by_id(sub.id).watermarked_image_path == output
    else:
        with pytest.raises(ValueError, match="not an image|could not be decoded safely"):
            ArtworkWatermarker().reconcile(sub.id)
        assert legacy.is_file()
        assert repo.get_by_id(sub.id).watermarked_image_path == str(legacy)
        assert not (Path(settings.ARTWORK_DIR) / f"{sub.id}.jpg").exists()


def _image_bytes(size=(600, 600), color=(180, 40, 80)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _transparent_corner_image_bytes(size=(600, 600), color=(180, 40, 80, 255)):
    """An RGBA image whose top-left quadrant is fully transparent."""
    image = Image.new("RGBA", size, color)
    hole = Image.new("RGBA", (size[0] // 2, size[1] // 2), (0, 0, 0, 0))
    image.paste(hole, (0, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _watermarked_subscription(slug, source_bytes, monkeypatch):
    """Create a subscription with watermarking enabled and a stubbed artwork download."""
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET default_watermark_artwork = 1 WHERE id = 1")
        conn.commit()

    sub = repo.create(
        SubscriptionCreate(feed_url=f"https://example.com/{slug}.xml"),
        slug.title(),
        slug,
        image_url=f"https://example.com/{slug}.png",
    )
    monkeypatch.setattr(
        ArtworkWatermarker, "_download", staticmethod(lambda url: source_bytes)
    )
    return repo, sub


def test_watermark_generation_is_inherited_cached_and_removable(isolated_data_dir, monkeypatch):
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET default_watermark_artwork = 1 WHERE id = 1")
        conn.commit()

    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/feed.xml"),
        "Watermarked",
        "watermarked",
        image_url="https://example.com/cover.png",
    )
    monkeypatch.setattr(ArtworkWatermarker, "_download", staticmethod(lambda url: _image_bytes()))

    service = ArtworkWatermarker()
    generated = service.reconcile(sub.id)
    refreshed = repo.get_by_id(sub.id)

    assert generated is not None
    assert generated.endswith(".jpg")
    assert Path(generated).is_file()
    assert refreshed.watermark_artwork is True
    assert refreshed.watermarked_image_path == generated
    assert len(refreshed.watermarked_image_hash) == 64
    assert effective_artwork_url(refreshed, "https://podcasts.example").startswith(
        f"https://podcasts.example/artwork/{sub.id}.jpg?v="
    )

    with Image.open(generated) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (600, 600)
        assert image.getpixel((560, 560))[:3] != (180, 40, 80)

    assert service.reconcile(sub.id) == generated
    service.clear(sub.id)
    cleared = repo.get_by_id(sub.id)
    assert not Path(generated).exists()
    assert cleared.watermarked_image_path is None


def test_watermark_downscales_large_source_to_the_output_cap(isolated_data_dir, monkeypatch):
    _, sub = _watermarked_subscription("large", _image_bytes(size=(3000, 3000)), monkeypatch)

    generated = ArtworkWatermarker().reconcile(sub.id)

    with Image.open(generated) as image:
        assert image.size == (1400, 1400)
    assert Path(generated).stat().st_size < 500 * 1024


def test_watermark_does_not_upscale_a_small_source(isolated_data_dir, monkeypatch):
    _, sub = _watermarked_subscription("small", _image_bytes(size=(800, 800)), monkeypatch)

    generated = ArtworkWatermarker().reconcile(sub.id)

    with Image.open(generated) as image:
        assert image.size == (800, 800)


def test_watermark_flattens_transparency_onto_white(isolated_data_dir, monkeypatch):
    _, sub = _watermarked_subscription("alpha", _transparent_corner_image_bytes(), monkeypatch)

    generated = ArtworkWatermarker().reconcile(sub.id)

    with Image.open(generated) as image:
        assert image.mode == "RGB"
        red, green, blue = image.getpixel((10, 10))
    assert min(red, green, blue) > 240


def test_reconcile_regenerates_and_removes_a_stale_png_output(isolated_data_dir, monkeypatch):
    """A pre-migration PNG output is replaced by a JPEG and its stored hash changes."""
    source = _image_bytes()
    repo, sub = _watermarked_subscription("stale", source, monkeypatch)

    legacy_png = Path(settings.ARTWORK_DIR) / f"{sub.id}.png"
    legacy_png.parent.mkdir(parents=True, exist_ok=True)
    legacy_png.write_bytes(source)
    legacy_digest = hashlib.sha256(source + BADGE_PATH.read_bytes() + b"artwork-v1").hexdigest()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE subscriptions
            SET watermarked_image_path = ?, watermarked_image_hash = ?
            WHERE id = ?
            """,
            (str(legacy_png), legacy_digest, sub.id),
        )
        conn.commit()

    generated = ArtworkWatermarker().reconcile(sub.id)
    refreshed = repo.get_by_id(sub.id)

    assert generated.endswith(".jpg")
    assert Path(generated).is_file()
    assert not legacy_png.exists()
    assert refreshed.watermarked_image_hash != legacy_digest


def test_effective_artwork_url_follows_the_stored_file_extension(isolated_data_dir, monkeypatch):
    """During migration the URL must match the file actually on disk."""
    repo, sub = _watermarked_subscription("transition", _image_bytes(), monkeypatch)

    legacy_png = Path(settings.ARTWORK_DIR) / f"{sub.id}.png"
    legacy_png.parent.mkdir(parents=True, exist_ok=True)
    legacy_png.write_bytes(_image_bytes())
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE subscriptions
            SET watermarked_image_path = ?, watermarked_image_hash = ?
            WHERE id = ?
            """,
            (str(legacy_png), "a" * 64, sub.id),
        )
        conn.commit()

    before = effective_artwork_url(repo.get_by_id(sub.id), "https://podcasts.example")
    assert before.startswith(f"https://podcasts.example/artwork/{sub.id}.png?v=")

    ArtworkWatermarker().reconcile(sub.id)

    after = effective_artwork_url(repo.get_by_id(sub.id), "https://podcasts.example")
    assert after.startswith(f"https://podcasts.example/artwork/{sub.id}.jpg?v=")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["execute", "commit"])
async def test_failed_jpeg_publication_keeps_legacy_artwork_available(
    isolated_data_dir, monkeypatch, failure_point
):
    source = _image_bytes()
    repo, sub = _watermarked_subscription("failed-publication", source, monkeypatch)
    legacy = Path(settings.ARTWORK_DIR) / f"{sub.id}.png"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(source)
    old_hash = hashlib.sha256(source + BADGE_PATH.read_bytes() + b"artwork-v1").hexdigest()
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE subscriptions SET watermarked_image_path=?, watermarked_image_hash=? WHERE id=?",
            (str(legacy), old_hash, sub.id),
        )
        conn.commit()

    @contextmanager
    def failing_connection():
        with get_db_connection() as conn:
            class FailingPublication:
                def execute(self, *args):
                    if failure_point == "execute":
                        raise sqlite3.OperationalError("database is locked")
                    return conn.execute(*args)

                def commit(self):
                    raise sqlite3.OperationalError("commit failed")

            yield FailingPublication()

    with monkeypatch.context() as patch:
        patch.setattr("app.core.artwork.get_db_connection", failing_connection)
        with pytest.raises(sqlite3.OperationalError):
            ArtworkWatermarker().reconcile(sub.id)

    saved = repo.get_by_id(sub.id)
    assert saved.watermarked_image_path == str(legacy)
    assert saved.watermarked_image_hash == old_hash
    assert legacy.read_bytes() == source
    response = await serve_watermarked_artwork(sub.id)
    assert response.media_type == "image/png"
    assert Path(response.path).read_bytes() == source

    # A normal retry must finish the transition and remove the superseded PNG.
    generated = ArtworkWatermarker().reconcile(sub.id)
    assert generated.endswith(".jpg")
    assert not legacy.exists()
    assert repo.get_by_id(sub.id).watermarked_image_path == generated
    assert (await serve_watermarked_artwork(sub.id)).media_type == "image/jpeg"


def test_cached_jpeg_retries_interrupted_legacy_cleanup(isolated_data_dir, monkeypatch):
    _, sub = _watermarked_subscription("interrupted-cleanup", _image_bytes(), monkeypatch)
    service = ArtworkWatermarker()
    generated = service.reconcile(sub.id)
    legacy = Path(settings.ARTWORK_DIR) / f"{sub.id}.png"
    legacy.write_bytes(_image_bytes())
    before = Path(generated).stat().st_mtime_ns

    assert service.reconcile(sub.id) == generated
    assert Path(generated).stat().st_mtime_ns == before
    assert not legacy.exists()


@pytest.mark.asyncio
async def test_artwork_route_serves_jpeg_with_immutable_caching(isolated_data_dir, monkeypatch):
    _, sub = _watermarked_subscription("served", _image_bytes(), monkeypatch)
    generated = ArtworkWatermarker().reconcile(sub.id)

    response = await serve_watermarked_artwork(sub.id)

    assert str(response.path) == generated
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"] == CACHE_CONTROL


def test_artwork_route_still_accepts_the_legacy_png_path():
    """Clients holding stale feed XML must not get a 404 for the old .png URL."""
    paths = {getattr(route, "path", None) for route in web_router.routes}

    assert "/artwork/{subscription_id}.jpg" in paths
    assert "/artwork/{subscription_id}.png" in paths


def test_watermark_rejects_extreme_dimensions(isolated_data_dir, monkeypatch):
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET default_watermark_artwork = 1 WHERE id = 1")
        conn.commit()
    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/wide.xml"),
        "Too Wide",
        "too-wide",
        image_url="https://example.com/wide.png",
    )
    monkeypatch.setattr(
        ArtworkWatermarker,
        "_download",
        staticmethod(lambda url: _image_bytes(size=(8001, 1))),
    )

    try:
        ArtworkWatermarker().reconcile(sub.id)
    except ValueError as exc:
        assert "dimensions exceed" in str(exc)
    else:
        raise AssertionError("Expected oversized artwork to be rejected")


def test_original_artwork_is_used_when_watermark_is_disabled(isolated_data_dir):
    init_db()
    sub = SubscriptionRepository().create(
        SubscriptionCreate(feed_url="https://example.com/plain.xml"),
        "Plain",
        "plain",
        image_url="https://example.com/plain.png",
    )

    assert effective_artwork_url(sub, "https://podcasts.example") == sub.image_url
