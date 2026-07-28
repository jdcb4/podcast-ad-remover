import io
from pathlib import Path

from PIL import Image

from app.core.artwork import ArtworkWatermarker, effective_artwork_url
from app.core.models import SubscriptionCreate
from app.infra.database import get_db_connection, init_db
from app.infra.repository import SubscriptionRepository


def _image_bytes(size=(600, 600), color=(180, 40, 80)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


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
    assert Path(generated).is_file()
    assert refreshed.watermark_artwork is True
    assert refreshed.watermarked_image_path == generated
    assert len(refreshed.watermarked_image_hash) == 64
    assert effective_artwork_url(refreshed, "https://podcasts.example").startswith(
        f"https://podcasts.example/artwork/{sub.id}.png?v="
    )

    with Image.open(generated) as image:
        assert image.size == (600, 600)
        assert image.getpixel((560, 560))[:3] != (180, 40, 80)

    assert service.reconcile(sub.id) == generated
    service.clear(sub.id)
    cleared = repo.get_by_id(sub.id)
    assert not Path(generated).exists()
    assert cleared.watermarked_image_path is None


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
