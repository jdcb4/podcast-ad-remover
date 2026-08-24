from pathlib import Path
from types import SimpleNamespace
from xml.etree.ElementTree import fromstring

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.rss_gen import RSSGenerator
from app.core.unified_feed import (
    DEFAULT_UNIFIED_FEED_DESCRIPTION,
    DEFAULT_UNIFIED_FEED_TITLE,
    normalize_unified_feed_settings,
    resolve_unified_feed_settings,
)
from app.infra.database import get_db_connection, init_db


ITUNES_NAMESPACE = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


class UnifiedEpisodeRepository:
    def get_completed_with_subscription_info(self):
        return [
            {
                "subscription_id": 1,
                "podcast_title": "Example & Friends",
                "podcast_slug": "example",
                "title": "An <interesting> episode",
                "guid": "episode-guid",
                "original_url": "https://example.com/episodes/1",
                "pub_date": None,
                "duration": None,
                "local_filename": str(
                    settings.PODCASTS_DIR + "/example/episode-guid/episode.mp3"
                ),
                "file_size": 123,
                "ai_summary": None,
                "description": "Episode notes",
            }
        ]


class UnifiedSubscriptionRepository:
    def get_all(self):
        return [
            SimpleNamespace(
                id=1,
                image_url="https://example.com/show.png",
                watermark_artwork=False,
                watermarked_image_path=None,
                watermarked_image_hash=None,
            )
        ]


def generate_unified_xml(monkeypatch, global_settings):
    import app.core.utils as utils

    monkeypatch.setattr(utils, "get_global_settings", lambda: global_settings)
    generator = RSSGenerator()
    generator.ep_repo = UnifiedEpisodeRepository()
    generator.sub_repo = UnifiedSubscriptionRepository()
    output_path = generator.generate_unified_feed()
    return fromstring(Path(output_path).read_text(encoding="utf-8"))


def test_unified_feed_settings_resolve_legacy_defaults():
    resolved = resolve_unified_feed_settings({}, "https://podcasts.example/")

    assert resolved == {
        "title": DEFAULT_UNIFIED_FEED_TITLE,
        "description": DEFAULT_UNIFIED_FEED_DESCRIPTION,
        "include_podcast_name": True,
        "custom_artwork_url": None,
        "artwork_url": "https://podcasts.example/static/unified_feed_cover.png",
    }


def test_unified_feed_settings_normalize_custom_values():
    normalized = normalize_unified_feed_settings(
        "  My Feed  ",
        "  My combined podcasts  ",
        False,
        "  http://media-server.local/cover.webp  ",
    )

    assert normalized == {
        "title": "My Feed",
        "description": "My combined podcasts",
        "include_podcast_name": False,
        "custom_artwork_url": "http://media-server.local/cover.webp",
    }


@pytest.mark.parametrize(
    "artwork_url",
    ["file:///tmp/cover.png", "javascript:alert(1)", "cover.png"],
)
def test_unified_feed_settings_reject_invalid_artwork_urls(artwork_url):
    with pytest.raises(ValueError, match="HTTP and HTTPS"):
        normalize_unified_feed_settings(
            DEFAULT_UNIFIED_FEED_TITLE,
            DEFAULT_UNIFIED_FEED_DESCRIPTION,
            True,
            artwork_url,
        )


def test_unified_feed_generation_preserves_defaults(isolated_data_dir, monkeypatch):
    rss = generate_unified_xml(
        monkeypatch,
        {"app_external_url": "https://podcasts.example"},
    )
    channel = rss.find("channel")

    assert channel.findtext("title") == DEFAULT_UNIFIED_FEED_TITLE
    assert channel.findtext("description") == DEFAULT_UNIFIED_FEED_DESCRIPTION
    assert channel.findtext("item/title") == "[Example & Friends] An <interesting> episode"
    assert (
        channel.find("itunes:image", ITUNES_NAMESPACE).attrib["href"]
        == "https://podcasts.example/static/unified_feed_cover.png"
    )


def test_unified_feed_generation_uses_custom_preferences(isolated_data_dir, monkeypatch):
    rss = generate_unified_xml(
        monkeypatch,
        {
            "app_external_url": "https://podcasts.example",
            "unified_feed_title": "Paul's Shows & More",
            "unified_feed_description": "A <custom> description & collection",
            "unified_feed_include_podcast_name": 0,
            "unified_feed_artwork_url": "https://images.example/cover.png?size=large&v=2",
        },
    )
    channel = rss.find("channel")

    assert channel.findtext("title") == "Paul's Shows & More"
    assert channel.findtext("description") == "A <custom> description & collection"
    assert channel.findtext("item/title") == "An <interesting> episode"
    assert (
        channel.find("itunes:image", ITUNES_NAMESPACE).attrib["href"]
        == "https://images.example/cover.png?size=large&v=2"
    )


def test_unified_feed_admin_page_updates_and_resets_settings(
    isolated_data_dir,
    monkeypatch,
):
    init_db()
    monkeypatch.setattr(settings, "PROCESSOR_ENABLED", False)
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET app_external_url = 'https://podcasts.example' WHERE id = 1"
        )
        conn.commit()

    from app.main import app

    with TestClient(app) as client:
        page = client.get("/admin/unified-feed")
        assert page.status_code == 200
        assert DEFAULT_UNIFIED_FEED_TITLE in page.text
        assert "https://podcasts.example/feed/unified.xml" in page.text

        response = client.post(
            "/admin/unified-feed/update",
            data={
                "unified_feed_title": "My Combined Feed",
                "unified_feed_description": "Everything in one place",
                "unified_feed_artwork_url": "https://images.example/combined.png",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        customized_page = client.get("/admin/unified-feed")
        assert "My Combined Feed" in customized_page.text
        assert "https://images.example/combined.png" in customized_page.text

        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT unified_feed_title, unified_feed_description,
                       unified_feed_include_podcast_name, unified_feed_artwork_url
                FROM app_settings WHERE id = 1
                """
            ).fetchone()

        assert row["unified_feed_title"] == "My Combined Feed"
        assert row["unified_feed_description"] == "Everything in one place"
        assert row["unified_feed_include_podcast_name"] == 0
        assert row["unified_feed_artwork_url"] == "https://images.example/combined.png"

        reset_response = client.post(
            "/admin/unified-feed/reset",
            follow_redirects=False,
        )
        assert reset_response.status_code == 303

    with get_db_connection() as conn:
        reset_row = conn.execute(
            """
            SELECT unified_feed_title, unified_feed_description,
                   unified_feed_include_podcast_name, unified_feed_artwork_url
            FROM app_settings WHERE id = 1
            """
        ).fetchone()

    assert reset_row["unified_feed_title"] == DEFAULT_UNIFIED_FEED_TITLE
    assert reset_row["unified_feed_description"] == DEFAULT_UNIFIED_FEED_DESCRIPTION
    assert reset_row["unified_feed_include_podcast_name"] == 1
    assert reset_row["unified_feed_artwork_url"] is None


def test_unified_feed_admin_rejects_invalid_artwork_url(
    isolated_data_dir,
    monkeypatch,
):
    init_db()
    monkeypatch.setattr(settings, "PROCESSOR_ENABLED", False)
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET app_external_url = 'https://podcasts.example' WHERE id = 1"
        )
        conn.commit()

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/admin/unified-feed/update",
            data={
                "unified_feed_title": "Unsafe Feed",
                "unified_feed_description": "Should not be saved",
                "unified_feed_include_podcast_name": "true",
                "unified_feed_artwork_url": "file:///tmp/cover.png",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/unified-feed?error=")
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT unified_feed_title FROM app_settings WHERE id = 1"
        ).fetchone()
    assert row["unified_feed_title"] == DEFAULT_UNIFIED_FEED_TITLE
