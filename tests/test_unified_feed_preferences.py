from pathlib import Path
import sqlite3
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
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
                "published_guid": "stable-published-guid",
                "original_url": "https://example.com/episodes/1",
                "pub_date": None,
                "duration": 60,
                "output_duration": 51,
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
    assert channel.findtext("item/guid") == "stable-published-guid"
    assert channel.findtext("item/itunes:duration", namespaces=ITUNES_NAMESPACE) == "51"
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


@pytest.mark.parametrize("field", ["title", "description", "artwork_url"])
@pytest.mark.parametrize("character", ["\x00", "\x0c", "\ud800", "\uffff"])
def test_unified_feed_rejects_xml_invalid_characters(field, character):
    values = {
        "title": "My feed",
        "description": "My description",
        "include_podcast_name": True,
        "artwork_url": "https://images.example/cover.png",
    }
    values[field] += character + "suffix"
    with pytest.raises(ValueError, match="characters that cannot be used in RSS"):
        normalize_unified_feed_settings(**values)


@pytest.mark.parametrize("field", ["title", "description", "artwork_url"])
def test_invalid_metadata_keeps_saved_settings_and_feed(
    isolated_data_dir, monkeypatch, field,
):
    init_db()
    monkeypatch.setattr(settings, "PROCESSOR_ENABLED", False)
    from app.main import app

    with TestClient(app) as client:
        feed = Path(RSSGenerator().generate_unified_feed())
        before = feed.read_bytes()
        values = {
            "unified_feed_title": "Changed title",
            "unified_feed_description": "Changed description",
            "unified_feed_artwork_url": "https://images.example/cover.png",
        }
        values[f"unified_feed_{field}"] += "\x0csuffix"
        response = client.post(
            "/admin/unified-feed/update", data=values, follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/unified-feed?error=")
    assert feed.read_bytes() == before
    assert fromstring(before).findtext("channel/title") == DEFAULT_UNIFIED_FEED_TITLE
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
    assert row["unified_feed_title"] == DEFAULT_UNIFIED_FEED_TITLE
    assert row["unified_feed_description"] == DEFAULT_UNIFIED_FEED_DESCRIPTION
    assert row["unified_feed_artwork_url"] is None


def test_legacy_metadata_still_generates_valid_xml(isolated_data_dir, monkeypatch):
    rss = generate_unified_xml(monkeypatch, {
        "app_external_url": "https://podcasts.example",
        "unified_feed_title": "Café\x0b 🎧",
        "unified_feed_description": "First\x0c page\nSecond\tpage",
        "unified_feed_artwork_url": "https://images.example/co\x01ver.png",
    })
    channel = rss.find("channel")
    assert channel.findtext("title") == "Café 🎧"
    assert channel.findtext("description") == "First page\nSecond\tpage"
    assert channel.find("itunes:image", ITUNES_NAMESPACE).attrib["href"] == (
        "https://images.example/cover.png"
    )


def test_settings_feed_address_uses_shared_session_token(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, "PROCESSOR_ENABLED", False)
    monkeypatch.setattr(settings, "SESSION_SECRET_KEY", "unified-feed-test-secret")
    with get_db_connection() as conn:
        conn.execute("""UPDATE app_settings SET enable_feed_auth = 1,
                     app_external_url = 'http://testserver' WHERE id = 1""")
        conn.execute("""INSERT INTO subscriptions (feed_url, title, slug)
                     VALUES ('https://example.com/feed.xml', 'Example', 'example')""")
        conn.commit()
    from app.main import app
    from app.infra.repository import FeedTokenRepository

    with TestClient(app) as client:
        page = client.get("/admin/unified-feed")
        feed_url = page.context["feed_url"]
        token = parse_qs(urlsplit(feed_url).query)["token"][0]
        assert feed_url in page.text
        assert client.get(feed_url).status_code == 200
        assert client.get("/feed/unified.xml").status_code == 401
        assert client.get("/admin/unified-feed").context["feed_url"] == feed_url
        assert client.get("/").context["unified_links"]["direct"] == feed_url
        assert client.get("/subscribe").context["unified_links"]["direct"] == (
            "http://testserver/feed/unified.xml"
        )
        FeedTokenRepository().revoke(token)
        refreshed_url = client.get("/admin/unified-feed").context["feed_url"]
        assert refreshed_url != feed_url
        assert client.get(refreshed_url).status_code == 200


@pytest.mark.parametrize("page_url, artwork_url, expected_preview", [
    ("https://testserver", "https://images.example/cover.png", "https://images.example/cover.png"),
    ("http://testserver", "http://media-server.local/cover.png", None),
    ("https://testserver", "http://testserver/cover.png", None),
    ("http://testserver", "http://testserver:80/cover.png", "http://testserver:80/cover.png"),
    ("http://testserver", "http://testserver:8080/cover.png", None),
    ("http://testserver", "", "/static/unified_feed_cover.png"),
])
def test_artwork_preview_matches_browser_policy(
    isolated_data_dir, monkeypatch, page_url, artwork_url, expected_preview,
):
    init_db()
    monkeypatch.setattr(settings, "PROCESSOR_ENABLED", False)
    from app.main import app

    with TestClient(app, base_url=page_url) as client:
        response = client.post("/admin/unified-feed/update", data={
            "unified_feed_title": "My feed",
            "unified_feed_description": "My description",
            "unified_feed_artwork_url": artwork_url,
        }, follow_redirects=False)
        assert response.status_code == 303
        page = client.get("/admin/unified-feed")
        assert "img-src 'self' data: blob: https:;" in page.headers["content-security-policy"]
        if expected_preview:
            assert f'src="{expected_preview}" alt="Current unified feed artwork"' in page.text
        else:
            assert 'alt="Current unified feed artwork"' not in page.text
            assert "Use an HTTPS artwork URL to preview it here" in page.text
            channel = fromstring(client.get("/feed/unified.xml").content).find("channel")
            assert channel.find("itunes:image", ITUNES_NAMESPACE).attrib["href"] == artwork_url


@pytest.mark.parametrize("previous_version", ["dev", "contributor_pr"])
def test_unified_feed_upgrade_preserves_both_database_histories(
    isolated_data_dir, monkeypatch, previous_version,
):
    import app.infra.database as database

    preference_migration = "20260824_0013_unified_feed_preferences"
    migrations = database.FORMAL_MIGRATIONS
    if previous_version == "dev":
        previous_migrations = [item for item in migrations if item[0] != preference_migration]
    else:
        previous_migrations = [item for item in migrations if item[0] <= preference_migration]
    with monkeypatch.context() as previous:
        previous.setattr(database, "FORMAL_MIGRATIONS", previous_migrations)
        init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET whisper_model = 'tiny', retention_days = 123 WHERE id = 1")
        if previous_version == "contributor_pr":
            conn.execute("""UPDATE app_settings SET unified_feed_title = 'Existing custom feed',
                         unified_feed_include_podcast_name = 0 WHERE id = 1""")
        conn.commit()

    init_db()
    init_db()

    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        applied = {record[0] for record in conn.execute("SELECT version FROM schema_migrations")}
    assert applied == {version for version, _ in migrations}
    assert row["whisper_model"] == "tiny"
    assert row["retention_days"] == 123
    assert row["unified_feed_title"] == (
        "Existing custom feed" if previous_version == "contributor_pr" else DEFAULT_UNIFIED_FEED_TITLE
    )
    assert row["unified_feed_include_podcast_name"] == (0 if previous_version == "contributor_pr" else 1)
    backups = list((isolated_data_dir / "backups").glob("*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert {record[0] for record in backup.execute("SELECT version FROM schema_migrations")} == {
            version for version, _ in previous_migrations
        }
