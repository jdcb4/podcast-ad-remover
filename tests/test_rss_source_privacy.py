from xml.etree.ElementTree import fromstring

from app.core.rss_gen import RSSGenerator
from app.infra.database import get_db_connection, init_db

FEED_URL = "https://example.com/feed.rss?token=FEEDSECRET"
MEDIA_URL = "https://cdn.example.com/episode.mp3?token=MEDIASECRET"


def seed_library(tmp_path):
    init_db()
    audio = tmp_path / "podcasts" / "show" / "episode-1" / "processed.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET app_external_url='https://par.example.net' WHERE id=1")
        conn.execute("INSERT INTO users (id,username,password_hash) VALUES (5,'member','unused')")
        conn.execute(
            "INSERT INTO subscriptions (id,feed_url,title,slug,owner_user_id) VALUES (1,?,'Show','show',5)",
            (FEED_URL,),
        )
        conn.execute("INSERT INTO user_subscriptions (user_id,subscription_id) VALUES (5,1)")
        # No description, so the feed would otherwise fall back to quoting the source URL.
        conn.execute(
            """INSERT INTO episodes (id,subscription_id,guid,title,original_url,status,pub_date,local_filename,file_size)
               VALUES (1,1,'guid-1','One',?,'completed','2026-01-01 00:00:00',?,5)""",
            (MEDIA_URL, str(audio)),
        )
        conn.commit()


def read_feed(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def assert_private(xml):
    assert "FEEDSECRET" not in xml
    assert "MEDIASECRET" not in xml
    assert "cdn.example.com" not in xml
    root = fromstring(xml)
    assert root.findtext("./channel/link") == "https://par.example.net"
    items = root.findall("./channel/item")
    assert len(items) == 1
    assert items[0].find("link") is None
    assert items[0].find("enclosure").get("url") == "https://par.example.net/audio/show/episode-1/processed.mp3"


def test_podcast_feed_does_not_publish_source_urls(isolated_data_dir):
    seed_library(isolated_data_dir)
    assert_private(read_feed(RSSGenerator().generate_feed(1)))


def test_unified_feeds_do_not_publish_source_urls(isolated_data_dir):
    seed_library(isolated_data_dir)
    generator = RSSGenerator()
    assert_private(read_feed(generator.generate_unified_feed()))
    assert_private(read_feed(generator.generate_unified_feed(user_id=5)))


def test_regenerate_all_corrects_feeds_already_on_disk(isolated_data_dir):
    seed_library(isolated_data_dir)
    feeds = isolated_data_dir / "feeds"
    stale = f"<rss><channel><link>{FEED_URL}</link></channel></rss>"
    for name in ("show.xml", "unified.xml", "unified-user-5.xml"):
        (feeds / name).write_text(stale, encoding="utf-8")
    (feeds / "unified-user-notes.xml").write_text("unrelated", encoding="utf-8")

    RSSGenerator().regenerate_all()

    for name in ("show.xml", "unified.xml", "unified-user-5.xml"):
        assert_private((feeds / name).read_text(encoding="utf-8"))
    assert (feeds / "unified-user-notes.xml").read_text(encoding="utf-8") == "unrelated"


def test_regenerate_all_continues_after_a_failure(isolated_data_dir, monkeypatch):
    seed_library(isolated_data_dir)
    generator = RSSGenerator()

    def fail(_subscription_id):
        raise OSError("disk problem")

    monkeypatch.setattr(generator, "generate_feed", fail)
    generator.regenerate_all()
    assert_private((isolated_data_dir / "feeds" / "unified.xml").read_text(encoding="utf-8"))
