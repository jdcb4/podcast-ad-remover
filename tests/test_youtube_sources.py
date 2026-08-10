import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.models import SubscriptionCreate
from app.core.processor import Processor
from app.core.youtube import (
    PLAYLIST_SCAN_LIMIT,
    YouTubeUrlError,
    YouTubeDownloadCancelled,
    classify_youtube_url,
    discover_youtube_entries,
    download_youtube_audio,
    hydrate_youtube_entry,
    resolve_youtube_source,
)
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, SourceItemRepository, SubscriptionRepository


@pytest.mark.parametrize(
    ("url", "source_type"),
    [
        ("https://www.youtube.com/@example", "youtube_channel"),
        ("https://youtube.com/channel/UC123/videos", "youtube_channel"),
        ("https://m.youtube.com/user/example", "youtube_channel"),
        ("https://www.youtube.com/playlist?list=PL123", "youtube_playlist"),
    ],
)
def test_classify_supported_youtube_sources(url, source_type):
    assert classify_youtube_url(url).source_type == source_type


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/abc123",
        "https://www.youtube.com/watch?v=abc123",
        "https://www.youtube.com/watch?v=abc123&list=PL123",
        "https://www.youtube.com/shorts/abc123",
        "https://www.youtube.com/@example/shorts",
        "https://www.youtube.com/@example/streams",
        "https://www.youtube.com/results?search_query=test",
    ],
)
def test_rejects_non_subscription_youtube_urls(url):
    with pytest.raises(YouTubeUrlError):
        classify_youtube_url(url)


def test_non_youtube_url_is_not_claimed():
    assert classify_youtube_url("https://example.com/feed.xml") is None


def test_resolve_channel_canonicalizes_to_channel_id(monkeypatch):
    class FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, url, download=False):
            assert url.endswith("/@friendly/videos")
            return {
                "id": "uploads",
                "channel_id": "UC_CANONICAL",
                "channel": "Friendly Channel",
                "description": "Channel notes",
                "thumbnail": "https://img.example/cover.jpg",
            }

    monkeypatch.setattr("app.core.youtube._youtube_dl", lambda _options: FakeYDL())
    resolved = resolve_youtube_source("https://www.youtube.com/@friendly")

    assert resolved.external_id == "UC_CANONICAL"
    assert resolved.canonical_url == "https://www.youtube.com/channel/UC_CANONICAL/videos"
    assert resolved.title == "Friendly Channel"


def test_hydration_excludes_vertical_shorts_and_live_content(monkeypatch):
    responses = [
        {
            "id": "short",
            "title": "Vertical short",
            "availability": "public",
            "live_status": "not_live",
            "duration": 79,
            "aspect_ratio": 0.56,
        },
        {
            "id": "live",
            "title": "Live now",
            "availability": "public",
            "live_status": "is_live",
        },
    ]

    class FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _url, download=False):
            return responses.pop(0)

    monkeypatch.setattr("app.core.youtube._youtube_dl", lambda _options: FakeYDL())
    assert hydrate_youtube_entry("short") == (None, "short", False)
    assert hydrate_youtube_entry("live") == (None, "is_live", True)


@pytest.mark.parametrize(
    ("source_type", "count", "expected_limit"),
    [("youtube_channel", 52, 50), ("youtube_playlist", 502, PLAYLIST_SCAN_LIMIT)],
)
def test_discovery_is_bounded_and_reports_truncation(monkeypatch, source_type, count, expected_limit):
    captured = {}

    class FakeYDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _url, download=False):
            return {"entries": [{"id": str(index)} for index in range(count)]}

    monkeypatch.setattr("app.core.youtube._youtube_dl", lambda options: FakeYDL(options))
    result = discover_youtube_entries(source_type, "https://www.youtube.com/source")

    assert len(result.entries) == expected_limit
    assert result.truncated is True
    assert captured["playlistend"] == expected_limit + 1


def test_playlist_discovery_initial_cap_and_new_old_member(isolated_data_dir, monkeypatch):
    init_db()
    subscriptions = SubscriptionRepository()
    subscription = subscriptions.create(
        SubscriptionCreate(feed_url="https://www.youtube.com/playlist?list=PL_TEST"),
        "Test Playlist",
        "test-playlist",
        retention_limit=20,
        source_type="youtube_playlist",
        source_external_id="PL_TEST",
    )

    processor = Processor.__new__(Processor)
    processor.sub_repo = subscriptions
    processor.ep_repo = EpisodeRepository()
    processor.source_item_repo = SourceItemRepository()

    first_entries = [{"id": f"video-{index}"} for index in range(7)]
    discoveries = [
        SimpleNamespace(entries=first_entries, truncated=False),
        SimpleNamespace(entries=first_entries + [{"id": "old-video"}], truncated=False),
    ]
    monkeypatch.setattr("app.core.sources.discover_youtube_entries", lambda *_args: discoveries.pop(0))

    def hydrate(video_id):
        return ({
            "guid": video_id,
            "title": video_id,
            "pub_date": None,
            "original_url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": 60,
            "description": "",
            "file_size": 0,
        }, None, False)

    monkeypatch.setattr("app.core.processor.hydrate_youtube_entry", hydrate)

    asyncio.run(processor.check_feeds(subscription_id=subscription.id, limit=5))
    with get_db_connection() as conn:
        statuses = [
            row["status"]
            for row in conn.execute(
                "SELECT status FROM episodes WHERE subscription_id = ? ORDER BY id",
                (subscription.id,),
            ).fetchall()
        ]
    assert statuses == ["pending"] * 5 + ["unprocessed"] * 2

    asyncio.run(processor.check_feeds(subscription_id=subscription.id, limit=5))
    with get_db_connection() as conn:
        old = conn.execute(
            "SELECT status, pub_date, discovered_at FROM episodes WHERE guid = 'old-video'"
        ).fetchone()
        retained = conn.execute(
            "SELECT COUNT(*) AS count FROM episodes WHERE subscription_id = ?",
            (subscription.id,),
        ).fetchone()
    assert old["status"] == "pending"
    assert old["discovered_at"] is not None
    assert retained["count"] == 8


def test_playlist_reconciliation_marks_removed_members_without_deleting_episode(isolated_data_dir, monkeypatch):
    init_db()
    subscriptions = SubscriptionRepository()
    subscription = subscriptions.create(
        SubscriptionCreate(feed_url="https://www.youtube.com/playlist?list=PL_REMOVE"),
        "Removal Test",
        "removal-test",
        source_type="youtube_playlist",
        source_external_id="PL_REMOVE",
    )
    processor = Processor.__new__(Processor)
    processor.sub_repo = subscriptions
    processor.ep_repo = EpisodeRepository()
    processor.source_item_repo = SourceItemRepository()
    discoveries = [
        SimpleNamespace(entries=[{"id": "keep"}, {"id": "removed"}], truncated=False),
        SimpleNamespace(entries=[{"id": "keep"}], truncated=False),
    ]
    monkeypatch.setattr("app.core.sources.discover_youtube_entries", lambda *_args: discoveries.pop(0))
    monkeypatch.setattr(
        "app.core.processor.hydrate_youtube_entry",
        lambda video_id: ({
            "guid": video_id,
            "title": video_id,
            "pub_date": None,
            "original_url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": 1,
            "description": "",
            "file_size": 0,
        }, None, False),
    )

    asyncio.run(processor.check_feeds(subscription_id=subscription.id, limit=0))
    asyncio.run(processor.check_feeds(subscription_id=subscription.id, limit=0))

    with get_db_connection() as conn:
        membership = conn.execute(
            "SELECT is_present FROM source_items WHERE subscription_id = ? AND external_id = 'removed'",
            (subscription.id,),
        ).fetchone()
        episode = conn.execute(
            "SELECT id FROM episodes WHERE subscription_id = ? AND guid = 'removed'",
            (subscription.id,),
        ).fetchone()
    assert membership["is_present"] == 0
    assert episode is not None


def test_transiently_excluded_video_is_queued_when_it_becomes_eligible(isolated_data_dir, monkeypatch):
    init_db()
    subscriptions = SubscriptionRepository()
    subscription = subscriptions.create(
        SubscriptionCreate(feed_url="https://www.youtube.com/channel/UC_TRANSIENT/videos"),
        "Transient Test",
        "transient-test",
        source_type="youtube_channel",
        source_external_id="UC_TRANSIENT",
    )
    processor = Processor.__new__(Processor)
    processor.sub_repo = subscriptions
    processor.ep_repo = EpisodeRepository()
    processor.source_item_repo = SourceItemRepository()
    monkeypatch.setattr(
        "app.core.sources.discover_youtube_entries",
        lambda *_args: SimpleNamespace(entries=[{"id": "eventual"}], truncated=False),
    )
    hydration = [
        (None, "is_upcoming", True),
        ({
            "guid": "eventual",
            "title": "Now available",
            "pub_date": None,
            "original_url": "https://www.youtube.com/watch?v=eventual",
            "duration": 60,
            "description": "",
            "file_size": 0,
        }, None, False),
    ]
    monkeypatch.setattr("app.core.processor.hydrate_youtube_entry", lambda _video_id: hydration.pop(0))

    asyncio.run(processor.check_feeds(subscription_id=subscription.id, limit=5))
    asyncio.run(processor.check_feeds(subscription_id=subscription.id, limit=5))

    with get_db_connection() as conn:
        episode = conn.execute("SELECT status FROM episodes WHERE guid = 'eventual'").fetchone()
    assert episode["status"] == "pending"


def test_one_youtube_source_failure_does_not_abort_other_sources(isolated_data_dir, monkeypatch):
    init_db()
    subscriptions = SubscriptionRepository()
    broken = subscriptions.create(
        SubscriptionCreate(feed_url="https://www.youtube.com/channel/UC_BROKEN/videos"),
        "Broken",
        "broken",
        source_type="youtube_channel",
        source_external_id="UC_BROKEN",
    )
    working = subscriptions.create(
        SubscriptionCreate(feed_url="https://www.youtube.com/channel/UC_WORKING/videos"),
        "Working",
        "working",
        source_type="youtube_channel",
        source_external_id="UC_WORKING",
    )
    processor = Processor.__new__(Processor)
    processor.sub_repo = subscriptions
    processor.ep_repo = EpisodeRepository()
    processor.source_item_repo = SourceItemRepository()

    def discover(_source_type, url):
        if "UC_BROKEN" in url:
            raise RuntimeError("extractor unavailable")
        return SimpleNamespace(entries=[{"id": "working-video"}], truncated=False)

    monkeypatch.setattr("app.core.sources.discover_youtube_entries", discover)
    monkeypatch.setattr(
        "app.core.processor.hydrate_youtube_entry",
        lambda video_id: ({
            "guid": video_id,
            "title": "Working video",
            "pub_date": None,
            "original_url": f"https://www.youtube.com/watch?v={video_id}",
            "duration": 60,
            "description": "",
            "file_size": 0,
        }, None, False),
    )

    asyncio.run(processor.check_feeds(limit=1))

    with get_db_connection() as conn:
        broken_state = conn.execute(
            "SELECT last_check_error FROM subscriptions WHERE id = ?", (broken.id,)
        ).fetchone()
        working_episode = conn.execute(
            "SELECT status FROM episodes WHERE subscription_id = ?", (working.id,)
        ).fetchone()
    assert "extractor unavailable" in broken_state["last_check_error"]
    assert working_episode["status"] == "pending"


def test_youtube_download_uses_native_extension_and_atomic_name(isolated_data_dir, monkeypatch):
    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _url, download=True):
            output = Path(self.options["outtmpl"].replace("%(ext)s", "webm"))
            output.write_bytes(b"audio")
            for hook in self.options["progress_hooks"]:
                hook({"downloaded_bytes": 5, "total_bytes": 5})
            return {"requested_downloads": [{"filepath": str(output)}]}

    monkeypatch.setattr("app.core.youtube._youtube_dl", lambda options: FakeYDL(options))
    progress = []
    episode_dir = isolated_data_dir / "podcasts" / "show" / "episode"
    output = download_youtube_audio(
        "https://www.youtube.com/watch?v=video",
        str(episode_dir),
        progress_callback=progress.append,
    )

    assert Path(output).name == "original.webm"
    assert Path(output).read_bytes() == b"audio"
    assert progress == [100]
    assert not list(episode_dir.glob("youtube-download.*"))


def test_youtube_download_cancellation_removes_partial_files(isolated_data_dir, monkeypatch):
    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _url, download=True):
            output = Path(self.options["outtmpl"].replace("%(ext)s", "webm"))
            output.write_bytes(b"partial")
            self.options["progress_hooks"][0]({"downloaded_bytes": 7, "total_bytes": 20})

    monkeypatch.setattr("app.core.youtube._youtube_dl", lambda options: FakeYDL(options))
    episode_dir = isolated_data_dir / "podcasts" / "show" / "cancelled"

    with pytest.raises(YouTubeDownloadCancelled):
        download_youtube_audio(
            "https://www.youtube.com/watch?v=video",
            str(episode_dir),
            cancellation_callback=lambda: True,
        )

    assert not list(episode_dir.glob("youtube-download.*"))


def test_source_identity_index_deduplicates_alternate_urls(isolated_data_dir):
    init_db()
    subscriptions = SubscriptionRepository()
    subscriptions.create(
        SubscriptionCreate(feed_url="https://www.youtube.com/@friendly"),
        "Friendly",
        "friendly",
        source_type="youtube_channel",
        source_external_id="UC_SAME",
    )
    with pytest.raises(ValueError, match="already exists"):
        subscriptions.create(
            SubscriptionCreate(feed_url="https://www.youtube.com/channel/UC_SAME/videos"),
            "Friendly Again",
            "friendly-again",
            source_type="youtube_channel",
            source_external_id="UC_SAME",
        )


def test_migration_adds_youtube_source_state(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        subscription_columns = {row["name"] for row in conn.execute("PRAGMA table_info(subscriptions)")}
        episode_columns = {row["name"] for row in conn.execute("PRAGMA table_info(episodes)")}
        source_item_columns = {row["name"] for row in conn.execute("PRAGMA table_info(source_items)")}
        migration = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = '20260810_0012_youtube_sources'"
        ).fetchone()

    assert {"source_type", "source_external_id", "last_check_error", "source_truncated"} <= subscription_columns
    assert {"discovered_at", "source_media_path"} <= episode_columns
    assert {"subscription_id", "external_id", "eligibility", "is_present"} <= source_item_columns
    assert migration is not None
