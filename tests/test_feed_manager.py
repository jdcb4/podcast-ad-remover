import pytest

from app.core.config import settings
from app.core.feed import FeedManager


class FakeStreamResponse:
    def __init__(self, *, headers=None, chunks=None, url="https://example.com/feed.xml"):
        self.headers = headers or {}
        self._chunks = chunks or []
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield from self._chunks


class FakeHttpClient:
    response = FakeStreamResponse()

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url):
        return self.response


def test_fetch_feed_rejects_oversized_content_length(monkeypatch):
    monkeypatch.setattr(settings, "MAX_FEED_BYTES", 10)
    FakeHttpClient.response = FakeStreamResponse(
        headers={"Content-Length": "11"},
        chunks=[b""],
    )
    monkeypatch.setattr("app.core.feed.httpx.Client", FakeHttpClient)

    with pytest.raises(ValueError, match="larger than the configured maximum"):
        FeedManager._fetch_feed("https://example.com/feed.xml")


def test_fetch_feed_rejects_stream_that_exceeds_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_FEED_BYTES", 10)
    FakeHttpClient.response = FakeStreamResponse(
        chunks=[b"12345", b"67890", b"x"],
    )
    monkeypatch.setattr("app.core.feed.httpx.Client", FakeHttpClient)

    with pytest.raises(ValueError, match="larger than the configured maximum"):
        FeedManager._fetch_feed("https://example.com/feed.xml")


def test_parse_feed_and_episodes_from_sample_rss(monkeypatch):
    sample_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Example Podcast!</title>
    <description>Useful show notes.</description>
    <itunes:image href="https://example.com/art.jpg" />
    <item>
      <guid>episode-one</guid>
      <title>Episode One</title>
      <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
      <description>Episode notes.</description>
      <itunes:duration>01:02:03</itunes:duration>
      <enclosure url="https://cdn.example.com/episode-one.mp3" type="audio/mpeg" length="12345" />
    </item>
    <item>
      <guid>non-audio</guid>
      <title>Skipped</title>
      <enclosure url="https://cdn.example.com/file.txt" type="text/plain" length="9" />
    </item>
  </channel>
</rss>
"""
    monkeypatch.setattr(FeedManager, "_fetch_feed", staticmethod(lambda url: sample_rss))

    title, slug, image_url, description = FeedManager.parse_feed("https://example.com/feed.xml")
    episodes = FeedManager.parse_episodes("https://example.com/feed.xml")

    assert title == "Example Podcast!"
    assert slug == "example-podcast"
    assert image_url == "https://example.com/art.jpg"
    assert description == "Useful show notes."
    assert episodes == [
        {
            "guid": "episode-one",
            "title": "Episode One",
            "pub_date": episodes[0]["pub_date"],
            "original_url": "https://cdn.example.com/episode-one.mp3",
            "duration": 3723,
            "description": "Episode notes.",
            "file_size": 12345,
        }
    ]
    assert episodes[0]["pub_date"].year == 2024
