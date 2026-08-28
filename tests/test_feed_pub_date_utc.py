from email.utils import format_datetime

from app.core.feed import FeedManager
from tests.conftest import DST_TEST_INSTANT

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title>
<item>
  <title>Ep</title>
  <pubDate>{pub_date}</pubDate>
  <enclosure url="https://cdn.example.com/e.mp3" type="audio/mpeg" length="1"/>
  <guid>g-1</guid>
</item>
</channel></rss>""".format(pub_date=format_datetime(DST_TEST_INSTANT, usegmt=True))

EXPECTED_PUB_DATE = DST_TEST_INSTANT.replace(tzinfo=None)


def test_parse_episodes_pub_date_is_utc_under_dst(monkeypatch, dst_timezone):
    """Regression guard.

    feedparser sets tm_isdst=0 on published_parsed, so the old code's
    mktime() encoded it with the zone's standard offset while
    fromtimestamp() decoded it with the zone's actual (DST) offset for the
    date, skewing pub_date by the DST delta. America/New_York in August is
    in DST, so this genuinely fails against the pre-fix code (06:05
    instead of 05:05) and passes against the fix.

    The RSS pubDate above and this expectation both derive from
    tests.conftest.DST_TEST_INSTANT — the same instant that
    assert_dst_active() validates is inside DST — so they cannot drift
    apart the way three independently-typed literals could.
    """
    monkeypatch.setattr(FeedManager, "_fetch_feed", staticmethod(lambda url: RSS))

    episodes = FeedManager.parse_episodes("https://example.com/feed.xml")

    assert episodes[0]["pub_date"] == EXPECTED_PUB_DATE


def test_parse_episodes_pub_date_is_utc_under_large_fixed_offset(monkeypatch, non_utc_timezone):
    """Branch coverage, not a regression guard.

    Pacific/Kiritimati has no DST rule (time.daylight == 0), so mktime()
    and fromtimestamp() are exact inverses there and this assertion holds
    even against the pre-fix code — it cannot detect the bug on its own.
    Kept alongside the DST test above to confirm the fix also behaves
    correctly under a large fixed UTC offset.
    """
    monkeypatch.setattr(FeedManager, "_fetch_feed", staticmethod(lambda url: RSS))

    episodes = FeedManager.parse_episodes("https://example.com/feed.xml")

    assert episodes[0]["pub_date"] == EXPECTED_PUB_DATE
