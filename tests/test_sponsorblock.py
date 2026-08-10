import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from app.core.config import settings
from app.core.processor import Processor
from app.core.sponsorblock import SponsorBlockClient, categories_for_subscription


def test_categories_map_to_existing_content_controls():
    subscription = SimpleNamespace(
        remove_ads=True,
        remove_promos=True,
        remove_intros=False,
        remove_outros=True,
    )
    assert categories_for_subscription(subscription) == [
        "sponsor",
        "selfpromo",
        "interaction",
        "outro",
    ]


def test_sponsorblock_filters_hash_prefix_response(monkeypatch):
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {
            "videoID": "target",
            "segments": [
                {
                    "segment": [10, 20],
                    "UUID": "uuid-1",
                    "category": "sponsor",
                    "actionType": "skip",
                    "votes": 12,
                    "locked": 1,
                },
                {"segment": [30, 40], "category": "preview", "actionType": "skip"},
                {"segment": [50, 60], "category": "sponsor", "actionType": "mute"},
            ],
        },
        {"videoID": "collision", "segments": [{"segment": [1, 2], "category": "sponsor"}]},
    ]
    request = Mock(return_value=response)
    monkeypatch.setattr("app.core.sponsorblock.httpx.get", request)

    segments = SponsorBlockClient().fetch_segments("target", ["sponsor"])

    assert len(segments) == 1
    assert segments[0]["start"] == 10.0
    assert segments[0]["evidence"][0] == {
        "source": "sponsorblock",
        "category": "sponsor",
        "uuid": "uuid-1",
        "votes": 12,
        "locked": 1,
        "action_type": "skip",
    }
    assert "/api/skipSegments/" in request.call_args.args[0]


def test_sponsorblock_failure_is_fail_open(monkeypatch):
    monkeypatch.setattr(
        "app.core.sponsorblock.httpx.get",
        Mock(side_effect=TimeoutError("offline")),
    )
    assert SponsorBlockClient().fetch_segments("target", ["sponsor"]) == []


def test_environment_gate_prevents_sponsorblock_client_call(monkeypatch):
    processor = Processor.__new__(Processor)
    processor.sponsorblock = Mock()
    subscription = SimpleNamespace(
        source_type="youtube_channel",
        remove_ads=True,
        remove_promos=False,
        remove_intros=False,
        remove_outros=False,
    )
    episode = SimpleNamespace(original_url="https://www.youtube.com/watch?v=target")
    monkeypatch.setattr(settings, "SPONSORBLOCK_ENABLED", False)

    assert asyncio.run(processor._fetch_sponsorblock_segments(subscription, episode)) == []
    processor.sponsorblock.fetch_segments.assert_not_called()


def test_merged_segments_preserve_llm_and_sponsorblock_evidence():
    merged = Processor._merge_remove_segments([
        {
            "start": 10,
            "end": 20,
            "label": "Ad",
            "evidence": [{"source": "llm", "reason": "ad read"}],
        },
        {
            "start": 18,
            "end": 25,
            "label": "SponsorBlock: sponsor",
            "evidence": [{"source": "sponsorblock", "category": "sponsor", "uuid": "u"}],
        },
    ])

    assert len(merged) == 1
    assert merged[0]["end"] == 25.0
    assert merged[0]["sources"] == ["llm", "sponsorblock"]
    assert len(merged[0]["evidence"]) == 2
