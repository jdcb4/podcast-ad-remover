import pytest
from unittest.mock import patch
from app.core.processor import Processor


def test_prepare_remove_segments_sorts_merges_close_gaps_and_does_not_shrink_contained_segments():
    remove_segments = Processor._prepare_remove_segments(
        [
            {"start": 30, "end": 60, "label": "Ad"},
            {"start": 5, "end": 10, "label": "Ad"},
            {"start": 15, "end": 20, "label": "Promo"},
            {"start": 35, "end": 40, "label": "Contained"},
            {"start": "bad", "end": 70},
            {"start": 80, "end": 80},
        ],
        whitelist_mode=False,
    )

    assert remove_segments == [
        {"start": 5.0, "end": 20.0, "label": "Ad"},
        {"start": 30.0, "end": 60.0, "label": "Ad"},
    ]


def test_prepare_remove_segments_whitelist_inverts_content_windows():
    remove_segments = Processor._prepare_remove_segments(
        [
            {"start": 50, "end": 70, "label": "Content"},
            {"start": 10, "end": 30, "label": "Content"},
            {"start": 110, "end": 120, "label": "Content"},
            {"start": 75, "end": 80, "label": "Ad"},
        ],
        whitelist_mode=True,
        total_duration=100.0,
    )

    assert remove_segments == [
        {
            "start": 0.0,
            "end": 10.0,
            "label": "Non-Content",
            "reason": "Not labeled as content (whitelist mode)",
        },
        {
            "start": 30.0,
            "end": 50.0,
            "label": "Non-Content",
            "reason": "Not labeled as content (whitelist mode)",
        },
        {
            "start": 70.0,
            "end": 100.0,
            "label": "Non-Content",
            "reason": "Trailing non-content (whitelist mode)",
        },
    ]


def test_prepare_remove_segments_whitelist_overlapping_content_does_not_create_negative_remove_windows():
    remove_segments = Processor._prepare_remove_segments(
        [
            {"start": 10, "end": 30, "label": "Content"},
            {"start": 20, "end": 40, "label": "Content"},
            {"start": 40, "end": 50, "label": "Content"},
        ],
        whitelist_mode=True,
        total_duration=60.0,
    )

    assert remove_segments == [
        {
            "start": 0.0,
            "end": 10.0,
            "label": "Non-Content",
            "reason": "Not labeled as content (whitelist mode)",
        },
        {
            "start": 50.0,
            "end": 60.0,
            "label": "Non-Content",
            "reason": "Trailing non-content (whitelist mode)",
        },
    ]


def test_prepare_remove_segments_whitelist_without_content_falls_back_to_non_content_rows():
    remove_segments = Processor._prepare_remove_segments(
        [
            {"start": 3, "end": 6, "label": "Ad"},
            {"start": 8, "end": 9, "label": "Promo"},
        ],
        whitelist_mode=True,
        total_duration=20.0,
    )

    assert remove_segments == [
        {"start": 3.0, "end": 9.0, "label": "Ad"},
    ]


def test_prepare_remove_segments_whitelist_without_duration_keeps_episode_uncut():
    assert (
        Processor._prepare_remove_segments(
            [{"start": 3, "end": 6, "label": "Content"}],
            whitelist_mode=True,
            total_duration=0.0,
        )
        == []
    )

def test_prepare_remove_segments_full_content_coverage(monkeypatch):
    """Test that if all segments are marked as Content, no remove windows are generated in whitelist mode."""
    # Mock the internal _normalize_segment to ensure float conversion works for testing
    with patch.object(Processor, '_normalize_segment', side_effect=lambda segment, total_duration: segment):
        remove_segments = Processor._prepare_remove_segments(
            [
                {"start": 0, "end": 10, "label": "Content"},
                {"start": 10, "end": 20, "label": "Content"},
                {"start": 20, "end": 30, "label": "Content"},
            ],
            whitelist_mode=True,
            total_duration=30.0,
        )
        assert remove_segments == []

def test_prepare_remove_segments_blacklist_merges_gaps_and_handles_non_contiguous_ads():
    """Test merging of segments that are close but not perfectly contiguous."""
    # Gap is 10.0 seconds, so 5s gap should merge (30-35=5 < 10)
    remove_segments = Processor._prepare_remove_segments(
        [
            {"start": 10, "end": 20, "label": "Ad"}, # Start
            {"start": 25, "end": 30, "label": "Promo"}, # Gap of 5s (merges)
            {"start": 35, "end": 45, "label": "Outro"}, # End
        ],
        whitelist_mode=False,
    )

    assert remove_segments == [
        {"start": 10.0, "end": 45.0, "label": "Ad"}, # Merged: 10 to 20 + gap + 25 to 45 = 10 to 45
    ]

def test_prepare_remove_segments_whitelist_with_gaps_and_boundaries(monkeypatch):
    """Test whitelist mode with gaps at start, end, and middle."""
    # Total duration is 100s. Content is [30-70]. Gaps are [0-30] and [70-100].
    with patch.object(Processor, '_normalize_segment', side_effect=lambda segment, total_duration=None: segment):
        remove_segments = Processor._prepare_remove_segments(
            [
                {"start": 30, "end": 70, "label": "Content"},
            ],
            whitelist_mode=True,
            total_duration=100.0,
        )

    assert remove_segments == [
        {
            "start": 0.0,
            "end": 30.0,
            "label": "Non-Content",
            "reason": "Not labeled as content (whitelist mode)",
        },
        {
            "start": 70.0,
            "end": 100.0,
            "label": "Non-Content",
            "reason": "Trailing non-content (whitelist mode)",
        }
    ]