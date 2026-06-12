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
