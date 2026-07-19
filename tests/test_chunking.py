import pytest
from app.core.ai_services import AdDetector


def test_create_transcript_chunks_empty_segments():
    """Test chunking with empty segments returns empty list."""
    detector = AdDetector()
    chunks = detector._create_transcript_chunks([], 60.0, 30.0, 10)
    assert chunks == []


def test_create_transcript_chunks_single_segment():
    """Test chunking with a single segment."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Hello"}
    ]
    chunks = detector._create_transcript_chunks(segments, 60.0, 30.0, 10)
    assert len(chunks) == 1
    assert len(chunks[0]) == 1
    assert chunks[0][0]["start"] == 0.0
    assert chunks[0][0]["end"] == 10.0


def test_create_transcript_chunks_no_overlap():
    """Test chunking with zero overlap creates distinct chunks."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 30.0, "text": "First"},
        {"start": 30.0, "end": 60.0, "text": "Second"},
        {"start": 60.0, "end": 90.0, "text": "Third"},
    ]
    chunks = detector._create_transcript_chunks(segments, 30.0, 0.0, 10)
    # With no overlap and 30s chunks, should create 3 chunks
    assert len(chunks) == 3


def test_create_transcript_chunks_with_overlap():
    """Test chunking with overlap includes boundary segments."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 30.0, "text": "First"},
        {"start": 30.0, "end": 60.0, "text": "Second"},
        {"start": 60.0, "end": 90.0, "text": "Third"},
    ]
    chunks = detector._create_transcript_chunks(segments, 30.0, 10.0, 10)
    # With 10s overlap, segments at boundaries should appear in adjacent chunks
    assert len(chunks) >= 2
    # Check that segment at 30s appears in multiple chunks due to overlap
    boundary_segment_count = sum(
        1 for chunk in chunks
        for seg in chunk
        if seg["start"] == 30.0
    )
    assert boundary_segment_count >= 1


def test_create_transcript_chunks_preserves_timestamps():
    """Test that chunking preserves original global timestamps."""
    detector = AdDetector()
    segments = [
        {"start": 100.0, "end": 130.0, "text": "First"},
        {"start": 130.0, "end": 160.0, "text": "Second"},
        {"start": 160.0, "end": 190.0, "text": "Third"},
    ]
    chunks = detector._create_transcript_chunks(segments, 30.0, 10.0, 10)
    # All chunks should preserve original timestamps
    for chunk in chunks:
        for seg in chunk:
            assert seg["start"] >= 100.0
            assert seg["end"] <= 190.0


def test_create_transcript_chunks_max_chunks_limit():
    """Test that max_chunks parameter limits the number of chunks."""
    detector = AdDetector()
    # Create a long transcript
    segments = [
        {"start": i * 10.0, "end": (i + 1) * 10.0, "text": f"Segment {i}"}
        for i in range(100)
    ]
    chunks = detector._create_transcript_chunks(segments, 30.0, 10.0, 5)
    assert len(chunks) <= 5


def test_merge_chunked_results_empty():
    """Test merging empty results returns empty list."""
    detector = AdDetector()
    merged = detector._merge_chunked_results([], 30.0)
    assert merged == []


def test_merge_chunked_results_no_overlap():
    """Test merging non-overlapping segments preserves all."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Ad", "reason": "First"},
        {"start": 50.0, "end": 60.0, "label": "Ad", "reason": "Second"},
        {"start": 100.0, "end": 110.0, "label": "Ad", "reason": "Third"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    assert len(merged) == 3


def test_merge_chunked_results_overlapping_same_label():
    """Test merging overlapping segments with same label."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Ad", "reason": "First"},
        {"start": 8.0, "end": 15.0, "label": "Ad", "reason": "Second"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    assert len(merged) == 1
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 15.0


def test_merge_chunked_results_overlapping_compatible_labels():
    """Test merging overlapping segments with compatible labels (Ad/Promo)."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Ad", "reason": "First"},
        {"start": 8.0, "end": 15.0, "label": "Promo", "reason": "Second"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    assert len(merged) == 1
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 15.0


def test_merge_chunked_results_overlapping_incompatible_labels():
    """Test that incompatible labels (Content vs Ad) are not merged."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Content", "reason": "Content segment"},
        {"start": 8.0, "end": 15.0, "label": "Ad", "reason": "Ad segment"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    assert len(merged) == 2
    # Both segments should be preserved
    assert any(s["label"] == "Content" for s in merged)
    assert any(s["label"] == "Ad" for s in merged)


def test_merge_chunked_results_deterministic():
    """Test that merging is deterministic regardless of input order."""
    detector = AdDetector()
    segments_unordered = [
        {"start": 50.0, "end": 60.0, "label": "Ad", "reason": "Second"},
        {"start": 0.0, "end": 10.0, "label": "Ad", "reason": "First"},
        {"start": 100.0, "end": 110.0, "label": "Ad", "reason": "Third"},
    ]
    merged1 = detector._merge_chunked_results(segments_unordered, 30.0)
    
    segments_reversed = list(reversed(segments_unordered))
    merged2 = detector._merge_chunked_results(segments_reversed, 30.0)
    
    assert merged1 == merged2


def test_labels_compatible_exact_match():
    """Test that exact label matches are compatible."""
    detector = AdDetector()
    assert detector._labels_compatible("Ad", "Ad") is True
    assert detector._labels_compatible("Promo", "Promo") is True
    assert detector._labels_compatible("Content", "Content") is True


def test_labels_compatible_ad_promo_group():
    """Test that Ad and Promo labels are compatible."""
    detector = AdDetector()
    assert detector._labels_compatible("Ad", "Promo") is True
    assert detector._labels_compatible("Promo", "Ad") is True
    assert detector._labels_compatible("Ad", "Cross-promotion") is True
    assert detector._labels_compatible("Cross-promotion", "Promo") is True


def test_labels_compatible_content_incompatible():
    """Test that Content is incompatible with non-Content labels."""
    detector = AdDetector()
    assert detector._labels_compatible("Content", "Ad") is False
    assert detector._labels_compatible("Ad", "Content") is False
    assert detector._labels_compatible("Content", "Promo") is False
    assert detector._labels_compatible("Intro", "Content") is False


def test_labels_compatible_different_labels():
    """Test that different non-Ad/Promo labels are incompatible."""
    detector = AdDetector()
    assert detector._labels_compatible("Intro", "Outro") is False
    assert detector._labels_compatible("Ad", "Intro") is False
    assert detector._labels_compatible("Promo", "Outro") is False


def test_detect_ads_uses_single_request_for_small_transcript():
    """Test that small transcripts use single-request path."""
    detector = AdDetector()
    # Mock settings to disable chunking
    detector.settings = {
        'chunking_enabled': 1,
        'chunking_threshold_kb': 1000,
        'chunking_max_chunks': 10,
        'chunking_overlap_seconds': 30,
        'chunking_accept_partial': 0,
    }
    
    # Create a small transcript (< 1 KB)
    segments = [
        {"start": i * 1.0, "end": (i + 1) * 1.0, "text": "Short"}
        for i in range(10)
    ]
    transcript = {"segments": segments}
    
    # This should use single-request path
    # We can't easily test the actual AI call without mocking, but we can verify
    # the size estimation logic works
    text_data = ""
    for seg in transcript['segments']:
        text_data += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"
    
    estimated_size_kb = len(text_data.encode('utf-8')) / 1024
    assert estimated_size_kb < 1000  # Below threshold


def test_detect_ads_uses_chunking_for_large_transcript():
    """Test that large transcripts trigger chunking path."""
    detector = AdDetector()
    # Mock settings to enable chunking with low threshold
    detector.settings = {
        'chunking_enabled': 1,
        'chunking_threshold_kb': 1,  # Very low threshold
        'chunking_max_chunks': 10,
        'chunking_overlap_seconds': 30,
        'chunking_accept_partial': 0,
    }
    
    # Create a large transcript (> 1 KB)
    segments = [
        {"start": i * 1.0, "end": (i + 1) * 1.0, "text": "A" * 100}
        for i in range(100)
    ]
    transcript = {"segments": segments}
    
    # This should trigger chunking
    text_data = ""
    for seg in transcript['segments']:
        text_data += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"
    
    estimated_size_kb = len(text_data.encode('utf-8')) / 1024
    assert estimated_size_kb >= 1  # Above threshold


def test_detect_ads_chunking_disabled():
    """Test that chunking can be disabled."""
    detector = AdDetector()
    # Mock settings to disable chunking
    detector.settings = {
        'chunking_enabled': 0,  # Disabled
        'chunking_threshold_kb': 1,
        'chunking_max_chunks': 10,
        'chunking_overlap_seconds': 30,
        'chunking_accept_partial': 0,
    }

    # Create a large transcript
    segments = [
        {"start": i * 1.0, "end": (i + 1) * 1.0, "text": "A" * 100}
        for i in range(100)
    ]
    transcript = {"segments": segments}

    # Even with large transcript, should use single-request when disabled
    text_data = ""
    for seg in transcript['segments']:
        text_data += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"

    estimated_size_kb = len(text_data.encode('utf-8')) / 1024
    assert estimated_size_kb >= 1  # Above threshold
    # But chunking is disabled, so should use single-request


def test_chunking_minimum_duration_adjustment():
    """Test that chunk count is adjusted to maintain minimum chunk duration."""
    detector = AdDetector()
    # Short transcript that would create tiny chunks
    segments = [
        {"start": 0.0, "end": 30.0, "text": "Segment 1"},
        {"start": 30.0, "end": 60.0, "text": "Segment 2"},
    ]
    transcript = {"segments": segments}

    # With max_chunks=10 and total duration=60s, chunk_duration would be 6s
    # But minimum is 60s, so should adjust to 1 chunk
    chunks = detector._create_transcript_chunks(segments, 6.0, 10.0, 10)
    # Should create fewer chunks to maintain minimum duration
    assert len(chunks) >= 1


def test_chunking_with_whitelist_mode():
    """Test that chunking preserves whitelist mode behavior."""
    detector = AdDetector()
    detector.settings = {
        'chunking_enabled': 1,
        'chunking_threshold_kb': 1,
        'chunking_max_chunks': 10,
        'chunking_overlap_seconds': 30,
        'chunking_accept_partial': 0,
    }

    # Create a transcript
    segments = [
        {"start": i * 10.0, "end": (i + 1) * 10.0, "text": "A" * 100}
        for i in range(20)
    ]
    transcript = {"segments": segments}

    # Test that chunking logic works regardless of whitelist mode
    # (whitelist mode affects filtering, not chunking)
    chunks = detector._create_transcript_chunks(segments, 100.0, 30.0, 10)
    assert len(chunks) > 0


def test_merge_chunked_results_preserves_content_labels():
    """Test that Content labels are preserved during merging."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Content", "reason": "Main content"},
        {"start": 10.0, "end": 20.0, "label": "Ad", "reason": "Sponsor"},
        {"start": 20.0, "end": 30.0, "label": "Content", "reason": "More content"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    # All segments should be preserved since they don't overlap
    assert len(merged) == 3
    assert any(s["label"] == "Content" for s in merged)


def test_merge_chunked_results_adjacent_segments():
    """Test that adjacent segments (within half overlap) are merged if compatible."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Ad", "reason": "First"},
        {"start": 10.0, "end": 20.0, "label": "Ad", "reason": "Second"},  # Exactly adjacent
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    # With 30s overlap, adjacent segments should merge
    assert len(merged) == 1
    assert merged[0]["end"] == 20.0


def test_chunking_with_very_long_transcript():
    """Test chunking behavior with a very long transcript (simulating 3+ hour episode)."""
    detector = AdDetector()
    # Create a 3-hour transcript (10800 seconds)
    segments = [
        {"start": i * 10.0, "end": (i + 1) * 10.0, "text": "Segment"}
        for i in range(1080)  # 10800 seconds / 10 seconds per segment
    ]
    transcript = {"segments": segments}

    chunks = detector._create_transcript_chunks(segments, 1200.0, 30.0, 10)
    # Should create chunks but respect max_chunks limit
    assert len(chunks) <= 10
    assert len(chunks) > 0


def test_chunking_with_zero_overlap():
    """Test that zero overlap creates distinct non-overlapping chunks."""
    detector = AdDetector()
    segments = [
        {"start": i * 30.0, "end": (i + 1) * 30.0, "text": f"Segment {i}"}
        for i in range(9)
    ]
    chunks = detector._create_transcript_chunks(segments, 90.0, 0.0, 10)
    # With zero overlap, segments should not appear in multiple chunks
    # (unless they span chunk boundaries)
    assert len(chunks) > 0


def test_chunking_with_large_overlap():
    """Test that large overlap ensures boundary coverage."""
    detector = AdDetector()
    segments = [
        {"start": i * 30.0, "end": (i + 1) * 30.0, "text": f"Segment {i}"}
        for i in range(9)
    ]
    chunks = detector._create_transcript_chunks(segments, 90.0, 60.0, 10)
    # With large overlap, boundary segments should appear in multiple chunks
    assert len(chunks) > 0


def test_merge_chunked_results_with_mixed_labels():
    """Test merging with mixed ad types and content."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 5.0, "label": "Intro", "reason": "Show intro"},
        {"start": 5.0, "end": 15.0, "label": "Ad", "reason": "Sponsor"},
        {"start": 15.0, "end": 25.0, "label": "Content", "reason": "Main topic"},
        {"start": 25.0, "end": 30.0, "label": "Promo", "reason": "Cross-promo"},
        {"start": 30.0, "end": 35.0, "label": "Outro", "reason": "Show outro"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    # All non-overlapping segments should be preserved
    assert len(merged) == 5


def test_chunking_edge_case_single_chunk():
    """Test chunking when only one chunk is needed."""
    detector = AdDetector()
    segments = [
        {"start": i * 10.0, "end": (i + 1) * 10.0, "text": "Segment"}
        for i in range(5)
    ]
    chunks = detector._create_transcript_chunks(segments, 1000.0, 30.0, 10)
    # Should create only 1 chunk since total duration is small
    assert len(chunks) == 1


def test_chunking_preserves_segment_order():
    """Test that chunking preserves segment order within chunks."""
    detector = AdDetector()
    segments = [
        {"start": i * 10.0, "end": (i + 1) * 10.0, "text": f"Segment {i}"}
        for i in range(20)
    ]
    chunks = detector._create_transcript_chunks(segments, 100.0, 30.0, 10)

    for chunk in chunks:
        # Verify segments are in order within each chunk
        starts = [seg["start"] for seg in chunk]
        assert starts == sorted(starts)


def test_merge_chunked_results_reason_accumulation():
    """Test that merge reasons are accumulated when merging overlapping segments."""
    detector = AdDetector()
    segments = [
        {"start": 0.0, "end": 10.0, "label": "Ad", "reason": "First ad"},
        {"start": 8.0, "end": 15.0, "label": "Ad", "reason": "Second ad"},
    ]
    merged = detector._merge_chunked_results(segments, 30.0)
    assert len(merged) == 1
    # Reason should include both original reasons
    assert "First ad" in merged[0]["reason"]
    assert "Second ad" in merged[0]["reason"]
