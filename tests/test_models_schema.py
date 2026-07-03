from datetime import datetime

from app.core.models import Episode, Subscription


def test_episode_model_keeps_retry_manual_and_listen_fields():
    next_retry = datetime(2026, 1, 2, 3, 4, 5)

    episode = Episode.model_validate({
        "id": 1,
        "subscription_id": 2,
        "guid": "episode-guid",
        "title": "Episode",
        "pub_date": None,
        "original_url": "https://example.com/audio.mp3",
        "duration": 123,
        "status": "failed",
        "processed_at": None,
        "error_message": "temporary failure",
        "processing_step": "retry scheduled",
        "progress": 0,
        "transcript_path": None,
        "ai_summary": None,
        "ad_report_path": None,
        "processing_flags": None,
        "description": "Description",
        "report_path": None,
        "file_size": 456,
        "local_filename": None,
        "retry_count": 2,
        "next_retry_at": next_retry,
        "is_manual_download": True,
        "listen_count": 7,
    })

    assert episode.retry_count == 2
    assert episode.next_retry_at == next_retry
    assert episode.is_manual_download is True
    assert episode.listen_count == 7


def test_subscription_model_includes_download_order():
    """Test that Subscription model includes download_order field with default."""
    subscription = Subscription.model_validate({
        "id": 1,
        "feed_url": "https://example.com/feed.xml",
        "title": "Test Podcast",
        "is_active": True,
        "created_at": datetime.now(),
    })

    # Default should be 'newest'
    assert subscription.download_order == "newest"

    # Test explicit value
    subscription_oldest = Subscription.model_validate({
        "id": 2,
        "feed_url": "https://example.com/feed2.xml",
        "title": "Test Podcast 2",
        "is_active": True,
        "created_at": datetime.now(),
        "download_order": "oldest",
    })

    assert subscription_oldest.download_order == "oldest"
