import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.core.processor import Processor

@pytest.fixture
def mock_processor():
    """Fixture to provide a Processor instance with mocked external dependencies."""
    with patch("app.core.processor.EpisodeRepository") as mock_ep_repo, \
            patch("app.core.processor.SubscriptionRepository") as mock_sub_repo, \
            patch("app.core.processor.JobRepository"), \
            patch("app.core.processor.Transcriber"), \
            patch("app.core.processor.AdDetector"), \
            patch("app.core.processor.RSSGenerator"):

        processor = Processor()
        yield processor

@patch("app.core.processor.FeedManager")
def test_check_feeds_creates_new_episodes_from_feed(mock_feed_manager, mock_processor):
    """Test that new episodes are queued (status: pending) from feed."""
    mock_sub = MagicMock()
    mock_sub.id = 1
    mock_sub.feed_url = "https://example.com/feed"
    mock_sub.retention_limit = 5

    mock_processor.sub_repo.get_all.return_value = [mock_sub]

    # Mock FeedManager to return one parsed episode
    mock_feed_manager.parse_episodes.return_value = [
        {'title': 'Episode 1', 'guid': 'guid-123'}
    ]

    # True means the episode did not exist and was created
    mock_processor.ep_repo.create_or_ignore.return_value = True

    asyncio.run(mock_processor.check_feeds())

    # Verify FeedManager was used instead of requests.get
    mock_feed_manager.parse_episodes.assert_called_once_with("https://example.com/feed")

    # Verify it tried to save to the database with the correct pending status
    mock_processor.ep_repo.create_or_ignore.assert_called_once()
    saved_ep_data = mock_processor.ep_repo.create_or_ignore.call_args[0][0]
    assert saved_ep_data['title'] == 'Episode 1'
    assert saved_ep_data['status'] == 'pending'


@patch("app.core.processor.FeedManager")
def test_check_feeds_skips_existing_episodes(mock_feed_manager, mock_processor):
    """Test that existing episodes are handled correctly (skipped or backfilled)."""
    mock_sub = MagicMock()
    mock_sub.id = 1
    mock_sub.retention_limit = 5
    mock_processor.sub_repo.get_all.return_value = [mock_sub]

    mock_feed_manager.parse_episodes.return_value = [
        {'title': 'Episode 1', 'guid': 'guid-123'}
    ]

    # False means the episode already exists in the database
    mock_processor.ep_repo.create_or_ignore.return_value = False

    asyncio.run(mock_processor.check_feeds())

    # Should attempt to backfill/update status since it's within the limit
    mock_processor.ep_repo.update_status_by_guid.assert_called_once_with(
        1, 'guid-123', 'pending', condition_status='unprocessed'
    )


@patch("app.core.processor.FeedManager")
def test_check_feeds_respects_retention_limit(mock_feed_manager, mock_processor):
    """Test that retention limit is respected by marking older episodes unprocessed."""
    mock_sub = MagicMock()
    mock_sub.id = 1
    mock_sub.retention_limit = 1 # Only the first episode should be pending
    mock_processor.sub_repo.get_all.return_value = [mock_sub]

    mock_feed_manager.parse_episodes.return_value = [
        {'title': 'Episode 1', 'guid': 'guid-1'},
        {'title': 'Episode 2', 'guid': 'guid-2'}
    ]

    asyncio.run(mock_processor.check_feeds())

    assert mock_processor.ep_repo.create_or_ignore.call_count == 2

    # Inspect the payloads sent to the database
    call_1_args = mock_processor.ep_repo.create_or_ignore.call_args_list[0][0][0]
    call_2_args = mock_processor.ep_repo.create_or_ignore.call_args_list[1][0][0]

    assert call_1_args['status'] == 'pending'
    assert call_2_args['status'] == 'unprocessed'


@patch("app.core.processor.FeedManager")
def test_check_feeds_handles_feed_parsing_error(mock_feed_manager, mock_processor):
    """Test that feed parsing errors are handled gracefully without crashing the loop."""
    mock_sub = MagicMock()
    mock_processor.sub_repo.get_all.return_value = [mock_sub]

    # Simulate a parsing exception
    mock_feed_manager.parse_episodes.side_effect = Exception("Invalid XML")

    # This should execute and catch the exception internally, preventing a crash
    asyncio.run(mock_processor.check_feeds())

    mock_processor.ep_repo.create_or_ignore.assert_not_called()


@patch("app.core.processor.FeedManager")
def test_check_feeds_all_subscriptions(mock_feed_manager, mock_processor):
    """Test processing loops over all active subscriptions."""
    mock_sub1 = MagicMock(id=1, feed_url="https://feed1.com")
    mock_sub2 = MagicMock(id=2, feed_url="https://feed2.com")

    mock_processor.sub_repo.get_all.return_value = [mock_sub1, mock_sub2]
    mock_feed_manager.parse_episodes.return_value = []

    asyncio.run(mock_processor.check_feeds())

    assert mock_feed_manager.parse_episodes.call_count == 2


@patch("app.core.processor.FeedManager")
def test_check_feeds_with_zero_limit_skips_initial_downloads(mock_feed_manager, mock_processor):
    """Test that a retention limit of zero sets status to unprocessed (skips downloads)."""
    mock_sub = MagicMock()
    mock_sub.id = 1
    mock_sub.retention_limit = 0 # 0 means skip initial downloads
    mock_processor.sub_repo.get_all.return_value = [mock_sub]

    mock_feed_manager.parse_episodes.return_value = [
        {'title': 'Episode 1', 'guid': 'guid-1'}
    ]

    asyncio.run(mock_processor.check_feeds())

    call_args = mock_processor.ep_repo.create_or_ignore.call_args[0][0]
    assert call_args['status'] == 'unprocessed'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])