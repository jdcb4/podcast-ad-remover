import asyncio
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.processor import Processor
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository, SubscriptionRepository
from app.main import app


class RecordingRSSGenerator:
    def __init__(self):
        self.subscription_calls = []
        self.unified_calls = 0

    def generate_feed(self, subscription_id):
        self.subscription_calls.append(subscription_id)

    def generate_unified_feed(self):
        self.unified_calls += 1


def _create_subscription_with_pending_episodes(count=2):
    with get_db_connection() as conn:
        subscription_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("https://example.com/feed.xml", "Example", "example"),
        ).lastrowid
        conn.commit()

    episode_repo = EpisodeRepository()
    episode_ids = []
    for index in range(count):
        episode_repo.create_or_ignore(
            {
                "subscription_id": subscription_id,
                "guid": f"episode-{index}",
                "title": f"Episode {index}",
                "pub_date": None,
                "original_url": f"https://example.com/episode-{index}.mp3",
                "duration": 100,
                "description": "",
                "status": "pending",
                "file_size": 0,
            }
        )
        episode_ids.append(episode_repo.get_by_subscription(subscription_id)[-1].id)

    subscription_dir = Path(settings.PODCASTS_DIR) / "example"
    subscription_dir.mkdir(parents=True)
    (subscription_dir / "worker-owned.wav").write_bytes(b"audio")
    feed_path = Path(settings.FEEDS_DIR) / "example.xml"
    feed_path.write_text("feed", encoding="utf-8")
    return subscription_id, episode_ids, subscription_dir, feed_path


def test_single_episode_worker_cleans_before_acknowledging_cancellation(
    isolated_data_dir, monkeypatch
):
    init_db()
    subscription_id, _episode_ids, subscription_dir, _feed_path = _create_subscription_with_pending_episodes(1)
    job_repo = JobRepository()
    episode_repo = EpisodeRepository()
    claim = job_repo.claim_due(1, worker_id="single-episode-worker")[0]
    episode_id = claim['id']
    episode = episode_repo.get_by_id(episode_id)
    episode_dir = subscription_dir / f"episode-{episode_id}" / "attempt-test"
    episode_dir.mkdir(parents=True)
    (episode_dir / "original.mp3.clean.wav").write_bytes(b"worker-owned")
    published = episode_dir.parent / 'published.mp3'
    published.write_bytes(b'previous publication')
    processor = Processor()
    processor.ep_repo = EpisodeRepository(attempt=(claim['job_id'], claim['claim_token']))
    processor._attempt_dir = episode_dir
    assert episode_repo.request_deletion(episode_id) is True
    assert processor._check_cancellation(episode) is False
    assert not episode_dir.exists()
    assert published.exists()  # The deletion coordinator removes publication separately.
    assert not job_repo.is_running_for_episode(episode_id)


@pytest.mark.asyncio
async def test_subscription_deletion_cancels_before_cleanup_without_blocking_event_loop(
    isolated_data_dir, monkeypatch
):
    init_db()
    subscription_id, _episode_ids, subscription_dir, feed_path = _create_subscription_with_pending_episodes()
    job_repo = JobRepository()
    episode_repo = EpisodeRepository()
    monkeypatch.setattr(settings, "PROCESSOR_ENABLED", False)
    claimed = job_repo.claim_due(1, worker_id="slow-test-worker")
    assert len(claimed) == 1
    running_episode_id = claimed[0]["id"]

    processor = Processor()
    rss = RecordingRSSGenerator()
    processor.rss_gen = rss
    order = []
    original_remove = processor._remove_subscription_directory

    def slow_remove(subscription_slug):
        with get_db_connection() as conn:
            statuses = {
                row["status"]
                for row in conn.execute(
                    """
                    SELECT j.status
                    FROM jobs j
                    JOIN episodes e ON e.id = j.episode_id
                    WHERE e.subscription_id = ?
                    """,
                    (subscription_id,),
                ).fetchall()
            }
        assert statuses == {"cancelled"}
        assert SubscriptionRepository().count_running_deletion_jobs(subscription_id) == 0
        order.append("filesystem_cleanup")
        time.sleep(0.15)
        return original_remove(subscription_slug)

    monkeypatch.setattr(processor, "_remove_subscription_directory", slow_remove)

    async def slow_worker():
        while episode_repo.get_status(running_episode_id) == "processing":
            await asyncio.sleep(0.005)
        assert subscription_dir.exists()
        assert job_repo.claim_due(10, worker_id="late-worker") == []
        order.append("worker_acknowledged")
        job_repo.acknowledge(claimed[0]["job_id"], claimed[0]["claim_token"])

    worker_task = asyncio.create_task(slow_worker())
    delete_task = asyncio.create_task(processor.delete_subscription(subscription_id))
    heartbeat_count = 0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        while not delete_task.done():
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}
            heartbeat_count += 1
            await asyncio.sleep(0.01)

    assert await delete_task == "deleted"
    await worker_task

    assert heartbeat_count >= 5
    assert order == ["worker_acknowledged", "filesystem_cleanup"]
    assert not subscription_dir.exists()
    assert not feed_path.exists()
    assert SubscriptionRepository().get_by_id(subscription_id) is None
    assert job_repo.claim_due(10, worker_id="post-delete-worker") == []
    with get_db_connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE subscription_id = ?", (subscription_id,)
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert rss.subscription_calls == []
    assert rss.unified_calls == 1


@pytest.mark.asyncio
async def test_subscription_deletion_timeout_is_bounded_and_processor_retries(
    isolated_data_dir
):
    init_db()
    subscription_id, _episode_ids, subscription_dir, _feed_path = _create_subscription_with_pending_episodes(1)
    job_repo = JobRepository()
    claimed = job_repo.claim_due(1, worker_id="stuck-test-worker")
    running_episode_id = claimed[0]["id"]

    processor = Processor()
    processor.rss_gen = RecordingRSSGenerator()
    processor.DELETION_ACK_TIMEOUT_SECONDS = 0.03
    processor.DELETION_POLL_INTERVAL_SECONDS = 0.005

    started = time.monotonic()
    assert await processor.delete_subscription(subscription_id) == "deletion_pending"
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert subscription_dir.exists()
    assert SubscriptionRepository().get_by_id(subscription_id).deletion_status == "pending"

    job_repo.acknowledge(claimed[0]["job_id"], claimed[0]["claim_token"])
    assert await processor.finalize_pending_subscription_deletions() == 1
    assert SubscriptionRepository().get_by_id(subscription_id) is None
    assert processor.rss_gen.unified_calls == 1


@pytest.mark.asyncio
async def test_subscription_cleanup_failure_is_retryable_and_repeated_delete_is_idempotent(
    isolated_data_dir, monkeypatch
):
    init_db()
    subscription_id, _episode_ids, subscription_dir, _feed_path = _create_subscription_with_pending_episodes(1)
    processor = Processor()
    rss = RecordingRSSGenerator()
    processor.rss_gen = rss
    original_remove = processor._remove_subscription_directory
    attempts = 0

    def flaky_remove(subscription_slug):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated cleanup failure")
        return original_remove(subscription_slug)

    monkeypatch.setattr(processor, "_remove_subscription_directory", flaky_remove)

    assert await processor.delete_subscription(subscription_id) == "deletion_pending"
    failed = SubscriptionRepository().get_by_id(subscription_id)
    assert failed.deletion_status == "failed"
    assert "simulated cleanup failure" in failed.deletion_error
    assert subscription_dir.exists()
    with get_db_connection() as conn:
        assert {
            row["status"] for row in conn.execute("SELECT status FROM jobs").fetchall()
        } == {"cancelled"}

    assert await processor.delete_subscription(subscription_id) == "deleted"
    assert await processor.delete_subscription(subscription_id) == "deleted"
    assert attempts == 2
    assert rss.subscription_calls == []
    assert rss.unified_calls == 1
    assert SubscriptionRepository().get_by_id(subscription_id) is None
