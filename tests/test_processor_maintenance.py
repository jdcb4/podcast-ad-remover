from pathlib import Path
from datetime import datetime, timedelta
import os

import pytest

from app.core.config import settings
from app.core.models import SubscriptionCreate
from app.core.processor import Processor
from app.infra.database import get_db_connection, init_db
from app.infra.repository import SubscriptionRepository


def test_processor_log_cleanup_does_not_rewrite_app_log():
    source = Path("app/core/processor.py").read_text(encoding="utf-8")
    cleanup_body = source.split("async def cleanup_old_logs", 1)[1].split("async def cleanup_old_episodes", 1)[0]

    assert "app.log" not in cleanup_body
    assert "readlines()" not in cleanup_body
    assert "writelines" not in cleanup_body


def test_remove_episode_directory_requires_podcast_root(monkeypatch, tmp_path):
    podcast_root = tmp_path / "podcasts"
    podcast_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    processor = object.__new__(Processor)

    assert processor._remove_episode_directory(str(outside_dir), "delete") is False
    assert outside_dir.exists()


def test_remove_episode_directory_removes_directory_inside_podcast_root(monkeypatch, tmp_path):
    podcast_root = tmp_path / "podcasts"
    episode_dir = podcast_root / "podcast" / "episode"
    episode_dir.mkdir(parents=True)
    (episode_dir / "audio.mp3").write_text("fake", encoding="utf-8")

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))

    processor = object.__new__(Processor)

    assert processor._remove_episode_directory(str(episode_dir), "delete") is True
    assert not episode_dir.exists()


def test_remove_file_if_exists_removes_regular_file(tmp_path):
    path = tmp_path / "original.mp3.part"
    path.write_text("partial", encoding="utf-8")

    processor = object.__new__(Processor)

    processor._remove_file_if_exists(str(path), "partial download")

    assert not path.exists()


def test_episode_download_uses_partial_file_before_final_audio():
    source = Path("app/core/processor.py").read_text(encoding="utf-8")

    assert 'temp_input_path = f"{input_path}.part"' in source
    assert 'aiofiles.open(temp_input_path, "wb")' in source
    assert "os.replace(temp_input_path, input_path)" in source


def test_cleanup_stale_temporary_files_only_removes_old_temp_files(monkeypatch, tmp_path):
    podcast_root = tmp_path / "podcasts"
    episode_dir = podcast_root / "podcast" / "episode"
    episode_dir.mkdir(parents=True)

    old_part = episode_dir / "original.mp3.part"
    old_tmp = episode_dir / "cleaned.mp3.tmp.mp3"
    recent_part = episode_dir / "active.mp3.part"
    real_audio = episode_dir / "cleaned.mp3"

    for path in [old_part, old_tmp, recent_part, real_audio]:
        path.write_text("audio", encoding="utf-8")

    old_timestamp = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(old_part, (old_timestamp, old_timestamp))
    os.utime(old_tmp, (old_timestamp, old_timestamp))

    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    processor = object.__new__(Processor)

    assert processor._cleanup_stale_temporary_files(max_age_hours=24) == 2
    assert not old_part.exists()
    assert not old_tmp.exists()
    assert recent_part.exists()
    assert real_audio.exists()


def test_validate_download_response_rejects_oversized_content_length(monkeypatch):
    processor = object.__new__(Processor)
    monkeypatch.setattr(settings, "MAX_DOWNLOAD_BYTES", 100)
    monkeypatch.setattr(settings, "MIN_FREE_SPACE_BYTES", 10)

    with pytest.raises(RuntimeError, match="maximum size"):
        processor._validate_download_response(
            "https://example.com/audio.mp3",
            "https://example.com/audio.mp3",
            {"Content-Length": "101", "Content-Type": "audio/mpeg"},
            free_space=1000,
        )


def test_validate_download_response_rejects_download_that_would_cross_free_space_floor(monkeypatch):
    processor = object.__new__(Processor)
    monkeypatch.setattr(settings, "MAX_DOWNLOAD_BYTES", 1000)
    monkeypatch.setattr(settings, "MIN_FREE_SPACE_BYTES", 100)

    with pytest.raises(RuntimeError, match="minimum free disk space"):
        processor._validate_download_response(
            "https://example.com/audio.mp3",
            "https://example.com/audio.mp3",
            {"Content-Length": "450", "Content-Type": "audio/mpeg"},
            free_space=500,
        )


def test_validate_download_response_rejects_non_audio_content(monkeypatch):
    processor = object.__new__(Processor)
    monkeypatch.setattr(settings, "MAX_DOWNLOAD_BYTES", 1000)
    monkeypatch.setattr(settings, "MIN_FREE_SPACE_BYTES", 100)

    with pytest.raises(RuntimeError, match="did not return audio"):
        processor._validate_download_response(
            "https://example.com/audio.mp3",
            "https://example.com/audio.mp3",
            {"Content-Length": "50", "Content-Type": "text/html"},
            free_space=500,
        )


def test_validate_download_response_ignores_invalid_content_length(monkeypatch):
    processor = object.__new__(Processor)
    monkeypatch.setattr(settings, "MAX_DOWNLOAD_BYTES", 1000)
    monkeypatch.setattr(settings, "MIN_FREE_SPACE_BYTES", 100)

    total = processor._validate_download_response(
        "https://example.com/audio.mp3",
        "https://example.com/audio.mp3",
        {"Content-Length": "not-a-number", "Content-Type": "audio/mpeg"},
        free_space=500,
    )

    assert total == 0


@pytest.mark.asyncio
async def test_cleanup_uses_current_global_retention_for_inheriting_podcasts(
    isolated_data_dir,
    monkeypatch,
):
    init_db()
    repository = SubscriptionRepository()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET default_retention_limit = 3,
                default_manual_retention_days = 30
            WHERE id = 1
            """
        )
        conn.commit()

    subscription = repository.create(
        SubscriptionCreate(feed_url="https://example.com/real-footy.xml"),
        "Real Footy",
        "real-footy",
        retention_limit=1,
        inherit_retention=True,
    )
    with get_db_connection() as conn:
        for index in range(4):
            conn.execute(
                """
                INSERT INTO episodes
                    (subscription_id, guid, title, pub_date, original_url, status, processed_at, is_manual_download)
                VALUES (?, ?, ?, datetime('now', ?), ?, 'completed', datetime('now', ?), 0)
                """,
                (
                    subscription.id,
                    f"auto-{index}",
                    f"Auto {index}",
                    f"-{index} days",
                    f"https://example.com/auto-{index}.mp3",
                    f"-{index} days",
                ),
            )
        manual_id = conn.execute(
            """
            INSERT INTO episodes
                (subscription_id, guid, title, pub_date, original_url, status, processed_at, is_manual_download)
            VALUES (?, 'manual', 'Manual', datetime('now'), 'https://example.com/manual.mp3',
                    'completed', datetime('now', '-20 days'), 1)
            """,
            (subscription.id,),
        ).lastrowid
        oldest_auto_id = conn.execute(
            "SELECT id FROM episodes WHERE guid = 'auto-3'"
        ).fetchone()["id"]
        conn.commit()

    deleted = []
    processor = object.__new__(Processor)

    async def record_delete(episode_id):
        deleted.append(episode_id)

    monkeypatch.setattr(processor, "delete_episode", record_delete)
    await processor.cleanup_old_episodes()

    assert deleted == [oldest_auto_id]
    assert manual_id not in deleted

    deleted.clear()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET default_retention_limit = 2,
                default_manual_retention_days = 14
            WHERE id = 1
            """
        )
        conn.commit()

    await processor.cleanup_old_episodes()

    assert manual_id in deleted
    assert len(deleted) == 3
