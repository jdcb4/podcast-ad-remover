import pytest
from fastapi import HTTPException

from app.api.audio_routes import _resolve_audio_file_path
from app.core.config import settings


def test_resolve_audio_file_path_allows_files_inside_podcast_root(isolated_data_dir):
    audio_path = isolated_data_dir / "podcasts" / "show" / "episode" / "audio.mp3"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_text("fake audio", encoding="utf-8")

    resolved = _resolve_audio_file_path("show/episode/audio.mp3")

    assert resolved == audio_path.resolve()


def test_resolve_audio_file_path_rejects_absolute_paths_outside_podcast_root(isolated_data_dir):
    outside_path = isolated_data_dir / "secret.mp3"
    outside_path.write_text("not podcast audio", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        _resolve_audio_file_path(str(outside_path))

    assert exc.value.status_code == 403


def test_resolve_audio_file_path_rejects_parent_traversal(isolated_data_dir):
    outside_path = isolated_data_dir / "db" / "podcasts.db"
    outside_path.parent.mkdir(parents=True, exist_ok=True)
    outside_path.write_text("sqlite", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        _resolve_audio_file_path("../db/podcasts.db")

    assert exc.value.status_code == 403


def test_resolve_audio_file_path_rejects_directories(isolated_data_dir):
    directory = isolated_data_dir / "podcasts" / "show"
    directory.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc:
        _resolve_audio_file_path("show")

    assert exc.value.status_code == 404
    assert settings.PODCASTS_DIR == str(isolated_data_dir / "podcasts")
