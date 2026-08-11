import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from app.web.router import router
from app.infra.database import get_db_connection, init_db
from unittest.mock import patch


@pytest.fixture
def authenticated_client(isolated_data_dir):
    """Create a test client with authenticated admin user."""
    init_db()
    
    with get_db_connection() as conn:
        # Create admin user
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
            ("testadmin", "hashed_password")
        )
        conn.commit()
    
    # Create a test FastAPI app with the router
    test_app = FastAPI()
    test_app.add_middleware(SessionMiddleware, secret_key="test-secret-key")
    test_app.include_router(router)
    
    # Create client with follow_redirects=False to check redirect status
    client = TestClient(test_app, follow_redirects=False)
    
    # Mock the authentication dependency
    with patch('app.web.router.require_admin'):
        yield client


def test_update_ai_settings_saves_transcription_engine_faster_whisper(authenticated_client):
    """Test that transcription_engine can be set to faster-whisper."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx' WHERE id = 1"
        )
        conn.commit()
    
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_transcription",
            "whisper_model": "base",
            "transcription_engine": "faster-whisper",
            "redirect_to": "/admin/ai/transcription"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT transcription_engine FROM app_settings WHERE id = 1"
        ).fetchone()
    
    assert row["transcription_engine"] == "faster-whisper"


def test_update_ai_settings_saves_transcription_engine_whisperx(authenticated_client):
    """Test that transcription_engine can be set to whisperx."""
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_transcription",
            "whisper_model": "base",
            "transcription_engine": "whisperx",
            "redirect_to": "/admin/ai/transcription"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT transcription_engine FROM app_settings WHERE id = 1"
        ).fetchone()
    
    assert row["transcription_engine"] == "whisperx"


def test_update_ai_settings_defaults_transcription_engine_to_faster_whisper(authenticated_client):
    """Test that invalid transcription_engine defaults to faster-whisper."""
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_transcription",
            "whisper_model": "base",
            "transcription_engine": "invalid-engine",
            "redirect_to": "/admin/ai/transcription"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT transcription_engine FROM app_settings WHERE id = 1"
        ).fetchone()
    
    assert row["transcription_engine"] == "faster-whisper"


def test_update_ai_settings_handles_null_transcription_engine(authenticated_client):
    """Test that null transcription_engine defaults to faster-whisper."""
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_transcription",
            "whisper_model": "base",
            "redirect_to": "/admin/ai/transcription"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT transcription_engine FROM app_settings WHERE id = 1"
        ).fetchone()
    
    assert row["transcription_engine"] == "faster-whisper"


def test_update_ai_settings_transcription_section_updates_both_fields(authenticated_client):
    """Test that transcription section updates both whisper_model and transcription_engine."""
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_transcription",
            "whisper_model": "tiny",
            "transcription_engine": "whisperx",
            "redirect_to": "/admin/ai/transcription"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT whisper_model, transcription_engine FROM app_settings WHERE id = 1"
        ).fetchone()
    
    assert row["whisper_model"] == "tiny"
    assert row["transcription_engine"] == "whisperx"


def test_update_ai_settings_other_sections_do_not_change_transcription_engine(authenticated_client):
    """Test that other sections (ai_voice, ai_text) don't affect transcription_engine."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx' WHERE id = 1"
        )
        conn.commit()
    
    # Update ai_voice section
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_voice",
            "tts_provider": "piper",
            "piper_model": "en_GB-cori-high.onnx",
            "redirect_to": "/admin/ai/voice"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT transcription_engine FROM app_settings WHERE id = 1"
        ).fetchone()
    
    # Should still be whisperx
    assert row["transcription_engine"] == "whisperx"


def test_update_ai_settings_invalid_whisper_model_defaults_to_base(authenticated_client):
    """Test that invalid whisper_model defaults to base."""
    response = authenticated_client.post(
        "/admin/ai/update",
        data={
            "section": "ai_transcription",
            "whisper_model": "invalid-model",
            "transcription_engine": "faster-whisper",
            "redirect_to": "/admin/ai/transcription"
        }
    )
    
    assert response.status_code == 303
    
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT whisper_model FROM app_settings WHERE id = 1"
        ).fetchone()
    
    assert row["whisper_model"] == "base"
