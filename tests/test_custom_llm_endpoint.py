import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app.core.ai_services import (
    AdDetector,
    OpenAIProvider,
    normalize_openai_base_url,
)
from app.infra.database import get_db_connection, init_db


@pytest.mark.parametrize(
    "value",
    [
        "ftp://localhost:11434/v1",
        "http://user:password@localhost:11434/v1",
        "http://localhost:11434/v1?api_key=secret",
        "http://localhost:11434/v1#fragment",
        "localhost:11434/v1",
        "",
    ],
)
def test_custom_base_url_rejects_unsafe_forms(value):
    with pytest.raises(ValueError):
        normalize_openai_base_url(value)


def test_custom_base_url_normalizes_trailing_slash():
    assert (
        normalize_openai_base_url(" http://localhost:11434/v1/ ")
        == "http://localhost:11434/v1"
    )


def test_custom_provider_is_keyless_and_never_reuses_openai_cloud_key(caplog):
    detector = AdDetector()
    detector.settings = {
        "openai_api_key": "cloud-secret-must-not-leak",
        "custom_llm_base_url": "http://127.0.0.1:11434/v1/",
        "custom_llm_model": '["qwen2.5:7b"]',
    }

    with caplog.at_level(logging.INFO):
        provider = detector.create_provider("custom")

    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "http://127.0.0.1:11434/v1"
    assert provider.api_keys == ["keyless-local-endpoint"]
    assert provider.models == ["qwen2.5:7b"]
    assert "cloud-secret-must-not-leak" not in caplog.text
    assert "keyless-local-endpoint" not in caplog.text


def test_custom_provider_uses_only_its_explicit_key():
    detector = AdDetector()
    detector.settings = {
        "openai_api_key": "cloud-secret-must-not-leak",
        "custom_llm_api_key": "custom-endpoint-key",
        "custom_llm_base_url": "https://llm.example.test/v1",
        "custom_llm_model": '["vendor/arbitrary-model"]',
    }

    provider = detector.create_provider("custom")

    assert provider.api_keys == ["custom-endpoint-key"]
    assert provider.models == ["vendor/arbitrary-model"]


class CompatibleHandler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.__class__.calls.append(("GET", self.path, self.headers.get("Authorization")))
        payload = {
            "object": "list",
            "data": [{"id": "vendor/arbitrary-local-model", "object": "model"}],
        }
        self._send_json(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = json.loads(self.rfile.read(length))
        self.__class__.calls.append(
            ("POST", self.path, self.headers.get("Authorization"), request_body)
        )
        self._send_json(
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": request_body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "[]"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    def _send_json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def compatible_endpoint():
    CompatibleHandler.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), CompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_keyless_endpoint_uses_same_base_for_listing_and_generation(
    compatible_endpoint,
):
    detector = AdDetector()
    detector.settings = {
        "custom_llm_base_url": compatible_endpoint,
        "custom_llm_model": '["vendor/arbitrary-local-model"]',
    }
    provider = detector.create_provider("custom")

    assert provider.list_models() == ["vendor/arbitrary-local-model"]
    assert provider.generate("Return []") == "[]"
    assert CompatibleHandler.calls[0][0:2] == ("GET", "/v1/models")
    assert CompatibleHandler.calls[1][0:2] == ("POST", "/v1/chat/completions")
    assert all("keyless-local-endpoint" in call[2] for call in CompatibleHandler.calls)


def test_custom_endpoint_migration_adds_only_endpoint_settings(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(app_settings)").fetchall()
        }
        migration = conn.execute(
            "SELECT version FROM schema_migrations WHERE version = ?",
            ("20260724_0009_local_llm_chunking",),
        ).fetchone()

    assert {
        "custom_llm_base_url",
        "custom_llm_api_key",
        "custom_llm_model",
    }.issubset(columns)
    assert not {
        "ad_chunking_enabled",
        "ad_chunk_context_tokens",
        "ad_chunk_overlap_seconds",
        "ad_chunk_max_chunks",
        "ad_include_reasons",
    }.intersection(columns)
    assert migration is not None


def test_ai_section_update_preserves_other_sections_and_saved_secrets(
    isolated_data_dir,
):
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET openai_api_key = 'saved-openai-key',
                active_ai_provider = 'openai',
                openai_model = '["gpt-test"]',
                piper_model = 'saved-voice.onnx'
            WHERE id = 1
            """
        )
        conn.commit()

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/admin/ai/update",
            data={
                "section": "ai_transcription",
                "whisper_model": "small",
                "redirect_to": "/admin/ai/transcription",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
    assert row["whisper_model"] == "small"
    assert row["active_ai_provider"] == "openai"
    assert row["openai_api_key"] == "saved-openai-key"
    assert row["openai_model"] == '["gpt-test"]'
    assert row["piper_model"] == "saved-voice.onnx"


def test_text_settings_store_arbitrary_custom_endpoint_and_model(
    isolated_data_dir,
):
    init_db()
    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/admin/ai/update",
            data={
                "section": "ai_text",
                "active_ai_provider": "custom",
                "custom_llm_base_url": "http://ollama:11434/v1/",
                "custom_llm_api_key": "saved-custom-key",
                "custom_llm_model": '["company/nonstandard:model-tag"]',
                "redirect_to": "/admin/ai/text-analysis",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
    assert row["active_ai_provider"] == "custom"
    assert row["custom_llm_base_url"] == "http://ollama:11434/v1"
    assert row["custom_llm_api_key"] == "saved-custom-key"
    assert json.loads(row["custom_llm_model"]) == ["company/nonstandard:model-tag"]


def test_text_settings_reject_unsafe_custom_endpoint(isolated_data_dir):
    init_db()
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/admin/ai/update",
            data={
                "section": "ai_text",
                "active_ai_provider": "custom",
                "custom_llm_base_url": "http://user:password@ollama:11434/v1",
                "custom_llm_model": '["model"]',
            },
        )

    assert response.status_code == 400


def test_custom_model_refresh_uses_post_body_and_supplied_base_url(
    isolated_data_dir,
    compatible_endpoint,
):
    init_db()
    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/admin/ai/refresh",
            data={
                "provider": "custom",
                "base_url": compatible_endpoint,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"models": ["vendor/arbitrary-local-model"]}
    assert CompatibleHandler.calls[0][0:2] == ("GET", "/v1/models")


def test_ai_page_never_renders_saved_api_key_values(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET openai_api_key = 'visible-openai-secret',
                custom_llm_api_key = 'visible-custom-secret',
                gemini_api_keys = '["visible-gemini-secret"]'
            WHERE id = 1
            """
        )
        conn.commit()

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/admin/ai/text-analysis")

    assert response.status_code == 200
    assert "visible-openai-secret" not in response.text
    assert "visible-custom-secret" not in response.text
    assert "visible-gemini-secret" not in response.text
    assert "Transcript Chunking" not in response.text
