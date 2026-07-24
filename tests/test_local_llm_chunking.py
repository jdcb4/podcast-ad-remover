import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.ai_services import (
    AdDetectionParseError,
    AdDetector,
    OpenAIProvider,
    TranscriptChunkingError,
    normalize_openai_base_url,
)
from app.core.processor import Processor
from app.infra.database import get_db_connection, init_db


class FakeProvider:
    provider_name = "Fake local"
    last_model = "fake/local"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        assert max_tokens == AdDetector.CHUNK_OUTPUT_RESERVE_TOKENS
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EchoContentProvider:
    provider_name = "Fake local"
    last_model = "fake/local"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens=None):
        import re

        self.prompts.append(prompt)
        assert max_tokens == AdDetector.WHITELIST_OUTPUT_RESERVE_TOKENS
        transcript_text = prompt.split("Transcript:\n", 1)[1]
        rows = []
        for start, end in re.findall(r"\[(\d+\.\d+)-(\d+\.\d+)\]", transcript_text):
            rows.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "label": "Content",
                }
            )
        return json.dumps(rows)


def make_segment(index: int, text_size: int = 80):
    return {
        "start": float(index * 10),
        "end": float(index * 10 + 9),
        "text": f"SEGMENT_{index:04d}_" + ("x" * text_size),
    }


def configured_detector(monkeypatch, settings, provider):
    detector = AdDetector()
    monkeypatch.setattr(detector, "_load_settings", lambda: dict(settings))
    monkeypatch.setattr(detector, "_get_provider", lambda: provider)
    return detector


def default_options():
    return {
        "remove_ads": True,
        "remove_promos": True,
        "remove_intros": False,
        "remove_outros": False,
        "custom_instructions": None,
    }


def test_chunking_keeps_short_transcript_on_one_request(monkeypatch):
    provider = FakeProvider(["[]"])
    detector = configured_detector(
        monkeypatch,
        {
            "active_ai_provider": "custom",
            "ad_chunking_enabled": 1,
            "ad_chunk_context_tokens": 8192,
            "ad_chunk_overlap_seconds": 30,
            "ad_chunk_max_chunks": 32,
            "ad_include_reasons": 1,
        },
        provider,
    )

    result = detector.detect_ads({"segments": [make_segment(0)]}, default_options())

    assert result == []
    assert len(provider.prompts) == 1
    assert detector.last_detection_metadata["chunk_count"] == 1
    assert detector.last_detection_metadata["complete"] is True


def test_chunking_covers_long_transcript_without_empty_trailing_request(monkeypatch):
    settings = {
        "active_ai_provider": "custom",
        "ad_chunking_enabled": 1,
        "ad_chunk_context_tokens": 4096,
        "ad_chunk_overlap_seconds": 20,
        "ad_chunk_max_chunks": 64,
        "ad_include_reasons": 0,
    }
    transcript_segments = [make_segment(index, text_size=160) for index in range(80)]
    planning_detector = AdDetector()
    planning_detector.settings = settings
    chunks = planning_detector._build_transcript_chunks(
        transcript_segments, default_options(), whitelist_mode=False
    )
    provider = FakeProvider(["[]"] * len(chunks))
    detector = configured_detector(monkeypatch, settings, provider)

    detector.detect_ads({"segments": transcript_segments}, default_options())

    assert len(provider.prompts) == len(chunks)
    assert len(provider.prompts) > 1
    assert all("Transcript:\n" in prompt for prompt in provider.prompts)
    for index in range(len(transcript_segments)):
        marker = f"SEGMENT_{index:04d}_"
        assert any(marker in prompt for prompt in provider.prompts)
    assert not any('"reason"' in prompt.split("Transcript:\n", 1)[0] for prompt in provider.prompts)


def test_chunk_failure_propagates_instead_of_accepting_partial_results(monkeypatch):
    settings = {
        "active_ai_provider": "custom",
        "ad_chunking_enabled": 1,
        "ad_chunk_context_tokens": 4096,
        "ad_chunk_overlap_seconds": 20,
        "ad_chunk_max_chunks": 64,
        "ad_include_reasons": 1,
    }
    transcript_segments = [make_segment(index, text_size=200) for index in range(80)]
    planner = AdDetector()
    planner.settings = settings
    chunks = planner._build_transcript_chunks(
        transcript_segments, default_options(), whitelist_mode=False
    )
    assert len(chunks) > 1
    provider = FakeProvider(["[]", RuntimeError("local endpoint unavailable")])
    detector = configured_detector(monkeypatch, settings, provider)

    with pytest.raises(RuntimeError, match="local endpoint unavailable"):
        detector.detect_ads({"segments": transcript_segments}, default_options())

    assert detector.last_detection_metadata["complete"] is False
    assert detector.last_detection_metadata["failed_chunk"] == 2
    assert detector.last_detection_metadata["completed_chunks"] == 1


def test_whitelist_chunking_preserves_every_primary_transcript_segment(monkeypatch):
    settings = {
        "active_ai_provider": "custom",
        "ad_chunking_enabled": 1,
        "ad_chunk_context_tokens": 4096,
        "ad_chunk_overlap_seconds": 20,
        "ad_chunk_max_chunks": 64,
        "ad_include_reasons": 0,
    }
    transcript_segments = [make_segment(index, text_size=180) for index in range(80)]
    provider = EchoContentProvider()
    detector = configured_detector(monkeypatch, settings, provider)

    result = detector.detect_ads(
        {"segments": transcript_segments},
        default_options(),
        whitelist_mode=True,
    )

    assert len(provider.prompts) > 1
    assert [(segment["start"], segment["end"]) for segment in result] == [
        (segment["start"], segment["end"]) for segment in transcript_segments
    ]


def test_whitelist_uses_smaller_chunks_to_protect_output_budget():
    detector = AdDetector()
    detector.settings = {
        "ad_chunk_context_tokens": 32768,
        "ad_chunk_overlap_seconds": 30,
        "ad_chunk_max_chunks": 64,
        "ad_include_reasons": 0,
    }
    transcript_segments = [make_segment(index, text_size=160) for index in range(80)]

    whitelist_chunks = detector._build_transcript_chunks(
        transcript_segments, default_options(), whitelist_mode=True
    )
    blacklist_chunks = detector._build_transcript_chunks(
        transcript_segments, default_options(), whitelist_mode=False
    )

    assert len(whitelist_chunks) > 1
    assert len(whitelist_chunks) > len(blacklist_chunks)
    assert all(
        chunk["estimated_tokens"] < 7000
        for chunk in whitelist_chunks
    )


def test_chunk_limit_is_enforced_before_requests():
    detector = AdDetector()
    detector.settings = {
        "ad_chunk_context_tokens": 4096,
        "ad_chunk_overlap_seconds": 0,
        "ad_chunk_max_chunks": 1,
        "ad_include_reasons": 1,
    }

    with pytest.raises(TranscriptChunkingError, match="exceeding"):
        detector._build_transcript_chunks(
            [make_segment(index, text_size=240) for index in range(120)],
            default_options(),
            whitelist_mode=False,
        )


def test_chunk_merge_deduplicates_same_labels_but_not_incompatible_labels():
    detector = AdDetector()

    result = detector._merge_chunk_detections(
        [
            {"start": 10.0, "end": 20.0, "label": "Ad", "reason": "first"},
            {"start": 19.5, "end": 25.0, "label": "Ad", "reason": "duplicate"},
            {"start": 19.0, "end": 24.0, "label": "Content", "reason": ""},
        ]
    )

    assert result == [
        {"start": 10.0, "end": 25.0, "label": "Ad", "reason": "first"},
        {"start": 19.0, "end": 24.0, "label": "Content", "reason": ""},
    ]


def test_reason_setting_changes_prompt_but_parser_accepts_extra_reason():
    detector = AdDetector()
    detector.settings = {
        "ad_include_reasons": 0,
        "ad_prompt_base": """
Targets: {targets}
{custom_instr}
Return a JSON array with a "reason" field.
Example: [{"start": 0, "end": 1, "label": "Ad", "reason": "legacy"}]
""",
    }

    prompt = detector._build_ad_prompt(default_options(), "[0.00-1.00] hello", False)
    parsed = detector._parse_ad_response(
        '[{"start":0,"end":1,"label":"Ad","reason":"returned anyway"}]'
    )

    assert "Do not include a reason field." in prompt
    assert '"reason": "legacy"' not in prompt
    assert parsed[0]["reason"] == "returned anyway"


@pytest.mark.parametrize(
    "value",
    [
        "ftp://localhost:11434/v1",
        "http://user:password@localhost:11434/v1",
        "http://localhost:11434/v1?api_key=secret",
        "http://localhost:11434/v1#fragment",
    ],
)
def test_custom_base_url_rejects_unsafe_forms(value):
    with pytest.raises(ValueError):
        normalize_openai_base_url(value)


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


class CompatibleHandler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.__class__.calls.append(("GET", self.path, self.headers.get("Authorization")))
        payload = {"object": "list", "data": [{"id": "local-test-model", "object": "model"}]}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request_body = json.loads(self.rfile.read(length))
        self.__class__.calls.append(
            ("POST", self.path, self.headers.get("Authorization"), request_body)
        )
        payload = {
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_keyless_compatible_endpoint_uses_same_base_for_listing_and_generation():
    CompatibleHandler.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), CompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        detector = AdDetector()
        detector.settings = {
            "custom_llm_base_url": base_url,
            "custom_llm_model": '["local-test-model"]',
        }
        provider = detector.create_provider("custom")

        assert provider.list_models() == ["local-test-model"]
        assert provider.generate("Return []") == "[]"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert CompatibleHandler.calls[0][0:2] == ("GET", "/v1/models")
    assert CompatibleHandler.calls[1][0:2] == ("POST", "/v1/chat/completions")
    assert all("keyless-local-endpoint" in call[2] for call in CompatibleHandler.calls)


def test_summary_features_are_disabled_by_chunking_without_losing_preferences():
    subscription = SimpleNamespace(
        ai_rewrite_description=True,
        ai_audio_summary=True,
        append_summary=False,
    )

    assert Processor._summary_feature_flags(subscription, {"ad_chunking_enabled": 1}) == (
        False,
        False,
        True,
    )
    assert Processor._summary_feature_flags(subscription, {"ad_chunking_enabled": 0}) == (
        True,
        True,
        False,
    )


def test_local_llm_migration_adds_settings(isolated_data_dir):
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
        "ad_chunking_enabled",
        "ad_chunk_context_tokens",
        "ad_chunk_overlap_seconds",
        "ad_chunk_max_chunks",
        "ad_include_reasons",
    }.issubset(columns)
    assert migration is not None


def test_ai_section_update_preserves_other_sections_and_saved_secrets(isolated_data_dir):
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
    assert "Summary generation is disabled while this is enabled" in response.text
