import base64
import wave

import pytest

from app.core.ai_services import AdDetector, OpenAIProvider, piper_tts_available
from app.infra.database import get_db_connection, init_db


def test_gemini_uses_openai_compatible_provider_with_multiple_keys():
    detector = AdDetector()
    detector.settings = {"gemini_api_keys": '["key-one", "key-two"]'}

    provider = detector.create_provider("gemini", model='["gemini-2.5-flash"]')

    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == AdDetector.GEMINI_OPENAI_BASE_URL
    assert provider.provider_name == "Gemini"
    assert provider.api_keys == ["key-one", "key-two"]
    assert provider.models == ["gemini-2.5-flash"]
    assert provider.model_prefixes == ("gemini-",)


def test_gemini_explicit_key_overrides_saved_keys():
    detector = AdDetector()
    detector.settings = {"gemini_api_keys": '["saved-key"]'}

    provider = detector.create_provider("gemini", api_key="explicit-one,explicit-two", model="gemini-2.5-flash")

    assert provider.api_keys == ["explicit-one", "explicit-two"]


def test_openrouter_keeps_openai_compatible_provider_without_model_filter():
    detector = AdDetector()
    detector.settings = {"openrouter_api_key": "openrouter-key"}

    provider = detector.create_provider("openrouter", model="google/gemini-3.1-flash-lite")

    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_name == "OpenRouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.model_prefixes is None


def test_tts_can_be_disabled_by_image_environment(monkeypatch):
    monkeypatch.setenv("TTS_ENABLED", "0")

    assert piper_tts_available() is False


class FakeGeminiTtsResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeAsyncClient:
    calls = []
    responses = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_gemini_tts_generates_wav_with_selected_voice(monkeypatch, isolated_data_dir, tmp_path):
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET tts_provider = 'gemini',
                gemini_api_keys = ?,
                gemini_tts_voice = 'Enceladus',
                gemini_tts_model_cascade = ?
            WHERE id = 1
            """,
            ('["test-key"]', '["gemini-3.1-flash-tts-preview"]'),
        )
        conn.commit()

    pcm_audio = b"\x00\x00" * 240
    FakeAsyncClient.calls = []
    FakeAsyncClient.responses = [
        FakeGeminiTtsResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "data": base64.b64encode(pcm_audio).decode("ascii")
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )
    ]
    monkeypatch.setattr("app.core.ai_services.httpx.AsyncClient", FakeAsyncClient)

    output_path = tmp_path / "tts.mp3"
    detector = AdDetector()
    await detector.generate_audio('"Hello" **world**', str(output_path))

    assert len(FakeAsyncClient.calls) == 1
    call = FakeAsyncClient.calls[0]
    assert call["url"].endswith("/models/gemini-3.1-flash-tts-preview:generateContent")
    assert call["headers"]["x-goog-api-key"] == "test-key"
    assert call["json"]["contents"][0]["parts"][0]["text"] == "Hello world"
    voice = call["json"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert voice["voiceName"] == "Enceladus"

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24000


@pytest.mark.asyncio
async def test_gemini_tts_falls_back_to_next_model(monkeypatch, isolated_data_dir, tmp_path):
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET tts_provider = 'gemini',
                gemini_api_keys = ?,
                gemini_tts_model_cascade = ?
            WHERE id = 1
            """,
            ('["test-key"]', '["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]'),
        )
        conn.commit()

    FakeAsyncClient.calls = []
    FakeAsyncClient.responses = [
        FakeGeminiTtsResponse(status_code=429, text="quota"),
        FakeGeminiTtsResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "data": base64.b64encode(b"\x00\x00" * 120).decode("ascii")
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    ]
    monkeypatch.setattr("app.core.ai_services.httpx.AsyncClient", FakeAsyncClient)

    output_path = tmp_path / "tts.mp3"
    detector = AdDetector()
    await detector.generate_audio("fallback test", str(output_path))

    assert [call["url"].split("/models/")[1].split(":")[0] for call in FakeAsyncClient.calls] == [
        "gemini-3.1-flash-tts-preview",
        "gemini-2.5-flash-preview-tts",
    ]
    assert output_path.exists()
