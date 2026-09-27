from unittest.mock import Mock
import pytest
from app.core.ai_services import AdDetector, OpenAIProvider
from app.core.config import settings
from app.core.speech import speech_configuration
from app.core import timeline


def test_environment_credential_and_one_model(monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'env-first,env-second')
    detector = AdDetector()
    detector.settings = {'gemini_api_keys': '["saved"]'}
    provider = detector.create_provider('gemini', model='["first", "second"]')
    assert provider.api_keys == ['env-first']
    assert provider.models == ['first']


def test_schema_failure_never_downgrades_or_changes_model(monkeypatch):
    provider = OpenAIProvider('fixture', ['one', 'two'])
    create = Mock(side_effect=ValueError('json_schema unsupported'))
    monkeypatch.setattr(provider.client.chat.completions, 'create', create)
    with pytest.raises(ValueError, match='unsupported'):
        provider.generate_structured([], timeline.SCHEMA)
    assert create.call_count == 1
    assert create.call_args.kwargs['response_format']['type'] == 'json_schema'


@pytest.mark.parametrize('text', ['{"a":1}', '```json\n{"a":1}\n```', 'Here: {"a":1} done.'])
def test_minimal_json_unwrapping(text):
    assert timeline.decode_json(text) == {'a': 1}


@pytest.mark.parametrize('text', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1} {"b":2}', '{"a":', "{'a':1}"])
def test_invalid_json_is_not_repaired(text):
    with pytest.raises(ValueError):
        timeline.decode_json(text)


def test_piper_requires_explicit_speech_setup():
    with pytest.raises(ValueError, match='needs configuration'):
        speech_configuration({'tts_provider': 'piper'})


def test_custom_speech_never_uses_cloud_key(monkeypatch):
    monkeypatch.setattr(settings, 'OPENAI_API_KEY', 'private-cloud-key')
    result = speech_configuration({'tts_provider': 'custom', 'tts_model': 'one',
        'tts_voice': 'voice', 'tts_base_url': 'http://localhost:9000/v1'})
    assert result[3] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['gemini', 'openai', 'openrouter', 'custom'])
async def test_speech_adapter_uses_one_request_and_preserves_failure_output(tmp_path, monkeypatch, provider):
    import base64
    import httpx
    from app.core.speech import generate_speech
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', 'fixture')
    monkeypatch.setattr(settings, 'OPENAI_API_KEY', 'fixture')
    monkeypatch.setattr(settings, 'OPENROUTER_API_KEY', 'fixture')
    monkeypatch.setattr('app.core.audio.AudioProcessor.get_duration', lambda _: 1)
    requests = []
    def respond(request):
        requests.append(request)
        if provider == 'gemini':
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'inlineData': {
                'mimeType': 'audio/L16;rate=24000', 'data': base64.b64encode(b'\0' * 48).decode()}}]}}]})
        return httpx.Response(200, content=b'fixture-audio', headers={'content-type': 'audio/mpeg'})
    client = httpx.AsyncClient
    monkeypatch.setattr('app.core.speech.httpx.AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    output = tmp_path / 'speech.mp3'
    values = {'tts_provider': provider, 'tts_model': 'model', 'tts_voice': 'voice', 'tts_base_url': 'http://localhost:9000/v1'}
    await generate_speech('Hello', str(output), values)
    assert len(requests) == 1 and output.stat().st_size > 0
    original = output.read_bytes()
    monkeypatch.setattr('app.core.audio.AudioProcessor.get_duration', lambda _: 0)
    with pytest.raises(ValueError, match='invalid audio'):
        await generate_speech('Hello', str(output), values)
    assert output.read_bytes() == original
