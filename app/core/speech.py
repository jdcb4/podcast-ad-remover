"""API-only speech, one provider/model/voice with no implicit paid fallback."""
import base64
import os
from pathlib import Path
import tempfile
import wave

import httpx

from app.core.config import settings
from app.core.provider_settings import credential
from app.core.provider_budget import provider_request

PROVIDERS = ('gemini', 'openai', 'openrouter', 'custom')
VOICES = {
    'gemini': ('Orus', 'Enceladus', 'Laomedeia', 'Kore', 'Puck', 'Charon', 'Fenrir', 'Aoede'),
    'openai': ('alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse', 'marin', 'cedar'),
}


def speech_configuration(values):
    from app.core.ai_services import normalize_openai_base_url
    provider = values.get('tts_provider')
    if provider not in PROVIDERS:
        raise ValueError('Speech needs configuration in Settings > Voice. Local speech is no longer supported.')
    model = (values.get('tts_model') or '').strip()
    voice = (values.get('tts_voice') or '').strip()
    if not model or not voice:
        raise ValueError('Choose a speech model and voice.')
    if any(c in model for c in ('/', '?', '#')) and provider == 'gemini':
        raise ValueError('Use a Gemini model ID without a URL or path.')
    key = values.get('tts_api_key') if provider == 'custom' else credential(provider, values)
    if not key and provider != 'custom':
        raise ValueError(f'Configure a {provider} API key to enable speech.')
    url = normalize_openai_base_url(values.get('tts_base_url')) if provider == 'custom' else {
        'gemini': 'https://generativelanguage.googleapis.com/v1beta',
        'openai': 'https://api.openai.com/v1', 'openrouter': 'https://openrouter.ai/api/v1',
    }[provider]
    return provider, model, voice, key, url


async def generate_speech(text, output_path, values):
    from app.core.ai_services import raise_permanent_provider_error, rate_limit_error
    provider, model, voice, key, base = speech_configuration(values)
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Speech text is empty')
    if provider == 'gemini':
        url = f'{base}/models/{model}:generateContent'
        payload = {'contents': [{'parts': [{'text': text}]}], 'generationConfig': {
            'responseModalities': ['AUDIO'], 'speechConfig': {'voiceConfig': {
                'prebuiltVoiceConfig': {'voiceName': voice}}}}}
        headers = {'x-goog-api-key': key}
    else:
        url = base.rstrip('/') + '/audio/speech'
        payload = {'model': model, 'voice': voice, 'input': text, 'response_format': 'mp3'}
        headers = {'Authorization': f'Bearer {key}'} if key else {}
    try:
        from app.core.gemini_quota import estimate_input_tokens
        quota = provider == 'gemini' and bool(values.get('gemini_free_tier_enabled'))
        with provider_request(provider + '_tts', model, gemini_free_tier=quota,
                              input_tokens=estimate_input_tokens(payload) if quota else 0):
            async with httpx.AsyncClient(timeout=settings.PROVIDER_TIMEOUT_SECONDS, follow_redirects=False) as client:
                async with client.stream('POST', url, headers=headers, json=payload) as response:
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 32 * 1024 * 1024:
                            raise ValueError('Speech response exceeds 32 MiB')
                        chunks.append(chunk)
                    data = b''.join(chunks)
                    if response.status_code >= 400:
                        bounded = httpx.Response(response.status_code, headers=response.headers, content=data, request=response.request)
                        raise httpx.HTTPStatusError(f'Speech HTTP {response.status_code}', request=response.request, response=bounded)
        if provider == 'gemini':
            import json
            result = json.loads(data)
            parts = [part for candidate in result.get('candidates', [])
                     for part in candidate.get('content', {}).get('parts', [])]
            inline = next((part.get('inlineData') for part in parts if part.get('inlineData')), None)
            if not inline or not inline.get('mimeType', '').startswith('audio/'):
                raise ValueError('Speech response contained no audio')
            data = base64.b64decode(inline['data'], validate=True)
        elif not response.headers.get('content-type', '').startswith('audio/'):
            raise ValueError('Speech endpoint returned a non-audio response')
        if not data:
            raise ValueError('Speech endpoint returned empty audio')
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.speech')
        os.close(fd)
        try:
            if provider == 'gemini':
                # Gemini native speech documents 24kHz, 16-bit, mono PCM.
                with wave.open(temporary, 'wb') as output:
                    output.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                    output.writeframes(data)
            else:
                Path(temporary).write_bytes(data)
            from app.core.audio import AudioProcessor
            if AudioProcessor.get_duration(temporary) <= 0:
                raise ValueError('Speech endpoint returned invalid audio')
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    except Exception as error:
        raise_permanent_provider_error(error)
        if provider == 'gemini' and bool(values.get('gemini_free_tier_enabled')):
            from app.core.gemini_quota import GeminiCooldown, record_error
            cooldown = error if isinstance(error, GeminiCooldown) else record_error(model, error)
            if cooldown:
                from datetime import datetime, timezone
                from app.core.ai_services import RateLimitError
                raise RateLimitError(str(cooldown), provider='gemini',
                    retry_at=datetime.fromtimestamp(cooldown.until, timezone.utc).replace(tzinfo=None)) from error
        if getattr(getattr(error, 'response', None), 'status_code', None) == 429:
            raise rate_limit_error(error, provider) from error
        raise
