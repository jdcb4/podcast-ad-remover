"""Read-only speech discovery. No synthesis, redirects, or credential-bearing errors."""
import asyncio
import httpx
from app.core.speech import VOICES
from app.core.provider_settings import credential
from app.core.ai_services import normalize_openai_base_url

LEGACY_OPENAI_VOICES = 'alloy ash coral echo fable onyx nova sage shimmer'.split()


def known_voices(provider, model):
    if provider == 'gemini' or (provider == 'openrouter' and model.startswith('google/gemini-')):
        return list(VOICES['gemini'])
    if provider == 'openai' or (provider == 'openrouter' and model.startswith('openai/')):
        return LEGACY_OPENAI_VOICES if model.removeprefix('openai/').startswith('tts-1') else list(VOICES['openai'])
    return []


async def _json(client, url, headers, params=None):
    async with client.stream('GET', url, headers=headers, params=params) as response:
        response.raise_for_status()
        data = bytearray()
        async for chunk in response.aiter_bytes():
            data.extend(chunk)
            if len(data) > 4*1024*1024:
                raise ValueError('Catalog response is too large')
    import json
    return json.loads(data)


async def refresh_catalog(provider, model, values):
    if provider not in ('gemini','openai','openrouter','custom'):
        raise ValueError('Choose a speech provider first.')
    key = values.get('tts_api_key') if provider == 'custom' else credential(provider, values)
    if not key and provider in ('gemini','openai'):
        raise ValueError('Enter or save an API key to refresh this provider.')
    base = normalize_openai_base_url(values.get('tts_base_url')) if provider == 'custom' else {
        'gemini':'https://generativelanguage.googleapis.com/v1beta',
        'openai':'https://api.openai.com/v1','openrouter':'https://openrouter.ai/api/v1'}[provider]
    headers = {'x-goog-api-key':key} if provider == 'gemini' else ({'Authorization':f'Bearer {key}'} if key else {})
    voices = known_voices(provider, model)
    notes = []
    async with asyncio.timeout(30):
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            models, token = [], None
            for _ in range(20):
                params = {'output_modalities':'speech'} if provider == 'openrouter' else {}
                if provider == 'gemini':
                    params['pageSize'] = 1000
                    if token: params['pageToken'] = token
                data = await _json(client, base+'/models', headers, params)
                for entry in data.get('models' if provider=='gemini' else 'data', []):
                    name = entry.get('name' if provider=='gemini' else 'id','').removeprefix('models/')
                    if name and (provider in ('openrouter','custom') or 'tts' in name.lower()):
                        models.append(name)
                token = data.get('nextPageToken') if provider=='gemini' else None
                if not token: break
            else: raise ValueError('Model catalog pagination limit reached; narrow the endpoint catalog.')
            if provider == 'gemini':
                try:
                    discovered, token = [], None
                    for _ in range(20):
                        params = {'type':'prebuilt','page_size':1000}
                        if token: params['page_token'] = token
                        data = await _json(client,base+'/voices',headers,params)
                        discovered.extend(v['id'] for v in data.get('voices',[]) if v.get('id') and v.get('type')=='prebuilt')
                        token = data.get('next_page_token')
                        if not token: break
                    else: raise ValueError('Voice catalog pagination limit reached.')
                    if discovered: voices = discovered
                    else: notes.append('No live prebuilt voices returned; showing the 30 documented studio voices.')
                except (httpx.HTTPError, ValueError):
                    notes.append('Voice catalog unavailable; showing the 30 documented studio voices.')
            elif provider == 'custom':
                try:
                    data = await _json(client,base+'/audio/voices',headers,{'model':model})
                    entries = data.get('voices',data.get('data',[]))
                    voices = [v if isinstance(v,str) else v.get('id',v.get('voice_id','')) for v in entries]
                except (httpx.HTTPError, ValueError):
                    notes.append('This endpoint does not expose a voice catalog. Enter a voice ID from its documentation.')
            elif provider == 'openrouter':
                notes.append('Voices depend on the selected model. Documented OpenAI/Gemini voices are listed; enter a provider voice ID for other models.')
            else:
                notes.append('Voices are the documented built-in options for this model; OpenAI does not expose a built-in voice-list endpoint.')
    if not models: notes.append('No speech models returned. You can still enter a supported model ID.')
    return {'models':sorted(set(models)), 'voices':sorted(set(v for v in voices if v)), 'note':' '.join(notes)}
