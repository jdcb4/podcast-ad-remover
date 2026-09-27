import asyncio
import httpx
import pytest
from app.core import speech_catalog as catalog
from app.core.config import settings


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(catalog.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))


def test_gemini_catalog_pages_models_and_prebuilt_voices(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','environment-key')
    seen=[]
    def handler(request):
        seen.append(request)
        assert request.headers['x-goog-api-key']=='environment-key'
        assert request.method=='GET'
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'models':[{'name':'models/gemini-2.5-flash-preview-tts'},{'name':'models/not-speech'}]})
        if not request.url.params.get('page_token'):
            return httpx.Response(200,json={'voices':[{'id':'Kore','type':'prebuilt'}],'next_page_token':'next'})
        return httpx.Response(200,json={'voices':[{'id':'Puck','type':'prebuilt'}]})
    transport(monkeypatch,handler)
    result=asyncio.run(catalog.refresh_catalog('gemini','gemini-2.5-flash-preview-tts',{'gemini_api_key':'saved'}))
    assert result['models']==['gemini-2.5-flash-preview-tts']
    assert result['voices']==['Kore','Puck']
    assert len(seen)==3


def test_openrouter_discovery_filters_speech_and_does_not_invent_voices(monkeypatch):
    def handler(request):
        assert request.url.params['output_modalities']=='speech'
        return httpx.Response(200,json={'data':[{'id':'vendor/speech'}]})
    transport(monkeypatch,handler)
    result=asyncio.run(catalog.refresh_catalog('openrouter','vendor/speech',{}))
    assert result['models']==['vendor/speech'] and result['voices']==[]
    assert 'depend on the selected model' in result['note']


def test_custom_credentials_isolated_and_missing_voice_catalog_explained(monkeypatch):
    monkeypatch.setattr(settings,'OPENAI_API_KEY','not-for-custom')
    def handler(request):
        assert request.headers['authorization']=='Bearer custom-only'
        if request.url.path.endswith('/models'):return httpx.Response(200,json={'data':[{'id':'my-voice-model'}]})
        return httpx.Response(404)
    transport(monkeypatch,handler)
    result=asyncio.run(catalog.refresh_catalog('custom','my-voice-model',{'tts_api_key':'custom-only','tts_base_url':'http://localhost:8080/v1'}))
    assert result['models']==['my-voice-model'] and 'does not expose' in result['note']


def test_documented_voice_sets_are_model_specific():
    assert len(catalog.known_voices('gemini',''))==30
    assert len(catalog.known_voices('openai','gpt-4o-mini-tts'))==13
    assert 'marin' not in catalog.known_voices('openai','tts-1')
    assert catalog.known_voices('custom','tts-1')==[]
