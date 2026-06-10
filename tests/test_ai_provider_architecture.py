from app.core.ai_services import AdDetector, OpenAIProvider


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
