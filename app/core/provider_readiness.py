"""Configuration readiness for the selected provider; never calls a remote API."""
import json

from app.core.config import settings


def provider_configuration_error(runtime: dict) -> str | None:
    provider = runtime.get('active_ai_provider') or 'gemini'
    if provider == 'custom':
        from app.core.ai_services import normalize_openai_base_url
        try:
            normalize_openai_base_url(runtime.get('custom_llm_base_url'))
        except ValueError as error:
            return str(error)
        if not runtime.get('custom_llm_model') or runtime.get('custom_llm_model') == '[]':
            return 'Choose a model for the custom endpoint.'
        return None  # Keyless local endpoints are supported.
    if provider not in ('gemini', 'openai', 'anthropic', 'openrouter'):
        return 'Choose a supported AI provider.'
    keys = [runtime.get(provider + '_api_key'), getattr(settings, provider.upper() + '_API_KEY', None)]
    if provider == 'gemini':
        try:
            parsed = json.loads(runtime.get('gemini_api_keys') or '[]')
            if isinstance(parsed, list):
                keys.extend(parsed)
        except (ValueError, TypeError):
            pass
    if not any(isinstance(key, str) and key.strip() for key in keys):
        return f'Configure an API key for the selected {provider} provider.'
    column = 'ai_model_cascade' if provider == 'gemini' else provider + '_model'
    if runtime.get(column) == '[]':
        return f'Choose at least one {provider} model.'
    return None
