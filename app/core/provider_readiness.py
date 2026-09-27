"""Configuration readiness for the active provider; no API calls."""
from app.core.provider_settings import MODEL_FIELDS, credential, first_value
from app.core.model_defaults import MODEL_DEFAULTS


def provider_configuration_error(runtime: dict) -> str | None:
    provider = runtime.get('active_ai_provider') or 'gemini'
    if provider not in MODEL_FIELDS:
        return 'Choose a supported AI provider.'
    if provider == 'custom':
        from app.core.ai_services import normalize_openai_base_url
        try:
            normalize_openai_base_url(runtime.get('custom_llm_base_url'))
        except ValueError as error:
            return str(error)
    elif not credential(provider,runtime):
        return f'Configure an API key for the selected {provider} provider.'
    models=MODEL_DEFAULTS.get(provider,[])
    value=runtime.get(MODEL_FIELDS[provider],models[0] if models else '')
    if not first_value(value):
        return f'Choose a model for the {provider} endpoint.'
    return None
