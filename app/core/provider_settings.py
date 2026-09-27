"""Single-model configuration and credential resolution shared by UI and workers."""
import json

from app.core.config import settings

MODEL_FIELDS = {
    'gemini': 'ai_model_cascade', 'openai': 'openai_model',
    'anthropic': 'anthropic_model', 'openrouter': 'openrouter_model',
    'custom': 'custom_llm_model',
}


def first_value(value, default=''):
    """Read an old ordered setting once without maintaining fallback behavior."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
    else:
        decoded = value
    values = decoded if isinstance(decoded, list) else [decoded]
    return next((v.strip() for v in values if isinstance(v, str) and v.strip()), default)


def credential(provider, values):
    """Environment wins; a custom endpoint never inherits a cloud credential."""
    if provider == 'custom':
        return first_value(values.get('custom_llm_api_key'))
    env = getattr(settings, provider.upper() + '_API_KEY', None)
    if env and env.strip():
        return next((key.strip() for key in env.split(',') if key.strip()), '')
    if provider == 'gemini':
        return first_value(values.get('gemini_api_keys')) or first_value(values.get('gemini_api_key'))
    return first_value(values.get(provider + '_api_key'))
