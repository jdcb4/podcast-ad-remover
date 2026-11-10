"""Ordered model and credential resolution shared by UI and workers."""
import json
import re

from app.core.config import settings

MODEL_FIELDS = {
    'gemini': 'ai_model_cascade', 'openai': 'openai_model',
    'anthropic': 'anthropic_model', 'openrouter': 'openrouter_model',
    'custom': 'custom_llm_model',
}
MAX_CONFIGURED_VALUES = 20


def all_values(value):
    """Read an ordered setting: a JSON list, or text separated by commas or newlines."""
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
    if isinstance(decoded, str):
        decoded = re.split(r'[,\r\n]+', decoded)
    if not isinstance(decoded, list):
        return []
    cleaned = (item.strip() for item in decoded if isinstance(item, str))
    return list(dict.fromkeys(item for item in cleaned if item))


def first_value(value, default=''):
    return next(iter(all_values(value)), default)


def store_values(values):
    """One value stays plain text, so a single-model setting keeps its existing format."""
    return json.dumps(values) if len(values) > 1 else (values[0] if values else '')


def saved_credentials(provider, values):
    if provider == 'custom':
        return all_values(values.get('custom_llm_api_key'))
    if provider == 'gemini':
        return all_values(values.get('gemini_api_keys')) or all_values(values.get('gemini_api_key'))
    return all_values(values.get(provider + '_api_key'))


def credentials(provider, values):
    """Environment wins; a custom endpoint never inherits a cloud credential."""
    env = None if provider == 'custom' else getattr(settings, provider.upper() + '_API_KEY', None)
    return all_values(env) if env and env.strip() else saved_credentials(provider, values)


def credential(provider, values):
    return next(iter(credentials(provider, values)), '')
