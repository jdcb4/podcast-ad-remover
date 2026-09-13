"""Ordered model defaults shared by runtime, settings UI and database setup."""

MODEL_DEFAULTS = {
    'gemini': [
        'gemini-3.8-flash',
        'gemini-3.7-flash',
        'gemini-3.6-flash',
        'gemini-3.5-flash',
        'gemini-3-flash-preview',
        'gemini-3.5-flash-lite',
        'gemini-3.1-flash-lite',
    ],
    'gemini_tts': [
        'gemini-3.1-flash-tts-preview',
        'gemini-2.5-flash-preview-tts',
    ],
    'openai': [
        'gpt-6-astra',
        'gpt-5.6-sol',
        'gpt-5.6-terra',
        'gpt-5.6-luna',
    ],
    'anthropic': [
        'claude-fable-5-1',
        'claude-opus-5',
        'claude-sonnet-5',
        'claude-haiku-4-5-20251001',
    ],
    'openrouter': [
        'openai/gpt-5.6-terra',
        'openai/gpt-5.6-luna',
        'anthropic/claude-sonnet-5',
        'anthropic/claude-haiku-4.5',
        'google/gemini-3.8-flash',
        'google/gemini-3.5-flash-lite',
        'tencent/hy4-preview',
        'tencent/hy3',
        'z-ai/glm-5.3-flash',
        'z-ai/glm-5.3',
        'deepseek/deepseek-v4.1-flash',
        'deepseek/deepseek-v4-pro',
    ],
    'custom': [
    ],
}
