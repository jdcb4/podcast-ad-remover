"""Resolve effective podcast settings from explicit group inheritance flags."""

from __future__ import annotations

from typing import Any, Mapping


GROUP_FIELDS = {
    "content_removal": ("remove_ads", "remove_promos", "remove_intros", "remove_outros"),
    "retention": ("retention_limit", "retention_days", "manual_retention_days"),
    "default_features": (
        "ai_rewrite_description",
        "ai_audio_summary",
        "append_title_intro",
        "watermark_artwork",
    ),
    "custom_instructions": ("custom_instructions",),
}

GLOBAL_FIELDS = {
    "remove_ads": "default_remove_ads",
    "remove_promos": "default_remove_promos",
    "remove_intros": "default_remove_intros",
    "remove_outros": "default_remove_outros",
    "retention_limit": "default_retention_limit",
    "retention_days": "default_retention_days",
    "manual_retention_days": "default_manual_retention_days",
    "ai_rewrite_description": "default_ai_rewrite_description",
    "ai_audio_summary": "default_ai_audio_summary",
    "append_title_intro": "default_append_title_intro",
    "watermark_artwork": "default_watermark_artwork",
    "custom_instructions": "default_custom_instructions",
}

BOOLEAN_FIELDS = {
    "remove_ads",
    "remove_promos",
    "remove_intros",
    "remove_outros",
    "ai_rewrite_description",
    "ai_audio_summary",
    "append_title_intro",
    "watermark_artwork",
}

FALLBACKS = {
    "remove_ads": True,
    "remove_promos": True,
    "remove_intros": False,
    "remove_outros": False,
    "retention_limit": 1,
    "retention_days": 30,
    "manual_retention_days": 14,
    "ai_rewrite_description": False,
    "ai_audio_summary": False,
    "append_title_intro": False,
    "watermark_artwork": False,
    "custom_instructions": None,
}


def resolve_subscription_row(
    row: Mapping[str, Any],
    global_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return model-ready data containing effective values and stored overrides."""
    data = dict(row)
    overrides = {field: data.get(field) for field in GLOBAL_FIELDS}
    overrides["append_summary"] = data.get("append_summary")

    for group, fields in GROUP_FIELDS.items():
        if not bool(data.get(f"inherit_{group}", 0)):
            continue
        for field in fields:
            value = global_settings.get(GLOBAL_FIELDS[field], FALLBACKS[field])
            if value is None and field != "custom_instructions":
                value = FALLBACKS[field]
            data[field] = bool(value) if field in BOOLEAN_FIELDS else value

    # append_summary is a legacy umbrella flag. Inherited features use only the
    # three current, independently named global feature defaults.
    if bool(data.get("inherit_default_features", 0)):
        data["append_summary"] = False

    data["setting_overrides"] = overrides
    return data
