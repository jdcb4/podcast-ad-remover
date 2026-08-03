from app.core.models import SubscriptionCreate
from app.core.subscription_settings import resolve_subscription_row
from app.infra.database import get_db_connection, init_db
from app.infra.repository import SubscriptionRepository


def _global_settings(**updates):
    settings = {
        "default_remove_ads": 1,
        "default_remove_promos": 0,
        "default_remove_intros": 1,
        "default_remove_outros": 0,
        "default_retention_limit": 5,
        "default_retention_days": 45,
        "default_manual_retention_days": 21,
        "default_ai_rewrite_description": 1,
        "default_ai_audio_summary": 0,
        "default_append_title_intro": 1,
        "default_custom_instructions": "Use the global instructions",
    }
    settings.update(updates)
    return settings


def test_resolver_uses_four_group_flags_without_overwriting_stored_values():
    row = {
        "inherit_content_removal": 1,
        "inherit_retention": 1,
        "inherit_default_features": 1,
        "inherit_custom_instructions": 1,
        "remove_ads": 0,
        "remove_promos": 1,
        "remove_intros": 0,
        "remove_outros": 1,
        "retention_limit": 1,
        "retention_days": 30,
        "manual_retention_days": 14,
        "ai_rewrite_description": 0,
        "ai_audio_summary": 1,
        "append_title_intro": 0,
        "append_summary": 1,
        "custom_instructions": "Stored override",
    }

    resolved = resolve_subscription_row(row, _global_settings())

    assert resolved["remove_ads"] is True
    assert resolved["remove_promos"] is False
    assert resolved["retention_limit"] == 5
    assert resolved["manual_retention_days"] == 21
    assert resolved["ai_rewrite_description"] is True
    assert resolved["ai_audio_summary"] is False
    assert resolved["append_title_intro"] is True
    assert resolved["append_summary"] is False
    assert resolved["custom_instructions"] == "Use the global instructions"
    assert resolved["setting_overrides"]["remove_ads"] == 0
    assert resolved["setting_overrides"]["retention_limit"] == 1
    assert resolved["setting_overrides"]["custom_instructions"] == "Stored override"
    assert resolved["setting_overrides"]["append_summary"] == 1


def test_new_subscription_inherits_all_groups_and_restores_stored_overrides(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET default_remove_ads = 0,
                default_retention_limit = 10,
                default_custom_instructions = 'Global rule'
            WHERE id = 1
            """
        )
        conn.commit()

    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/inherited.xml"),
        "Inherited",
        "inherited",
        retention_limit=3,
    )

    assert sub.inherit_content_removal is True
    assert sub.inherit_retention is True
    assert sub.inherit_default_features is True
    assert sub.inherit_custom_instructions is True
    assert sub.remove_ads is False
    assert sub.retention_limit == 10
    assert sub.custom_instructions == "Global rule"
    assert sub.setting_overrides["retention_limit"] == 3

    repo.update_settings(
        sub.id,
        False,
        True,
        False,
        False,
        "Podcast rule",
        False,
        False,
        False,
        False,
        retention_days=30,
        manual_retention_days=14,
        retention_limit=3,
        inherit_content_removal=False,
        inherit_retention=False,
        inherit_default_features=False,
        inherit_custom_instructions=False,
    )
    explicit = repo.get_by_id(sub.id)

    assert explicit.remove_ads is False
    assert explicit.retention_limit == 3
    assert explicit.custom_instructions == "Podcast rule"


def test_settings_form_displays_effective_inherited_values_and_keeps_overrides_reversible():
    template = open("app/web/templates/episodes.html", encoding="utf-8").read()
    script = open("app/web/static/js/settings-inheritance.js", encoding="utf-8").read()

    assert "displayed_retention_limit = subscription.retention_limit if subscription.inherit_retention" in template
    assert 'data-effective-value="{{ settings.default_retention_limit' in template
    assert 'data-override-value="{{ stored.get(\'retention_limit\')' in template
    assert "storeOverride(control)" in script
    assert "showValue(control, 'effective')" in script
    assert "showValue(control, 'override')" in script
