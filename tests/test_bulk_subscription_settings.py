from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.core.models import SubscriptionCreate
from app.infra.database import get_db_connection, init_db
from app.infra.repository import SubscriptionRepository
from app.web.router import bulk_delete_subscriptions, bulk_update_subscription_settings


async def _bulk(user, ids, **overrides):
    values = {
        "background_tasks": BackgroundTasks(),
        "subscription_ids": ids,
        "content_mode": "unchanged",
        "remove_ads": False,
        "remove_promos": False,
        "remove_intros": False,
        "remove_outros": False,
        "retention_mode": "unchanged",
        "retention_limit": 1,
        "retention_days": 30,
        "manual_retention_days": 14,
        "features_mode": "unchanged",
        "ai_rewrite_description": False,
        "ai_audio_summary": False,
        "append_title_intro": False,
        "watermark_artwork": False,
        "instructions_mode": "unchanged",
        "custom_instructions": "",
        "owner_mode": "unchanged",
        "owner_user_id": "",
        "user": user,
    }
    values.update(overrides)
    return await bulk_update_subscription_settings(**values)


def _create_user(conn, username, is_admin=False):
    return conn.execute(
        "INSERT INTO users (username, password_hash, is_admin) VALUES (?, 'hash', ?)",
        (username, int(is_admin)),
    ).lastrowid


@pytest.mark.asyncio
async def test_bulk_update_applies_group_modes_atomically(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        owner_id = _create_user(conn, "owner")
        conn.commit()
    first = repo.create(
        SubscriptionCreate(feed_url="https://example.com/one.xml"),
        "One",
        "one",
        owner_user_id=owner_id,
    )
    second = repo.create(
        SubscriptionCreate(feed_url="https://example.com/two.xml"),
        "Two",
        "two",
        owner_user_id=owner_id,
    )

    response = await _bulk(
        SimpleNamespace(id=owner_id, is_admin=False),
        [first.id, second.id],
        content_mode="override",
        remove_ads=True,
        retention_mode="inherit",
        features_mode="override",
        watermark_artwork=True,
        instructions_mode="override",
        custom_instructions="Remove local sponsor reads",
    )

    assert response.status_code == 303
    for subscription_id in (first.id, second.id):
        sub = repo.get_by_id(subscription_id)
        assert sub.inherit_content_removal is False
        assert sub.remove_ads is True
        assert sub.remove_promos is False
        assert sub.inherit_retention is True
        assert sub.inherit_default_features is False
        assert sub.watermark_artwork is True
        assert sub.inherit_custom_instructions is False
        assert sub.custom_instructions == "Remove local sponsor reads"


@pytest.mark.asyncio
async def test_bulk_update_rejects_mixed_unauthorized_selection_without_changes(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        first_owner = _create_user(conn, "first")
        second_owner = _create_user(conn, "second")
        conn.commit()
    first = repo.create(
        SubscriptionCreate(feed_url="https://example.com/first.xml"),
        "First",
        "first",
        owner_user_id=first_owner,
    )
    second = repo.create(
        SubscriptionCreate(feed_url="https://example.com/second.xml"),
        "Second",
        "second",
        owner_user_id=second_owner,
    )

    with pytest.raises(HTTPException) as exc:
        await _bulk(
            SimpleNamespace(id=first_owner, is_admin=False),
            [first.id, second.id],
            content_mode="override",
            remove_ads=False,
        )

    assert exc.value.status_code == 403
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, inherit_content_removal FROM subscriptions ORDER BY id"
        ).fetchall()
    assert [(row["id"], row["inherit_content_removal"]) for row in rows] == [
        (first.id, 1),
        (second.id, 1),
    ]


@pytest.mark.asyncio
async def test_bulk_owner_assignment_is_admin_only_and_adds_membership(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    with get_db_connection() as conn:
        admin_id = _create_user(conn, "admin", is_admin=True)
        owner_id = _create_user(conn, "new-owner")
        conn.commit()
    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/admin.xml"),
        "Admin",
        "admin",
    )

    await _bulk(
        SimpleNamespace(id=admin_id, is_admin=True),
        [sub.id],
        owner_mode="set",
        owner_user_id=str(owner_id),
    )

    updated = repo.get_by_id(sub.id)
    assert updated.owner_user_id == owner_id
    assert repo.is_in_user_library(owner_id, sub.id) is True


def test_dashboard_contains_compact_table_and_bulk_editor():
    template = open("app/web/templates/index.html", encoding="utf-8").read()
    script = open("app/web/static/js/bulk-subscriptions.js", encoding="utf-8").read()

    assert 'id="podcast-table-container"' in template
    assert 'id="bulk-settings-form"' in template
    assert 'name="subscription_ids"' in template
    assert "appConfirm" in script


@pytest.mark.asyncio
async def test_bulk_delete_requires_explicit_confirmation_before_scheduling(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    sub = repo.create(
        SubscriptionCreate(feed_url="https://example.com/delete.xml"),
        "Delete Me",
        "delete-me",
    )
    tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as exc:
        await bulk_delete_subscriptions(
            background_tasks=tasks,
            subscription_ids=[sub.id],
            delete_confirmation="",
            admin_user=SimpleNamespace(id=1, is_admin=True),
        )

    assert exc.value.status_code == 400
    assert repo.get_by_id(sub.id) is not None
    assert len(tasks.tasks) == 0


@pytest.mark.asyncio
async def test_bulk_delete_uses_durable_subscription_cleanup(isolated_data_dir):
    init_db()
    repo = SubscriptionRepository()
    first = repo.create(
        SubscriptionCreate(feed_url="https://example.com/delete-one.xml"),
        "Delete One",
        "delete-one",
    )
    second = repo.create(
        SubscriptionCreate(feed_url="https://example.com/delete-two.xml"),
        "Delete Two",
        "delete-two",
    )
    tasks = BackgroundTasks()

    response = await bulk_delete_subscriptions(
        background_tasks=tasks,
        subscription_ids=[first.id, second.id, first.id],
        delete_confirmation="delete",
        admin_user=SimpleNamespace(id=1, is_admin=True),
    )

    assert response.status_code == 303
    assert len(tasks.tasks) == 1
    assert repo.get_by_id(first.id).deletion_status == "pending"
    assert repo.get_by_id(second.id).deletion_status == "pending"

    await tasks()

    assert repo.get_by_id(first.id) is None
    assert repo.get_by_id(second.id) is None


def test_bulk_delete_control_is_admin_only_and_warns_about_file_removal():
    template = open("app/web/templates/index.html", encoding="utf-8").read()
    script = open("app/web/static/js/bulk-subscriptions.js", encoding="utf-8").read()

    assert '{% if user.is_admin %}' in template
    assert 'formaction="/subscriptions/bulk-delete"' in template
    assert 'data-bulk-action="delete"' in template
    assert "downloaded audio, processed files, transcripts, reports" in script
    assert "{ danger: true }" in script
    assert "confirmation.value = 'delete'" in script


def test_select_all_uses_live_checkbox_after_dashboard_view_replacement():
    script = open("app/web/static/js/bulk-subscriptions.js", encoding="utf-8").read()

    assert "const selectAll = () => document.getElementById('select-all-podcasts');" in script
    assert "event.target.matches('#select-all-podcasts')" in script
    assert "checkbox.checked = event.target.checked" in script
    assert "const selectAll = document.getElementById('select-all-podcasts');" not in script
