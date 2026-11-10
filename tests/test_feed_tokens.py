from types import SimpleNamespace

import pytest

from app.infra.database import get_db_connection, init_db
from app.infra.repository import FeedTokenRepository
from app.web.router import delete_user


def test_feed_token_validates_and_revokes(isolated_data_dir):
    init_db()
    repo = FeedTokenRepository()

    token = repo.create(name="pytest")

    assert token
    assert repo.validate(token) is True
    assert repo.validate("not-the-token") is False

    repo.revoke(token)

    assert repo.validate(token) is False


def test_feed_tokens_can_be_listed_and_revoked_by_id(isolated_data_dir):
    init_db()
    repo = FeedTokenRepository()

    token = repo.create(name="Podcast client")
    active_tokens = repo.list_active()

    assert len(active_tokens) == 1
    assert active_tokens[0]["name"] == "Podcast client"
    assert "token_hash" not in active_tokens[0]
    assert repo.revoke_by_id(active_tokens[0]["id"]) is True
    assert repo.revoke_by_id(active_tokens[0]["id"]) is False
    assert repo.validate(token) is False
    assert repo.list_active() == []


def test_feed_token_stops_working_when_its_user_is_gone(isolated_data_dir):
    init_db()
    repo = FeedTokenRepository()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (id,username,password_hash) VALUES (7,'member','unused')")
        conn.commit()
    linked = repo.create(user_id=7, name="Member app")
    shared = repo.create(name="Shared app")
    assert repo.validate(linked) is True

    # Reproduces rows left by earlier versions, where deletion did not touch tokens.
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id=7")
        conn.commit()

    assert repo.validate(linked) is False
    assert repo.validate(shared) is True
    assert [token["name"] for token in repo.list_active()] == ["Shared app"]


@pytest.mark.asyncio
async def test_deleting_a_user_revokes_their_feed_tokens(isolated_data_dir):
    init_db()
    repo = FeedTokenRepository()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (id,username,password_hash) VALUES (7,'member','unused'), (8,'other','unused')")
        conn.commit()
    deleted = repo.create(user_id=7, name="Member app")
    kept = repo.create(user_id=8, name="Other app")

    await delete_user(7, request=None, user=SimpleNamespace(id=1, is_admin=True))

    with get_db_connection() as conn:
        revoked = conn.execute("SELECT revoked_at FROM feed_tokens WHERE user_id=7").fetchone()[0]
    assert revoked is not None
    assert repo.validate(deleted) is False
    assert repo.validate(kept) is True
