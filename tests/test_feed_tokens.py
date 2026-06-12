from app.infra.database import init_db
from app.infra.repository import FeedTokenRepository


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
