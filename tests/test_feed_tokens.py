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
