from datetime import datetime

import pytest

from app.api import subscriptions
from app.core.models import Subscription, SubscriptionCreate
from app.web import router as web_router


class FakeSubscriptionRepository:
    def __init__(self):
        self.created = None

    def get_by_url(self, url):
        return None

    def create(self, sub, title, slug, image_url=None, description=None, retention_limit=1, owner_user_id=None):
        self.created = {
            "sub": sub,
            "title": title,
            "slug": slug,
            "image_url": image_url,
            "description": description,
            "retention_limit": retention_limit,
            "owner_user_id": owner_user_id,
        }
        return Subscription(
            id=1,
            feed_url=sub.feed_url,
            title=title,
            slug=slug,
            image_url=image_url,
            description=description,
            is_active=True,
            created_at=datetime(2026, 1, 1),
        )


class FakeProcessor:
    def __init__(self):
        self.checked = None

    async def check_feeds(self, subscription_id=None, limit=5):
        self.checked = {"subscription_id": subscription_id, "limit": limit}


@pytest.mark.asyncio
async def test_create_subscription_accepts_parse_feed_description(monkeypatch):
    fake_repo = FakeSubscriptionRepository()
    fake_processor = FakeProcessor()

    monkeypatch.setattr(subscriptions, "repo", fake_repo)
    monkeypatch.setattr(
        subscriptions.FeedManager,
        "parse_feed",
        staticmethod(lambda url: ("Example", "example", "https://example.com/art.jpg", "Feed notes")),
    )
    monkeypatch.setattr(subscriptions, "get_processor", lambda: fake_processor)

    created = await subscriptions.create_subscription(
        SubscriptionCreate(feed_url="https://example.com/feed.xml"),
        initial_count=3,
        user=object(),
    )

    assert created.description == "Feed notes"
    assert fake_repo.created["description"] == "Feed notes"
    assert fake_processor.checked == {"subscription_id": 1, "limit": 3}


@pytest.mark.asyncio
async def test_web_add_subscription_validates_feed_url_before_placeholder_insert(monkeypatch):
    fake_repo = FakeSubscriptionRepository()
    rendered = {}

    def fake_render_index(request, error=None):
        rendered["error"] = error
        return {"error": error}

    monkeypatch.setattr(web_router, "sub_repo", fake_repo)
    monkeypatch.setattr(web_router, "_render_index", fake_render_index)

    response = await web_router.add_subscription(
        request=object(),
        background_tasks=object(),
        feed_url="file:///etc/passwd",
        initial_count=1,
        user=object(),
    )

    assert response == {"error": "Only HTTP and HTTPS URLs are supported"}
    assert rendered["error"] == "Only HTTP and HTTPS URLs are supported"
    assert fake_repo.created is None
