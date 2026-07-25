import json
from types import SimpleNamespace

import pytest

from app.web import router as web_router


class FakeLibraryRepository:
    def __init__(self, initially_member=False):
        self.member = initially_member

    def get_by_id(self, subscription_id):
        if subscription_id != 7:
            return None
        return SimpleNamespace(id=7)

    def is_in_user_library(self, user_id, subscription_id):
        return self.member

    def add_to_user_library(self, user_id, subscription_id):
        changed = not self.member
        self.member = True
        return changed

    def remove_from_user_library(self, user_id, subscription_id):
        changed = self.member
        self.member = False
        return changed

    def count_user_library_members(self, subscription_id):
        return 1 if self.member else 0


@pytest.mark.asyncio
async def test_library_membership_can_return_json_without_navigation(monkeypatch):
    repository = FakeLibraryRepository()
    monkeypatch.setattr(web_router, "sub_repo", repository)
    request = SimpleNamespace(headers={"accept": "application/json"})
    user = SimpleNamespace(id=42)

    response = await web_router.update_user_library_membership(
        request=request,
        id=7,
        action="add",
        redirect_to="/?view=library",
        user=user,
    )

    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload == {
        "status": "updated",
        "subscription_id": 7,
        "changed": True,
        "in_user_library": True,
        "user_library_count": 1,
        "message": "Podcast added to My Podcasts",
    }


@pytest.mark.asyncio
async def test_library_membership_keeps_redirect_fallback(monkeypatch):
    repository = FakeLibraryRepository(initially_member=True)
    monkeypatch.setattr(web_router, "sub_repo", repository)
    request = SimpleNamespace(headers={"accept": "text/html"})
    user = SimpleNamespace(id=42)

    response = await web_router.update_user_library_membership(
        request=request,
        id=7,
        action="remove",
        redirect_to="/?view=mine",
        user=user,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/?view=mine&success=")
    assert repository.member is False


def test_dashboard_membership_script_is_progressively_enhanced():
    template = open("app/web/templates/index.html", encoding="utf-8").read()
    script = open("app/web/static/js/dashboard-library.js", encoding="utf-8").read()

    assert 'class="shrink-0 library-membership-form"' in template
    assert 'src="/static/js/dashboard-library.js"' in template
    assert "event.preventDefault()" in script
    assert "library-membership-changed" in script
