from app.core.config import settings
from app.core import utils


class DummyRequest:
    base_url = "http://podcast-host.local:8000/"


def test_base_url_prefers_configured_external_url():
    assert utils.get_app_base_url({"app_external_url": "http://example.test/"}, DummyRequest()) == "http://example.test"


def test_base_url_prefers_request_when_external_url_missing(monkeypatch):
    monkeypatch.setattr(utils, "is_running_in_container", lambda: True)

    assert utils.get_app_base_url({"app_external_url": ""}, DummyRequest()) == "http://podcast-host.local:8000"


def test_base_url_uses_base_url_in_container_without_request(monkeypatch):
    original_base_url = settings.BASE_URL
    monkeypatch.setattr(utils, "is_running_in_container", lambda: True)
    settings.BASE_URL = "http://configured-host:8000"
    try:
        assert utils.get_app_base_url({"app_external_url": None}) == "http://configured-host:8000"
    finally:
        settings.BASE_URL = original_base_url
