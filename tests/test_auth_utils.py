from types import SimpleNamespace
from pathlib import Path
import re

from app.core.config import settings
from app.web.auth_utils import (
    get_client_ip,
    get_request_base_origin,
    hash_password,
    is_bcrypt_hash,
    is_ip_allowed,
    is_same_origin_request,
    verify_feed_password,
)


def make_request(headers=None, host="203.0.113.10", method="GET", scheme="http", netloc=None):
    headers = headers or {}
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=host),
        method=method,
        url=SimpleNamespace(scheme=scheme, netloc=netloc or headers.get("host") or host),
    )


def test_client_ip_ignores_forwarded_headers_by_default(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)

    request = make_request(
        headers={
            "X-Forwarded-For": "198.51.100.1, 10.0.0.2",
            "X-Real-IP": "198.51.100.2",
            "CF-Connecting-IP": "198.51.100.3",
        },
        host="203.0.113.10",
    )

    assert get_client_ip(request) == "203.0.113.10"


def test_client_ip_uses_forwarded_headers_when_proxy_headers_are_trusted(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)

    request = make_request(
        headers={"X-Forwarded-For": "198.51.100.1, 10.0.0.2"},
        host="203.0.113.10",
    )

    assert get_client_ip(request) == "198.51.100.1"


def test_ip_allowlist_uses_effective_client_ip(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    request = make_request(headers={"X-Forwarded-For": "198.51.100.1"}, host="203.0.113.10")

    assert is_ip_allowed(get_client_ip(request), "198.51.100.1") is False
    assert is_ip_allowed(get_client_ip(request), "203.0.113.10") is True


def test_ip_allowlist_supports_cidr_ranges():
    assert is_ip_allowed("192.168.1.42", "10.0.0.5, 192.168.1.0/24") is True
    assert is_ip_allowed("192.168.2.42", "10.0.0.5, 192.168.1.0/24") is False


def test_ip_allowlist_keeps_exact_match_fallback_for_unparseable_values():
    assert is_ip_allowed("unknown", "127.0.0.1, unknown") is True
    assert is_ip_allowed("unknown", "127.0.0.1") is False


def test_feed_password_verifier_accepts_bcrypt_and_legacy_plaintext():
    hashed_password = hash_password("feed-secret")

    assert is_bcrypt_hash(hashed_password) is True
    assert verify_feed_password("feed-secret", hashed_password) is True
    assert verify_feed_password("wrong", hashed_password) is False
    assert verify_feed_password("legacy-secret", "legacy-secret") is True
    assert verify_feed_password("wrong", "legacy-secret") is False
    assert verify_feed_password("anything", None) is False


def test_same_origin_allows_safe_and_headerless_unsafe_requests(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)

    assert is_same_origin_request(make_request(method="GET", headers={"origin": "https://evil.example"})) is True
    assert is_same_origin_request(make_request(method="POST", headers={"host": "app.local"})) is True


def test_same_origin_blocks_mismatched_origin(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)

    request = make_request(
        method="POST",
        headers={"host": "app.local", "origin": "https://evil.example"},
        scheme="https",
    )

    assert is_same_origin_request(request) is False


def test_same_origin_allows_configured_public_url(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)

    request = make_request(
        method="POST",
        headers={"host": "127.0.0.1:8000", "origin": "https://podcasts.example.com"},
    )

    assert is_same_origin_request(request, "https://podcasts.example.com") is True


def test_same_origin_uses_forwarded_host_only_when_trusted(monkeypatch):
    request = make_request(
        method="POST",
        headers={
            "host": "127.0.0.1:8000",
            "x-forwarded-proto": "https",
            "x-forwarded-host": "podcasts.example.com",
            "origin": "https://podcasts.example.com",
        },
    )

    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    assert get_request_base_origin(request) == "http://127.0.0.1:8000"
    assert is_same_origin_request(request) is False

    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    assert get_request_base_origin(request) == "https://podcasts.example.com"
    assert is_same_origin_request(request) is True


def test_dashboard_password_is_not_stored_in_session():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")

    assert 'session["user_pass"] =' not in router_source


def test_new_standalone_feed_passwords_are_hashed_before_storage():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")

    assert "hashed_feed_password = hash_password(feed_auth_password) if feed_auth_password else None" in router_source


def test_sensitive_admin_routes_have_route_level_admin_dependency():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")
    sensitive_handlers = [
        "update_system_settings",
        "update_ai_settings",
        "test_ai_connection",
        "refresh_models",
        "save_prompts",
        "reset_prompts",
        "cancel_episode",
        "retry_episode",
        "revoke_feed_token",
        "update_global_subscription_settings",
        "delete_user_post",
        "approve_access_request",
        "deny_access_request",
        "update_user_username",
        "update_request_username",
    ]

    for handler in sensitive_handlers:
        match = re.search(rf"async def {handler}\((.*?)\):", router_source, re.S)
        assert match, f"{handler} not found"
        assert "Depends(require_admin)" in match.group(1), f"{handler} lacks route-level admin dependency"


def test_management_api_routes_have_route_level_auth_dependency():
    api_source = Path("app/api/subscriptions.py").read_text(encoding="utf-8")
    web_source = Path("app/web/router.py").read_text(encoding="utf-8")

    api_handlers = [
        "list_subscriptions",
        "create_subscription",
        "delete_subscription",
        "delete_episode",
        "check_subscription_updates",
        "process_episode",
        "cancel_episode",
        "search_podcasts",
        "track_listen",
    ]
    web_handlers = [
        "api_reprocess_episode",
        "api_ignore_episode",
        "manual_download_episode",
        "add_subscription",
        "update_settings",
    ]

    for handler in api_handlers:
        match = re.search(rf"async def {handler}\((.*?)\):", api_source, re.S)
        assert match, f"{handler} not found"
        assert "Depends(require_auth)" in match.group(1), f"{handler} lacks route-level auth dependency"

    for handler in web_handlers:
        match = re.search(rf"async def {handler}\((.*?)\):", web_source, re.S)
        assert match, f"{handler} not found"
        assert "Depends(require_auth)" in match.group(1), f"{handler} lacks route-level auth dependency"


def test_user_management_delete_uses_post_route():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")
    users_template = Path("app/web/templates/admin/users.html").read_text(encoding="utf-8")

    assert '@router.post("/admin/users/{user_id}/delete")' in router_source
    assert 'action="/admin/users/{{ active_user.id }}/delete"' in users_template
    assert 'action="/admin/users/delete/' not in users_template


def test_login_page_offers_public_subscribe_link_when_enabled():
    template_source = Path("app/web/templates/login.html").read_text(encoding="utf-8")
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")

    assert "public_subscribe_page_enabled" in template_source
    assert 'href="/subscribe"' in template_source
    assert "Browse and subscribe here" in template_source
    assert "SELECT auth_enabled, initial_password, public_subscribe_page_enabled" in router_source
    assert '"public_subscribe_page_enabled": settings' in router_source


def test_protected_feed_links_warn_that_tokens_are_bearer_secrets():
    feed_access_template = Path("app/web/templates/admin/feed_access.html").read_text(encoding="utf-8")
    index_template = Path("app/web/templates/index.html").read_text(encoding="utf-8")
    episodes_template = Path("app/web/templates/episodes.html").read_text(encoding="utf-8")
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")

    for template_source in [feed_access_template, index_template, episodes_template]:
        assert "bearer secret" in template_source
        assert "until the token is revoked" in template_source

    assert '"settings": global_settings' in router_source
