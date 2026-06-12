import socket

import pytest

from app.core.url_utils import is_audio_content_type, validate_http_url, validate_redirect_target


def test_validate_http_url_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        validate_http_url("file:///etc/passwd")


def test_validate_http_url_allows_http_url_without_private_check():
    assert validate_http_url("https://example.com/feed.xml") == "https://example.com/feed.xml"


def test_audio_content_type_accepts_audio_and_octet_stream():
    assert is_audio_content_type("audio/mpeg") is True
    assert is_audio_content_type("audio/mp4; charset=binary") is True
    assert is_audio_content_type("application/octet-stream") is True


def test_audio_content_type_rejects_html():
    assert is_audio_content_type("text/html") is False


def test_redirect_target_rejects_private_final_url_when_hardened(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="Redirect target"):
        validate_redirect_target(
            "https://example.com/feed.xml",
            "http://internal.local/feed.xml",
            allow_private=False,
        )


def test_redirect_target_allows_public_final_url_when_hardened(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert (
        validate_redirect_target(
            "https://example.com/feed.xml",
            "https://cdn.example.com/feed.xml",
            allow_private=False,
        )
        == "https://cdn.example.com/feed.xml"
    )
