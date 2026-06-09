import pytest

from app.core.url_utils import is_audio_content_type, validate_http_url


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
