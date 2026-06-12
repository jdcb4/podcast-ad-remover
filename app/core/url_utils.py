import ipaddress
import socket
from urllib.parse import urlparse


def validate_http_url(url: str, allow_private: bool = True) -> str:
    """Validate an HTTP(S) URL and optionally reject private network targets."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must include a host")

    if allow_private:
        return url

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL host: {parsed.hostname}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
            raise ValueError("Private, loopback, link-local, and multicast URLs are not allowed")

    return url


def validate_redirect_target(original_url: str, final_url: str, allow_private: bool = True) -> str:
    """Validate a final response URL after redirects."""
    try:
        validate_http_url(final_url, allow_private=allow_private)
    except ValueError as exc:
        raise ValueError(f"Redirect target for {original_url} is not allowed: {final_url}") from exc
    return final_url


def is_audio_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    media_type = content_type.split(";", 1)[0].strip().lower()
    return (
        media_type.startswith("audio/")
        or media_type in {"application/octet-stream", "binary/octet-stream"}
    )
