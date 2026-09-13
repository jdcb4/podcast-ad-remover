"""Bounded redirects; restricted mode connects only to the validated address.

HTTPX's documented sni_hostname extension preserves TLS hostname verification
when connecting to a pinned IP: https://www.python-httpx.org/advanced/extensions/
"""
import asyncio
import ipaddress
import socket
from contextlib import contextmanager, asynccontextmanager
from urllib.parse import urljoin

import httpx

from app.core.config import settings
from app.core.url_utils import validate_http_url

def max_download_redirects() -> int:
    """Return the configured download redirect cap."""
    value = getattr(settings, 'MAX_DOWNLOAD_REDIRECTS', 5)
    try:
        from app.core.utils import get_global_settings

        global_settings = get_global_settings()
        value = global_settings.get('download_max_redirects', value)
    except Exception:
        pass

    try:
        return max(0, min(50, int(value)))
    except (TypeError, ValueError):
        return 5


def request_target(url: str):
    validate_http_url(url, allow_private=True)
    parsed = httpx.URL(url)
    if settings.ALLOW_PRIVATE_FEEDS:
        return url, {}, {}
    if parsed.username or parsed.password:
        raise ValueError('URL credentials are not supported in restricted network mode')
    addresses = socket.getaddrinfo(parsed.host, parsed.port or (443 if parsed.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    ips = list(dict.fromkeys(address[4][0] for address in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError('Private or non-public network target is not allowed')
    return parsed.copy_with(host=ips[0]), {'Host': parsed.netloc.decode('ascii')}, {'sni_hostname': parsed.host}


@contextmanager
def stream_get(client, url: str, *, timeout=30.0):
    redirect_limit = max_download_redirects()
    for hop in range(redirect_limit + 1):
        target, headers, extensions = request_target(url)
        with client.stream('GET', target, headers=headers, extensions=extensions, follow_redirects=False, timeout=timeout) as response:
            if not response.is_redirect:
                yield response
                return
            location = response.headers.get('location')
            if not location or hop == redirect_limit:
                raise ValueError('Invalid redirect or too many redirects')
            url = urljoin(url, location)


@asynccontextmanager
async def async_stream_get(client, url: str, *, timeout=300.0):
    redirect_limit = max_download_redirects()
    for hop in range(redirect_limit + 1):
        target, headers, extensions = await asyncio.to_thread(request_target, url)
        async with client.stream('GET', target, headers=headers, extensions=extensions, follow_redirects=False, timeout=timeout) as response:
            if not response.is_redirect:
                yield response
                return
            location = response.headers.get('location')
            if not location or hop == redirect_limit:
                raise ValueError('Invalid redirect or too many redirects')
            url = urljoin(url, location)
