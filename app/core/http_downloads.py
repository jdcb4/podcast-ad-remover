"""Bounded redirects; restricted mode connects only to the validated address.

HTTPX's documented sni_hostname extension preserves TLS hostname verification
when connecting to a pinned IP: https://www.python-httpx.org/advanced/extensions/
"""
import asyncio
import ipaddress
import socket
import sqlite3
from contextlib import contextmanager, asynccontextmanager
from urllib.parse import urljoin

import httpx

from app.core.config import settings
from app.core.url_utils import validate_http_url

MAX_REDIRECTS = 8


def max_download_redirects() -> int:
    """Snapshot the shared RSS/audio/artwork limit once per download."""
    from app.infra.database import get_db_connection
    try:
        with get_db_connection() as conn:
            row = conn.execute('SELECT download_max_redirects FROM app_settings WHERE id = 1').fetchone()
        value = row[0] if row else None
        return max(0, min(50, int(value))) if value is not None else MAX_REDIRECTS
    except (sqlite3.Error, OSError, TypeError, ValueError, OverflowError):
        # Bootstrap/older databases and invalid stored values retain the stable default.
        return MAX_REDIRECTS


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
    visited = set()
    for hop in range(redirect_limit + 1):
        if url in visited:
            raise ValueError('Redirect loop detected')
        visited.add(url)
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
    redirect_limit = await asyncio.to_thread(max_download_redirects)
    visited = set()
    for hop in range(redirect_limit + 1):
        if url in visited:
            raise ValueError('Redirect loop detected')
        visited.add(url)
        target, headers, extensions = await asyncio.to_thread(request_target, url)
        async with client.stream('GET', target, headers=headers, extensions=extensions, follow_redirects=False, timeout=timeout) as response:
            if not response.is_redirect:
                yield response
                return
            location = response.headers.get('location')
            if not location or hop == redirect_limit:
                raise ValueError('Invalid redirect or too many redirects')
            url = urljoin(url, location)
