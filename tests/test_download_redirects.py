import socket

import httpx
import pytest

from app.core.config import settings
from app.core.http_downloads import stream_get, async_stream_get
from app.infra.database import get_db_connection, init_db


@pytest.fixture
def restricted(monkeypatch):
    monkeypatch.setattr(settings, 'ALLOW_PRIVATE_FEEDS', False)
    def resolve(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1' if host == 'private.test' else '8.8.8.8', 443))]
    monkeypatch.setattr(socket, 'getaddrinfo', resolve)


def test_redirect_is_validated_before_target_request(restricted):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'https://private.test/secret'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='non-public'):
            with stream_get(client, 'https://public.test/feed'):
                pytest.fail('Private redirect accepted')
    assert len(requests) == 1
    assert requests[0].url.host == '8.8.8.8'
    assert requests[0].headers['host'] == 'public.test'
    assert requests[0].extensions['sni_hostname'] == 'public.test'


@pytest.mark.asyncio
async def test_async_download_pins_dns_and_checks_each_redirect(restricted):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'https://private.test/audio'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            async with async_stream_get(client, 'https://public.test/audio'):
                pytest.fail('Private redirect accepted')
    assert len(requests) == 1


def test_download_redirect_limit_comes_from_system_settings(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET download_max_redirects = 7 WHERE id = 1")
        conn.commit()

    seen = []

    def handler(request):
        seen.append(str(request.url))
        if len(seen) <= 6:
            return httpx.Response(302, headers={"Location": f"https://example.test/hop-{len(seen)}"})
        return httpx.Response(200, headers={"Content-Type": "audio/mpeg"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with stream_get(client, "https://example.test/start") as response:
            assert response.status_code == 200

    assert len(seen) == 7


def test_download_redirect_limit_still_rejects_excessive_redirects(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET download_max_redirects = 2 WHERE id = 1")
        conn.commit()

    def handler(request):
        return httpx.Response(302, headers={"Location": "https://example.test/next"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="too many redirects"):
            with stream_get(client, "https://example.test/start"):
                pytest.fail("redirect chain should have been rejected")
