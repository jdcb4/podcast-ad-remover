import socket

import httpx
import pytest

from app.core.config import settings
from app.core.http_downloads import stream_get, async_stream_get


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
