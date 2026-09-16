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


@pytest.mark.parametrize('hops', [8, 9])
def test_sync_redirect_boundary(restricted, hops):
    visited = []
    def handler(request):
        hop = int(request.url.path.strip('/'))
        visited.append(hop)
        return httpx.Response(302, headers={'Location': f'/{hop+1}'}) if hop < hops else httpx.Response(200, content=b'audio')
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if hops == 8:
            with stream_get(client, 'https://public.test/0') as response:
                assert response.read() == b'audio'
        else:
            with pytest.raises(ValueError, match='too many redirects'):
                with stream_get(client, 'https://public.test/0'):
                    pytest.fail('Ninth redirect accepted')
    assert visited == list(range(9))


@pytest.mark.asyncio
@pytest.mark.parametrize('hops', [8, 9])
async def test_async_redirect_boundary(restricted, hops):
    visited = []
    def handler(request):
        hop = int(request.url.path.strip('/'))
        visited.append(hop)
        return httpx.Response(302, headers={'Location': f'/{hop+1}'}) if hop < hops else httpx.Response(200, content=b'audio')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if hops == 8:
            async with async_stream_get(client, 'https://public.test/0') as response:
                assert await response.aread() == b'audio'
        else:
            with pytest.raises(ValueError, match='too many redirects'):
                async with async_stream_get(client, 'https://public.test/0'):
                    pytest.fail('Ninth redirect accepted')
    assert visited == list(range(9))


@pytest.mark.asyncio
async def test_redirect_loops_close_both_transports(restricted):
    responses = []
    def handler(request):
        response = httpx.Response(302, headers={'Location': '/loop'})
        responses.append(response)
        return response
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='loop'):
            with stream_get(client, 'https://public.test/loop'):
                pytest.fail('Loop accepted')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='loop'):
            async with async_stream_get(client, 'https://public.test/loop'):
                pytest.fail('Loop accepted')
    assert len(responses) == 2 and all(response.is_closed for response in responses)


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
