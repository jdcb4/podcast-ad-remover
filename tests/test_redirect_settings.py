import sqlite3

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.core.http_downloads import stream_get, async_stream_get, max_download_redirects
from app.infra import database
from app.infra.database import get_db_connection, init_db


@pytest.fixture(autouse=True)
def redirect_database(isolated_data_dir):
    init_db()


@pytest.mark.asyncio
@pytest.mark.parametrize('limit', [0, 1, 12, 50])
@pytest.mark.parametrize('excess', [0, 1])
async def test_configurable_boundary_for_both_transports(limit, excess):
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET download_max_redirects=?', (limit,))
        conn.commit()
    seen = []
    def handler(request):
        hop = int(request.url.path.strip('/'))
        seen.append(hop)
        return httpx.Response(302, headers={'Location': f'/{hop+1}'}) if hop < limit + excess else httpx.Response(200)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if excess:
            with pytest.raises(ValueError, match='too many redirects'):
                with stream_get(client, 'https://public.test/0'):
                    pytest.fail('Exceeded limit')
        else:
            with stream_get(client, 'https://public.test/0') as response:
                assert response.status_code == 200
    assert seen == list(range(limit + 1))
    seen.clear()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if excess:
            with pytest.raises(ValueError, match='too many redirects'):
                async with async_stream_get(client, 'https://public.test/0'):
                    pytest.fail('Exceeded limit')
        else:
            async with async_stream_get(client, 'https://public.test/0') as response:
                assert response.status_code == 200
    assert seen == list(range(limit + 1))


@pytest.mark.parametrize('stored,expected', [('invalid', 8), (-1, 0), (100, 50)])
def test_invalid_persisted_limit_is_bounded(stored, expected):
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET download_max_redirects=?', (stored,))
        conn.commit()
    assert max_download_redirects() == expected


def test_settings_zero_round_trip_validation_and_omission():
    from app.web import router as web
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='test-only')
    app.include_router(web.router)
    with TestClient(app) as client:
        for value in [12, 0, 50]:
            response = client.post('/admin/system/update', data={'download_max_redirects': value}, follow_redirects=False)
            assert response.status_code == 303
            page = client.get('/admin/system')
            assert page.status_code == 200
            assert f'value="{value}"' in page.text.split('id="download-max-redirects"')[1].split('>')[0]
        for value in [-1, 51, 'invalid', '1.5']:
            assert client.post('/admin/system/update', data={'download_max_redirects': value}).status_code == 422
        assert client.post('/admin/system/update', data={}, follow_redirects=False).status_code == 303
    with get_db_connection() as conn:
        assert conn.execute('SELECT download_max_redirects FROM app_settings').fetchone()[0] == 50


@pytest.mark.parametrize('contributor_value', [None, 0, 12])
def test_upgrade_backup_and_existing_contributor_values(isolated_data_dir, contributor_value):
    # Recreate the schema immediately before this feature, including a PR #29 variant.
    with get_db_connection() as conn:
        conn.execute('ALTER TABLE app_settings DROP COLUMN download_max_redirects')
        conn.execute('DELETE FROM schema_migrations WHERE version=?', (database.DOWNLOAD_REDIRECT_MIGRATION,))
        if contributor_value is not None:
            conn.execute('ALTER TABLE app_settings ADD COLUMN download_max_redirects INTEGER DEFAULT 5')
            conn.execute('UPDATE app_settings SET download_max_redirects=?', (contributor_value,))
        conn.commit()
    before = set((isolated_data_dir / 'backups').glob('*.db'))
    init_db()
    backups = set((isolated_data_dir / 'backups').glob('*.db')) - before
    assert backups
    with sqlite3.connect(next(iter(backups))) as conn:
        assert conn.execute('SELECT count(*) FROM schema_migrations WHERE version=?', (database.DOWNLOAD_REDIRECT_MIGRATION,)).fetchone()[0] == 0
    init_db()
    with get_db_connection() as conn:
        assert conn.execute('SELECT download_max_redirects FROM app_settings').fetchone()[0] == (8 if contributor_value is None else contributor_value)
        assert conn.execute('SELECT count(*) FROM schema_migrations WHERE version=?', (database.DOWNLOAD_REDIRECT_MIGRATION,)).fetchone()[0] == 1
