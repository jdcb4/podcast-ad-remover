import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.core import podcast_import as imports
from app.core.models import SubscriptionCreate
from app.core.sources import ResolvedSource
from app.infra.database import init_db, get_db_connection
from app.infra.repository import SubscriptionRepository, ApiTokenRepository
from tests.conftest import make_client, enable_ai_api, auth_header


@pytest.fixture()
def import_db(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        ids = [conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (name, 'hash')).lastrowid for name in ('owner', 'importer')]
        conn.commit()
    return ids


def test_nested_opml_and_conservative_duplicates(import_db):
    owner, user = import_db
    repo = SubscriptionRepository()
    sub = repo.create(SubscriptionCreate(feed_url='https://EXAMPLE.com:443/feed#old'), 'Existing', 'existing', owner_user_id=owner)
    opml = '''<?xml version="1.0"?><opml version="2.0"><body><outline text="Folder">
      <outline text="Old title" xmlUrl="https://example.com/feed"/>
      <outline xmlUrl="https://EXAMPLE.com:443/feed#second"/>
      <outline title="New" xmlUrl="https://example.com/Feed?a=1&amp;b=2"/>
      <outline xmlUrl="file:///etc/passwd"/>
    </outline></body></opml>'''
    rows = imports.preview_import(opml, user)
    assert [r['status'] for r in rows] == ['join', 'duplicate', 'ready', 'invalid']
    assert rows[0]['title'] == 'Existing'
    assert rows[2]['url'].endswith('/Feed?a=1&b=2')
    assert len(repo.get_all()) == 1
    assert not repo.is_in_user_library(user, sub.id)


@pytest.mark.parametrize('content', [
    '<!DOCTYPE opml [<!ENTITY x "boom">]><opml><body>&x;</body></opml>',
    '<opml>', '<rss/>', '<opml><body><outline text="folder"/></body></opml>',
    ' ', 'https://example.com\n' * 101, 'x' * (imports.MAX_BYTES + 1),
], ids=['entity', 'malformed', 'rss', 'empty-opml', 'empty', 'too-many', 'too-large'])
def test_invalid_documents_rejected(content):
    with pytest.raises(ValueError):
        imports.parse_import(content)


@pytest.mark.parametrize('url', ['ftp://example.com', 'https://name:secret@example.com', 'https://example.com:invalid/a', 'https://exa mple.com', 'http:///x'])
def test_invalid_urls(url):
    with pytest.raises(ValueError):
        imports.feed_key(url)


def test_create_join_replay_and_source_identity(import_db, monkeypatch):
    owner, user = import_db
    source = ResolvedSource('rss', None, 'https://example.com/feed', 'Show', 'show', None, 'Description')
    monkeypatch.setattr(imports, 'resolve_source', lambda url: source)
    first = imports.import_feed('https://example.com/feed', owner)
    second = imports.import_feed('https://EXAMPLE.com:443/feed#x', user)
    assert (first['status'], second['status']) == ('added', 'joined')
    assert first['subscription_id'] == second['subscription_id']
    assert imports.import_feed('https://example.com/feed', user)['status'] == 'existing'
    repo = SubscriptionRepository()
    sub = repo.get_by_id(first['subscription_id'])
    assert sub.owner_user_id == owner
    assert sub.inherit_retention and sub.inherit_default_features
    assert len(repo.get_all()) == 1
    # An alias resolving to the canonical URL is reused too.
    assert imports.import_feed('https://example.com/alias', user)['status'] == 'existing'


def test_failed_resolve_is_sanitized(import_db, monkeypatch):
    def fail(url):
        raise RuntimeError('token=private-secret')
    monkeypatch.setattr(imports, 'resolve_source', fail)
    with pytest.raises(ValueError, match='Could not read this feed') as error:
        imports.import_feed('https://example.com/feed', import_db[0])
    assert 'private-secret' not in str(error.value)


def test_api_preview_permissions_partial_import_and_retry(import_db, monkeypatch):
    enable_ai_api()
    client = make_client()
    token = ApiTokenRepository().create('Importer', scopes=['write'], user_id=import_db[1])
    read_token = ApiTokenRepository().create('Reader', scopes=['read'], user_id=import_db[1])
    payload = {'content': 'https://example.com/ok\nhttps://example.com/bad\nhttps://example.com/ok'}
    assert client.post('/api/v1/subscriptions/import', json=payload).status_code == 401
    assert client.post('/api/v1/subscriptions/import', json=payload, headers=auth_header(read_token)).status_code == 403
    def resolve(url):
        if url.endswith('bad'):
            raise ValueError('Unavailable')
        return ResolvedSource('rss', None, url, 'Show', 'show', None, '')
    monkeypatch.setattr(imports, 'resolve_source', lambda url: pytest.fail('Preview must not fetch feeds'))
    response = client.post('/api/v1/subscriptions/import', json=payload, headers=auth_header(token))
    assert response.status_code == 200
    assert response.json()['dry_run'] is True
    assert SubscriptionRepository().get_all() == []
    monkeypatch.setattr(imports, 'resolve_source', resolve)
    payload['dry_run'] = False
    response = client.post('/api/v1/subscriptions/import', json=payload, headers=auth_header(token))
    assert [r['status'] for r in response.json()['items']] == ['added', 'error', 'duplicate']
    response = client.post('/api/v1/subscriptions/import', json=payload, headers=auth_header(token))
    assert [r['status'] for r in response.json()['items']] == ['existing', 'error', 'duplicate']
    assert len(SubscriptionRepository().get_all()) == 1


def test_web_upload_text_and_origin(import_db):
    from app.web.podcast_import import router
    from app.web.auth import require_auth
    from types import SimpleNamespace
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = lambda: SimpleNamespace(id=import_db[1], is_admin=False)
    client = TestClient(app)
    response = client.post('/import/preview', files={'file': ('feeds.opml', b'\xef\xbb\xbf<opml><body><outline xmlUrl="https://example.com/feed"/></body></opml>')})
    assert response.status_code == 200 and response.json()['items'][0]['status'] == 'ready'
    assert client.post('/import/preview', files={'file': ('feeds.txt', b'\xff\xff')}).status_code == 400
    assert client.post('/import/preview', files={'file': ('feeds.txt', b'x' * (imports.MAX_BYTES + 1))}).status_code == 400
    assert client.post('/import/preview', data={'text': 'https://example.com'}, files={'file': ('feeds.txt', b'x')}).status_code == 400
    assert client.post('/import/preview', data={'text': 'https://example.com'}, headers={'Origin': 'https://evil.example'}).status_code == 403


def test_import_browser_selection_failure_retry_and_safe_text(import_db):
    import json
    import shutil
    import subprocess
    from pathlib import Path
    from starlette.middleware.sessions import SessionMiddleware
    from app.web import router as web
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='import-test-only')
    app.include_router(web.router)
    html = TestClient(app).get('/import').text
    script = r'''
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const fs = require('node:fs');
const dom = new JSDOM(JSON.parse(fs.readFileSync(0, 'utf8')), {url:'http://example.test/import',runScripts:'outside-only'});
const w=dom.window, d=w.document;
const calls=[]; let fail=true;
w.fetch=async (url, options) => {
  calls.push([url, options.body.get('feed_url')]);
  if(url.endsWith('preview')) return {ok:true,json:async()=>({items:[
    {url:'https://example.com/a',title:'<img src=x onerror=alert(1)>',status:'ready'},
    {url:'https://example.com/b',title:'Second',status:'ready'},
    {url:'https://example.com/a',title:'Duplicate',status:'duplicate'}
  ]})};
  if(fail) return {ok:false,json:async()=>({detail:'Feed temporarily unavailable'})};
  return {ok:true,json:async()=>({url:'https://example.com/a',title:'Added',status:'added',subscription_id:1})};
};
w.eval(fs.readFileSync('app/web/static/js/podcast-import.js','utf8'));
const tick=()=>new Promise(resolve=>setTimeout(resolve,5));
(async()=>{
 d.querySelector('#import-form').dispatchEvent(new w.Event('submit',{cancelable:true})); await tick();
 assert.equal(d.querySelector('#import-review').hidden,false);
 assert.equal(d.querySelector('#import-items img'),null);
 const boxes=d.querySelectorAll('#import-items input');
 assert.equal(boxes[2].disabled,true); boxes[1].click();
 assert.match(d.querySelector('#start-import').textContent, /\(1\)/);
 d.querySelector('#start-import').click(); await tick();
 assert.match(d.querySelector('#import-progress').textContent,/1 failed/);
 assert.match(d.querySelector('#import-items').textContent,/Feed temporarily unavailable/);
 assert.equal(d.querySelector('#import-form textarea').disabled,false);
 fail=false; d.querySelector('#start-import').click(); await tick();
 assert.equal(d.querySelector('#start-import').disabled,true);
 assert.equal(calls.filter(c=>c[0].endsWith('/add')).length,2);
 assert.ok(calls.filter(c=>c[0].endsWith('/add')).every(c=>c[1]==='https://example.com/a'));
 d.querySelector('#import-text').dispatchEvent(new w.Event('input',{bubbles:true}));
 assert.equal(d.querySelector('#import-review').hidden,true);
 dom.window.close();
})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run([shutil.which('node'), '-e', script], input=json.dumps(html), text=True,
                            capture_output=True, timeout=20, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr


def test_repository_identity_matches_historical_urls_without_folding_path_case(import_db):
    repo = SubscriptionRepository()
    sub = repo.create(SubscriptionCreate(feed_url='https://example.com/Feed?key=A'), 'One', 'one')
    with get_db_connection() as conn:
        conn.execute('UPDATE subscriptions SET feed_url=? WHERE id=?', ('https://EXAMPLE.com:443/Feed?key=A#old', sub.id))
        conn.commit()
    assert repo.get_by_url('https://example.com/Feed?key=A').id == sub.id
    with pytest.raises(ValueError, match='already exists'):
        repo.create(SubscriptionCreate(feed_url='https://example.com/Feed?key=A'), 'Duplicate', 'duplicate')
    distinct = repo.create(SubscriptionCreate(feed_url='https://example.com/feed?key=A'), 'Distinct', 'distinct')
    assert distinct.id != sub.id
    assert len(repo.get_all()) == 2
