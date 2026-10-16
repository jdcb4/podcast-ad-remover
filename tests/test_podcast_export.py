import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, parse_qs
from xml.etree import ElementTree as ET

import pytest

from app.infra.database import get_db_connection
from tests.test_management_http import populated_client


def test_export_replacement_feeds_with_current_user_token_and_no_processing_changes(populated_client):
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET enable_feed_auth=1,app_external_url='https://par.example'")
        conn.execute("UPDATE subscriptions SET title=?,feed_url=? WHERE id=90",
                     ('Books & <Stories> “today”', 'https://source.example/private?secret=ingestion-key'))
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug,source_type,owner_user_id) VALUES(93,'https://youtube.com/@show','Video show','video-show','youtube_channel',91)")
        conn.commit()
        before = {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                  for table in ('subscriptions', 'episodes', 'jobs', 'user_subscriptions')}
    response = populated_client.post('/subscriptions/export-opml', data={'subscription_ids': ['90', '93', '90']})
    assert response.status_code == 200
    assert response.headers['content-disposition'] == 'attachment; filename="par-subscriptions.opml"'
    assert response.headers['cache-control'] == 'private, no-store'
    assert 'ingestion-key' not in response.text and 'youtube.com' not in response.text
    root = ET.fromstring(response.content)
    assert root.attrib['version'] == '2.0'
    rows = root.findall('./body/outline')
    assert [r.attrib['title'] for r in rows] == ['Books & <Stories> “today”', 'Video show']
    assert [urlsplit(r.attrib['xmlUrl']).path for r in rows] == ['/feeds/show.xml', '/feeds/video-show.xml']
    tokens = [parse_qs(urlsplit(r.attrib['xmlUrl']).query)['token'][0] for r in rows]
    assert tokens[0] == tokens[1]
    from app.infra.repository import FeedTokenRepository
    assert FeedTokenRepository().validate(tokens[0])
    with get_db_connection() as conn:
        assert conn.execute('SELECT user_id FROM feed_tokens WHERE token_hash=?',
                            (FeedTokenRepository.hash_token(tokens[0]),)).fetchone()[0] == 92
        assert before == {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in before}


@pytest.mark.parametrize('ids,status', [([], 422), (['90', '99999'], 404),
                                        ([str(i) for i in range(501)], 400)])
def test_export_invalid_selection_is_atomic(populated_client, ids, status):
    response = populated_client.post('/subscriptions/export-opml', data={'subscription_ids': ids})
    assert response.status_code == status
    with get_db_connection() as conn:
        assert conn.execute('SELECT count(*) FROM subscriptions').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0


def test_deleted_selection_and_cross_origin_export_rejected(populated_client):
    response = populated_client.post('/subscriptions/export-opml', data={'subscription_ids': '90'},
                                     headers={'Origin': 'https://unrelated.example'})
    assert response.status_code == 403
    with get_db_connection() as conn:
        conn.execute("UPDATE subscriptions SET deletion_status='pending' WHERE id=90")
        conn.commit()
    assert populated_client.post('/subscriptions/export-opml', data={'subscription_ids': '90'}).status_code == 404


def test_export_requires_authentication(populated_client):
    from app.web.auth import require_auth
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET auth_enabled=1')
        conn.commit()
    populated_client.app.dependency_overrides.pop(require_auth)
    assert populated_client.post('/subscriptions/export-opml', data={'subscription_ids': '90'}).status_code == 401


def test_ingestion_url_only_in_owner_management(populated_client, monkeypatch):
    from app.web import auth
    with get_db_connection() as conn:
        conn.execute('UPDATE subscriptions SET feed_url=? WHERE id=90', ('https://source.example/private?key=secret&other=1',))
        conn.commit()
    monkeypatch.setattr(auth, 'get_current_user', lambda _: SimpleNamespace(id=91, is_admin=False, username='owner'))
    page = populated_client.get('/subscriptions/90')
    assert page.status_code == 200
    assert 'id="ingestion-url"' in page.text and 'key=secret&amp;other=1' in page.text
    monkeypatch.setattr(auth, 'get_current_user', lambda _: SimpleNamespace(id=92, is_admin=False, username='reader'))
    page = populated_client.get('/subscriptions/90')
    assert 'id="ingestion-url"' not in page.text and 'key=secret' not in page.text


def test_selection_sync_export_and_view_replacement():
    script = r"""
const {JSDOM}=require('jsdom'),fs=require('node:fs'),assert=require('node:assert/strict');
const dom=new JSDOM(`<form id="opml-export-form"><span id="bulk-selection-count"></span><div id="opml-selected-podcasts"></div><button id="export-opml" disabled></button><p id="bulk-export-only" hidden></p></form><form id="bulk-settings-form"></form><div id="dashboard-podcast-results"></div>`,{runScripts:'outside-only'});
const w=dom.window,d=w.document;
function rows(){return `<input id="select-all-podcasts" type="checkbox">`+['mobile','grid','table'].map(kind=>[1,2].map(id=>`<article><input type="checkbox" class="podcast-bulk-checkbox" value="${id}" data-can-manage="${id===1}" data-kind="${kind}"></article>`).join('')).join('');}
d.getElementById('dashboard-podcast-results').innerHTML=rows();
w.eval(fs.readFileSync('app/web/static/js/bulk-subscriptions.js','utf8'));
const fields=()=>Array.from(d.querySelectorAll('#opml-selected-podcasts input')).map(x=>x.value);
const checkbox=(id,kind)=>d.querySelector(`[value="${id}"][data-kind="${kind}"]`);
const change=box=>box.dispatchEvent(new w.Event('change',{bubbles:true}));
assert.equal(d.getElementById('export-opml').disabled,true);
checkbox(1,'mobile').checked=true;change(checkbox(1,'mobile'));
assert.deepEqual(fields(),['1']);assert.equal(d.getElementById('bulk-selection-count').textContent,'1');
assert.equal(checkbox(1,'table').checked,true);assert.equal(checkbox(1,'grid').checked,true);
assert.equal(d.getElementById('export-opml').disabled,false);
assert.equal(d.getElementById('bulk-settings-form').classList.contains('hidden'),false);
checkbox(2,'grid').checked=true;change(checkbox(2,'grid'));
assert.deepEqual(fields(),['1','2']);assert.equal(d.getElementById('bulk-selection-count').textContent,'2');
assert.equal(d.getElementById('bulk-settings-form').classList.contains('hidden'),true);
assert.equal(d.getElementById('bulk-export-only').hidden,false);
checkbox(2,'table').checked=false;change(checkbox(2,'table'));assert.deepEqual(fields(),['1']);
d.getElementById('dashboard-podcast-results').innerHTML=rows();
d.dispatchEvent(new w.CustomEvent('dashboard-library-results-changed'));
assert.deepEqual(fields(),[]);assert.equal(d.getElementById('export-opml').disabled,true);
const all=d.getElementById('select-all-podcasts');all.checked=true;change(all);
assert.deepEqual(fields(),['1','2']);assert.equal(d.getElementById('bulk-selection-count').textContent,'2');
all.checked=false;change(all);assert.deepEqual(fields(),[]);
dom.window.close();
"""
    result = subprocess.run([shutil.which('node'), '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
