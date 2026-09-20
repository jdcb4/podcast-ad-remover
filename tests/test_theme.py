"""Browser preference behavior against the real shared header and theme script."""
import json
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.infra.database import init_db
from app.web import router as web


def test_browser_theme_preference(isolated_data_dir):
    init_db()
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key='theme-test-only')
    app.include_router(web.router)
    with TestClient(app) as client:
        html = client.get('/').text
        for path in ['/admin/system', '/login', '/request-access']:
            assert 'data-theme-toggle' in client.get(path).text
    assert html.index('/static/js/theme.js') < html.index('/static/css/output.css')
    script = r'''
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const fs = require('node:fs');
const html = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync('app/web/static/js/theme.js', 'utf8');
const key = 'podcast-ad-remover-theme';
function open(saved, blocked=false) {
    const dom = new JSDOM(html, {url:'https://example.test/', runScripts:'outside-only'});
    const w = dom.window;
    if (saved !== null) w.localStorage.setItem(key, saved);
    if (blocked) Object.defineProperty(w, 'localStorage', {get(){throw new Error('Storage disabled');}});
    w.matchMedia = () => {throw new Error('System preference must not be consulted');};
    w.eval(source);
    const root = w.document.documentElement;
    // Preference is applied while still in the head, before DOMContentLoaded.
    assert.equal(root.dataset.theme, saved === 'light' && !blocked ? 'light' : 'dark');
    w.document.dispatchEvent(new w.Event('DOMContentLoaded'));
    return {dom, w, root, button:w.document.querySelector('[data-theme-toggle]')};
}
for (const saved of [null, 'dark', 'light', 'system', 'invalid']) {
    const {dom,w,root,button} = open(saved);
    const first = saved === 'light' ? 'light' : 'dark';
    assert.equal(button.hidden, false);
    assert.equal(button.getAttribute('aria-pressed'), String(first === 'light'));
    button.focus(); button.click();
    const next = first === 'dark' ? 'light' : 'dark';
    assert.equal(root.dataset.theme, next);
    assert.equal(root.classList.contains('dark'), next === 'dark');
    assert.equal(w.document.activeElement, button);
    assert.equal(w.localStorage.getItem(key), next);
    assert.equal(button.querySelector('[data-theme-label]').textContent, next === 'light' ? 'Light' : 'Dark');
    const reload = open(w.localStorage.getItem(key));
    assert.equal(reload.root.dataset.theme, next);
    reload.dom.window.close();
    button.click();
    assert.equal(w.localStorage.getItem(key), first);
    w.dispatchEvent(new w.StorageEvent('storage', {key, newValue:'light'}));
    assert.equal(root.dataset.theme, 'light');
    w.dispatchEvent(new w.StorageEvent('storage', {key:null, newValue:null}));
    assert.equal(root.dataset.theme, 'dark');
    dom.window.close();
}
const blocked = open(null,true);
blocked.button.click();
assert.equal(blocked.root.dataset.theme,'light');
blocked.button.click();
assert.equal(blocked.root.dataset.theme,'dark');
blocked.dom.window.close();
console.log('Theme default, toggle, persistence, reload, storage events and blocked storage passed');
'''
    result = subprocess.run([shutil.which('node'), '-e', script], input=json.dumps(html),
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
