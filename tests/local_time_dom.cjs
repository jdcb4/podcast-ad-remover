const {JSDOM} = require('jsdom');
const fs = require('node:fs');
const assert = require('node:assert/strict');

async function run() {
  const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
  const dom = new JSDOM(payload.views.mine, {url: 'http://testserver/?view=mine', runScripts: 'outside-only'});
  const w = dom.window;
  try {
    assert.equal(w.Intl.DateTimeFormat().resolvedOptions().timeZone, payload.zone);
    const legacy = w.document.createElement('aside');
    legacy.innerHTML = payload.legacy;
    w.document.body.append(legacy);
    const originalLegacy = legacy.textContent;
    assert(originalLegacy.includes('timezone unknown'));
    const dates = () => [...w.document.querySelectorAll('#dashboard-podcast-results time[data-lt]')];
    assert(dates().length > 0);
    const fallback = dates()[0].textContent;
    w.eval(fs.readFileSync('app/web/static/js/local-time.js', 'utf8'));
    w.eval(fs.readFileSync('app/web/static/js/dashboard-refresh.js', 'utf8'));
    await new Promise(resolve => w.addEventListener('load', resolve, {once: true}));
    const expected = dates()[0].textContent;
    assert.notEqual(expected, fallback, 'the fixture must cross a UTC/local date boundary');
    function checkDates() {
      for (const el of dates()) {
        assert.equal(el.textContent, expected);
        assert(el.title.includes('UTC:'));
      }
      assert.equal(legacy.textContent, originalLegacy, 'unknown historical times must stay unchanged');
      assert.equal(legacy.querySelector('time'), null);
    }
    checkDates();
    w.fetch = async url => {
      const view = new URL(url, w.location.href).searchParams.get('view');
      return {ok: true, text: async () => payload.views[view]};
    };
    function changed() {
      return new Promise(resolve => w.document.addEventListener('dashboard-library-results-changed', resolve, {once: true}));
    }
    for (const view of ['library', 'mine']) {
      const next = changed();
      w.document.querySelector(`[data-library-view-link="${view}"]`).click();
      await next;
      assert.equal(w.document.getElementById('dashboard-podcast-results').dataset.libraryView, view);
      checkDates();
    }
    const back = changed();
    w.history.back();
    await back;
    assert.equal(w.document.getElementById('dashboard-podcast-results').dataset.libraryView, 'library');
    checkDates();
  } finally {
    dom.window.close();
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
