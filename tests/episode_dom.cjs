// Execute the shipped script against actual server-rendered HTML. No browser/network.
const {JSDOM} = require('jsdom');
const fs = require('node:fs');
const assert = require('node:assert/strict');

async function run() {
  const dom = new JSDOM(fs.readFileSync(0, 'utf8'), {url: 'http://localhost/subscriptions/90', runScripts: 'outside-only'});
  const w = dom.window;
  const messages = [];
  const calls = [];
  w.appConfirm = async () => true;
  w.appToast = message => messages.push(message);
  const episode = {id: 90, title: 'Result', status: 'unprocessed', guid: '..', description: 'Description'};
  const page = title => ({episodes: [{...episode, title}], total: 1, has_more: false});
  w.fetch = async (url, options) => {
    calls.push({url, options});
    return Response.json(options?.method ? {status: 'deleted'} : page('Result'));
  };
  w.eval(fs.readFileSync('app/web/static/js/episodes.js', 'utf8'));
  await w.deleteEpisode(90); // There is deliberately no btn-delete-90 element.
  assert(calls.some(c => c.options?.method === 'DELETE'));
  assert(messages.some(m => m.includes('removal requested')));

  let releaseOld;
  w.fetch = url => {
    if (url.includes('search=older')) return new Promise(resolve => { releaseOld = resolve; });
    return Promise.resolve(Response.json(page('Latest result')));
  };
  const older = w.performServerSearch('older');
  await w.performServerSearch('newer');
  releaseOld(Response.json(page('Stale result')));
  await older;
  assert(w.document.getElementById('episodes-grid').textContent.includes('Latest result'));
  assert(!w.document.getElementById('episodes-grid').textContent.includes('Stale result'));

  w.fetch = async () => Response.json({detail: 'Forbidden by ownership policy'}, {status: 403});
  messages.length = 0;
  await w.deleteEpisode(90);
  assert.deepEqual(messages, ['Forbidden by ownership policy']);

  const markup = w.createEpisodeCardHTML({...episode, title: '<img src=x onerror=alert(1)>', description: '<script>bad()</script>'}, 'show');
  const fragment = w.document.createElement('div');
  fragment.innerHTML = markup;
  assert.equal(fragment.querySelectorAll('script,img').length, 0);
  assert.equal(fragment.querySelector('.episode-checkbox').getAttribute('aria-label'), 'Select <img src=x onerror=alert(1)>');
  const toggle = w.document.querySelector('[aria-controls="settings-form"]');
  w.toggleProcessingSettings(toggle);
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  assert.equal(w.document.getElementById('action-sheet').tagName, 'DIALOG');
  dom.window.close();
}
run().catch(error => { console.error(error); process.exitCode = 1; });
