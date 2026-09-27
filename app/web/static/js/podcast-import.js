(() => {
  const form = document.getElementById('import-form');
  if (!form) return;
  const error = document.getElementById('import-error');
  const review = document.getElementById('import-review');
  const list = document.getElementById('import-items');
  const progress = document.getElementById('import-progress');
  const start = document.getElementById('start-import');
  const stop = document.getElementById('stop-import');
  let rows = [], busy = false, stopped = false;
  form.addEventListener('input', () => { if (!busy) { review.hidden = true; rows = []; } });
  const labels = {ready: 'New feed — checked when imported', join: 'Add from library', duplicate: 'Repeated in this import', existing: 'Already in My Podcasts', added: 'Imported', joined: 'Added from library'};
  const eligible = row => ['ready', 'join', 'error'].includes(row.status);
  async function post(url, data) {
    const response = await fetch(url, {method: 'POST', body: data, headers: {Accept: 'application/json'}});
    if (response.redirected) throw new Error('Your session expired. Log in again, then retry.');
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Could not read the import. Check the file or feed URLs and retry.');
    return body;
  }
  function showError(message) { error.textContent = message; error.hidden = !message; }
  function updateButton() {
    const count = rows.filter(row => row.selected && eligible(row)).length;
    start.textContent = `Import selected (${count})`;
    start.disabled = busy || count === 0;
  }
  function render() {
    list.replaceChildren();
    rows.forEach(row => {
      const li = document.createElement('li');
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox'; checkbox.checked = row.selected;
      checkbox.disabled = busy || !eligible(row);
      checkbox.addEventListener('change', () => { row.selected = checkbox.checked; updateButton(); });
      const content = document.createElement('span');
      const title = document.createElement('strong'); title.textContent = row.title || row.url;
      const url = document.createElement('span'); url.textContent = row.title ? row.url : '';
      const status = document.createElement('span'); status.className = 'import-status';
      status.textContent = labels[row.status] || row.detail || row.status;
      content.append(title, url, status); label.append(checkbox, content); li.append(label); list.append(li);
    });
    updateButton();
  }
  function lock(value) {
    busy = value;
    for (const input of form.elements) input.disabled = value;
    list.setAttribute('aria-busy', String(value));
    stop.hidden = !value; start.disabled = value;
  }
  form.addEventListener('submit', async event => {
    event.preventDefault(); if (busy) return;
    showError(''); review.hidden = true;
    const data = new FormData(form);
    lock(true); stop.hidden = true;
    document.getElementById('preview-import').textContent = 'Reviewing…';
    try {
      const result = await post('/import/preview', data);
      rows = result.items.map(row => ({...row, selected: eligible(row)}));
      review.hidden = false;
      progress.textContent = `${rows.length} entries reviewed. Uncheck any feeds you don’t want to add.`;
      document.getElementById('import-review-title').focus();
    } catch (err) { showError(err.message || 'Unable to review this import. Please retry.'); }
    finally { lock(false); document.getElementById('preview-import').textContent = 'Review import'; render(); }
  });
  stop.addEventListener('click', () => { stopped = true; stop.disabled = true; stop.textContent = 'Stopping after this feed…'; });
  start.addEventListener('click', async () => {
    if (busy) return;
    showError(''); stopped = false; lock(true); render();
    let completed = 0;
    const selected = rows.filter(row => row.selected && eligible(row));
    for (const row of selected) {
      if (stopped) break;
      progress.textContent = `Importing ${completed + 1} of ${selected.length}: ${row.title || row.url}`;
      const data = new FormData(); data.set('feed_url', row.url);
      try { Object.assign(row, await post('/import/add', data), {selected: false}); }
      catch (err) { row.status = 'error'; row.detail = err.message || 'Could not import this feed. Retry it below.'; }
      completed += 1; render();
    }
    lock(false); stop.disabled = false; stop.textContent = 'Stop after this feed'; render();
    const failed = rows.filter(row => row.status === 'error').length;
    progress.textContent = `${stopped ? 'Stopped' : 'Import finished'}. ${completed} feeds checked${failed ? `; ${failed} failed. Retry selected feeds below.` : '.'}`;
  });
  window.addEventListener('beforeunload', event => { if (busy) { event.preventDefault(); event.returnValue = ''; } });
})();
