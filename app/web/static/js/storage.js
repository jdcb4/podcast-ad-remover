(() => {
  const state = document.getElementById('storage-state');
  if (!state) return;
  const active = ['waiting', 'copying', 'cleaning', 'paused', 'error'];
  async function refresh() {
    try {
      const response = await fetch('/admin/system/storage/status', {headers: {Accept: 'application/json'}});
      if (!response.ok) throw new Error('Status unavailable');
      const s = await response.json();
      state.textContent = s.status.charAt(0).toUpperCase() + s.status.slice(1);
      document.getElementById('storage-progress').max = s.total || 1;
      document.getElementById('storage-progress').value = s.completed;
      document.getElementById('storage-count').textContent = `${s.completed} of ${s.total} files complete`;
      document.getElementById('storage-job-error').textContent = s.error || s.availability_error || '';
      document.getElementById('storage-detail').textContent = s.status === 'cancelling' ? 'Finishing the current file before ending the operation.' : s.status === 'waiting' ? 'Waiting for running episodes to finish. New processing is paused.' : ['paused', 'error'].includes(s.status) ? 'Processing stays paused. Resume the operation, or end it to allow processing again.' : s.status === 'complete' ? 'Operation complete. Processing can continue. Originals are retained unless you chose cleanup.' : '';
      document.getElementById('storage-pause').hidden = !['waiting', 'copying', 'cleaning'].includes(s.status);
      document.getElementById('storage-resume').hidden = !['paused', 'error'].includes(s.status);
      document.getElementById('storage-cancel').hidden = !active.includes(s.status);
      const retained = document.getElementById('storage-retained');
      if (retained) retained.textContent = `${s.retained} local copies recorded`;
    } catch (_) { state.textContent = 'Cannot refresh progress. Reconnecting; the background operation continues.'; }
    setTimeout(refresh, 3000);
  }
  setTimeout(refresh, 3000);
})();
