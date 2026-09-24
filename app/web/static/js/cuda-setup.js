(() => {
    'use strict';
    const set = (id, text) => { const node = document.getElementById(id); if (node) node.textContent = text; };
    async function refresh() {
        try {
            const response = await fetch('/admin/ai/cuda/status', {credentials: 'same-origin'});
            if (!response.ok) return;
            const state = await response.json();
            set('cuda-phase', state.phase || 'idle');
            set('cuda-message', state.message || '');
            set('cuda-effective', `${state.effective_device || 'cpu'} / ${state.effective_compute_type || 'float32'}`);
            set('cuda-download', state.phase === 'downloading' ? `${Math.round((state.downloaded || 0) / 1048576)} MB downloaded` : '');
            // Avoid a reload: the user may be editing unsaved settings.
            if (state.phase === 'ready') set('cuda-message', `${state.message} Refresh this page to see the saved device and precision choices.`);
        } catch (_) { /* A temporary connection loss must not interrupt settings edits. */ }
    }
    const timer = setInterval(refresh, 3000);
    window.addEventListener('pagehide', () => clearInterval(timer), {once: true});
})();
