(() => {
    const queueRegion = document.getElementById('dashboard-queue-region');
    const queueContent = document.getElementById('queue-content');
    const queueCount = document.getElementById('queue-total-count');
    const queueStatus = document.getElementById('queue-refresh-status');
    const toggle = document.getElementById('auto-refresh-toggle');
    const intervalInput = document.getElementById('refresh-interval');
    const countdown = document.getElementById('refresh-countdown');
    let timeLeft = 30;
    let timer = null;

    function statusBadge(item) {
        const badge = document.createElement('span');
        if (item.status === 'processing') {
            badge.className = 'badge-info animate-pulse';
            badge.textContent = 'Processing';
        } else if (item.status === 'rate_limited') {
            badge.className = 'badge-warning';
            badge.textContent = 'Rate Limited';
        } else if (item.status === 'failed') {
            badge.className = 'badge-secondary';
            badge.textContent = 'Retry Pending';
        } else {
            badge.className = 'badge-neutral';
            badge.textContent = 'Queued';
        }
        return badge;
    }

    function renderQueue(items) {
        queueContent.replaceChildren();
        items.forEach((item) => {
            const card = document.createElement('div');
            card.className = 'queue-item p-4 bg-surface-elevated rounded-xl border border-white/[0.06]';

            const header = document.createElement('div');
            header.className = 'flex items-start justify-between mb-2';
            const text = document.createElement('div');
            text.className = 'flex-1 min-w-0 mr-4';
            const title = document.createElement('p');
            title.className = 'font-medium text-sm truncate';
            title.textContent = item.title;
            const podcast = document.createElement('p');
            podcast.className = 'text-xs text-text-muted';
            podcast.textContent = item.podcast_title;
            const badgeWrap = document.createElement('div');
            badgeWrap.className = 'flex-shrink-0';
            text.append(title, podcast);
            badgeWrap.append(statusBadge(item));
            header.append(text, badgeWrap);
            card.append(header);

            if (item.status === 'processing' && item.processing_step) {
                const progressWrap = document.createElement('div');
                progressWrap.className = 'mt-3';
                const labels = document.createElement('div');
                labels.className = 'flex justify-between items-center text-xs mb-1.5';
                const step = document.createElement('span');
                step.className = 'text-primary-400 font-medium';
                step.textContent = item.processing_step;
                const amount = document.createElement('span');
                amount.className = 'text-text-muted';
                amount.textContent = `${item.progress}%`;
                const bar = document.createElement('div');
                bar.className = 'progress-bar';
                const fill = document.createElement('div');
                fill.className = 'progress-bar-fill';
                fill.style.width = `${item.progress}%`;
                labels.append(step, amount);
                bar.append(fill);
                progressWrap.append(labels, bar);
                card.append(progressWrap);
            }
            queueContent.append(card);
        });

        queueCount.textContent = `${items.length} item${items.length === 1 ? '' : 's'}`;
        queueRegion.classList.toggle('hidden', items.length === 0);
    }

    async function refreshQueue() {
        if (!queueRegion || !queueContent) return;
        queueRegion.setAttribute('aria-busy', 'true');
        try {
            const response = await fetch('/api/dashboard/queue', {
                headers: { Accept: 'application/json', 'X-Requested-With': 'fetch' },
                credentials: 'same-origin',
            });
            if (!response.ok) throw new Error(`Request failed (${response.status})`);
            const result = await response.json();
            renderQueue(Array.isArray(result.items) ? result.items : []);
            queueStatus.classList.add('hidden');
            queueStatus.textContent = '';
        } catch (error) {
            queueStatus.textContent = 'Queue refresh failed. Use Refresh to reload the full page.';
            queueStatus.classList.remove('hidden');
        } finally {
            queueRegion.setAttribute('aria-busy', 'false');
        }
    }

    function normalizedInterval() {
        let value = Number.parseInt(intervalInput.value, 10);
        if (!Number.isFinite(value)) value = 30;
        value = Math.max(5, Math.min(300, value));
        intervalInput.value = String(value);
        return value;
    }

    function resetCountdown() {
        timeLeft = normalizedInterval();
        countdown.textContent = toggle.checked ? `${timeLeft}s` : 'Off';
        countdown.classList.toggle('bg-primary-500/20', toggle.checked);
        countdown.classList.toggle('text-primary-400', toggle.checked);
        countdown.classList.toggle('bg-white/5', !toggle.checked);
        countdown.classList.toggle('text-text-muted', !toggle.checked);
    }

    function tick() {
        if (!toggle.checked) return;
        timeLeft -= 1;
        if (timeLeft <= 0) {
            refreshQueue();
            resetCountdown();
        } else {
            countdown.textContent = `${timeLeft}s`;
        }
    }

    if (queueRegion && queueContent && queueCount && queueStatus && toggle && intervalInput && countdown) {
        const savedInterval = localStorage.getItem('dashboard-refresh-interval');
        const savedToggle = localStorage.getItem('dashboard-refresh-enabled');
        if (savedInterval) intervalInput.value = savedInterval;
        if (savedToggle !== null) toggle.checked = savedToggle === 'true';
        resetCountdown();
        timer = window.setInterval(tick, 1000);
        toggle.addEventListener('change', () => {
            localStorage.setItem('dashboard-refresh-enabled', String(toggle.checked));
            resetCountdown();
        });
        intervalInput.addEventListener('change', () => {
            localStorage.setItem('dashboard-refresh-interval', String(normalizedInterval()));
            resetCountdown();
        });
    }

    async function switchLibraryView(view, { updateHistory = true } = {}) {
        const region = document.getElementById('dashboard-library-region');
        const currentResults = document.getElementById('dashboard-podcast-results');
        if (!region || !currentResults || !['mine', 'library'].includes(view)) return;
        region.setAttribute('aria-busy', 'true');

        try {
            const response = await fetch(`/?view=${encodeURIComponent(view)}`, {
                headers: { Accept: 'text/html', 'X-Requested-With': 'fetch' },
                credentials: 'same-origin',
            });
            if (!response.ok) throw new Error(`Request failed (${response.status})`);
            const html = await response.text();
            const page = new DOMParser().parseFromString(html, 'text/html');
            const nextResults = page.getElementById('dashboard-podcast-results');
            if (!nextResults) throw new Error('Podcast results were missing');

            currentResults.replaceChildren(...Array.from(nextResults.childNodes).map((node) => node.cloneNode(true)));
            currentResults.dataset.libraryView = view;
            const grid = document.getElementById('podcast-grid');
            if (grid) grid.dataset.libraryView = view;
            document.querySelectorAll('[data-library-view-link]').forEach((link) => {
                const active = link.dataset.libraryViewLink === view;
                link.classList.toggle('btn-primary', active);
                link.classList.toggle('btn-secondary', !active);
                link.setAttribute('aria-current', active ? 'page' : 'false');
            });
            if (updateHistory) history.pushState({ libraryView: view }, '', `/?view=${view}`);
            document.dispatchEvent(new CustomEvent('dashboard-library-results-changed', { detail: { view } }));
        } catch (error) {
            window.location.assign(`/?view=${encodeURIComponent(view)}`);
        } finally {
            region.setAttribute('aria-busy', 'false');
        }
    }

    document.addEventListener('click', (event) => {
        const link = event.target.closest?.('[data-library-view-link]');
        if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
        event.preventDefault();
        switchLibraryView(link.dataset.libraryViewLink);
    });
    window.addEventListener('popstate', () => {
        const view = new URL(window.location.href).searchParams.get('view') || 'mine';
        switchLibraryView(view, { updateHistory: false });
    });
})();
