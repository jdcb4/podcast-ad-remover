const episodePage = JSON.parse(document.getElementById('episode-page-data').textContent);
    // Lazy Loading State
    let currentOffset = episodePage.offset;
    const pageSize = episodePage.pageSize;
    const subscriptionId = episodePage.subscriptionId;
    let isLoading = false;
    let totalEpisodes = episodePage.total;

    // Search State
    let currentSearchTerm = '';
    let searchDebounceTimer = null;

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (char) => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[char]);
    }

    // Episode Card Template Generator
    function createEpisodeCardHTML(ep, subscriptionSlug) {
        const allowedStatuses = ['completed', 'processing', 'ignored', 'deleted', 'failed', 'rate_limited', 'unprocessed', 'pending'];
        const status = allowedStatuses.includes(ep.status) ? ep.status : 'unprocessed';
        const isManual = ep.is_manual_download ? 'true' : 'false';
        const episodeId = Number.parseInt(ep.id, 10) || 0;
        const title = escapeHtml(ep.title || '');
        const titleSearch = escapeHtml((ep.title || '').toLowerCase());
        const description = escapeHtml(ep.description || '');
        const pubDate = escapeHtml(ep.pub_date || '');
        const listenCount = Number.parseInt(ep.listen_count, 10) || 0;
        const formatDuration = (seconds) => {
            if (!seconds) return '-';
            const totalSeconds = Number.parseInt(seconds, 10) || 0;
            const m = Math.floor((totalSeconds % 3600) / 60);
            const s = totalSeconds % 60;
            const h = Math.floor(totalSeconds / 3600);
            if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
            return `${m}:${s.toString().padStart(2, '0')}`;
        };

        // Status badge HTML
        let statusBadge = '';
        if (status === 'completed') {
            statusBadge = `<span class="inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-emerald-500/30 text-emerald-400 text-[8px] font-black tracking-widest leading-none hover:bg-emerald-500/40 transition-colors"><span class="w-1 h-1 rounded-full bg-emerald-400"></span>DOWNLOADED</span>`;
        } else if (status === 'processing') {
            statusBadge = `<span class="inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-amber-500/30 text-amber-400 text-[8px] font-black tracking-widest leading-none animate-pulse"><span class="w-1 h-1 rounded-full bg-amber-400"></span>PROCESSING</span>`;
        } else if (status === 'ignored' || status === 'deleted') {
            statusBadge = `<span class="inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-slate-500/30 text-slate-400 text-[8px] font-black tracking-widest leading-none hover:bg-slate-500/40 transition-colors"><span class="w-1 h-1 rounded-full bg-slate-400"></span>DELETED</span>`;
        } else if (status === 'failed') {
            statusBadge = `<span class="inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-red-500/30 text-red-400 text-[8px] font-black tracking-widest leading-none hover:bg-red-500/40 transition-colors"><span class="w-1 h-1 rounded-full bg-red-400"></span>FAILED</span>`;
        } else if (status === 'rate_limited') {
            statusBadge = `<span class="inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-amber-500/30 text-amber-400 text-[8px] font-black tracking-widest leading-none"><span class="w-1 h-1 rounded-full bg-amber-400"></span>RATE LIMITED</span>`;
        }

        // Play count badge
        const playsBadge = listenCount > 0 ? `<span class="ml-3 text-primary-400 text-[11px] font-bold tracking-wider uppercase">${listenCount} play${listenCount !== 1 ? 's' : ''}</span>` : '';

        // Action buttons
        let actionButtons = '';
        if (ep.local_filename && status !== 'ignored') {
            const audioHref = `/episodes/${episodeId}/audio`;
            actionButtons = `<a href="${audioHref}" target="_blank"  class="p-2 bg-primary-500 rounded-full text-white hover:bg-primary-400 hover:scale-105 transition-all shadow-lg" title="Play"><svg class="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg></a>`;
        } else if (episodePage.canManage && (status === 'unprocessed' || status === 'ignored')) {
            actionButtons = `<button onclick="downloadEpisode(${episodeId})" class="p-2 text-text-muted hover:text-white hover:bg-white/10 rounded-full transition-colors" title="Download"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg></button>`;
        }

        const hasTranscript = ep.transcript_path ? 'true' : 'false';
        const hasReport = (ep.report_path || ep.ad_report_path) ? 'true' : 'false';

        return `
        <div class="episode-card card flex flex-col relative overflow-visible group transition-all duration-300 h-full border hover:border-white/10"
            data-status="${status}" data-manual="${isManual}"
            data-title="${titleSearch}" data-date="${pubDate}" data-listen-count="${listenCount}"
            id="card-${episodeId}">
            <div class="checkbox-wrapper absolute top-4 left-4 z-20 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity" onclick="event.stopPropagation();">
                <input ${episodePage.canManage ? '' : 'disabled hidden'} type="checkbox" name="episode_ids" value="${episodeId}" aria-label="Select ${title}" class="h-5 w-5 rounded border-white/20 bg-surface-elevated text-primary-500 focus:ring-primary-500 cursor-pointer shadow-lg episode-checkbox" onchange="updateBatchActions()">
            </div>
            <div class="flex-1 p-4 flex flex-col cursor-pointer relative" onclick="toggleDescription('${episodeId}')">
                <div class="flex justify-between items-start mb-3">
                    <div class="flex items-center gap-2 text-[11px] font-bold tracking-wider text-text-muted uppercase">
                        <span class="date-relative" data-date="${pubDate}">${pubDate}</span>
                        <span class="text-white/10 dot-separator">•</span>
                        <span>${escapeHtml(formatDuration(ep.duration))}</span>
                    </div>
                </div>
                <h3 class="font-display text-lg font-bold text-text-primary leading-snug mb-3 group-hover:text-primary-300 transition-colors" title="${title}">
                    ${statusBadge}
                    <span class="block mt-5">${title}${playsBadge}</span>
                </h3>
                <div class="mb-4 text-sm leading-relaxed text-text-secondary cursor-pointer group/ep-desc relative" onclick="this.classList.toggle('expanded'); event.stopPropagation();">
                    ${description ? `<div class="py-1 px-1"><div class="line-clamp-4 group-[.expanded]/ep-desc:line-clamp-none transition-all duration-500 ease-in-out text-text-secondary/90">${description}</div></div>` : ''}
                </div>
            </div>
            <div class="px-4 py-3 flex justify-between items-center bg-surface-base/95 md:bg-surface-base/30 md:backdrop-blur-sm" onclick="event.stopPropagation();">
                <div class="flex items-center gap-2 list-actions">
                    ${actionButtons}
                    <div class="relative inline-block text-left">
                        <button onclick="showActionSheet(event, ${episodeId}, ${hasTranscript}, ${hasReport})" class="p-2 text-text-muted hover:text-white transition-colors rounded-full hover:bg-white/10" title="More">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v.01M12 12v.01M12 19v.01"></path></svg>
                        </button>
                    </div>
                </div>
            </div>
            <button type="button" data-description-toggle aria-expanded="false" onclick="toggleDescription(${episodeId})" class="sr-only focus:not-sr-only">Toggle description for ${title}</button>
        </div>`;
    }

    let currentFilter = localStorage.getItem('episode-filter-preference') || 'all';
    let pageRequest = null;
    let pageSequence = 0;
    let retryReset = false;

    async function fetchEpisodePage(reset = false) {
        if (isLoading && !reset) return;
        if (pageRequest) pageRequest.abort();
        const controller = new AbortController();
        pageRequest = controller;
        const sequence = ++pageSequence;
        isLoading = true;
        const grid = document.getElementById('episodes-grid');
        const count = document.getElementById('episode-count');
        const button = document.getElementById('load-more-btn');
        grid.setAttribute('aria-busy', 'true');
        if (button) button.disabled = true;
        if (count) count.textContent = 'Loading episodes…';
        const params = new URLSearchParams({limit: pageSize, offset: reset ? 0 : currentOffset, search: currentSearchTerm, filter: currentFilter});
        try {
            const response = await fetch(`/api/subscriptions/${subscriptionId}/episodes?${params}`, {signal: controller.signal, headers: {Accept: 'application/json'}});
            if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('Could not load episodes. Please retry.');
            const data = await response.json();
            if (sequence !== pageSequence) return;
            const selected = new Set(Array.from(document.querySelectorAll('.episode-checkbox:checked'), cb => cb.value));
            if (reset) grid.replaceChildren();
            data.episodes.forEach(ep => grid.insertAdjacentHTML('beforeend', createEpisodeCardHTML(ep, data.subscription_slug)));
            currentOffset = (reset ? 0 : currentOffset) + data.episodes.length;
            totalEpisodes = data.total;
            retryReset = false;
            if (!currentOffset) grid.innerHTML = '<p class="col-span-full card p-6">No episodes match these filters. Try another filter or search.</p>';
            for (const checkbox of grid.querySelectorAll('.episode-checkbox')) checkbox.checked = selected.has(checkbox.value);
            updateBatchActions();
            if (count) count.textContent = `Showing ${currentOffset} of ${totalEpisodes} matching episodes`;
            const more = document.getElementById('load-more-container');
            if (more) more.style.display = data.has_more ? 'flex' : 'none';
            const moreCount = document.getElementById('load-more-count');
            if (moreCount) moreCount.textContent = `${currentOffset} of ${totalEpisodes}`;
        } catch (error) {
            if (error.name !== 'AbortError' && sequence === pageSequence) {
                retryReset = reset;
                if (count) count.textContent = 'Could not load episodes. Your current results are kept. Use Load more to retry.';
                appToast(error.message, {type: 'error'});
                const more = document.getElementById('load-more-container');
                if (more) more.style.display = 'flex';
            }
        } finally {
            if (sequence === pageSequence) {
                isLoading = false;
                grid.setAttribute('aria-busy', 'false');
                if (button) button.disabled = false;
            }
        }
    }

    function loadMoreEpisodes() { return fetchEpisodePage(retryReset); }
    function performServerSearch(term) {
        currentSearchTerm = term.trim();
        return fetchEpisodePage(true);
    }
    document.addEventListener('DOMContentLoaded', () => {
        const input = document.getElementById('episode-search');
        input?.addEventListener('input', () => {
            clearTimeout(searchDebounceTimer);
            pageRequest?.abort();
            ++pageSequence;
            currentSearchTerm = input.value.trim();
            searchDebounceTimer = setTimeout(() => fetchEpisodePage(true), 300);
        });
    });

    // Batch Actions Functions
    function updateBatchActions() {
        const checkboxes = document.querySelectorAll('.episode-checkbox:checked');
        const toolbar = document.getElementById('batch-actions');
        const countEl = document.getElementById('batch-count');

        if (checkboxes.length > 0) {
            if (toolbar) {
                toolbar.style.display = 'flex';
                // Trigger reflow for transition
                toolbar.offsetHeight;
                toolbar.style.opacity = '1';
                toolbar.style.pointerEvents = 'auto';
                toolbar.style.transform = 'translateY(0)';
            }
            if (countEl) countEl.textContent = checkboxes.length + ' selected';
        } else {
            if (toolbar) {
                toolbar.style.opacity = '0';
                toolbar.style.transform = 'translateY(20px)';
                toolbar.style.pointerEvents = 'none';
                setTimeout(() => {
                    if (document.querySelectorAll('.episode-checkbox:checked').length === 0) {
                        toolbar.style.display = 'none';
                    }
                }, 300);
            }
        }
    }

    function selectAllEpisodes() {
        const visibleCards = document.querySelectorAll('.episode-card:not(.hidden)');
        visibleCards.forEach(card => {
            const cb = card.querySelector('.episode-checkbox');
            if (cb) cb.checked = true;
        });
        updateBatchActions();
    }

    function clearSelection() {
        document.querySelectorAll('.episode-checkbox:checked').forEach(cb => cb.checked = false);
        updateBatchActions();
    }

    async function managementRequest(url, method = 'POST') {
        const response = await fetch(url, {method, headers: {Accept: 'application/json'}});
        let data;
        try { data = await response.json(); } catch { throw new Error('Session expired or an unexpected response was received. Reload and try again.'); }
        if (!response.ok || ['ignored', 'error', 'failed'].includes(data.status)) throw new Error(data.detail || data.reason || `Request failed (${response.status})`);
        return data;
    }

    async function batchAction(action) {
        const ids = Array.from(document.querySelectorAll('.episode-checkbox:checked'), cb => cb.value);
        if (!ids.length || !await appConfirm(`${action === 'ignore' ? 'Ignore and remove files for' : 'Reprocess'} ${ids.length} episode(s)?`, {danger: action === 'ignore'})) return;
        let completed = 0;
        const failed = [];
        for (const id of ids) {
            try {
                await managementRequest(`/api/episodes/${id}/${action}`);
                const checkbox = document.querySelector(`.episode-checkbox[value="${id}"]`);
                if (checkbox) checkbox.checked = false;
                completed++;
            } catch (error) { failed.push(`${id}: ${error.message}`); }
        }
        appToast(`${completed} succeeded; ${failed.length} failed.${failed.length ? ' Failed items remain selected. ' + failed.join('; ') : ''}`, {type: failed.length ? 'error' : 'success', timeout: 10000});
        if (failed.length) updateBatchActions();
        else await fetchEpisodePage(true);
    }
    function batchReprocess() { return batchAction('reprocess'); }
    function batchDelete() { return batchAction('ignore'); }

    function filterEpisodes(filter) {
        currentFilter = filter || localStorage.getItem('episode-filter-preference') || 'all';
        localStorage.setItem('episode-filter-preference', currentFilter);
        document.querySelectorAll('#episode-filters .filter-btn').forEach(button => {
            const active = button.dataset.filter === currentFilter;
            button.classList.toggle('badge-info', active);
            button.classList.toggle('badge-soft', !active);
            button.setAttribute('aria-pressed', String(active));
        });
        return fetchEpisodePage(true);
    }

    function toggleProcessingSettings(button) {
        const panel = document.getElementById('settings-form');
        if (!panel) return;
        const expanded = panel.classList.toggle('hidden') === false;
        button.setAttribute('aria-expanded', String(expanded));
    }

    function timeAgo(dateParam) {
        if (!dateParam) return null;
        const date = typeof dateParam === 'object' ? dateParam : new Date(dateParam);
        const today = new Date();
        const seconds = Math.round((today - date) / 1000);
        const minutes = Math.round(seconds / 60);
        const isToday = today.toDateString() === date.toDateString();

        if (isToday) return 'Today';
        if (seconds < 86400 * 2 && today.getDate() - date.getDate() === 1) return 'Yesterday';
        if (seconds < 86400 * 7) return `${Math.floor(seconds / 86400)} days ago`;

        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    // Format Dates on Load with Relative Logic
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.date-relative').forEach(el => {
            try {
                const dateStr = el.getAttribute('data-date');
                const rel = timeAgo(dateStr);
                if (rel) el.textContent = rel;
            } catch (e) { }
        });
    });

    function toggleDescription(id) {
        const card = document.getElementById('card-' + id);
        if (card) {
            const expanded = card.classList.toggle('expanded');
            card.querySelector('[data-description-toggle]')?.setAttribute('aria-expanded', String(expanded));
        }
    }



    // Format Dates on Load
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.date-formatted').forEach(el => {
            try {
                const dateStr = el.textContent.trim();
                const date = new Date(dateStr);
                if (!isNaN(date)) {
                    // Use simple format: "Dec 23, 2023"
                    el.textContent = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
                }
            } catch (e) { }
        });
    });

    async function episodeAction(id, url, message, confirmation, method = 'POST', danger = false) {
        if (!await appConfirm(confirmation, {danger})) return;
        const card = document.getElementById(`card-${id}`);
        card?.setAttribute('aria-busy', 'true');
        try {
            await managementRequest(url, method);
            appToast(message, {type: 'success'});
            await fetchEpisodePage(true);
        } catch (error) { appToast(error.message, {type: 'error'}); }
        finally { card?.setAttribute('aria-busy', 'false'); }
    }
    function deleteEpisode(id) {
        return episodeAction(id, `/api/episodes/${id}`, 'Episode ignored and file removal requested.', 'Ignore this episode and remove its local files?', 'DELETE', true);
    }
    function downloadEpisode(id) {
        return episodeAction(id, `/episodes/${id}/download`, 'Download queued.', 'Download and process this episode?');
    }

    function showActionSheet(e, episodeId, hasTranscript, hasReport) {
        e.stopPropagation();
        const sheet = document.getElementById('action-sheet');
        const content = document.getElementById('action-sheet-content');

        // Build menu items
        let html = '';
        if (hasTranscript) {
            html += `<a href="/episodes/${episodeId}/transcript" class="block w-full text-left px-4 py-3 text-sm font-semibold text-text-secondary hover:text-white hover:bg-white/5 rounded-xl flex items-center gap-3">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                View Transcript
            </a>`;
        }
        if (hasReport) {
            html += `<a href="/artifacts/report/${episodeId}" target="_blank" class="block w-full text-left px-4 py-3 text-sm font-semibold text-text-secondary hover:text-white hover:bg-white/5 rounded-xl flex items-center gap-3">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                View Report
            </a>`;
        }
        if (episodePage.canManage) {
        html += `<button onclick="closeActionSheet(); processEpisode(${episodeId}, false)" class="w-full text-left px-4 py-3 text-sm font-semibold text-text-secondary hover:text-white hover:bg-white/5 rounded-xl flex items-center gap-3">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
            Full Reprocess
        </button>`;
        if (hasTranscript) {
            html += `<button onclick="closeActionSheet(); processEpisode(${episodeId}, true)" class="w-full text-left px-4 py-3 text-sm font-semibold text-text-secondary hover:text-white hover:bg-white/5 rounded-xl flex items-center gap-3">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                Reprocess (Keep Transcript)
            </button>`;
        }
        html += `<button onclick="closeActionSheet(); deleteEpisode(${episodeId})" class="w-full text-left px-4 py-3 text-sm font-semibold text-red-400 hover:bg-red-500/10 rounded-xl flex items-center gap-3">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            Ignore Episode and Remove Files
        </button>`;

        }
        content.innerHTML = html;
        sheet.classList.remove('hidden');
        sheet.showModal();
    }

    function closeActionSheet() {
        const sheet = document.getElementById('action-sheet');
        sheet.close();
        sheet.classList.add('hidden');
    }

    window.onclick = function (event) {
        if (!event.target.closest('.relative.inline-block.text-left > button') && !event.target.closest('[id^=menu-]')) {
            document.querySelectorAll('[id^=menu-]').forEach(x => x.classList.add('hidden'));
        }
    }

    async function checkNow(id) {
        if (!await appConfirm('Check for new episodes now?')) return;
        try {
            await managementRequest(`/api/subscriptions/${id}/check`);
            appToast('Feed check triggered.', {type: 'success'});
            await fetchEpisodePage(true);
        } catch (error) { appToast(error.message, {type: 'error'}); }
    }
    function processEpisode(id, skipTranscription) {
        return episodeAction(id, `/api/episodes/${id}/reprocess?skip_transcription=${Boolean(skipTranscription)}`, 'Processing queued. Previous audio remains available.', 'Reprocess using the current settings?');
    }
    function cancelEpisode(id) {
        return episodeAction(id, `/api/episodes/${id}/cancel`, 'Cancellation requested.', 'Cancel processing and remove this episode’s local files?', 'POST', true);
    }

    async function deleteSubscription(id) {
        if (!await appConfirm('Delete this subscription and all local files for its episodes? This removes the subscription from the app.', { danger: true })) return;
        const res = await fetch(`/api/subscriptions/${id}`, { method: 'DELETE' });
        if (res.ok) window.location.href = '/';
        else appToast('Failed to delete subscription.', { type: 'error' });
    }

    if (episodePage.hasProcessing) {
    (function () {
        const toggle = document.getElementById('auto-refresh-toggle');
        const intervalInput = document.getElementById('refresh-interval');
        const countdownDisplay = document.getElementById('refresh-countdown');

        const savedInterval = localStorage.getItem('episodes-refresh-interval');
        const savedToggle = localStorage.getItem('episodes-refresh-enabled');
        if (savedInterval) intervalInput.value = savedInterval;
        if (savedToggle !== null) toggle.checked = savedToggle === 'true';

        let timeLeft = parseInt(intervalInput.value);
        let intervalId = null;

        function updateCountdown() {
            if (!toggle.checked) return;
            timeLeft--;
            if (timeLeft <= 0) {
                localStorage.setItem('episodes-refresh-enabled', toggle.checked);
                localStorage.setItem('episodes-refresh-interval', intervalInput.value);
                fetchEpisodePage(true);
                timeLeft = Number(intervalInput.value) || 30;
            } else { countdownDisplay.textContent = timeLeft + 's'; }
        }

        function startTimer() {
            if (intervalId) clearInterval(intervalId);
            timeLeft = parseInt(intervalInput.value);
            if (isNaN(timeLeft) || timeLeft < 5) timeLeft = 5;
            countdownDisplay.textContent = timeLeft + 's';
            if (toggle.checked) {
                countdownDisplay.classList.remove('bg-white/5', 'text-text-muted');
                countdownDisplay.classList.add('bg-primary-500/20', 'text-primary-400');
                intervalId = setInterval(updateCountdown, 1000);
            } else {
                countdownDisplay.textContent = 'Off';
                countdownDisplay.classList.remove('bg-primary-500/20', 'text-primary-400');
                countdownDisplay.classList.add('bg-white/5', 'text-text-muted');
            }
        }

        toggle.onchange = function () { localStorage.setItem('episodes-refresh-enabled', toggle.checked); startTimer(); };
        intervalInput.onchange = function () {
            let val = parseInt(this.value);
            if (isNaN(val) || val < 5) { val = 5; this.value = 5; }
            if (val > 300) { val = 300; this.value = 300; }
            localStorage.setItem('episodes-refresh-interval', val);
            startTimer();
        };
        startTimer();
    })();
    }
    // View Toggle Logic for Episodes
    document.addEventListener('DOMContentLoaded', () => {
        function setEpisodeView(mode) {
            const grid = document.getElementById('episodes-grid');
            if (!grid) return;

            // Set attribute
            grid.setAttribute('data-view', mode);

            // Save preference (reset to v4 for the new default)
            try {
                localStorage.setItem('episodes-view-mode-v4', mode);
            } catch (e) { }

            // Update buttons
            ['grid', 'list'].forEach(m => {
                const btn = document.getElementById(`view-${m}`);
                if (btn) {
                    if (m === mode) {
                        btn.classList.add('bg-primary-500', 'text-white');
                        btn.classList.remove('text-text-muted', 'hover:bg-white/[0.1]');
                    } else {
                        btn.classList.remove('bg-primary-500', 'text-white');
                        btn.classList.add('text-text-muted', 'hover:bg-white/[0.1]');
                    }
                }
            });
        }

        // Init view (using v4 to force the new Grid default)
        const savedView = localStorage.getItem('episodes-view-mode-v4') || 'grid';
        setEpisodeView(savedView);

        // Attach listeners
        ['grid', 'list'].forEach(m => {
            const btn = document.getElementById(`view-${m}`);
            if (btn) {
                btn.addEventListener('click', () => setEpisodeView(m));
            }
        });

        // Hoist toolbar to body to avoid stacking context/transform issues
        const toolbar = document.getElementById('batch-actions');
        if (toolbar) {
            document.body.appendChild(toolbar);
        }
    });

document.addEventListener('DOMContentLoaded', () => {
    const sheet = document.getElementById('action-sheet');
    if (currentFilter !== 'all') filterEpisodes(currentFilter);
    sheet?.addEventListener('cancel', event => { event.preventDefault(); closeActionSheet(); });
});
