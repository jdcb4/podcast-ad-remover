(() => {
    const form = document.getElementById('bulk-settings-form');
    if (!form) return;

    const checkboxes = () => Array.from(document.querySelectorAll('.podcast-bulk-checkbox'));
    const selectAll = () => document.getElementById('select-all-podcasts');
    const count = document.getElementById('bulk-selection-count');
    const exportButton = document.getElementById('export-opml');
    const exportFields = document.getElementById('opml-selected-podcasts');
    const selectedPodcasts = () => Array.from(new Map(checkboxes()
        .filter((checkbox) => checkbox.checked).map((checkbox) => [checkbox.value, checkbox])).values());

    function syncSelection() {
        const available = checkboxes().filter((checkbox) => {
            const row = checkbox.closest('tr, article');
            return !row?.hidden && row?.style.display !== 'none';
        });
        const selected = selectedPodcasts();
        const canManage = selected.every((checkbox) => checkbox.dataset.canManage !== 'false');
        form.classList.toggle('hidden', selected.length === 0 || !canManage);
        const disclosure = document.getElementById('bulk-settings-details');
        if (disclosure) {
            disclosure.classList.toggle('hidden', selected.length === 0 || !canManage);
            if (!selected.length || !canManage) disclosure.open = false;
        }
        if (count) count.textContent = String(selected.length);
        if (exportButton) exportButton.disabled = selected.length === 0;
        const exportOnly = document.getElementById('bulk-export-only');
        if (exportOnly) exportOnly.hidden = selected.length === 0 || canManage;
        if (exportFields) {
            exportFields.replaceChildren(...selected.map((checkbox) => {
                const field = document.createElement('input');
                field.type = 'hidden'; field.name = 'subscription_ids'; field.value = checkbox.value;
                return field;
            }));
        }
        const selectAllCheckbox = selectAll();
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = available.length > 0 && available.every((checkbox) => checkbox.checked);
            selectAllCheckbox.indeterminate = available.some((checkbox) => checkbox.checked) && !selectAllCheckbox.checked;
        }
    }

    document.addEventListener('change', (event) => {
        if (event.target.matches('#select-all-podcasts')) {
            checkboxes().forEach((checkbox) => {
                const row = checkbox.closest('tr, article');
                if (!row?.hidden && row?.style.display !== 'none') checkbox.checked = event.target.checked;
            });
            syncSelection();
            return;
        }
        if (event.target.matches('.podcast-bulk-checkbox')) {
            checkboxes().filter((checkbox) => checkbox.value === event.target.value)
                .forEach((checkbox) => { checkbox.checked = event.target.checked; });
            syncSelection();
        }
    });
    document.addEventListener('dashboard-library-results-changed', syncSelection);
    document.addEventListener('library-membership-changed', () => queueMicrotask(syncSelection));

    form.querySelectorAll('[data-bulk-mode]').forEach((mode) => {
        const controls = mode.parentElement.querySelector('[data-bulk-controls]');
        const sync = () => {
            if (!controls) return;
            const enabled = mode.value === 'override';
            controls.querySelectorAll('input, select, textarea').forEach((control) => {
                control.disabled = !enabled || control.hasAttribute('data-unavailable');
            });
            controls.classList.toggle('opacity-50', !enabled);
        };
        mode.addEventListener('change', sync);
        sync();
    });

    const ownerMode = form.querySelector('[data-owner-mode]');
    const ownerSelect = form.querySelector('[data-owner-select]');
    ownerMode?.addEventListener('change', () => {
        if (ownerSelect) ownerSelect.disabled = ownerMode.value !== 'set';
    });

    form.addEventListener('submit', async (event) => {
        const selected = selectedPodcasts().length;
        if (!selected) {
            event.preventDefault();
            return;
        }

        event.preventDefault();
        if (selectedPodcasts().some((checkbox) => checkbox.dataset.canManage === 'false')) return;
        const submitter = event.submitter;
        const deleting = submitter?.dataset.bulkAction === 'delete';
        const confirmation = form.querySelector('[data-delete-confirmation]');
        if (confirmation) confirmation.value = '';

        const message = deleting
            ? `Permanently delete ${selected} selected podcast${selected === 1 ? '' : 's'}?\n\nThis removes downloaded audio, processed files, transcripts, reports, generated feeds, artwork, and related database records. This cannot be undone from the app.`
            : `Apply the selected bulk settings to ${selected} podcast${selected === 1 ? '' : 's'}?`;
        const options = deleting ? { danger: true } : {};
        const confirmed = window.appConfirm
            ? await window.appConfirm(message, options)
            : window.confirm(message);
        if (confirmed) {
            if (deleting && confirmation) confirmation.value = 'delete';
            form.setAttribute(
                'action',
                deleting ? '/subscriptions/bulk-delete' : '/subscriptions/bulk-settings'
            );
            form.setAttribute('method', 'post');
            HTMLFormElement.prototype.submit.call(form);
        }
    });

    syncSelection();
})();
