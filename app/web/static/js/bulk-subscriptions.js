(() => {
    const form = document.getElementById('bulk-settings-form');
    if (!form) return;

    const checkboxes = () => Array.from(document.querySelectorAll('.podcast-bulk-checkbox'));
    const selectAll = () => document.getElementById('select-all-podcasts');
    const count = document.getElementById('bulk-selection-count');

    function syncSelection() {
        const available = checkboxes().filter((checkbox) => checkbox.closest('tr')?.style.display !== 'none');
        const selected = checkboxes().filter((checkbox) => checkbox.checked);
        form.classList.toggle('hidden', selected.length === 0);
        if (count) count.textContent = String(selected.length);
        const selectAllCheckbox = selectAll();
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = available.length > 0 && available.every((checkbox) => checkbox.checked);
            selectAllCheckbox.indeterminate = available.some((checkbox) => checkbox.checked) && !selectAllCheckbox.checked;
        }
    }

    document.addEventListener('change', (event) => {
        if (event.target.matches('#select-all-podcasts')) {
            checkboxes().forEach((checkbox) => {
                if (checkbox.closest('tr')?.style.display !== 'none') checkbox.checked = event.target.checked;
            });
            syncSelection();
            return;
        }
        if (event.target.matches('.podcast-bulk-checkbox')) syncSelection();
    });
    document.addEventListener('dashboard-library-results-changed', syncSelection);

    form.querySelectorAll('[data-bulk-mode]').forEach((mode) => {
        const controls = mode.parentElement.querySelector('[data-bulk-controls]');
        const sync = () => {
            if (!controls) return;
            const enabled = mode.value === 'override';
            controls.querySelectorAll('input, select, textarea').forEach((control) => {
                control.disabled = !enabled;
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
        const selected = checkboxes().filter((checkbox) => checkbox.checked).length;
        if (!selected) {
            event.preventDefault();
            return;
        }

        event.preventDefault();
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
