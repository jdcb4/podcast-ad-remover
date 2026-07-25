(() => {
    const form = document.getElementById('bulk-settings-form');
    if (!form) return;

    const checkboxes = () => Array.from(document.querySelectorAll('.podcast-bulk-checkbox'));
    const selectAll = document.getElementById('select-all-podcasts');
    const count = document.getElementById('bulk-selection-count');

    function syncSelection() {
        const available = checkboxes().filter((checkbox) => checkbox.closest('tr')?.style.display !== 'none');
        const selected = checkboxes().filter((checkbox) => checkbox.checked);
        form.classList.toggle('hidden', selected.length === 0);
        if (count) count.textContent = String(selected.length);
        if (selectAll) {
            selectAll.checked = available.length > 0 && available.every((checkbox) => checkbox.checked);
            selectAll.indeterminate = available.some((checkbox) => checkbox.checked) && !selectAll.checked;
        }
    }

    selectAll?.addEventListener('change', () => {
        checkboxes().forEach((checkbox) => {
            if (checkbox.closest('tr')?.style.display !== 'none') checkbox.checked = selectAll.checked;
        });
        syncSelection();
    });
    document.addEventListener('change', (event) => {
        if (event.target.matches('.podcast-bulk-checkbox')) syncSelection();
    });

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
        if (window.appConfirm) {
            event.preventDefault();
            const confirmed = await window.appConfirm(
                `Apply the selected bulk settings to ${selected} podcast${selected === 1 ? '' : 's'}?`
            );
            if (confirmed) HTMLFormElement.prototype.submit.call(form);
        }
    });

    syncSelection();
})();
