(() => {
    function storeOverride(control) {
        if (control.type === 'checkbox') {
            control.dataset.overrideChecked = String(control.checked);
        } else {
            control.dataset.overrideValue = control.value;
        }
    }

    function showValue(control, source) {
        if (control.type === 'checkbox') {
            control.checked = control.dataset[`${source}Checked`] === 'true';
        } else if (control.dataset[`${source}Value`] !== undefined) {
            control.value = control.dataset[`${source}Value`];
        }
    }

    function syncGroup(group, initial = false) {
        const toggle = group.querySelector('[data-inheritance-toggle]');
        if (!toggle) return;
        group.querySelectorAll('[data-inherited-control]').forEach((control) => {
            if (!initial) {
                if (toggle.checked) {
                    storeOverride(control);
                    showValue(control, 'effective');
                } else {
                    showValue(control, 'override');
                }
            }
            control.disabled = toggle.checked;
        });
        group.classList.toggle('opacity-75', toggle.checked);
    }

    document.querySelectorAll('[data-inheritance-group]').forEach((group) => {
        const toggle = group.querySelector('[data-inheritance-toggle]');
        if (!toggle) return;
        syncGroup(group, true);
        toggle.addEventListener('change', () => syncGroup(group));
    });
})();
