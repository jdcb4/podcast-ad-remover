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
        const controls = [...group.querySelectorAll('[data-inherited-control]')];
        if (group.closest('#podcast-settings-workspace')) {
            document.querySelectorAll('[data-inheritance-source="' + toggle.name.replace('inherit_', '') + '"] [data-inherited-control]').forEach(control => controls.push(control));
        }
        controls.forEach((control) => {
            if (!initial) {
                if (toggle.checked) {
                    storeOverride(control);
                    showValue(control, 'effective');
                } else {
                    showValue(control, 'override');
                }
            }
            control.disabled = toggle.checked || control.hasAttribute('data-unavailable');
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
