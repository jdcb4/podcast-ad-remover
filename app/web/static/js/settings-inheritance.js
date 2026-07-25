(() => {
    function syncGroup(group) {
        const toggle = group.querySelector('[data-inheritance-toggle]');
        if (!toggle) return;
        group.querySelectorAll('[data-inherited-control]').forEach((control) => {
            control.disabled = toggle.checked;
        });
        group.classList.toggle('opacity-75', toggle.checked);
    }

    document.querySelectorAll('[data-inheritance-group]').forEach((group) => {
        const toggle = group.querySelector('[data-inheritance-toggle]');
        if (!toggle) return;
        syncGroup(group);
        toggle.addEventListener('change', () => syncGroup(group));
    });
})();
