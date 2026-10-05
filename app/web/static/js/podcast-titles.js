(() => {
    const example = document.getElementById('podcast-title-example');
    const fields = ['prefix', 'suffix'].map(kind => ({
        kind,
        toggle: document.getElementById(`podcast_title_${kind}_enabled`),
        input: document.getElementById(`podcast_title_${kind}`),
    }));
    function update() {
        let title = 'The Rest is History';
        fields.forEach(({kind, toggle, input}) => {
            input.disabled = !toggle.checked;
            input.required = toggle.checked;
            const value = input.value;
            const invalid = /[<>\p{C}\u2028\u2029]/u.test(value);
            input.setCustomValidity(toggle.checked && (invalid || !value.trim())
                ? 'Enter a single line of plain text without markup or control characters.' : '');
            if (!toggle.checked || !value.trim()) return;
            if (kind === 'prefix') title = value + (/\s$/u.test(value) ? '' : ' ') + title;
            else title += (/^\s/u.test(value) ? '' : ' ') + value;
        });
        example.textContent = title;
    }
    fields.forEach(({toggle, input}) => {
        toggle.addEventListener('change', update);
        input.addEventListener('input', update);
    });
    update();
})();
