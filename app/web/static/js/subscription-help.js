(() => {
    const overlay = document.getElementById('subscription-help-overlay');
    const input = document.getElementById('subscription-help-url');
    const closeButton = document.getElementById('subscription-help-close');
    const copyButton = document.getElementById('subscription-help-copy');
    const copyStatus = document.getElementById('subscription-help-copy-status');
    let previousFocus = null;

    if (!overlay || !input || !closeButton || !copyButton || !copyStatus) return;

    function closeDialog() {
        overlay.classList.add('hidden');
        copyStatus.textContent = '';
        if (previousFocus && typeof previousFocus.focus === 'function') previousFocus.focus();
    }

    function openDialog(link) {
        previousFocus = link;
        input.value = link.dataset.feedUrl || '';
        copyStatus.textContent = '';
        overlay.classList.remove('hidden');
        copyButton.focus();
    }

    async function copyFeedUrl() {
        let copied = false;
        try {
            await navigator.clipboard.writeText(input.value);
            copied = true;
        } catch (error) {
            input.select();
            copied = document.execCommand('copy');
        }
        copyStatus.textContent = copied
            ? 'Feed URL copied.'
            : 'Could not copy automatically. Select the URL and copy it manually.';
    }

    document.addEventListener('click', (event) => {
        const link = event.target.closest?.('[data-subscription-help]');
        if (!link) return;
        event.preventDefault();
        openDialog(link);
    });
    closeButton.addEventListener('click', closeDialog);
    copyButton.addEventListener('click', copyFeedUrl);
    overlay.addEventListener('click', (event) => {
        if (event.target === overlay) closeDialog();
    });
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !overlay.classList.contains('hidden')) closeDialog();
    });
})();
