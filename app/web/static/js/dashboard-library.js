(() => {
    function setButtonState(form, inLibrary) {
        const button = form.querySelector('button[type="submit"]');
        const action = form.querySelector('input[name="action"]');
        const icon = button?.querySelector('svg');
        if (!button || !action) return;

        action.value = inLibrary ? 'remove' : 'add';
        button.title = inLibrary ? 'Remove from My Podcasts' : 'Add to My Podcasts';
        button.setAttribute('aria-label', button.title);
        button.classList.toggle('text-primary-400', inLibrary);
        button.classList.toggle('text-text-muted', !inLibrary);
        if (icon) icon.setAttribute('fill', inLibrary ? 'currentColor' : 'none');
    }

    function updateCounts(card, result) {
        const membershipCount = card?.querySelector('.library-user-count');
        if (membershipCount) {
            const count = Number(result.user_library_count || 0);
            membershipCount.dataset.count = String(count);
            membershipCount.textContent = `${count} user${count === 1 ? '' : 's'}`;
        }

        const myCount = document.getElementById('my-podcast-count');
        if (myCount && result.changed) {
            const current = Number(myCount.textContent || 0);
            myCount.textContent = String(Math.max(0, current + (result.in_user_library ? 1 : -1)));
        }
    }

    async function submitMembership(form) {
        const button = form.querySelector('button[type="submit"]');
        const card = form.closest('.podcast-card');
        if (button) button.disabled = true;

        try {
            const response = await fetch(form.action, {
                method: 'POST',
                body: new FormData(form),
                headers: {
                    Accept: 'application/json',
                    'X-Requested-With': 'fetch',
                },
                credentials: 'same-origin',
            });
            if (!response.ok) throw new Error(`Request failed (${response.status})`);

            const result = await response.json();
            setButtonState(form, result.in_user_library);
            updateCounts(card, result);

            const grid = document.getElementById('podcast-grid');
            if (grid?.dataset.libraryView === 'mine' && !result.in_user_library && card) {
                card.remove();
                document.querySelector(
                    `.podcast-table-row[data-subscription-id="${result.subscription_id}"]`
                )?.remove();
            }

            document.dispatchEvent(new CustomEvent('library-membership-changed', { detail: result }));
            if (window.appToast) window.appToast(result.message, { type: 'success' });
        } catch (error) {
            if (window.appToast) {
                window.appToast('Could not update My Podcasts. Please try again.', { type: 'error' });
            }
        } finally {
            if (button && button.isConnected) button.disabled = false;
        }
    }

    document.addEventListener('submit', (event) => {
        const form = event.target.closest?.('.library-membership-form');
        if (!form) return;
        event.preventDefault();
        submitMembership(form);
    });
})();
