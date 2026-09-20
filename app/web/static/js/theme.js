/* Loaded synchronously in <head> so the saved theme applies before first paint. */
(() => {
    'use strict';
    const key = 'podcast-ad-remover-theme';
    const root = document.documentElement;
    let theme = 'dark';
    try {
        if (localStorage.getItem(key) === 'light') theme = 'light';
    } catch (_) {
        // Storage can be disabled; the toggle still works for the current page.
    }

    function apply(value) {
        theme = value === 'light' ? 'light' : 'dark';
        root.classList.toggle('dark', theme === 'dark');
        root.dataset.theme = theme;
        document.querySelectorAll('[data-theme-toggle]').forEach(button => {
            button.setAttribute('aria-pressed', String(theme === 'light'));
            button.title = theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode';
            button.querySelector('[data-theme-label]').textContent = theme === 'light' ? 'Light' : 'Dark';
        });
    }
    apply(theme);
    document.addEventListener('DOMContentLoaded', () => {
        apply(theme);
        document.querySelectorAll('[data-theme-toggle]').forEach(button => {
            button.hidden = false;
            button.addEventListener('click', () => {
                apply(theme === 'light' ? 'dark' : 'light');
                try { localStorage.setItem(key, theme); } catch (_) { /* Optional persistence. */ }
            });
        });
    });
    window.addEventListener('storage', event => {
        if (event.key === key || event.key === null) apply(event.newValue);
    });
})();
