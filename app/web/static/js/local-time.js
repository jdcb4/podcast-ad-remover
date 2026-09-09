// Rewrites <time data-lt> elements (UTC instants, 'Z'-suffixed) into the
// viewer's local timezone. Formats: compact, datetime, date, relative.
//
// Load-order note (verified): this script is loaded with `defer` from
// base.html, ABOVE {% block content %}. A synchronous (non-deferred) inline
// <script> placed by a child template inside its content block still runs
// BEFORE this file executes, because deferred scripts don't run until the
// whole document has finished parsing. So `window.AppLocalTime` is
// undefined at that point. Inline consumers must only call
// `window.AppLocalTime.*` from async/event callbacks (fetch().then(...),
// DOMContentLoaded listeners, click handlers, etc.) that run after the page
// has finished loading, or guard the call with `if (window.AppLocalTime)`.
(function () {
    function parseIso(iso) {
        if (!iso) return null;
        // Accept both the 'Z'-suffixed instants this filter normally emits
        // and the raw naive-UTC shapes ("YYYY-MM-DD HH:MM:SS") that can leak
        // through when a server-side `utc_iso(...)` conversion fails and a
        // caller falls back to the raw DB string. Without this, `new Date()`
        // treats a bare (no zone) string as browser-local time, silently
        // misrendering by the viewer's UTC offset. An explicit offset
        // (e.g. '+05:00') is left alone and respected as-is.
        let s = String(iso).trim().replace(' ', 'T');
        if (!/[Zz]$|[+-]\d\d:?\d\d$/.test(s)) s += 'Z';
        const date = new Date(s);
        return isNaN(date.getTime()) ? null : date;
    }

    function formatCompact(date) {
        const now = new Date();
        if (date.getFullYear() !== now.getFullYear()) {
            return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
        }
        return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ' ' +
            date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    }

    function formatDateTime(date) {
        return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) + ' ' +
            date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    }

    function formatRelative(date) {
        const now = new Date();
        if (now.toDateString() === date.toDateString()) return 'Today';
        const yesterday = new Date(now);
        yesterday.setDate(now.getDate() - 1);
        if (yesterday.toDateString() === date.toDateString()) return 'Yesterday';
        const seconds = Math.round((now - date) / 1000);
        if (seconds > 0 && seconds < 86400 * 7) {
            const days = Math.floor(seconds / 86400);
            return days + (days === 1 ? ' day ago' : ' days ago');
        }
        const opts = { month: 'short', day: 'numeric' };
        if (date.getFullYear() !== now.getFullYear()) opts.year = 'numeric';
        return date.toLocaleDateString(undefined, opts);
    }

    function format(date, fmt) {
        if (fmt === 'relative') return formatRelative(date);
        if (fmt === 'date') return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
        if (fmt === 'datetime') return formatDateTime(date);
        return formatCompact(date);
    }

    function formatIso(iso, fmt) {
        const date = parseIso(iso);
        return date ? format(date, fmt) : (iso || '');
    }

    function hydrate(root) {
        (root || document).querySelectorAll('time[data-lt]').forEach(function (el) {
            const raw = el.getAttribute('datetime');
            const date = parseIso(raw);
            if (!date) return;
            el.textContent = format(date, el.getAttribute('data-lt'));
            // Local rendering first (Task 10's smoke test hovers for a local
            // tooltip), but keep the UTC instant alongside it rather than
            // discarding Task 4's deliberate no-JS/pre-hydration UTC marker.
            el.title = date.toLocaleString() + ' (UTC: ' + raw + ')';
        });
    }

    window.AppLocalTime = { hydrate: hydrate, formatIso: formatIso };
    document.addEventListener('DOMContentLoaded', function () { hydrate(document); });
    document.addEventListener('dashboard-library-results-changed', function () {
        hydrate(document.getElementById('dashboard-podcast-results'));
    });
})();
