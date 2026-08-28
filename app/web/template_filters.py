import html
import re
from datetime import datetime

from markupsafe import Markup, escape

from app.core.time_utils import now_utc, utc_iso


def format_duration(seconds):
    if not seconds:
        return '-'
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes}:{seconds:02d}'


def simple_markdown(text):
    """Convert a small safe markdown subset to HTML."""
    if not text:
        return ""

    lines = text.replace("\r\n", "\n").split("\n")
    result = []
    in_list = False
    list_items = []

    def apply_bold(value):
        escaped = html.escape(value)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*(.+?)\*", r"<strong>\1</strong>", escaped)
        return escaped

    def flush_list():
        nonlocal in_list, list_items
        if in_list and list_items:
            list_html = (
                '<ul class="list-disc ml-8 space-y-2 mb-4 text-white/90" '
                'style="list-style-type: disc; margin-left: 2rem; margin-bottom: 1rem;">\n'
                + "\n".join(list_items)
                + "\n</ul>"
            )
            result.append(list_html)
            list_items = []
            in_list = False

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if in_list:
                flush_list()
            continue

        bullet_match = re.match(r"^([\*\-\u2022]|\d+\.)\s+(.+)$", stripped_line)

        if bullet_match:
            if not in_list:
                in_list = True
            content = apply_bold(bullet_match.group(2).strip())
            list_items.append(f'<li class="mb-1">{content}</li>')
        else:
            if in_list:
                flush_list()
            processed_line = apply_bold(stripped_line)
            result.append(f'<p class="mb-2 text-white/90 leading-relaxed">{processed_line}</p>')

    if in_list:
        flush_list()

    return "\n".join(result)


def clean_description(text):
    """Clean episode description: remove URLs, sponsor footers, and HTML tags."""
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    lines = text.split("\n")
    cleaned_lines = []
    cutoff_keywords = [
        "Sponsors:",
        "Support the show:",
        "Brought to you by:",
        "Advertise with us:",
        "See omnystudio.com/listener",
    ]

    for line in lines:
        stripped = line.strip()
        if any(keyword in stripped for keyword in cutoff_keywords):
            break
        if stripped:
            cleaned_lines.append(stripped)

    result = " ".join(cleaned_lines)
    return re.sub(r"\s+", " ", result).strip()


def _parse_for_display(value):
    """Parse value the way every fallback renderer below needs it parsed.

    Returns a `datetime` on success. On failure it returns the same
    degraded string a caller should show verbatim: "Never" for falsy
    input, `value[:16]` for a string that doesn't parse, or `str(value)`
    for a non-string, non-datetime value. Callers must check
    `isinstance(result, datetime)` before formatting.
    """
    if not value:
        return "Never"

    dt = value
    if isinstance(value, str):
        normalized = value.replace("T", " ").split(".")[0]
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return value[:16]

    if not isinstance(dt, datetime):
        return str(value)

    return dt


def compact_datetime(value):
    """Render SQLite/Python timestamps compactly.

    This is local_time's server-rendered fallback body for fmt='compact'
    (and, for lack of a better option, fmt='relative' — see
    _FALLBACK_RENDERERS). It remains registered as a standalone Jinja
    filter, but no template calls it directly anymore.
    """
    dt = _parse_for_display(value)
    if not isinstance(dt, datetime):
        return dt

    now = now_utc()
    if dt.year == now.year:
        return dt.strftime("%d %b %H:%M")
    return dt.strftime("%d %b %Y")


def _datetime_fallback(value):
    """Fallback body for fmt='datetime': date and time, always with the year.

    Matches the shape of local-time.js's formatDateTime() so the
    pre-hydration render doesn't visibly reformat once JS lands, and so a
    JS-less viewer still sees a time of day even for a past-year value.
    """
    dt = _parse_for_display(value)
    if not isinstance(dt, datetime):
        return dt
    return dt.strftime("%d %b %Y %H:%M")


def _date_fallback(value):
    """Fallback body for fmt='date': date only, matching local-time.js's date format."""
    dt = _parse_for_display(value)
    if not isinstance(dt, datetime):
        return dt
    return dt.strftime("%d %b %Y")


# Which renderer produces local_time's server-rendered fallback body for
# each fmt. Keyed by fmt so the body's shape always matches what
# local-time.js's hydrator will replace it with, instead of local_time
# always reaching for compact_datetime regardless of fmt.
#
# 'relative' has no server-side renderer of its own: relative wording
# ("Today", "3 days ago") depends on the viewer's "today", which the
# server cannot know. A concrete timestamp (compact_datetime) is the
# honest fallback rather than a guess that may be wrong once hydrated.
_FALLBACK_RENDERERS = {
    "datetime": _datetime_fallback,
    "date": _date_fallback,
    "compact": compact_datetime,
    "relative": compact_datetime,
}


def local_time(value, fmt="compact"):
    """Render a stored naive-UTC timestamp as a localized <time> element.

    The datetime attribute carries the UTC instant ('...Z'); local-time.js
    rewrites the text into the viewer's timezone on load. The body holds a
    server-rendered UTC fallback for no-JS rendering, and the title
    attribute carries that same UTC instant explicitly so a viewer without
    JS (or looking at the page in the instant before hydration) isn't
    misled into reading it as their own local time. fmt is one of:
    compact, datetime, date, relative.

    Always returns markupsafe.Markup, fully escaped: never chain |safe
    onto this filter's output, and never use it inside an HTML attribute
    context (e.g. `<div title="{{ v|local_time }}">`) — the value is
    itself a full <time> element with its own quoted attributes, and
    embedding it inside another attribute corrupts the surrounding tag.
    Use utc_isoformat there instead.
    """
    if not value:
        return Markup(escape("Never"))
    iso = utc_iso(value)
    if iso is None:
        return Markup(escape(str(value)))
    renderer = _FALLBACK_RENDERERS.get(fmt, compact_datetime)
    return Markup('<time datetime="{}" data-lt="{}" title="{}">{}</time>').format(
        iso, fmt, iso, renderer(value)
    )


def utc_isoformat(value):
    """Plain 'YYYY-MM-DDTHH:MM:SSZ' string for data-* attributes (no markup)."""
    return utc_iso(value) or ""
