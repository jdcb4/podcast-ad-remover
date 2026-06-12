import html
import re
from datetime import datetime


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


def compact_datetime(value):
    """Render SQLite/Python timestamps compactly for admin tables."""
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

    now = datetime.now()
    if dt.year == now.year:
        return dt.strftime("%d %b %H:%M")
    return dt.strftime("%d %b %Y")
