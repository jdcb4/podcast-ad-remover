import re
from pathlib import Path

from app.web.template_filters import simple_markdown


def test_router_registers_standalone_template_filters():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")

    assert "templates.env.filters['simple_markdown'] = safe_simple_markdown" in router_source
    assert "templates.env.filters['clean_description'] = safe_clean_description" in router_source


def test_simple_markdown_escapes_html_before_safe_rendering():
    rendered = simple_markdown('<script>alert("x")</script> **safe**')

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<strong>safe</strong>" in rendered


def test_simple_markdown_escapes_bullet_content():
    rendered = simple_markdown("- <img src=x onerror=alert(1)>")

    assert "<img" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert '<li class="mb-1">' in rendered


def test_podcast_search_template_escapes_dynamic_result_fields():
    template_source = Path("app/web/templates/index.html").read_text(encoding="utf-8")

    assert "function escapeHtml(value)" in template_source
    assert "const image = escapeHtml(pod.image || '')" in template_source
    assert "const title = escapeHtml(pod.title || 'Untitled podcast')" in template_source
    assert "const description = escapeHtml(pod.description || '')" in template_source
    assert "const feedUrl = escapeHtml(pod.feed_url || '')" in template_source
    assert "${pod.title}" not in template_source
    assert "${pod.description}" not in template_source
    assert "${pod.feed_url}" not in template_source


def test_admin_ai_template_escapes_dynamic_model_names():
    template_source = Path("app/web/templates/admin/ai.html").read_text(encoding="utf-8")

    assert "function escapeHtml(value)" in template_source
    assert "<span>${escapeHtml(m)}</span>" in template_source
    assert "onclick=\"shuttleManager.remove('${provider}', '${m}')\"" not in template_source
    assert "div.querySelector('button').onclick = () => this.remove(provider, m)" in template_source


def test_admin_logs_template_escapes_lines_before_highlighting():
    template_source = Path("app/web/templates/admin/logs.html").read_text(encoding="utf-8")

    assert "function escapeHtml(value)" in template_source
    assert "const escapedLine = escapeHtml(line)" in template_source
    assert "${line}</span>" not in template_source
    assert "return escapedLine;" in template_source


def test_admin_prompts_alerts_use_text_content():
    template_source = Path("app/web/templates/admin/prompts.html").read_text(encoding="utf-8")

    assert "appToast(message, { type })" in template_source
    assert "alertContainer.innerHTML = `<div" not in template_source


def test_templates_use_app_notifications_instead_of_browser_popups():
    template_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("app/web/templates").rglob("*.html")
    )

    assert "function (message, options = {})" in template_sources
    assert "window.appToast" in template_sources
    assert "window.appConfirm" in template_sources
    assert "window.appPrompt" in template_sources
    assert "alert(" not in template_sources
    assert "confirm(" not in template_sources
    assert "prompt(" not in template_sources


from datetime import datetime as _datetime

from app.web.template_filters import local_time, utc_isoformat


def test_local_time_wraps_value_in_time_element():
    rendered = str(local_time("2026-08-27 17:47:12"))

    assert rendered.startswith('<time datetime="2026-08-27T17:47:12Z"')
    assert 'data-lt="compact"' in rendered
    assert rendered.endswith("</time>")


def test_local_time_accepts_datetime_objects_and_format_arg():
    rendered = str(local_time(_datetime(2026, 8, 27, 17, 47, 12), "relative"))

    assert 'datetime="2026-08-27T17:47:12Z"' in rendered
    assert 'data-lt="relative"' in rendered


def test_local_time_fallback_body_is_the_compact_rendering():
    rendered = str(local_time("2025-08-27 17:47:12"))

    assert ">27 Aug 2025</time>" in rendered


def test_local_time_datetime_format_fallback_body_includes_time_and_year_current_year():
    rendered = str(local_time("2026-08-27 17:47:12", "datetime"))

    assert ">27 Aug 2026 17:47</time>" in rendered


def test_local_time_datetime_format_fallback_body_includes_time_for_past_year():
    """The motivating bug: a past-year datetime fallback must not drop the clock time."""
    rendered = str(local_time("2025-08-27 17:47:12", "datetime"))

    assert ">27 Aug 2025 17:47</time>" in rendered


def test_local_time_datetime_format_fallback_body_accepts_datetime_objects():
    rendered = str(local_time(_datetime(2025, 8, 27, 17, 47, 12), "datetime"))

    assert ">27 Aug 2025 17:47</time>" in rendered


def test_local_time_datetime_format_fallback_body_accepts_fractional_seconds_shape():
    rendered = str(local_time("2025-08-27T17:47:12.123456", "datetime"))

    assert ">27 Aug 2025 17:47</time>" in rendered


def test_local_time_date_format_fallback_body_is_date_only():
    rendered = str(local_time("2026-08-27 17:47:12", "date"))

    assert rendered.endswith(">27 Aug 2026</time>")


def test_local_time_date_format_fallback_body_matches_across_years():
    rendered = str(local_time("2025-08-27 17:47:12", "date"))

    assert ">27 Aug 2025</time>" in rendered


def test_local_time_relative_format_fallback_body_is_the_compact_rendering():
    """relative can't know the viewer's 'today' server-side; fall back to compact_datetime."""
    rendered = str(local_time("2025-08-27 17:47:12", "relative"))

    assert ">27 Aug 2025</time>" in rendered


def test_local_time_relative_format_fallback_body_current_year_is_compact():
    rendered = str(local_time("2026-08-27 17:47:12", "relative"))

    assert ">27 Aug 17:47</time>" in rendered


def test_local_time_handles_empty_and_garbage():
    assert local_time(None) == "Never"
    assert local_time("") == "Never"
    assert local_time("not a date") == "not a date"


def test_utc_isoformat_returns_plain_attribute_string():
    assert utc_isoformat("2026-08-27 17:47:12") == "2026-08-27T17:47:12Z"
    assert utc_isoformat(None) == ""


def test_router_registers_local_time_filters():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")

    assert "templates.env.filters['local_time'] = local_time" in router_source
    assert "templates.env.filters['utc_isoformat'] = utc_isoformat" in router_source


import jinja2


def _autoescaping_env():
    env = jinja2.Environment(autoescape=True)
    env.filters['local_time'] = local_time
    env.filters['utc_isoformat'] = utc_isoformat
    return env


def test_local_time_renders_a_literal_time_element_through_jinja():
    rendered = _autoescaping_env().from_string("{{ v|local_time }}").render(
        v="2026-08-27 17:47:12"
    )

    assert rendered.startswith("<time ")
    assert "&lt;time" not in rendered


def test_local_time_hostile_fmt_cannot_break_out_of_data_lt_attribute():
    rendered = _autoescaping_env().from_string("{{ v|local_time(f) }}").render(
        v="2026-08-27 17:47:12", f='"><script>x</script>'
    )

    assert "<script>" not in rendered
    assert 'data-lt="&#34;&gt;&lt;script&gt;x&lt;/script&gt;"' in rendered


def test_local_time_unparseable_hostile_value_is_escaped_through_jinja():
    rendered = _autoescaping_env().from_string("{{ v|local_time }}").render(
        v="<script>x</script>"
    )

    assert "<script>" not in rendered
    assert rendered == "&lt;script&gt;x&lt;/script&gt;"


def test_local_time_safe_filter_does_not_reopen_the_footgun():
    rendered = _autoescaping_env().from_string("{{ v|local_time|safe }}").render(
        v="<script>x</script>"
    )

    assert "<script>" not in rendered
    assert rendered == "&lt;script&gt;x&lt;/script&gt;"


def test_local_time_safe_filter_is_a_no_op_on_the_success_path():
    rendered = _autoescaping_env().from_string("{{ v|local_time|safe }}").render(
        v="2026-08-27 17:47:12"
    )

    assert rendered == (
        '<time datetime="2026-08-27T17:47:12Z" data-lt="compact" '
        'title="2026-08-27T17:47:12Z">27 Aug 17:47</time>'
    )


# `name="... |local_time ..."` — a local_time call inside a quoted HTML
# attribute value. Deliberately simple, and blind to anything that puts a
# quote character in the attribute value ahead of the `|local_time` it's
# looking for: an apostrophe in ordinary prose
# (`title="Don't miss: {{ v|local_time }}"`), an earlier filter's own
# quoted argument (`|default('')|local_time`), an unquoted attribute
# (`data-x={{ v|local_time }}`), or `title = "..."` with spaces around
# `=`. Reviewers confirmed none of those patterns occur in the current
# templates and no near-misses exist, so it still catches the realistic
# mistake of reaching for the element filter in a title=""/aria-label=""
# slot without a false sense of completeness beyond that.
_LOCAL_TIME_IN_ATTRIBUTE = re.compile(r'\b[\w-]+=(["\'])[^"\']*\|\s*local_time\b')


def test_no_template_pipes_local_time_into_an_html_attribute():
    """local_time renders a whole <time ...> element, quotes included.

    Inside an attribute value its first quote terminates the attribute and
    corrupts the enclosing tag. utc_isoformat is the attribute-safe filter,
    and local_time's docstring says so — this makes it enforceable.
    """
    offenders = []
    for path in sorted(Path("app/web/templates").rglob("*.html")):
        source = path.read_text(encoding="utf-8")
        offenders += [
            f"{path}: {match.group(0)!r}"
            for match in _LOCAL_TIME_IN_ATTRIBUTE.finditer(source)
        ]

    assert offenders == [], (
        "local_time emits a <time> element and corrupts the enclosing tag; "
        "use utc_isoformat in attribute context:\n" + "\n".join(offenders)
    )
