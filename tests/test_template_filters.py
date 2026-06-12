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


def test_lazy_episode_template_escapes_dynamic_episode_fields():
    template_source = Path("app/web/templates/episodes.html").read_text(encoding="utf-8")

    assert "function escapeHtml(value)" in template_source
    assert "function encodePathSegment(value)" in template_source
    assert "const allowedStatuses =" in template_source
    assert "const title = escapeHtml(ep.title || '')" in template_source
    assert "const description = escapeHtml(ep.description || '')" in template_source
    assert "const pubDate = escapeHtml(ep.pub_date || '')" in template_source
    assert "${ep.title}" not in template_source
    assert "${ep.description}" not in template_source
    assert "${ep.pub_date}" not in template_source
    assert "/audio/${subscriptionSlug}/${guid}/${filename}" not in template_source


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
