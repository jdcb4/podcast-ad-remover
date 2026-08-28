"""Task 9: dashboard, transcript, and public subscribe pages render dates
localized via local_time, and the dashboard's data-recent attribute (fed
from router.py's latest_episode_date) is Z-suffixed UTC.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX_TEMPLATE = Path("app/web/templates/index.html")
TRANSCRIPT_TEMPLATE = Path("app/web/templates/transcript.html")
PUBLIC_SUBSCRIBE_TEMPLATE = Path("app/web/templates/public_subscribe.html")
ROUTER = Path("app/web/router.py")


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_line(source: str, needle: str) -> str:
    return next(line for line in source.splitlines() if needle in line)


# --- Source assertions: new filter present, old constructs gone -----------


def test_dashboard_recent_episode_list_uses_local_time_and_drops_split():
    source = _source(INDEX_TEMPLATE)

    assert "{% if ep.published_date %}{{ ep.published_date|local_time('date') }}{% endif %}" in source
    # The old two-branch split(' ')[0] / bare-value construct must be gone.
    assert "published_date.split(' ')" not in source
    assert "ep.published_date is string" not in source
    assert "{{ ep.published_date }}" not in source


def test_transcript_page_uses_local_time_and_drops_bare_pub_date():
    source = _source(TRANSCRIPT_TEMPLATE)

    assert "{{ episode.pub_date|local_time('date') }}" in source
    # The bare, unfiltered interpolation must not remain anywhere.
    assert "{{ episode.pub_date }}" not in source


def test_public_subscribe_badge_uses_local_time_and_drops_truncate_chain():
    source = _source(PUBLIC_SUBSCRIBE_TEMPLATE)

    assert "{{ item.latest_episode.pub_date|local_time('date') }}" in source
    assert "|string|truncate(10, True, '')" not in source


def test_local_time_never_lands_in_an_html_attribute_at_these_three_sites():
    """Belt-and-suspenders companion to the repo-wide attribute guard: confirm
    none of the three edits accidentally put local_time inside quotes.
    """
    import re

    for path in (INDEX_TEMPLATE, TRANSCRIPT_TEMPLATE, PUBLIC_SUBSCRIBE_TEMPLATE):
        source = _source(path)
        assert not re.search(r'="[^"]*\|\s*local_time', source), path


def test_router_latest_episode_date_uses_utc_iso_not_raw_isoformat():
    source = _source(ROUTER)

    assert "latest_episode_date = utc_iso(latest_any_ep['pub_date']) or str(latest_any_ep['pub_date'])" in source
    # The old ad-hoc hasattr/isoformat branching must be gone.
    assert "hasattr(d, 'isoformat')" not in source
    assert "d.isoformat()" not in source


# --- Real Jinja rendering: truthy / falsy / unparseable, all three sites --


@pytest.fixture()
def env():
    from app.web.router import templates

    return templates.env


def test_dashboard_recent_line_renders_time_element_for_a_real_date(env):
    line = _find_line(_source(INDEX_TEMPLATE), "ep.published_date|local_time")
    template = env.from_string(line)

    html = template.render(ep={"published_date": "2026-01-15 09:30:00"}).strip()

    assert html.startswith("<time ")
    assert html.endswith("</time>")
    assert 'datetime="2026-01-15T09:30:00Z"' in html
    assert 'data-lt="date"' in html
    assert 'title="2026-01-15T09:30:00Z"' in html
    assert "15 Jan 2026" in html


def test_dashboard_recent_line_renders_nothing_for_a_falsy_date(env):
    """The {% if %} guard means a falsy published_date renders empty, not
    'Never' -- local_time's falsy branch is never reached at this site.
    """
    line = _find_line(_source(INDEX_TEMPLATE), "ep.published_date|local_time")
    template = env.from_string(line)

    html = template.render(ep={"published_date": None}).strip()

    assert html == ""


def test_dashboard_recent_line_falls_back_for_an_unparseable_date(env):
    line = _find_line(_source(INDEX_TEMPLATE), "ep.published_date|local_time")
    template = env.from_string(line)

    html = template.render(ep={"published_date": "not-a-date"}).strip()

    assert html == "not-a-date"
    assert "<time" not in html


def test_transcript_line_renders_time_element_for_a_real_date(env):
    line = _find_line(_source(TRANSCRIPT_TEMPLATE), "episode.pub_date|local_time")
    template = env.from_string(line)

    html = template.render(
        episode={"pub_date": "2026-01-15 09:30:00", "duration": 0},
        format_duration=lambda seconds: "-",
    ).strip()

    assert "<time datetime=\"2026-01-15T09:30:00Z\" data-lt=\"date\" title=\"2026-01-15T09:30:00Z\">15 Jan 2026</time>" in html
    # Surrounding markup (the bullet separator and duration) survives intact.
    assert "• -" in html


def test_transcript_line_shows_never_for_a_falsy_pub_date(env):
    line = _find_line(_source(TRANSCRIPT_TEMPLATE), "episode.pub_date|local_time")
    template = env.from_string(line)

    html = template.render(
        episode={"pub_date": None, "duration": 0},
        format_duration=lambda seconds: "-",
    ).strip()

    assert "Never" in html
    assert "<time" not in html
    # Before this task the bare `{{ episode.pub_date }}` would have rendered
    # the literal string "None" here -- confirm that regression is gone too.
    assert "None" not in html


def test_transcript_line_falls_back_for_an_unparseable_pub_date(env):
    line = _find_line(_source(TRANSCRIPT_TEMPLATE), "episode.pub_date|local_time")
    template = env.from_string(line)

    html = template.render(
        episode={"pub_date": "not-a-date", "duration": 0},
        format_duration=lambda seconds: "-",
    ).strip()

    assert "not-a-date" in html
    assert "<time" not in html


def test_public_subscribe_badge_renders_time_element_for_a_real_date(env):
    line = _find_line(_source(PUBLIC_SUBSCRIBE_TEMPLATE), "item.latest_episode.pub_date|local_time")
    template = env.from_string(line)

    html = template.render(item={"latest_episode": {"pub_date": "2026-01-15 09:30:00"}}).strip()

    assert html == (
        '<span class="badge-soft">Latest: '
        '<time datetime="2026-01-15T09:30:00Z" data-lt="date" title="2026-01-15T09:30:00Z">15 Jan 2026</time>'
        "</span>"
    )


def test_public_subscribe_badge_shows_never_for_a_falsy_pub_date(env):
    line = _find_line(_source(PUBLIC_SUBSCRIBE_TEMPLATE), "item.latest_episode.pub_date|local_time")
    template = env.from_string(line)

    html = template.render(item={"latest_episode": {"pub_date": None}}).strip()

    assert "Never" in html
    assert "<time" not in html


def test_public_subscribe_badge_falls_back_for_an_unparseable_pub_date(env):
    line = _find_line(_source(PUBLIC_SUBSCRIBE_TEMPLATE), "item.latest_episode.pub_date|local_time")
    template = env.from_string(line)

    html = template.render(item={"latest_episode": {"pub_date": "not-a-date"}}).strip()

    assert "not-a-date" in html
    assert "<time" not in html


# --- Real behavioural test: the dashboard's data-recent attribute ---------


@pytest.mark.asyncio
async def test_dashboard_data_recent_attribute_is_z_suffixed_utc(isolated_data_dir):
    """End-to-end through the real app: GET '/' with a seeded episode and
    confirm both data-recent attributes (table row + grid card) carry the
    Z-suffixed instant, not a bare local-naive string.
    """
    from httpx import ASGITransport, AsyncClient

    from app.infra.database import get_db_connection, init_db
    from app.main import app

    init_db()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id, feed_url, title, slug) "
            "VALUES (1, 'https://example.com/feed.xml', 'Show', 'show')"
        )
        conn.execute(
            """
            INSERT INTO episodes (id, subscription_id, guid, title, pub_date, original_url,
                                  duration, status, discovered_at)
            VALUES (7, 1, 'g1', 'Dated Ep', '2026-01-15 09:30:00',
                    'https://cdn.example.com/1.mp3', 60, 'completed', CURRENT_TIMESTAMP)
            """
        )
        conn.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    body = response.text

    assert body.count('data-recent="2026-01-15T09:30:00Z"') == 2, body.count(
        'data-recent="2026-01-15T09:30:00Z"'
    )
    # Neither the bare naive-local string nor a T-but-no-Z variant survives.
    assert 'data-recent="2026-01-15 09:30:00"' not in body
    assert 'data-recent="2026-01-15T09:30:00"' not in body


# --- Regression: a subscription with zero episodes ever ------------------
#
# router.py leaves latest_episode_date as a real Python None (not Undefined)
# when a subscription has no episodes at all -- e.g. just created, before
# its first feed poll. Jinja's `|default('')` only substitutes for
# Undefined, never a real None, so the template used to render the literal
# string data-recent="None". new Date("None") is a truthy Invalid Date, so
# the dashboard's "Recently updated" filter guard (!recentDate is false;
# NaN comparisons are always false) never excluded it: a podcast that has
# never had an episode wrongly showed up as "recently updated".


@pytest.mark.asyncio
async def test_dashboard_data_recent_attribute_empty_for_zero_episode_subscription(
    isolated_data_dir,
):
    """End-to-end through the real app: GET '/' for a subscription with no
    episode rows at all must render data-recent="", never data-recent="None".
    """
    from httpx import ASGITransport, AsyncClient

    from app.infra.database import get_db_connection, init_db
    from app.main import app

    init_db()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id, feed_url, title, slug) "
            "VALUES (1, 'https://example.com/feed.xml', 'Show', 'show')"
        )
        conn.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    body = response.text

    assert 'data-recent="None"' not in body
    # Both the table row and the grid card render the empty attribute.
    assert body.count('data-recent=""') == 2, body.count('data-recent=""')


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_recent_filter_excludes_empty_data_recent_and_catches_the_none_bug():
    """Extract the dashboard's actual 'recently updated' filter guard from
    index.html verbatim and run it in node against the value the router now
    emits for a zero-episode subscription ("") and the literal value it used
    to emit ("None"), plus real stale/fresh timestamps for sanity.

    The "None" case is fault injection: it proves this test would have
    failed against the pre-fix router (which is exactly what happened when
    this test was written and run before the router.py change below), so
    the "" assertion below is not passing vacuously.
    """
    source = _source(INDEX_TEMPLATE)
    recent_date_line = _find_line(source, "const recentDate = card.dataset.recent").strip()
    guard_line = _find_line(source, "if (!recentDate || recentDate < sevenDaysAgo)").strip()
    guard_line = guard_line.replace("return false;", "result = true;")

    script = f"""
    function excluded(datasetRecent) {{
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        const card = {{ dataset: {{ recent: datasetRecent }} }};
        {recent_date_line}
        let result = false;
        {guard_line}
        return result;
    }}

    console.log(JSON.stringify({{
        empty: excluded(""),
        none: excluded("None"),
        staleZ: excluded("2020-01-01T00:00:00Z"),
        freshZ: excluded(new Date().toISOString()),
    }}));
    """

    proc = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    result = json.loads(proc.stdout)

    # The fix: the empty string the router now emits for a zero-episode
    # subscription is correctly excluded from "recently updated".
    assert result["empty"] is True, result
    # Fault injection: the pre-fix literal string "None" is NOT excluded by
    # this same guard -- new Date("None") is a truthy Invalid Date and NaN
    # comparisons are always false. This confirms the guard genuinely
    # discriminates between the fixed and buggy attribute values.
    assert result["none"] is False, result
    # Sanity: a genuinely stale timestamp is still excluded, and a genuinely
    # fresh one is still included -- the fix doesn't break real dates.
    assert result["staleZ"] is True, result
    assert result["freshZ"] is False, result
