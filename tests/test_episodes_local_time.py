import re
from pathlib import Path

# dev's "Reuse one server renderer for episode cards" decision (2026-09-06)
# split this page into three files: the page shell, the card partial that
# both the first paint and pagination render, and the extracted page JS.
EPISODES_TEMPLATE = Path("app/web/templates/episodes.html")
CARDS_TEMPLATE = Path("app/web/templates/_episode_cards.html")
EPISODES_JS = Path("app/web/static/js/episodes.js")
BASE_TEMPLATE = Path("app/web/templates/base.html")


def template_source() -> str:
    return EPISODES_TEMPLATE.read_text(encoding="utf-8")


def cards_source() -> str:
    return CARDS_TEMPLATE.read_text(encoding="utf-8")


def script_source() -> str:
    """The episodes page's JavaScript, now a shipped external file."""
    return EPISODES_JS.read_text(encoding="utf-8")


def test_server_rendered_episode_card_localizes_pub_date():
    """The visible date is a <time> element; the data-* attribute is a bare instant."""
    source = cards_source()

    # Element context: the full <time> element, hydrated by local-time.js.
    assert "<span class=\"date-relative\">{{ ep.pub_date|local_time('relative') }}</span>" in source
    # Attribute context: utc_isoformat, because a <time> element inside an
    # attribute would corrupt the surrounding tag.
    assert 'data-date="{{ ep.pub_date|utc_isoformat }}"' in source
    # The raw, unfiltered timestamp must not reach the page anywhere.
    assert "{{ ep.pub_date }}" not in source


def test_local_time_is_never_used_in_an_attribute_context():
    """local_time renders a whole <time> element and cannot sit inside quotes."""
    for source in (template_source(), cards_source()):
        assert not re.search(r'="[^"]*\|\s*local_time', source)


def test_one_server_renderer_feeds_both_the_first_paint_and_pagination():
    """There is no second, JS-side card template to keep in sync any more.

    PR #20's follow-up collapsed the duplicated markup into
    `_episode_cards.html`, so the localized <time> element only has to be
    correct in one place. If a JS card builder ever comes back, the
    hydratable-shape assertions deleted alongside this test come back too.
    """
    assert '{% include "_episode_cards.html" %}' in template_source()
    assert "_episode_cards.html" in Path("app/web/router.py").read_text(encoding="utf-8")
    # The old client-side builder is gone; nothing rebuilds a card in JS.
    assert "createEpisodeCardHTML" not in script_source()


def test_pagination_hydrates_the_grid_it_just_filled():
    """Server-rendered fragments arrive after local-time.js has already run,
    so the newly inserted <time> elements need an explicit hydrate pass."""
    script = script_source()

    insert = script.index("grid.insertAdjacentHTML('beforeend', data.html);")
    after = script[insert : insert + 200]
    assert "formatEpisodeDates(grid)" in after, (
        "inserted cards must be hydrated immediately after insertion"
    )

    # ...and that helper is a thin delegation, not a second date implementation.
    assert "window.AppLocalTime?.hydrate(root);" in script
    assert "querySelectorAll('.date-relative')" not in script


def test_hydrator_is_loaded_before_the_page_script_that_calls_it():
    """Load-order guard.

    Both files are deferred, and deferred scripts execute in document order.
    local-time.js is in base.html's head and episodes.js is emitted in the
    page body, so window.AppLocalTime exists by the time episodes.js runs.
    The optional-chaining guard keeps a hydrate() call harmless even if that
    order is ever disturbed -- so assert both the order and the guard.
    """
    assert 'src="/static/js/local-time.js' in BASE_TEMPLATE.read_text(encoding="utf-8")
    assert "defer" in next(
        line for line in BASE_TEMPLATE.read_text(encoding="utf-8").splitlines()
        if "local-time.js" in line
    )
    assert "defer" in next(
        line for line in template_source().splitlines() if "episodes.js" in line
    )

    script = script_source()
    # Every reference to the hydrator goes through the optional-chained call.
    assert script.count("AppLocalTime") == 1
    assert "window.AppLocalTime?.hydrate(" in script


def test_undated_episode_renders_the_shared_placeholder():
    """A NULL pub_date must read as local_time's placeholder, not an empty cell.

    The expected text is not hard-coded: it is whatever the real Jinja
    environment renders for a falsy pub_date, so re-wording local_time's
    placeholder cannot silently drift away from what the card shows.
    """
    from app.web.router import templates

    ssr_line = next(
        line for line in cards_source().splitlines()
        if 'class="date-relative">{{ ep.pub_date|local_time' in line
    )
    ssr = templates.env.from_string(ssr_line).render(ep={"pub_date": None}).strip()
    placeholder = ssr[ssr.index(">") + 1 : ssr.rindex("<")]

    assert placeholder, "SSR falsy render produced no placeholder text"
    assert "<time" not in ssr, "a falsy pub_date must not emit a <time> element"


def test_hand_rolled_episode_date_code_is_gone():
    """timeAgo, its DOMContentLoaded hydrator and the dead .date-formatted
    loop are deleted -- from the page shell and from the extracted JS, which
    is where dev moved them."""
    for source in (template_source(), script_source()):
        assert "timeAgo" not in source
        assert "date-formatted" not in source
        # No date parsing or locale formatting is left at all.
        assert "new Date(" not in source
        assert "toLocaleDateString" not in source


def test_episodes_api_serializes_pub_date_as_a_z_suffixed_instant(isolated_data_dir):
    import asyncio

    from app.infra.database import get_db_connection, init_db
    from app.web.router import get_subscription_episodes_api

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
            VALUES (7, 1, 'g1', 'Dated Ep', '2026-01-01 10:00:00',
                    'https://cdn.example.com/1.mp3', 60, 'unprocessed', CURRENT_TIMESTAMP)
            """
        )
        conn.commit()

    payload = asyncio.run(get_subscription_episodes_api(None, 1, user=None))

    assert payload["episodes"][0]["pub_date"] == "2026-01-01T10:00:00Z"


def test_episodes_api_leaves_a_missing_pub_date_alone(isolated_data_dir):
    """A NULL pub_date must stay falsy rather than becoming the string 'None'."""
    import asyncio

    from app.infra.database import get_db_connection, init_db
    from app.web.router import get_subscription_episodes_api

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
            VALUES (8, 1, 'g2', 'Undated Ep', NULL,
                    'https://cdn.example.com/2.mp3', 60, 'unprocessed', CURRENT_TIMESTAMP)
            """
        )
        conn.commit()

    payload = asyncio.run(get_subscription_episodes_api(None, 1, user=None))

    assert not payload["episodes"][0]["pub_date"]


def test_episodes_api_z_suffixes_every_timestamp_in_the_row(isolated_data_dir):
    """SELECT * returns four timestamp columns; shipping one normalized and
    three raw is the same "two formats in one row" defect Task 7 removed
    from the queue endpoints.
    """
    import asyncio

    from app.infra.database import get_db_connection, init_db
    from app.web.router import get_subscription_episodes_api

    init_db()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (id, feed_url, title, slug) "
            "VALUES (1, 'https://example.com/feed.xml', 'Show', 'show')"
        )
        conn.execute(
            """
            INSERT INTO episodes (id, subscription_id, guid, title, pub_date, original_url,
                                  duration, status, discovered_at, processed_at, next_retry_at, processed_at_is_utc)
            VALUES (9, 1, 'g3', 'Fully Dated Ep', '2026-01-01 10:00:00',
                    'https://cdn.example.com/3.mp3', 60, 'completed',
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            """
        )
        conn.commit()

    row = asyncio.run(get_subscription_episodes_api(None, 1, user=None))["episodes"][0]

    for field in ("pub_date", "discovered_at", "processed_at", "next_retry_at"):
        assert row[field], f"seed did not populate {field}"
        assert row[field].endswith("Z"), f"{field} shipped raw: {row[field]!r}"

    # And nothing else timestamp-shaped slipped through unnormalized.
    raw_timestamp = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
    assert {
        key: value
        for key, value in row.items()
        if isinstance(value, str) and raw_timestamp.match(value) and not value.endswith("Z")
    } == {}
