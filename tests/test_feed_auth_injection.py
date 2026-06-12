from app.web.router import _append_feed_access_to_enclosures


def test_feed_access_injection_url_encodes_legacy_basic_auth_values():
    xml = '<enclosure url="https://podcasts.example.com/audio/show/episode.mp3" type="audio/mpeg" />'

    rendered = _append_feed_access_to_enclosures(xml, "auth", "abc+/=")

    assert 'url="https://podcasts.example.com/audio/show/episode.mp3?auth=abc%2B%2F%3D"' in rendered
    assert "abc+/=" not in rendered


def test_feed_access_injection_uses_xml_safe_separator_for_existing_query():
    xml = '<enclosure url="https://podcasts.example.com/audio/show/episode.mp3?download=1" type="audio/mpeg" />'

    rendered = _append_feed_access_to_enclosures(xml, "token", "feed-token")

    assert 'episode.mp3?download=1&amp;token=feed-token' in rendered


def test_feed_access_injection_leaves_non_enclosure_links_unchanged():
    xml = '<link>https://podcasts.example.com/audio/show/episode.mp3</link>'

    assert _append_feed_access_to_enclosures(xml, "token", "feed-token") == xml
