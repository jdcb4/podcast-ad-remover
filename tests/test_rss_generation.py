from xml.etree.ElementTree import Element, SubElement, fromstring

import app.core.utils as utils
from app.core.config import settings
from app.core.rss_gen import _cdata, _get_feed_base_url, _serialize_rss


def test_serialize_rss_emits_html_description_as_cdata():
    description = (
        '<p>Hosted by <a href="https://example.com">Example</a></p>'
        '<p>Tom &amp; Jerry&nbsp;</p>'
    )
    rss = Element("rss")
    item = SubElement(rss, "item")
    SubElement(item, "description").text = _cdata(description)

    xml = _serialize_rss(rss)
    parsed_description = fromstring(xml).findtext("./item/description")

    assert parsed_description == description
    assert f"<![CDATA[{description}]]>" in xml
    assert "&lt;p&gt;Hosted by" not in xml


def test_serialize_rss_safely_round_trips_cdata_terminator_text():
    rss = Element("rss")
    item = SubElement(rss, "item")
    SubElement(item, "description").text = _cdata("before ]]> after")

    xml = _serialize_rss(rss)

    assert fromstring(xml).findtext("./item/description") == "before ]]> after"
    assert "<![CDATA[before ]]]]><![CDATA[> after]]>" in xml


def test_serialize_rss_restores_cdata_marker_after_serialization():
    rss = Element("rss")
    description = SubElement(SubElement(rss, "item"), "description")
    description.text = _cdata("<p>Notes</p>")

    _serialize_rss(rss)

    assert description.text == "<p>Notes</p>"
    assert type(description.text).__name__ == "_CDATA"


def test_feed_base_url_prefers_configured_external_url():
    assert _get_feed_base_url({"app_external_url": "https://podcasts.example.com/"}) == "https://podcasts.example.com"


def test_feed_base_url_preserves_existing_lan_fallback(monkeypatch):
    monkeypatch.setattr(settings, "BASE_URL", "http://localhost:9000")
    monkeypatch.setattr(utils, "get_lan_ip", lambda: "192.168.1.20")

    assert _get_feed_base_url({"app_external_url": ""}) == "http://192.168.1.20:9000"
