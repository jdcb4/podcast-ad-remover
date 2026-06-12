from xml.etree.ElementTree import Element, SubElement

import app.core.utils as utils
from app.core.config import settings
from app.core.rss_gen import _get_feed_base_url, _safe_cdata, _serialize_rss


def test_safe_cdata_splits_embedded_cdata_terminator():
    rendered = _safe_cdata("before ]]> after")

    assert rendered == "<![CDATA[before ]]]]><![CDATA[> after]]>"


def test_safe_cdata_unescapes_existing_entities():
    assert _safe_cdata("Tom &amp; Jerry") == "<![CDATA[Tom & Jerry]]>"


def test_serialize_rss_preserves_safe_cdata_sections():
    rss = Element("rss")
    item = SubElement(rss, "item")
    SubElement(item, "description").text = _safe_cdata("before ]]> after")

    xml = _serialize_rss(rss)

    assert "<![CDATA[before ]]]]><![CDATA[> after]]>" in xml
    assert "&lt;![CDATA[" not in xml


def test_feed_base_url_prefers_configured_external_url():
    assert _get_feed_base_url({"app_external_url": "https://podcasts.example.com/"}) == "https://podcasts.example.com"


def test_feed_base_url_preserves_existing_lan_fallback(monkeypatch):
    monkeypatch.setattr(settings, "BASE_URL", "http://localhost:9000")
    monkeypatch.setattr(utils, "get_lan_ip", lambda: "192.168.1.20")

    assert _get_feed_base_url({"app_external_url": ""}) == "http://192.168.1.20:9000"
