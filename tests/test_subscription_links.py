from app.web.subscription_links import (
    build_best_effort_subscribe_url,
    build_subscribe_instruction_context,
    build_subscription_links,
)


def test_subscription_links_encode_tokenized_feed_urls():
    feed_url = "https://example.com/feeds/show.xml?token=a+b&name=one two"

    links = build_subscription_links(feed_url)

    assert links["direct"] == feed_url
    assert links["apps"] == "/subscribe/apps?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Da%2Bb%26name%3Done%20two"
    assert links["apple"] == "/subscribe/apple?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Da%2Bb%26name%3Done%20two"
    assert links["pocket_casts"] == "/subscribe/pocket_casts?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Da%2Bb%26name%3Done%20two"
    assert links["overcast"] == "overcast://x-callback-url/add?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Da%2Bb%26name%3Done%20two"
    assert links["castbox"] == "/subscribe/castbox?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Da%2Bb%26name%3Done%20two"
    assert links["podcast_addict"] == "/subscribe/podcast_addict?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Da%2Bb%26name%3Done%20two"


def test_subscription_app_links_offer_two_clear_workflows():
    links = build_subscription_links("https://example.com/feed.xml")

    labels = [link["label"] for link in links["app_links"]]
    urls = {link["key"]: link["url"] for link in links["app_links"]}

    assert labels == ["Direct link", "Use your favourite app"]
    assert urls["direct"] == "https://example.com/feed.xml"
    assert urls["apps"].startswith("/subscribe/apps?")


def test_generic_app_instructions_describe_the_shared_private_feed_workflow():
    context = build_subscribe_instruction_context("apps", "https://example.com/feed.xml")

    assert context["label"] == "Use your favourite app"
    assert context["best_effort_url"] is None
    assert "Add by URL" in context["intro"]


def test_best_effort_links_do_not_include_raw_protocol_prefix_for_supported_scheme_paths():
    feed_url = "https://example.com/feeds/show.xml?token=secret"

    assert build_best_effort_subscribe_url("pocket_casts", feed_url) == (
        "pktc://subscribe/example.com%2Ffeeds%2Fshow.xml%3Ftoken%3Dsecret"
    )
    assert build_best_effort_subscribe_url("podcast_addict", feed_url) == (
        "podcastaddict://example.com%2Ffeeds%2Fshow.xml%3Ftoken%3Dsecret"
    )
    assert build_best_effort_subscribe_url("castbox", feed_url) == (
        "castbox://subscribe?url=https%3A%2F%2Fexample.com%2Ffeeds%2Fshow.xml%3Ftoken%3Dsecret"
    )


def test_subscribe_instruction_context_marks_downgraded_apps_as_best_effort():
    context = build_subscribe_instruction_context("pocket_casts", "https://example.com/feed.xml")

    assert context["label"] == "Pocket Casts"
    assert context["best_effort_url"].startswith("pktc://subscribe/")
    assert "might work" in context["intro"]


def test_apple_instruction_context_has_no_best_effort_link():
    context = build_subscribe_instruction_context("apple", "https://example.com/feed.xml")

    assert context["label"] == "Apple"
    assert context["best_effort_url"] is None


def test_subscribe_instruction_copy_button_does_not_claim_success_on_failed_fallback():
    template = open("app/web/templates/subscribe_instructions.html", encoding="utf-8").read()

    assert "copied = document.execCommand('copy')" in template
    assert "if (copied)" in template
    assert "showCopyFailure()" in template
    assert "Could not copy automatically" in template
