import re
from urllib.parse import quote


SUBSCRIBE_CLIENTS = {
    "apps": {
        "label": "Use your favourite app",
        "title": "Use your favourite podcast app",
        "intro": "Private and self-hosted podcast feeds use the same reliable workflow in most podcast apps: copy the feed URL and use Add by URL, Follow by URL, or private RSS.",
        "steps_title": "Add this feed",
        "steps": [
            "Copy the feed URL above.",
            "Open your podcast app and find its Add by URL, Follow by URL, or private RSS option.",
            "Paste the feed URL and confirm the subscription.",
            "If your app does not support custom RSS feeds, choose another podcast app that does.",
        ],
    },
    "apple": {
        "label": "Apple",
        "title": "Subscribe on Apple Podcasts",
        "intro": "Apple Podcasts requires adding private feeds manually. Follow these steps:",
        "app_url": "podcasts://open",
        "app_link_label": "Open Apple Podcasts",
        "steps_title": "Add to Apple Podcasts",
        "steps": [
            "Open the Podcasts app.",
            "Go to the Library tab.",
            "Tap the ... (More) button in the top-right corner.",
            "Select Follow a Show by URL...",
            "Paste the URL you copied above and tap Follow.",
        ],
    },
    "pocket_casts": {
        "label": "Pocket Casts",
        "title": "Subscribe on Pocket Casts",
        "intro": "Pocket Casts does not provide a stable cross-platform private-feed link. This link might work on some iOS installs; if not, follow the instructions below.",
        "steps_title": "Add to Pocket Casts",
        "steps": [
            "Copy the feed URL above.",
            "Open Pocket Casts.",
            "Paste the feed URL into Discover or search.",
            "Open the matching private feed and subscribe.",
        ],
    },
    "castbox": {
        "label": "Castbox",
        "title": "Subscribe on Castbox",
        "intro": "Castbox documents private RSS subscription through manual feed entry. This link might work in some installs; if not, follow the instructions below.",
        "steps_title": "Add to Castbox",
        "steps": [
            "Copy the feed URL above.",
            "Open Castbox.",
            "Use the RSS feed subscription option.",
            "Paste the feed URL and subscribe.",
        ],
    },
    "podcast_addict": {
        "label": "Podcast Addict",
        "title": "Subscribe on Podcast Addict",
        "intro": "Podcast Addict documents private RSS subscription through manual feed entry. This link might work in some installs; if not, follow the instructions below.",
        "steps_title": "Add to Podcast Addict",
        "steps": [
            "Copy the feed URL above.",
            "Open Podcast Addict.",
            "Choose the RSS feed/custom feed option.",
            "Paste the feed URL and subscribe.",
        ],
    },
}


def _encoded_url(url: str) -> str:
    return quote(url, safe="")


def _without_http_scheme(url: str) -> str:
    return re.sub(r"^https?://", "", url, flags=re.IGNORECASE)


def build_best_effort_subscribe_url(client_key: str, rss_url: str) -> str | None:
    encoded_rss_url = _encoded_url(rss_url)
    encoded_without_scheme = _encoded_url(_without_http_scheme(rss_url))
    if client_key == "pocket_casts":
        return f"pktc://subscribe/{encoded_without_scheme}"
    if client_key == "castbox":
        return f"castbox://subscribe?url={encoded_rss_url}"
    if client_key == "podcast_addict":
        return f"podcastaddict://{encoded_without_scheme}"
    return None


def subscribe_instruction_url(client_key: str, rss_url: str) -> str:
    return f"/subscribe/{client_key}?url={_encoded_url(rss_url)}"


def build_subscription_links(rss_url: str) -> dict:
    encoded_rss_url = _encoded_url(rss_url)
    links = {
        "rss": rss_url,
        "direct": rss_url,
        "apps": subscribe_instruction_url("apps", rss_url),
        "apple": subscribe_instruction_url("apple", rss_url),
        "pocket_casts": subscribe_instruction_url("pocket_casts", rss_url),
        "overcast": f"overcast://x-callback-url/add?url={encoded_rss_url}",
        "castbox": subscribe_instruction_url("castbox", rss_url),
        "podcast_addict": subscribe_instruction_url("podcast_addict", rss_url),
    }
    links["app_links"] = [
        {"key": "direct", "label": "Direct link", "url": links["direct"], "target": "_blank"},
        {"key": "apps", "label": "Use your favourite app", "url": links["apps"]},
    ]
    return links


def build_subscribe_instruction_context(client_key: str, feed_url: str) -> dict | None:
    client = SUBSCRIBE_CLIENTS.get(client_key)
    if not client:
        return None

    best_effort_url = build_best_effort_subscribe_url(client_key, feed_url)
    return {
        **client,
        "key": client_key,
        "feed_url": feed_url,
        "best_effort_url": best_effort_url,
    }
