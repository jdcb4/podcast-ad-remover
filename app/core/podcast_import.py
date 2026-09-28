"""Bounded OPML/text import. Preview is local; adding rechecks shared-library identity."""
import re
import threading
from app.core.feed_urls import feed_key
from xml.etree import ElementTree

from app.core.models import SubscriptionCreate
from app.core.sources import resolve_source
from app.infra.repository import SubscriptionRepository

MAX_BYTES = 1024 * 1024
MAX_FEEDS = 100
_create_lock = threading.Lock()



def parse_import(content: str) -> list[dict]:
    if len(content.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Use a file or list smaller than 1 MiB.")
    content = content.lstrip("\ufeff \t\r\n")
    if not content:
        raise ValueError("Choose an OPML/text file or paste feed URLs first.")
    entries = []
    if content.startswith("<"):
        # UTF-8 input only; no DTD or entity declarations, including internal entities.
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", content, re.I):
            raise ValueError("OPML files cannot contain DTD or entity declarations.")
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as exc:
            raise ValueError("This file is not valid OPML XML.") from exc
        if root.tag.rsplit("}", 1)[-1].lower() != "opml":
            raise ValueError("Choose an OPML export, not a podcast RSS feed.")
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1].lower() == "outline":
                attrs = {key.lower(): val for key, val in node.attrib.items()}
                if "xmlurl" in attrs:
                    entries.append({"url": attrs["xmlurl"].strip(), "title": attrs.get("title") or attrs.get("text") or ""})
    else:
        entries = [{"url": line.strip(), "title": ""} for line in content.splitlines() if line.strip()]
    if not entries:
        raise ValueError("No feed URLs were found. OPML subscriptions need an xmlUrl attribute.")
    if len(entries) > MAX_FEEDS:
        raise ValueError(f"Import up to {MAX_FEEDS} feeds at a time. Split this list into smaller batches.")
    return entries


def _existing(repo, url, subscriptions=None):
    key = feed_key(url)
    for sub in subscriptions if subscriptions is not None else repo.get_all(include_deleting=True):
        try:
            if feed_key(sub.feed_url) == key:
                return sub
        except ValueError:
            continue
    return None


def preview_import(content: str, user_id: int | None) -> list[dict]:
    repo = SubscriptionRepository()
    entries = parse_import(content)
    subscriptions = repo.get_all(include_deleting=True)
    seen = set()
    rows = []
    for entry in entries:
        row = {**entry, "status": "ready", "detail": "New feed — checked when imported", "subscription_id": None}
        try:
            key = feed_key(entry["url"])
            row["url"] = key
            if key in seen:
                row.update(status="duplicate", detail="Repeated in this import")
            else:
                seen.add(key)
                sub = _existing(repo, key, subscriptions)
                if sub:
                    row.update(title=sub.title, subscription_id=sub.id)
                    if sub.deletion_status:
                        row.update(status="invalid", detail="This podcast is being deleted. Try again later.")
                    elif not user_id or repo.is_in_user_library(user_id, sub.id):
                        row.update(status="existing", detail="Already in My Podcasts")
                    else:
                        row.update(status="join", detail="Add from the existing library")
        except (ValueError, UnicodeError):
            row.update(status="invalid", detail="Invalid feed URL. Use HTTP or HTTPS without spaces or embedded credentials.")
        rows.append(row)
    return rows


def import_feed(url: str, user_id: int | None) -> dict:
    """Import one entry, without eagerly downloading an entire exported library."""
    repo = SubscriptionRepository()
    url = feed_key(url)

    def reuse(sub):
        if sub.deletion_status:
            raise ValueError("This podcast is being deleted. Try again later.")
        added = repo.add_to_user_library(user_id, sub.id)
        return {"url": url, "title": sub.title, "status": "joined" if added else "existing", "subscription_id": sub.id}

    existing = _existing(repo, url)
    if existing:
        return reuse(existing)
    # Network resolution uses the same redirect/private-feed policy as single additions.
    try:
        source = resolve_source(url)
    except Exception as exc:
        # Provider exceptions can contain private URLs/credentials. Do not reflect them.
        raise ValueError("Could not read this feed. Check its URL and availability, then retry.") from exc
    with _create_lock:
        existing = repo.get_by_source_identity(source.source_type, source.external_id) or _existing(repo, source.canonical_url)
        if existing:
            return reuse(existing)
        try:
            sub = repo.create(
                SubscriptionCreate(feed_url=feed_key(source.canonical_url)), source.title,
                repo.available_slug(source.slug, source.external_id), source.image_url, source.description,
                owner_user_id=user_id, source_type=source.source_type, source_external_id=source.external_id,
            )
        except ValueError:
            # A different worker/single-add request may have won the unique constraint.
            existing = repo.get_by_source_identity(source.source_type, source.external_id) or _existing(repo, source.canonical_url)
            if existing:
                return reuse(existing)
            raise ValueError("This podcast could not be added. Retry the import.") from None
    return {"url": url, "title": sub.title, "status": "added", "subscription_id": sub.id}
