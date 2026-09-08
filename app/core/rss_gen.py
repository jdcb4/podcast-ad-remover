import os
from app.core.publication import atomic_write, serialized_feed
import logging
from datetime import datetime
from email.utils import format_datetime
from uuid import uuid4
from xml.etree.ElementTree import Element, SubElement, tostring
from app.core.config import settings
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.artwork import effective_artwork_url
from app.core.unified_feed import resolve_unified_feed_settings

logger = logging.getLogger(__name__)


class _CDATA(str):
    """Mark element text that must be serialized as an XML CDATA section."""


def _cdata(text: str) -> _CDATA:
    return _CDATA(text or "")


def _serialize_rss(rss: Element) -> str:
    replacements = []
    try:
        for element in rss.iter():
            if isinstance(element.text, _CDATA):
                original = element.text
                token = f"PODCAST_AD_REMOVER_CDATA_{uuid4().hex}"
                replacements.append((element, original, token))
                element.text = token

        xml_str = tostring(rss, encoding='unicode')
    finally:
        for element, original, _token in replacements:
            element.text = original

    for _element, original, token in replacements:
        safe_text = str(original).replace("]]>", "]]]]><![CDATA[>")
        xml_str = xml_str.replace(token, f"<![CDATA[{safe_text}]]>")

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str


def _get_feed_base_url(global_settings: dict) -> str:
    external_url = global_settings.get("app_external_url")
    if external_url and external_url.strip():
        return external_url.rstrip("/")

    base_url = settings.BASE_URL.rstrip("/")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        from app.core.utils import get_lan_ip
        import re

        lan_ip = get_lan_ip()
        if lan_ip and lan_ip != "localhost":
            return re.sub(r"(https?://)(localhost|127\.0\.0\.1)", rf"\g<1>{lan_ip}", base_url)
    return base_url


class RSSGenerator:
    def __init__(self):
        self.sub_repo = SubscriptionRepository()
        self.ep_repo = EpisodeRepository()

    @serialized_feed
    def generate_feed(self, subscription_id: int):
        sub = self.sub_repo.get_by_id(subscription_id)
        if not sub:
            return
            
        from app.core.utils import get_global_settings
        global_settings = get_global_settings()
        base_url = _get_feed_base_url(global_settings)

        episodes = self.ep_repo.get_completed_by_subscription(subscription_id)
            
        rss = Element('rss', version='2.0', **{'xmlns:itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd'})
        channel = SubElement(rss, 'channel')
        
        SubElement(channel, 'title').text = f"{sub.title} (Ad-Free)"
        
        # Build description with latest episode info
        # Build description
        description = f"Ad-free version of {sub.title}"
        if sub.description:
            description += f"\n\n{sub.description}"
        elif episodes:
            # Fallback if no description: show latest episode
            latest = episodes[0]
            description += f"\n\nLatest: {latest['title']}"
        
        SubElement(channel, 'description').text = description
        SubElement(channel, 'link').text = sub.feed_url
        
        # Image
        artwork_url = effective_artwork_url(sub, base_url)
        if artwork_url:
            itunes_image = SubElement(channel, 'itunes:image')
            itunes_image.set('href', artwork_url)

        for ep_row in episodes:
            ep = dict(ep_row)
            item = SubElement(channel, 'item')
            SubElement(item, 'title').text = ep['title']
            SubElement(item, 'guid').text = ep.get('published_guid') or ep['guid']
            SubElement(item, 'link').text = ep['original_url']
            
            # PubDate
            if ep['pub_date']:
                # Ensure we have a datetime object
                dt = datetime.fromisoformat(ep['pub_date']) if isinstance(ep['pub_date'], str) else ep['pub_date']
                SubElement(item, 'pubDate').text = format_datetime(dt)
            
            # Duration
            if ep.get('output_duration') or ep['duration']:
                SubElement(item, 'itunes:duration').text = str(round(ep.get('output_duration') or ep['duration']))

            # Enclosure
            enclosure = SubElement(item, 'enclosure')
            # Extract relative path from PODCASTS_DIR to the local file
            # This ensures we include the podcast_slug/episode_slug/ structure
            try:
                rel_path = os.path.relpath(ep['local_filename'], settings.PODCASTS_DIR)
                if rel_path.startswith(".."):
                    raise ValueError("Path mismatch")
                # Ensure we use forward slashes for the URL
                url_path = rel_path.replace(os.sep, '/')
                url = f"{base_url}/audio/{url_path}"
            except Exception:
                # Fallback to current (potentially broken) logic if path math fails
                filename = os.path.basename(ep['local_filename'])
                url = f"{base_url}/audio/{sub.slug}/{filename}"
                
            enclosure.set('url', url)
            enclosure.set('type', 'audio/mpeg')
            if ep['file_size']:
                enclosure.set('length', str(ep['file_size']))
            
            # Prioritize AI summary for description if available
            description = ep['ai_summary'] if ep.get('ai_summary') else ep['description']
            if not description:
                description = f"Original: {ep['original_url']}\n\nProcessed by Podcast Ad Remover."
            
            desc_element = SubElement(item, 'description')
            desc_element.text = _cdata(description)



        # Save to file - use basic tostring to avoid minidom.toprettyxml() URL corruption bug
        # minidom.toprettyxml() has a bug that corrupts URLs like "http://192" into "O2"
        xml_str = _serialize_rss(rss)
        
        output_path = os.path.join(settings.FEEDS_DIR, f"{sub.slug}.xml")
        
        atomic_write(output_path, xml_str)
            
        return output_path

    @serialized_feed
    def generate_unified_feed(self):
        """Generate a single RSS feed containing all episodes from all subscriptions."""
        
        from app.core.utils import get_global_settings
        global_settings = get_global_settings()
        base_url = _get_feed_base_url(global_settings)
        feed_settings = resolve_unified_feed_settings(global_settings, base_url)

        episodes = self.ep_repo.get_completed_with_subscription_info()
        subscriptions = {sub.id: sub for sub in self.sub_repo.get_all()}

        rss = Element('rss', version='2.0', **{'xmlns:itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd'})
        channel = SubElement(rss, 'channel')
        
        SubElement(channel, 'title').text = feed_settings["title"]
        SubElement(channel, 'description').text = feed_settings["description"]
        SubElement(channel, 'link').text = base_url
        
        itunes_image = SubElement(channel, 'itunes:image')
        itunes_image.set('href', feed_settings["artwork_url"])

        for ep_row in episodes:
            ep = dict(ep_row)
            item = SubElement(channel, 'item')
            episode_title = ep['title']
            if feed_settings["include_podcast_name"] and ep.get('podcast_title'):
                episode_title = f"[{ep['podcast_title']}] {episode_title}"
            SubElement(item, 'title').text = episode_title
            SubElement(item, 'guid').text = ep.get('published_guid') or ep['guid']
            SubElement(item, 'link').text = ep['original_url']
            
            if ep['pub_date']:
                dt = datetime.fromisoformat(ep['pub_date']) if isinstance(ep['pub_date'], str) else ep['pub_date']
                SubElement(item, 'pubDate').text = format_datetime(dt)
            
            if ep.get('output_duration') or ep['duration']:
                SubElement(item, 'itunes:duration').text = str(round(ep.get('output_duration') or ep['duration']))

            enclosure = SubElement(item, 'enclosure')
            # Construct URL using the same relative path logic
            try:
                rel_path = os.path.relpath(ep['local_filename'], settings.PODCASTS_DIR)
                if rel_path.startswith(".."):
                    raise ValueError("Path mismatch")
                url_path = rel_path.replace(os.sep, '/')
                url = f"{base_url}/audio/{url_path}"
            except Exception:
                filename = os.path.basename(ep['local_filename'])
                url = f"{base_url}/audio/{ep['podcast_slug']}/{filename}"
                
            enclosure.set('url', url)
            enclosure.set('type', 'audio/mpeg')
            if ep['file_size']:
                enclosure.set('length', str(ep['file_size']))
            
            # Prioritize AI summary for description if available
            description = ep['ai_summary'] if ep.get('ai_summary') else ep['description']
            if not description:
                description = ""
            
            # Optionally add podcast name to description as well
            if ep.get('podcast_title'):
                description = f"From: {ep['podcast_title']}\n\n" + description

            # Episode Artwork - Use podcast image for each item
            episode_artwork = effective_artwork_url(
                subscriptions.get(ep.get("subscription_id")),
                base_url,
            )
            if episode_artwork:
                itunes_ep_image = SubElement(item, 'itunes:image')
                itunes_ep_image.set('href', episode_artwork)

            desc_element = SubElement(item, 'description')
            desc_element.text = _cdata(description)

        # Save to file - use basic tostring to avoid minidom.toprettyxml() URL corruption bug
        # minidom.toprettyxml() has a bug that corrupts URLs like "http://192" into "O2"
        xml_str = _serialize_rss(rss)
        
        output_path = os.path.join(settings.FEEDS_DIR, "unified.xml")
        
        atomic_write(output_path, xml_str)
            
        return output_path
