import os
import logging
from datetime import datetime
from email.utils import format_datetime
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import html
from app.core.config import settings
from app.infra.repository import SubscriptionRepository, EpisodeRepository

logger = logging.getLogger(__name__)

class RSSGenerator:
    def __init__(self):
        self.sub_repo = SubscriptionRepository()
        self.ep_repo = EpisodeRepository() # We might need a get_by_subscription method

    def generate_feed(self, subscription_id: int):
        sub = self.sub_repo.get_by_id(subscription_id)
        if not sub:
            return
            
        # Get base URL from settings
        from app.core.utils import get_global_settings
        from app.core.utils import get_lan_ip
        global_settings = get_global_settings()
        external_url = global_settings.get("app_external_url")
        
        # Use placeholder if no external URL is set
        if external_url and external_url.strip():
            base_url = external_url.rstrip("/")
        else:
            # Check if we're on localhost and can use LAN IP
            base_url = settings.BASE_URL.rstrip("/")
            if "localhost" in base_url or "127.0.0.1" in base_url:
                lan_ip = get_lan_ip()
                if lan_ip and lan_ip != "localhost":
                    import re
                    # Use \g<1> to avoid ambiguity when lan_ip starts with digits (e.g. "192" would be interpreted as \1192)
                    base_url = re.sub(r"(https?://)(localhost|127\.0\.0\.1)", rf"\g<1>{lan_ip}", base_url)

        # Get processed episodes
        # TODO: Add get_processed_by_subscription to EpisodeRepository
        # For now, let's assume we have a list of dicts
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            episodes = conn.execute(
                "SELECT * FROM episodes WHERE subscription_id = ? AND status = 'completed' ORDER BY pub_date DESC", 
                (subscription_id,)
            ).fetchall()
            
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
        if sub.image_url:
            itunes_image = SubElement(channel, 'itunes:image')
            itunes_image.set('href', sub.image_url) 

        for ep_row in episodes:
            ep = dict(ep_row)
            item = SubElement(channel, 'item')
            SubElement(item, 'title').text = ep['title']
            SubElement(item, 'guid').text = ep['guid']
            
            # PubDate
            if ep['pub_date']:
                # Ensure we have a datetime object
                dt = datetime.fromisoformat(ep['pub_date']) if isinstance(ep['pub_date'], str) else ep['pub_date']
                SubElement(item, 'pubDate').text = format_datetime(dt)
            
            # Duration
            if ep['duration']:
                SubElement(item, 'itunes:duration').text = str(ep['duration'])

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
            
            # Use CDATA for HTML support and to prevent double-escaping
            # We unescape first in case the source was already escaped in the DB
            clean_description = html.unescape(description)
            desc_element = SubElement(item, 'description')
            desc_element.text = f"<![CDATA[{clean_description}]]>"



        # Save to file - use basic tostring to avoid minidom.toprettyxml() URL corruption bug
        # minidom.toprettyxml() has a bug that corrupts URLs like "http://192" into "O2"
        xml_str = tostring(rss, encoding='unicode')
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
        xml_str = xml_str.replace('&lt;![CDATA[', '<![CDATA[').replace(']]&gt;', ']]>')
        
        output_path = os.path.join(settings.FEEDS_DIR, f"{sub.slug}.xml")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
            
        return output_path

    def generate_unified_feed(self):
        """Generate a single RSS feed containing all episodes from all subscriptions."""
        
        # Get base URL from settings (reuse logic from generate_feed)
        from app.core.utils import get_global_settings
        from app.core.utils import get_lan_ip
        global_settings = get_global_settings()
        external_url = global_settings.get("app_external_url")
        
        #  Use placeholder if no external URL is set
        if external_url and external_url.strip():
            base_url = external_url.rstrip("/")
        else:
            # Check if we're on localhost and can use LAN IP
            base_url = settings.BASE_URL.rstrip("/")
            if "localhost" in base_url or "127.0.0.1" in base_url:
                lan_ip = get_lan_ip()
                if lan_ip and lan_ip != "localhost":
                    import re
                    # Use \g<1> to avoid ambiguity when lan_ip starts with digits (e.g. "192" would be interpreted as \1192)
                    base_url = re.sub(r"(https?://)(localhost|127\.0\.0\.1)", rf"\g<1>{lan_ip}", base_url)
        


        # Query all completed episodes with subscription info
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            episodes = conn.execute("""
                SELECT e.*, s.title as podcast_title, s.slug as podcast_slug, s.image_url as podcast_image 
                FROM episodes e 
                JOIN subscriptions s ON e.subscription_id = s.id 
                WHERE e.status = 'completed' 
                ORDER BY e.pub_date DESC
            """).fetchall()

        rss = Element('rss', version='2.0', **{'xmlns:itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd'})
        channel = SubElement(rss, 'channel')
        
        SubElement(channel, 'title').text = "Unified Feed (Ad-Free)"
        SubElement(channel, 'description').text = "All your ad-free podcasts in one place."
        SubElement(channel, 'link').text = base_url
        
        # Use custom unified feed cover image
        unified_cover_url = f"{base_url}/static/unified_feed_cover.png"
        itunes_image = SubElement(channel, 'itunes:image')
        itunes_image.set('href', unified_cover_url)

        for ep_row in episodes:
            ep = dict(ep_row)
            item = SubElement(channel, 'item')
            # Prefix title with Podcast Name
            SubElement(item, 'title').text = f"[{ep['podcast_title']}] {ep['title']}"
            SubElement(item, 'guid').text = ep['guid']
            
            if ep['pub_date']:
                dt = datetime.fromisoformat(ep['pub_date']) if isinstance(ep['pub_date'], str) else ep['pub_date']
                SubElement(item, 'pubDate').text = format_datetime(dt)
            
            if ep['duration']:
                SubElement(item, 'itunes:duration').text = str(ep['duration'])

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

            # Use CDATA for HTML support and to prevent double-escaping
            clean_description = html.unescape(description)
            desc_element = SubElement(item, 'description')
            desc_element.text = f"<![CDATA[{clean_description}]]>"

        # Save to file - use basic tostring to avoid minidom.toprettyxml() URL corruption bug
        # minidom.toprettyxml() has a bug that corrupts URLs like "http://192" into "O2"
        xml_str = tostring(rss, encoding='unicode')
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
        xml_str = xml_str.replace('&lt;![CDATA[', '<![CDATA[').replace(']]&gt;', ']]>')
        
        output_path = os.path.join(settings.FEEDS_DIR, "unified.xml")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
            
        return output_path
