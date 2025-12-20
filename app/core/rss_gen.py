import os
from datetime import datetime
from email.utils import format_datetime
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from app.core.config import settings
from app.infra.repository import SubscriptionRepository, EpisodeRepository

class RSSGenerator:
    def __init__(self):
        self.sub_repo = SubscriptionRepository()
        self.ep_repo = EpisodeRepository() # We might need a get_by_subscription method

    def generate_feed(self, subscription_id: int):
        sub = self.sub_repo.get_by_id(subscription_id)
        if not sub:
            return
            
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
        SubElement(channel, 'description').text = f"Ad-free version of {sub.title}"
        SubElement(channel, 'link').text = sub.feed_url
        
        # Image
        if sub.image_url:
            itunes_image = SubElement(channel, 'itunes:image')
            itunes_image.set('href', sub.image_url) 

        for ep in episodes:
            item = SubElement(channel, 'item')
            SubElement(item, 'title').text = ep['title']
            SubElement(item, 'guid').text = ep['guid']
            
            # PubDate
            if ep['pub_date']:
                # Ensure we have a datetime object
                dt = datetime.fromisoformat(ep['pub_date']) if isinstance(ep['pub_date'], str) else ep['pub_date']
                SubElement(item, 'pubDate').text = format_datetime(dt)
            
            # Enclosure
            enclosure = SubElement(item, 'enclosure')
            filename = os.path.basename(ep['local_filename'])
            url = f"{settings.BASE_URL}/audio/{sub.slug}/{filename}"
            enclosure.set('url', url)
            enclosure.set('type', 'audio/mpeg')
            # enclosure.set('length', str(file_size)) # Optional but good
            
            description = ep['description'] if ep['description'] else f"Original: {ep['original_url']}\n\nProcessed by Podcast Ad Remover."
            if ep['error_message']: # Re-purposing error message for stats if needed, or add new column
                 pass
            SubElement(item, 'description').text = description

        # Save to file
        xml_str = minidom.parseString(tostring(rss)).toprettyxml(indent="  ")
        output_path = os.path.join(settings.FEEDS_DIR, f"{sub.slug}.xml")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
            
        return output_path
