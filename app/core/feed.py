import feedparser
import re
from datetime import datetime
from time import mktime
from typing import Optional, Tuple

def slugify(text: str) -> str:
    """Convert text to a filename-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text

class FeedManager:
    @staticmethod
    def parse_feed(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse feed and return (title, slug, image_url, description). Raises error if invalid."""
        d = feedparser.parse(url)
        if d.bozo:
            raise ValueError(f"Invalid feed: {d.bozo_exception}")
        
        if not hasattr(d, 'feed') or not hasattr(d.feed, 'title'):
             raise ValueError("Feed has no title")
             
        title = d.feed.title
        slug = slugify(title)
        
        description = d.feed.get('summary', d.feed.get('description', ''))
        
        image_url = None
        if hasattr(d.feed, 'image') and hasattr(d.feed.image, 'href'):
            image_url = d.feed.image.href
        elif hasattr(d.feed, 'itunes_image') and hasattr(d.feed.itunes_image, 'href'):
             image_url = d.feed.itunes_image.href
             
        return title, slug, image_url, description

    @staticmethod
    def parse_episodes(url: str) -> list:
        """Parse all episodes from feed."""
        d = feedparser.parse(url)
        episodes = []
        
        for entry in d.entries:
            # Find audio enclosure
            enclosure = next((l for l in entry.get('links', []) if l.get('type', '').startswith('audio/')), None)
            if not enclosure:
                continue
                
            pub_date = None
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime.fromtimestamp(mktime(entry.published_parsed))
            
            description = entry.get('summary', entry.get('description', ''))

            # Parse duration
            duration = 0
            itunes_duration = entry.get('itunes_duration')
            if itunes_duration:
                try:
                    if ':' in itunes_duration:
                        parts = itunes_duration.split(':')
                        if len(parts) == 3:
                            h, m, s = map(int, parts)
                            duration = h * 3600 + m * 60 + s
                        elif len(parts) == 2:
                            m, s = map(int, parts)
                            duration = m * 60 + s
                    else:
                        duration = int(itunes_duration)
                except ValueError:
                    pass

            episodes.append({
                'guid': entry.get('id', enclosure.href),
                'title': entry.get('title', 'Unknown Episode'),
                'pub_date': pub_date,
                'original_url': enclosure.href,
                'duration': duration,
                'description': description,
                'file_size': int(enclosure.length) if hasattr(enclosure, 'length') and enclosure.length else 0
            })
            
        return episodes
