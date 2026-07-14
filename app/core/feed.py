import feedparser
import httpx
import re
from datetime import datetime
from time import mktime
from typing import Optional, Tuple
from app.core.config import settings
from app.core.url_utils import validate_http_url, validate_redirect_target

def slugify(text: str) -> str:
    """Convert text to a filename-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text

class FeedManager:
    @staticmethod
    def _fetch_feed(url: str) -> bytes:
        validate_http_url(url, allow_private=settings.ALLOW_PRIVATE_FEEDS)
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                validate_redirect_target(url, str(response.url), allow_private=settings.ALLOW_PRIVATE_FEEDS)
                content_length = int(response.headers.get("Content-Length") or 0)
                if content_length and content_length > settings.MAX_FEED_BYTES:
                    raise ValueError("Feed is larger than the configured maximum size")

                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > settings.MAX_FEED_BYTES:
                        raise ValueError("Feed is larger than the configured maximum size")
                    chunks.append(chunk)
                return b"".join(chunks)

    @staticmethod
    def parse_feed(url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse feed and return (title, slug, image_url, description). Raises error if invalid."""
        d = feedparser.parse(FeedManager._fetch_feed(url))
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
        d = feedparser.parse(FeedManager._fetch_feed(url))
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
                'file_size': int(enclosure.length) if hasattr(enclosure, 'length') and enclosure.length and str(enclosure.length).isdigit() else 0
            })

        return episodes
