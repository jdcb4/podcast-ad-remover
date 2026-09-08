import httpx
import asyncio
from typing import List, Dict, Optional
from app.core.sources import is_youtube_input, resolve_source

class PodcastSearcher:
    BASE_URL = "https://itunes.apple.com/search"

    @staticmethod
    async def search(term: str, limit: int = 10) -> List[Dict]:
        """Resolve direct YouTube sources or search podcasts through iTunes."""
        if is_youtube_input(term) or term.strip().lower().startswith(('https://', 'http://')):
            source = await asyncio.to_thread(resolve_source, term.strip())
            return [source.search_result()]
        params = {
            "term": term,
            "media": "podcast",
            "entity": "podcast",
            "limit": limit
        }
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(PodcastSearcher.BASE_URL, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("collectionName"),
                        "feed_url": item.get("feedUrl"),
                        "image": item.get("artworkUrl600"),
                        "description": item.get("artistName"), # iTunes doesn't give full desc in search
                        "source_type": "rss",
                        "source_external_id": None,
                    })
                return results
            except Exception as e:
                raise ValueError("Podcast search is unavailable. Try again or paste a direct RSS URL.") from e
