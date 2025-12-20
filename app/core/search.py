import httpx
from typing import List, Dict, Optional

class PodcastSearcher:
    BASE_URL = "https://itunes.apple.com/search"

    @staticmethod
    async def search(term: str, limit: int = 10) -> List[Dict]:
        """Search for podcasts using iTunes API."""
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
                        "description": item.get("artistName") # iTunes doesn't give full desc in search
                    })
                return results
            except Exception as e:
                print(f"Search failed: {e}")
                return []
