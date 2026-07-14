from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from app.core.models import Subscription, SubscriptionCreate
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.feed import FeedManager
from app.core.processor import Processor
from app.core.search import PodcastSearcher
from app.core.notifications import EVENT_NEW_PODCAST, send_notification_async
from app.web.auth import require_auth
from pydantic import BaseModel
import shutil
import os
from app.core.config import settings


router = APIRouter()
repo = SubscriptionRepository()


def _real_user_id(user) -> int | None:
    user_id = getattr(user, "id", None)
    return user_id if user_id and user_id > 0 else None


def _can_manage_subscription(user, sub) -> bool:
    if getattr(user, "is_admin", False):
        return True
    user_id = _real_user_id(user)
    return bool(user_id and getattr(sub, "owner_user_id", None) == user_id)

# Helper to get processor (in a real app, use dependency injection)
def get_processor():
    return Processor()

@router.get("/subscriptions", response_model=List[Subscription])
async def list_subscriptions(user = Depends(require_auth)):
    return repo.get_all(user_id=_real_user_id(user), only_user=True)

@router.post("/subscriptions", response_model=Subscription)
async def create_subscription(sub: SubscriptionCreate, initial_count: int = 5, user = Depends(require_auth)):
    existing = repo.get_by_url(sub.feed_url)
    if existing:
        added = repo.add_to_user_library(_real_user_id(user), existing.id)
        if added:
            return existing
        raise HTTPException(status_code=400, detail="Subscription already exists in your podcasts")
    
    try:
        # Parse feed to get title
        print(f"Parsing feed {sub.feed_url}")
        title, slug, image_url, description = FeedManager.parse_feed(sub.feed_url)

        print(f"Parsing feed complete title: {title}, slug: {slug}, image_url: {image_url}, description: {description}")

        # Save to DB
        new_sub = repo.create(sub, title, slug, image_url, description=description, owner_user_id=_real_user_id(user))
        await send_notification_async(
            EVENT_NEW_PODCAST,
            "Podcast added",
            f"{title} was added to the global podcast library.",
            severity="success",
        )
        
        # Trigger initial check
        proc = get_processor()
        await proc.check_feeds(subscription_id=new_sub.id, limit=initial_count)
        
        # Processor loop will pick up 'pending' items automatically
        

        
        return new_sub
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/subscriptions/{id}")
async def delete_subscription(id: int, user = Depends(require_auth)):
    """Delete a subscription and remove local files for its episodes."""
    sub = repo.get_by_id(id)
    
    if sub:
        if not getattr(user, "is_admin", False):
            removed = repo.remove_from_user_library(_real_user_id(user), id)
            return {"status": "removed_from_my_podcasts" if removed else "not_in_my_podcasts"}

        # 1. Delete all episodes using Processor (cleans DB + all artifact files)
        proc = get_processor()
        ep_repo = EpisodeRepository()
        
        # We need a way to get all episodes IDs first
        # Assuming get_by_subscription returns models with IDs
        episodes = ep_repo.get_by_subscription(sub.id)
        for ep in episodes:
            await proc.delete_episode(ep.id)
            
        # 2. Delete subscription-level folders/files
        
        # Audio directory (Subscription folder)
        dir_path = os.path.join(settings.AUDIO_DIR, sub.slug)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")
        
        # Feed file
        feed_path = os.path.join(settings.FEEDS_DIR, f"{sub.slug}.xml")
        if os.path.exists(feed_path):
            try:
                os.remove(feed_path)
            except Exception as e:
                 print(f"Error deleting feed file {feed_path}: {e}")

        # 3. Delete Subscription from DB
        repo.delete(id)
    
    return {"status": "deleted"}

@router.delete("/episodes/{id}")
async def delete_episode(id: int, user = Depends(require_auth)):
    """Ignore a specific episode and remove its local files."""
    proc = get_processor()
    success = await proc.delete_episode(id)
    if not success:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"status": "deleted"}

@router.post("/subscriptions/{id}/check")
async def check_subscription_updates(id: int, background_tasks: BackgroundTasks, user = Depends(require_auth)):
    """Trigger a check for new episodes."""
    # We allow running check_feeds in background task because it's I/O bound (network)
    # and doesn't use heavy CPU. The background loop will then pick up any new pending episodes.
    proc = get_processor()
    background_tasks.add_task(proc.check_feeds, subscription_id=id)
    return {"status": "check_triggered"}

@router.post("/episodes/{id}/process")
async def process_episode(id: int, skip_transcription: bool = False, user = Depends(require_auth)):
    """Manually trigger processing for an episode."""
    ep_repo = EpisodeRepository()
    
    import json
    flags = {'skip_transcription': skip_transcription}
    flags_json = json.dumps(flags)
    
    # Using reset_status to ensure clean state but with flags
    ep_repo.reset_status(id, processing_flags=flags_json)
    ep_repo.update_status(id, "pending")
    
    # Background processor (polling) will pick this up
    return {"status": "processing_triggered"}

@router.post("/episodes/{id}/cancel")
async def cancel_episode(id: int, user = Depends(require_auth)):
    """Cancel processing, ignore the episode, and remove local files."""
    proc = get_processor()
    success = await proc.delete_episode(id)
    if not success:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {"status": "cancelled"}

class SearchQuery(BaseModel):
    query: str

@router.post("/search")
async def search_podcasts(q: SearchQuery, user = Depends(require_auth)):
    return await PodcastSearcher.search(q.query)

@router.post("/episodes/{id}/track-listen")
async def track_listen(id: int, user = Depends(require_auth)):
    """Increment listen count for an episode."""
    ep_repo = EpisodeRepository()
    ep = ep_repo.get_by_id(id)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    ep_repo.increment_listen_count(id)
    return {"status": "tracked", "episode_id": id}
