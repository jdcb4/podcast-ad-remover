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
import asyncio
from app.core.sources import resolve_source


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
    try:
        source = await asyncio.to_thread(resolve_source, sub.feed_url)
        if source.source_type.startswith("youtube_") and initial_count not in {0, 1, 3, 5}:
            raise ValueError("YouTube initial import must be 0, 1, 3, or 5 videos")
        identity_lookup = getattr(repo, "get_by_source_identity", lambda *_args: None)
        existing = identity_lookup(source.source_type, source.external_id) or repo.get_by_url(source.canonical_url)
        if existing:
            added = repo.add_to_user_library(_real_user_id(user), existing.id)
            if added:
                return existing
            raise ValueError("Subscription already exists in your podcasts")
        slug_method = getattr(repo, "available_slug", lambda base, _external: base)
        slug = slug_method(source.slug, source.external_id)
        create_args = {
            "description": source.description,
            "owner_user_id": _real_user_id(user),
        }
        if source.source_type != "rss":
            create_args.update(source_type=source.source_type, source_external_id=source.external_id)
        new_sub = repo.create(
            SubscriptionCreate(feed_url=source.canonical_url),
            source.title,
            slug,
            source.image_url,
            **create_args,
        )
        await send_notification_async(
            EVENT_NEW_PODCAST,
            "Podcast added",
            f"{source.title} was added to the global podcast library.",
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

        proc = get_processor()
        status = await proc.delete_subscription(id)
        return {"status": status}

    # Repeated deletion is intentionally idempotent.
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
    try:
        return await PodcastSearcher.search(q.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/episodes/{id}/track-listen")
async def track_listen(id: int, user = Depends(require_auth)):
    """Increment listen count for an episode."""
    ep_repo = EpisodeRepository()
    ep = ep_repo.get_by_id(id)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    ep_repo.increment_listen_count(id)
    return {"status": "tracked", "episode_id": id}
