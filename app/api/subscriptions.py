from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from app.core.models import Subscription, SubscriptionCreate
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.feed import FeedManager
from app.core.processor import Processor
from app.core.search import PodcastSearcher
from pydantic import BaseModel
import shutil
import os
from app.core.config import settings


router = APIRouter()
repo = SubscriptionRepository()

# Helper to get processor (in a real app, use dependency injection)
def get_processor():
    return Processor()

@router.get("/subscriptions", response_model=List[Subscription])
async def list_subscriptions():
    return repo.get_all()

@router.post("/subscriptions", response_model=Subscription)
async def create_subscription(sub: SubscriptionCreate, initial_count: int = 5):
    existing = repo.get_by_url(sub.feed_url)
    if existing:
        raise HTTPException(status_code=400, detail="Subscription already exists")
    
    try:
        # Parse feed to get title
        title, slug, image_url = FeedManager.parse_feed(sub.feed_url)
        
        # Save to DB
        new_sub = repo.create(sub, title, slug, image_url)
        
        # Trigger initial check
        proc = get_processor()
        await proc.check_feeds(subscription_id=new_sub.id, limit=initial_count)
        
        # Trigger processing in background
        # In a simple app, we might just let the loop handle it, or trigger it explicitly
        # For now, let's just trigger the queue processing
        await proc.process_queue()
        

        
        return new_sub
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/subscriptions/{id}")
async def delete_subscription(id: int):
    """Delete a subscription and its episodes."""
    sub = repo.get_by_id(id)
    
    if sub:
        # Delete audio directory
        dir_path = os.path.join(settings.AUDIO_DIR, sub.slug)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")
                
        repo.delete(id)
    
    # Regenerate static site
    return {"status": "deleted"}

@router.post("/subscriptions/{id}/check")
async def check_subscription_updates(id: int, background_tasks: BackgroundTasks):
    """Trigger a check for new episodes."""
    proc = get_processor()
    background_tasks.add_task(proc.check_feeds, subscription_id=id)
    background_tasks.add_task(proc.process_queue)
    return {"status": "check_triggered"}

@router.post("/episodes/{id}/process")
async def process_episode(id: int, background_tasks: BackgroundTasks, skip_transcription: bool = False):
    """Manually trigger processing for an episode."""
    ep_repo = EpisodeRepository()
    
    import json
    flags = json.dumps({'skip_transcription': skip_transcription}) if skip_transcription else None
    
    # Using reset_status to ensure clean state but with flags
    ep_repo.reset_status(id, processing_flags=flags)
    ep_repo.update_status(id, "pending")
    
    proc = get_processor()
    background_tasks.add_task(proc.process_queue)
    return {"status": "processing_triggered"}

@router.post("/episodes/{id}/cancel")
async def cancel_episode(id: int):
    """Cancel processing and reset status."""
    ep_repo = EpisodeRepository()
    ep = ep_repo.get_by_id(id)
    
    if ep:
        # Clean up artifacts if they exist
        if ep.transcript_path and os.path.exists(ep.transcript_path):
            try:
                os.remove(ep.transcript_path)
            except: pass
            
        if ep.ad_report_path and os.path.exists(ep.ad_report_path):
            try:
                os.remove(ep.ad_report_path)
            except: pass
            
        if ep.report_path and os.path.exists(ep.report_path):
            try:
                os.remove(ep.report_path)
            except: pass

    ep_repo.reset_status(id)
    return {"status": "cancelled"}

class SearchQuery(BaseModel):
    query: str

@router.post("/search")
async def search_podcasts(q: SearchQuery):
    return await PodcastSearcher.search(q.query)
