from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class SubscriptionBase(BaseModel):
    feed_url: str

class SubscriptionCreate(SubscriptionBase):
    pass

class Subscription(SubscriptionBase):
    id: int
    title: Optional[str] = None
    slug: Optional[str] = None
    image_url: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_checked_at: Optional[datetime] = None
    
    # Granular Ad Removal Settings
    remove_ads: bool = True
    remove_promos: bool = True
    remove_intros: bool = False
    remove_outros: bool = False
    custom_instructions: Optional[str] = None
    
    # New Features
    append_summary: bool = False
    append_title_intro: bool = False

    class Config:
        from_attributes = True

class EpisodeBase(BaseModel):
    guid: str
    title: str
    pub_date: Optional[datetime] = None
    original_url: str
    duration: Optional[int] = None

class Episode(EpisodeBase):
    id: int
    subscription_id: int
    status: str
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_step: Optional[str] = None
    progress: int = 0
    transcript_path: Optional[str] = None
    ad_report_path: Optional[str] = None
    processing_flags: Optional[str] = None
    description: Optional[str] = None
    report_path: Optional[str] = None
    file_size: Optional[int] = None

    class Config:
        from_attributes = True
