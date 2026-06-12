from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

class SubscriptionBase(BaseModel):
    feed_url: str

class SubscriptionCreate(SubscriptionBase):
    pass

class Subscription(SubscriptionBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: Optional[str] = None
    description: Optional[str] = None
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
    ai_rewrite_description: bool = False
    ai_audio_summary: bool = False
    owner_user_id: Optional[int] = None
    
    # Retention
    retention_days: Optional[int] = 30
    manual_retention_days: Optional[int] = 14
    retention_limit: Optional[int] = 1

class EpisodeBase(BaseModel):
    guid: str
    title: str
    pub_date: Optional[datetime] = None
    original_url: str
    duration: Optional[int] = None

class Episode(EpisodeBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subscription_id: int
    status: str
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_step: Optional[str] = None
    progress: int = 0
    transcript_path: Optional[str] = None
    ai_summary: Optional[str] = None
    ad_report_path: Optional[str] = None
    processing_flags: Optional[str] = None
    description: Optional[str] = None
    report_path: Optional[str] = None
    file_size: Optional[int] = None
    local_filename: Optional[str] = None
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    is_manual_download: bool = False
    listen_count: int = 0

class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    username: str
    password_hash: str
    is_admin: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class AccessRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    username: str
    email: Optional[str] = None
    reason: Optional[str] = None
    password_hash: Optional[str] = None
    requested_at: Optional[datetime] = None
    status: str = "pending"
    ip_address: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

class LoginAttempt(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    success: bool
    timestamp: Optional[datetime] = None
    user_agent: Optional[str] = None
