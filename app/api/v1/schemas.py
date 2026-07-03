from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ApiPrincipal(BaseModel):
    token_id: int
    token_prefix: str
    name: str
    scopes: set[str]
    user_id: int | None = None
    username: str | None = None
    is_admin: bool = False
    requests_per_minute: int | None = None
    requests_per_day: int | None = None


class ApiStatus(BaseModel):
    status: str
    enabled: bool


class CapabilityResponse(BaseModel):
    name: str = "Podcast Ad Remover API"
    version: str = "v1"
    auth: str = "bearer"
    scopes: list[str]
    rate_limits: dict[str, int]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)


class SubscriptionCreateRequest(BaseModel):
    feed_url: str = Field(..., min_length=1)
    initial_count: int = Field(default=5, ge=0, le=50)


class SubscriptionSettingsUpdate(BaseModel):
    remove_ads: bool | None = None
    remove_promos: bool | None = None
    remove_intros: bool | None = None
    remove_outros: bool | None = None
    custom_instructions: str | None = None
    append_summary: bool | None = None
    append_title_intro: bool | None = None
    ai_rewrite_description: bool | None = None
    ai_audio_summary: bool | None = None
    feed_url: str | None = None
    retention_days: int | None = Field(default=None, ge=0)
    manual_retention_days: int | None = Field(default=None, ge=0)
    retention_limit: int | None = Field(default=None, ge=0)


class ActionResponse(BaseModel):
    status: str
    id: int | None = None
    detail: str | None = None


class PaginatedEpisodes(BaseModel):
    episodes: list[dict[str, Any]]
    total: int
    offset: int
    limit: int
    search: str | None = None
    has_more: bool


class TranscriptResponse(BaseModel):
    episode_id: int
    transcript: Any


class ReportResponse(BaseModel):
    episode_id: int
    content_type: str
    report: Any


class QueueResponse(BaseModel):
    queue: list[dict[str, Any]]
    recently_processed: list[dict[str, Any]]
    operation_status: dict[str, Any]


class ApiTokenCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    requests_per_minute: int | None = Field(default=None, ge=1)
    requests_per_day: int | None = Field(default=None, ge=1)
    user_id: int | None = None


class ApiTokenCreateResponse(BaseModel):
    token: str
    name: str
    scopes: list[str]
    created_at: datetime | None = None
