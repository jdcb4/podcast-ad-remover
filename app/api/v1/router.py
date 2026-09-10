from app.core.artifacts import artifact_path as _artifact_path
import json
import os
import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.openapi.utils import get_openapi

from app.api.v1.dependencies import ensure_ai_api_enabled, get_api_settings, require_scopes
from app.api.v1.schemas import (
    ActionResponse,
    ApiPrincipal,
    ApiStatus,
    CapabilityResponse,
    PaginatedEpisodes,
    QueueResponse,
    ReportResponse,
    SearchRequest,
    SubscriptionCreateRequest,
    SubscriptionSettingsUpdate,
    TranscriptResponse,
)
from app.core.config import settings
from app.core.feed import FeedManager
from app.core.models import Subscription, SubscriptionCreate
from app.core.notifications import EVENT_NEW_PODCAST, send_notification_async
from app.core.processor import Processor
from app.core.search import PodcastSearcher
from app.core.sources import resolve_source
from app.core.queue_status import get_queue_payload
from app.core.system_status import get_operation_status
from app.core.url_utils import validate_http_url
from app.infra.database import get_db_connection
from app.infra.repository import EpisodeRepository, SubscriptionRepository

router = APIRouter(tags=["AI API v1"])
sub_repo = SubscriptionRepository()
ep_repo = EpisodeRepository()


def _processor() -> Processor:
    return Processor()


def _token_has_global_access(principal: ApiPrincipal) -> bool:
    return principal.is_admin or principal.user_id is None


def _require_admin_user(principal: ApiPrincipal) -> None:
    if principal.user_id is not None and not principal.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API token is not linked to an admin user")


def _can_manage_subscription(principal: ApiPrincipal, sub: Subscription) -> bool:
    if _token_has_global_access(principal):
        return True
    return sub.owner_user_id == principal.user_id


def _require_manage_subscription(principal: ApiPrincipal, sub: Subscription, detail: str) -> None:
    if not _can_manage_subscription(principal, sub):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _get_episode_row(episode_id: int) -> dict[str, Any]:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT e.*, s.slug AS subscription_slug, s.title AS podcast_title
            FROM episodes e
            JOIN subscriptions s ON s.id = e.subscription_id
            WHERE e.id = ?
            """,
            (episode_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Episode not found")
        return dict(row)


def _subscription_or_404(subscription_id: int) -> Subscription:
    sub = sub_repo.get_by_id(subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return sub


def _subscription_manage_or_403(principal: ApiPrincipal, subscription_id: int) -> Subscription:
    sub = _subscription_or_404(subscription_id)
    _require_manage_subscription(
        principal,
        sub,
        "Only admins and the podcast owner can manage this podcast",
    )
    return sub


def _episode_manage_or_403(principal: ApiPrincipal, episode_id: int) -> dict[str, Any]:
    row = _get_episode_row(episode_id)
    sub = _subscription_or_404(row["subscription_id"])
    _require_manage_subscription(
        principal,
        sub,
        "Only admins and the podcast owner can manage this episode",
    )
    return row


@router.get("/health", response_model=ApiStatus)
async def api_health():
    api_settings = get_api_settings()
    return {"status": "healthy", "enabled": bool(api_settings.get("ai_api_enabled"))}


@router.get("/capabilities", response_model=CapabilityResponse)
async def api_capabilities(api_settings: dict = Depends(ensure_ai_api_enabled)):
    return {
        "scopes": ["read", "write", "process", "admin"],
        "rate_limits": {
            "default_requests_per_minute": int(api_settings.get("ai_api_default_requests_per_minute") or 60),
            "default_requests_per_day": int(api_settings.get("ai_api_default_requests_per_day") or 1000),
            "unauth_requests_per_minute": int(api_settings.get("ai_api_unauth_requests_per_minute") or 10),
        },
    }


@router.get("/openapi.json")
async def api_openapi(_api_settings: dict = Depends(ensure_ai_api_enabled)):
    schema_app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    schema_app.include_router(router, prefix="/api/v1")
    return get_openapi(
        title="Podcast Ad Remover AI API",
        version="v1",
        description="Opt-in REST API for AI agents and automation clients.",
        routes=schema_app.routes,
    )


@router.get("/system/status")
async def system_status(principal: ApiPrincipal = Depends(require_scopes(["admin"]))):
    _require_admin_user(principal)
    return get_operation_status()


@router.get("/queue", response_model=QueueResponse)
async def queue_status(_principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    return get_queue_payload()


@router.get("/subscriptions", response_model=list[Subscription])
async def list_subscriptions(_principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    return sub_repo.get_all()


@router.get("/subscriptions/{subscription_id}", response_model=Subscription)
async def get_subscription(subscription_id: int, _principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    return _subscription_or_404(subscription_id)


@router.get("/subscriptions/{subscription_id}/episodes", response_model=PaginatedEpisodes)
async def list_subscription_episodes(
    subscription_id: int,
    limit: int = 20,
    offset: int = 0,
    search: str | None = None,
    _principal: ApiPrincipal = Depends(require_scopes(["read"])),
):
    _subscription_or_404(subscription_id)
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    episodes = [dict(row) for row in ep_repo.get_by_subscription_paginated(subscription_id, safe_limit, safe_offset, search)]
    total = ep_repo.count_by_subscription(subscription_id, search)
    return {
        "episodes": episodes,
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "search": search,
        "has_more": safe_offset + len(episodes) < total,
    }


@router.get("/episodes/{episode_id}")
async def get_episode(episode_id: int, _principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    return _get_episode_row(episode_id)


@router.get("/episodes/{episode_id}/transcript", response_model=TranscriptResponse)
async def get_episode_transcript(episode_id: int, _principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    row = _get_episode_row(episode_id)
    path = _artifact_path(row, "transcript_path", "transcript.json")
    if not path:
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(path, "r", encoding="utf-8") as handle:
        transcript = json.load(handle)
    return {"episode_id": episode_id, "transcript": transcript}


@router.get("/episodes/{episode_id}/report", response_model=ReportResponse)
async def get_episode_report(episode_id: int, _principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    row = _get_episode_row(episode_id)
    report_path = _artifact_path(row, "report_path", "report.json") or _artifact_path(row, "ad_report_path", "report.json")
    if not report_path:
        episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
        html_path = os.path.join(settings.get_episode_dir(row["subscription_slug"], episode_slug), "report.html")
        report_path = html_path if os.path.exists(html_path) else None
    if not report_path:
        raise HTTPException(status_code=404, detail="Report not found")

    if report_path.endswith(".json"):
        with open(report_path, "r", encoding="utf-8") as handle:
            return {"episode_id": episode_id, "content_type": "application/json", "report": json.load(handle)}

    with open(report_path, "r", encoding="utf-8") as handle:
        return {"episode_id": episode_id, "content_type": "text/html", "report": handle.read()}


@router.post("/search")
async def search_podcasts(request_body: SearchRequest, _principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    try:
        return await PodcastSearcher.search(request_body.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/subscriptions", response_model=Subscription)
async def create_subscription(
    request_body: SubscriptionCreateRequest,
    principal: ApiPrincipal = Depends(require_scopes(["write"])),
):
    try:
        source = await asyncio.to_thread(resolve_source, request_body.feed_url)
        if source.source_type.startswith("youtube_") and request_body.initial_count not in {0, 1, 3, 5}:
            raise ValueError("YouTube initial import must be 0, 1, 3, or 5 videos")
        existing = (
            sub_repo.get_by_source_identity(source.source_type, source.external_id)
            or sub_repo.get_by_url(source.canonical_url)
        )
        if existing:
            if principal.user_id:
                sub_repo.add_to_user_library(principal.user_id, existing.id)
            return existing
        slug = sub_repo.available_slug(source.slug, source.external_id)
        new_sub = sub_repo.create(
            sub=SubscriptionCreate(feed_url=source.canonical_url),
            title=source.title,
            slug=slug,
            image_url=source.image_url,
            description=source.description,
            owner_user_id=principal.user_id,
            source_type=source.source_type,
            source_external_id=source.external_id,
        )
        await send_notification_async(
            EVENT_NEW_PODCAST,
            "Podcast added",
            f"{source.title} was added to the global podcast library.",
            severity="success",
        )
        await _processor().check_feeds(subscription_id=new_sub.id, limit=request_body.initial_count)
        return new_sub
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/subscriptions/{subscription_id}/settings", response_model=ActionResponse)
async def update_subscription_settings(
    subscription_id: int,
    request_body: SubscriptionSettingsUpdate,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_scopes(["write"])),
):
    sub = _subscription_or_404(subscription_id)
    _require_manage_subscription(
        principal,
        sub,
        "Only admins and the podcast owner can change podcast settings",
    )
    updates = request_body.model_dump(exclude_unset=True)
    # New optional fields treat null like omission, including inheritance effects.
    for field in ('processing_workflow', 'inherit_processing_workflow', 'remove_editorial_non_speech',
                  'remove_non_editorial_non_speech', 'minimum_retained_seconds'):
        if updates.get(field) is None:
            updates.pop(field, None)
    stored = sub.setting_overrides
    content_fields = {"remove_ads", "remove_promos", "remove_intros", "remove_outros",
                      "remove_editorial_non_speech", "remove_non_editorial_non_speech", "minimum_retained_seconds"}
    retention_fields = {"retention_days", "manual_retention_days", "retention_limit"}
    feature_fields = {
        "append_summary",
        "append_title_intro",
        "ai_rewrite_description",
        "ai_audio_summary",
        "watermark_artwork",
    }

    inherit_content_removal = updates.get(
        "inherit_content_removal",
        False if content_fields.intersection(updates) else sub.inherit_content_removal,
    )
    inherit_retention = updates.get(
        "inherit_retention",
        False if retention_fields.intersection(updates) else sub.inherit_retention,
    )
    inherit_default_features = updates.get(
        "inherit_default_features",
        False if feature_fields.intersection(updates) else sub.inherit_default_features,
    )
    inherit_custom_instructions = updates.get(
        "inherit_custom_instructions",
        sub.inherit_custom_instructions,
    )
    if "custom_instructions" in updates:
        if updates["custom_instructions"] is None or not updates["custom_instructions"].strip():
            inherit_custom_instructions = True
        elif "inherit_custom_instructions" not in updates:
            inherit_custom_instructions = False

    sub_repo.update_settings(
        subscription_id,
        updates.get("remove_ads", stored.get("remove_ads")),
        updates.get("remove_promos", stored.get("remove_promos")),
        updates.get("remove_intros", stored.get("remove_intros")),
        updates.get("remove_outros", stored.get("remove_outros")),
        updates.get("custom_instructions", stored.get("custom_instructions")),
        updates.get("append_summary", stored.get("append_summary")),
        updates.get("append_title_intro", stored.get("append_title_intro")),
        updates.get("ai_rewrite_description", stored.get("ai_rewrite_description")),
        updates.get("ai_audio_summary", stored.get("ai_audio_summary")),
        updates.get("retention_days", stored.get("retention_days") or 30),
        updates.get("manual_retention_days", stored.get("manual_retention_days") or 14),
        (
            updates["retention_limit"]
            if "retention_limit" in updates
            else stored.get("retention_limit")
            if stored.get("retention_limit") is not None
            else 1
        ),
        inherit_content_removal=inherit_content_removal,
        inherit_retention=inherit_retention,
        inherit_default_features=inherit_default_features,
        inherit_custom_instructions=inherit_custom_instructions,
        watermark_artwork=updates.get("watermark_artwork", stored.get("watermark_artwork")),
        processing_workflow=updates.get("processing_workflow"),
        inherit_processing_workflow=updates.get("inherit_processing_workflow",
                                               False if updates.get("processing_workflow") else None),
        remove_editorial_non_speech=updates.get("remove_editorial_non_speech"),
        remove_non_editorial_non_speech=updates.get("remove_non_editorial_non_speech"),
        minimum_retained_seconds=updates.get("minimum_retained_seconds"),
    )

    proc = _processor()

    async def post_update_tasks(sub_id: int):
        await proc.cleanup_old_episodes()
        await proc.check_feeds(sub_id)

    background_tasks.add_task(post_update_tasks, subscription_id)
    return {"status": "updated", "id": subscription_id}


@router.post("/subscriptions/{subscription_id}/check", response_model=ActionResponse)
async def check_subscription(
    subscription_id: int,
    background_tasks: BackgroundTasks,
    principal: ApiPrincipal = Depends(require_scopes(["process"])),
):
    _subscription_manage_or_403(principal, subscription_id)
    background_tasks.add_task(_processor().check_feeds, subscription_id=subscription_id)
    return {"status": "check_triggered", "id": subscription_id}


@router.post("/episodes/{episode_id}/download", response_model=ActionResponse)
async def download_episode(episode_id: int, principal: ApiPrincipal = Depends(require_scopes(["process"]))):
    _episode_manage_or_403(principal, episode_id)
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET is_manual_download = 1 WHERE id = ?", (episode_id,))
        conn.commit()
    ep_repo.update_status(episode_id, "pending")
    return {"status": "download_queued", "id": episode_id}


@router.post("/episodes/{episode_id}/reprocess", response_model=ActionResponse)
async def reprocess_episode(
    episode_id: int,
    skip_transcription: bool = False,
    principal: ApiPrincipal = Depends(require_scopes(["process"])),
):
    _episode_manage_or_403(principal, episode_id)
    current_status = ep_repo.get_status(episode_id)
    if current_status == "processing":
        return {"status": "ignored", "id": episode_id, "detail": "already_processing"}

    proc = _processor()
    await proc.version_episode(episode_id)
    ep_repo.reset_status(episode_id, processing_flags=json.dumps({"skip_transcription": skip_transcription}))
    ep_repo.update_status(episode_id, "pending")
    return {"status": "reprocess_queued", "id": episode_id}


@router.post("/episodes/{episode_id}/cancel", response_model=ActionResponse)
async def cancel_episode(episode_id: int, principal: ApiPrincipal = Depends(require_scopes(["process"]))):
    _episode_manage_or_403(principal, episode_id)
    ep_repo.reset_status(episode_id)
    return {"status": "cancelled", "id": episode_id}


@router.post("/episodes/{episode_id}/ignore", response_model=ActionResponse)
async def ignore_episode(episode_id: int, principal: ApiPrincipal = Depends(require_scopes(["process"]))):
    _episode_manage_or_403(principal, episode_id)
    await _processor().delete_episode(episode_id)
    return {"status": "ignored", "id": episode_id}
