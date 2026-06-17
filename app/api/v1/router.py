import json
import os
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
from app.core.system_status import get_operation_status
from app.core.url_utils import validate_http_url
from app.infra.database import get_db_connection
from app.infra.repository import EpisodeRepository, SubscriptionRepository

router = APIRouter(tags=["AI API v1"])
sub_repo = SubscriptionRepository()
ep_repo = EpisodeRepository()


def _processor() -> Processor:
    return Processor()


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


def _artifact_path(row: dict[str, Any], preferred_column: str, fallback_name: str) -> str | None:
    recorded = row.get(preferred_column)
    if recorded and os.path.exists(recorded):
        return recorded

    episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
    candidate = os.path.join(settings.get_episode_dir(row["subscription_slug"], episode_slug), fallback_name)
    return candidate if os.path.exists(candidate) else None


def _subscription_or_404(subscription_id: int) -> Subscription:
    sub = sub_repo.get_by_id(subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return sub


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
async def system_status(_principal: ApiPrincipal = Depends(require_scopes(["admin"]))):
    return get_operation_status()


@router.get("/queue", response_model=QueueResponse)
async def queue_status(_principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    return {
        "queue": ep_repo.get_queue(),
        "recently_processed": ep_repo.get_recently_processed(days=3),
        "operation_status": get_operation_status(),
    }


@router.get("/subscriptions", response_model=list[Subscription])
async def list_subscriptions(principal: ApiPrincipal = Depends(require_scopes(["read"]))):
    return sub_repo.get_all(user_id=principal.user_id, only_user=bool(principal.user_id))


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
    return await PodcastSearcher.search(request_body.query)


@router.post("/subscriptions", response_model=Subscription)
async def create_subscription(
    request_body: SubscriptionCreateRequest,
    principal: ApiPrincipal = Depends(require_scopes(["write"])),
):
    try:
        validate_http_url(request_body.feed_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = sub_repo.get_by_url(request_body.feed_url)
    if existing:
        if principal.user_id:
            sub_repo.add_to_user_library(principal.user_id, existing.id)
        return existing

    try:
        title, slug, image_url, description = FeedManager.parse_feed(request_body.feed_url)
        new_sub = sub_repo.create(
            sub=SubscriptionCreate(feed_url=request_body.feed_url),
            title=title,
            slug=slug,
            image_url=image_url,
            description=description,
            owner_user_id=principal.user_id,
        )
        await send_notification_async(
            EVENT_NEW_PODCAST,
            "Podcast added",
            f"{title} was added to the global podcast library.",
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
    _principal: ApiPrincipal = Depends(require_scopes(["write"])),
):
    sub = _subscription_or_404(subscription_id)
    updates = request_body.model_dump(exclude_unset=True)
    sub_repo.update_settings(
        subscription_id,
        updates.get("remove_ads", sub.remove_ads),
        updates.get("remove_promos", sub.remove_promos),
        updates.get("remove_intros", sub.remove_intros),
        updates.get("remove_outros", sub.remove_outros),
        updates.get("custom_instructions", sub.custom_instructions),
        updates.get("append_summary", sub.append_summary),
        updates.get("append_title_intro", sub.append_title_intro),
        updates.get("ai_rewrite_description", sub.ai_rewrite_description),
        updates.get("ai_audio_summary", sub.ai_audio_summary),
        updates.get("retention_days", sub.retention_days or 30),
        updates.get("manual_retention_days", sub.manual_retention_days or 14),
        updates.get("retention_limit", sub.retention_limit or 1),
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
    _principal: ApiPrincipal = Depends(require_scopes(["process"])),
):
    _subscription_or_404(subscription_id)
    background_tasks.add_task(_processor().check_feeds, subscription_id=subscription_id)
    return {"status": "check_triggered", "id": subscription_id}


@router.post("/episodes/{episode_id}/download", response_model=ActionResponse)
async def download_episode(episode_id: int, _principal: ApiPrincipal = Depends(require_scopes(["process"]))):
    _get_episode_row(episode_id)
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET is_manual_download = 1 WHERE id = ?", (episode_id,))
        conn.commit()
    ep_repo.update_status(episode_id, "pending")
    return {"status": "download_queued", "id": episode_id}


@router.post("/episodes/{episode_id}/reprocess", response_model=ActionResponse)
async def reprocess_episode(
    episode_id: int,
    skip_transcription: bool = False,
    _principal: ApiPrincipal = Depends(require_scopes(["process"])),
):
    _get_episode_row(episode_id)
    current_status = ep_repo.get_status(episode_id)
    if current_status == "processing":
        return {"status": "ignored", "id": episode_id, "detail": "already_processing"}

    proc = _processor()
    await proc.version_episode(episode_id)
    ep_repo.reset_status(episode_id, processing_flags=json.dumps({"skip_transcription": skip_transcription}))
    ep_repo.update_status(episode_id, "pending")
    return {"status": "reprocess_queued", "id": episode_id}


@router.post("/episodes/{episode_id}/cancel", response_model=ActionResponse)
async def cancel_episode(episode_id: int, _principal: ApiPrincipal = Depends(require_scopes(["process"]))):
    _get_episode_row(episode_id)
    ep_repo.reset_status(episode_id)
    return {"status": "cancelled", "id": episode_id}


@router.post("/episodes/{episode_id}/ignore", response_model=ActionResponse)
async def ignore_episode(episode_id: int, _principal: ApiPrincipal = Depends(require_scopes(["process"]))):
    _get_episode_row(episode_id)
    await _processor().delete_episode(episode_id)
    return {"status": "ignored", "id": episode_id}
