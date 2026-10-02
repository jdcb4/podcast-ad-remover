"""Shared, permission-checked web/API transport for reviewed podcast actions."""
import asyncio
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from app.core import podcast_operations as service
from app.core.permissions import can_manage_subscription
from app.infra.repository import SubscriptionRepository
from app.web.auth import require_auth
from app.web.podcast_import import require_import_origin
from app.api.v1.dependencies import require_scopes


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['feed', 'archive']
    url: str | None = Field(default=None, max_length=4096)


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    preview_id: str = Field(min_length=32, max_length=32)
    choices: dict[str, int | None] = Field(default_factory=dict, max_length=10000)


class BatchRequest(BaseModel):
    action: Literal['pause', 'resume', 'cancel']


def authorize(subscription_id, user, api=False):
    sub = SubscriptionRepository().get_by_id(subscription_id)
    if not sub:
        raise HTTPException(404, 'Podcast not found')
    permitted = can_manage_subscription(user, sub)
    if api:
        permitted = user.is_admin or user.user_id is None or sub.owner_user_id == user.user_id
    if not permitted:
        raise HTTPException(403, 'Only the owner or an administrator can manage this podcast.')


async def call(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def routes(api=False, router=None):
    router = router if router is not None else APIRouter(dependencies=[] if api else [Depends(require_import_origin)])
    auth = require_scopes(['write']) if api else require_auth

    @router.post('/subscriptions/{subscription_id}/operations/preview')
    async def preview(subscription_id: int, payload: PreviewRequest, user=Depends(auth)):
        authorize(subscription_id, user, api)
        return await call(service.preview, subscription_id, payload.kind, payload.url)

    @router.post('/subscriptions/{subscription_id}/operations/confirm')
    async def confirm(subscription_id: int, payload: ConfirmRequest, user=Depends(auth)):
        authorize(subscription_id, user, api)
        return await call(service.confirm, subscription_id, payload.preview_id, payload.choices)

    @router.get('/subscriptions/{subscription_id}/archive-batches')
    async def batches(subscription_id: int, user=Depends(auth)):
        authorize(subscription_id, user, api)
        return await call(service.batches, subscription_id)

    @router.post('/subscriptions/{subscription_id}/archive-batches/{batch_id}')
    async def control(subscription_id: int, batch_id: int, payload: BatchRequest, user=Depends(auth)):
        authorize(subscription_id, user, api)
        return await call(service.control, subscription_id, batch_id, payload.action)

    return router


router = routes()

