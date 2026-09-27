"""Authenticated import page and sequential per-feed operations."""
import asyncio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from app.core.podcast_import import MAX_BYTES, import_feed, preview_import
from app.web.auth import require_auth

router = APIRouter()


def require_import_origin(request: Request):
    from app.web.auth_utils import is_same_origin_request
    from app.web.router import get_global_settings
    if not is_same_origin_request(request, get_global_settings().get('app_external_url')):
        raise HTTPException(403, 'Cross-origin management requests are not allowed')


@router.get('/import')
async def import_page(request: Request, user=Depends(require_auth)):
    from app.web.router import templates, get_csp_nonce, get_global_settings
    return templates.TemplateResponse(request=request, name='import.html', context={
        'user': user, 'settings': get_global_settings(), 'csp_nonce': get_csp_nonce(request),
    })


@router.post('/import/preview', dependencies=[Depends(require_import_origin)])
async def preview(text: str = Form(''), file: UploadFile | None = File(None), user=Depends(require_auth)):
    from app.web.router import _real_user_id
    try:
        if file and file.filename:
            if text.strip():
                raise ValueError('Choose a file or paste a list, not both.')
            raw = await file.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError('Use a file smaller than 1 MiB.')
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeError as exc:
                raise ValueError('Save this file as UTF-8 and try again.') from exc
        return {'items': await asyncio.to_thread(preview_import, text, _real_user_id(user))}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post('/import/add', dependencies=[Depends(require_import_origin)])
async def add(feed_url: str = Form(...), user=Depends(require_auth)):
    from app.web.router import _real_user_id
    try:
        return await asyncio.to_thread(import_feed, feed_url, _real_user_id(user))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, str(exc)) from exc
