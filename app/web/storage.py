"""Admin-only storage operations; never accept filesystem paths from a request."""
import asyncio
from fastapi import APIRouter, Depends, Form, Request
from filelock import Timeout
from app.web.auth import require_admin
from app.core import media_storage as storage

router = APIRouter()


@router.get('/admin/system/storage')
async def storage_page(request: Request, user=Depends(require_admin)):
    from app.web.router import templates, _admin_context
    context = _admin_context(request, 'system')
    context['storage'] = await asyncio.to_thread(storage.status)
    return templates.TemplateResponse(request=request, name='admin/storage.html', context=context)


@router.get('/admin/system/storage/status')
async def storage_status(user=Depends(require_admin)):
    return await asyncio.to_thread(storage.status)


@router.post('/admin/system/storage')
async def storage_action(request: Request, action: str = Form(...), confirmed: bool = Form(False), user=Depends(require_admin)):
    from app.web.router import templates, _admin_context
    context = _admin_context(request, 'system')
    try:
        if action == 'preview':
            context['preview'] = await asyncio.to_thread(storage.preview)
        elif action == 'enable':
            await asyncio.to_thread(storage.enable)
        elif action in {'copy', 'cleanup'}:
            if not confirmed:
                raise ValueError('Confirm the operation before starting.')
            await asyncio.to_thread(storage.start, action)
        elif action in {'pause', 'resume', 'cancel'}:
            await asyncio.to_thread(storage.control, action)
        else:
            raise ValueError('Unknown storage action')
    except (OSError, ValueError, storage.StorageUnavailable, Timeout) as exc:
        context['storage_error'] = str(exc) if not isinstance(exc, Timeout) else 'A storage operation is busy. Try again shortly.'
    context['storage'] = await asyncio.to_thread(storage.status)
    return templates.TemplateResponse(request=request, name='admin/storage.html', context=context,
                                      status_code=400 if context.get('storage_error') else 200)
