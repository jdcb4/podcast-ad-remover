"""Optional first-run setup. Each page saves only its own small setting group."""
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.web.auth import require_admin
from app.infra.database import get_db_connection
from app.core.url_utils import validate_http_url
from app.core.provider_settings import MODEL_FIELDS, credential

router = APIRouter()


@router.get('/admin/setup')
async def setup(request: Request, step: int = 1, user=Depends(require_admin)):
    from app.web.router import templates, _admin_context
    context = _admin_context(request, 'setup')
    context['step'] = max(1, min(4, step))
    return templates.TemplateResponse(request=request, name='admin/setup.html', context=context)


@router.post('/admin/setup')
async def save_setup(request: Request, step: int = Form(1), action: str = Form('next'),
                     user=Depends(require_admin)):
    from app.web.router import get_global_settings
    from app.core.provider_readiness import provider_configuration_error
    values = get_global_settings()
    if action == 'dismiss':
        updates = {'onboarding_status': 'dismissed'}
        target = '/'
    elif action == 'finish':
        updates = {'onboarding_status': 'completed'}
        target = '/#add'
    else:
        form = await request.form()
        updates = {'onboarding_status': 'in_progress'}
        try:
            if step == 1:
                url = str(form.get('app_external_url', '')).strip()
                if url:
                    validate_http_url(url, allow_private=True)
                updates['app_external_url'] = url or None
            elif step == 2:
                provider = str(form.get('provider', ''))
                if provider not in MODEL_FIELDS:
                    raise ValueError('Choose a text analysis provider')
                model = str(form.get('model', '')).strip()
                if not model or model.startswith('['):
                    raise ValueError('Enter one model ID')
                updates.update(active_ai_provider=provider)
                updates[MODEL_FIELDS[provider]] = model
                if provider == 'custom':
                    updates['custom_llm_base_url'] = str(form.get('base_url', '')).strip()
                key = str(form.get('api_key', '')).strip()
                if key:
                    updates['custom_llm_api_key' if provider == 'custom' else provider + '_api_key'] = key
                error = provider_configuration_error({**values, **updates})
                if error:
                    raise ValueError(error)
            elif step == 3:
                for field in ('ads', 'promos', 'intros', 'outros', 'non_editorial_non_speech'):
                    updates['default_remove_' + field] = int(form.get('remove_' + field) == 'true')
            else:
                raise ValueError('Unknown setup step')
        except ValueError as exc:
            from urllib.parse import quote
            return RedirectResponse(f'/admin/setup?step={step}&error={quote(str(exc))}', status_code=303)
        target = f'/admin/setup?step={step + 1}'
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET ' + ', '.join(f'{key} = ?' for key in updates) + ' WHERE id = 1', tuple(updates.values()))
        conn.commit()
    return RedirectResponse(target, status_code=303)
