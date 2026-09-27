"""Optional setup: server-memory drafts, explicit review/apply, no secrets in cookies."""
import secrets
import time
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from app.web.auth import require_admin
from app.infra.database import get_db_connection
from app.core.url_utils import validate_http_url
from app.core.provider_settings import MODEL_FIELDS, credential

router = APIRouter()
_DRAFTS = {}
_TTL = 1800


def draft_for(request, values):
    now = time.monotonic()
    for token in list(_DRAFTS):
        if _DRAFTS[token]['expires'] < now:
            del _DRAFTS[token]
    token = request.session.get('setup_draft')
    if token not in _DRAFTS:
        token = secrets.token_urlsafe(24)
        request.session['setup_draft'] = token
        _DRAFTS[token] = {'baseline': dict(values), 'changes': {}, 'expires': now + _TTL}
    draft = _DRAFTS[token]
    draft['expires'] = now + _TTL
    return draft


def discard(request):
    _DRAFTS.pop(request.session.pop('setup_draft', None), None)


def _label(key):
    labels = {'app_external_url':'Application URL', 'active_ai_provider':'Analysis provider',
              'ai_model_cascade':'Gemini model', 'custom_llm_model':'Custom analysis model',
              'custom_llm_base_url':'Custom endpoint URL', 'whisper_model':'Transcription model',
              'whisper_device':'Transcription device'}
    return labels.get(key, key.replace('default_', '').replace('_', ' ').capitalize())


def render(request, step, draft, error=None, entered=None):
    from app.web.router import templates, _admin_context
    context = _admin_context(request, 'setup')
    effective = {**context['settings'], **draft['changes']}
    context.update(step=max(1, min(5, step)), settings=effective, error=error, entered=entered or {},
                   changes={_label(k): ('Configured' if 'api_key' in k else v) for k,v in draft['changes'].items()})
    context['credential_configured'] = bool(credential(effective.get('active_ai_provider','gemini'), effective))
    return templates.TemplateResponse(request=request, name='admin/setup.html', context=context, status_code=400 if error else 200)


@router.get('/admin/setup')
async def setup(request: Request, step: int = 1, user=Depends(require_admin)):
    from app.web.router import get_global_settings
    return render(request, step, draft_for(request, get_global_settings()))


@router.post('/admin/setup')
async def save_setup(request: Request, step: int = Form(1), action: str = Form('next'), user=Depends(require_admin)):
    from app.web.router import get_global_settings
    from app.core.provider_readiness import provider_configuration_error
    values = get_global_settings()
    if action == 'dismiss':
        discard(request)
        with get_db_connection() as conn:
            conn.execute("UPDATE app_settings SET onboarding_status='dismissed' WHERE id=1")
            conn.commit()
        return RedirectResponse('/', status_code=303)
    token = request.session.get('setup_draft')
    if action == 'finish' and (token not in _DRAFTS or _DRAFTS[token]['expires'] < time.monotonic()):
        discard(request)
        return RedirectResponse('/admin/setup?error=Setup+expired.+Please+review+again.', status_code=303)
    draft = draft_for(request, values)
    if action == 'finish':
        changes = draft['changes']
        with get_db_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = dict(conn.execute('SELECT * FROM app_settings WHERE id=1').fetchone())
            conflicts = [key for key in changes if current.get(key) != draft['baseline'].get(key)]
            if conflicts:
                conn.rollback()
                return render(request, 5, draft, 'Settings changed elsewhere. Cancel and restart setup to review the current values.')
            updates = {**changes, 'onboarding_status':'completed'}
            conn.execute('UPDATE app_settings SET '+', '.join(f'{key}=?' for key in updates)+' WHERE id=1',tuple(updates.values()))
            conn.commit()
        discard(request)
        return RedirectResponse('/#add', status_code=303)
    form = await request.form()
    updates = {}
    try:
        if step == 1:
            url = str(form.get('app_external_url','')).strip()
            if url:
                validate_http_url(url, allow_private=True)
            updates['app_external_url'] = url or None
        elif step == 2:
            if action == 'skip':
                return RedirectResponse('/admin/setup?step=3', status_code=303)
            provider = str(form.get('provider',''))
            if provider not in MODEL_FIELDS:
                raise ValueError('Choose a text analysis provider')
            model = str(form.get('model','')).strip()
            if not model or model.startswith('['):
                raise ValueError('Enter one model ID')
            updates.update(active_ai_provider=provider)
            updates[MODEL_FIELDS[provider]] = model
            if provider == 'custom':
                url = str(form.get('base_url','')).strip()
                validate_http_url(url, allow_private=True)
                updates['custom_llm_base_url'] = url
            key = str(form.get('api_key','')).strip()
            if key:
                updates['custom_llm_api_key' if provider == 'custom' else provider+'_api_key'] = key
            error = provider_configuration_error({**values, **draft['changes'], **updates})
            if error:
                raise ValueError(error)
        elif step == 3:
            model, device = str(form.get('whisper_model','base')), str(form.get('whisper_device','cpu'))
            if model not in {'tiny','base','small','medium','large'} or device not in {'cpu','cuda'}:
                raise ValueError('Choose a supported transcription model and device')
            updates.update(whisper_model=model, whisper_device=device)
        elif step == 4:
            for field in ('ads','promos','intros','outros','non_editorial_non_speech'):
                updates['default_remove_'+field] = int(form.get('remove_'+field)=='true')
            preset = str(form.get('retention','unchanged'))
            if preset == 'standard':
                updates.update(default_retention_days=30, default_manual_retention_days=14, default_retention_limit=1)
            elif preset != 'unchanged':
                raise ValueError('Choose a retention preset')
        else:
            raise ValueError('Unknown setup step')
    except ValueError as exc:
        # Keep typed values, including a password input, in this same-origin response only.
        return render(request, step, draft, str(exc), dict(form))
    for key,value in updates.items():
        if value != draft['baseline'].get(key):
            draft['changes'][key] = value
        else:
            draft['changes'].pop(key, None)
    return RedirectResponse(f'/admin/setup?step={step+1}', status_code=303)
