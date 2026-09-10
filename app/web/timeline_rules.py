"""Admin-only editing and request preview for the opt-in classification workflow."""
import json

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core import timeline
from app.core.utils import get_global_settings
from app.infra.database import get_db_connection
from app.infra.repository import SubscriptionRepository
from app.web.auth import require_admin

router = APIRouter()


def prompt_updates(form, current):
    effective = timeline.definitions(current)
    for label in timeline.DEFINITIONS:
        field = 'definition_' + label
        if field in form:
            effective[label] = str(form[field]).strip() or timeline.DEFINITIONS[label]
    overrides = {k: v for k, v in effective.items() if v != timeline.DEFINITIONS[k]}
    try:
        timeline.definitions({'timeline_definitions': overrides})
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    summary = str(form.get('timeline_summary_instructions', current.get('timeline_summary_instructions') or timeline.SUMMARY_DEFAULT)).strip()
    if len(summary) > 10000:
        raise HTTPException(400, 'Summary instructions must be at most 10000 characters')
    mode = form.get('timeline_output_mode', current.get('timeline_output_mode', 'auto'))
    if mode not in timeline.OUTPUT_MODES:
        raise HTTPException(400, 'Choose Auto, Require schema or JSON compatibility')
    return {'timeline_definitions': json.dumps(overrides) if overrides else None,
            'timeline_summary_instructions': summary if summary and summary != timeline.SUMMARY_DEFAULT else None,
            'timeline_output_mode': mode}


@router.post('/admin/prompts/timeline')
async def save_timeline_prompts(request: Request, admin=Depends(require_admin)):
    updates = prompt_updates(await request.form(), get_global_settings())
    with get_db_connection() as conn:
        conn.execute('''UPDATE app_settings SET timeline_definitions=?, timeline_summary_instructions=?,
                        timeline_output_mode=? WHERE id=1''',
                     (updates['timeline_definitions'], updates['timeline_summary_instructions'], updates['timeline_output_mode']))
        conn.commit()
    return {'status': 'success', 'detail': 'Saved for newly queued Complete Timeline jobs. Legacy and queued jobs are unchanged.'}


@router.post('/admin/prompts/timeline/preview')
async def preview_timeline_prompt(request: Request, admin=Depends(require_admin)):
    form = await request.form()
    current = get_global_settings()
    proposed = {**current, **prompt_updates(form, current)}
    custom = current.get('default_custom_instructions')
    if form.get('preview_subscription_id'):
        try:
            sub = SubscriptionRepository().get_by_id(int(form['preview_subscription_id']))
        except ValueError as error:
            raise HTTPException(400, 'Invalid podcast') from error
        if not sub:
            raise HTTPException(404, 'Podcast not found')
        custom = sub.custom_instructions
    return {'system_prompt': timeline.build_prompt(proposed, custom),
            'schema': timeline.SCHEMA, 'prompt_version': timeline.PROMPT_VERSION,
            'provider_settings': timeline.model_settings(proposed),
            'transcript_input': 'At processing time, the user message contains the complete numbered timeline, explicit gaps, measured duration and episode metadata. It is treated as source data.',
            'temperature': 'Provider default (omitted)', 'reasoning': 'Provider default (omitted)',
            'fallback': 'Configured model cascade within the selected provider only'}
