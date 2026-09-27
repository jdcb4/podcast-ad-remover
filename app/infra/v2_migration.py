"""One-time v2 data conversion. Historical publications and ownership are untouched."""
import json

from app.core.provider_settings import MODEL_FIELDS, first_value


def migrate(conn):
    from app.core import timeline
    from app.core.subscription_settings import resolve_subscription_row

    if conn.execute("SELECT 1 FROM jobs WHERE status='running' LIMIT 1").fetchone():
        raise RuntimeError('Stop processing and drain running jobs before upgrading to v2')
    current = dict(conn.execute('SELECT * FROM app_settings WHERE id=1').fetchone())
    changes = {'models': [], 'speech_setup_required': current.get('tts_provider') != 'gemini',
        'credential_policy': 'First configured credential retained; environment takes precedence. Extra credentials are not used.',
        'cut_tones': 'Enabled at all applicable cuts' if any(current.get('warning_tone_'+p) for p in ('start','middle','end')) else 'Off',
        'retired': ['Legacy classification', 'Global free-form detection instructions', 'Editorial non-speech removal', 'SponsorBlock', 'Custom unified-feed description'],
        'ownership': 'Unchanged', 'speech_action': 'Configure provider, model and voice in Settings > Voice; optional speech preferences are retained.'}

    for field in [*MODEL_FIELDS.values(), 'gemini_tts_model_cascade']:
        value = first_value(current.get(field))
        conn.execute(f'UPDATE app_settings SET {field}=? WHERE id=1', (value,))
        if value != current.get(field):
            changes['models'].append({'setting':field,'selected':value,'previous':current.get(field)})
    key = first_value(current.get('gemini_api_keys')) or first_value(current.get('gemini_api_key'))
    conn.execute("UPDATE app_settings SET gemini_api_key=?, gemini_api_keys=NULL WHERE id=1", (key or None,))
    conn.execute('''UPDATE app_settings SET
        tts_provider=CASE WHEN tts_provider='gemini' THEN 'gemini' ELSE 'unconfigured' END,
        tts_model=?, tts_voice=gemini_tts_voice,
        cut_tone_enabled=(warning_tone_start OR warning_tone_middle OR warning_tone_end),
        default_processing_workflow='complete_timeline', timeline_output_mode='strict',
        default_remove_editorial_non_speech=0, default_custom_instructions=NULL, whitelist_mode=0
        WHERE id=1''', (first_value(current.get('gemini_tts_model_cascade')),))
    conn.execute("UPDATE app_settings SET unified_feed_artwork_source='url' WHERE trim(coalesce(unified_feed_artwork_url,'')) != ''")
    conn.execute('''UPDATE subscriptions SET processing_workflow='complete_timeline',
        remove_editorial_non_speech=0,
        ai_rewrite_description=(ai_rewrite_description OR append_summary),
        ai_audio_summary=(ai_audio_summary OR append_summary), append_summary=0''')
    values = dict(conn.execute('SELECT * FROM app_settings WHERE id=1').fetchone())
    count = 0
    for job in conn.execute("SELECT id,episode_id,processing_snapshot FROM jobs WHERE status IN ('queued','retry_scheduled','rate_limited')").fetchall():
        sub = conn.execute('SELECT s.* FROM subscriptions s JOIN episodes e ON e.subscription_id=s.id WHERE e.id=?', (job['episode_id'],)).fetchone()
        if not sub:
            continue
        snapshot = timeline.make_snapshot(resolve_subscription_row(dict(sub), values), values)
        # Retain compatible frozen classification and cut choices for existing timeline jobs.
        old = json.loads(job['processing_snapshot']) if job['processing_snapshot'] else {}
        if old.get('workflow') == 'complete_timeline':
            snapshot['options'].update(old.get('options', {}))
            snapshot['options']['remove_editorial_non_speech'] = False
            snapshot['settings'].update(old.get('settings', {}))
            for field in MODEL_FIELDS.values():
                if field in snapshot['settings']:
                    snapshot['settings'][field] = first_value(snapshot['settings'][field])
            snapshot['settings']['timeline_output_mode'] = 'strict'
            for field in ('prompt', 'summary_instructions'):
                if field in old:
                    snapshot[field] = old[field]
        snapshot['migration'] = 'v2'
        conn.execute('UPDATE jobs SET processing_snapshot=? WHERE id=?', (json.dumps(snapshot), job['id']))
        count += 1
    changes['converted_jobs'] = count
    conn.execute('UPDATE app_settings SET v2_migration_report=? WHERE id=1', (json.dumps(changes),))
