"""Durable Gemini quotas, shared conservatively by every configured project/key.

API keys do not identify their Google project. Never grant another allowance on
key rotation: this installation uses one shared pool per model. Reservations and
job budgets are committed in the same SQLite write transaction before sending.
"""
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

from app.infra.database import get_db_connection


LIMITS = {
    'gemini-3.5-flash': (5, 250_000, 20),
    'gemini-3.5-flash-lite': (15, 250_000, 500),
    'gemini-3.6-flash': (5, 250_000, 20),
    'gemini-3.8-flash': (5, 250_000, 20),
    'gemini-2.5-flash': (5, 250_000, 20),
    'gemini-3.1-flash-lite': (15, 250_000, 500),
    'gemini-3.7-flash': (5, 250_000, 20),
}
PACIFIC = ZoneInfo('America/Los_Angeles')


class GeminiCooldown(RuntimeError):
    def __init__(self, model, until, reason):
        self.until = until
        self.reason = reason
        super().__init__(f'{model}: {reason}')


def day_window(now):
    local = datetime.fromtimestamp(now, PACIFIC)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return local.date().isoformat(), midnight.timestamp()


def estimate_input_tokens(payload):
    # UTF-8 bytes provide a conservative bound for text tokenization, including
    # schema/system-message overhead. Reconcile to provider input usage afterwards.
    return len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) + 256


def _availability(conn, model, tokens, now):
    day, midnight = day_window(now)
    state = conn.execute('SELECT * FROM gemini_quota_state WHERE model=?', (model,)).fetchone()
    requests = state['requests'] if state and state['day'] == day else 0
    recent = conn.execute('SELECT started,input_tokens FROM gemini_quota_requests WHERE model=? AND started>? ORDER BY started',
                          (model, now - 60)).fetchall()
    blockers = []
    if state and state['cooldown_until'] > now:
        blockers.append((state['cooldown_until'], state['reason']))
    limits = LIMITS.get(model)
    if limits:
        rpm, tpm, rpd = limits
        if requests >= rpd:
            blockers.append((midnight, 'daily request quota exhausted; resets at midnight Pacific'))
        if len(recent) >= rpm:
            blockers.append((recent[len(recent) - rpm]['started'] + 60, 'minute request quota exhausted'))
        remaining = sum(row['input_tokens'] for row in recent)
        if remaining + tokens > tpm:
            for row in recent:
                remaining -= row['input_tokens']
                if remaining + tokens <= tpm:
                    blockers.append((row['started'] + 60, 'minute input-token quota exhausted'))
                    break
    until, reason = max(blockers, default=(0, 'eligible'))
    return {'model': model, 'day_requests': requests, 'minute_requests': len(recent),
            'minute_tokens': sum(row['input_tokens'] for row in recent),
            'limits': limits, 'until': until, 'reason': reason}


def reserve(conn, model, tokens):
    """Caller must hold BEGIN IMMEDIATE; denied reservations consume no budget."""
    now = time.time()
    if model in LIMITS and tokens > LIMITS[model][1]:
        raise ValueError(f'{model}: request exceeds the free-tier input-token minute allowance')
    state = _availability(conn, model, tokens, now)
    if state['until'] > now:
        raise GeminiCooldown(model, state['until'], state['reason'])
    day, _ = day_window(now)
    conn.execute('DELETE FROM gemini_quota_requests WHERE started<=?', (now - 60,))
    conn.execute('''INSERT INTO gemini_quota_state(model,day,requests) VALUES(?,?,1)
        ON CONFLICT(model) DO UPDATE SET day=excluded.day,
        requests=CASE WHEN day=excluded.day THEN requests+1 ELSE 1 END''', (model, day))
    return conn.execute('INSERT INTO gemini_quota_requests(model,started,input_tokens) VALUES(?,?,?)',
                        (model, now, tokens)).lastrowid


def reconcile(request_id, input_tokens):
    if input_tokens is not None:
        with get_db_connection() as conn:
            conn.execute('UPDATE gemini_quota_requests SET input_tokens=? WHERE id=?',
                         (max(0, int(input_tokens)), request_id))
            conn.commit()


def error_cooldown(error, now=None):
    """QuotaFailure identifiers beat generic messages; overload is not daily quota."""
    now = time.time() if now is None else now
    response = getattr(error, 'response', None)
    status = getattr(error, 'status_code', None) or getattr(response, 'status_code', None)
    body = getattr(error, 'body', None)
    if body is None and response is not None:
        try:
            body = response.json()
        except (ValueError, AttributeError):
            pass
    text = (str(error) + ' ' + json.dumps(body, default=str)).lower()
    timing = []
    headers = getattr(response, 'headers', {}) or {}
    value = headers.get('retry-after')
    if value:
        try:
            timing.append(float(value))
        except (ValueError, TypeError):
            try:
                timing.append(parsedate_to_datetime(value).timestamp() - now)
            except (ValueError, TypeError, OverflowError):
                pass

    def retry_delays(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'retryDelay':
                    try:
                        timing.append(float(item.rstrip('s')) if isinstance(item, str)
                                      else float(item.get('seconds', 0)) + float(item.get('nanos', 0)) / 1e9)
                    except (ValueError, TypeError, AttributeError):
                        pass
                else:
                    retry_delays(item)
        elif isinstance(value, list):
            for item in value:
                retry_delays(item)
    retry_delays(body)
    match = re.search(r'retry in\s+([\d.]+)s', text)
    if match:
        timing.append(float(match.group(1)))
    provider_until = now + max([0, *timing])
    daily = any(marker in text for marker in ('perday', 'per_day', 'per day', 'daily'))
    minute = any(marker in text for marker in ('perminute', 'per_minute', 'per minute'))
    quota = any(marker in text for marker in ('quotafailure', 'quota exceeded', 'quota exhausted', 'rate limit', 'resource_exhausted'))
    if daily and (quota or status == 429):
        return max(day_window(now)[1], provider_until), 'provider daily quota exhausted; resets at midnight Pacific'
    if status in (500, 502, 503, 504) and not minute and 'quotafailure' not in text:
        return max(now + 30, provider_until), 'provider temporarily overloaded'
    if minute or status == 429 or quota:
        # Explicit overload messages without quota violations are temporary capacity.
        if not minute and 'quotafailure' not in text and any(s in text for s in ('overload', 'capacity', 'unavailable')):
            return max(now + 30, provider_until), 'provider temporarily overloaded'
        return max(now + 60, provider_until), 'provider minute quota/rate limit'
    if status in (500, 502, 503, 504) or any(s in text for s in ('overload', 'temporarily unavailable')):
        return max(now + 30, provider_until), 'provider temporarily overloaded'
    return None


def record_error(model, error):
    cooldown = error_cooldown(error)
    if cooldown is None:
        return None
    until, reason = cooldown
    day, _ = day_window(time.time())
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('''INSERT INTO gemini_quota_state(model,day,cooldown_until,reason) VALUES(?,?,?,?)
            ON CONFLICT(model) DO UPDATE SET
            reason=CASE WHEN excluded.cooldown_until>cooldown_until THEN excluded.reason ELSE reason END,
            cooldown_until=MAX(cooldown_until,excluded.cooldown_until)''', (model, day, until, reason))
        state = conn.execute('SELECT cooldown_until,reason FROM gemini_quota_state WHERE model=?', (model,)).fetchone()
        conn.commit()
    return GeminiCooldown(model, state['cooldown_until'], state['reason'])


def usage(models):
    now = time.time()
    with get_db_connection() as conn:
        rows = [_availability(conn, model, 1, now) for model in dict.fromkeys(models)]
    for row in rows:
        row['retry_at'] = datetime.fromtimestamp(row['until'], timezone.utc) if row['until'] else None
    return rows
