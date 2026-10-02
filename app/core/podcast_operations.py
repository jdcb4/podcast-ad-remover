"""Reviewed RSS replacement and restart-safe archive batches."""
import hashlib
import json
import time
from uuid import uuid4
from urllib.parse import urljoin

import feedparser

from app.core.feed import FeedManager
from app.core.feed_urls import feed_key
from app.infra.database import get_db_connection

DISCOVERY_SECONDS = 120


def discover_history(url):
    entries, seen_pages, seen_guids = [], set(), set()
    size = 0
    started = time.monotonic()
    for _ in range(50):
        if time.monotonic() - started > DISCOVERY_SECONDS:
            raise ValueError('Discovery time limit reached; history is incomplete.')
        if url in seen_pages:
            raise ValueError('Feed pagination loops; history discovery is incomplete.')
        seen_pages.add(url)
        raw = FeedManager._fetch_feed(url)
        size += len(raw)
        if size > 50 * 1024 * 1024:
            raise ValueError('History exceeds the 50 MiB discovery limit; nothing was queued.')
        parsed = feedparser.parse(raw)
        if parsed.bozo or not parsed.feed.get('title'):
            raise ValueError('The source did not return a valid podcast feed.')
        for item in FeedManager.episodes_from_parsed(parsed):
            if item['guid'] not in seen_guids:
                seen_guids.add(item['guid'])
                item['pub_date'] = item['pub_date'].isoformat() if item['pub_date'] else None
                entries.append(item)
        if len(entries) > 10000:
            raise ValueError('History exceeds 10,000 episodes; nothing was queued.')
        links = [x['href'] for x in parsed.feed.get('links', []) if x.get('rel') == 'next' and x.get('href')]
        if not links:
            return entries
        if len(set(links)) != 1:
            raise ValueError('Ambiguous feed pagination; nothing was queued.')
        url = urljoin(url, links[0])
    raise ValueError('History exceeds 50 pages; nothing was queued.')


def _subscription(conn, sub_id):
    row = conn.execute('SELECT * FROM subscriptions WHERE id=? AND deletion_status IS NULL', (sub_id,)).fetchone()
    if not row or row['source_type'] != 'rss' or not row['is_active']:
        raise ValueError('This action requires an active RSS podcast.')
    return dict(row)


def _fingerprint(conn, sub_id):
    sub = _subscription(conn, sub_id)
    for key in ('last_checked', 'last_check_error', 'last_check_at', 'last_successful_check', 'last_check_truncated'):
        sub.pop(key, None)
    data = [sub,
            [tuple(r) for r in conn.execute('SELECT id,guid,original_url,status FROM episodes WHERE subscription_id=? ORDER BY id', (sub_id,))],
            [tuple(r) for r in conn.execute('SELECT j.id,j.status FROM jobs j JOIN episodes e ON e.id=j.episode_id WHERE e.subscription_id=? ORDER BY j.id', (sub_id,))]]
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _match(conn, sub_id, item):
    rows = conn.execute('''SELECT e.* FROM episodes e WHERE e.subscription_id=? AND
        (e.guid=? OR e.id IN (SELECT episode_id FROM episode_source_aliases WHERE subscription_id=? AND guid=?))''',
        (sub_id, item['guid'], sub_id, item['guid'])).fetchall()
    if len(rows) > 1:
        raise ValueError('Conflicting source identifiers require correction before changing this feed.')
    if len(rows) == 1:
        return dict(rows[0])
    rows = conn.execute('SELECT * FROM episodes WHERE subscription_id=? AND original_url=?', (sub_id, item['original_url'])).fetchall()
    return dict(rows[0]) if len(rows) == 1 else None


def reconcile_batches(conn):
    conn.execute("""UPDATE archive_batches SET status='completed' WHERE status IN ('active','paused')
        AND NOT EXISTS (SELECT 1 FROM jobs WHERE archive_batch_id=archive_batches.id
        AND status IN ('queued','running','retry_scheduled','rate_limited'))""")


def preview(sub_id, kind, url=None):
    if kind not in ('archive', 'feed'):
        raise ValueError('Unknown podcast operation.')
    with get_db_connection() as conn:
        sub = _subscription(conn, sub_id)
        fingerprint = _fingerprint(conn, sub_id)
    if kind == 'feed' and not url:
        raise ValueError('Enter a replacement feed URL.')
    url = feed_key(url) if kind == 'feed' else sub['feed_url']
    if kind == 'feed' and url == sub['feed_url']:
        raise ValueError('Choose a different feed URL.')
    try:
        entries = discover_history(url)
    except Exception as exc:
        # Network exceptions can include credential-bearing source URLs.
        raise ValueError('Could not discover complete available RSS history. Check the URL, feed validity and discovery limits (50 pages / 10,000 episodes / 50 MiB / 120 seconds).') from exc
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if fingerprint != _fingerprint(conn, sub_id):
            raise ValueError('Podcast changed during discovery. Preview again.')
        reconcile_batches(conn)
        _available(conn, sub_id, kind, url)
        result, used = [], set()
        existing = [dict(r) for r in conn.execute('SELECT id,title,pub_date FROM episodes WHERE subscription_id=? ORDER BY pub_date DESC,id', (sub_id,))] if kind == 'feed' else []
        for item in entries:
            old = _match(conn, sub_id, item)
            if old and old['id'] in used:
                raise ValueError('Multiple feed entries match one episode. Resolve the feed identities first.')
            if old:
                used.add(old['id'])
            active = bool(old and conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status IN ('queued','running','retry_scheduled','rate_limited')", (old['id'],)).fetchone())
            status = ('completed' if old and old['status'] == 'completed' else 'active' if active else
                      'excluded' if old and old['status'] in ('ignored', 'deleted', 'processing') else 'eligible')
            candidates = []
            if kind == 'feed' and not old:
                candidates = [r['id'] for r in existing if r['title'].casefold().strip() == item['title'].casefold().strip() or (r['pub_date'] and item['pub_date'] and r['pub_date'][:10] == item['pub_date'][:10])][:20]
            result.append({'entry': item, 'episode_id': old['id'] if old else None, 'status': status, 'candidates': candidates})
        token = uuid4().hex
        payload = {'url': url, 'items': result}
        conn.execute('DELETE FROM podcast_previews WHERE expires_at<?', (int(time.time()),))
        conn.execute('INSERT INTO podcast_previews VALUES(?,?,?,?,?,?)',
                     (token, sub_id, kind, fingerprint, json.dumps(payload), int(time.time()) + 1800))
        conn.commit()
    return {'preview_id': token, 'kind': kind, 'items': result,
            'counts': {s: sum(x['status'] == s for x in result) for s in ('completed','active','excluded','eligible')}, 'total': len(result), 'existing_episodes': existing}


def _available(conn, sub_id, kind, url):
    if conn.execute("SELECT 1 FROM archive_batches WHERE subscription_id=? AND status IN ('active','paused')", (sub_id,)).fetchone():
        raise ValueError('Finish or cancel the existing archive batch first.')
    if kind == 'feed':
        if conn.execute("SELECT 1 FROM jobs j JOIN episodes e ON e.id=j.episode_id WHERE e.subscription_id=? AND j.status IN ('queued','running','retry_scheduled','rate_limited')", (sub_id,)).fetchone():
            raise ValueError('Finish or cancel this podcast’s queued/running work before changing its source.')
        for row in conn.execute('SELECT id,feed_url FROM subscriptions WHERE id!=?', (sub_id,)):
            if feed_key(row['feed_url']) == url:
                raise ValueError('Another podcast already uses this feed URL.')


def confirm(sub_id, token, choices=None):
    from app.infra.repository import _enqueue_job
    choices = choices or {}
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT * FROM podcast_previews WHERE id=? AND subscription_id=?', (token, sub_id)).fetchone()
        if not row or row['expires_at'] < time.time() or row['fingerprint'] != _fingerprint(conn, sub_id):
            raise ValueError('Preview expired or the podcast changed. Preview again.')
        payload, kind = json.loads(row['payload']), row['kind']
        reconcile_batches(conn)
        _available(conn, sub_id, kind, payload['url'])
        items = payload['items']
        batch_id = None
        if kind == 'archive':
            if not any(x['status'] == 'eligible' for x in items):
                raise ValueError('There are no eligible episodes to process.')
            batch_id = conn.execute("INSERT INTO archive_batches(subscription_id,status) VALUES(?,'active')", (sub_id,)).lastrowid
            # Freeze the effective discovery window before disabling inheritance.
            conn.execute('''UPDATE subscriptions SET
                retention_limit=CASE WHEN inherit_retention THEN (SELECT default_retention_limit FROM app_settings WHERE id=1) ELSE retention_limit END,
                retention_days=CASE WHEN inherit_retention THEN (SELECT default_retention_days FROM app_settings WHERE id=1) ELSE retention_days END,
                manual_retention_days=CASE WHEN inherit_retention THEN (SELECT default_manual_retention_days FROM app_settings WHERE id=1) ELSE manual_retention_days END,
                keep_whole_show=1,inherit_retention=0 WHERE id=?''', (sub_id,))
        items.sort(key=lambda x: (x['entry']['pub_date'] is None, x['entry']['pub_date'] or '', x['entry']['guid']))
        used = set()
        for position, item in enumerate(items):
            data, episode_id = item['entry'], item['episode_id']
            if kind == 'archive' and item['status'] != 'eligible':
                continue
            if kind == 'feed' and episode_id is None:
                if data['guid'] not in choices:
                    raise ValueError('Review every unmatched entry before changing the feed.')
                episode_id = choices[data['guid']]
                if episode_id is not None and not conn.execute('SELECT 1 FROM episodes WHERE id=? AND subscription_id=?', (episode_id, sub_id)).fetchone():
                    raise ValueError('Selected episode does not belong to this podcast.')
            if episode_id is not None and episode_id in used:
                raise ValueError('Two incoming entries cannot map to one episode.')
            if episode_id is None:
                episode_id = conn.execute('''INSERT INTO episodes(subscription_id,guid,title,pub_date,original_url,duration,description,status,file_size,discovered_at)
                    VALUES(?,?,?,?,?,?,?,'unprocessed',?,CURRENT_TIMESTAMP)''',
                    (sub_id, data['guid'], data['title'], data['pub_date'], data['original_url'], data['duration'], data['description'], data['file_size'])).lastrowid
            used.add(episode_id)
            if kind == 'feed':
                conn.execute('INSERT INTO episode_source_aliases VALUES(?,?,?) ON CONFLICT(subscription_id,guid) DO UPDATE SET episode_id=excluded.episode_id', (sub_id, data['guid'], episode_id))
                conn.execute('UPDATE episodes SET original_url=? WHERE id=?', (data['original_url'], episode_id))
            else:
                conn.execute("UPDATE episodes SET status='pending',next_retry_at=NULL,retry_count=0 WHERE id=?", (episode_id,))
                job_id = _enqueue_job(conn, episode_id, priority=200)
                conn.execute('UPDATE jobs SET archive_batch_id=?,archive_position=? WHERE id=?', (batch_id, position, job_id))
        if kind == 'feed':
            conn.execute('UPDATE subscriptions SET feed_url=?,last_check_error=NULL WHERE id=?', (payload['url'], sub_id))
        conn.execute('DELETE FROM podcast_previews WHERE id=?', (token,))
        conn.commit()
    return {'status': 'ok', 'batch_id': batch_id}


def batches(sub_id):
    with get_db_connection() as conn:
        reconcile_batches(conn)
        rows = [dict(r) for r in conn.execute('SELECT * FROM archive_batches WHERE subscription_id=? ORDER BY id DESC LIMIT 10', (sub_id,))]
        for row in rows:
            row['counts'] = dict(conn.execute('SELECT status,count(*) FROM jobs WHERE archive_batch_id=? GROUP BY status', (row['id'],)).fetchall())
        conn.commit()
        return rows


def control(sub_id, batch_id, action):
    if action not in ('pause','resume','cancel'):
        raise ValueError('Unknown batch action.')
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT * FROM archive_batches WHERE id=? AND subscription_id=?', (batch_id, sub_id)).fetchone()
        if not row or row['status'] not in ('active','paused'):
            raise ValueError('This batch is no longer active.')
        conn.execute('UPDATE archive_batches SET status=? WHERE id=?', ({'pause':'paused','resume':'active','cancel':'cancelled'}[action], batch_id))
        if action == 'cancel':
            conn.execute("UPDATE episodes SET status='unprocessed',next_retry_at=NULL WHERE id IN (SELECT episode_id FROM jobs WHERE archive_batch_id=? AND status IN ('queued','retry_scheduled','rate_limited'))", (batch_id,))
            conn.execute("UPDATE jobs SET status='cancelled',updated_at=CURRENT_TIMESTAMP WHERE archive_batch_id=? AND status IN ('queued','retry_scheduled','rate_limited')", (batch_id,))
        conn.commit()
    return {'status': 'ok'}
