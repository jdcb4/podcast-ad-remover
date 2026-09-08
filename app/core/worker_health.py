"""Minimal persisted worker liveness and real scheduler timestamps (UTC)."""
from datetime import datetime, timedelta

from app.core.config import settings
from app.infra.database import get_db_connection


def record_heartbeat(worker_id: str):
    with get_db_connection() as conn:
        conn.execute("""INSERT INTO worker_status(id,worker_id,heartbeat_at) VALUES(1,?,CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET worker_id=excluded.worker_id, heartbeat_at=excluded.heartbeat_at""", (worker_id,))
        conn.commit()


def record_feed_check(interval_minutes: int):
    now = datetime.utcnow()
    with get_db_connection() as conn:
        conn.execute('UPDATE worker_status SET last_feed_check=?, next_feed_check=? WHERE id=1',
                     (now.isoformat(sep=' ', timespec='seconds'), (now + timedelta(minutes=interval_minutes)).isoformat(sep=' ', timespec='seconds')))
        conn.commit()


def worker_status():
    if not settings.PROCESSOR_ENABLED:
        return {'state': 'disabled', 'label': 'Automatic processing disabled; manual actions available', 'next_feed_check': None}
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM worker_status WHERE id=1').fetchone()
    if not row or not row['heartbeat_at']:
        return {'state': 'starting', 'label': 'Waiting for processor heartbeat', 'next_feed_check': None}
    result = dict(row)
    age = (datetime.utcnow() - datetime.fromisoformat(row['heartbeat_at'])).total_seconds()
    result.update(state='healthy' if age < 90 else 'stale', label='Processor available' if age < 90 else 'Processor heartbeat is stale')
    return result
