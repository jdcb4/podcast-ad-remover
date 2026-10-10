"""Durable, unique-episode processing totals and current library holdings."""
from app.infra.database import get_db_connection


def record_completion(conn, episode_id, transcription_seconds=None):
    # Reprocessing updates measurements without counting the same episode twice or
    # moving its first completion into a later reporting period.
    conn.execute("""INSERT INTO processing_history
        (episode_id,subscription_id,processed_at,source_seconds,output_seconds,transcription_seconds)
        SELECT id,subscription_id,processed_at,duration,output_duration,? FROM episodes WHERE id=?
        ON CONFLICT(episode_id) DO UPDATE SET source_seconds=excluded.source_seconds,
        output_seconds=excluded.output_seconds,
        transcription_seconds=CASE WHEN excluded.transcription_seconds IS NULL THEN processing_history.transcription_seconds ELSE COALESCE(processing_history.transcription_seconds,0)+excluded.transcription_seconds END
        """, (transcription_seconds, episode_id))
    conn.execute("""INSERT OR IGNORE INTO processing_history_users
        SELECT ?,user_id FROM user_subscriptions WHERE subscription_id=(
            SELECT subscription_id FROM processing_history WHERE episode_id=?)
        AND datetime(added_at)<=datetime((SELECT processed_at FROM processing_history WHERE episode_id=?))
        """, (episode_id, episode_id, episode_id))


def get_statistics(*, user_id=None, period="all", historical=False):
    periods = {"all": "1=1", "week": "datetime({date})>=datetime('now','weekday 0','-6 days','start of day')",
               "month": "datetime({date})>=datetime('now','start of month')",
               "year": "datetime({date})>=datetime('now','start of year')"}
    date_filter = periods[period]
    with get_db_connection() as conn:
        params = () if user_id is None else (user_id,)
        if historical:
            membership = "" if user_id is None else "AND EXISTS (SELECT 1 FROM processing_history_users u WHERE u.episode_id=h.episode_id AND u.user_id=?)"
            row = conn.execute(f"""SELECT COUNT(*) episodes, COUNT(DISTINCT subscription_id) podcasts,
                COALESCE(SUM(COALESCE(output_seconds,source_seconds)),0) seconds,
                COALESCE(SUM(CASE WHEN source_seconds IS NOT NULL AND output_seconds IS NOT NULL THEN MAX(0,source_seconds-output_seconds) ELSE 0 END),0) saved,
                COALESCE(SUM(transcription_seconds),0) transcription_seconds,
                COUNT(transcription_seconds) transcription_count,
                SUM(output_seconds IS NULL OR source_seconds IS NULL) unknown_savings
                FROM processing_history h WHERE {date_filter.format(date='h.processed_at')} {membership}""", params).fetchone()
            result = dict(row)
        else:
            membership = "" if user_id is None else "AND EXISTS (SELECT 1 FROM user_subscriptions u WHERE u.subscription_id=s.id AND u.user_id=?)"
            row = conn.execute(f"""SELECT COUNT(e.id) episodes,
                COALESCE(SUM(COALESCE(e.output_duration,e.duration)),0) seconds,
                COALESCE(SUM(CASE WHEN e.duration IS NOT NULL AND e.output_duration IS NOT NULL THEN MAX(0,e.duration-e.output_duration) ELSE 0 END),0) saved,
                SUM(e.id IS NOT NULL AND (e.output_duration IS NULL OR e.duration IS NULL)) unknown_savings
                FROM subscriptions s LEFT JOIN episodes e ON e.subscription_id=s.id
                AND e.local_filename IS NOT NULL AND e.status!='ignored'
                WHERE s.deletion_status IS NULL {membership}""", params).fetchone()
            result = dict(row)
            result['podcasts'] = conn.execute(f"SELECT COUNT(*) FROM subscriptions s WHERE s.deletion_status IS NULL {membership}", params).fetchone()[0]
        if historical and user_id is None:
            usage = conn.execute(f"""SELECT COUNT(*) calls, COUNT(input_tokens) known_input,
                COUNT(output_tokens) known_output, COALESCE(SUM(input_tokens),0) input_tokens,
                COALESCE(SUM(output_tokens),0) output_tokens FROM provider_calls
                WHERE {date_filter.format(date='started_at')}""").fetchone()
            result['usage'] = dict(usage)
        result['hours'] = round(result['seconds']/3600, 1)
        result['saved_hours'] = round(result['saved']/3600, 1)
        result['transcription_hours'] = round(result.get('transcription_seconds',0)/3600, 1)
        return result
