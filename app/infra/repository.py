import sqlite3
from typing import List, Optional
from datetime import datetime
from app.infra.database import get_db_connection
from app.core.models import SubscriptionCreate, Subscription, Episode

class SubscriptionRepository:
    def create(self, sub: SubscriptionCreate, title: str, slug: str, image_url: str = None, description: str = None, retention_limit: int = 1) -> Subscription:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO subscriptions (feed_url, title, slug, image_url, description, retention_limit) VALUES (?, ?, ?, ?, ?, ?)",
                    (sub.feed_url, title, slug, image_url, description, retention_limit)
                )
                sub_id = cursor.lastrowid
                conn.commit()
                return self.get_by_id(sub_id)
            except sqlite3.IntegrityError:
                raise ValueError("Subscription already exists")

    def get_by_id(self, id: int) -> Optional[Subscription]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (id,)).fetchone()
            if row:
                return Subscription.model_validate(dict(row))
            return None

    def get_all(self) -> List[Subscription]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM subscriptions").fetchall()
            return [Subscription.model_validate(dict(row)) for row in rows]

    def get_by_url(self, url: str) -> Optional[Subscription]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM subscriptions WHERE feed_url = ?", (url,)).fetchone()
            if row:
                return Subscription.model_validate(dict(row))
            return None

    def get_by_slug(self, slug: str) -> Optional[Subscription]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM subscriptions WHERE slug = ?", (slug,)).fetchone()
            if row:
                return Subscription.model_validate(dict(row))
            return None

    def delete(self, id: int):
        with get_db_connection() as conn:
            conn.execute("DELETE FROM episodes WHERE subscription_id = ?", (id,))
            conn.execute("DELETE FROM subscriptions WHERE id = ?", (id,))
            conn.commit()

    def update_settings(self, id: int, remove_ads: bool, remove_promos: bool, remove_intros: bool, remove_outros: bool, custom_instructions: str, append_summary: bool, append_title_intro: bool, ai_rewrite_description: bool, ai_audio_summary: bool, retention_days: int = 30, manual_retention_days: int = 14, retention_limit: int = 1):
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE subscriptions 
                SET remove_ads = ?, 
                    remove_promos = ?, 
                    remove_intros = ?, 
                    remove_outros = ?, 
                    custom_instructions = ?,
                    append_summary = ?,
                    append_title_intro = ?,
                    ai_rewrite_description = ?,
                    ai_audio_summary = ?,
                    retention_days = ?,
                    manual_retention_days = ?,
                    retention_limit = ?
                WHERE id = ?
            """, (remove_ads, remove_promos, remove_intros, remove_outros, custom_instructions, append_summary, append_title_intro, ai_rewrite_description, ai_audio_summary, retention_days, manual_retention_days, retention_limit, id))
            conn.commit()

class EpisodeRepository:
    def create_or_ignore(self, episode: dict) -> bool:
        """Returns True if created, False if already exists."""
        with get_db_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO episodes (subscription_id, guid, title, pub_date, original_url, duration, description, status, file_size)
                    VALUES (:subscription_id, :guid, :title, :pub_date, :original_url, :duration, :description, :status, :file_size)
                """, episode)
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_pending(self) -> List[dict]:
        with get_db_connection() as conn:
            # Get pending episodes OR failed/rate_limited episodes that are due for retry
            rows = conn.execute("""
                SELECT * FROM episodes 
                WHERE status = 'pending' 
                OR (status = 'failed' AND next_retry_at IS NOT NULL AND next_retry_at <= CURRENT_TIMESTAMP)
                OR (status = 'rate_limited' AND next_retry_at IS NOT NULL AND next_retry_at <= CURRENT_TIMESTAMP)
            """).fetchall()
            return [dict(row) for row in rows]
            
    def get_queue(self) -> List[dict]:
        with get_db_connection() as conn:
            # Get full processing queue with details (including rate_limited)
            rows = conn.execute("""
                SELECT e.*, s.title as podcast_title 
                FROM episodes e
                JOIN subscriptions s ON e.subscription_id = s.id
                WHERE e.status IN ('processing', 'pending', 'rate_limited')
                OR (e.status = 'failed' AND e.next_retry_at IS NOT NULL)
                ORDER BY 
                    CASE e.status 
                        WHEN 'processing' THEN 1 
                        WHEN 'pending' THEN 2 
                        WHEN 'rate_limited' THEN 3
                        ELSE 4 
                    END,
                    e.id ASC
            """).fetchall()
            return [dict(row) for row in rows]

    def get_recently_processed(self, days: int = 3) -> List[dict]:
        """Get episodes completed or failed in the last N days for audit trail."""
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT e.*, s.title as podcast_title 
                FROM episodes e
                JOIN subscriptions s ON e.subscription_id = s.id
                WHERE e.status IN ('completed', 'failed', 'ignored')
                AND e.processed_at IS NOT NULL
                AND e.processed_at >= datetime('now', ?)
                ORDER BY e.processed_at DESC
                LIMIT 50
            """, (f'-{days} days',)).fetchall()
            return [dict(row) for row in rows]

    def get_by_id(self, id: int) -> Optional[Episode]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE id = ?", (id,)).fetchone()
            if row:
                return Episode.model_validate(dict(row))
            return None

    def get_by_subscription(self, subscription_id: int) -> List[Episode]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM episodes WHERE subscription_id = ?", (subscription_id,)).fetchall()
            return [Episode.model_validate(dict(row)) for row in rows]

    def get_by_subscription_paginated(self, subscription_id: int, limit: int = 20, offset: int = 0, search: str = None) -> list:
        """Get episodes for a subscription with pagination, ordered by pub_date descending.
        Optionally filter by search term (matches title)."""
        with get_db_connection() as conn:
            if search:
                return conn.execute(
                    "SELECT * FROM episodes WHERE subscription_id = ? AND title LIKE ? ORDER BY pub_date DESC LIMIT ? OFFSET ?",
                    (subscription_id, f"%{search}%", limit, offset)
                ).fetchall()
            return conn.execute(
                "SELECT * FROM episodes WHERE subscription_id = ? ORDER BY pub_date DESC LIMIT ? OFFSET ?",
                (subscription_id, limit, offset)
            ).fetchall()

    def count_by_subscription(self, subscription_id: int, search: str = None) -> int:
        """Count total episodes for a subscription, optionally filtered by search term."""
        with get_db_connection() as conn:
            if search:
                result = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE subscription_id = ? AND title LIKE ?",
                    (subscription_id, f"%{search}%")
                ).fetchone()
            else:
                result = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE subscription_id = ?",
                    (subscription_id,)
                ).fetchone()
            return result[0] if result else 0

    def get_status(self, id: int) -> Optional[str]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT status FROM episodes WHERE id = ?", (id,)).fetchone()
            if row:
                return row['status']
            return None

    def reset_status(self, id: int, processing_flags: str = None):
        """Reset episode status to unprocessed."""
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE episodes 
                SET status = 'unprocessed', 
                    processing_step = NULL, 
                    progress = 0, 
                    error_message = NULL,
                    retry_count = 0,
                    next_retry_at = NULL,
                    processing_flags = ?,
                    ai_summary = NULL
                WHERE id = ?
            """, (processing_flags, id))
            conn.commit()

    def requeue_stuck(self):
        """Reset all 'processing' episodes to 'failed' on startup."""
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE episodes 
                SET status = 'failed', 
                    error_message = 'Interrupted by system restart',
                    processing_step = 'interrupted', 
                    progress = 0,
                    next_retry_at = NULL
                WHERE status = 'processing'
            """)
            conn.commit()

    def update_retry(self, id: int, retry_count: int, next_retry_at: datetime, error: str):
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE episodes 
                SET status = 'failed', 
                    retry_count = ?, 
                    next_retry_at = ?, 
                    error_message = ? 
                WHERE id = ?
            """, (retry_count, next_retry_at, error, id))
            conn.commit()

    def update_rate_limited(self, id: int, next_retry_at: datetime, error: str):
        """Set episode to rate_limited status with scheduled retry at API quota reset."""
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE episodes 
                SET status = 'rate_limited', 
                    next_retry_at = ?, 
                    error_message = ?,
                    processing_step = 'Waiting for API quota reset'
                WHERE id = ?
            """, (next_retry_at, error, id))
            conn.commit()

    def update_status(self, id: int, status: str, error: str = None, filename: str = None, file_size: int = None):
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE episodes SET status = ?, error_message = ?, local_filename = ?, file_size = ?, processed_at = ?, next_retry_at = NULL WHERE id = ?",
                (status, error, filename, file_size, datetime.now() if status == 'completed' else None, id)
            )
            conn.commit()

    def update_progress(self, id: int, step: str, progress: int, transcript_path: str = None, ad_report_path: str = None, report_path: str = None):
        with get_db_connection() as conn:
            updates = ["processing_step = ?", "progress = ?"]
            params = [step, progress]
            
            if transcript_path:
                updates.append("transcript_path = ?")
                params.append(transcript_path)
            
            if ad_report_path:
                updates.append("ad_report_path = ?")
                params.append(ad_report_path)

            if report_path:
                updates.append("report_path = ?")
                params.append(report_path)
                
            params.append(id)
            
            sql = f"UPDATE episodes SET {', '.join(updates)} WHERE id = ?"
            conn.execute(sql, params)
            conn.commit()

    def update_description(self, id: int, description: str):
        with get_db_connection() as conn:
            conn.execute("UPDATE episodes SET description = ? WHERE id = ?", (description, id))
            conn.commit()

    def update_ai_summary(self, id: int, summary: str):
        with get_db_connection() as conn:
            conn.execute("UPDATE episodes SET ai_summary = ? WHERE id = ?", (summary, id))
            conn.commit()

    def update_status_by_guid(self, subscription_id: int, guid: str, status: str, condition_status: str = None):
        """Update status of an episode by GUID, optionally only if current status matches condition."""
        with get_db_connection() as conn:
            if condition_status:
                conn.execute("""
                    UPDATE episodes 
                    SET status = ? 
                    WHERE subscription_id = ? AND guid = ? AND (status = ? or status = 'pending_manual')
                """, (status, subscription_id, guid, condition_status))
            else:
                conn.execute("""
                    UPDATE episodes 
                    SET status = ? 
                    WHERE subscription_id = ? AND guid = ?
                """, (status, subscription_id, guid))
            conn.commit()

    def delete(self, id: int):
        with get_db_connection() as conn:
            conn.execute("DELETE FROM episodes WHERE id = ?", (id,))
            conn.commit()
    
    def increment_listen_count(self, id: int):
        """Increment the listen count for an episode."""
        with get_db_connection() as conn:
            conn.execute("UPDATE episodes SET listen_count = listen_count + 1 WHERE id = ?", (id,))
            conn.commit()
    
    def get_subscription_listen_count(self, subscription_id: int) -> int:
        """Get total listen count for all episodes in a subscription."""
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT SUM(listen_count) as total FROM episodes WHERE subscription_id = ? AND status = 'completed'",
                (subscription_id,)
            ).fetchone()
            return row['total'] if row and row['total'] else 0
    
    def get_by_subscription_and_filename(self, subscription_id: int, filename: str) -> Optional[Episode]:
        """Find episode by subscription and local filename (for audio tracking)."""
        with get_db_connection() as conn:
            # Try exact match first
            row = conn.execute(
                "SELECT * FROM episodes WHERE subscription_id = ? AND local_filename LIKE ?",
                (subscription_id, f"%{filename}")
            ).fetchone()
            if row:
                return Episode.model_validate(dict(row))
            return None

    def count_processing(self) -> int:
        """Count episodes currently in 'processing' status. Used for concurrent limit enforcement."""
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM episodes WHERE status = 'processing'").fetchone()
            return row['count'] if row else 0

    def soft_delete(self, id: int):
        """Mark episode as ignored and clear paths to free space."""
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE episodes 
                SET status = 'ignored',
                    local_filename = NULL,
                    transcript_path = NULL,
                    ad_report_path = NULL,
                    report_path = NULL,
                    progress = 0,
                    processing_step = NULL
                WHERE id = ?
            """, (id,))
            conn.commit()
