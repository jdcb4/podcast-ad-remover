import sqlite3
from typing import List, Optional
from datetime import datetime
from app.infra.database import get_db_connection
from app.core.models import SubscriptionCreate, Subscription, Episode

class SubscriptionRepository:
    def create(self, sub: SubscriptionCreate, title: str, slug: str, image_url: str = None) -> Subscription:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO subscriptions (feed_url, title, slug, image_url) VALUES (?, ?, ?, ?)",
                    (sub.feed_url, title, slug, image_url)
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

    def delete(self, id: int):
        with get_db_connection() as conn:
            conn.execute("DELETE FROM episodes WHERE subscription_id = ?", (id,))
            conn.execute("DELETE FROM subscriptions WHERE id = ?", (id,))
            conn.commit()

    def update_settings(self, id: int, remove_ads: bool, remove_promos: bool, remove_intros: bool, remove_outros: bool, custom_instructions: str, append_summary: bool, append_title_intro: bool):
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE subscriptions 
                SET remove_ads = ?, 
                    remove_promos = ?, 
                    remove_intros = ?, 
                    remove_outros = ?, 
                    custom_instructions = ?,
                    append_summary = ?,
                    append_title_intro = ?
                WHERE id = ?
            """, (remove_ads, remove_promos, remove_intros, remove_outros, custom_instructions, append_summary, append_title_intro, id))
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
            # Get pending episodes OR failed episodes that are due for retry
            rows = conn.execute("""
                SELECT * FROM episodes 
                WHERE status = 'pending' 
                OR (status = 'failed' AND next_retry_at IS NOT NULL AND next_retry_at <= CURRENT_TIMESTAMP)
            """).fetchall()
            return [dict(row) for row in rows]
            
    def get_queue(self) -> List[dict]:
        with get_db_connection() as conn:
            # Get full processing queue with details
            rows = conn.execute("""
                SELECT e.*, s.title as podcast_title 
                FROM episodes e
                JOIN subscriptions s ON e.subscription_id = s.id
                WHERE e.status IN ('processing', 'pending')
                OR (e.status = 'failed' AND e.next_retry_at IS NOT NULL)
                ORDER BY 
                    CASE e.status 
                        WHEN 'processing' THEN 1 
                        WHEN 'pending' THEN 2 
                        ELSE 3 
                    END,
                    e.id ASC
            """).fetchall()
            return [dict(row) for row in rows]

    def get_by_id(self, id: int) -> Optional[Episode]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE id = ?", (id,)).fetchone()
            if row:
                return Episode.model_validate(dict(row))
            return None

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
                    processing_flags = ?
                WHERE id = ?
            """, (processing_flags, id))
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

    def update_status(self, id: int, status: str, error: str = None, filename: str = None):
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE episodes SET status = ?, error_message = ?, local_filename = ?, processed_at = ? WHERE id = ?",
                (status, error, filename, datetime.now() if status == 'completed' else None, id)
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
