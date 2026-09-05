import sqlite3
from contextlib import contextmanager
from uuid import uuid4
import socket
import hashlib
import secrets
import time
from typing import List, Optional
from datetime import datetime
from app.infra.database import get_db_connection
from app.core.models import SubscriptionCreate, Subscription, Episode
from app.core.subscription_settings import resolve_subscription_row

class SubscriptionRepository:
    @staticmethod
    def _global_settings(conn) -> dict:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        return dict(row) if row else {}

    def _subscription_from_row(self, row, global_settings: dict) -> Subscription:
        return Subscription.model_validate(resolve_subscription_row(dict(row), global_settings))

    def create(
        self,
        sub: SubscriptionCreate,
        title: str,
        slug: str,
        image_url: str = None,
        description: str = None,
        retention_limit: int = 1,
        owner_user_id: int | None = None,
        inherit_retention: bool = True,
        source_type: str = "rss",
        source_external_id: str | None = None,
    ) -> Subscription:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO subscriptions
                        (feed_url, title, slug, image_url, description, retention_limit, owner_user_id,
                         inherit_content_removal, inherit_retention, source_type, source_external_id,
                         inherit_default_features, inherit_custom_instructions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 1, 1)
                    """,
                    (
                        sub.feed_url,
                        title,
                        slug,
                        image_url,
                        description,
                        retention_limit,
                        owner_user_id,
                        int(inherit_retention),
                        source_type,
                        source_external_id,
                    )
                )
                sub_id = cursor.lastrowid
                if owner_user_id and owner_user_id > 0:
                    cursor.execute(
                        "INSERT OR IGNORE INTO user_subscriptions (user_id, subscription_id) VALUES (?, ?)",
                        (owner_user_id, sub_id),
                    )
                conn.commit()
                return self.get_by_id(sub_id)
            except sqlite3.IntegrityError:
                raise ValueError("Subscription already exists")

    def get_by_id(self, id: int) -> Optional[Subscription]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (id,)).fetchone()
            if row:
                return self._subscription_from_row(row, self._global_settings(conn))
            return None

    def get_all(
        self,
        user_id: int | None = None,
        only_user: bool = False,
        include_deleting: bool = False,
    ) -> List[Subscription]:
        with get_db_connection() as conn:
            deletion_filter = "" if include_deleting else " AND s.deletion_status IS NULL"
            if only_user and user_id and user_id > 0:
                rows = conn.execute(
                    f"""
                    SELECT s.*
                    FROM subscriptions s
                    JOIN user_subscriptions us ON us.subscription_id = s.id
                    WHERE us.user_id = ?
                    {deletion_filter}
                    ORDER BY s.title COLLATE NOCASE
                    """,
                    (user_id,),
                ).fetchall()
            else:
                where_clause = "" if include_deleting else " WHERE deletion_status IS NULL"
                rows = conn.execute(
                    f"SELECT * FROM subscriptions{where_clause} ORDER BY title COLLATE NOCASE"
                ).fetchall()
            global_settings = self._global_settings(conn)
            return [self._subscription_from_row(row, global_settings) for row in rows]

    def add_to_user_library(self, user_id: int | None, subscription_id: int) -> bool:
        if not user_id or user_id <= 0:
            return False
        with get_db_connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO user_subscriptions (user_id, subscription_id) VALUES (?, ?)",
                (user_id, subscription_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def remove_from_user_library(self, user_id: int | None, subscription_id: int) -> bool:
        if not user_id or user_id <= 0:
            return False
        with get_db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM user_subscriptions WHERE user_id = ? AND subscription_id = ?",
                (user_id, subscription_id),
            )
            conn.execute(
                "UPDATE subscriptions SET owner_user_id = NULL WHERE id = ? AND owner_user_id = ?",
                (subscription_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def is_in_user_library(self, user_id: int | None, subscription_id: int) -> bool:
        if not user_id or user_id <= 0:
            return False
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM user_subscriptions WHERE user_id = ? AND subscription_id = ?",
                (user_id, subscription_id),
            ).fetchone()
            return row is not None

    def count_user_library_members(self, subscription_id: int) -> int:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM user_subscriptions WHERE subscription_id = ?",
                (subscription_id,),
            ).fetchone()
            return int(row["count"]) if row else 0

    def get_owner_username(self, subscription_id: int) -> str | None:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT u.username
                FROM subscriptions s
                JOIN users u ON u.id = s.owner_user_id
                WHERE s.id = ?
                """,
                (subscription_id,),
            ).fetchone()
            return row["username"] if row else None

    def set_owner(self, subscription_id: int, owner_user_id: int | None) -> bool:
        with get_db_connection() as conn:
            if owner_user_id is not None:
                user = conn.execute(
                    "SELECT id FROM users WHERE id = ?",
                    (owner_user_id,),
                ).fetchone()
                if not user:
                    return False

            result = conn.execute(
                "UPDATE subscriptions SET owner_user_id = ? WHERE id = ?",
                (owner_user_id, subscription_id),
            )
            if result.rowcount == 0:
                return False

            if owner_user_id is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO user_subscriptions (user_id, subscription_id) VALUES (?, ?)",
                    (owner_user_id, subscription_id),
                )

            conn.commit()
            return True

    def get_by_url(self, url: str) -> Optional[Subscription]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM subscriptions WHERE feed_url = ?", (url,)).fetchone()
            if row:
                return self._subscription_from_row(row, self._global_settings(conn))
            return None

    def get_by_source_identity(self, source_type: str, external_id: str | None) -> Optional[Subscription]:
        if not external_id:
            return None
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM subscriptions WHERE source_type = ? AND source_external_id = ?",
                (source_type, external_id),
            ).fetchone()
            if row:
                return self._subscription_from_row(row, self._global_settings(conn))
            return None

    def available_slug(self, base_slug: str, source_external_id: str | None = None) -> str:
        candidate = base_slug or "podcast"
        with get_db_connection() as conn:
            if not conn.execute("SELECT 1 FROM subscriptions WHERE slug = ?", (candidate,)).fetchone():
                return candidate
            suffix = (source_external_id or "source")[-8:].lower()
            candidate = f"{candidate}-{suffix}"
            counter = 2
            while conn.execute("SELECT 1 FROM subscriptions WHERE slug = ?", (candidate,)).fetchone():
                candidate = f"{base_slug or 'podcast'}-{suffix}-{counter}"
                counter += 1
            return candidate

    def record_check_success(self, subscription_id: int, *, truncated: bool = False) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE subscriptions
                SET last_checked_at = CURRENT_TIMESTAMP,
                    last_check_error = NULL,
                    last_check_error_at = NULL,
                    source_truncated = ?
                WHERE id = ?
                """,
                (int(truncated), subscription_id),
            )
            conn.commit()

    def record_check_error(self, subscription_id: int, error: str) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE subscriptions
                SET last_check_error = ?, last_check_error_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (error[:1000], subscription_id),
            )
            conn.commit()

    def get_by_slug(self, slug: str) -> Optional[Subscription]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM subscriptions WHERE slug = ?", (slug,)).fetchone()
            if row:
                return self._subscription_from_row(row, self._global_settings(conn))
            return None

    def begin_deletion(self, id: int) -> Optional[dict]:
        """Atomically deactivate a subscription and request cancellation of all work."""
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (id,)).fetchone()
            if not row:
                conn.commit()
                return None

            if row["deletion_status"] is None:
                conn.execute(
                    """
                    UPDATE subscriptions
                    SET is_active = 0,
                        deletion_status = 'pending',
                        deletion_started_at = CURRENT_TIMESTAMP,
                        deletion_updated_at = CURRENT_TIMESTAMP,
                        deletion_error = NULL
                    WHERE id = ?
                    """,
                    (id,),
                )
                conn.execute(
                    """
                    UPDATE episodes
                    SET status = 'ignored',
                        local_filename = NULL,
                        transcript_path = NULL,
                        ad_report_path = NULL,
                        report_path = NULL,
                        progress = 0,
                        processing_step = 'cancellation requested',
                        next_retry_at = NULL
                    WHERE subscription_id = ?
                    """,
                    (id,),
                )
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'cancelled',
                        locked_at = NULL,
                        locked_by = NULL,
                        error = 'Cancelled because subscription deletion was requested',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id IN (
                        SELECT id FROM episodes WHERE subscription_id = ?
                    )
                      AND type = 'process_episode'
                      AND status IN ('queued', 'retry_scheduled', 'rate_limited')
                    """,
                    (id,),
                )
                conn.execute(
                    """
                    UPDATE jobs
                    SET error = 'Cancellation requested because subscription is being deleted',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id IN (
                        SELECT id FROM episodes WHERE subscription_id = ?
                    )
                      AND type = 'process_episode'
                      AND status = 'running'
                    """,
                    (id,),
                )

            conn.commit()
            current = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (id,)).fetchone()
            return dict(current) if current else None

    def count_running_deletion_jobs(self, id: int) -> int:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                WHERE e.subscription_id = ?
                  AND j.type = 'process_episode'
                  AND j.status = 'running'
                """,
                (id,),
            ).fetchone()
            return row["count"] if row else 0

    def list_pending_deletions(self) -> List[int]:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id
                FROM subscriptions
                WHERE deletion_status IN ('pending', 'failed')
                   OR (
                       deletion_status = 'cleaning'
                       AND deletion_updated_at <= datetime(CURRENT_TIMESTAMP, '-5 minutes')
                   )
                ORDER BY deletion_started_at, id
                """
            ).fetchall()
            return [row["id"] for row in rows]

    def claim_deletion_cleanup(self, id: int) -> Optional[dict]:
        """Claim retryable physical cleanup after all running workers have acknowledged."""
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            running = conn.execute(
                """
                SELECT 1
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                WHERE e.subscription_id = ?
                  AND j.type = 'process_episode'
                  AND j.status = 'running'
                LIMIT 1
                """,
                (id,),
            ).fetchone()
            if running:
                conn.commit()
                return None

            cursor = conn.execute(
                """
                UPDATE subscriptions
                SET deletion_status = 'cleaning',
                    deletion_updated_at = CURRENT_TIMESTAMP,
                    deletion_error = NULL
                WHERE id = ?
                  AND (
                      deletion_status IN ('pending', 'failed')
                      OR (
                          deletion_status = 'cleaning'
                          AND deletion_updated_at <= datetime(CURRENT_TIMESTAMP, '-5 minutes')
                      )
                  )
                """,
                (id,),
            )
            if cursor.rowcount == 0:
                conn.commit()
                return None

            row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (id,)).fetchone()
            conn.commit()
            return dict(row) if row else None

    def mark_deletion_failed(self, id: int, error: str) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE subscriptions
                SET deletion_status = 'failed',
                    deletion_updated_at = CURRENT_TIMESTAMP,
                    deletion_error = ?
                WHERE id = ?
                """,
                (error, id),
            )
            conn.commit()

    def delete(self, id: int):
        with get_db_connection() as conn:
            conn.execute(
                "DELETE FROM jobs WHERE episode_id IN (SELECT id FROM episodes WHERE subscription_id = ?)",
                (id,),
            )
            conn.execute("DELETE FROM episodes WHERE subscription_id = ?", (id,))
            conn.execute("DELETE FROM source_items WHERE subscription_id = ?", (id,))
            conn.execute("DELETE FROM user_subscriptions WHERE subscription_id = ?", (id,))
            conn.execute("DELETE FROM subscriptions WHERE id = ?", (id,))
            conn.commit()

    def update_settings(
        self,
        id: int,
        remove_ads: bool,
        remove_promos: bool,
        remove_intros: bool,
        remove_outros: bool,
        custom_instructions: str,
        append_summary: bool,
        append_title_intro: bool,
        ai_rewrite_description: bool,
        ai_audio_summary: bool,
        retention_days: int = 30,
        manual_retention_days: int = 14,
        retention_limit: int = 1,
        inherit_content_removal: bool | None = None,
        inherit_retention: bool | None = None,
        inherit_default_features: bool | None = None,
        inherit_custom_instructions: bool | None = None,
        watermark_artwork: bool | None = None,
    ):
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
                    retention_limit = ?,
                    inherit_content_removal = COALESCE(?, inherit_content_removal),
                    inherit_retention = COALESCE(?, inherit_retention),
                    inherit_default_features = COALESCE(?, inherit_default_features),
                    inherit_custom_instructions = COALESCE(?, inherit_custom_instructions),
                    watermark_artwork = COALESCE(?, watermark_artwork)
                WHERE id = ?
            """, (
                remove_ads,
                remove_promos,
                remove_intros,
                remove_outros,
                custom_instructions,
                append_summary,
                append_title_intro,
                ai_rewrite_description,
                ai_audio_summary,
                retention_days,
                manual_retention_days,
                retention_limit,
                None if inherit_content_removal is None else int(inherit_content_removal),
                None if inherit_retention is None else int(inherit_retention),
                None if inherit_default_features is None else int(inherit_default_features),
                None if inherit_custom_instructions is None else int(inherit_custom_instructions),
                None if watermark_artwork is None else int(watermark_artwork),
                id,
            ))
            conn.commit()


class SourceItemRepository:
    """Track provider membership independently of retained episode rows."""

    def get(self, subscription_id: int, external_id: str) -> dict | None:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM source_items WHERE subscription_id = ? AND external_id = ?",
                (subscription_id, external_id),
            ).fetchone()
            return dict(row) if row else None

    def count(self, subscription_id: int) -> int:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM source_items WHERE subscription_id = ?",
                (subscription_id,),
            ).fetchone()
            return int(row["count"] if row else 0)

    def begin_reconciliation(self, subscription_id: int) -> None:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE source_items SET is_present = 0 WHERE subscription_id = ?",
                (subscription_id,),
            )
            conn.commit()

    def observe(
        self,
        subscription_id: int,
        external_id: str,
        canonical_url: str,
    ) -> tuple[dict, bool]:
        with get_db_connection() as conn:
            existing = conn.execute(
                "SELECT * FROM source_items WHERE subscription_id = ? AND external_id = ?",
                (subscription_id, external_id),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO source_items
                    (subscription_id, external_id, canonical_url, is_present)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(subscription_id, external_id) DO UPDATE SET
                    canonical_url = excluded.canonical_url,
                    last_seen_at = CURRENT_TIMESTAMP,
                    is_present = 1
                """,
                (subscription_id, external_id, canonical_url),
            )
            row = conn.execute(
                "SELECT * FROM source_items WHERE subscription_id = ? AND external_id = ?",
                (subscription_id, external_id),
            ).fetchone()
            conn.commit()
            return dict(row), existing is None

    def set_eligibility(
        self,
        subscription_id: int,
        external_id: str,
        eligibility: str,
        exclusion_reason: str | None = None,
    ) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE source_items
                SET eligibility = ?, exclusion_reason = ?
                WHERE subscription_id = ? AND external_id = ?
                """,
                (eligibility, exclusion_reason, subscription_id, external_id),
            )
            conn.commit()


class StaleAttempt(RuntimeError):
    """A cancelled/replaced worker must not write episode state."""


class EpisodeRepository:
    def __init__(self, attempt: tuple[int, str] | None = None):
        self.attempt = attempt
        self.pending_metadata = {}

    def owns_attempt(self, episode_id: int) -> bool:
        if self.attempt is None:
            return self.get_status(episode_id) == 'processing'
        with get_db_connection() as conn:
            return self._owns_attempt(conn, episode_id)

    def _owns_attempt(self, conn, episode_id):
        job_id, token = self.attempt
        return conn.execute("""
            SELECT 1 FROM jobs j JOIN episodes e ON e.id=j.episode_id
            JOIN subscriptions s ON s.id=e.subscription_id
            WHERE j.id=? AND j.episode_id=? AND j.locked_by=?
            AND j.status='running' AND j.cancel_requested=0
            AND e.status='processing' AND s.is_active=1 AND s.deletion_status IS NULL
        """, (job_id, episode_id, token)).fetchone() is not None

    @contextmanager
    def _write_connection(self, episode_id):
        with get_db_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if self.attempt and not self._owns_attempt(conn, episode_id):
                raise StaleAttempt('Processing attempt was cancelled or replaced')
            yield conn

    def create_or_ignore(self, episode: dict) -> bool:
        """Returns True if created, False if already exists."""
        with get_db_connection() as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO episodes (subscription_id, guid, title, pub_date, original_url, duration, description, status, file_size, discovered_at)
                    SELECT :subscription_id, :guid, :title, :pub_date, :original_url,
                           :duration, :description, :status, :file_size, CURRENT_TIMESTAMP
                    WHERE EXISTS (
                        SELECT 1 FROM subscriptions
                        WHERE id = :subscription_id
                          AND is_active = 1
                          AND deletion_status IS NULL
                    )
                """, episode)
                if cursor.rowcount == 0:
                    return False
                if episode.get("status") == "pending":
                    _enqueue_job(conn, cursor.lastrowid)
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
                SELECT e.*,
                       s.title as podcast_title,
                       j.id as job_id,
                       j.status as job_status,
                       j.attempts as job_attempts,
                       j.locked_at as job_locked_at,
                       j.locked_by as job_locked_by,
                       j.next_run_at as job_next_run_at,
                       j.error as job_error
                FROM episodes e
                JOIN subscriptions s ON e.subscription_id = s.id
                LEFT JOIN jobs j ON j.episode_id = e.id
                    AND j.type = 'process_episode'
                    AND j.status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
                WHERE e.status IN ('processing', 'pending', 'rate_limited')
                OR (e.status = 'failed' AND e.next_retry_at IS NOT NULL)
                ORDER BY 
                    CASE COALESCE(j.status, e.status)
                        WHEN 'running' THEN 1
                        WHEN 'processing' THEN 1 
                        WHEN 'queued' THEN 2
                        WHEN 'pending' THEN 2 
                        WHEN 'retry_scheduled' THEN 3
                        WHEN 'rate_limited' THEN 3
                        ELSE 4 
                    END,
                    COALESCE(j.priority, 100) ASC,
                    COALESCE(j.next_run_at, e.next_retry_at, e.pub_date) ASC,
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

    def get_completed_by_subscription(self, subscription_id: int) -> List[dict]:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM episodes
                WHERE subscription_id = ?
                  AND local_filename IS NOT NULL AND status != 'ignored'
                ORDER BY pub_date DESC
                """,
                (subscription_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_completed_with_subscription_info(self) -> List[dict]:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT e.*,
                       s.title AS podcast_title,
                       s.slug AS podcast_slug,
                       s.image_url AS podcast_image
                FROM episodes e
                JOIN subscriptions s ON e.subscription_id = s.id
                WHERE e.local_filename IS NOT NULL AND e.status != 'ignored' AND s.deletion_status IS NULL
                ORDER BY e.pub_date DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def _episode_filter(subscription_id, search=None, filter='all'):
        clauses, params = ['subscription_id = ?'], [subscription_id]
        if search:
            clauses.append("title LIKE ? ESCAPE '\\'")
            params.append('%' + search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%')
        filters = {
            'all': '1=1', 'completed': "local_filename IS NOT NULL AND status != 'ignored'",
            'manual': 'is_manual_download=1',
            'not-downloaded': "local_filename IS NULL AND status IN ('unprocessed','failed','pending','rate_limited')",
            'ignored': "status='ignored'", 'played': 'COALESCE(listen_count,0)>0',
        }
        if filter not in filters:
            raise ValueError('Unknown episode filter')
        clauses.append(filters[filter])
        return ' AND '.join(clauses), params

    def get_by_subscription_paginated(self, subscription_id: int, limit: int = 20, offset: int = 0, search: str = None, filter: str = 'all') -> list:
        where, params = self._episode_filter(subscription_id, search, filter)
        with get_db_connection() as conn:
            return conn.execute(f'SELECT * FROM episodes WHERE {where} ORDER BY pub_date DESC, id DESC LIMIT ? OFFSET ?', (*params, max(1, min(100, limit)), max(0, offset))).fetchall()

    def count_by_subscription(self, subscription_id: int, search: str = None, filter: str = 'all') -> int:
        where, params = self._episode_filter(subscription_id, search, filter)
        with get_db_connection() as conn:
            return conn.execute(f'SELECT COUNT(*) FROM episodes WHERE {where}', params).fetchone()[0]

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
                    ai_summary = ai_summary
                WHERE id = ?
                  AND EXISTS (
                      SELECT 1 FROM subscriptions s
                      WHERE s.id = episodes.subscription_id
                        AND s.is_active = 1
                        AND s.deletion_status IS NULL
                  )
            """, (processing_flags, id))
            JobRepository().cancel_active_for_episode(id, conn=conn)
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
            conn.execute("""
                UPDATE jobs
                SET status = CASE
                        WHEN episode_id IN (SELECT id FROM episodes WHERE status = 'ignored')
                            THEN 'cancelled'
                        ELSE 'retry_scheduled'
                    END,
                    locked_at = NULL,
                    locked_by = NULL,
                    error = CASE
                        WHEN episode_id IN (SELECT id FROM episodes WHERE status = 'ignored')
                            THEN 'Cancellation acknowledged after worker restart'
                        ELSE 'Interrupted by system restart'
                    END,
                    next_run_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'running'
            """)
            conn.commit()

    def update_retry(self, id: int, retry_count: int, next_retry_at: datetime, error: str):
        with self._write_connection(id) as conn:
            cursor = conn.execute("""
                UPDATE episodes 
                SET status = 'failed', 
                    retry_count = ?, 
                    next_retry_at = ?, 
                    error_message = ? 
                WHERE id = ?
                  AND status = 'processing'
                  AND EXISTS (
                      SELECT 1 FROM subscriptions s
                      WHERE s.id = episodes.subscription_id
                        AND s.is_active = 1
                        AND s.deletion_status IS NULL
                  )
            """, (retry_count, next_retry_at, error, id))
            if cursor.rowcount:
                _schedule_retry_job(conn, id, next_retry_at, error)
            else:
                JobRepository().cancel_active_for_episode(id, conn=conn)
            conn.commit()

    def update_rate_limited(self, id: int, next_retry_at: datetime, error: str):
        """Set episode to rate_limited status with scheduled retry at API quota reset."""
        with self._write_connection(id) as conn:
            cursor = conn.execute("""
                UPDATE episodes 
                SET status = 'rate_limited', 
                    next_retry_at = ?, 
                    error_message = ?,
                    processing_step = 'Waiting for API quota reset'
                WHERE id = ?
                  AND status = 'processing'
                  AND EXISTS (
                      SELECT 1 FROM subscriptions s
                      WHERE s.id = episodes.subscription_id
                        AND s.is_active = 1
                        AND s.deletion_status IS NULL
                  )
            """, (next_retry_at, error, id))
            if cursor.rowcount:
                _schedule_retry_job(conn, id, next_retry_at, error, status="rate_limited")
            else:
                JobRepository().cancel_active_for_episode(id, conn=conn)
            conn.commit()

    def update_status(self, id: int, status: str, error: str = None, filename: str = None, file_size: int = None):
        with self._write_connection(id) as conn:
            eligibility = ""
            if status in {"pending", "completed", "failed"}:
                eligibility = """
                  AND EXISTS (
                      SELECT 1 FROM subscriptions s
                      WHERE s.id = episodes.subscription_id
                        AND s.is_active = 1
                        AND s.deletion_status IS NULL
                  )
                """
            if status in {"completed", "failed"}:
                eligibility += " AND status = 'processing'"
            cursor = conn.execute(
                f"""
                UPDATE episodes
                SET status = ?, error_message = ?, local_filename = COALESCE(?, local_filename), file_size = COALESCE(?, file_size),
                    processed_at = COALESCE(?, processed_at), next_retry_at = NULL
                WHERE id = ? {eligibility}
                """,
                (status, error, filename, file_size, datetime.now() if status == 'completed' else None, id),
            )
            if status == 'completed' and cursor.rowcount:
                if self.pending_metadata:
                    columns = ', '.join(f'{key}=?' for key in self.pending_metadata)
                    conn.execute(f'UPDATE episodes SET {columns} WHERE id=?', (*self.pending_metadata.values(), id))
                conn.execute("UPDATE episodes SET publication_pending=1, processing_step='completed', progress=100 WHERE id=?", (id,))
            if status == "pending":
                if cursor.rowcount:
                    _enqueue_job(conn, id)
            elif status == "completed":
                if cursor.rowcount:
                    JobRepository().complete_for_episode(id, conn=conn)
                else:
                    JobRepository().cancel_active_for_episode(id, conn=conn)
            elif status == "failed":
                if cursor.rowcount:
                    JobRepository().fail_running_for_episode(id, error or "Processing failed", conn=conn)
                else:
                    JobRepository().cancel_active_for_episode(id, conn=conn)
            conn.commit()

    def update_progress(self, id: int, step: str, progress: int, transcript_path: str = None, ad_report_path: str = None, report_path: str = None):
        if self.attempt:
            for key, value in [('transcript_path', transcript_path), ('ad_report_path', ad_report_path), ('report_path', report_path)]:
                if value:
                    self.pending_metadata[key] = value
            transcript_path = ad_report_path = report_path = None
        with self._write_connection(id) as conn:
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
            
            sql = f"UPDATE episodes SET {', '.join(updates)} WHERE id = ? AND status = 'processing'"
            conn.execute(sql, params)
            conn.execute("""
                UPDATE jobs
                SET updated_at = CURRENT_TIMESTAMP
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status = 'running'
            """, (id,))
            conn.commit()

    def update_description(self, id: int, description: str):
        if self.attempt:
            self.pending_metadata["description"] = description
            return
        with self._write_connection(id) as conn:
            conn.execute("UPDATE episodes SET description = ? WHERE id = ?", (description, id))
            conn.commit()

    def update_source_media_path(self, id: int, source_media_path: str) -> None:
        with self._write_connection(id) as conn:
            conn.execute(
                "UPDATE episodes SET source_media_path = ? WHERE id = ?",
                (source_media_path, id),
            )
            conn.commit()

    def update_ai_summary(self, id: int, summary: str):
        if self.attempt:
            self.pending_metadata["ai_summary"] = summary
            return
        with self._write_connection(id) as conn:
            conn.execute("UPDATE episodes SET ai_summary = ? WHERE id = ?", (summary, id))
            conn.commit()

    def update_status_by_guid(self, subscription_id: int, guid: str, status: str, condition_status: str = None):
        """Update status of an episode by GUID, optionally only if current status matches condition."""
        with get_db_connection() as conn:
            if condition_status:
                cursor = conn.execute("""
                    UPDATE episodes 
                    SET status = ? 
                    WHERE subscription_id = ? AND guid = ? AND (status = ? or status = 'pending_manual')
                      AND EXISTS (
                          SELECT 1 FROM subscriptions s
                          WHERE s.id = episodes.subscription_id
                            AND s.is_active = 1
                            AND s.deletion_status IS NULL
                      )
                """, (status, subscription_id, guid, condition_status))
            else:
                cursor = conn.execute("""
                    UPDATE episodes 
                    SET status = ? 
                    WHERE subscription_id = ? AND guid = ?
                      AND EXISTS (
                          SELECT 1 FROM subscriptions s
                          WHERE s.id = episodes.subscription_id
                            AND s.is_active = 1
                            AND s.deletion_status IS NULL
                      )
                """, (status, subscription_id, guid))
            if status == "pending" and cursor.rowcount:
                row = conn.execute(
                    "SELECT id FROM episodes WHERE subscription_id = ? AND guid = ?",
                    (subscription_id, guid),
                ).fetchone()
                if row:
                    _enqueue_job(conn, row["id"])
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
                "SELECT * FROM episodes WHERE subscription_id = ? AND replace(local_filename, char(92), '/') = ?",
                (subscription_id, filename.replace("\\", "/"))
            ).fetchone()
            if row:
                return Episode.model_validate(dict(row))
            return None

    def count_processing(self) -> int:
        """Count episodes currently in 'processing' status. Used for concurrent limit enforcement."""
        with get_db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM episodes WHERE status = 'processing'").fetchone()
            return row['count'] if row else 0

    def request_deletion(self, id: int) -> bool:
        """Mark an episode ignored and cancel queued work while retaining running-job ownership."""
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute("""
                UPDATE episodes 
                SET status = 'ignored',
                    local_filename = NULL,
                    transcript_path = NULL,
                    ad_report_path = NULL,
                    report_path = NULL,
                    progress = 0,
                    processing_step = 'cancellation requested',
                    next_retry_at = NULL
                WHERE id = ?
            """, (id,))
            conn.execute("""
                UPDATE jobs
                SET status = 'cancelled',
                    locked_at = NULL,
                    locked_by = NULL,
                    error = 'Cancelled because episode deletion was requested',
                    updated_at = CURRENT_TIMESTAMP
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status IN ('queued', 'retry_scheduled', 'rate_limited')
            """, (id,))
            conn.execute("""
                UPDATE jobs
                SET error = 'Cancellation requested because episode is being deleted', cancel_requested=1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status = 'running'
            """, (id,))
            conn.commit()
            return cursor.rowcount > 0

    def soft_delete(self, id: int):
        """Backward-compatible alias for requesting safe episode deletion."""
        return self.request_deletion(id)


def _enqueue_job(conn: sqlite3.Connection, episode_id: int, job_type: str = "process_episode", priority: int = 100):
    eligible = conn.execute("""
        SELECT 1
        FROM episodes e
        JOIN subscriptions s ON s.id = e.subscription_id
        WHERE e.id = ?
          AND s.is_active = 1
          AND s.deletion_status IS NULL
    """, (episode_id,)).fetchone()
    if not eligible:
        return None

    existing = conn.execute("""
        SELECT id FROM jobs
        WHERE episode_id = ?
          AND type = ?
          AND status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
        LIMIT 1
    """, (episode_id, job_type)).fetchone()

    if existing:
        conn.execute("""
            UPDATE jobs
            SET status = CASE WHEN status = 'running' THEN status ELSE 'queued' END,
                priority = ?,
                next_run_at = CURRENT_TIMESTAMP,
                error = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (priority, existing["id"]))
        return existing["id"]

    cursor = conn.execute("""
        INSERT INTO jobs (episode_id, type, status, priority, next_run_at)
        VALUES (?, ?, 'queued', ?, CURRENT_TIMESTAMP)
    """, (episode_id, job_type, priority))
    return cursor.lastrowid


def _schedule_retry_job(
    conn: sqlite3.Connection,
    episode_id: int,
    next_run_at: datetime,
    error: str,
    status: str = "retry_scheduled",
):
    existing = conn.execute("""
        SELECT id FROM jobs
        WHERE episode_id = ?
          AND type = 'process_episode'
          AND status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
        LIMIT 1
    """, (episode_id,)).fetchone()

    if existing:
        conn.execute("""
            UPDATE jobs
            SET status = ?,
                locked_at = NULL,
                locked_by = NULL,
                next_run_at = ?,
                error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, next_run_at, error, existing["id"]))
        return

    conn.execute("""
        INSERT INTO jobs (episode_id, type, status, next_run_at, error)
        VALUES (?, 'process_episode', ?, ?, ?)
    """, (episode_id, status, next_run_at, error))


class JobRepository:
    def heartbeat(self, job_id: int, token: str):
        with get_db_connection() as conn:
            conn.execute("UPDATE jobs SET updated_at=CURRENT_TIMESTAMP WHERE id=? AND locked_by=? AND status='running'", (job_id, token))
            conn.commit()

    def acknowledge(self, job_id: int, token: str):
        """Release only this claim; cancellation/restart cannot release a successor."""
        with get_db_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT j.episode_id, e.status FROM jobs j JOIN episodes e ON e.id=j.episode_id WHERE j.id=? AND j.locked_by=? AND j.status='running'", (job_id, token)).fetchone()
            if not row:
                return
            requeue = row['status'] in ('pending', 'processing')
            if row['status'] == 'processing':
                conn.execute("UPDATE episodes SET status='pending', processing_step='worker interrupted' WHERE id=?", (row['episode_id'],))
            conn.execute("UPDATE jobs SET status=?, locked_by=NULL, locked_at=NULL, cancel_requested=0, updated_at=CURRENT_TIMESTAMP WHERE id=? AND locked_by=?", ('queued' if requeue else 'cancelled', job_id, token))
            conn.commit()


    """SQLite-backed processing jobs with transaction-based claiming."""

    DEFAULT_STALE_AFTER_MINUTES = 180

    def enqueue(self, episode_id: int, job_type: str = "process_episode", priority: int = 100):
        with get_db_connection() as conn:
            job_id = _enqueue_job(conn, episode_id, job_type, priority)
            conn.commit()
            return job_id

    def count_running(self) -> int:
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS count
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                WHERE j.status = 'running'
                  AND e.status IN ('processing', 'ignored')
            """).fetchone()
            return row["count"] if row else 0

    def is_running_for_episode(self, episode_id: int) -> bool:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM jobs
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status = 'running'
                LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            return row is not None

    def count_claimable(self) -> int:
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS count
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                JOIN subscriptions s ON s.id = e.subscription_id
                WHERE j.type = 'process_episode'
                  AND j.status IN ('queued', 'retry_scheduled', 'rate_limited')
                  AND (j.next_run_at IS NULL OR j.next_run_at <= CURRENT_TIMESTAMP)
                  AND e.status IN ('pending', 'failed', 'rate_limited')
                  AND s.is_active = 1
                  AND s.deletion_status IS NULL
            """).fetchone()
            return row["count"] if row else 0

    def repair_missing_active_jobs(self) -> int:
        """Recreate claimable jobs for episodes that are queue-visible but lack an active job row."""
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("""
                SELECT e.id, e.status, e.next_retry_at
                FROM episodes e
                JOIN subscriptions s ON s.id = e.subscription_id
                WHERE (
                    e.status = 'pending'
                    OR (
                        e.status IN ('failed', 'rate_limited')
                        AND e.next_retry_at IS NOT NULL
                    )
                )
                AND s.is_active = 1
                AND s.deletion_status IS NULL
                AND NOT EXISTS (
                    SELECT 1
                    FROM jobs j
                    WHERE j.episode_id = e.id
                      AND j.type = 'process_episode'
                      AND j.status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
                )
            """).fetchall()

            for row in rows:
                if row["status"] == "pending":
                    _enqueue_job(conn, row["id"])
                else:
                    _schedule_retry_job(
                        conn,
                        row["id"],
                        row["next_retry_at"],
                        "Recreated missing processing job",
                        status="rate_limited" if row["status"] == "rate_limited" else "retry_scheduled",
                    )

            conn.commit()
            return len(rows)

    def claim_due(self, limit: int, worker_id: str | None = None, *, max_running: int | None = None) -> List[dict]:
        if limit <= 0:
            return []

        worker_id = worker_id or f"{socket.gethostname()}:{uuid4().hex}"
        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if max_running is not None:
                running = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
                limit = min(limit, max(0, max_running - running))
            rows = conn.execute("""
                SELECT j.id AS job_id,
                       j.attempts AS job_attempts,
                       e.*
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                JOIN subscriptions s ON s.id = e.subscription_id
                WHERE j.type = 'process_episode'
                  AND j.status IN ('queued', 'retry_scheduled', 'rate_limited')
                  AND (j.next_run_at IS NULL OR j.next_run_at <= CURRENT_TIMESTAMP)
                  AND e.status IN ('pending', 'failed', 'rate_limited')
                  AND s.is_active = 1
                  AND s.deletion_status IS NULL
                ORDER BY j.priority ASC, j.created_at ASC
                LIMIT ?
            """, (limit,)).fetchall()

            job_ids = [row["job_id"] for row in rows]
            for job_id in job_ids:
                conn.execute("""
                    UPDATE jobs
                    SET status = 'running',
                        attempts = attempts + 1,
                        locked_at = CURRENT_TIMESTAMP,
                        locked_by = ?,
                        error = NULL, cancel_requested = 0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (worker_id, job_id))

            for row in rows:
                conn.execute("""
                    UPDATE episodes
                    SET status = 'processing',
                        processing_step = 'queued',
                        progress = 0,
                        error_message = NULL
                    WHERE id = ?
                """, (row["id"],))

            conn.commit()
            return [dict(row, claim_token=worker_id, claim_attempt=row["job_attempts"] + 1) for row in rows]

    def recover_stale_running(self, max_age_minutes: int | None = None) -> int:
        """Return stale running jobs to the queue after a worker crash or restart."""
        max_age_minutes = max_age_minutes or self.DEFAULT_STALE_AFTER_MINUTES
        stale_modifier = f"-{int(max_age_minutes)} minutes"

        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            inconsistent_rows = conn.execute("""
                SELECT j.id AS job_id, j.episode_id, e.status AS episode_status
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                WHERE j.type = 'process_episode'
                  AND j.status = 'running'
                  AND e.status != 'processing'
                  AND e.status != 'ignored'
                  AND j.cancel_requested = 0
            """).fetchall()

            for row in inconsistent_rows:
                replacement_status = "completed" if row["episode_status"] == "completed" else "cancelled"
                conn.execute("""
                    UPDATE jobs
                    SET status = ?,
                        locked_at = NULL,
                        locked_by = NULL,
                        error = 'Cleared inconsistent running job state',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (replacement_status, row["job_id"]))

            stale_cancellations = conn.execute("""
                SELECT j.id AS job_id
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                WHERE j.type = 'process_episode'
                  AND j.status = 'running'
                  AND (e.status = 'ignored' OR j.cancel_requested = 1)
                  AND (
                      COALESCE(j.updated_at, j.locked_at) IS NULL
                      OR COALESCE(j.updated_at, j.locked_at) <= datetime(CURRENT_TIMESTAMP, ?)
                  )
            """, (stale_modifier,)).fetchall()

            for row in stale_cancellations:
                conn.execute("""
                    UPDATE jobs
                    SET status = 'cancelled',
                        locked_at = NULL,
                        locked_by = NULL,
                        error = 'Cancellation acknowledged after stale worker timeout',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (row["job_id"],))

            rows = conn.execute("""
                SELECT j.id AS job_id, j.episode_id
                FROM jobs j
                JOIN episodes e ON e.id = j.episode_id
                WHERE j.type = 'process_episode'
                  AND j.status = 'running'
                  AND e.status = 'processing'
                  AND (
                    COALESCE(j.updated_at, j.locked_at) IS NULL
                    OR COALESCE(j.updated_at, j.locked_at) <= datetime(CURRENT_TIMESTAMP, ?)
                  )
            """, (stale_modifier,)).fetchall()

            if not rows and not inconsistent_rows and not stale_cancellations:
                conn.commit()
                return 0

            job_ids = [row["job_id"] for row in rows]
            episode_ids = [row["episode_id"] for row in rows]

            conn.executemany("""
                UPDATE jobs
                SET status = 'queued',
                    locked_at = NULL,
                    locked_by = NULL,
                    next_run_at = CURRENT_TIMESTAMP,
                    error = 'Recovered stale running job after worker interruption',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [(job_id,) for job_id in job_ids])

            conn.executemany("""
                UPDATE episodes
                SET status = 'pending',
                    processing_step = 'retry scheduled after worker interruption',
                    progress = 0
                WHERE id = ?
            """, [(episode_id,) for episode_id in episode_ids])

            conn.commit()
            return len(rows) + len(inconsistent_rows) + len(stale_cancellations)

    def complete_for_episode(self, episode_id: int, conn: sqlite3.Connection | None = None):
        if conn is None:
            with get_db_connection() as own_conn:
                self.complete_for_episode(episode_id, conn=own_conn)
                own_conn.commit()
            return

        conn.execute("""
                UPDATE jobs
                SET status = 'completed',
                    locked_at = NULL,
                    locked_by = NULL,
                    error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
            """, (episode_id,))

    def cancel_active_for_episode(self, episode_id: int, conn: sqlite3.Connection | None = None):
        if conn is None:
            with get_db_connection() as own_conn:
                self.cancel_active_for_episode(episode_id, conn=own_conn)
                own_conn.commit()
            return

        conn.execute("""
                UPDATE jobs
                SET status = CASE WHEN status='running' THEN 'running' ELSE 'cancelled' END,
                    cancel_requested = 1,
                    locked_at = CASE WHEN status='running' THEN locked_at ELSE NULL END,
                    locked_by = CASE WHEN status='running' THEN locked_by ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status IN ('queued', 'running', 'retry_scheduled', 'rate_limited')
            """, (episode_id,))

    def fail_running_for_episode(self, episode_id: int, error: str, conn: sqlite3.Connection | None = None):
        if conn is None:
            with get_db_connection() as own_conn:
                self.fail_running_for_episode(episode_id, error, conn=own_conn)
                own_conn.commit()
            return

        conn.execute("""
                UPDATE jobs
                SET status = 'failed',
                    locked_at = NULL,
                    locked_by = NULL,
                    error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE episode_id = ?
                  AND type = 'process_episode'
                  AND status = 'running'
            """, (error, episode_id))


class FeedTokenRepository:
    """Manage bearer tokens for protected RSS feeds and audio URLs."""

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(self, user_id: int | None = None, name: str = "Podcast app") -> str:
        token = secrets.token_urlsafe(32)
        token_hash = self.hash_token(token)
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO feed_tokens (user_id, token_hash, name)
                VALUES (?, ?, ?)
            """, (user_id, token_hash, name))
            conn.commit()
        return token

    def list_active(self) -> list[dict]:
        with get_db_connection() as conn:
            rows = conn.execute("""
                SELECT ft.id,
                       ft.user_id,
                       ft.name,
                       ft.created_at,
                       ft.last_used_at,
                       u.username
                FROM feed_tokens ft
                LEFT JOIN users u ON u.id = ft.user_id
                WHERE ft.revoked_at IS NULL
                ORDER BY ft.created_at DESC
            """).fetchall()
            return [dict(row) for row in rows]

    def validate(self, token: str | None) -> bool:
        if not token:
            return False

        token_hash = self.hash_token(token)
        with get_db_connection() as conn:
            row = conn.execute("""
                SELECT id FROM feed_tokens
                WHERE token_hash = ?
                  AND revoked_at IS NULL
            """, (token_hash,)).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE feed_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
                (row["id"],),
            )
            conn.commit()
            return True

    def revoke(self, token: str):
        token_hash = self.hash_token(token)
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE feed_tokens
                SET revoked_at = CURRENT_TIMESTAMP
                WHERE token_hash = ? AND revoked_at IS NULL
            """, (token_hash,))
            conn.commit()

    def revoke_by_id(self, token_id: int) -> bool:
        with get_db_connection() as conn:
            result = conn.execute("""
                UPDATE feed_tokens
                SET revoked_at = CURRENT_TIMESTAMP
                WHERE id = ? AND revoked_at IS NULL
            """, (token_id,))
            conn.commit()
            return result.rowcount > 0


class ApiTokenRepository:
    """Manage scoped bearer tokens for the AI-facing REST API."""

    TOKEN_PREFIX = "par_"
    DISPLAY_PREFIX_LENGTH = 12
    VALID_SCOPES = {"read", "write", "process", "admin"}

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def normalize_scopes(cls, scopes: list[str] | str | None) -> str:
        if isinstance(scopes, str):
            raw_scopes = [scope.strip() for scope in scopes.split(",")]
        else:
            raw_scopes = scopes or []

        cleaned = sorted({scope for scope in raw_scopes if scope in cls.VALID_SCOPES})
        if not cleaned:
            cleaned = ["read"]
        return ",".join(cleaned)

    def create(
        self,
        name: str,
        scopes: list[str] | str | None = None,
        user_id: int | None = None,
        requests_per_minute: int | None = None,
        requests_per_day: int | None = None,
    ) -> str:
        token = f"{self.TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
        token_hash = self.hash_token(token)
        token_prefix = token[: self.DISPLAY_PREFIX_LENGTH]
        normalized_scopes = self.normalize_scopes(scopes)
        safe_name = (name or "AI API token").strip()[:120] or "AI API token"

        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_tokens
                    (user_id, token_hash, token_prefix, name, scopes, requests_per_minute, requests_per_day)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    token_hash,
                    token_prefix,
                    safe_name,
                    normalized_scopes,
                    requests_per_minute if requests_per_minute and requests_per_minute > 0 else None,
                    requests_per_day if requests_per_day and requests_per_day > 0 else None,
                ),
            )
            conn.commit()

        return token

    def list_active(self) -> list[dict]:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT at.id,
                       at.user_id,
                       at.token_prefix,
                       at.name,
                       at.scopes,
                       at.requests_per_minute,
                       at.requests_per_day,
                       at.created_at,
                       at.last_used_at,
                       u.username,
                       COALESCE(u.is_admin, 0) AS is_admin
                FROM api_tokens at
                LEFT JOIN users u ON u.id = at.user_id
                WHERE at.revoked_at IS NULL
                ORDER BY at.created_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def validate(self, token: str | None) -> dict | None:
        if not token or not token.startswith(self.TOKEN_PREFIX):
            return None

        token_hash = self.hash_token(token)
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT at.id,
                       at.user_id,
                       at.token_prefix,
                       at.name,
                       at.scopes,
                       at.requests_per_minute,
                       at.requests_per_day,
                       u.username,
                       COALESCE(u.is_admin, 0) AS is_admin
                FROM api_tokens at
                LEFT JOIN users u ON u.id = at.user_id
                WHERE at.token_hash = ?
                  AND at.revoked_at IS NULL
                """,
                (token_hash,),
            ).fetchone()
            if not row:
                return None
            if row["user_id"] is not None and row["username"] is None:
                return None

            conn.execute(
                "UPDATE api_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
                (row["id"],),
            )
            conn.commit()
            return dict(row)

    def revoke_by_id(self, token_id: int) -> bool:
        with get_db_connection() as conn:
            result = conn.execute(
                """
                UPDATE api_tokens
                SET revoked_at = CURRENT_TIMESTAMP
                WHERE id = ? AND revoked_at IS NULL
                """,
                (token_id,),
            )
            conn.commit()
            return result.rowcount > 0


class ApiRateLimitRepository:
    """SQLite fixed-window counters for AI API request limits."""

    def check_and_increment(self, bucket_key: str, limit: int, window_seconds: int, window_name: str) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0

        now = int(time.time())
        window_start = now - (now % window_seconds)
        reset_after = window_start + window_seconds - now

        with get_db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT request_count
                FROM api_rate_limits
                WHERE bucket_key = ?
                  AND window_name = ?
                  AND window_start = ?
                """,
                (bucket_key, window_name, window_start),
            ).fetchone()

            if not row:
                conn.execute(
                    """
                    INSERT INTO api_rate_limits (bucket_key, window_name, window_start, request_count)
                    VALUES (?, ?, ?, 1)
                    """,
                    (bucket_key, window_name, window_start),
                )
                self._cleanup_old_windows(conn, now)
                conn.commit()
                return True, reset_after

            count = int(row["request_count"])
            if count >= limit:
                conn.commit()
                return False, reset_after

            conn.execute(
                """
                UPDATE api_rate_limits
                SET request_count = request_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE bucket_key = ?
                  AND window_name = ?
                  AND window_start = ?
                """,
                (bucket_key, window_name, window_start),
            )
            self._cleanup_old_windows(conn, now)
            conn.commit()
            return True, reset_after

    @staticmethod
    def _cleanup_old_windows(conn: sqlite3.Connection, now: int) -> None:
        conn.execute(
            "DELETE FROM api_rate_limits WHERE window_start < ?",
            (now - (3 * 24 * 60 * 60),),
        )
