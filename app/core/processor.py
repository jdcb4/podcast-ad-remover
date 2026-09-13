import asyncio
import copy
from uuid import uuid4
import os
import logging
import aiofiles
import json
import shutil
from datetime import datetime
from pathlib import Path
from app.core.artifacts import episode_directory, legacy_directory, source_fingerprint
from app.core.provider_budget import current_claim, ProviderBudgetExceeded
from app.core.resource_budget import require_scratch
from app.core.reports import render_ad_report
from app.core.config import settings
from app.core.models import Episode
from app.core.time_utils import now_utc
from app.infra.repository import EpisodeRepository, SubscriptionRepository, SourceItemRepository, JobRepository, StaleAttempt
from app.core.ai_services import Transcriber, AdDetector, RateLimitError, PermanentProviderError, AnalysisError
from app.core.audio import AudioProcessor
from app.core.rss_gen import RSSGenerator
from app.core.youtube import (
    YouTubeDownloadCancelled,
    hydrate_youtube_entry,
    video_id_from_url,
)
from app.core.sources import get_source_adapter
from app.core.notifications import (
    EVENT_BREAKING_ERROR,
    EVENT_EPISODE_DOWNLOAD,
    send_notification_async,
)
from app.core.sponsorblock import SponsorBlockClient, categories_for_subscription

logger = logging.getLogger(__name__)

class Processor:
    _is_background_worker = False
    _manual_processor = None
    _active_task_ids = set()
    _queue_lock = asyncio.Lock()  # Prevent race conditions in process_queue
    DELETION_ACK_TIMEOUT_SECONDS = 10.0
    DELETION_POLL_INTERVAL_SECONDS = 0.1

    def __init__(self):
        self.ep_repo = EpisodeRepository()
        self.sub_repo = SubscriptionRepository()
        self.source_item_repo = SourceItemRepository()
        self.job_repo = JobRepository()
        self.transcriber = Transcriber()
        self.ad_detector = AdDetector()
        self.rss_gen = RSSGenerator()
        self.sponsorblock = SponsorBlockClient()

    def _remove_episode_directory(self, episode_dir: str, action: str) -> bool:
        """Remove an episode directory only if it is contained by PODCASTS_DIR."""
        requested = Path(episode_dir).absolute()
        target = requested.resolve()
        podcasts_root = Path(settings.PODCASTS_DIR).resolve()

        try:
            relative = target.relative_to(podcasts_root)
        except ValueError:
            logger.error(f"Refusing to {action} outside podcast storage: {target}")
            return False

        if len(relative.parts) != 2 or requested != target:
            logger.error("Refusing to %s an aliased or non-episode path: %s", action, requested)
            return False

        if not target.exists():
            return True
        if not target.is_dir():
            logger.warning(f"Refusing to {action} non-directory episode path: {target}")
            return False

        shutil.rmtree(target)
        logger.info(f"{action.capitalize()} episode directory: {target}")
        return True

    def _remove_subscription_directory(self, subscription_slug: str) -> bool:
        """Idempotently remove one subscription directory within podcast storage."""
        podcasts_root = Path(settings.PODCASTS_DIR).resolve()
        target = (podcasts_root / subscription_slug).resolve()
        try:
            relative = target.relative_to(podcasts_root)
        except ValueError:
            logger.error(f"Refusing to delete subscription outside podcast storage: {target}")
            return False
        if len(relative.parts) != 1 or (podcasts_root / subscription_slug).absolute() != target:
            logger.error(f"Refusing to delete podcast storage root: {target}")
            return False
        if not target.exists():
            return True
        if not target.is_dir():
            logger.warning(f"Refusing to delete non-directory subscription path: {target}")
            return False
        shutil.rmtree(target)
        logger.info(f"Deleted subscription directory: {target}")
        return True

    def _remove_subscription_feed(self, subscription_slug: str) -> bool:
        """Idempotently remove one generated feed within the configured feed directory."""
        feeds_root = Path(settings.FEEDS_DIR).resolve()
        target = (feeds_root / f"{subscription_slug}.xml").resolve()
        try:
            relative = target.relative_to(feeds_root)
        except ValueError:
            logger.error(f"Refusing to delete subscription feed outside feed storage: {target}")
            return False
        if not relative.parts:
            logger.error(f"Refusing to delete feed storage root: {target}")
            return False
        if not target.exists():
            return True
        if not target.is_file():
            logger.warning(f"Refusing to delete non-file subscription feed path: {target}")
            return False
        target.unlink()
        logger.info(f"Deleted subscription feed: {target}")
        return True

    def _remove_file_if_exists(self, path: str, action: str) -> None:
        """Remove a single file and log failures without masking the original error."""
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to remove {action} file {path}: {e}")

    def _cleanup_stale_temporary_files(self, max_age_hours: int = 24) -> int:
        """Remove stale processor temp files that are safe to recreate."""
        podcasts_root = Path(settings.PODCASTS_DIR).resolve()
        if not podcasts_root.exists():
            return 0

        cutoff = datetime.now().timestamp() - (max_age_hours * 60 * 60)
        removed = 0
        for path in podcasts_root.rglob("*"):
            if not path.is_file():
                continue
            if not (path.name.endswith(".part") or path.name.endswith(".tmp.mp3")):
                continue
            try:
                path.relative_to(podcasts_root)
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"Failed to remove stale temporary file {path}: {e}")
            except ValueError:
                logger.warning(f"Refusing to remove temporary file outside podcast storage: {path}")
        return removed

    async def check_feeds(self, subscription_id: int = None, limit: int = 5):
        """Check subscriptions for new episodes."""
        
        if subscription_id:
            sub = self.sub_repo.get_by_id(subscription_id)
            subs = [sub] if sub else []
        else:
            subs = self.sub_repo.get_all()
            
        for sub in subs:
            try:
                source_type = getattr(sub, "source_type", "rss")
                if not isinstance(source_type, str):
                    source_type = "rss"
                source_adapter = get_source_adapter(source_type)
                if source_type in {"youtube_channel", "youtube_playlist"}:
                    await self._check_youtube_source(
                        sub,
                        source_adapter,
                        initial_limit=min(max(int(limit), 0), 5),
                    )
                    continue

                # Use subscription limit if set, else default. 
                # Limit of 0 is valid (means skip initial downloads)
                actual_limit = sub.retention_limit if sub.retention_limit is not None else limit
                logger.info(f"Checking {sub.title} (Sub Limit: {sub.retention_limit}, Ref Limit: {limit}, Final Limit: {actual_limit})...")

                # Fetch ALL episodes
                discovery = await source_adapter.discover(source_type, sub.feed_url)
                episodes = discovery.entries
                
                for i, ep_data in enumerate(episodes):
                    ep_data['subscription_id'] = sub.id
                    
                    # Determine status based on limit
                    should_be_pending = i < actual_limit
                    
                    if should_be_pending:
                        ep_data['status'] = 'pending'
                    else:
                        ep_data['status'] = 'unprocessed'
                        
                    # Try to create. If exists, it returns False.
                    if self.ep_repo.create_or_ignore(ep_data):
                        if should_be_pending:
                            logger.info(f"New episode queued: {ep_data['title']}")
                    else:
                        # Episode exists. Backfill if needed.
                        # If we want it pending, and it's currently unprocessed (or failed), retry it.
                        if should_be_pending:
                            self.ep_repo.update_status_by_guid(
                                sub.id, 
                                ep_data['guid'], 
                                'pending', 
                                condition_status='unprocessed'
                            )
                self.sub_repo.record_check_success(sub.id)
            except Exception as e:
                logger.error(f"Error checking feed {sub.feed_url}: {e}")
                self.sub_repo.record_check_error(sub.id, str(e))

    async def _check_youtube_source(self, sub, source_adapter, initial_limit: int) -> None:
        """Discover bounded public YouTube entries and queue each new item once."""
        is_initial = self.source_item_repo.count(sub.id) == 0
        discovery = await source_adapter.discover(sub.source_type, sub.feed_url)
        if sub.source_type == "youtube_playlist" and not discovery.truncated:
            self.source_item_repo.begin_reconciliation(sub.id)

        eligible_initial = 0
        for flat_entry in discovery.entries:
            external_id = str(flat_entry.get("id") or "").strip()
            if not external_id:
                continue
            canonical_url = f"https://www.youtube.com/watch?v={external_id}"
            observed, is_new = self.source_item_repo.observe(sub.id, external_id, canonical_url)

            if (
                sub.source_type == "youtube_channel"
                and not is_new
                and observed.get("eligibility") == "eligible"
            ):
                break

            should_hydrate = is_new or observed.get("eligibility") in {"unknown", "transient"}
            if not should_hydrate:
                continue
            if "/shorts/" in str(flat_entry.get("url") or ""):
                self.source_item_repo.set_eligibility(sub.id, external_id, "excluded", "short")
                continue
            episode_data, exclusion_reason, transient = await asyncio.to_thread(
                hydrate_youtube_entry,
                external_id,
            )
            if not episode_data:
                self.source_item_repo.set_eligibility(
                    sub.id,
                    external_id,
                    "transient" if transient else "excluded",
                    exclusion_reason,
                )
                continue

            self.source_item_repo.set_eligibility(sub.id, external_id, "eligible")
            episode_data["subscription_id"] = sub.id
            if is_initial:
                should_queue = eligible_initial < initial_limit
                eligible_initial += 1
            else:
                should_queue = is_new or observed.get("eligibility") == "transient"
            episode_data["status"] = "pending" if should_queue else "unprocessed"
            if self.ep_repo.create_or_ignore(episode_data) and should_queue:
                logger.info("New YouTube episode queued: %s", episode_data["title"])

        self.sub_repo.record_check_success(sub.id, truncated=discovery.truncated)

    async def delete_episode(self, episode_id: int):
        """Ignore an episode, wait for its worker, then remove artifacts safely."""
        ep = await asyncio.to_thread(self.ep_repo.get_by_id, episode_id)
        if not ep:
            return False
        
        # Get subscription for slug
        sub = await asyncio.to_thread(self.sub_repo.get_by_id, ep.subscription_id)
        if not sub:
            return False

        await asyncio.to_thread(self.ep_repo.request_deletion, episode_id)

        deadline = asyncio.get_running_loop().time() + self.DELETION_ACK_TIMEOUT_SECONDS
        while await asyncio.to_thread(self.job_repo.is_running_for_episode, episode_id):
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning(
                    f"Timed out waiting for episode {episode_id} worker cancellation; "
                    "the worker will clean up at its next checkpoint"
                )
                break
            await asyncio.sleep(self.DELETION_POLL_INTERVAL_SECONDS)

        if not await asyncio.to_thread(self.job_repo.is_running_for_episode, episode_id):
            await asyncio.to_thread(self._finalize_episode_deletion, episode_id)

        await asyncio.to_thread(self.rss_gen.generate_feed, sub.id)
        await asyncio.to_thread(self.rss_gen.generate_unified_feed)
        return True

    def _finalize_episode_deletion(self, episode_id: int):
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT e.*, s.slug FROM episodes e JOIN subscriptions s ON s.id=e.subscription_id WHERE e.id=? AND e.status='ignored'", (episode_id,)).fetchone()
            running = conn.execute("SELECT 1 FROM jobs WHERE episode_id=? AND status='running'", (episode_id,)).fetchone()
            if not row or running:
                return
            self._remove_episode_directory(str(episode_directory(row['slug'], episode_id)), 'delete')
            legacy = legacy_directory(row['slug'], row['guid'])
            # Legacy GUID sanitization was lossy. Never delete a directory another
            # episode could be using, even when this episode was explicitly deleted.
            if legacy:
                others = conn.execute('SELECT guid FROM episodes WHERE subscription_id=? AND id!=?', (row['subscription_id'], episode_id)).fetchall()
                if not any(legacy_directory(row['slug'], other['guid']) == legacy for other in others):
                    self._remove_episode_directory(str(legacy), 'delete')
            conn.execute("UPDATE episodes SET processing_step='deleted', file_size=0, publication_pending=1 WHERE id=?", (episode_id,))
            conn.commit()

    def _finalize_subscription_deletion(self, subscription_id: int) -> str:
        """Run one claimed, retryable subscription cleanup outside the web event loop."""
        sub = self.sub_repo.claim_deletion_cleanup(subscription_id)
        if not sub:
            return "deletion_pending"

        try:
            if not self._remove_subscription_directory(sub["slug"]):
                raise RuntimeError("subscription directory failed path-containment validation")
            if not self._remove_subscription_feed(sub["slug"]):
                raise RuntimeError("subscription feed failed path-containment validation")
            from app.core.artwork import ArtworkWatermarker
            ArtworkWatermarker().clear(subscription_id)
            self.rss_gen.generate_unified_feed()
            self.sub_repo.delete(subscription_id)
            logger.info(f"Completed deletion of subscription {subscription_id} ({sub['title']})")
            return "deleted"
        except Exception as e:
            logger.error(f"Subscription {subscription_id} cleanup failed and will be retried: {e}")
            self.sub_repo.mark_deletion_failed(subscription_id, str(e))
            return "deletion_pending"

    async def delete_subscription(self, subscription_id: int) -> str:
        """Atomically cancel a subscription, wait briefly for workers, and clean it up."""
        sub = await asyncio.to_thread(self.sub_repo.begin_deletion, subscription_id)
        if not sub:
            return "deleted"

        deadline = asyncio.get_running_loop().time() + self.DELETION_ACK_TIMEOUT_SECONDS
        while await asyncio.to_thread(self.sub_repo.count_running_deletion_jobs, subscription_id):
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning(
                    f"Subscription {subscription_id} deletion is pending worker acknowledgement; "
                    "cleanup will be retried by the processor loop"
                )
                return "deletion_pending"
            await asyncio.sleep(self.DELETION_POLL_INTERVAL_SECONDS)

        return await asyncio.to_thread(self._finalize_subscription_deletion, subscription_id)

    async def finalize_pending_subscription_deletions(self) -> int:
        """Retry safe cleanup for deletions whose workers have exited."""
        subscription_ids = await asyncio.to_thread(self.sub_repo.list_pending_deletions)
        completed = 0
        for subscription_id in subscription_ids:
            if await asyncio.to_thread(self.sub_repo.count_running_deletion_jobs, subscription_id):
                continue
            result = await asyncio.to_thread(self._finalize_subscription_deletion, subscription_id)
            if result == "deleted":
                completed += 1
        return completed

    async def version_episode(self, episode_id: int):
        """Reprocessing keeps the published GUID and paths until successful replacement."""
        return self.ep_repo.get_by_id(episode_id) is not None

    def _extract_text(self, start: float, end: float, segments: list) -> str:
        """Extract text from transcript overlapping with the given time range."""
        text = []
        for seg in segments:
            if seg['start'] < end and seg['end'] > start:
                text.append(seg['text'])
        return " ".join(text).strip()

    @staticmethod
    def _normalize_segment(segment: dict, total_duration: float | None = None) -> dict | None:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Skipping malformed segment: {segment}")
            return None

        if total_duration is not None:
            start = max(0.0, min(total_duration, start))
            end = max(0.0, min(total_duration, end))

        if end <= start:
            logger.warning(f"Skipping empty segment: {segment}")
            return None

        normalized = segment.copy()
        normalized["start"] = start
        normalized["end"] = end
        if normalized.get("evidence"):
            normalized["sources"] = sorted({
                str(item.get("source") or "unknown")
                for item in normalized["evidence"]
                if isinstance(item, dict)
            })
        return normalized

    @staticmethod
    def _merge_remove_segments(segments: list[dict], merge_gap: float = 10.0) -> list[dict]:
        normalized_segments = [
            normalized
            for segment in segments
            if (normalized := Processor._normalize_segment(segment)) is not None
        ]

        merged_segments = []
        for segment in sorted(normalized_segments, key=lambda item: item["start"]):
            if merged_segments and segment["start"] - merged_segments[-1]["end"] < merge_gap:
                old_end = merged_segments[-1]["end"]
                merged_segments[-1]["end"] = max(merged_segments[-1]["end"], segment["end"])
                if merged_segments[-1].get("evidence") or segment.get("evidence"):
                    existing_evidence = merged_segments[-1].setdefault("evidence", [])
                    for evidence in segment.get("evidence", []):
                        if evidence not in existing_evidence:
                            existing_evidence.append(evidence)
                    merged_segments[-1]["sources"] = sorted({
                        str(item.get("source") or "unknown")
                        for item in existing_evidence
                        if isinstance(item, dict)
                    })
                logger.info(
                    f"Merged segment {segment['start']}-{segment['end']} into "
                    f"{merged_segments[-1]['start']}-{merged_segments[-1]['end']}"
                )
                if merged_segments[-1]["end"] == old_end:
                    logger.info("Contained segment did not extend the merged remove window")
            else:
                merged_segments.append(segment.copy())

        return merged_segments

    @staticmethod
    def _invert_content_segments(content_segments: list[dict], total_duration: float) -> list[dict]:
        normalized_content = [
            normalized
            for segment in content_segments
            if (normalized := Processor._normalize_segment(segment, total_duration=total_duration)) is not None
        ]
        inverted_remove = []
        current_time = 0.0

        for content in sorted(normalized_content, key=lambda item: item["start"]):
            if content["start"] > current_time:
                inverted_remove.append({
                    "start": current_time,
                    "end": content["start"],
                    "label": "Non-Content",
                    "reason": "Not labeled as content (whitelist mode)",
                })
            current_time = max(current_time, content["end"])

        if current_time < total_duration:
            inverted_remove.append({
                "start": current_time,
                "end": total_duration,
                "label": "Non-Content",
                "reason": "Trailing non-content (whitelist mode)",
            })

        return inverted_remove

    @staticmethod
    def _prepare_remove_segments(
        ad_segments: list[dict],
        whitelist_mode: bool,
        total_duration: float | None = None,
    ) -> list[dict]:
        if whitelist_mode:
            content_segments = [s for s in ad_segments if s.get("label") == "Content"]
            non_content_segments = [s for s in ad_segments if s.get("label") != "Content"]

            if not content_segments:
                logger.warning("Whitelist mode: No Content segments found! Falling back to blacklist mode.")
                ad_segments = non_content_segments
            elif total_duration and total_duration > 0:
                logger.info(
                    f"Whitelist mode: {len(content_segments)} Content segments, "
                    f"{len(non_content_segments)} non-content segments"
                )
                ad_segments = Processor._invert_content_segments(content_segments, total_duration)
                logger.info(f"Whitelist mode: inverted to {len(ad_segments)} remove segments")
            else:
                logger.warning("Whitelist mode: Could not determine duration; keeping episode uncut.")
                ad_segments = []

        return Processor._merge_remove_segments(ad_segments)
    
    async def regenerate_all_feeds(self):
        """Regenerate all RSS feeds to ensure they use current base URL."""
        logger.info("Regenerating all RSS feeds...")
        try:
            subs = self.sub_repo.get_all()
            for sub in subs:
                self.rss_gen.generate_feed(sub.id)
            self.rss_gen.generate_unified_feed()
            logger.info(f"Successfully regenerated {len(subs) + 1} feeds.")
        except Exception as e:
            logger.error(f"Failed to regenerate feeds: {e}")

    async def process_queue(self):
        """Process pending episodes concurrently up to the configured limit."""
        if settings.PROCESSOR_ENABLED and not Processor._is_background_worker:
            return
        if not settings.PROCESSOR_ENABLED and Processor._manual_processor is None:
            Processor._manual_processor = self
        if not settings.PROCESSOR_ENABLED and Processor._manual_processor is not self:
            return await Processor._manual_processor.process_queue()
        # Use lock to prevent race conditions from multiple callers WITHIN this process
        async with Processor._queue_lock:
            # 1. Fetch limit from settings
            from app.core.utils import get_global_settings
            db_settings = get_global_settings()
            limit = db_settings.get('concurrent_downloads', 2)
            if not limit or limit < 1: limit = 2

            repaired_jobs = self.job_repo.repair_missing_active_jobs()
            if repaired_jobs:
                logger.warning(f"Recreated {repaired_jobs} missing processing job(s) for queued episode(s)")

            recovered_jobs = self.job_repo.recover_stale_running()
            if recovered_jobs:
                logger.warning(f"Recovered or cleared {recovered_jobs} stalled processing job(s)")

            # 2. Check if we have room using DATABASE count (cross-process safe)
            currently_processing = self.job_repo.count_running()
            if currently_processing >= limit:
                return

            # 3. Claim jobs in one SQLite transaction, then launch them.
            capacity = limit - currently_processing
            claimed = self.job_repo.claim_due(capacity, max_running=limit)
            if not claimed:
                return

            for ep_dict in claimed:
                ep_id = ep_dict['id']
                
                # Skip if already in our in-process tracking (for this process)
                if ep_id in Processor._active_task_ids:
                    continue
                    
                # Add to in-process set and launch
                Processor._active_task_ids.add(ep_id)
                
                # Start background task
                asyncio.create_task(self._process_single_episode_task(ep_dict))

    async def _process_single_episode_task(self, ep_dict: dict):
        """Wrapper to manage active task state and call the actual processor."""
        ep_id = ep_dict['id']
        try:
            # Re-validate episode and subscription
            ep = Episode.model_validate(ep_dict)
            sub = self.sub_repo.get_by_id(ep.subscription_id)
            if not sub:
                logger.error(f"Subscription {ep.subscription_id} not found for episode {ep.id}")
                self.job_repo.fail_running_for_episode(ep_id, "Subscription not found")
                await send_notification_async(
                    EVENT_BREAKING_ERROR,
                    "Episode processing failed",
                    f"Episode {ep.id} could not be processed because its podcast no longer exists.",
                    severity="error",
                )
                return
            if not sub.is_active or sub.deletion_status is not None:
                logger.info(f"Skipping episode {ep.id}; subscription deletion is in progress")
                self.job_repo.cancel_active_for_episode(ep_id)
                return

            # Actually run the processing
            worker = copy.copy(self)
            worker.ad_detector = copy.copy(self.ad_detector)
            worker.ep_repo = EpisodeRepository(attempt=(ep_dict['job_id'], ep_dict['claim_token']))
            heartbeat = asyncio.create_task(self._heartbeat_claim(ep_dict))
            budget_token = current_claim.set((ep_dict['job_id'], ep_dict['claim_token']))
            try:
                await worker._process_episode_inner(ep, sub, ep_dict)
            finally:
                current_claim.reset(budget_token)
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"Fatal error in episode task {ep_id}: {e}")
            await send_notification_async(
                EVENT_BREAKING_ERROR,
                "Episode task crashed",
                f"Episode task {ep_id} crashed before it could finish: {e}",
                severity="error",
            )
        finally:
            self.job_repo.acknowledge(ep_dict['job_id'], ep_dict['claim_token'])
            await asyncio.to_thread(self._finalize_episode_deletion, ep_id)
            # Ensure ID is removed from active set
            Processor._active_task_ids.discard(ep_id)
            try:
                from app.core.utils import get_global_settings
                global_settings = get_global_settings()
                if (
                    not Processor._active_task_ids
                    and global_settings.get("unload_whisper_after_job")
                    and self.job_repo.count_claimable() == 0
                ):
                    self.transcriber.unload_model()
            except Exception as e:
                logger.warning(f"Failed to unload Whisper model after job: {e}")
            # Proactively check queue again after finishing to keep pipeline full
            asyncio.create_task(self.process_queue())

    async def _heartbeat_claim(self, claim):
        while True:
            await asyncio.to_thread(self.job_repo.heartbeat, claim['job_id'], claim['claim_token'])
            await asyncio.sleep(20)

    async def _fetch_sponsorblock_segments(self, sub, ep: Episode) -> list[dict]:
        if not settings.SPONSORBLOCK_ENABLED:
            return []
        if sub.source_type not in {"youtube_channel", "youtube_playlist"}:
            return []
        categories = categories_for_subscription(sub)
        video_id = video_id_from_url(ep.original_url)
        if not categories or not video_id:
            return []
        return await asyncio.to_thread(
            self.sponsorblock.fetch_segments,
            video_id,
            categories,
        )

    async def _classify_complete_timeline(self, ep, sub, transcript, input_path, fingerprint, snapshot, episode_root):
        from app.core import timeline
        duration = await asyncio.to_thread(AudioProcessor.get_duration, input_path)
        units, notes = timeline.prepare_timeline(transcript, duration)
        metadata = {'podcast_name': sub.title, 'episode_title': ep.title,
                    'publication_date': str(ep.pub_date) if ep.pub_date else None}
        key = timeline.cache_key(fingerprint, transcript, duration, snapshot)
        cache_path = self._attempt_dir / 'analysis-cache.json'
        cache_paths = [cache_path]
        # A reprocess may reuse classification only if the newly acquired source
        # and transcript still match. Dynamic ad insertion changes the fingerprint.
        if ep.ad_report_path:
            previous = Path(ep.ad_report_path).resolve().parent
            if previous.is_relative_to(episode_root.resolve()):
                cache_paths.append(previous / 'analysis-cache.json')
        analysis = None
        for path in cache_paths:
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
                if cached['key'] != key:
                    continue
                candidate = cached['analysis']
                rows, summary = timeline.parse_response(json.dumps(candidate['response']), units)
                if summary is not None and not timeline.valid_summary(summary):
                    continue
                analysis = {**candidate, 'segments': rows, 'summary': summary}
                if summary is None:
                    summary, error = await asyncio.to_thread(
                        self.ad_detector.repair_cached_timeline_summary, units, duration, metadata, snapshot,
                    )
                    analysis.update(summary=summary, summary_error=error)
                    analysis['response']['summary'] = summary
                break
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if analysis is None:
            analysis = await asyncio.to_thread(
                self.ad_detector.classify_timeline, units, duration,
                metadata, snapshot,
            )
        cache_path.write_text(json.dumps({'key': key, 'analysis': analysis}), encoding='utf-8')
        analysis = {**analysis, 'timeline': units, 'normalization_notes': notes,
                    'duration': duration, 'source_fingerprint': fingerprint, 'cache_key': key}
        for row in analysis['segments']:
            row['text'] = self._extract_text(row['start'], row['end'], transcript['segments'])
        return analysis

    async def _process_episode_inner(self, ep: Episode, sub, ep_dict: dict):
        """Core multi-step processing logic for a single episode."""
        logger.info(f"Processing {ep.title}...")
        from app.core.utils import get_global_settings
        global_settings = get_global_settings()
        ffmpeg_threads = int(global_settings.get("ffmpeg_threads") or 0)
        
        try:
            from app.core import timeline
            snapshot = json.loads(ep_dict['processing_snapshot']) if ep_dict.get('processing_snapshot') else {'version': 1, 'workflow': 'legacy'}
            if not isinstance(snapshot, dict) or snapshot.get('version') != 1 or snapshot.get('workflow') not in timeline.WORKFLOWS:
                raise PermanentProviderError('Unknown queued processing workflow; requeue with supported settings')
            complete_timeline = snapshot['workflow'] == 'complete_timeline'
            if complete_timeline:
                if snapshot.get('schema_version') != timeline.SCHEMA_VERSION:
                    raise PermanentProviderError('Queued timeline schema is not supported by this application version')
                sub = sub.model_copy(update={**snapshot['options'], 'processing_workflow': 'complete_timeline'})
            analysis, edit_policy = None, None
            self.ep_repo.update_progress(ep.id, "Actively Processing", 0)
            
            if not self._check_cancellation(ep): return

            self.ep_repo.update_progress(ep.id, "processing", 10)
            
            # Check for skip flags
            skip_transcription = False
            if ep.processing_flags:
                try:
                    logger.info(f"Checking processing flags for {ep.title}: {ep.processing_flags}")
                    flags = json.loads(ep.processing_flags)
                    skip_transcription = flags.get('skip_transcription', False)
                    if skip_transcription:
                        logger.info(f"Targeting skip_transcription for {ep.title}")
                except Exception as e:
                    logger.error(f"Failed to parse processing flags: {e}")
                
            transcript = None
            
            # Create episode-specific directory
            episode_root = episode_directory(sub.slug, ep.id)
            episode_dir = str(episode_root / ('attempt-' + uuid4().hex))
            self._attempt_dir = Path(episode_dir)
            self._attempt_dir.mkdir(parents=True, exist_ok=False)
            
            input_path = os.path.join(episode_dir, "original.mp3")
            transcript_path = None
            resume = Path(ep_dict['resume_directory']) if ep_dict.get('resume_directory') else None
            if resume and resume.resolve() == resume and resume.is_relative_to(episode_root) and resume.is_dir():
                # Copy only finalized reusable inputs into the new owned stage. A stale
                # worker can never mutate the files used by its successor.
                manifest_path = resume / 'cache.json'
                try:
                    cache = json.loads(manifest_path.read_text(encoding='utf-8'))
                    cached_source = resume / cache['source_file']
                    if cached_source.parent == resume and cached_source.is_file():
                        input_path = str(self._attempt_dir / cached_source.name)
                        await asyncio.to_thread(shutil.copyfile, cached_source, input_path)
                        if await asyncio.to_thread(source_fingerprint, input_path) == cache['sha256']:
                            for name in ('transcript.json', 'analysis-cache.json'):
                                if (resume / name).is_file():
                                    await asyncio.to_thread(shutil.copyfile, resume / name, self._attempt_dir / name)
                            if (self._attempt_dir / 'transcript.json').is_file():
                                ep.transcript_path = str(self._attempt_dir / 'transcript.json')
                                skip_transcription = True
                except (OSError, ValueError, KeyError):
                    logger.info('No reusable verified source cache for this retry')
            if self.ep_repo.attempt:
                from app.infra.database import get_db_connection
                with self.ep_repo._write_connection(ep.id) as conn:
                    conn.execute('UPDATE jobs SET work_directory=? WHERE id=? AND locked_by=?', (episode_dir, *self.ep_repo.attempt))
                    conn.commit()
            
            if skip_transcription and ep.transcript_path and os.path.exists(ep.transcript_path):
                 logger.info(f"Attempting to skip transcription, using existing: {ep.transcript_path}")
                 transcript_path = ep.transcript_path
                 import ast
                 async with aiofiles.open(transcript_path, "r", encoding="utf-8") as f:
                     content = await f.read()
                     # Handle both JSON and Python dict string (legacy)
                     try:
                         transcript = json.loads(content)
                         logger.info(f"Successfully loaded JSON transcript for {ep.title}")
                     except:
                         try:
                             transcript = ast.literal_eval(content)
                             logger.info(f"Successfully loaded legacy dict transcript for {ep.title}")
                         except Exception as e:
                             logger.error(f"Failed to load transcript for {ep.title}: {e}")
                             # Fallback to re-transcribe if load fails
                             skip_transcription = False
                             transcript = None
            elif skip_transcription:
                logger.warning(f"skip_transcription requested for {ep.title} but transcript_path missing or file not found: {ep.transcript_path}")
                skip_transcription = False
            # 1. Ensure Audio Exists (Download if missing)
            if not os.path.exists(input_path):
                if not self._check_cancellation(ep): return
                logger.info(f"Downloading {ep.title}...")
                source_adapter = get_source_adapter(sub.source_type)
                try:
                    input_path = await source_adapter.download(
                        ep.original_url,
                        episode_dir,
                        progress_callback=lambda percent: self.ep_repo.update_progress(
                            ep.id, "downloading", percent
                        ),
                        cancellation_callback=lambda: not self._check_cancellation(ep),
                    )
                except YouTubeDownloadCancelled:
                    self._acknowledge_cancellation(ep)
                    return
                if sub.source_type in {"youtube_channel", "youtube_playlist"}:
                    self.ep_repo.update_source_media_path(ep.id, input_path)

                file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
                logger.info(f"Download complete: {file_size_mb:.2f} MB")
            
            fingerprint = await asyncio.to_thread(source_fingerprint, input_path)
            if transcript:
                provenance = transcript.get('_source', {}) if isinstance(transcript, dict) else {}
                if provenance.get('sha256') != fingerprint or provenance.get('whisper_model') != global_settings.get('whisper_model', settings.WHISPER_MODEL):
                    logger.info("Source or transcription settings changed; transcribing again")
                    transcript = None
            self.ep_repo.update_source_media_path(ep.id, input_path)
            (self._attempt_dir / 'cache.json').write_text(json.dumps({'source_file': Path(input_path).name, 'sha256': fingerprint}), encoding='utf-8')

            await asyncio.to_thread(require_scratch, ep.duration, os.path.getsize(input_path))
            # 2. Transcribe (If needed)
            if not transcript:
                self.ep_repo.update_progress(ep.id, "transcribing", 0)
                            
                start_time = datetime.now()
                logger.info(f"Starting transcription for {ep.title}...")
                
                # Shared state for callback to check
                cancellation_state = {'last_check': datetime.now(), 'is_cancelled': False}
                
                def transcribe_progress(current, total):
                    def format_time(seconds):
                        m, s = divmod(int(seconds), 60)
                        h, m = divmod(m, 60)
                        if h > 0:
                            return f"{h}:{m:02d}:{s:02d}"
                        return f"{m}:{s:02d}"
                    
                    # Check for cancellation every 2 seconds
                    if (datetime.now() - cancellation_state['last_check']).total_seconds() > 2.0:
                        cancellation_state['last_check'] = datetime.now()
                        # We need to check DB status. 
                        # Since this runs in a thread, we use a new connection or the repo method if it handles it.
                        # Repo methods open fresh connections so they are thread-safe.
                        status = self.ep_repo.get_status(ep.id)
                        if status != 'processing':
                            cancellation_state['is_cancelled'] = True
                            raise Exception("CancelledByUser")

                    percent = int((current / total) * 100) if total > 0 else 0
                    
                    remaining_str = ""
                    if current > 0 and total > 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed > 5: # Give it a few seconds to stabilize
                            speed = current / elapsed
                            remaining_secs = (total - current) / speed
                            remaining_str = f", ~{format_time(remaining_secs)} left"
                    
                    step = f"transcribing ({format_time(current)} / {format_time(total)}{remaining_str})"
                    self.ep_repo.update_progress(ep.id, step, percent)

                try:
                    transcript = await asyncio.to_thread(
                        self.transcriber.transcribe, input_path, progress_callback=transcribe_progress
                    )
                except Exception as e:
                    # Catch cancellation exception from thread
                    if "CancelledByUser" in str(e) or cancellation_state['is_cancelled']:
                        logger.warning(f"Transcription cancelled for {ep.title}")
                        self._acknowledge_cancellation(ep)
                        return # Stop processing this episode
                    raise e
                
                if not self._check_cancellation(ep): return # Check after transcribe
                
                # Double check state
                if cancellation_state['is_cancelled']:
                     self._acknowledge_cancellation(ep)
                     return

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Transcription complete in {duration:.1f}s")
                
                transcript['_source'] = {'sha256': fingerprint, 'whisper_model': global_settings.get('whisper_model', settings.WHISPER_MODEL)}
                # Save Transcript (Prefer JSON now)
                transcript_path = os.path.join(episode_dir, "transcript.json")
                async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(transcript))
                
            if transcript_path != os.path.join(episode_dir, 'transcript.json'):
                transcript_path = os.path.join(episode_dir, 'transcript.json')
                async with aiofiles.open(transcript_path, 'w', encoding='utf-8') as handle:
                    await handle.write(json.dumps(transcript))
            self.ep_repo.update_progress(ep.id, "detecting_ads", 50, transcript_path=transcript_path)
            
            if not self._check_cancellation(ep): return

            # 3. Classification is independent of cut preferences in the opt-in workflow.
            if complete_timeline:
                analysis = await self._classify_complete_timeline(
                    ep, sub, transcript, input_path, fingerprint, snapshot, episode_root
                )
                if not self._check_cancellation(ep): return
                sponsor_segments = await self._fetch_sponsorblock_segments(sub, ep)
                sponsor_segments = [normalized for s in sponsor_segments
                                    if (normalized := self._normalize_segment(s, analysis['duration'])) is not None]
                edit_policy = timeline.apply_preferences(analysis['segments'], snapshot['options'], sponsor_segments)
                ad_segments = edit_policy['segments']
            else:
                # 3. Detect Ads
                logger.info("Detecting ads...")
            
                detect_options = {
                    "remove_ads": sub.remove_ads,
                    "remove_promos": sub.remove_promos,
                    "remove_intros": sub.remove_intros,
                    "remove_outros": sub.remove_outros,
                    "custom_instructions": sub.custom_instructions
                }
            
                # Check whitelist mode from global settings
                whitelist_mode = bool(global_settings.get('whitelist_mode', 0))
            
                if whitelist_mode:
                    logger.info("Whitelist mode is ENABLED - will keep only Content segments")
            
                import hashlib
                policy = {key: value for key, value in global_settings.items() if key.startswith(('ad_', 'custom_llm_model', 'custom_llm_base_url')) or key in ('active_ai_provider', 'ai_model_cascade', 'openai_model', 'anthropic_model', 'openrouter_model')}
                cache_key = hashlib.sha256(json.dumps([fingerprint, transcript, detect_options, whitelist_mode, policy], sort_keys=True).encode()).hexdigest()
                analysis_cache_path = self._attempt_dir / 'analysis-cache.json'
                ad_segments = None
                try:
                    cached = json.loads(analysis_cache_path.read_text(encoding='utf-8'))
                    if cached['key'] == cache_key:
                        ad_segments = cached['segments']
                except (OSError, ValueError, KeyError):
                    pass
                if ad_segments is None:
                    ad_segments = await asyncio.to_thread(
                        self.ad_detector.detect_ads, transcript, detect_options, whitelist_mode=whitelist_mode
                    )
                    analysis_cache_path.write_text(json.dumps({'key': cache_key, 'segments': ad_segments}), encoding='utf-8')

                if not self._check_cancellation(ep): return

                logger.info(f"Found {len(ad_segments)} segments: {ad_segments}")

                total_duration = AudioProcessor.get_duration(input_path) if whitelist_mode else None
                ad_segments = self._prepare_remove_segments(ad_segments, whitelist_mode, total_duration=total_duration)
                for segment in ad_segments:
                    segment.setdefault("source", "llm")
                    segment.setdefault("evidence", [{
                        "source": "llm",
                        "label": segment.get("label"),
                        "reason": segment.get("reason"),
                    }])
                    segment.setdefault("sources", ["llm"])
                sponsor_segments = await self._fetch_sponsorblock_segments(sub, ep)
                if sponsor_segments:
                    if total_duration is None:
                        total_duration = AudioProcessor.get_duration(input_path)
                    sponsor_segments = [
                        normalized
                        for segment in sponsor_segments
                        if (normalized := self._normalize_segment(segment, total_duration)) is not None
                    ]
                    ad_segments = self._merge_remove_segments(ad_segments + sponsor_segments)
                    logger.info("Added %s SponsorBlock segments", len(sponsor_segments))
                logger.info(f"After merging: {len(ad_segments)} ad segments")
            
            # Enrich with Text
            for s in ad_segments:
                s['text'] = self._extract_text(s['start'], s['end'], transcript['segments'])

            # Save Ad Report (JSON)
            report_path = os.path.join(episode_dir, "report.json")
            report_data = {
                "episode_id": ep.id,
                "guid": ep.guid,
                "segments": ad_segments,
                "transcript_path": transcript_path
            }
            if complete_timeline:
                report_data.update(workflow='complete_timeline', analysis=analysis, edit_policy=edit_policy)
            async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(report_data, indent=2))

            # Generate Human-Readable Report (HTML)
            human_report_path = os.path.join(episode_dir, "report.html")
            
            html_content = render_ad_report(ep, ad_segments, analysis=analysis, edit_policy=edit_policy) if complete_timeline else render_ad_report(ep, ad_segments)

            async with aiofiles.open(human_report_path, "w", encoding="utf-8") as f:
                await f.write(html_content)

            self.ep_repo.update_progress(ep.id, "removing_ads", 75, ad_report_path=report_path, report_path=human_report_path)
            
            if not self._check_cancellation(ep): return

            # 4. Remove Ads
            output_path = os.path.join(episode_dir, "processed.mp3")
            
            await asyncio.to_thread(require_scratch, ep.duration, os.path.getsize(input_path))
            logger.info("Removing ads with FFmpeg...")
            tone_options = {position: bool(global_settings.get(f'warning_tone_{position}'))
                            for position in ('start', 'middle', 'end')}
            for position in ('start', 'middle', 'end'):
                tone_options[f'{position}_style'] = global_settings.get(f'warning_tone_{position}_style') or 'soft'
            audio_options = {'warning_tones': tone_options} if any(
                tone_options[position] for position in ('start', 'middle', 'end')) else {}

            await asyncio.to_thread(
                AudioProcessor.remove_segments, 
                input_path, 
                output_path, 
                ad_segments,
                ffmpeg_threads=ffmpeg_threads,
                **audio_options,
            )
            logger.info(f"Saved cleaned audio to {output_path}")
            
            # 4.5 Generate & Append Summary (If enabled)
            
            if not self._check_cancellation(ep): return

            # 4.5 Generate Intros (Title Intro & Summary)
            intro_files = []
            temp_clean_path = None
            try:
                self.ep_repo.update_progress(ep.id, "generating_intros", 90)

                # A. Title Intro
                if sub.append_title_intro:
                    try:
                        logger.info("Generating Title Intro...")
                        date_str = ep.pub_date.strftime('%B %d, %Y') if ep.pub_date else "recently"
                        p_title = sub.title or "Podcast"
                        e_title = ep.title or "Episode"
                        intro_text = f"You're listening to {p_title} from {date_str}, {e_title}"
                        intro_path = os.path.join(episode_dir, "title_intro.mp3")
                        
                        await self.ad_detector.generate_audio(intro_text, intro_path)
                        intro_files.append(intro_path)
                    except Exception as e:
                        logger.error(f"Failed to generate Title Intro: {e}")

                # B. AI Summary Features
                do_text = sub.ai_rewrite_description or sub.append_summary
                do_audio = sub.ai_audio_summary or sub.append_summary
                
                if complete_timeline:
                    # Keep the combined summary in the report; ai_summary controls RSS
                    # descriptions in existing feed readers, so populate it only on opt-in.
                    self.ep_repo.update_ai_summary(ep.id, analysis['summary'] if do_text else None)
                if do_text or do_audio:
                    summary_text = analysis['summary'] if complete_timeline else None
                    try:
                        logger.info(f"Generating episode summary for {ep.title}...")
                        # Build subscription settings dict for targets
                        sub_settings = {
                            'remove_ads': sub.remove_ads,
                            'remove_promos': sub.remove_promos,
                            'remove_intros': sub.remove_intros,
                            'remove_outros': sub.remove_outros
                        }
                        if not complete_timeline:
                            summary_text = await asyncio.to_thread(
                                self.ad_detector.generate_summary,
                                transcript,
                                sub.title or "Podcast",
                                ep.title,
                                str(ep.pub_date) if ep.pub_date else "recently",
                                sub_settings
                            )
                        elif not summary_text:
                            raise AnalysisError(analysis.get('summary_error') or 'Combined summary unavailable')
                        # Save to DB and file immediately
                        if not complete_timeline or do_text:
                            self.ep_repo.update_ai_summary(ep.id, summary_text)
                        summary_txt_path = os.path.join(episode_dir, "summary.txt")
                        async with aiofiles.open(summary_txt_path, "w") as f:
                            await f.write(summary_text)
                    except Exception as e:
                        logger.error(f"Failed to generate/save text summary: {e}")
                        if not summary_text and not complete_timeline:
                            summary_text = f"Welcome to {sub.title}. Today's episode is {ep.title}."

                    # Audio Summary (TTS)
                    if do_audio and summary_text:
                        try:
                            logger.info("Generating AI Audio Summary (TTS)...")
                            summary_path = os.path.join(episode_dir, "summary.mp3")
                            await self.ad_detector.validate_tts()
                            await self.ad_detector.generate_audio(summary_text, summary_path)
                            intro_files.append(summary_path)
                        except Exception as e:
                            logger.error(f"Failed to generate Audio Summary: {e}")
                
                # C. Prepend Intros to Episode
                if intro_files:
                    # Rename current output to use as input
                    temp_clean_path = output_path + ".tmp.mp3"
                    if os.path.exists(output_path):
                        os.rename(output_path, temp_clean_path)
                        
                        # Combine: [Intro, Summary, Episode]
                        concat_list = intro_files + [temp_clean_path]
                        
                        await asyncio.to_thread(
                            AudioProcessor.concat_files,
                            output_path,
                            concat_list,
                            ffmpeg_threads=ffmpeg_threads,
                        )
                        
                        # Cleanup
                        os.remove(temp_clean_path)
                        for f in intro_files:
                            if os.path.exists(f): 
                                os.remove(f)
                        logger.info("Intros prepended successfully.")
                        
            except Exception as e:
                logger.error(f"Failed to append intros: {e}")
                # Restore original if things failed and we moved it
                if temp_clean_path and os.path.exists(temp_clean_path) and not os.path.exists(output_path):
                    os.rename(temp_clean_path, output_path)
            
            if not self._check_cancellation(ep): return

            # Validate before committing the active pointer. Files keep their immutable
            # attempt directory name, so old podcast-client URLs remain available.
            output_duration = await asyncio.to_thread(AudioProcessor.get_duration, output_path)
            if output_duration <= 0:
                raise RuntimeError('Processed audio failed duration validation')
            self.ep_repo.pending_metadata['output_duration'] = output_duration
            if ep.local_filename:
                self.ep_repo.pending_metadata['published_guid'] = f'{ep.guid}#revision-{uuid4().hex}'
            file_size = os.path.getsize(output_path)
            (self._attempt_dir / 'published.json').write_text(json.dumps({'episode_id': ep.id}), encoding='utf-8')
            self.ep_repo.update_status(ep.id, 'completed', filename=output_path, file_size=file_size)
            self._attempt_dir = None  # Published files are never cancellation cleanup.
            self._remove_file_if_exists(input_path, 'source audio')
            # Feed publication has its own durable retry flag; failure keeps audio.
            await self.publish_pending_feeds()

            logger.info(f"Successfully processed {ep.title}")
            await send_notification_async(
                EVENT_EPISODE_DOWNLOAD,
                "Episode available",
                f"{ep.title} from {sub.title} finished processing and is available in the feed.",
                severity="success",
            )

        except StaleAttempt:
            self._acknowledge_cancellation(ep)
            return
        except (PermanentProviderError, ProviderBudgetExceeded) as e:
            if self.ep_repo.owns_attempt(ep.id):
                self.ep_repo.update_status(ep.id, 'failed', error=str(e))
            else:
                self._acknowledge_cancellation(ep)
            return
        except RateLimitError as e:
            if not self.ep_repo.owns_attempt(ep.id):
                logger.info(f"Discarding rate-limit result for cancelled episode {ep.id}")
                self._acknowledge_cancellation(ep)
                return
            # Special handling for API rate limits - wait until quota resets
            logger.warning(f"Rate limit hit for episode {ep.id}: {e}")
            next_retry = e.get_next_retry_time()
            retry_time_str = next_retry.strftime('%Y-%m-%d %H:%M:%S UTC')
            logger.info(f"Episode {ep.title} placed on hold until API quota resets at {retry_time_str}")
            self.ep_repo.update_rate_limited(ep.id, next_retry, str(e))
            
        except Exception as e:
            if not self.ep_repo.owns_attempt(ep.id):
                logger.info(f"Episode {ep.id} stopped because cancellation was requested")
                self._acknowledge_cancellation(ep)
                return

            logger.error(f"Failed to process episode {ep.id}: {e}")
            
            # Check if this might be a rate limit we didn't catch
            error_str = str(e).lower()
            rate_limit_patterns = ['resource_exhausted', 'quota', 'rate limit', '429', 'too many requests']
            if any(pattern in error_str for pattern in rate_limit_patterns):
                # Treat as rate limit
                logger.warning(f"Detected possible rate limit in error: {e}")
                rate_error = RateLimitError(str(e), is_daily_limit=False, provider="unknown")
                next_retry = rate_error.get_next_retry_time()
                self.ep_repo.update_rate_limited(ep.id, next_retry, str(e))
                return
            
            # Regular Retry Logic
            retry_count = ep_dict.get('retry_count', 0) + 1
            if retry_count <= 5:
                # Exponential backoff: 5, 10, 20, 40, 80 minutes
                delay_minutes = 5 * (2 ** (retry_count - 1))
                from datetime import timedelta
                next_retry = now_utc() + timedelta(minutes=delay_minutes)
                
                logger.info(f"Scheduling retry {retry_count}/5 for {ep.title} in {delay_minutes} minutes")
                self.ep_repo.update_retry(ep.id, retry_count, next_retry, str(e))
            else:
                logger.error(f"Max retries reached for {ep.title}")
                self.ep_repo.update_status(ep.id, "failed", error=str(e))
                await send_notification_async(
                    EVENT_BREAKING_ERROR,
                    "Episode processing failed",
                    f"{ep.title} from {sub.title} failed after all retry attempts: {e}",
                    severity="error",
                )

    def _check_cancellation(self, ep: Episode) -> bool:
        """
        Check if episode status in DB matches 'processing'.
        If it changed (e.g. to 'unprocessed'), abort and cleanup.
        Returns: True (Continue), False (Abort)
        """
        current_status = self.ep_repo.get_status(ep.id)
        if not self.ep_repo.owns_attempt(ep.id):
            logger.warning(f"Processing cancelled for {ep.title} (Status changed to {current_status})")
            self._acknowledge_cancellation(ep)
            return False
        return True

    def _acknowledge_cancellation(self, ep: Episode) -> None:
        self._cleanup_artifacts(ep)
        if self.ep_repo.attempt:
            self.job_repo.acknowledge(*self.ep_repo.attempt)

    def _cleanup_artifacts(self, ep: Episode):
        """Only discard this attempt's staging files, never a published revision."""
        directory = getattr(self, '_attempt_dir', None)
        if directory is None:
            return
        root = Path(settings.PODCASTS_DIR).resolve()
        resolved = directory.resolve()
        if directory == resolved and resolved.is_relative_to(root) and len(resolved.relative_to(root).parts) == 3 and resolved.name.startswith('attempt-'):
            shutil.rmtree(resolved, ignore_errors=False)
            self._attempt_dir = None

    async def publish_pending_feeds(self):
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            rows = conn.execute('SELECT id, subscription_id, local_filename FROM episodes WHERE publication_pending=1').fetchall()
        for row in rows:
            try:
                await asyncio.to_thread(self.rss_gen.generate_feed, row['subscription_id'])
                await asyncio.to_thread(self.rss_gen.generate_unified_feed)
                with get_db_connection() as conn:
                    conn.execute('UPDATE episodes SET publication_pending=0 WHERE id=? AND local_filename IS ?', (row['id'], row['local_filename']))
                    conn.commit()
            except Exception:
                logger.exception('Feed publication pending for episode %s; audio retained', row['id'])

    def _cleanup_abandoned_attempts(self, max_age_hours=48):
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            protected = {Path(row[0]).resolve() for row in conn.execute("SELECT work_directory FROM jobs WHERE status IN ('running','queued','retry_scheduled','rate_limited') AND work_directory IS NOT NULL")}
            protected.update(Path(row[0]).resolve().parent for row in conn.execute("SELECT local_filename FROM episodes WHERE local_filename IS NOT NULL"))
        root = Path(settings.PODCASTS_DIR).resolve()
        cutoff = datetime.now().timestamp() - max_age_hours * 3600
        for directory in root.glob('*/episode-*/attempt-*'):
            if directory.resolve() != directory or not directory.is_dir() or directory in protected or (directory / 'published.json').exists():
                continue
            if directory.stat().st_mtime < cutoff:
                shutil.rmtree(directory)

    async def cleanup_old_logs(self):
        """Clean up old login-attempt rows; log files are handled by rotation."""
        from datetime import timedelta
        from app.infra.database import get_db_connection
        
        try:
            await asyncio.to_thread(self._cleanup_abandoned_attempts)
            # Clean up login_attempts table
            thirty_days_ago = (now_utc() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
            with get_db_connection() as conn:
                result = conn.execute(
                    "DELETE FROM login_attempts WHERE timestamp < ?",
                    (thirty_days_ago,)
                )
                if result.rowcount > 0:
                    logger.info(f"Cleaned up {result.rowcount} old login attempts")
                conn.commit()
                
            # Clean up empty episode folders
            try:
                removed_temp_files = self._cleanup_stale_temporary_files()
                if removed_temp_files > 0:
                    logger.info(f"Cleaned up {removed_temp_files} stale temporary processor files")

                podcasts_dir = os.path.join(settings.DATA_DIR, "podcasts")
                if os.path.exists(podcasts_dir):
                    deleted_folders = 0
                    for subscription_folder in os.listdir(podcasts_dir):
                        sub_path = os.path.join(podcasts_dir, subscription_folder)
                        if not os.path.isdir(sub_path):
                            continue
                            
                        for episode_folder in os.listdir(sub_path):
                            ep_path = os.path.join(sub_path, episode_folder)
                            if not os.path.isdir(ep_path):
                                continue
                            
                            # Check if folder is empty
                            if not os.listdir(ep_path):
                                try:
                                    os.rmdir(ep_path)
                                    deleted_folders += 1
                                except Exception as e:
                                    logger.warning(f"Failed to delete empty folder {ep_path}: {e}")
                    
                        # Check if subscription folder is empty (inside the loop where sub_path is defined)
                        if os.path.exists(sub_path) and not os.listdir(sub_path):
                            try:
                                os.rmdir(sub_path)
                                logger.info(f"Deleted empty subscription folder: {subscription_folder}")
                            except Exception as e:
                                logger.warning(f"Failed to delete empty subscription folder {sub_path}: {e}")

                    if deleted_folders > 0:
                        logger.info(f"Cleaned up {deleted_folders} empty episode folders")
            except Exception as e:
                logger.warning(f"Folder cleanup failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Log cleanup failed: {e}")

    async def cleanup_old_episodes(self):
        """Clean up episodes per retention policies: Manual (Time) + Auto (Count)."""
        from app.infra.database import get_db_connection
        try:
            ids_to_delete = []
            with get_db_connection() as conn:
                # 1. Manual Downloads (Time Based)
                # processed_at < now - manual_retention_days
                cursor = conn.execute("""
                    SELECT e.id, e.title FROM episodes e
                    LEFT JOIN subscriptions s ON e.subscription_id = s.id
                    CROSS JOIN app_settings a
                    WHERE e.status = 'completed' 
                      AND e.is_manual_download = 1
                      AND datetime(e.processed_at) < datetime(
                          'now',
                          '-' || CASE
                              WHEN s.inherit_retention = 1
                                  THEN COALESCE(a.default_manual_retention_days, 14)
                              ELSE COALESCE(s.manual_retention_days, 14)
                          END || ' days'
                      )
                """)
                for row in cursor.fetchall():
                    logger.info(f"Cleanup: Expired Manual Download: {row['title']}")
                    ids_to_delete.append(row['id'])

                # 2. Auto Downloads (Count Based - Keep Last N)
                # Uses Window Functions (SQLite 3.25+)
                try:
                    cursor = conn.execute("""
                        SELECT t.id, t.title 
                        FROM (
                           SELECT e.id, e.title, e.subscription_id,
                                  ROW_NUMBER() OVER (
                                      PARTITION BY e.subscription_id
                                      ORDER BY CASE
                                          WHEN s2.source_type IN ('youtube_channel', 'youtube_playlist')
                                              THEN COALESCE(e.discovered_at, e.pub_date)
                                          ELSE e.pub_date
                                      END DESC
                                  ) as rn
                           FROM episodes e
                           JOIN subscriptions s2 ON s2.id = e.subscription_id
                           WHERE status='completed' 
                             AND (is_manual_download IS NULL OR is_manual_download=0)
                        ) t
                        JOIN subscriptions s ON t.subscription_id = s.id
                        CROSS JOIN app_settings a
                        WHERE t.rn > CASE
                            WHEN s.inherit_retention = 1
                                THEN COALESCE(a.default_retention_limit, 1)
                            ELSE COALESCE(s.retention_limit, 1)
                        END
                    """)
                    for row in cursor.fetchall():
                         logger.info(f"Cleanup: Auto Download Exceeds Limit: {row['title']}")
                         ids_to_delete.append(row['id'])
                except Exception as e:
                    logger.error(f"Cleanup Auto Error (Window Function?): {e}")

            for ep_id in set(ids_to_delete):
                await self.delete_episode(ep_id)
                
        except Exception as e:
            logger.error(f"Episode cleanup failed: {e}")

    async def run_loop(self):
        """Main loop."""
        # Requeue entries that were interrupted
        logger.info("Resuming interrupted processes...")
        try:
            self.ep_repo.requeue_stuck()
        except Exception as e:
            logger.error(f"Failed to requeue stuck episodes: {e}")
            
        # 1. Initial Feed Sync/Regen on startup to clear stale URLs
        await self.regenerate_all_feeds()
        
        # Track last feed check
        from datetime import datetime
        last_feed_check = datetime.min
        
        while True:
            # Get latest interval from DB
            from app.core.utils import get_global_settings
            db_settings = get_global_settings()
            interval_minutes = db_settings.get('check_interval_minutes', settings.CHECK_INTERVAL_MINUTES)
            interval_seconds = interval_minutes * 60
            
            try:
                await self.publish_pending_feeds()
                from app.infra.database import get_db_connection
                with get_db_connection() as conn:
                    deleted = conn.execute("SELECT id FROM episodes WHERE status='ignored' AND processing_step='cancellation requested'").fetchall()
                for row in deleted:
                    await asyncio.to_thread(self._finalize_episode_deletion, row['id'])
                # 1. Finish any subscription deletions whose workers have acknowledged cancellation.
                await self.finalize_pending_subscription_deletions()

                # 2. Always process queue (high frequency)
                await self.process_queue()
                
                # 3. Check Feeds (low frequency)
                now = datetime.now()
                if (now - last_feed_check).total_seconds() > interval_seconds:
                    logger.info("Interval reached. Checking feeds/maintenance...")
                    await self.cleanup_old_logs()
                    await self.cleanup_old_episodes()
                    await self.check_feeds()
                    last_feed_check = datetime.now()
                    from app.core.worker_health import record_feed_check
                    await asyncio.to_thread(record_feed_check, interval_minutes)
                
            except Exception as e:
                logger.error(f"Error in background processor loop: {e}")
                await send_notification_async(
                    EVENT_BREAKING_ERROR,
                    "Background processor error",
                    f"The background processor loop hit an unrecovered error: {e}",
                    severity="error",
                )
                
            # Short sleep to be responsive to new queue items (e.g. Manual Download/Reprocess)
            await asyncio.sleep(10)

def setup_background_logging():
    """Configure logging for the background process."""
    import logging
    from logging.handlers import RotatingFileHandler
    
    log_file = os.path.join(settings.DATA_DIR, "app.log")
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    fh = RotatingFileHandler(
        log_file,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT
    )
    fh.setFormatter(log_formatter)
    
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    root.addHandler(fh)
    root.addHandler(sh)

def start_processor_process():
    """Entry point for the background processor process."""
    import os
    import signal
    import asyncio
    
    # 0. Setup logging for the new process
    setup_background_logging()
    
    # 1. Lower priority for only this process
    try:
        os.nice(10)
    except Exception as e:
        print(f"Failed to set background priority: {e}")

    # 2. Setup isolated event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    Processor._is_background_worker = True
    processor = Processor()
    
    # 3. Handle stop signals gracefully
    stop_event = asyncio.Event()
    
    def handle_stop():
        print("Background processor receiving stop signal...")
        stop_event.set()
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle_stop)
        except NotImplementedError:
             # Signal handlers not supported on Windows in loop, but we are on Mac
             pass

    async def heartbeat_worker():
        from app.core.worker_health import record_heartbeat
        worker_id = f'{os.getpid()}:{uuid4().hex}'
        while True:
            await asyncio.to_thread(record_heartbeat, worker_id)
            await asyncio.sleep(10)

    async def run_until_stopped():
        heartbeat = asyncio.create_task(heartbeat_worker())
        runner = asyncio.create_task(processor.run_loop())
        stop = asyncio.create_task(stop_event.wait())
        try:
            done, _ = await asyncio.wait({runner, stop, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stop:
                    task.result()  # A failed loop/heartbeat exits for parent supervision.
        finally:
            for task in (heartbeat, runner, stop):
                task.cancel()
            await asyncio.gather(heartbeat, runner, stop, return_exceptions=True)

    try:
        loop.run_until_complete(run_until_stopped())
    finally:
        loop.close()
