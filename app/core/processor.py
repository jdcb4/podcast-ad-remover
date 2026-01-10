import asyncio
import os
import logging
import httpx
import aiofiles
import json
from datetime import datetime
from app.core.config import settings
from app.core.models import Episode
from app.infra.repository import EpisodeRepository, SubscriptionRepository
from app.core.ai_services import Transcriber, AdDetector, RateLimitError
from app.core.audio import AudioProcessor
from app.core.rss_gen import RSSGenerator
from app.core.feed import FeedManager

logger = logging.getLogger(__name__)

class Processor:
    _active_task_ids = set()
    _queue_lock = asyncio.Lock()  # Prevent race conditions in process_queue

    def __init__(self):
        self.ep_repo = EpisodeRepository()
        self.sub_repo = SubscriptionRepository()
        self.transcriber = Transcriber()
        self.ad_detector = AdDetector()
        self.rss_gen = RSSGenerator()

    async def check_feeds(self, subscription_id: int = None, limit: int = 5):
        """Check subscriptions for new episodes."""
        
        if subscription_id:
            sub = self.sub_repo.get_by_id(subscription_id)
            subs = [sub] if sub else []
        else:
            subs = self.sub_repo.get_all()
            
        for sub in subs:
            try:
                # Use subscription limit if set, else default. 
                # Limit of 0 is valid (means skip initial downloads)
                actual_limit = sub.retention_limit if sub.retention_limit is not None else limit
                logger.info(f"Checking {sub.title} (Sub Limit: {sub.retention_limit}, Ref Limit: {limit}, Final Limit: {actual_limit})...")

                # Fetch ALL episodes
                episodes = FeedManager.parse_episodes(sub.feed_url)
                
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
            except Exception as e:
                logger.error(f"Error checking feed {sub.feed_url}: {e}")

    async def process_episode(self, episode_id: int):
        """Force process a specific episode."""
        self.ep_repo.update_status(episode_id, "pending") # Reset to pending
        await self.process_queue() # Trigger queue processing

    async def delete_episode(self, episode_id: int):
        """Hard delete an episode and all associated files."""
        ep = self.ep_repo.get_by_id(episode_id)
        if not ep:
            return False
        
        # Get subscription for slug
        sub = self.sub_repo.get_by_id(ep.subscription_id)
        if not sub:
            return False
            
        # Delete entire episode directory
        episode_slug = f"{ep.guid}".replace("/", "_").replace(" ", "_")
        episode_dir = settings.get_episode_dir(sub.slug, episode_slug)
        
        if os.path.exists(episode_dir):
            try:
                import shutil
                shutil.rmtree(episode_dir)
                logger.info(f"Deleted episode directory: {episode_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete episode directory {episode_dir}: {e}")

        # Soft delete from DB (marks as ignored, keeps GUID to prevent re-download)
        self.ep_repo.soft_delete(episode_id)

        # Regenerate feeds immediately so the episode is removed
        self.rss_gen.generate_feed(sub.id)
        self.rss_gen.generate_unified_feed()
        
        return True

    async def version_episode(self, episode_id: int):
        """
        Increments the version suffix of an episode's GUID (e.g., _v2, _v3)
         and renames its physical directory to match.
        """
        import re
        ep = self.ep_repo.get_by_id(episode_id)
        if not ep:
            logger.error(f"Cannot version episode {episode_id}: Not found")
            return False
            
        sub = self.sub_repo.get_by_id(ep.subscription_id)
        if not sub:
            logger.error(f"Cannot version episode {episode_id}: Subscription {ep.subscription_id} not found")
            return False

        old_guid = ep.guid
        old_title = ep.title
        
        # Determine next version
        match = re.search(r'_v(\d+)$', old_guid)
        if match:
            v_num = int(match.group(1)) + 1
            new_guid = re.sub(r'_v\d+$', f"_v{v_num}", old_guid)
            # Update title version if present, else append
            if re.search(r' \(Reprocessed V\d+\)$', old_title):
                new_title = re.sub(r' \(Reprocessed V\d+\)$', f" (Reprocessed V{v_num})", old_title)
            else:
                new_title = f"{old_title} (Reprocessed V{v_num})"
        else:
            v_num = 1
            new_guid = f"{old_guid}_v{v_num}"
            new_title = f"{old_title} (Reprocessed V{v_num})"

        logger.info(f"Versioning episode {episode_id}: '{old_title}' ({old_guid}) -> '{new_title}' ({new_guid})")

        # Rename physical directory
        old_episode_slug = f"{old_guid}".replace("/", "_").replace(" ", "_")
        new_episode_slug = f"{new_guid}".replace("/", "_").replace(" ", "_")
        
        old_dir = settings.get_episode_dir(sub.slug, old_episode_slug)
        new_dir = settings.get_episode_dir(sub.slug, new_episode_slug)

        if os.path.exists(old_dir):
            try:
                os.rename(old_dir, new_dir)
                logger.info(f"Renamed episode directory: {old_dir} -> {new_dir}")
            except Exception as e:
                logger.error(f"Failed to rename directory {old_dir} to {new_dir}: {e}")
                # We continue anyway to update the DB, as the directory move is preferred but metadata is critical
        
        # Update database paths using a helper or manual dict update
        def update_path(p):
            if not p: return p
            return p.replace(old_dir, new_dir)

        # We need a way to update the GUID, Title and paths in the DB. 
        # Using a fresh connection helper from the database module
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE episodes SET 
                    guid = ?, 
                    title = ?,
                    local_filename = ?, 
                    transcript_path = ?, 
                    ad_report_path = ?, 
                    report_path = ? 
                WHERE id = ?
            """, (
                new_guid, 
                new_title,
                update_path(ep.local_filename),
                update_path(ep.transcript_path),
                update_path(ep.ad_report_path),
                update_path(ep.report_path),
                episode_id
            ))
            conn.commit()

        logger.info(f"Metadata and Title updated for episode {episode_id}")
        return True

    def _extract_text(self, start: float, end: float, segments: list) -> str:
        """Extract text from transcript overlapping with the given time range."""
        text = []
        for seg in segments:
            if seg['start'] < end and seg['end'] > start:
                text.append(seg['text'])
        return " ".join(text).strip()
    
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
        # Use lock to prevent race conditions from multiple callers WITHIN this process
        async with Processor._queue_lock:
            # 1. Fetch limit from settings
            from app.web.router import get_global_settings
            db_settings = get_global_settings()
            limit = db_settings.get('concurrent_downloads', 2)
            if not limit or limit < 1: limit = 2

            # 2. Check if we have room using DATABASE count (cross-process safe)
            currently_processing = self.ep_repo.count_processing()
            if currently_processing >= limit:
                return

            # 3. Get pending episodes
            pending = self.ep_repo.get_pending()
            if not pending:
                return

            # 4. Launch new tasks up to limit
            capacity = limit - currently_processing
            for ep_dict in pending:
                if capacity <= 0:
                    break
                    
                ep_id = ep_dict['id']
                
                # Skip if already in our in-process tracking (for this process)
                if ep_id in Processor._active_task_ids:
                    continue
                
                # CRITICAL: Mark as "processing" in DB BEFORE launching task
                # This prevents other processes from picking up the same episode
                self.ep_repo.update_status(ep_id, "processing")
                self.ep_repo.update_progress(ep_id, "Starting...", 0)
                    
                # Add to in-process set and launch
                Processor._active_task_ids.add(ep_id)
                capacity -= 1
                
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
                return

            # Actually run the processing
            await self._process_episode_inner(ep, sub, ep_dict)
            
        except Exception as e:
            logger.error(f"Fatal error in episode task {ep_id}: {e}")
        finally:
            # Ensure ID is removed from active set
            Processor._active_task_ids.discard(ep_id)
            # Proactively check queue again after finishing to keep pipeline full
            asyncio.create_task(self.process_queue())

    async def _process_episode_inner(self, ep: Episode, sub, ep_dict: dict):
        """Core multi-step processing logic for a single episode."""
        logger.info(f"Processing {ep.title}...")
        
        try:
            self.ep_repo.update_status(ep.id, "processing")
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
            episode_slug = f"{ep.guid}".replace("/", "_").replace(" ", "_")
            episode_dir = settings.get_episode_dir(sub.slug, episode_slug)
            os.makedirs(episode_dir, exist_ok=True)
            
            input_path = os.path.join(episode_dir, "original.mp3")
            transcript_path = None
            
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
                
                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", ep.original_url, follow_redirects=True, timeout=300.0) as resp:
                        resp.raise_for_status()
                        total = int(resp.headers.get("Content-Length", 0))
                        downloaded = 0
                        last_logged_percent = -1
                        last_cancel_check = datetime.now()
                        
                        async with aiofiles.open(input_path, "wb") as f:
                            async for chunk in resp.aiter_bytes():
                                await f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Periodic cancellation check (Time-based + Percent-based)
                                if (datetime.now() - last_cancel_check).total_seconds() > 2.0:
                                     if not self._check_cancellation(ep): return
                                     last_cancel_check = datetime.now()

                                if total > 0:
                                    percent = int((downloaded / total) * 100)
                                    # Update DB every 5% 
                                    if percent % 5 == 0 and percent != last_logged_percent:
                                        self.ep_repo.update_progress(ep.id, "downloading", percent)
                                        logger.info(f"Downloading {ep.title}: {percent}%")
                                        last_logged_percent = percent
                
                if not self._check_cancellation(ep): return # Check after download

                file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
                logger.info(f"Download complete: {file_size_mb:.2f} MB")
            
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
                        self._cleanup_artifacts(ep)
                        return # Stop processing this episode
                    raise e
                
                if not self._check_cancellation(ep): return # Check after transcribe
                
                # Double check state
                if cancellation_state['is_cancelled']:
                     self._cleanup_artifacts(ep)
                     return

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Transcription complete in {duration:.1f}s")
                
                # Save Transcript (Prefer JSON now)
                transcript_path = os.path.join(episode_dir, "transcript.json")
                async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(transcript))
                
            self.ep_repo.update_progress(ep.id, "detecting_ads", 50, transcript_path=transcript_path)
            
            if not self._check_cancellation(ep): return

            # 3. Detect Ads
            logger.info("Detecting ads...")
            
            detect_options = {
                "remove_ads": sub.remove_ads,
                "remove_promos": sub.remove_promos,
                "remove_intros": sub.remove_intros,
                "remove_outros": sub.remove_outros,
                "custom_instructions": sub.custom_instructions
            }
            
            ad_segments = await asyncio.to_thread(self.ad_detector.detect_ads, transcript, detect_options)
            
            if not self._check_cancellation(ep): return

            logger.info(f"Found {len(ad_segments)} ad segments: {ad_segments}")
            
            # Merge ad segments with gaps < 10 seconds
            merged_segments = []
            for segment in sorted(ad_segments, key=lambda x: x['start']):
                if merged_segments and segment['start'] - merged_segments[-1]['end'] < 10:
                    # Merge with previous segment
                    merged_segments[-1]['end'] = segment['end']
                    # Keep the label/reason of the first segment or combine? Let's keep first for simplicity
                    logger.info(f"Merged segment {segment['start']}-{segment['end']} into {merged_segments[-1]['start']}-{merged_segments[-1]['end']}")
                else:
                    merged_segments.append(segment.copy())
            
            ad_segments = merged_segments
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
            async with aiofiles.open(report_path, "w") as f:
                await f.write(json.dumps(report_data, indent=2))

            # Generate Human-Readable Report (HTML)
            human_report_path = os.path.join(episode_dir, "report.html")
            
            rows_html = ""
            for s in ad_segments:
                rows_html += f"""
                <div class="segment">
                    <div class="flex justify-between">
                        <strong>{s['start']}s - {s['end']}s</strong>
                        <span class="badge">{s.get('label', 'Ad')}</span>
                    </div>
                    <p class="reason">{s.get('reason', 'No reason provided')}</p>
                    <div class="transcript-text">
                        "{s.get('text', 'No text extracted')}"
                    </div>
                </div>
                """

            html_content = f"""
            <html>
            <head>
                <title>Ad Report: {ep.title}</title>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
                <style>
                    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                    body {{ 
                        font-family: 'Inter', sans-serif; 
                        max-width: 900px; 
                        margin: 0 auto; 
                        padding: 2rem 1rem;
                        background: #0a0a0f;
                        color: #fafafa;
                        line-height: 1.6;
                    }}
                    h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; }}
                    h1 {{ font-size: 2rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #a78bfa, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                    h2 {{ font-size: 1.5rem; margin-bottom: 1rem; color: #fafafa; }}
                    h3 {{ font-size: 1.125rem; margin: 1.5rem 0 1rem 0; color: #a1a1aa; }}
                    .meta {{ color: #52525b; font-size: 0.85em; margin-bottom: 1.5rem; font-family: monospace; }}
                    /* Scrollbar */
                    ::-webkit-scrollbar {{ width: 8px; }}
                    ::-webkit-scrollbar-track {{ background: #0a0a0f; }}
                    ::-webkit-scrollbar-thumb {{ background: #22222f; border-radius: 4px; }}
                    ::-webkit-scrollbar-thumb:hover {{ background: #3f3f46; }}

                    .segment {{ 
                        background: #1a1a25;
                        padding: 1.25rem; 
                        margin: 1rem 0; 
                        border-left: 4px solid #8b5cf6; 
                        border-radius: 0.75rem;
                        border: 1px solid rgba(255,255,255,0.08);
                    }}
                    .badge {{ 
                        background: rgba(139,92,246,0.15); 
                        color: #a78bfa; 
                        padding: 0.25rem 0.75rem; 
                        border-radius: 999px; 
                        font-size: 0.75em; 
                        font-weight: 600;
                        border: 1px solid rgba(139,92,246,0.2);
                    }}
                    .badge.intro {{ background: rgba(52,211,153,0.15); color: #34d399; border-color: rgba(52,211,153,0.2); }}
                    .badge.outro {{ background: rgba(251,191,36,0.15); color: #fbbf24; border-color: rgba(251,191,36,0.2); }}
                    .flex {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }}
                    .transcript-text {{ 
                        background: rgba(255,255,255,0.03); 
                        padding: 0.75rem 1rem; 
                        border-radius: 0.5rem; 
                        font-style: italic; 
                        color: #a1a1aa; 
                        font-size: 0.9em; 
                        margin-top: 0.75rem;
                        border: 1px solid rgba(255,255,255,0.06);
                    }}
                    .reason {{ margin: 0; font-weight: 600; color: #a78bfa; }}
                    .time {{ color: #fafafa; font-weight: 600; }}
                    a {{ color: #a78bfa; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    .total {{ 
                        display: inline-block;
                        background: rgba(139,92,246,0.1); 
                        color: #a78bfa; 
                        padding: 0.5rem 1rem; 
                        border-radius: 0.5rem;
                        font-weight: 600;
                        margin-bottom: 1rem;
                    }}
                </style>
            </head>
            <body>
                <div style="margin-bottom: 2rem;">
                    <a href="/" style="font-weight: 700; font-size: 1.25rem; color: #fafafa; text-decoration: none; display: flex; align-items: center; gap: 0.5rem;">
                         <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #a78bfa;"><path d="M4 11v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"></path><path d="M4 11l8-8 8 8"></path><path d="M12 19v-6"></path></svg>
                         Back to Dashboard
                    </a>
                </div>
                <h1>Ad Report</h1>
                <h2>{ep.title}</h2>
                <p class="meta">GUID: {ep.guid}</p>
                
                <h3>Detected Segments</h3>
                <p class="total">Total Segments: {len(ad_segments)}</p>
                
                {rows_html}
                
                <h3>Transcript</h3>
                <p><a href="/artifacts/transcript/{ep.id}" class="btn">View Full Transcript (JSON)</a></p>
            </body>
            </html>
            """
            
            async with aiofiles.open(human_report_path, "w", encoding="utf-8") as f:
                await f.write(html_content)

            self.ep_repo.update_progress(ep.id, "removing_ads", 75, ad_report_path=report_path, report_path=human_report_path)
            
            if not self._check_cancellation(ep): return

            # 4. Remove Ads
            output_path = os.path.join(episode_dir, "processed.mp3")
            
            logger.info("Removing ads with FFmpeg...")
            await asyncio.to_thread(
                AudioProcessor.remove_segments, 
                input_path, 
                output_path, 
                ad_segments
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
                
                if do_text or do_audio:
                    summary_text = None
                    try:
                        logger.info(f"Generating episode summary for {ep.title}...")
                        # Build subscription settings dict for targets
                        sub_settings = {
                            'remove_ads': sub.remove_ads,
                            'remove_promos': sub.remove_promos,
                            'remove_intros': sub.remove_intros,
                            'remove_outros': sub.remove_outros
                        }
                        summary_text = await asyncio.to_thread(
                            self.ad_detector.generate_summary,
                            transcript, 
                            sub.title or "Podcast", 
                            ep.title, 
                            str(ep.pub_date) if ep.pub_date else "recently",
                            sub_settings
                        )
                        # Save to DB and file immediately
                        self.ep_repo.update_ai_summary(ep.id, summary_text)
                        summary_txt_path = os.path.join(episode_dir, "summary.txt")
                        async with aiofiles.open(summary_txt_path, "w") as f:
                            await f.write(summary_text)
                    except Exception as e:
                        logger.error(f"Failed to generate/save text summary: {e}")
                        if not summary_text:
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
                            concat_list
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

            # 5. Cleanup & Save
            if os.path.exists(input_path):
                os.remove(input_path)
            
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            self.ep_repo.update_status(ep.id, "completed", filename=output_path, file_size=file_size)
            self.ep_repo.update_progress(ep.id, "completed", 100)
            
            # 6. Regenerate Feed
            self.rss_gen.generate_feed(sub.id)
            self.rss_gen.generate_unified_feed()
            
            logger.info(f"Successfully processed {ep.title}")

        except RateLimitError as e:
            # Special handling for API rate limits - wait until quota resets
            logger.warning(f"Rate limit hit for episode {ep.id}: {e}")
            next_retry = e.get_next_retry_time()
            retry_time_str = next_retry.strftime('%Y-%m-%d %H:%M:%S UTC')
            logger.info(f"Episode {ep.title} placed on hold until API quota resets at {retry_time_str}")
            self.ep_repo.update_rate_limited(ep.id, next_retry, str(e))
            
        except Exception as e:
            logger.error(f"Failed to process episode {ep.id}: {e}")
            
            # Check if this might be a rate limit we didn't catch
            error_str = str(e).lower()
            rate_limit_patterns = ['resource_exhausted', 'quota', 'rate limit', '429', 'too many requests']
            if any(pattern in error_str for pattern in rate_limit_patterns):
                # Treat as rate limit
                logger.warning(f"Detected possible rate limit in error: {e}")
                from app.core.ai_services import RateLimitError
                rate_error = RateLimitError(str(e), is_daily_limit=True, provider="unknown")
                next_retry = rate_error.get_next_retry_time()
                self.ep_repo.update_rate_limited(ep.id, next_retry, str(e))
                return
            
            # Regular Retry Logic
            retry_count = ep_dict.get('retry_count', 0) + 1
            if retry_count <= 5:
                # Exponential backoff: 5, 10, 20, 40, 80 minutes
                delay_minutes = 5 * (2 ** (retry_count - 1))
                from datetime import timedelta
                next_retry = datetime.now() + timedelta(minutes=delay_minutes)
                
                logger.info(f"Scheduling retry {retry_count}/5 for {ep.title} in {delay_minutes} minutes")
                self.ep_repo.update_retry(ep.id, retry_count, next_retry, str(e))
            else:
                logger.error(f"Max retries reached for {ep.title}")
                self.ep_repo.update_status(ep.id, "failed", error=str(e))

    def _check_cancellation(self, ep: Episode) -> bool:
        """
        Check if episode status in DB matches 'processing'.
        If it changed (e.g. to 'unprocessed'), abort and cleanup.
        Returns: True (Continue), False (Abort)
        """
        current_status = self.ep_repo.get_status(ep.id)
        if current_status != 'processing':
            logger.warning(f"Processing cancelled for {ep.title} (Status changed to {current_status})")
            self._cleanup_artifacts(ep)
            return False
        return True

    def _cleanup_artifacts(self, ep: Episode):
        """Cleanup temporary files for a cancelled episode."""
        try:
            logger.info(f"Cleaning up artifacts for {ep.title}...")
            # Get subscription for slug
            sub = self.sub_repo.get_by_id(ep.subscription_id)
            if not sub:
                return
                
            # Remove entire episode directory if it exists
            episode_slug = f"{ep.guid}".replace("/", "_").replace(" ", "_")
            episode_dir = settings.get_episode_dir(sub.slug, episode_slug)
            
            if os.path.exists(episode_dir):
                import shutil
                shutil.rmtree(episode_dir)
                logger.info(f"Cleaned up episode directory: {episode_dir}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def cleanup_old_logs(self):
        """Clean up logs older than 30 days."""
        from datetime import datetime, timedelta
        from app.infra.database import get_db_connection
        
        try:
            # Clean up login_attempts table
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
            with get_db_connection() as conn:
                result = conn.execute(
                    "DELETE FROM login_attempts WHERE timestamp < ?",
                    (thirty_days_ago,)
                )
                if result.rowcount > 0:
                    logger.info(f"Cleaned up {result.rowcount} old login attempts")
                conn.commit()
                
            # Clean up app.log file - keep only lines from last 30 days
            log_path = os.path.join(settings.DATA_DIR, "app.log")
            if os.path.exists(log_path):
                cutoff_date = datetime.now() - timedelta(days=30)
                cutoff_str = cutoff_date.strftime('%Y-%m-%d')
                
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                original_count = len(lines)
                # Keep lines that start with a date >= cutoff, or lines without date prefix
                kept_lines = []
                for line in lines:
                    try:
                        # Log format: "2025-12-21 17:34:24,050 - ..."
                        if len(line) >= 10 and line[4] == '-' and line[7] == '-':
                            line_date = line[:10]
                            if line_date >= cutoff_str:
                                kept_lines.append(line)
                        else:
                            kept_lines.append(line)  # Keep non-dated lines
                    except:
                        kept_lines.append(line)
                
                if len(kept_lines) < original_count:
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.writelines(kept_lines)
                    logger.info(f"Cleaned up {original_count - len(kept_lines)} old log lines")
        
            # Clean up empty episode folders
            try:
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
                    WHERE e.status = 'completed' 
                      AND e.is_manual_download = 1
                      AND datetime(e.processed_at) < datetime('now', '-' || COALESCE(s.manual_retention_days, 14) || ' days')
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
                           SELECT id, title, subscription_id,
                                  ROW_NUMBER() OVER (PARTITION BY subscription_id ORDER BY pub_date DESC) as rn
                           FROM episodes
                           WHERE status='completed' 
                             AND (is_manual_download IS NULL OR is_manual_download=0)
                        ) t
                        JOIN subscriptions s ON t.subscription_id = s.id
                        WHERE t.rn > COALESCE(s.retention_limit, 1)
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
            from app.web.router import get_global_settings
            db_settings = get_global_settings()
            interval_minutes = db_settings.get('check_interval_minutes', settings.CHECK_INTERVAL_MINUTES)
            interval_seconds = interval_minutes * 60
            
            try:
                # 1. Always process queue (high frequency)
                await self.process_queue()
                
                # 2. Check Feeds (low frequency)
                now = datetime.now()
                if (now - last_feed_check).total_seconds() > interval_seconds:
                    logger.info("Interval reached. Checking feeds/maintenance...")
                    await self.cleanup_old_logs()
                    await self.cleanup_old_episodes()
                    await self.check_feeds()
                    last_feed_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in background processor loop: {e}")
                
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

    async def run_until_stopped():
        runner = asyncio.create_task(processor.run_loop())
        await stop_event.wait()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
        print("Background processor stopped clean.")

    try:
        loop.run_until_complete(run_until_stopped())
    finally:
        loop.close()
