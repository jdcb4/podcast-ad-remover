import asyncio
import os
import logging
import httpx
import aiofiles
from datetime import datetime
from app.core.config import settings
from app.core.models import Episode
from app.infra.repository import EpisodeRepository, SubscriptionRepository
from app.core.ai_services import Transcriber, AdDetector
from app.core.audio import AudioProcessor
from app.core.rss_gen import RSSGenerator
from app.core.feed import FeedManager

logger = logging.getLogger(__name__)

class Processor:
    _is_processing = False

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
        """Process pending episodes sequentially."""
        if Processor._is_processing:
            logger.info("Queue processing already in progress. Skipping.")
            return

        Processor._is_processing = True
        try:
            while True:
                # Get next pending episode
                pending = self.ep_repo.get_pending()
                if not pending:
                    break
                
                # Process one episode at a time
                ep_dict = pending[0]
                ep = Episode.model_validate(ep_dict)
                sub = self.sub_repo.get_by_id(ep.subscription_id)
                
                logger.info(f"Processing {ep.title}...")
                
                try:
                    self.ep_repo.update_status(ep.id, "processing")
                    self.ep_repo.update_progress(ep.id, "Actively Processing", 0)
                    
                    if not self._check_cancellation(ep): continue

                    self.ep_repo.update_progress(ep.id, "processing", 10)
                    
                    # Check for skip flags
                    skip_transcription = False
                    if ep.processing_flags:
                        try:
                            import json
                            flags = json.loads(ep.processing_flags)
                            skip_transcription = flags.get('skip_transcription', False)
                        except: pass
                        
                    transcript = None
                    
                    # Create episode-specific directory
                    episode_slug = f"{ep.guid}".replace("/", "_").replace(" ", "_")
                    episode_dir = settings.get_episode_dir(sub.slug, episode_slug)
                    os.makedirs(episode_dir, exist_ok=True)
                    
                    input_path = os.path.join(episode_dir, "original.mp3")
                    transcript_path = None
                    
                    if skip_transcription and ep.transcript_path and os.path.exists(ep.transcript_path):
                         logger.info(f"Skipping transcription, using existing: {ep.transcript_path}")
                         transcript_path = ep.transcript_path
                         import ast
                         async with aiofiles.open(transcript_path, "r", encoding="utf-8") as f:
                             content = await f.read()
                             # Handle both JSON and Python dict string (legacy)
                             try:
                                 transcript = json.loads(content)
                             except:
                                 try:
                                     transcript = ast.literal_eval(content)
                                 except Exception as e:
                                     logger.error(f"Failed to load transcript: {e}")
                                     # Fallback to re-transcribe if load fails
                                     skip_transcription = False
                    
                    # 1. Ensure Audio Exists (Download if missing)
                    if not os.path.exists(input_path):
                        if not self._check_cancellation(ep): continue
                        
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
                                             if not self._check_cancellation(ep): continue
                                             last_cancel_check = datetime.now()

                                        if total > 0:
                                            percent = int((downloaded / total) * 100)
                                            # Update DB every 5% 
                                            if percent % 5 == 0 and percent != last_logged_percent:
                                                self.ep_repo.update_progress(ep.id, "downloading", percent)
                                                logger.info(f"Downloading {ep.title}: {percent}%")
                                                last_logged_percent = percent
                        
                        if not self._check_cancellation(ep): continue # Check after download

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
                                continue # Stop processing this episode
                            raise e
                        
                        if not self._check_cancellation(ep): continue # Check after transcribe
                        
                        # Double check state
                        if cancellation_state['is_cancelled']:
                             self._cleanup_artifacts(ep)
                             continue

                        duration = (datetime.now() - start_time).total_seconds()
                        logger.info(f"Transcription complete in {duration:.1f}s")
                        
                        # Save Transcript (Prefer JSON now)
                        transcript_path = os.path.join(episode_dir, "transcript.json")
                        import json
                        async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                            await f.write(json.dumps(transcript))
                        
                    self.ep_repo.update_progress(ep.id, "detecting_ads", 50, transcript_path=transcript_path)
                    
                    if not self._check_cancellation(ep): continue

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
                    
                    if not self._check_cancellation(ep): continue

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
                    import json
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
                        <style>
                            body {{ font-family: sans-serif; max_width: 800px; margin: 2rem auto; padding: 0 1rem; }}
                            .segment {{ background: #ffebee; padding: 15px; margin: 10px 0; border-left: 4px solid #f44336; border-radius: 4px; }}
                            .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
                            .badge {{ background: #e53935; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
                            .flex {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }}
                            .transcript-text {{ background: rgba(255,255,255,0.5); padding: 8px; border-radius: 4px; font-style: italic; color: #444; font-size: 0.9em; margin-top: 10px; }}
                            .reason {{ margin: 0 0 5px 0; font-weight: bold; color: #b71c1c; }}
                        </style>
                    </head>
                    <body>
                        <h1>Ad Report</h1>
                        <h2>{ep.title}</h2>
                        <p class="meta">GUID: {ep.guid}</p>
                        
                        <h3>Detected Segments</h3>
                        <p>Total Segments: {len(ad_segments)}</p>
                        
                        {rows_html}
                        
                        <h3>Transcript</h3>
                        <p><a href="file://{transcript_path}">View Full Transcript</a></p>
                    </body>
                    </html>
                    """
                    
                    async with aiofiles.open(human_report_path, "w", encoding="utf-8") as f:
                        await f.write(html_content)

                    self.ep_repo.update_progress(ep.id, "removing_ads", 75, ad_report_path=report_path, report_path=human_report_path)
                    
                    if not self._check_cancellation(ep): continue

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
                    
                    if not self._check_cancellation(ep): continue

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
                                summary_text = await asyncio.to_thread(
                                    self.ad_detector.generate_summary,
                                    transcript, 
                                    sub.title or "Podcast", 
                                    ep.title, 
                                    str(ep.pub_date) if ep.pub_date else "recently"
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
                    
                    if not self._check_cancellation(ep): continue

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

                except Exception as e:
                    logger.error(f"Failed to process episode {ep.id}: {e}")
                    
                    # Retry Logic
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
        
        finally:
            Processor._is_processing = False

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
                    
                    # Check if subscription folder is empty
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
