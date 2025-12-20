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
        logger.info(f"Checking feeds (Limit: {limit})...")
        
        if subscription_id:
            sub = self.sub_repo.get_by_id(subscription_id)
            subs = [sub] if sub else []
        else:
            subs = self.sub_repo.get_all()
            
        for sub in subs:
            try:
                logger.info(f"Checking {sub.title}...")
                # Fetch ALL episodes
                episodes = FeedManager.parse_episodes(sub.feed_url)
                
                for i, ep_data in enumerate(episodes):
                    ep_data['subscription_id'] = sub.id
                    
                    # Determine status based on limit
                    if i < limit:
                        ep_data['status'] = 'pending'
                    else:
                        ep_data['status'] = 'unprocessed'
                        
                    # Try to create. If exists, it returns False.
                    if self.ep_repo.create_or_ignore(ep_data):
                        logger.info(f"New episode found: {ep_data['title']} ({ep_data['status']})")
            except Exception as e:
                logger.error(f"Error checking feed {sub.feed_url}: {e}")

    async def process_episode(self, episode_id: int):
        """Force process a specific episode."""
        self.ep_repo.update_status(episode_id, "pending") # Reset to pending
        await self.process_queue() # Trigger queue processing

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
                    self.ep_repo.update_progress(ep.id, "downloading", 0)
                    
                    if not self._check_cancellation(ep): break

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
                    temp_filename = f"{ep.guid}.mp3".replace("/", "_")
                    input_path = os.path.join(settings.DOWNLOADS_DIR, temp_filename)
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
                        if not self._check_cancellation(ep): break
                        
                        logger.info(f"Downloading {ep.title}...")
                        
                        async with httpx.AsyncClient() as client:
                            async with client.stream("GET", ep.original_url, follow_redirects=True, timeout=300.0) as resp:
                                resp.raise_for_status()
                                total = int(resp.headers.get("Content-Length", 0))
                                downloaded = 0
                                last_logged_percent = -1
                                
                                async with aiofiles.open(input_path, "wb") as f:
                                    async for chunk in resp.aiter_bytes():
                                        await f.write(chunk)
                                        downloaded += len(chunk)
                                        if total > 0:
                                            percent = int((downloaded / total) * 100)
                                            # Update DB every 10% to avoid spamming
                                            if percent % 10 == 0 and percent != last_logged_percent:
                                                if not self._check_cancellation(ep): break # Check during download
                                                self.ep_repo.update_progress(ep.id, "downloading", percent)
                                                logger.info(f"Downloading {ep.title}: {percent}%")
                                                last_logged_percent = percent
                        
                        if not self._check_cancellation(ep): break # Check after download

                        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
                        logger.info(f"Download complete: {file_size_mb:.2f} MB")
                    
                    # 2. Transcribe (If needed)
                    if not transcript:
                        self.ep_repo.update_progress(ep.id, "transcribing", 0)
                                    
                        start_time = datetime.now()
                        logger.info(f"Starting transcription for {ep.title}...")
                        transcript = await asyncio.to_thread(self.transcriber.transcribe, input_path)
                        
                        if not self._check_cancellation(ep): break # Check after transcribe

                        duration = (datetime.now() - start_time).total_seconds()
                        logger.info(f"Transcription complete in {duration:.1f}s")
                        
                        # Save Transcript (Prefer JSON now)
                        transcript_filename = f"{ep.guid}.txt".replace("/", "_")
                        transcript_path = os.path.join(settings.TRANSCRIPTS_DIR, transcript_filename)
                        import json
                        async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                            await f.write(json.dumps(transcript))
                        
                    self.ep_repo.update_progress(ep.id, "detecting_ads", 50, transcript_path=transcript_path)
                    
                    if not self._check_cancellation(ep): break

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
                    
                    if not self._check_cancellation(ep): break

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
                    
                    # Save Ad Report (JSON)
                    import json
                    report_filename = f"{ep.guid}_report.json".replace("/", "_")
                    report_path = os.path.join(settings.TRANSCRIPTS_DIR, report_filename)
                    report_data = {
                        "episode_id": ep.id,
                        "guid": ep.guid,
                        "segments": ad_segments,
                        "transcript_path": transcript_path
                    }
                    async with aiofiles.open(report_path, "w") as f:
                        await f.write(json.dumps(report_data, indent=2))

                    # Generate Human-Readable Report (HTML)
                    human_report_filename = f"{ep.guid}_report.html".replace("/", "_")
                    human_report_path = os.path.join(settings.TRANSCRIPTS_DIR, human_report_filename)
                    
                    rows_html = ""
                    for s in ad_segments:
                        rows_html += f"""
                        <div class="segment">
                            <div class="flex justify-between">
                                <strong>{s['start']}s - {s['end']}s</strong>
                                <span class="badge">{s.get('label', 'Ad')}</span>
                            </div>
                            <p>{s.get('reason', 'No reason provided')}</p>
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
                    
                    if not self._check_cancellation(ep): break

                    # 4. Remove Ads
                    output_dir = os.path.join(settings.AUDIO_DIR, sub.slug)
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f"{ep.guid}_clean.mp3".replace("/", "_")
                    output_path = os.path.join(output_dir, output_filename)
                    
                    logger.info("Removing ads with FFmpeg...")
                    await asyncio.to_thread(
                        AudioProcessor.remove_segments, 
                        input_path, 
                        output_path, 
                        ad_segments
                    )
                    logger.info(f"Saved cleaned audio to {output_path}")
                    
                    # 4.5 Generate & Append Summary (If enabled)
                    
                    if not self._check_cancellation(ep): break

                    # 4.5 Generate Intros (Title Intro & Summary)
                    intro_files = []
                    temp_clean_path = None
                    try:
                        self.ep_repo.update_progress(ep.id, "generating_intros", 90)

                        # A. Title Intro
                        if sub.append_title_intro:
                            logger.info("Generating Title Intro...")
                            date_str = ep.pub_date.strftime('%B %d, %Y') if ep.pub_date else "recently"
                            # Ensure we handle potential None values gracefully
                            p_title = sub.title or "Podcast"
                            e_title = ep.title or "Episode"
                            
                            intro_text = f"You're listening to {p_title} from {date_str}, {e_title}"
                            
                            intro_filename = f"{ep.guid}_title_intro.mp3".replace("/", "_")
                            intro_path = os.path.join(settings.TRANSCRIPTS_DIR, intro_filename)
                            
                            await self.ad_detector.generate_audio(intro_text, intro_path)
                            intro_files.append(intro_path)

                        # B. AI Summary
                        if sub.append_summary:
                            logger.info("Generating episode summary & TTS...")
                            
                            summary_text = await asyncio.to_thread(
                                self.ad_detector.generate_summary,
                                transcript, 
                                sub.title or "Podcast", 
                                ep.title, 
                                str(ep.pub_date) if ep.pub_date else "recently"
                            )
                            
                            # Update DB description
                            self.ep_repo.update_description(ep.id, summary_text)
                            
                            summary_filename = f"{ep.guid}_summary.mp3".replace("/", "_")
                            summary_path = os.path.join(settings.TRANSCRIPTS_DIR, summary_filename)
                            
                            # Also save summary text
                            summary_txt_path = summary_path.replace(".mp3", ".txt")
                            async with aiofiles.open(summary_txt_path, "w") as f:
                                await f.write(summary_text)
                            
                            await self.ad_detector.generate_audio(summary_text, summary_path)
                            intro_files.append(summary_path)
                        
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
                    
                    if not self._check_cancellation(ep): break

                    # 5. Cleanup & Save
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    
                    self.ep_repo.update_status(ep.id, "completed", filename=output_path)
                    self.ep_repo.update_progress(ep.id, "completed", 100)
                    
                    # 6. Regenerate Feed
                    self.rss_gen.generate_feed(sub.id)
                    
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
            # 1. Input download
            temp_filename = f"{ep.guid}.mp3".replace("/", "_")
            input_path = os.path.join(settings.DOWNLOADS_DIR, temp_filename)
            if os.path.exists(input_path): os.remove(input_path)
            
            # 2. Transcript (if incomplete/cancelled immediately)
            # Maybe we keep transcript if we are just cancelling logic? 
            # User request: "delete partical artifacts". 
            # If we are mid-transcription, we might not have a full file.
            pass 
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def run_loop(self):
        """Main loop."""
        while True:
            await self.check_feeds()
            await self.process_queue()
            logger.info(f"Sleeping for {settings.CHECK_INTERVAL_MINUTES} minutes...")
            await asyncio.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
