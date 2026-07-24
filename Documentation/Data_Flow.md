# Data Flow

## 1. Subscription & Polling
1.  **User** adds a Podcast RSS URL via Web UI.
2.  **System** saves one global podcast row to `subscriptions`, or reuses the existing global row if the feed is already known.
3.  **System** adds the podcast to the user's `user_subscriptions` list. New podcasts record the first adding user as `subscriptions.owner_user_id`.
4.  **Scheduler** wakes up (e.g., every hour) and iterates active subscriptions.
5.  **Feed Manager** fetches the remote RSS feed.
6.  **System** compares remote episodes with `episodes` table (by GUID).
7.  **System** queues new episodes for processing.

## 2. Episode Processing Pipeline
For each queued episode:

1.  **Download**:
    - Fetch audio from `enclosure` URL.
    - Save episode artifacts under `/data/podcasts/{podcast_slug}/{episode_slug}/`.

2.  **Transcribe (Whisper)**:
    - Load Whisper model (if not loaded).
    - Process audio file -> generate text segments with timestamps.

3.  **Ad Detection (Configured LLM)**:
    - Build a timestamped prompt for Gemini by default, or another configured cloud/custom OpenAI-compatible provider.
    - When context-aware chunking is enabled and the prompt exceeds the configured budget, partition it at transcript-segment boundaries with bounded overlap. Short prompts still use one request.
    - Require every request to return a valid JSON array. A timeout, rate limit, malformed response, or other failed chunk fails the complete attempt and enters normal retry handling.
    - Clip chunk results to their primary time windows and deterministically merge same-label overlap without combining incompatible labels.

4.  **Ad Removal (FFmpeg)**:
    - Calculate "keep" segments (total duration minus ad segments).
    - Use FFmpeg to cut and concatenate "keep" segments.
    - Save processed audio in the episode artifact directory.

    AI description rewrites and audio summaries are skipped when ad-detection chunking is enabled,
    because summary generation is not yet chunk-aware. Spoken title intros are unaffected.

5.  **Finalize**:
    - Update database with processing stats (time saved, ad count).
    - Clean up temporary/intermediate files according to the processor settings.
    - Regenerate the podcast's local RSS feed XML in `/data/feeds/`.

## 3. Consumption
1.  **User** points their Podcast Player to `http://{host}/feeds/{podcast_slug}.xml`.
2.  **Player** requests the feed.
3.  **System** serves the static XML file.
4.  **Player** requests an episode.
5.  **System** serves the processed audio file from the stored episode path.

The public feed/audio path is not tied to a logged-in account by default. Admin-visible podcast stats can show how many user libraries include each podcast and the existing aggregate episode play count. Per-user download attribution would require token-attributed audio access logging and is not currently part of the data flow.

## 4. Subscription Deletion

1. The delete request atomically deactivates the subscription, marks its episodes ignored, and cancels all queued or retryable jobs.
2. Running jobs retain their worker lock while cancellation is requested. Workers stop at their next safe checkpoint, remove worker-owned temporary artifacts, and then mark the job cancelled.
3. The request waits asynchronously for up to ten seconds. If a worker has not stopped, it returns a pending result and leaves all podcast files in place.
4. Once no running jobs remain, a single cleanup claimant removes the subscription directory and generated feed, regenerates the unified feed once, and deletes the related database rows.
5. Partial filesystem or feed cleanup is recorded as failed and retried idempotently by the processor loop. A process interruption during cleanup can be reclaimed after five minutes.
