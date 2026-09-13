# Data Flow

## 1. Subscription & Polling
1.  **User** searches for a podcast or pastes a direct RSS, YouTube channel, or explicit YouTube playlist URL.
2.  **System** saves one global podcast row to `subscriptions`, or reuses the existing global row if the feed is already known. New rows inherit the current global processing-workflow, content-removal, retention, default-feature, and custom-instruction groups. Pre-migration podcasts stay explicitly on Legacy until opted in.
3.  **System** adds the podcast to the user's `user_subscriptions` list. New podcasts record the first adding user as `subscriptions.owner_user_id`.
4.  **Scheduler** wakes up (e.g., every hour) and iterates active subscriptions.
5.  **Source adapter** fetches RSS entries or performs bounded YouTube discovery (50 recent channel entries or 500 flat playlist members).
6.  **System** compares remote episodes with `episodes` table (by GUID).
7.  **System** queues new episodes for processing. New Complete Timeline jobs freeze their prompt, schema, model cascade and removal choices without credentials. Existing jobs keep their workflow; NULL snapshots mean Legacy.

## 2. Episode Processing Pipeline
For each queued episode:

1.  **Download**:
    - Fetch RSS audio from its enclosure, or use pinned yt-dlp to download a YouTube video's best audio-only format.
    - Claim a durable job with a unique token and scratch reservation. Save new artifacts under `/data/podcasts/{podcast_slug}/episode-{database_id}/attempt-{unique_token}/`; keep prior published paths intact.

2.  **Transcribe (Whisper)**:
    - Load Whisper model (if not loaded).
    - Process audio file -> generate text segments with timestamps.

3.  **Classification (configured LLM)**:
    - Legacy: send the transcript with the effective prompts/removal settings and validate a JSON array of finite, ordered, known-label intervals. A valid empty array means no cuts. Existing whitelist behaviour is preserved.
    - Complete Timeline: add explicit gaps through the measured episode duration, send numbered items separately from the rules, and request complete ID ranges plus a summary. Validate exact coverage and known labels, then map IDs to source boundaries. Schema constraints are requested according to the selected output mode.
    - Both use durable request budgets. Invalid/refused/truncated classifications fail analysis. Cascades stay within the selected provider.

4.  **Optional SponsorBlock Evidence**:
    - Only for YouTube episodes and only when `SPONSORBLOCK_ENABLED=true`, query read-only crowdsourced timestamps for categories enabled by the podcast's existing removal settings.
    - Fail open and combine valid timestamps with selected LLM intervals on the original media timeline, retaining separate evidence.

5.  **Ad Removal (FFmpeg)**:
    - Complete Timeline first selects categories according to frozen removal preferences. Then it bridges only retained islands strictly shorter than the configured threshold between two selected cuts. Zero disables bridging. Reports preserve model classifications and extra island cuts separately. Legacy keeps its existing merge policy.
    - Calculate "keep" segments (total duration minus ad segments).
    - Use FFmpeg to cut and concatenate "keep" segments.
    - If global warning tones are enabled, splice bundled beginning/ending cues only at removed edges and one low cue at each interior removal seam. No runtime synthesis is used.
    - Save processed audio in the episode artifact directory.

6.  **Finalize**:
    - Complete Timeline reuses its combined summary for enabled description/TTS features; Legacy retains separate summary generation. A summary-only format failure can be repaired without rerunning valid classification.
    - Validate output MP3 duration (non-MP3 no-cut sources are encoded to MP3).
    - Switch published artifact pointers and stats in one claim-guarded SQLite transaction.
    - Serialize and atomically replace podcast/unified RSS; clear the matching publication-pending flag only after success.
    - Keep completed audio if RSS fails. Retry publication without another download, model call or cut.
    - Reuse only source-fingerprint/model/prompt-compatible stage artifacts on retry; clean owned temporary files and later expire abandoned unpublished attempts.

## 3. Consumption
1.  **User** points their Podcast Player to `http://{host}/feeds/{podcast_slug}.xml`.
2.  **Player** requests the feed.
3.  **System** serves the static XML file.
4.  **Player** requests an episode.
5.  **System** serves the processed audio file from the stored episode path.

The public feed/audio path is not tied to a logged-in account by default. Admin-visible podcast stats can show how many user libraries include each podcast and the existing aggregate episode play count. Per-user download attribution would require token-attributed audio access logging and is not currently part of the data flow.

## 4. Effective Settings And Artwork

1. Repository reads resolve each inherited setting group against the current global settings.
2. Explicit podcast values stay stored while inheritance is enabled; they become effective again if the group toggle is disabled.
3. Global-setting changes therefore affect all inheriting podcasts without rewriting their rows. Complete Timeline classification settings already frozen in queued jobs stay unchanged; future jobs use the new effective settings.
4. If artwork badging is effectively enabled, validated source art is composited with the bundled badge and cached under `/data/artwork/`.
5. Feed generation uses the derived artwork URL and its content hash. Disabling the feature clears the cached derivative and restores source artwork.

## 5. Subscription Deletion

1. The delete request atomically deactivates the subscription, marks its episodes ignored, and cancels all queued or retryable jobs.
2. Running jobs retain their worker lock while cancellation is requested. Workers stop at their next safe checkpoint, remove worker-owned temporary artifacts, and then mark the job cancelled.
3. The request waits asynchronously for up to ten seconds. If a worker has not stopped, it returns a pending result and leaves all podcast files in place.
4. Once no running jobs remain, a single cleanup claimant removes the subscription directory, generated feed, and derived artwork, regenerates the unified feed once, and deletes the related database rows.
5. Partial filesystem or feed cleanup is recorded as failed and retried idempotently by the processor loop. A process interruption during cleanup can be reclaimed after five minutes.

## 6. Budgets and observability

A job retains its provider request count across automatic retries. SDK retries are disabled and
explicit request timeouts bound each call. Reported input/output tokens and request outcomes are
stored without prompts or secrets; the queue shows 24-hour totals, not a billing estimate.
Admission reserves estimated scratch space across workers and stages recheck free space. Other
host writers can still exhaust the disk; failures retain the last published audio and safe retry cache.
