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

3.  **Ad Detection (Gemini)**:
    - Send transcript to Gemini API with a prompt to identify ad segments.
    - Receive JSON response containing start/end times of ads.

4.  **Ad Removal (FFmpeg)**:
    - Calculate "keep" segments (total duration minus ad segments).
    - Use FFmpeg to cut and concatenate "keep" segments.
    - Save processed audio in the episode artifact directory.

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
