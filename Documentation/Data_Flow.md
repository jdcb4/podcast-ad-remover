# Data Flow

## 1. Subscription & Polling
1.  **User** adds a Podcast RSS URL via Web UI.
2.  **System** saves subscription to `subscriptions` table.
3.  **Scheduler** wakes up (e.g., every hour) and iterates active subscriptions.
4.  **Feed Manager** fetches the remote RSS feed.
5.  **System** compares remote episodes with `episodes` table (by GUID).
6.  **System** queues new episodes for processing.

## 2. Episode Processing Pipeline
For each queued episode:

1.  **Download**:
    - Fetch audio from `enclosure` URL.
    - Save to `/data/downloads/{uuid}.mp3`.

2.  **Transcribe (Whisper)**:
    - Load Whisper model (if not loaded).
    - Process audio file -> generate text segments with timestamps.

3.  **Ad Detection (Gemini)**:
    - Send transcript to Gemini API with a prompt to identify ad segments.
    - Receive JSON response containing start/end times of ads.

4.  **Ad Removal (FFmpeg)**:
    - Calculate "keep" segments (total duration minus ad segments).
    - Use FFmpeg to cut and concatenate "keep" segments.
    - Save output to `/data/public/audio/{podcast_slug}/{uuid}.mp3`.

5.  **Finalize**:
    - Update database with processing stats (time saved, ad count).
    - Delete temporary download file.
    - Regenerate the Podcast's local RSS feed XML in `/data/public/feeds/`.

## 3. Consumption
1.  **User** points their Podcast Player to `http://{host}/feeds/{podcast_slug}.xml`.
2.  **Player** requests the feed.
3.  **System** serves the static XML file.
4.  **Player** requests an episode.
5.  **System** serves the static MP3 file from `/data/public/audio/...`.
