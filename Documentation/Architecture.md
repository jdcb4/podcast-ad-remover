# Architecture

## Overview

Podcast Ad Remover is a Dockerized FastAPI application that subscribes to podcast RSS feeds, downloads episodes, processes the audio to remove ads or promotional segments, and republishes replacement RSS feeds for podcast clients.

The application is intentionally simple: one web app, one SQLite database, local filesystem storage under `/data`, and a separate processor process launched by the app at startup.

## Technology Stack

- Python 3.11.
- FastAPI for web and API routes.
- Jinja templates for the server-rendered UI.
- SQLite for application state.
- FFmpeg for audio processing.
- Whisper/faster-whisper for local transcription.
- Gemini, OpenAI, Anthropic, or OpenRouter for LLM-backed segment detection and summaries.
- Piper or Gemini TTS for optional spoken title intros and audio summaries.
- Tailwind CSS for styling.
- Docker for deployment.

## Main Components

### Web App

`app/main.py` creates the FastAPI app, configures middleware and routes, and starts the background processor process.

Key areas:
- `app/web/router.py`: HTML UI routes and admin actions.
- `app/web/templates/`: Jinja templates.
- `app/api/`: subscription and audio endpoints.
- `app/web/auth.py` and `app/web/middleware.py`: authentication and request handling.

### Processing Core

`app/core/processor.py` coordinates the episode lifecycle:

1. Discover episodes from subscribed feeds.
2. Download source audio.
3. Transcribe locally.
4. Ask the configured LLM provider to identify removable segments.
5. Cut and concatenate audio with FFmpeg.
6. Update SQLite state and regenerate RSS feeds.

Supporting modules include:
- `app/core/audio.py`: FFmpeg helpers.
- `app/core/ai_services.py`: provider integrations, transcription, summaries, and TTS.
- `app/core/rss_gen.py`: generated feed output.
- `app/core/feed.py`: feed parsing.

### Text-To-Speech

TTS is only used for optional spoken title intros and audio summaries. `app_settings.tts_provider` selects the engine:

- `piper`: default local/offline provider. It uses the configured `piper_model` and stores downloaded voice models under `/data/models/piper`.
- `gemini`: optional API-backed provider. It reuses saved Gemini API keys, sends speech requests through Google's REST `generateContent` endpoint, tries `gemini_tts_model_cascade` in order, and writes returned 24 kHz mono PCM as a WAV file for FFmpeg.

The currently exposed Gemini voices are `Orus`, `Enceladus`, and `Laomedeia`.

### Infrastructure

- `app/core/config.py`: environment settings and canonical filesystem paths.
- `app/infra/database.py`: SQLite initialization and schema evolution.
- `app/infra/repository.py`: database access methods.

### Podcast Library and Ownership

`subscriptions` is the global podcast library. There is still only one row, episode set, media directory, and generated RSS feed per podcast. User-specific interest is tracked separately in:

```text
user_subscriptions(user_id, subscription_id, added_at)
```

The dashboard defaults logged-in users to a "My Podcasts" view backed by `user_subscriptions`, with a Library view for all global podcasts. Adding an existing podcast from search or the Library only adds that global podcast to the user's list.

`subscriptions.owner_user_id` records the user who first added a podcast. Admins can reassign or clear a podcast owner and can change settings for any podcast. Assigning a new owner also adds that podcast to the new owner's My Podcasts list. The owner can change settings for their podcast while they own it. Other users can view, subscribe, refresh, and trigger downloads, but cannot change per-podcast settings. Only admins can delete the global podcast and local files; when an owner removes a podcast from their own list, the podcast becomes unowned instead.

### Job State

Processing is coordinated through a durable SQLite `jobs` table. Episodes still keep a user-facing `episodes.status`, while workers claim due jobs transactionally and update job state as work runs, retries, completes, or is cancelled.

Active job columns include:

```text
jobs(id, episode_id, type, status, priority, attempts, locked_at, locked_by, next_run_at, error, created_at, updated_at)
```

Current job statuses are:

- `queued`
- `running`
- `retry_scheduled`
- `rate_limited`
- `completed`
- `failed`
- `cancelled`

Startup migration creates a `schema_migrations` table and backs up the current database to `/data/backups/` before applying formal migrations.

Manual downloads and reprocess actions must use repository status helpers that enqueue a `jobs` row, not raw episode status updates. Queue cancellation is non-destructive: it marks the active job cancelled and returns the episode to `unprocessed`; deleting or ignoring an episode remains a separate action.

### Feed Access

RSS feeds and audio files remain public when feed authentication is disabled. When feed authentication is enabled, generated dashboard links use bearer tokens:

```text
/feeds/<slug>.xml?token=<generated-token>
```

Tokens are stored as SHA-256 hashes in `feed_tokens` and can be listed or revoked from the admin Feed Access page. Basic Auth and the older `?auth=base64(username:password)` format are still accepted for compatibility with existing podcast-client subscriptions.

### Access Requests

Users requesting dashboard access choose a password during the request. The pending `access_requests` row stores only `password_hash`; admins can approve or deny the request but do not see or transmit the user's password. On approval, the stored hash is copied into the new `users` row.

### Notifications

Admin notifications are optional and disabled by default. Settings are stored in `app_settings` and include a newline-separated list of Apprise URLs plus per-event toggles.

The app currently emits notification events for:

- access requests submitted from `/request-access`;
- new global podcasts after feed metadata is resolved;
- completed episodes after processing succeeds and feeds are regenerated;
- breaking processing errors such as max-retry episode failures, missing subscription rows during processing, and top-level background worker loop errors.

Notification delivery uses the `apprise` Python library directly in the app process. Notification failures are logged and do not block access requests, podcast creation, or episode processing.

## Data Layout

Persistent data should be mounted at `/data`.

```text
/data/
  db/
    podcasts.db
  podcasts/
    <podcast_slug>/
      <episode_slug>/
        episode artifacts
  feeds/
    generated RSS files
  models/
    downloaded local model files
  app.log
```

Deprecated path helpers still exist for older code paths, but new work should use the podcast/episode directory structure exposed by `settings.get_episode_dir()`.

## Episode Statuses

The main episode status values are:

- `pending`
- `unprocessed`
- `processing`
- `completed`
- `failed`
- `rate_limited`
- `ignored`

There is also legacy handling for `pending_manual`. See `Documentation/NAMING.md` before adding or renaming statuses.

## Release Architecture

Release images are built from the repository Dockerfile and published to Docker Hub as:

```text
jdcb4/podcast-ad-remover:<version>
jdcb4/podcast-ad-remover:latest
```

Versioning and verification rules live in `Documentation/VERSIONING.md` and `Documentation/VERIFICATION.md`.
