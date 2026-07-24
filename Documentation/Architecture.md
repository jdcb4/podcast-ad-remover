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
- Gemini, OpenAI, Anthropic, OpenRouter, or an explicitly configured OpenAI-compatible endpoint for LLM-backed segment detection and summaries.
- Piper or Gemini TTS for optional spoken title intros and audio summaries.
- Tailwind CSS for styling.
- Docker for deployment.

## Main Components

### Web App

`app/main.py` creates the FastAPI app, configures middleware and routes, and starts the background processor process.

Key areas:
- `app/web/router.py`: HTML UI routes and admin actions.
- `app/web/templates/`: Jinja templates.
- `app/api/`: dashboard API, audio endpoints, and the optional AI-facing `/api/v1` REST API.
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

### Text Analysis Providers And Chunking

Gemini remains the default provider. The custom provider is a separate, opt-in OpenAI-compatible
configuration with its own base URL, model cascade, and optional credential. It never inherits the
OpenAI provider credential. Keyless endpoints receive an internal non-secret SDK placeholder because
the OpenAI client requires a non-empty key value.

Ad-detection chunking is global and disabled by default. When enabled, `AdDetector` estimates the
complete prompt against the configured context window, preserves the one-request path when it fits,
and otherwise partitions timestamped Whisper segments into bounded sequential requests with limited
overlap. Results are clipped to each chunk's primary ownership window and same-label overlaps are
merged deterministically. A failed, rate-limited, or malformed chunk fails the whole analysis.

The summary generator is not chunk-aware and still uses a single truncated full-transcript prompt.
The processor consequently disables description rewrites and audio summaries whenever ad-detection
chunking is enabled. It retains per-podcast preferences for use if chunking is turned off later.

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

Subscription deletion uses a durable two-phase lifecycle because the web app and processor run in separate processes. The first SQLite transaction sets `subscriptions.is_active = 0`, records `deletion_status = pending`, marks every episode ignored, cancels queued/retry jobs, and leaves running jobs locked until their workers acknowledge cancellation. Job repair and claiming only consider active subscriptions that are not being deleted.

After a bounded asynchronous wait, cleanup is claimed through the subscription row and runs outside the FastAPI event loop. It removes the contained podcast directory and feed file, regenerates the unified feed once, and then removes the subscription, episode, and job rows. Failures leave the inactive subscription in `deletion_status = failed`; the processor loop retries failed or interrupted cleanup idempotently. A stale `cleaning` claim can be reclaimed after five minutes.

### Feed Access

RSS feeds and audio files remain public when feed authentication is disabled. When feed authentication is enabled, generated dashboard links use bearer tokens:

```text
/feeds/<slug>.xml?token=<generated-token>
```

Tokens are stored as SHA-256 hashes in `feed_tokens` and can be listed or revoked from the admin Feed Access page. Basic Auth and the older `?auth=base64(username:password)` format are still accepted for compatibility with existing podcast-client subscriptions.

Dashboard and public subscribe pages build podcast-client links through one server-side helper so tokenized feed URLs are encoded consistently. Direct RSS and Overcast links are emitted directly. Apple, Pocket Casts, Castbox, and Podcast Addict route through local instruction pages; Pocket Casts, Castbox, and Podcast Addict include a clearly labelled best-effort app link before the manual RSS instructions.

### AI API Access

The AI-facing REST API is disabled by default and lives under `/api/v1`. It uses scoped bearer tokens stored in `api_tokens` as SHA-256 hashes and SQLite-backed request counters in `api_rate_limits`.

API tokens are independent from dashboard sessions and feed tokens. Dashboard authentication middleware bypasses `/api/v1/*` after the global IP allowlist so API clients receive JSON `401`, `403`, and `429` responses instead of browser login redirects. Feed tokens continue to grant only RSS/audio access.

Initial API scopes are `read`, `write`, `process`, and `admin`. Hard global podcast deletion is intentionally not exposed in API v1.

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
