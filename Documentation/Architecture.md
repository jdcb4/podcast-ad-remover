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
- `app/core/ai_services.py`: provider integrations, transcription, and summaries.
- `app/core/rss_gen.py`: generated feed output.
- `app/core/feed.py`: feed parsing.

### Infrastructure

- `app/core/config.py`: environment settings and canonical filesystem paths.
- `app/infra/database.py`: SQLite initialization and schema evolution.
- `app/infra/repository.py`: database access methods.

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
