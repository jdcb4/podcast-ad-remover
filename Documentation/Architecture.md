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
- `app/core/artwork.py`: safe source-image retrieval and cached ad-free artwork generation.
- `app/core/rss_gen.py`: generated feed output.
- `app/core/feed.py`: feed parsing.
- `app/core/sources.py`: formal RSS/YouTube adapter boundary for source resolution, discovery, and media download.
- `app/core/youtube.py`: strict public YouTube URL handling, bounded discovery, and audio-only yt-dlp downloads.
- `app/core/sponsorblock.py`: read-only, fail-open SponsorBlock timestamp lookup.
- `app/core/subscription_settings.py`: effective per-podcast setting resolution.

### Text Analysis Providers

Gemini remains the default provider. The custom provider is a separate, opt-in OpenAI-compatible
configuration with its own base URL, model cascade, and optional credential. It never inherits the
OpenAI provider credential. Keyless endpoints receive an internal non-secret SDK placeholder because
the OpenAI client requires a non-empty key value.

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

Subscriptions carry a source type and canonical external identity. Existing rows remain `rss`.
YouTube channel aliases resolve to a canonical channel ID and explicit playlists to their list ID,
preventing duplicate global subscriptions. `source_items` records provider membership independently
of retained episode media, so an old video newly added to a playlist queues once and removed playlist
members are not deleted or rediscovered. YouTube output retention uses discovery time while RSS and
the generated feed continue to expose original publication/upload dates.

The dashboard defaults logged-in users to a "My Podcasts" view backed by `user_subscriptions`, with a Library view for all global podcasts. Adding an existing podcast from search or the Library only adds that global podcast to the user's list.

`subscriptions.owner_user_id` records the user who first added a podcast. Admins can reassign or clear a podcast owner and can change settings for any podcast. Assigning a new owner also adds that podcast to the new owner's My Podcasts list. The owner can change settings for their podcast while they own it. Other users can view, subscribe and refresh discovery. Episode processing, cancellation and file removal require the podcast owner or an admin, as do per-podcast settings. Only admins can delete the global podcast and local files; when an owner removes a podcast from their own list, the podcast becomes unowned instead.

Library membership changes use a JSON response when requested by the dashboard. My Podcasts and
Library navigation progressively enhances ordinary links: the browser fetches the server-rendered
view and replaces only the podcast-result region, preserving the toolbar, active filters, display
mode, and scroll position. Failed enhancement falls back to normal navigation.

The compact table displays retention and inheritance sources and supplies row selection for bulk
updates. The server validates every selected podcast before opening one SQLite write transaction, so
a mixed unauthorised selection changes nothing. Owners can bulk-edit podcasts they manage; ownership
reassignment remains admin-only. Administrators can also bulk-delete selected podcasts. The dashboard
shows a destructive-action warning and requires an explicit confirmation before submitting, while the
server requires a separate confirmation value and starts the same durable, retryable cleanup lifecycle
used by single-subscription deletion.

### Subscription Setting Inheritance

Subscriptions have four explicit inheritance flags:

- `inherit_content_removal`
- `inherit_retention`
- `inherit_default_features`
- `inherit_custom_instructions`

`SubscriptionRepository` resolves inheriting groups against the current `app_settings` row whenever
it returns a subscription. The stored podcast-specific values remain available as internal
`setting_overrides`; changing global settings affects inheriting podcasts immediately, and disabling
inheritance restores the stored values. New subscriptions inherit all groups.

Podcast detail controls show the effective global values while a group inherits. The browser keeps
the stored overrides separately in data attributes, so disabling inheritance restores those values
instead of converting the last global value into a podcast override. Processor feed discovery uses
repository-resolved subscriptions, while retention cleanup resolves the same flags and current
global values in its SQLite queries.

Migration `20260725_0010_subscription_setting_inheritance` is additive. Existing settings remain
explicit except blank or NULL custom instructions, which migrate to inheritance. This preserves the
old meaning of blank custom instructions without treating NULL booleans as an inheritance signal.

The default-features group contains description rewriting, audio summary, title intro, and artwork
badging. The legacy `append_summary` umbrella flag remains explicit for compatibility and is
suppressed while the default-features group inherits.

### Derived Podcast Artwork

When the effective artwork-badge setting is enabled, `app/core/artwork.py` retrieves HTTP(S) source
artwork with redirect and size validation, composites the bundled `AD FREE` badge, and atomically
caches a JPEG under `/data/artwork/`. Output is capped at 1400x1400 and encoded at quality 82 to
reduce client downloads. Source aspect ratios are preserved and small images are not enlarged.
Generated feeds and the web UI use the local `/artwork/{id}.jpg` route with a content
hash cache key; the encoder settings are part of that hash, so changing them republishes every URL.
The older `/artwork/{id}.png` route is still served for clients holding cached feed XML. Disabling
the feature clears the derived file and restores the source artwork URL. An artwork failure is
logged but does not prevent subscription creation or feed processing.

Existing PNGs remain available until the JPEG path and hash commit successfully to SQLite.
Legacy cleanup is retried on a cache hit if it was interrupted after the commit. To convert
existing subscriptions after upgrading, save the global subscription settings once; this
reconciles artwork and republishes feeds with the new cache keys.

### Job State

Processing is coordinated through a durable SQLite `jobs` table. Episodes still keep a user-facing `episodes.status`, while workers claim due jobs transactionally and update job state as work runs, retries, completes, or is cancelled.

Active job columns include:

```text
jobs(id, episode_id, type, status, priority, attempts, locked_at, locked_by, cancel_requested, next_run_at, provider_call_count, reserved_bytes, work_directory, error, created_at, updated_at)
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

Manual downloads and reprocess actions must use repository status helpers that enqueue a `jobs` row, not raw episode status updates. Queue, legacy API and v1 cancellation are non-destructive: they return the episode to
`unprocessed`, cancel queued work, and set `cancel_requested` on a running claim. The worker
retains its lock until acknowledgement. Every worker mutation checks its unique claim token,
current episode state and active subscription in the same SQLite transaction. Stale workers
cannot publish or overwrite a newer attempt. Atomic claims enforce global capacity and scratch
reservations, including competing processes.

Each attempt stages its own files. Source fingerprints and cache provenance allow safe retries;
reprocessing leaves the prior publication readable. After output duration validation, one guarded
transaction switches artifact pointers and marks publication pending. RSS writers serialize the
complete query/write under a cross-process lock and fsync/replace the XML atomically. A feed error
leaves audio intact; the processor retries only publication. See `RECOVERY.md` for manual recovery.

The processor writes a persisted heartbeat and actual scheduler deadlines. The web parent
supervises its child; `/health` combines database readiness and worker liveness. Use one web worker
per data directory. `PROCESSOR_ENABLED=false` disables scheduled work and explicitly reports it;
manual actions share one bounded in-process runner.

Subscription deletion uses a durable two-phase lifecycle because the web app and processor run in separate processes. The first SQLite transaction sets `subscriptions.is_active = 0`, records `deletion_status = pending`, marks every episode ignored, cancels queued/retry jobs, and leaves running jobs locked until their workers acknowledge cancellation. Job repair and claiming only consider active subscriptions that are not being deleted.

After a bounded asynchronous wait, cleanup is claimed through the subscription row and runs outside the FastAPI event loop. It removes the contained podcast directory and feed file, regenerates the unified feed once, and then removes the subscription, episode, and job rows. Failures leave the inactive subscription in `deletion_status = failed`; the processor loop retries failed or interrupted cleanup idempotently. A stale `cleaning` claim can be reclaimed after five minutes.

### Unified Feed Preferences

The unified feed remains available at `/feed/unified.xml`. Administrators can change its channel
title and description, choose whether item titles use the `[Podcast Name] Episode Title` prefix, and
provide an optional external HTTP(S) channel-artwork URL from **Podcast Preferences > Unified
Feed**. Defaults preserve the original generated RSS output, and clearing the external artwork URL
restores the bundled cover. The server validates but does not retrieve external unified-feed
artwork; the URL must therefore be reachable by each podcast client.

The settings-page feed address uses the same session feed token as dashboard subscription links
when feed authentication is enabled. Public Subscribe links remain unauthenticated. Metadata
validation rejects XML-invalid characters; resolution removes such characters from older saved
preferences so the RSS remains readable. HTTPS and same-origin HTTP artwork can be previewed;
other HTTP artwork keeps its direct RSS URL and shows an explanatory message in the web UI.
The bundled preview uses a local static path, and the page's content security policy is unchanged.

Per-episode unified-feed descriptions continue to identify the source podcast, and item artwork
continues to use the corresponding podcast artwork. Presentation-setting changes regenerate only
the unified RSS file and do not reprocess audio.

Migration `20260824_0013_unified_feed_preferences` adds four columns to `app_settings`. Its full
identifier is retained for compatibility with existing PR #20 installations and is distinct from
`20260905_0013_processing_recovery`. Both migration histories can upgrade without losing saved
preferences. See [Recovery](RECOVERY.md) for the backup and rollback procedure.

### Feed Access

RSS feeds and audio files remain public when feed authentication is disabled. When feed authentication is enabled, generated dashboard links use bearer tokens:

```text
/feeds/<slug>.xml?token=<generated-token>
```

Tokens are stored as SHA-256 hashes in `feed_tokens` and can be listed or revoked from the admin Feed Access page. Basic Auth and the older `?auth=base64(username:password)` format are still accepted for compatibility with existing podcast-client subscriptions.

Dashboard and public subscribe pages build links through one server-side helper so tokenized feed
URLs are encoded consistently. The visible choices are Direct link and Use your favourite app. The
latter opens a shared copy-and-paste guide because adding a private/custom RSS URL is the common,
reliable workflow across clients. Older Apple, Pocket Casts, Overcast, Castbox, and Podcast Addict
URLs remain available for backward compatibility, but uncertain platform-specific deep links are no
longer promoted in the main UI.

### Dashboard Refresh Islands

The dashboard remains a FastAPI/Jinja application rather than a single-page app. Small
progressively enhanced regions own live behavior:

- `/api/dashboard/queue` returns only safe queue fields; the queue renderer updates that panel on
  its saved schedule and leaves the rest of the page untouched.
- My Podcasts and Library replace only `dashboard-podcast-results`, then reapply the existing
  client-side filter, sort, and layout preferences.
- ordinary links and a visible full Refresh action remain available when JavaScript or a partial
  request fails.

This approach addresses independent dashboard updates without introducing client-side routing,
duplicating all server templates in JavaScript, or changing the Docker deployment model. A full SPA
would be a substantially larger project involving an API contract for every dashboard action,
client-side rendering and state management, authentication/error handling changes, and parallel
accessibility and browser-test coverage; it is not currently justified.

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
      episode-<database-id>/
        attempt-<unique-token>/
          validated audio, report, transcript, cache provenance
      <legacy-guid-slug>/
        historical artifacts (read compatibility)
  feeds/
    generated RSS files
  artwork/
    cached ad-free podcast artwork
  models/
    downloaded local model files
  app.log
```

`app/core/artifacts.py` owns new ID-based directory identity and compatible legacy lookups.
Recorded artifact paths are authoritative; legacy fallback is constrained to contained paths.
Deleting an episode removes its contained ID root after worker acknowledgement. Published
revisions remain available until episode retention/deletion; abandoned unpublished attempts have
48-hour retention. Existing data is not bulk-renamed.

Small shared modules isolate permissions (`permissions.py`), safe HTTP redirects (`http_downloads.py`),
report escaping (`reports.py`), atomic publication (`publication.py`), worker liveness
(`worker_health.py`), request budgets (`provider_budget.py`) and online backups (`infra/backup.py`).
Initial and incremental episode cards share `_episode_cards.html`; `static/js/episodes.js` owns
requests and interactions. No frontend framework, external queue or new database service is needed.

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
