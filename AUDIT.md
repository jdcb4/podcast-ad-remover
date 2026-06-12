# Podcast Ad Remover Audit

Date: 2026-05-18
Branch: `audit`

## Executive Summary

The application is a solid homelab MVP: it has the core workflow you need, uses SQLite appropriately for a single-user/self-hosted tool, has useful queue controls, and already includes practical features such as feed generation, retry handling, retention cleanup, transcript reuse, multiple AI providers, and local Whisper/Piper processing.

The main risk is that the app has grown past the shape of its current architecture. A few very large files now own unrelated responsibilities: routing, settings, user management, feed generation, queue control, download/transcription, LLM calls, FFmpeg, reports, cleanup, and UI behavior. That makes future fixes harder and increases the chance of regressions.

My recommendation is not a full rewrite first. Keep the existing SQLite database and data directory, then do an incremental rebuild around the current data model: formal migrations, a real job table/worker boundary, safer auth/feed-token handling, and a cleaner UI/API split. A full rebuild is only worth it if you want multi-user/public exposure, mobile-grade UX, or more reliable large-scale processing.

## What Works Well

- The core product loop exists end to end: subscribe to RSS, download episodes, transcribe, detect removable segments, cut audio, regenerate RSS, and serve audio to podcast clients.
- SQLite is a reasonable persistence choice for a homelab app with one main writer and simple operational needs.
- The data directory layout is moving in the right direction: `settings.get_episode_dir()` stores per-podcast/per-episode artifacts under `/data/podcasts`.
- Processing state is visible in the UI through status, step, and progress fields.
- Feed generation uses local files and normal HTTP endpoints, which keeps podcast-client consumption simple.
- The app has useful admin controls for AI provider settings, prompts, global defaults, queue status, logs, users, and access requests.
- Python syntax compilation passes with `python -m compileall -q app`.
- After `npm ci`, the Tailwind CSS build completes with `npm run build:css`.

## Key Findings

### Functionality

1. There is no atomic job claim in the queue. `Processor.process_queue()` counts current jobs, reads pending jobs, then updates them to `processing` in separate steps (`app/core/processor.py:245-270`). This is vulnerable if more than one process or request-triggered processor races the background worker. SQLite can support an atomic `UPDATE ... RETURNING` or transaction-based claim.

2. Manual processor objects are created from API routes (`app/api/subscriptions.py:17-19`, `app/api/subscriptions.py:98-119`) while the real processor runs in a separate process (`app/main.py:82-95`). That split is workable, but right now coordination is implicit through status polling. It should become an explicit job queue.

3. Feed parsing uses `feedparser.parse(url)` directly (`app/core/feed.py:18`, `app/core/feed.py:41`) with no network timeout, size limit, host validation, or cache headers. A slow or hostile feed can hang or consume resources.

4. Audio downloads stream to disk with a 300 second timeout but no maximum size, content-type validation, or minimum free-space check (`app/core/processor.py:363-386`). A bad enclosure URL can fill disk.

5. Cancellation deletes the entire episode directory when status changes away from `processing` (`app/core/processor.py:851-879`). That is simple, but dangerous if the directory path is ever wrong or reused; cancellation should only remove known temporary files or use a staging directory.

6. RSS generation embeds CDATA by writing escaped text and replacing strings afterward (`app/core/rss_gen.py:117-129`, `app/core/rss_gen.py:230-239`). This is brittle for descriptions containing `]]>` and is easy to break with malformed source descriptions.

7. There is documentation drift. The docs still describe `/public` feeds/audio, while code serves `/data/feeds` and `/data/podcasts` (`app/core/config.py:35-64`, `Documentation/Deployment.md`). This will confuse deployment and migration.

### UX

1. The app is feature rich, but the main episode page is overloaded. `episodes.html` is about 1,400 lines and mixes markup, state, fetch calls, mobile actions, filtering, auto-refresh, batch operations, and card rendering.

2. Many actions rely on `confirm()` and delayed reloads, for example reprocess waits 20 seconds before refreshing (`app/web/templates/episodes.html:1389-1398`). Users cannot always tell whether work was queued, started, failed, or blocked on rate limits.

3. Destructive labels are inconsistent: UI text says "Delete File", "Deleted", "Ignore", and backend methods use `delete_episode`, `soft_delete`, and `cancel` for overlapping behaviors. This can make it hard to know whether an episode can be recovered.

4. Feed auth links embed credentials as `?auth=base64(...)` (`app/web/router.py:167-171`, `app/web/router.py:1646-1659`, `app/web/router.py:1734-1753`). This may be convenient for podcast clients, but users should see a warning that these URLs are bearer secrets.

5. The admin and subscription views need clearer operational status: last feed check, next retry time, current worker state, disk free space, estimated processing time, and exactly why a job is waiting.

### Code Quality

1. The largest files are too large for safe ongoing development:
   - `app/web/router.py`: about 1,662 lines.
   - `app/web/templates/episodes.html`: about 1,400 lines.
   - `app/core/processor.py`: about 1,070 lines.
   - `app/core/ai_services.py`: about 914 lines.

2. Database migrations are ad hoc `ALTER TABLE` statements with ignored `OperationalError`s (`app/infra/database.py:168-245`). This keeps old databases limping forward but gives no schema version, no rollback strategy, and no validation that an existing database is actually compatible.

3. Models do not fully match the schema. The database has fields such as `retry_count`, `next_retry_at`, `is_manual_download`, and `listen_count`, but `Episode` omits several of them (`app/core/models.py`). This pushes parts of the app into raw dicts and weakens validation.

4. There is repeated settings access through raw SQL across router, processor, AI services, RSS generation, and middleware. A typed settings repository/service would reduce drift.

5. The API and web layers are blurred. Some `/api/...` endpoints live in `app/web/router.py`, while other API endpoints live in `app/api/subscriptions.py`.

6. There are signs of accidental or rushed code: duplicate returns in `simple_markdown`, duplicate `raise` in `AudioProcessor.prepend_audio()` (`app/core/audio.py:122-125`), comments that disagree with current behavior, and mojibake characters in docs/comments.

7. No automated test suite is present. `package.json` has `"test": "echo \"Error: no test specified\" && exit 1"`. The most important missing tests are migration tests, RSS generation tests, queue state transition tests, and audio segment math tests.

### Security

1. Default session secret is unsafe (`app/core/config.py:13`). In production, the app should fail startup if `SESSION_SECRET_KEY` is still the default and auth is enabled.

2. Session cookies are always configured with `https_only=False` (`app/main.py:127-133`), while `SECURITY.md` says secure cookies are enabled in production. This is a direct documentation/code mismatch.

3. Auth is disabled by default and every visitor becomes admin when disabled (`app/web/auth.py:21-33`, `app/web/auth.py:46-61`). That may be acceptable for a private LAN deployment, but it is dangerous for anything exposed beyond trusted hosts.

4. IP allowlist does not apply to `/feeds`, `/feed`, or `/audio` because those paths skip the auth middleware before allowlist checks (`app/web/auth.py:87-95`). The comment says the allowlist applies to everything, but the code excludes the public feed/audio endpoints.

5. Proxy headers are trusted from any client (`app/web/auth_utils.py:33-44`, `app/api/audio_routes.py:112-115`). A direct client can spoof `X-Forwarded-For` and bypass IP rate limiting or poison listen tracking unless the app is behind a trusted reverse proxy that strips headers.

6. Global feed auth password is stored plaintext (`app/web/router.py:392-394`, `app/web/middleware.py:65-80`) and dashboard passwords are stored in the session as plaintext to generate feed URLs (`app/web/router.py:260-262`). This is the biggest credential-handling issue.

7. Feed auth tokens are just base64 username/password and are accepted in query strings (`app/web/middleware.py:30-45`). Query strings leak through logs, browser history, reverse proxies, and podcast apps.

8. If feed auth is enabled but credentials are missing, the middleware allows access (`app/web/middleware.py:69-71`). Security features should fail closed.

9. CSP allows inline scripts and styles (`app/web/security_headers.py:30-46`) because templates use inline handlers heavily. That limits XSS protection.

10. `simple_markdown()` builds HTML without escaping input (`app/web/router.py:27-79`) and the template renders AI summary with `|safe` (`app/web/templates/episodes.html:663`). LLM output or poisoned transcript-derived text can become stored XSS.

11. SSRF risk exists in feed fetches, podcast search/add flow, enclosure downloads, and model downloads. Feed URLs and enclosure URLs should be validated against private IP ranges if the app can be accessed by untrusted users.

12. `npm audit` reports 3 frontend dev dependency vulnerabilities: high severity `picomatch`, moderate `postcss`, and moderate `yaml`. They are dev/build-chain issues, but should still be fixed.

### Resource Usage

1. Whisper runs CPU-only with `float32` (`app/core/ai_services.py`) and the README already notes that it can saturate small hosts. The app should expose thread/core limits and process niceness as first-class settings.

2. Concurrent processing is controlled by DB setting, but FFmpeg and Whisper each create their own internal thread pools. `concurrent_downloads=2` can mean more than two CPU-heavy workloads.

3. Chunked transcription creates large intermediate WAV/chunk files (`app/core/audio.py:193-237`). This is useful for stability but can spike disk usage. Add free-space checks and cleanup-on-crash recovery.

4. Full log cleanup reads the whole log file into memory (`app/core/processor.py:901-928`) even though a rotating log handler is already configured. This is unnecessary and can block the worker.

5. Listen dedupe uses an in-memory dictionary (`app/api/audio_routes.py:19-49`). It is fine for a single process, but it resets on restart and is not shared across workers.

6. Docker installs `torch`, `torchvision`, and `torchaudio` even though only faster-whisper is used directly (`Dockerfile:11-16`). This increases image size and build time.

7. `docker-compose.yml` bind-mounts the whole repo into `/app` (`docker-compose.yml:10-12`). That is good for development but not for a stable app deployment because it bypasses the image contents and can expose source/config files.

## Recommended Improvement Plan

### Phase 1: Make the Current App Safer Without Breaking Data

1. Add real migration management with a `schema_migrations` table and idempotent migration files. Keep existing table and column names.
2. Add a startup database backup before applying migrations, for example `/data/backups/podcasts-YYYYMMDD-HHMMSS.db`.
3. Fail startup if `SESSION_SECRET_KEY` is the default and auth or feed auth is enabled.
4. Change session cookies to `https_only=settings.ENVIRONMENT == "production"` or add an explicit `COOKIE_SECURE` setting.
5. Replace query-string base64 credentials with generated feed tokens:
   - New table: `feed_tokens(id, user_id, token_hash, name, created_at, last_used_at, revoked_at)`.
   - Keep old query auth temporarily for migration, but show a deprecation warning.
6. Make feed auth fail closed if enabled but credentials/tokens are missing.
7. Apply IP allowlist before skipping feed/audio paths, or create separate `feed_ip_allowlist` behavior.
8. Escape AI/user text before converting markdown, or use a sanitizer such as Bleach with a small allowlist.
9. Add feed/enclosure URL validation and deny loopback, link-local, RFC1918/private ranges unless an explicit "allow private feeds" setting is enabled.
10. Add max download size, content-type checks, and disk free-space checks.

### Phase 2: Stabilize Processing

1. Introduce explicit jobs:
   - `jobs(id, episode_id, type, status, priority, attempts, locked_at, locked_by, next_run_at, error, created_at, updated_at)`.
   - Keep `episodes.status` for user-facing state, but make workers claim `jobs` atomically.
2. Split processor responsibilities:
   - `feed_sync_service.py`
   - `download_service.py`
   - `transcription_service.py`
   - `ad_detection_service.py`
   - `audio_edit_service.py`
   - `rss_service.py`
   - `retention_service.py`
3. Use staging paths such as `episode_dir/.work/{job_id}` and only move final artifacts into place after success.
4. Add a recovery task that finds orphaned `.work` folders and interrupted jobs.
5. Add resource controls:
   - worker concurrency
   - Whisper CPU thread count
   - FFmpeg thread count
   - max simultaneous transcriptions
   - max disk usage or minimum free space
6. Store structured processing reports in the DB or JSON with a version field.

### Phase 3: Clean Up UX

1. Replace delayed reloads with live job state polling or server-sent events.
2. Make queue states explicit: queued, downloading, transcribing, detecting ads, cutting audio, generating feed, rate-limited, retry scheduled, completed, failed, ignored.
3. Add an operation dashboard:
   - active job
   - CPU/memory/disk usage
   - free disk space
   - next feed check
   - retry schedule
   - recent failures
4. Clarify destructive actions:
   - "Remove download but keep episode"
   - "Ignore episode"
   - "Reprocess"
   - "Delete subscription and all local files"
5. Split `episodes.html` into smaller template partials and move JavaScript into static files.
6. Improve first-run setup: require admin password creation, show base URL/feed URL test, and explain whether feeds are public or token-protected.

### Phase 4: Test and Package

1. Add Python tests with pytest:
   - migrations against empty and legacy databases
   - feed parsing with sample RSS files
   - RSS output validity
   - segment removal math
   - queue claiming/retry transitions
   - auth/feed token behavior
2. Add a small test fixture database copied from the current schema with fake podcast data.
3. Add linting/formatting: Ruff, mypy or pyright where practical, and a CI workflow.
4. Pin Python dependency versions or use a lockfile.
5. Fix `npm audit` findings with `npm audit fix` and commit the updated lockfile.
6. Separate development and production compose files:
   - Production: only `./data:/data`.
   - Development: source bind mount and reload.

## Rebuild Recommendation

I would not do a ground-up rewrite immediately. The app already works and your priority is preserving existing subscriptions and processed episodes. A full rewrite would create more migration risk than value at this stage.

The best path is an incremental rebuild inside the same repo:

- Keep FastAPI, SQLite, Jinja/templates initially.
- Preserve `/data/db/podcasts.db`, `/data/podcasts`, and `/data/feeds`.
- Add formal migrations and backup first.
- Introduce a proper job table and worker service.
- Move large modules into services.
- Harden auth and feed token handling.
- Gradually extract frontend JavaScript/CSS without changing database semantics.

Consider a full rebuild only if you want:

- multiple users with separate libraries,
- public internet exposure as a supported deployment mode,
- a richer SPA-style interface,
- distributed workers/GPU machines,
- or very high reliability for many subscriptions.

If rebuilding, I would still keep SQLite by default and add Postgres only as an optional advanced backend. A good rebuilt shape would be:

- FastAPI API layer.
- Server-rendered UI or a small React/Vue/Svelte frontend.
- SQLite with Alembic migrations.
- A job queue backed by SQLite for homelab simplicity.
- Separate worker process with clear job leases.
- Artifact storage under versioned episode directories.
- Token-based feed access.
- Typed settings and repositories.

## Database-Preserving Migration Strategy

1. Freeze the current schema by introspecting existing user databases:
   - `PRAGMA table_info(subscriptions)`
   - `PRAGMA table_info(episodes)`
   - `PRAGMA table_info(app_settings)`
2. Add `schema_migrations(version TEXT PRIMARY KEY, applied_at TIMESTAMP NOT NULL)`.
3. Convert the current ad hoc migrations into numbered migration files.
4. On startup:
   - open DB,
   - create backup,
   - apply unapplied migrations in a transaction,
   - run sanity checks,
   - start worker only after migration success.
5. Do not rename or remove existing columns in the first pass.
6. Add new tables rather than replacing old fields:
   - `jobs`
   - `feed_tokens`
   - `episode_artifacts`
   - optionally `settings_history`
7. Backfill new tables from old columns:
   - create completed artifact rows from `episodes.local_filename`, `transcript_path`, `report_path`, `ad_report_path`.
   - create initial job rows only for `pending`, `failed` with retry due, or `rate_limited` episodes.
8. Keep compatibility reads for old path fields for at least one release.
9. Add a migration dry-run command that copies the DB to a temp file and applies migrations before touching the real DB.

## Suggested Priority List

1. Add formal migrations plus automatic DB backup.
2. Fix session/feed credential handling.
3. Add feed/enclosure URL validation, download size limits, and disk free-space checks.
4. Replace queue claiming with atomic jobs.
5. Split the processor into services.
6. Escape/sanitize markdown and remove unsafe template rendering.
7. Split `router.py` and move `/api` routes into the API package.
8. Split `episodes.html` JavaScript into static files and remove inline handlers.
9. Add pytest coverage for migrations, queue transitions, RSS generation, and audio segment math.
10. Clean deployment files and docs so `/data` layout is consistent.

## Validation Performed

- `python -m compileall -q app`: passed.
- `npm ci`: passed, but reported 3 vulnerabilities.
- `npm audit --audit-level=moderate`: failed due to `picomatch`, `postcss`, and `yaml` advisories.
- `npm run build:css`: passed after `npm ci`, with an outdated Browserslist warning.
