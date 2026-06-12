# Audit Status

Date: 2026-06-11
Branch: `audit-work`

This document tracks the current audit branch findings, what was implemented, and what is intentionally deferred. It is not a release note; use `CHANGELOG.md` for release-facing changes.

## Priorities

- Preserve existing `/data/db/podcasts.db` databases and podcast files.
- Keep the default homelab/localhost experience simple.
- Make optional authentication robust enough that the public subscribe page can reasonably sit behind a reverse proxy.
- Avoid rebuilding the app until incremental improvements stop buying down real risk.

## Implemented In This Branch

### Functionality And UX

- Added a read-only public subscribe page at `/subscribe`.
- Added a login-page link to `/subscribe` when the public subscribe page is enabled.
- Clarified destructive action labels for removing downloads, ignoring episodes, reprocessing, and deleting subscriptions.
- Added a setup checklist for admin credentials, base URL, subscribe page, and feed URL checks.
- Added live queue polling and an operation dashboard for current job, resource, retry, and failure state.
- Fixed API subscription creation to preserve feed descriptions from `FeedManager.parse_feed()`.

### Reliability

- Added a formal migration table and migration backup path before new schema work.
- Added synthetic legacy database migration coverage to prove old subscription, setting, and episode rows survive startup migration.
- Added a migration dry-run helper that applies startup migrations to a copied database without touching the source.
- Added a durable `jobs` table and atomic job claiming for processor work.
- Kept `episodes.status` as the user-facing episode state for compatibility.
- Added WAL and a 30-second SQLite busy timeout for both startup migrations and runtime DB access.
- Added stale running job recovery so interrupted workers do not permanently consume queue capacity.
- Wrote downloads to `.part` files first and moved them into place only after completion.
- Added stale temporary processor artifact cleanup for old `.part` and `.tmp.mp3` files.
- Removed whole-file log rewriting; rotating log handlers own log size.
- Added containment checks before deleting episode directories or serving audio files.
- Hardened feed and audio URL redirect validation when private-network feed access is disabled.
- Hardened RSS CDATA serialization for descriptions containing `]]>`.
- Hardened audio segment keep-window calculation for overlapping, unsorted, out-of-range, or malformed model output.
- Hardened processor ad-segment post-processing so whitelist inversion and close-gap merging are deterministic and covered.

### Security

- Added optional feed/audio authentication with generated feed tokens while keeping legacy Basic Auth and query links compatible.
- URL-encoded injected feed access parameters in RSS enclosure URLs and shared the logic between individual and unified feeds.
- Added feed-token visibility and revocation from the admin Feed Access page.
- Added UI warnings that protected feed URLs containing generated tokens are bearer secrets until revoked.
- Kept public feed and subscribe access available when feed auth is disabled.
- Added `TRUST_PROXY_HEADERS` so reverse proxy headers are ignored by default and only trusted when explicitly enabled.
- Added CIDR support to the IP allowlist.
- Applied the IP allowlist before public feed/audio/subscribe bypasses.
- Added same-origin checks for authenticated mutating management requests.
- Restricted System Settings `redirect_to` targets to local app paths.
- Stopped storing plaintext dashboard passwords in session cookies.
- Added startup validation so dashboard or feed authentication cannot run with the default session secret.
- Added System Settings warnings and form guards so dashboard or feed authentication cannot be enabled while `SESSION_SECRET_KEY` is still the default.
- Added a setup checklist warning when authentication is enabled on an HTTPS base URL while `COOKIE_SECURE=false`.
- Hashed newly saved standalone feed Basic Auth passwords while accepting legacy plaintext values.
- Added an explicit System Settings error when standalone feed authentication is enabled without a username and password.
- Added route-level auth/admin dependencies to management endpoints as defense in depth.
- Escaped AI summary markdown output before applying the supported formatting subset.
- Escaped dynamic podcast search results and lazy-loaded episode card fields before inserting client-rendered HTML.
- Escaped dynamic AI model names, log lines, and prompt alerts before client-side HTML insertion.

### Resource Usage

- Removed unused PyTorch packages from the production image.
- Excluded local development artifacts from Docker builds.
- Added configurable Whisper CPU threads, FFmpeg threads, and optional Whisper unload after the queue empties.
- Added per-category `/data` storage reporting for podcast files, models, feeds, database, backups, and logs.
- Added a production Compose file that uses the published image and only mounts `/data`.
- Added `RESOURCE_AUDIT.md` with live measurement commands and current findings.

### Maintainability And Verification

- Added regression tests for migrations, job claiming, feed parsing and size limits, episode download response guardrails, audio segment math, processor ad-segment transitions, URL validation, RSS generation, template filters, auth helpers, auth middleware contracts, audio path containment, and processor maintenance helpers.
- Added feed-auth middleware regression coverage for generated token access, revoked-token rejection, and hashed standalone Basic Auth.
- Added regression coverage for URL-encoded feed access injection in RSS enclosure URLs.
- Aligned the `Episode` Pydantic model with retry, manual-download, and listen-count columns from the live SQLite schema.
- Added regression tests for default session-secret fail-closed behavior when optional auth is enabled.
- Added explicit pytest-asyncio loop-scope configuration to keep async test behavior stable across pytest-asyncio releases.
- Refreshed Browserslist build metadata to the latest npm-available `caniuse-lite` data.
- Fixed npm audit findings in frontend build dependencies.
- Added `npm audit --audit-level=moderate` to the standard verification gate.
- Changed `npm test` from the old placeholder to the standard verification gate.
- Added GitHub Actions PR verification for the standard `npm run verify` gate.
- Added experimental Docker tagging for branch testing without updating `latest`.
- Refactored direct Gemini access through the OpenAI-compatible provider path.
- Moved completed-episode RSS queries into `EpisodeRepository` and added regression coverage.
- Extracted duplicated RSS feed base-URL selection into a shared helper with regression coverage.
- Removed a duplicate unreachable `raise` from audio prepending error handling.

## Deferred

These remain useful, but are not required before trialing the audit branch.

- Split `app/web/router.py` into smaller routers and move more inline JavaScript into static files; this should also make a stricter CSP practical later.
- Split `app/core/processor.py` and `app/core/ai_services.py` into smaller feed sync, download, transcription, ad detection, audio edit, RSS, and retention services after more processor transition tests exist.
- Replace the partial-file download staging with full per-job `.work` directories and orphaned work-folder recovery once processor services are split.
- Add versioned structured processing reports or an `episode_artifacts` table after the worker/job boundary has settled.
- Add a realistic copied-database migration test using a sanitized existing `podcasts.db`.
- Add broader processor transition tests for the full download/transcribe/detect/cut/feed lifecycle.
- Add Ruff and optional type checking once the current large modules have been reduced.
- Add a Python dependency lockfile if release reproducibility becomes a practical problem.

## Not Currently Planned

- Requiring authentication for public subscribe or feed URLs by default. This would work against the primary podcast-client workflow.
- Automatically deriving secure-cookie behavior from `ENVIRONMENT`. The branch keeps an explicit `COOKIE_SECURE` setting because many installs are LAN HTTP or HTTPS-terminated behind a reverse proxy.
- Replacing SQLite before there is evidence that SQLite plus WAL, busy timeout, and durable jobs cannot handle the target deployment.
- Replacing the lightweight `schema_migrations` table with Alembic in this audit pass. Alembic can still be introduced later if schema changes become more complex.
- Replacing the FastAPI/Jinja app with a full frontend rebuild. The current architecture is still serviceable once route and template size are reduced incrementally.
- Removing local Whisper or Piper from the standard image. They are part of the offline/local processing value of the project; smaller image variants can be considered later.

## Current Verification Evidence

Latest local checks on this branch:

- `npm run verify`: passing, 113 tests.
- `npm run verify:docker`: passing, including a local Docker image build tagged `podcast-ad-remover:verify`.
- `npm audit --audit-level=moderate`: included in `npm run verify`, 0 vulnerabilities.
- `docker compose -f docker-compose.prod.yml config --quiet`: passing.
- Throwaway Docker deployment smoke test: passing with a temporary `/data` mount and `BASE_URL=http://localhost:18000`; `/health`, `/login`, `/subscribe`, and `/admin/system` all returned HTTP 200, and the test container/data directory were removed afterward.
- `git diff --check`: passing with Windows CRLF warnings only.

The local Tailwind build can still print a Browserslist old-data warning under the current system date even after `npx update-browserslist-db@latest`; `npm ls caniuse-lite browserslist` shows the installed metadata has been refreshed to the latest available npm versions.
