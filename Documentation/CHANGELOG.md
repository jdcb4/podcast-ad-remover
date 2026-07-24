# Changelog

## Unreleased

- Added an opt-in custom OpenAI-compatible text-analysis provider with arbitrary API base URL/model slugs, keyed or keyless endpoint support, strict URL validation, separate credentials, and Docker networking guidance.
- Added opt-in context-budget transcript chunking for ad detection with a single-request short path, bounded overlap and request caps, deterministic same-label merging, strict complete-failure semantics, and safe report metadata.
- Recorded aggregate provider token usage across ad-detection chunks when the provider returns usage data.
- Added an explicit 1,024-token ad-detection completion cap and a more conservative timestamped-prompt estimator after constrained-context OpenRouter validation exposed provider-side context reservation failures.
- Added an option to omit per-segment detection reasons for lower-output local models while keeping reasons enabled by default.
- Normalized common case and naming variants for supported segment labels so smaller models do not break whitelist filtering with responses such as `content` or `advertisement`.
- Prevented saved AI credentials from being rendered back into admin forms and made AI settings updates section-specific so saving one page cannot reset another.
- Disabled AI description rewrites and audio summaries while transcript chunking is enabled, with processing enforcement and explanatory UI while retaining podcast preferences.

## 1.8.0 - 2026-07-22

- Fixed subscription deletion during active processing by atomically deactivating the podcast, cancelling queued work, waiting a bounded time for running workers to acknowledge cancellation, and deferring retryable file cleanup when a worker is still active.
- Moved subscription file and feed cleanup off the FastAPI event loop, prevented new episode claims after deletion starts, and regenerated the unified feed once after successful cleanup.
- Added durable subscription-deletion state and regression coverage for concurrent deletion, route responsiveness, bounded waits, cleanup failures, retries, and idempotent repeated deletion.

## 1.7.2 - 2026-07-18

- Fixed feed processing so malformed non-numeric podcast enclosure lengths are treated as unknown instead of aborting the feed, with regression coverage for feed parsing and episode discovery.

## 1.7.1 - 2026-07-13

- Fixed generated RSS episode descriptions so source HTML is emitted in genuine CDATA sections and formatting and links render correctly across podcast clients instead of appearing as literal tags.

## 1.7.0 - 2026-06-18

- Added direct admin user creation from User Management.
- Changed new AI API token creation to require selecting the dashboard user the token acts as, while allowing linked tokens to browse the global library and limiting non-admin management actions to podcasts owned by that user.
- Added `scripts/link_api_token_user.py` to list and link existing API tokens to users without recreating the token.

## 1.6.1 - 2026-06-17

- Fixed AI API subscription creation so valid HTTP/HTTPS feed URLs are accepted instead of treated as validation errors.
- Made the AI API OpenAPI endpoint generate its schema directly from the v1 router so `/api/v1/openapi.json` includes all v1 paths.
- Expanded the AI API documentation into a user guide with workflows, endpoint behavior, request bodies, and response examples.

## 1.6.0 - 2026-06-17

- Added an optional token-protected AI REST API under `/api/v1`, with scoped admin-managed tokens, SQLite-backed rate limits, OpenAPI discovery, and System Settings controls.

## 1.5.2 - 2026-06-14

- Fixed the feed instruction page copy button so it reports success only after a confirmed clipboard write and shows manual copy guidance when browsers block automatic copying.

## 1.5.1 - 2026-06-14

- Added a Direct link subscription option beside the podcast-client app links while keeping the existing RSS feed link.
- Centralized podcast-client subscription link generation, fixed URL encoding for tokenized feeds, and moved lower-confidence app schemes to local best-effort instruction pages.

## 1.5.0 - 2026-06-14

- Fixed GitHub Actions verification by installing the async pytest plugin in CI and updating the workflow to Node 24-compatible GitHub action versions.
- Added admin podcast owner reassignment from the podcast detail page.

## 1.4.1 - 2026-06-13

- Added queue self-repair, FFmpeg operation timeouts, and running-job heartbeats so missing job rows, stuck audio operations, or orphaned running jobs do not hold the processing queue indefinitely.

## 1.4.0 - 2026-06-12

- Split Admin AI Configuration into separate Transcription, Voice and TTS, and Text Analysis pages under a new AI Settings sidebar group.
- Reorganized admin navigation into AI Settings, Podcast Preferences, System, and User Management groups.
- Moved Whitelist Mode from System Settings to Global Settings.
- Fixed manual downloads so they create durable queue jobs immediately instead of only changing episode status to `pending`.
- Changed Processing Queue cancellation to remove queued/running work without deleting or ignoring the episode.
- Added optional Gemini TTS for spoken title intros and audio summaries, with selectable Piper/Gemini providers, Gemini voices, and a dedicated Gemini TTS fallback cascade.
- Tidied root maintenance files by removing obsolete alternate-agent pointers and legacy shell helpers, moving the Unraid template under `Documentation/unraid/`, and fixing the GitHub Actions verification trigger.
- Added optional Apprise-backed admin notifications for access requests, new podcasts, completed episodes, and breaking processing errors.
- Split the overloaded admin access page into User Management, Access Requests, and Feed Access pages, and fixed admin user deletion from the UI.
- Compact admin user/login timestamps and adjust user tables/cards to avoid default horizontal scrolling.
- Changed access requests so users choose a password during the request; the app stores only the hash and admins no longer need to send temporary passwords.
- Added a global podcast library plus per-user "My Podcasts" membership without duplicating podcast rows.
- Added podcast ownership rules: admins can change any podcast settings, owners can change settings for podcasts they added, and only admins can delete the global podcast/files.
- Added admin-visible per-podcast library-user counts alongside existing total play counts.
- Split Piper TTS into an optional Docker dependency layer while keeping it enabled for default builds.
- Added an experimental no-TTS `linux/arm64` Docker build command for Apple Silicon / ARM testing.
- Refreshed the README to match the current app state and added current UI screenshots.
- Updated agent-maintenance guidance to list relevant root and `Documentation/` guidance files, require documentation updates alongside changes, and prefer commits after significant verified changes.
- Updated the default Gemini and OpenRouter Gemini fallback cascades to the current Flash/Lite order.
- Documented current Gemini free-tier RPM, TPM, and RPD limits in the README and environment documentation.
- Added a shared in-app toast and confirmation dialog system and replaced browser-native alert, confirm, and prompt popups in the web UI.
- Added a toggleable public read-only subscription page at `/subscribe`.
- Added backup-aware formal migration scaffolding and a durable SQLite jobs table.
- Added a migration dry-run helper for validating startup migrations against a copied `podcasts.db`.
- Added hashed feed tokens for protected podcast feed/audio links, while keeping Basic Auth and legacy `auth` links compatible.
- Added UI warnings that protected feed URLs containing generated tokens are bearer secrets until revoked.
- URL-encoded injected feed access parameters in RSS enclosure URLs and shared the logic between individual and unified feeds.
- Added atomic job claiming for the processor queue.
- Added an operation dashboard to the admin queue with active job, disk, memory/load, feed check, and retry state.
- Added live queue status polling through `/api/queue/status`.
- Added feed fetch and episode download guardrails for timeouts, size limits, content type, private URL policy, and free disk space.
- Added direct regression coverage for episode download size, content-type, and free-space response validation.
- Added private-network validation for final feed and episode download URLs after redirects when hardened URL policy is enabled.
- Added initial pytest coverage for migrations, job claiming, feed tokens, and URL guardrails.
- Added a setup checklist to System Settings with admin-account creation, base URL, subscribe page, and unified feed checks.
- Added migration backup tests for fresh and existing database initialization.
- Added synthetic legacy database migration coverage for preserving existing subscriptions, settings, episodes, and queued work.
- Added feed parsing and feed-size guardrail tests using deterministic sample RSS and fake HTTP streams.
- Fixed API subscription creation to handle feed descriptions returned by `FeedManager.parse_feed()`.
- Aligned the `Episode` model with retry, manual-download, and listen-count columns already present in the SQLite schema.
- Reduced the production Docker image by removing unused PyTorch packages and excluding local development artifacts.
- Added a production Docker Compose file that uses the published image and only mounts `/data`.
- Added a resource audit with runtime measurement commands and follow-up recommendations.
- Added optional resource tuning for Whisper CPU threads, FFmpeg threads, and unloading Whisper after the queue empties.
- Fixed fresh Docker installs so the public app URL is not auto-set to the container's internal IP address.
- Updated default OpenRouter models to cheaper Gemini flash/lite options.
- Hardened ad-detection response parsing so malformed model rows are skipped instead of crashing processing.
- Hardened audio segment keep-window calculation so overlapping, unsorted, out-of-range, or malformed remove segments are normalized before FFmpeg filtering.
- Extracted and tested processor ad-segment post-processing, including whitelist inversion and close-gap merging.
- Added a non-release Docker helper for publishing experimental tags without touching `latest`.
- Refactored direct Gemini access onto the OpenAI-compatible provider path and removed the `google-genai` runtime dependency.
- Updated Pydantic model/settings configuration to the v2 style, removing class-based config deprecation warnings.
- Escaped markdown summary rendering before applying the supported formatting subset.
- Escaped dynamic podcast search results and lazy-loaded episode card fields before inserting client-rendered HTML.
- Escaped dynamic AI model names, log lines, and prompt alerts before client-side HTML insertion.
- Added `TRUST_PROXY_HEADERS` so reverse-proxy deployments can explicitly opt into forwarded client IP headers.
- Added CIDR support to the IP allowlist while preserving exact IP entries.
- Stopped storing dashboard plaintext passwords in signed session cookies; feed links use generated tokens instead.
- New standalone feed Basic Auth passwords are now stored as bcrypt hashes while legacy plain-text settings remain accepted.
- Added admin visibility and revocation for active feed tokens.
- Added route-level admin dependencies to sensitive management endpoints as defense in depth beyond middleware.
- Added same-origin Origin/Referer checks for authenticated mutating management requests.
- Restricted System Settings `redirect_to` targets to local app paths.
- Added route-level auth dependencies to podcast-management API endpoints as defense in depth beyond middleware.
- Added startup validation and regression coverage so dashboard or feed authentication cannot run with the default session secret.
- Added System Settings warnings and form guards so dashboard or feed authentication cannot be enabled from the UI while `SESSION_SECRET_KEY` is still the default.
- Added a setup checklist warning when authentication is enabled on an HTTPS base URL while `COOKIE_SECURE=false`.
- Added behavior coverage for the intended split between authenticated management pages, public subscribe pages, and optional feed/audio authentication.
- Added middleware coverage for protected feed tokens, revoked-token rejection, and hashed standalone feed Basic Auth.
- Moved template filters into a standalone module, removed the router-local duplicate, and added regression coverage for escaped AI summary markdown rendering.
- Moved completed-episode RSS queries into `EpisodeRepository` and added regression coverage for ordering and subscription metadata.
- Extracted RSS feed base-URL selection into a shared helper while preserving LAN fallback behavior.
- Fixed npm audit findings for frontend build dependencies.
- Added `npm audit --audit-level=moderate` to the standard verification gate.
- Changed `npm test` to run the standard verification gate.
- Added GitHub Actions verification for pull requests and pushes to `master`/`audit-work`.
- Added explicit pytest-asyncio loop-scope configuration and refreshed Browserslist metadata used by the CSS build.
- Removed redundant whole-file `app.log` cleanup; log size is handled by rotating log handlers.
- Applied the same WAL and busy-timeout SQLite connection settings during startup migrations and runtime access.
- Added recovery for stale running processor jobs so interrupted workers do not permanently consume queue capacity.
- Downloaded episodes now write to a partial file and atomically move into place after completion to avoid treating interrupted downloads as valid audio.
- Added stale temporary processor artifact cleanup for old `.part` and `.tmp.mp3` files.
- Added containment checks before removing episode artifact directories.
- Resolved audio request paths inside podcast storage before file existence checks to prevent outside-path probing.
- Added per-category storage reporting for podcast files, models, feeds, database, backups, and logs.
- Hardened RSS CDATA description serialization, including descriptions containing `]]>`.
- Made feed authentication fail closed when enabled without credentials.
- Added an explicit System Settings error when standalone feed authentication is enabled without a username and password.
- Applied the IP allowlist before public feed/audio/subscribe route bypasses.
- Clarified feed protection as an optional podcast subscription security mode.
- Clarified destructive episode and subscription action labels.
- Removed a duplicate unreachable `raise` from audio prepending error handling.

## 1.3.1 - 2026-06-09

- Fixed `TemplateResponse` compatibility with modern FastAPI and Starlette releases.
- Fixed the Admin Queue context regression so the recently processed section renders again.
- Fixed the AI test connection response shape to match the admin UI expectations.
- Fixed dashboard AI configuration detection for the plural `gemini_api_keys` setting.
- Fixed `get_app_base_url()` usage in admin access routes.
- Added project maintenance docs for versioning, verification, naming, roadmap, decisions, and agent guidance.
- Added repeatable verification and Docker build/publish helper scripts.
- Updated release metadata to use `jdcb4/podcast-ad-remover` and MIT licensing.

## 1.3.0 - 2026-03-06

- Normalized the previous `1.3` release label to SemVer `1.3.0`.
- Added whitelist processing mode.
- Improved subprocess handling for non-ASCII paths and output.
