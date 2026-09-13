# Changelog

## Unreleased

- Simplify removal-tone settings to three enable switches using fixed Wooden notes. Remove in-app sound selection and previews; ignore legacy style values without deleting stored settings.

- Select Wooden notes as the default beginning, ending and middle-removal sounds; migrate the former Soft chime defaults without changing enable switches or other selected styles.


- Update Gemini, OpenAI, Anthropic and OpenRouter default cascades to the requested model lists; share runtime/UI defaults and migrate recognized old defaults once without replacing custom selections. Gemini speech defaults stay unchanged.
- Add opt-in global removal tones with separate beginning, middle and ending switches and sound selections. Ship six previewable original sound sets as tiny WAV files; insert cues only at removed content, without runtime synthesis or TTS dependencies.


- Show the fixed global Complete Timeline prompt template as read only in AI Prompt Rules, with category definitions arranged vertically at full width.
- Add opt-in Complete Timeline processing with editable categories, contextual editorial/non-editorial gaps, a combined episode summary, and a configurable short-island rule (10 seconds; 0 disables). Preserve substantive openings and conclusions. Reports separate model classifications from extra cuts; schema support is detected through requests and provider fallback stays within the configured provider. Tidy AI Prompt Rules with separate Complete Timeline/Legacy controls and an effective-prompt preview. Add a backward-compatible migration that keeps existing podcasts/jobs on Legacy and preserves custom prompts, whitelist settings and published audio. See `Documentation/COMPLETE_TIMELINE.md` for opt-in, cache reuse, backup and rollback.
- Accept valid podcast artwork served with the generic `application/octet-stream` content type. Keep download limits and image decoding checks, and preserve cached artwork when the response is not a valid image.
- Reduce badged podcast artwork to JPEG at quality 82, with a maximum edge of 1400px, contributed by [Paul McManus (@pmacca) in PR #22](https://github.com/jdcb4/podcast-ad-remover/pull/22). Keep cached PNG feed URLs working and retain the old image until the replacement path commits successfully. Save global subscription settings once after upgrading to regenerate existing artwork and feeds.
- Add administrator-managed unified-feed preferences for its name, description, podcast-name episode-title prefix, and external artwork URL, contributed by [Paul McManus (@pmacca) in PR #20](https://github.com/jdcb4/podcast-ad-remover/pull/20). Preserve the existing address and defaults, validate RSS-safe metadata, include authentication in the settings-page feed address, and explain unavailable HTTP artwork previews.
- Make `dev` the GitHub default and rename production `master` to `main`; align CI, release guards and Unraid template URLs. Require verification on both long-lived branches, clean up merged branches, and retain the abandoned local-LLM experiment under an archive tag.
- Render dates with known timezones in the viewer's browser-local timezone, contributed by [Paul McManus (@pmacca) in PR #21](https://github.com/jdcb4/podcast-ad-remover/pull/21). Preserve localization after dashboard view switches, browser history navigation, and episode pagination.
- Normalize new timestamp writes to naive UTC and fix feed-date ingest during daylight saving. Preserve historical login and processing times exactly; an additive migration tracks new UTC writes while older ambiguous values display "timezone unknown". See `Documentation/RECOVERY.md` for backup and rollback.
- Fixed a subscription with no episodes at all (e.g. just created, before its first feed poll) wrongly appearing under the dashboard's "Recently updated" filter.
- **API wire-format change:** queue and system-status responses use Z-suffixed ISO timestamps for known UTC instants. Historical `processed_at` values remain raw with `processed_at_is_utc=0`; clients must not guess their timezone. See the endpoint-specific contract in `Documentation/API.md` before updating parsers that expect bare database strings.
- The admin log viewer is unchanged: its timestamps come from the logging formatter, not the database.

## 1.13.0 - 2026-09-06

- Keep published-episode counts, previews and listen totals stable while replacement processing is queued, running, cancelled or failed.

- Honor explicit RSS API initial-download counts, including zero, and reject negative counts before creating a subscription.

- Pin the Python dependency set and Python/Deno image stages, audit Python packages in the standard gate, and ship offline container/backup/publication recovery commands.
- Bound standalone speech fallback requests and stop speech retries on authentication failures.
- Replace obsolete active audit/setup/security claims with current behavior; preserve historical audit evidence and remove unused wrappers without deleting legacy data.

- Share episode card rendering between initial pages and pagination; fix keyboard description expansion, mobile list layout and native confirmation dialogs. Cancellation consistently retains published audio; Ignore remains the file-removal action.

- Bound provider calls across automatic retries, disable hidden SDK retries, honor Retry-After, stop retrying authentication/billing failures, record provider usage, reserve scratch capacity, and reuse verified stage outputs after failures.

- Expose real worker heartbeat/scheduler status, supervise child failures, add a Docker readiness check, report container memory and cache storage scans. Honor the initial Whisper environment setting and validate the selected AI provider including keyless custom endpoints.

- Fix episode deletion, make settings and action menus keyboard accessible, label episode selection, paginate filters/search on the server, preserve newer search results, report action failures and support direct RSS URLs. Extract episode JavaScript and correct stored-audio/count labels.

- Use stable database IDs for new episode storage, fence every worker write by its claim, retain cancellation leases until acknowledgement, and enforce the global claim limit in one transaction.
- Stage replacement audio independently, preserve previous published files and GUIDs during failed reprocessing, validate source fingerprints before transcript reuse, and publish measured output duration.
- Atomically replace serialized RSS snapshots and retain completed audio with a durable feed-publication retry flag when feed writing fails. Existing artifact paths remain readable.

- Validate every feed, artwork and RSS audio redirect before connecting; pin public IPs in restricted mode and recheck disk space during streaming.

- Reject invalid AI segmentation responses instead of publishing them as no-ads results, and encode non-MP3 sources correctly when no cuts are needed.

- Enforce owner/admin episode permissions, fix library membership responses, return deliberate 403 responses, and attribute audio listens by full path.

- Reject known session-secret placeholders and honor Compose environment values.

- Create integrity-checked SQLite snapshots before migrations and for dry runs, including committed WAL data; add a backup CLI and recovery runbook.
- Escape generated report text and sandbox legacy reports.

- Reject episode cleanup paths that resolve to storage roots, subscription roots or filesystem aliases.

## 1.12.0 - 2026-08-11

- Isolated deterministic AI-provider tests from workstation credential fallbacks so release verification cannot change retry behavior or expose configured keys in assertion output.
- Added administrator-only bulk subscription deletion with an explicit permanent-file-removal warning and the existing durable, retryable cleanup workflow; confirmed actions explicitly target the deletion endpoint instead of falling back to a no-op bulk-settings update, and content-derived static-asset URLs prevent Cloudflare from serving obsolete dashboard JavaScript after deployment.
- Fixed the My Podcasts/Library **Select all** checkbox after in-place view changes by binding it through the stable document and resolving the current results control dynamically.
- Added public YouTube channel and explicit-playlist subscriptions with bounded discovery, pinned audio-only yt-dlp/Deno extraction, native-format pipeline input, canonical deduplication, and discovery-based YouTube retention.
- Added read-only SponsorBlock timestamp merging with report provenance behind `SPONSORBLOCK_ENABLED=false`, including fail-open behavior and licence documentation.
- Made `dev` the primary integration branch, reserved `master` for explicitly approved production promotions, and added traceable Docker Dev builds tagged `dev` and `dev-<git-sha>`.
- Updated the frontend dependency lock to resolve the high-severity `nanoid` audit advisory.
- Added `PROCESSOR_ENABLED=false` for isolated Dev clones that need normal startup and migration checks without background polling or episode processing.

## 1.11.0 - 2026-07-28

- Made the mobile dashboard more compact, reduced feed subscription choices to Direct link and a generic “Use your favourite app” workflow, and deemphasized the unified feed.
- Limited the visible processing queue to roughly three items and replaced scheduled whole-page reloads with a queue-only JSON refresh plus an explicit full-refresh fallback.
- Made My Podcasts and Library switch in place while preserving dashboard search, filters, sorting, display mode, and scroll position; ordinary links remain as progressive-enhancement fallbacks.
- Fixed inherited podcast controls to display their current effective global values while preserving stored overrides for restoration, with regression coverage proving repository reads, feed discovery, and cleanup use updated global retention.

## 1.10.0 - 2026-07-25

- Added four explicit per-podcast inheritance groups for content removal, retention, default features, and custom instructions. New podcasts inherit all groups; migration preserves existing explicit values except blank or NULL custom instructions, which now inherit.
- Added an optional inherited or per-podcast ad-free artwork badge with validated source-image fetching, cached generated artwork, and RSS/dashboard integration.
- Added a compact subscription table with settings-source indicators and permission-checked, atomic bulk updates for processing settings, retention, instructions, and admin ownership.
- Preserved dashboard view, filters, sort order, and scroll position when adding or removing podcasts from a user's Library.

## 1.9.0 - 2026-07-24

- Added an opt-in custom OpenAI-compatible text-analysis provider with arbitrary API base URL/model slugs, keyed or keyless endpoint support, strict URL validation, separate credentials, and Docker networking guidance.
- Prevented saved AI credentials from being rendered back into admin forms and made AI settings updates section-specific so saving one page cannot reset another.
- Published the archived local-model transcript-chunking research report and exact interval comparisons; the unreliable chunking implementation remains experimental and is not included in the production application.
- Excluded local benchmark workspaces and agent attachments from Docker build contexts so development-only transcripts and artifacts cannot enter release images.

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
