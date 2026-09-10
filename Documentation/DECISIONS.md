# Decisions

This is a lightweight decision log. Keep entries short, dated, and focused on choices that future maintainers may otherwise revisit.

## 2026-09-10: Retain API v1 for the UTC timestamp update

The maintainer accepted the timestamp parser compatibility risk given the limited known API
usage. Keep `/api/v1` and the new UTC timestamp representation; do not introduce an API v2
or duplicate legacy timestamp fields for this change. Target the next minor application
release, 1.14.0, as an explicit exception to the usual major-version rule for incompatible
API changes. Preserve the warning in [API.md](API.md#timestamp-formats) and the release notes
so clients using strict parsers can update. Historical timezone provenance remains unchanged.
This decision accepts the compatibility tradeoff; production promotion remains a separate step.

## 2026-09-06: Keep dev and main as the only long-lived branches

Joe requested `dev` as the working/default branch and `main` as production. Rename `master` to
`main` without rewriting history. Apply the existing deletion/force-push protection to both real
branches and require the GitHub Actions `verify` check. Keep the existing administrator recovery
bypass; do not use it for routine changes. Feature branches and worktrees exist only while work is
active and unmerged, and merged pull-request branches are deleted automatically.

Preserve the concluded local-LLM experiment at tag `archive/local-llm-transcript-chunking` before
removing its inactive branch. Open contributor pull requests remain open and target `dev`.
Repository-maintenance changes authorized for this rename do not publish or deploy a new app
release. See [Git workflow](GIT_WORKFLOW.md) for the operating procedure and cleanup record.

## 2026-08-10: Integrate on dev and explicitly promote production releases

`dev` is the primary integration branch and `main` (named `master` until 2026-09-06) is the production branch. Development images share the production Docker repository but use `dev` as a rolling tag and `dev-<git-sha>` as the traceable immutable tag. They never update SemVer tags or `latest`. After a Dev image has been tested, merging to `main` and publishing production tags still require Joe's explicit instruction.

## 2026-05-19: Keep SQLite and `/data` as the migration anchor

Existing users already have SQLite databases and downloaded podcast artifacts under `/data`. Improvements should preserve that layout unless there is a clear migration path, backup guidance, and a versioned release note.

## 2026-05-19: Publish Docker releases to Docker Hub

The release image is `jdcb4/podcast-ad-remover`. Every release should publish both a SemVer tag and `latest` so users can either pin a version or follow the current release.

## 2026-05-19: Use lightweight Python release scripts

The project is primarily Python, so verification and Docker publish helpers live in `scripts/` and are exposed through npm scripts. This keeps the commands easy to run on Windows and Linux while avoiding a larger build system.

## 2026-06-11: Keep public subscription access optional and unauthenticated by default

The main workflow is subscribing from podcast clients, many of which handle authentication inconsistently. Dashboard management can require login, while the public subscribe page and feeds remain unauthenticated unless feed authentication is explicitly enabled.

## 2026-06-11: Keep SQLite as the default database for this audit pass

The target deployment is a personal Docker install with existing SQLite data under `/data`. The audit branch improves SQLite safety with WAL, a longer busy timeout, migration backups, and durable job rows rather than introducing a database migration to a larger service.

## 2026-06-11: Improve the current FastAPI/Jinja app incrementally before considering a rebuild

The current app has oversized modules and templates, but the core architecture still matches the deployment model. The audit branch should reduce risk through tests, auth hardening, resource controls, and focused refactors before considering a full frontend or backend rebuild.

## 2026-06-11: Prefer Gemini Flash/Lite fallback models for default ad detection

Gemini remains the recommended provider for most homelab installs because the free tier is usually enough for this app's transcript-analysis workload. The default cascade now follows the current Flash/Lite fallback order: Gemini 3.5 Flash, Gemini 3 Flash, Gemini 3.1 Flash Lite, Gemini 2.5 Flash, then Gemini 2.5 Flash Lite. OpenRouter uses the same order with `google/` model IDs.

## 2026-06-12: Keep amd64 primary and make ARM64 experimental without Piper TTS

The default Docker image remains `linux/amd64` with Piper TTS installed. Apple Silicon / ARM64 is useful to test, but Piper's phonemizer dependency is not currently simple to install from Linux arm64 wheels. The experimental ARM64 build uses `INSTALL_TTS=0` so the core podcast workflow can be tested without local Piper; Gemini TTS can still provide spoken title intros and summaries when configured.

## 2026-06-12: Separate global podcast records from user libraries

Podcast rows remain global so the app only downloads, processes, stores, and publishes one copy of each feed. User-specific interest is tracked through `user_subscriptions`. The user who first adds a podcast becomes its settings owner, but only admins can delete the global podcast and files. If an owner removes the podcast from their own library, the global podcast becomes unowned for an admin to review.

## 2026-06-12: Let access-request users choose their password

Access requests should not make admins copy generated passwords back to users. New requests collect a password, store only the bcrypt hash, and copy that hash into `users` if approved. Admins approve identity/access, not credentials.

## 2026-06-12: Use Apprise for optional admin notifications

Notifications should stay optional and provider-agnostic. Embedding the Apprise Python library keeps the app to one container while supporting ntfy, Gotify, Pushover, Discord, email, webhooks, and other targets through configuration rather than provider-specific code.

## 2026-06-12: Keep Piper default while adding optional Gemini TTS

Piper remains the default because it is local, offline, and does not consume API quota. Gemini TTS is available as an optional provider for installs that prefer hosted speech generation or experimental images without Piper. It reuses Gemini API keys, has its own TTS model cascade, and keeps text-analysis model settings separate from voice settings in the admin UI.

## 2026-06-17: Keep the AI API opt-in and token-scoped

The AI-facing integration surface is a REST API under `/api/v1`, disabled by default and protected by admin-managed bearer tokens. API tokens are separate from feed tokens, use explicit scopes, and have SQLite-backed rate limits so the feature fits the existing single-container SQLite deployment model.

## 2026-07-22: Coordinate destructive subscription cleanup through SQLite

The FastAPI app and episode processor run in separate processes, so in-memory locks cannot safely coordinate subscription deletion. Deletion uses durable subscription state and existing job locks: the database transaction prevents new claims and requests cancellation, running workers release their locks only after reaching a safe checkpoint, and one retryable cleanup claimant removes files and regenerates feeds outside the web event loop.

## 2026-07-24: Keep cloud LLMs primary and make custom compatibility explicit

Gemini and the existing cloud cascades remain the default text-analysis path. A custom endpoint is
an opt-in, separately credentialed OpenAI-compatible provider so an OpenAI cloud key cannot be sent
to an operator-supplied URL. Arbitrary model slugs and keyed or keyless endpoints are supported.

## 2026-07-24: Do not ship experimental transcript chunking

The 4B-to-72B evaluation completed only 20 of 30 runs and passed 6 of 30 interval-quality gates.
Every local-class candidate missed the short promotion, and structured-output failures remained
common. The chunking implementation and harness are preserved on
tag `archive/local-llm-transcript-chunking` (formerly the experimental branch) for research, but are not part of the production
application and have no planned further development. Sanitized HTML, Markdown, and JSON results
remain on `main` so the decision and exact detections are inspectable.

## 2026-07-25: Use explicit group inheritance without discarding podcast overrides

Per-podcast settings inherit through four independent flags: content removal, retention, default
features, and custom instructions. Effective values are resolved when subscriptions are read, while
the stored podcast-specific values remain unchanged so turning inheritance off restores them. New
podcasts inherit all four groups. The additive migration keeps every existing content-removal,
retention, and feature group explicit; only blank or NULL custom instructions migrate to inheritance,
matching their previous behavior. Boolean values are therefore never overloaded with NULL to mean
inheritance.

## 2026-07-28: Use progressively enhanced dashboard islands instead of a SPA

The server-rendered FastAPI/Jinja architecture remains the default. Live queue updates and
My Podcasts/Library switching update bounded DOM regions while ordinary links and a manual full
refresh remain functional fallbacks. This preserves scroll position and in-progress dashboard state
without adding a client-side router or duplicating the full UI as a JavaScript application. Feed
subscription choices expose the two reliable workflows—direct RSS and copy into a preferred
podcast app—while legacy app-specific instruction routes remain compatible.

## 2026-08-10: Treat YouTube as a source adapter and gate SponsorBlock at deployment

Public YouTube channels and explicit playlists use pinned yt-dlp plus bundled EJS scripts and Deno,
not the official YouTube Data API or another long-running sidecar. Discovery remains bounded and
provider-specific while downstream transcription, detection, cutting, retention, and RSS generation
stay shared. SponsorBlock timestamps complement rather than pre-cut the LLM workflow, fail open, and
are protected by an environment-only flag that defaults off because the API/database licence is
CC BY-NC-SA 4.0.

## 2026-08-27: Store and compare every timestamp as naive UTC

New database timestamps use naive UTC, matching SQLite's `CURRENT_TIMESTAMP`; generate them with
`app/core/time_utils.now_utc()`. Elapsed-time measurements and human-readable backup filenames
may use other clocks because they are not interpreted as stored UTC instants.

Earlier releases wrote `users.last_login` and `episodes.processed_at` using the server's local
clock without its timezone. Migration `20260910_0015_timestamp_provenance` preserves those values
and adds `last_login_is_utc` / `processed_at_is_utc`, defaulting to false. Successful login and
episode-completion writes set the matching flag atomically with a new UTC value. Failed attempts
and requeueing preserve both the previous timestamp and its flag. No timezone is inferred from
the current server configuration: it may differ from the one used for the original write.

Historical values without a known offset render with a "timezone unknown" label and stay raw in
JSON alongside their flag. Values containing an explicit offset can be localized safely. This
also applies to existing PR #21 installs, whose older naive values cannot be reliably distinguished
from pre-PR server-local values. Existing timestamps are never shifted automatically.

Rendering follows the same split: `local_time` (in `app/web/template_filters.py`) renders a whole
`<time datetime="...Z">` element and is for element text only; `utc_isoformat` renders a bare
`...Z` string and is for HTML attribute contexts, because a `<time>` element's own quotes would
corrupt the enclosing tag if `local_time` were used inside one. A repo-wide test
(`tests/test_template_filters.py::test_no_template_pipes_local_time_into_an_html_attribute`) scans
every template and fails if `local_time` is ever piped into an attribute.

Pass the provenance flag as `local_time(value, format, known_utc)` for the two historical fields.
The shared formatter also handles dashboard replacement events so My Podcasts/Library switches
and browser history retain local dates.

## 2026-09-05: retain publications and fence attempts

Episode identity remains the SQLite ID and source GUID. New artifacts live in
`podcasts/<subscription>/episode-<id>/attempt-<random>/`; each attempt owns only that
staging directory. A validated result commits its audio/report pointers and output
duration together. Reprocessing changes the feed GUID only after successful replacement.
Existing files and old playback URLs remain available until explicit deletion or retention.
Legacy GUID-derived paths are read-compatible; ambiguous legacy deletion paths are preserved.

SQLite claims include a unique token. Worker writes check that token, cancellation,
episode status and subscription eligibility within the same write transaction. Cancel
requests retain running leases until acknowledgement; the capacity check and claim are
one transaction. Enabled installations process only in the dedicated child; disabled
installations share one manual processor. This avoids duplicate model owners in the web process.

Feed generation is serialized across processes and uses temporary-file replacement.
`publication_pending` survives feed errors without deleting audio or repeating AI work.
The existing model-tooling dependency `filelock` is now explicit for portable publication
locking; no queue service, ORM or external coordination service was added.

## 2026-09-05: test the rendered episode page

Episode behavior lives in `static/js/episodes.js`; its only server input is an escaped
JSON configuration block. SQL applies search and status filters before pagination.
Each request has a sequence and abort controller so old results cannot replace newer ones.
Native dialog behavior provides keyboard modality without an additional UI framework.

`jsdom` is a development-only dependency for running shipped JavaScript against rendered
HTML. It reproduces the missing-button failure that source-text assertions missed and
checks stale searches, injection escaping and error messages without downloading browsers.
Browser smoke testing remains necessary for layout, native focus and media playback.

## 2026-09-06: Reuse one server renderer for episode cards

Browser verification found drift between initial and incrementally loaded cards. Both now render
`_episode_cards.html`; the existing paginated JSON retains its data fields and also supplies HTML.
The client owns request ordering, selection and dialogs. This removes duplicated status/summary/
permission markup and a conflicting legacy list stylesheet without adding a frontend framework.
Native confirmation dialogs make the page behind them inert and restore focus on dismissal.

## 2026-09-06: Pin reviewed dependencies and keep integration tests lightweight

A universal Python 3.11 constraint set covers runtime, dev and optional Piper installations. Base
Python/Deno stages are digest-pinned; the tested immutable image remains the deployment/rollback
unit because OS package repositories can change. `pip-audit` joins the existing verification gate.
`jsdom` is a dev-only dependency justified by regressions that required real rendered HTML plus the
shipped JavaScript to reproduce. `filelock`, already transitive in model tooling, is explicit because
atomic feed publication must serialize writers across processes. No database, broker or UI framework
is added. Unused wrappers/path properties are removed after call-site and compatibility checks;
legacy on-disk artifacts continue to resolve through constrained compatibility reads.
