# Decisions

This is a lightweight decision log. Keep entries short, dated, and focused on choices that future maintainers may otherwise revisit.

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
`experimental/local-llm-transcript-chunking` for research, but are not part of the production
application and have no planned further development. Sanitized HTML, Markdown, and JSON results
remain on `master` so the decision and exact detections are inspectable.
