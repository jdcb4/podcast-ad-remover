# Changelog

## Unreleased

- Added a toggleable public read-only subscription page at `/subscribe`.
- Added backup-aware formal migration scaffolding and a durable SQLite jobs table.
- Added hashed feed tokens for protected podcast feed/audio links, while keeping Basic Auth and legacy `auth` links compatible.
- Added atomic job claiming for the processor queue.
- Added an operation dashboard to the admin queue with active job, disk, memory/load, feed check, and retry state.
- Added live queue status polling through `/api/queue/status`.
- Added feed fetch and episode download guardrails for timeouts, size limits, content type, private URL policy, and free disk space.
- Added initial pytest coverage for migrations, job claiming, feed tokens, and URL guardrails.
- Added a setup checklist to System Settings with admin-account creation, base URL, subscribe page, and unified feed checks.
- Added migration backup tests for fresh and existing database initialization.
- Reduced the production Docker image by removing unused PyTorch packages and excluding local development artifacts.
- Added a resource audit with runtime measurement commands and follow-up recommendations.
- Added optional resource tuning for Whisper CPU threads, FFmpeg threads, and unloading Whisper after the queue empties.
- Fixed fresh Docker installs so the public app URL is not auto-set to the container's internal IP address.
- Updated default OpenRouter models to cheaper Gemini flash/lite options.
- Hardened ad-detection response parsing so malformed model rows are skipped instead of crashing processing.
- Added a non-release Docker helper for publishing experimental tags without touching `latest`.
- Escaped markdown summary rendering before applying the supported formatting subset.
- Made feed authentication fail closed when enabled without credentials.
- Applied the IP allowlist before public feed/audio/subscribe route bypasses.
- Clarified feed protection as an optional podcast subscription security mode.
- Clarified destructive episode and subscription action labels.

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
