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
