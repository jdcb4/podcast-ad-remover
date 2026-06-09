# Changelog

## Unreleased

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
