# Agent Guide

This file is the first stop for coding agents and maintainers working on Podcast Ad Remover.

## Project Shape

Podcast Ad Remover is a Dockerized FastAPI application for downloading podcast episodes, removing ads with local transcription plus LLM analysis, and publishing replacement RSS feeds for podcast clients.

Core stack:
- Python 3.11, FastAPI, Jinja templates, SQLite.
- `faster-whisper`/Whisper for local transcription and FFmpeg for audio operations.
- Gemini, OpenAI, Anthropic, and OpenRouter integrations for ad detection and summaries.
- Tailwind CSS built through npm.
- Docker for normal deployment.

## Before Changing Code

1. Read `Documentation/PROJECT_INDEX.md`.
2. Read the relevant docs for the area being changed.
3. Inspect the current implementation before assuming a documented behavior is still true.
4. Keep user data compatibility as a default requirement. Existing installs use `/data/db/podcasts.db` and `/data` media folders.

## Additional Agent Guidance

Treat the following root-level guidance files as having the same working status as this file when they are relevant to the task:

- `README.md`: user-facing overview, quick start, and documentation index.
- `CONTRIBUTING.md`: contributor workflow and pull request expectations.
- `SECURITY.md`: supported security posture and reporting guidance.
- `AUDIT.md`: historical audit notes and priorities.

Treat the following `Documentation/` files as active guidance when they apply:

- `Documentation/PROJECT_INDEX.md`: first stop for project layout and verification commands.
- `Documentation/Architecture.md`: system architecture and major module responsibilities.
- `Documentation/Data_Flow.md`: processing pipeline behavior.
- `Documentation/Deployment.md` and `Documentation/Unraid_Deployment.md`: Docker, compose, and Unraid deployment expectations.
- `Documentation/Environment_Variables.md`: environment configuration and database-backed runtime settings.
- `Documentation/VERSIONING.md`: version bump and release-tag rules.
- `Documentation/VERIFICATION.md`: required local, Docker, and release checks.
- `Documentation/CHANGELOG.md`: unreleased and released change history.
- `Documentation/DECISIONS.md`: active decision log.
- `Documentation/Design_Decisions.md`: historical context only; prefer `Documentation/DECISIONS.md` for current choices.
- `Documentation/RESOURCE_AUDIT.md`: resource measurements and tuning guidance.
- `Documentation/ROADMAP.md`: current priorities and deferred work.
- `Documentation/NAMING.md`: naming conventions.
- `Documentation/AUDIT_STATUS.md`: current audit-branch state and deferred audit items.

Update relevant documentation as part of the same change. For user-visible behavior, consider `README.md`, `Documentation/CHANGELOG.md`, `Documentation/DECISIONS.md`, `Documentation/PROJECT_INDEX.md`, `Documentation/VERIFICATION.md`, and the area-specific documentation before finishing.

Commit after each significant coherent change, once verification appropriate to that change has passed. Keep commits scoped and descriptive. Do not commit secrets, local data, downloaded media, generated models, or throwaway test artifacts.

## Hard Rules

- Do not delete, rewrite, or reset existing `/data` content as part of a code change.
- Any database schema change needs a backward-compatible migration path and a rollback/backup note.
- Do not commit secrets, API keys, real session secrets, downloaded audio, transcripts, generated models, or local database files.
- Do not push a Docker release unless explicitly asked.
- Keep `package.json` and `package-lock.json` versions aligned.
- Update `Documentation/CHANGELOG.md` in the same change as a version bump.
- Docker releases use `jdcb4/podcast-ad-remover:<version>` and `jdcb4/podcast-ad-remover:latest`.

## Verification

Run this before saying a change is complete:

```bash
npm run verify
```

For release work, also run:

```bash
npm run verify:docker
```

Publishing a release image is explicit:

```bash
npm run docker:publish
```

`npm run verify` currently checks Python syntax with `compileall` and rebuilds Tailwind CSS. A proper Python test suite should be added before relying on this as the only release gate.

## Local Development Notes

- On Windows, use PowerShell commands unless the user asks otherwise.
- Use `rg` for code search.
- Prefer small, scoped changes that match the current app layout.
- Treat `Documentation/Design_Decisions.md` as historical context and `Documentation/DECISIONS.md` as the active lightweight decision log.

## Useful Paths

- `app/main.py`: FastAPI app startup and processor process launch.
- `app/core/processor.py`: episode download, transcription, ad detection, cutting, and cleanup.
- `app/core/config.py`: environment settings and `/data` paths.
- `app/infra/database.py`: SQLite schema initialization and migrations.
- `app/infra/repository.py`: database access patterns.
- `app/web/router.py`: web UI routes.
- `app/api/`: JSON/API and audio routes.
- `Documentation/`: project docs.
- `scripts/`: verification and release helper scripts.
