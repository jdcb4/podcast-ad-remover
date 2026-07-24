# Project Index

Podcast Ad Remover downloads podcast episodes, processes them to remove ads or promotional segments, and republishes RSS feeds that can be subscribed to from a podcast client.

## Current Stack

- Python 3.11, FastAPI, Jinja templates, SQLite.
- FFmpeg for audio cutting and concatenation.
- Whisper/faster-whisper for local transcription.
- Gemini, OpenAI, Anthropic, and OpenRouter for LLM-based segment detection.
- Piper or Gemini TTS for optional spoken title intros and audio summaries.
- Apprise for optional admin notifications.
- Tailwind CSS compiled with npm.
- Docker for normal deployment.

## Important Folders

- `app/main.py`: application entry point and processor process startup.
- `app/core/`: podcast, audio, AI, RSS, search, and processing logic.
- `app/infra/`: SQLite initialization and repository access.
- `app/web/`: web routes, templates, authentication helpers, and static files.
- `app/api/`: dashboard API, audio endpoints, and the optional AI-facing `/api/v1` REST API.
- `Documentation/`: architecture, deployment, release, and maintenance docs.
- `Documentation/unraid/podcast-ad-remover.xml`: Unraid Docker template.
- `scripts/`: verification, migration dry-run, API-token user-linking, and Docker release helpers.
- `Dockerfile` and `docker-compose.yml`: container build and local compose configuration.

## Common Commands

```bash
npm ci
npm run build:css
npm run verify
npm run verify:docker
npm run docker:build
npm run docker:publish
npm run docker:experimental:arm64
docker compose up -d --build
```

`npm run verify` is the normal pre-change completion check. `npm run verify:docker` adds a local Docker image build and should be used before release tagging or publishing.

## Documentation Map

- `Documentation/Architecture.md`: current application structure and data layout.
- `Documentation/API.md`: optional AI-facing REST API, auth, scopes, and rate limits.
- `Documentation/Deployment.md`: Docker and Docker Compose deployment.
- `Documentation/Environment_Variables.md`: environment configuration.
- `Documentation/VERSIONING.md`: version bump and Docker tag rules.
- `Documentation/VERIFICATION.md`: checks to run before merging or releasing.
- `Documentation/CHANGELOG.md`: release notes.
- `Documentation/AUDIT_STATUS.md`: current audit branch findings, implementation status, and deferred work.
- `Documentation/DECISIONS.md`: lightweight decision log.
- `Documentation/ROADMAP.md`: improvement candidates and future direction.
- `Documentation/RESOURCE_AUDIT.md`: image size, runtime resource findings, and live-container measurement commands.
- `Documentation/LOCAL_LLM_EVALUATION.md`: archived local-model and transcript-chunking research findings.
- `Documentation/LOCAL_LLM_EVALUATION_REPORT.html`: self-contained pass/fail and exact detection comparison.
- `Documentation/LOCAL_LLM_EVALUATION_RESULTS.json`: sanitized machine-readable research results.
- `Documentation/NAMING.md`: naming conventions for code, statuses, docs, and Docker artifacts.
- `AGENTS.md`: maintenance rules for coding agents.
