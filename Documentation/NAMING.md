# Naming

Use these conventions when adding new code, settings, docs, or release artifacts.

## Code

- Python modules and functions use `snake_case`.
- Classes use `PascalCase`.
- Constants use `UPPER_SNAKE_CASE`.
- Database columns use lower `snake_case`.
- Template files use lower `snake_case.html`.

## Episode Statuses

Keep status values lowercase strings. Current episode statuses include:

- `pending`: queued for processing.
- `unprocessed`: known but not queued.
- `processing`: actively being processed.
- `completed`: processed and available in feeds.
- `failed`: processing failed.
- `rate_limited`: waiting for LLM quota reset.
- `ignored`: hidden or soft-deleted.
- `pending_manual`: legacy/manual status still referenced by update logic.

Add a migration and UI handling before introducing a new status value.

## Job Statuses

Jobs use lowercase strings and describe worker state rather than the user-facing episode state:

- `queued`: ready to claim.
- `running`: claimed by a worker.
- `retry_scheduled`: failed but has a future retry time.
- `rate_limited`: waiting for provider quota reset.
- `completed`: finished successfully.
- `failed`: exhausted or stopped by an unrecoverable error.
- `cancelled`: cancelled because the episode was ignored or reset.

## Storage

- Persistent application data lives under `/data`.
- The SQLite database is `/data/db/podcasts.db`.
- Podcast episode artifacts live under `/data/podcasts/<podcast_slug>/<episode_slug>/`.
- Generated RSS feeds live under `/data/feeds/`.
- Downloaded models live under `/data/models/`.

## Docker

- Release image: `jdcb4/podcast-ad-remover`.
- Version tags must be full SemVer, for example `1.3.0`.
- Release publishes should also update `latest`.

## Documentation

- New maintenance docs should live in `Documentation/`.
- Prefer clear descriptive names such as `VERSIONING.md`, `VERIFICATION.md`, and `ROADMAP.md`.
- Keep `README.md` focused on users and `AGENTS.md` focused on maintainers and coding agents.
