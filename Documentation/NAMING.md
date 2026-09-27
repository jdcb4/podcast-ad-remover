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
- `ignored`: skipped/non-episode or explicitly ignored; the Ignore action removes generated artifacts after worker acknowledgement. It is not just a hide toggle.
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
- New artifacts live under `/data/podcasts/<podcast_slug>/episode-<database-id>/attempt-<token>/`; legacy GUID-derived directories remain readable. Do not rename existing data to match the newer layout.
- Generated RSS feeds live under `/data/feeds/`.
- Downloaded models live under `/data/models/`.

## Docker

- Release image: `jdcb4/podcast-ad-remover`.
- Rolling development tag: `dev`.
- Commit-specific development tag: `dev-<short-git-sha>`.
- Version tags must be full SemVer, for example `1.3.0`.
- Release publishes should also update `latest`.
- Development publishes must never update `latest` or a SemVer tag.

## Documentation

- New maintenance docs should live in `Documentation/`.
- Prefer clear descriptive names such as `VERSIONING.md`, `VERIFICATION.md`, and `ROADMAP.md`.
- Keep `README.md` focused on users and `AGENTS.md` focused on maintainers and coding agents.

V2 UI terms are **My Podcasts**, **Library**, **Tasks**, **Settings**, **Users & access**, **Complete Timeline**, **Voice**, and **Insert tone at content cuts**. Historical database names containing `cascade`, `warning_tone` or Legacy fields may remain for compatibility; do not infer that those old features are still supported.
