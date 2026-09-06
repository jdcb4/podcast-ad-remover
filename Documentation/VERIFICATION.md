# Verification

Verification is the minimum set of repeatable checks to run before merging, tagging, or publishing a Docker image.

## Setup

Install Node dependencies before running frontend-related checks:

```bash
npm ci
```

Use Python 3.11 and Node 24 for parity with CI. Python dependencies are normally installed through Docker. For local Python work, install `requirements.txt` in your chosen virtual environment.
For local verification and tests, install the development requirements:

```bash
pip install -r requirements-dev.txt
```

## Standard Check

Run:

```bash
npm run verify
```

`npm test` runs the same standard verification gate.

This currently performs:

- Python syntax compilation for `app/` and `scripts/`.
- Python unit/integration tests with `pytest`, including real SQLite, two-process claims, rendered DOM and short FFmpeg fixtures.
- Tailwind CSS rebuild from `app/web/static/css/input.css` to `app/web/static/css/output.css`.
- Frontend dependency audit with `npm audit --audit-level=moderate`.
- Python dependency audit with `pip-audit` in the active environment.

If Tailwind reports stale Browserslist data, refresh the lockfile metadata with:

```bash
npx update-browserslist-db@latest
```

This is a maintenance update only; confirm the resulting `package-lock.json` changes are limited to Browserslist-related dependency metadata.

## Docker Check

Run this before a version increment or release publish:

```bash
npm run verify:docker
```

This runs the standard check and builds a local image tagged `podcast-ad-remover:verify`.

## Opt-In Live YouTube Smoke Test

The deterministic suite mocks YouTube and SponsorBlock. To verify the currently pinned extractor
against live public sources, supply a channel, explicit playlist, and short public video fixture:

```bash
python scripts/youtube_smoke_test.py --channel "https://www.youtube.com/@handle" --playlist "https://www.youtube.com/playlist?list=LIST_ID" --download-video "https://www.youtube.com/watch?v=VIDEO_ID"
```

The script resolves both sources and downloads best audio-only into a temporary directory. It uses
no cookies or captions and deletes the fixture when complete. This network-dependent test is kept
outside `npm run verify` so upstream availability cannot make the deterministic gate flaky.

## Custom OpenAI-Compatible Endpoint Checks

Automated coverage uses a temporary OpenAI-compatible HTTP server and verifies keyed/keyless model
listing and generation, URL validation, arbitrary model slugs, credential isolation, settings
persistence across separate AI pages, and prevention of saved credential rendering.

The rejected transcript-chunking experiment and its provider comparison are documented in
`Documentation/LOCAL_LLM_EVALUATION.md` and `Documentation/LOCAL_LLM_EVALUATION_REPORT.html`.
Those research artifacts are not a production release gate.

## Subscription Management Checks

Automated coverage verifies:

- new-subscription inheritance and the migration rule that only blank or NULL custom instructions inherit;
- effective global values, effective inherited-control display, restoration of stored podcast overrides,
  and cleanup behavior after global retention changes;
- backward-compatible API updates and all four inheritance flags;
- artwork URL validation, image limits, compositing/cache behavior, cleanup, and RSS output;
- in-place Library membership changes without dashboard navigation;
- atomic bulk updates, rejection of mixed unauthorised selections, and admin owner assignment;
- admin-only bulk deletion, its explicit destructive confirmation, and durable file cleanup.

For UI changes, manually confirm the compact table at a desktop width, select multiple manageable
podcasts, and verify that choosing an override enables only that group's controls.
As an administrator, cancel the Delete Selected warning and confirm no podcasts change. Then verify a
confirmed deletion removes each selected podcast and its local files. Confirm that non-admin users do
not see the bulk-delete control.

At a mobile width, also confirm the four compact statistics remain readable, feed actions retain
accessible touch targets, and approximately three queue rows are visible before internal scrolling.
Switch My Podcasts/Library after setting a search, filter, sort, and display mode; the toolbar,
scroll position, and layout should remain in place. Let queue auto-refresh run once and confirm only
the queue changes. Disable JavaScript or simulate a failed request to confirm the normal view links
and full Refresh action still work.

## Migration Dry Run

Before upgrading a valuable existing install, validate migrations against a copy of the database:

```bash
npm run db:migration-dry-run -- --db-path /data/db/podcasts.db
```

To keep the migrated copy for inspection:

```bash
npm run db:migration-dry-run -- --db-path /data/db/podcasts.db --keep-copy /tmp/podcast-ad-remover-migration-check
```

The helper takes an integrity-checked SQLite online backup, including committed WAL data, into a temporary data directory and runs startup migrations there. It does not modify the source. See `RECOVERY.md` for matching media/image restoration.

## Branch And Pull Request Check

GitHub Actions runs `npm run verify` on pull requests and pushes to `dev` and `main`. The `verify` job from GitHub Actions is a required check for both branches. Pull requests normally target `dev`; `main` is reserved for explicitly approved production promotions. The workflow sets `DATA_DIR` to a temporary Linux runner path so tests do not depend on `/data` being writable. See [Git workflow](GIT_WORKFLOW.md) for branch rules and maintenance.

## Dev Image Check And Publish

From a clean committed `dev` checkout, build the development image locally:

```bash
npm run docker:dev
```

To publish the development channel after verification:

```bash
npm run docker:dev:publish
```

Both commands produce:

- `jdcb4/podcast-ad-remover:dev`
- `jdcb4/podcast-ad-remover:dev-<git-sha>`

The rolling tag is convenient for the Dev environment; the SHA tag records the exact tested build. The helper refuses dirty checkouts and branches other than `dev`.

## Release Publish Check

Only run the production release path after the tested Dev revision has received explicit promotion approval and has been merged into `main`. The release helper requires a clean `main` checkout, reads the version from `package.json`, validates that it is `MAJOR.MINOR.PATCH`, runs verification, and builds two tags:

```bash
npm run docker:build
```

To push to Docker Hub:

```bash
npm run docker:publish
```

The pushed tags are:

- `jdcb4/podcast-ad-remover:<version>`
- `jdcb4/podcast-ad-remover:latest`

## Experimental Docker Tags

For audit or trial builds that must not update `latest` or a version tag:

```bash
npm run docker:experimental -- --push
```

By default this publishes:

- `jdcb4/podcast-ad-remover:experimental`
- `jdcb4/podcast-ad-remover:audit-work`
- `jdcb4/podcast-ad-remover:audit-work-<git-sha>`

The helper refuses `latest` and SemVer-looking tags.

## Experimental ARM64 Docker Tag

`linux/amd64` remains the primary release target. For Apple Silicon / ARM64 trial builds, use the no-TTS experimental helper:

```bash
npm run docker:experimental:arm64 -- --push
```

This publishes `jdcb4/podcast-ad-remover:experimental-arm64` when pushed. It passes `INSTALL_TTS=0`, so Piper TTS is not installed. Spoken summaries and title intros can still be tested with Gemini TTS when a Gemini API key is configured; the ARM64 experimental target is intended to test the core podcast download, transcription, ad detection, cutting, feed, and web UI path.

## Current Gaps

- Python coverage should continue expanding around full processor lifecycles and service boundaries.
- Migration tests cover additive schema and data transforms, but a copied realistic `podcasts.db`
  dry run remains a release-time check rather than a routine automated test.

The Python suite also runs `tests/episode_dom.cjs` with Node against server-rendered HTML.
Run `npm ci` before pytest; Node 24 is the CI baseline. FFmpeg enables short MP3/AAC/Opus
integration tests. Offline tests stub external downloads/models, not database repositories.

## Reproducible dependencies and container smoke

`constraints.txt` pins the Python 3.11 runtime, verification and optional Piper dependency set,
including platform markers. Requirements install through it. `requirements-build.txt` also pins
pip/setuptools so CI and the image do not inherit outdated bootstrap tools. Refresh deliberately in a clean
Python 3.11 environment, inspect the diff, then run the full gate and image smoke:

```bash
uv pip compile --python-version 3.11 --universal requirements-dev.txt requirements-tts.txt --output-file constraints.txt
pip install -r requirements-dev.txt
npm ci
npm run verify:docker
docker run --rm --network none podcast-ad-remover:verify python scripts/container_smoke.py
docker run --rm --network none podcast-ad-remover:verify python -m pip check
```

`uv` is an optional lock-maintenance tool, not an application dependency. Python/Deno image
stages are digest-pinned; system packages are resolved during builds, so retain the tested
immutable image for exact rollback. Do not silently ignore dependency advisories. Network access
is needed for the audits; an unavailable registry is a failed/incomplete gate, not a clean audit.

The smoke script creates its own temporary data, checks native imports, migrations/backup,
AAC conversion, publication recovery, startup, health, dashboard and assets. Use an isolated
container with no live data mount. It does not download models or call paid providers. CI installs
FFmpeg so codec tests run instead of skipping. `jsdom` is a development-only dependency that runs
the shipped JavaScript against actual Jinja output; native dialog/focus/layout still need browser QA.

Before deployment, rehearse `migration_dry_run.py` on an online snapshot of the target database,
record the immutable image and media backup, and check the disabled-processing clone before
re-enabling work. A build alone does not validate production data or paid provider behavior.
