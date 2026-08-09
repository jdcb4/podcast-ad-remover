# Verification

Verification is the minimum set of repeatable checks to run before merging, tagging, or publishing a Docker image.

## Setup

Install Node dependencies before running frontend-related checks:

```bash
npm ci
```

Python dependencies are normally installed through Docker. For local Python work, install `requirements.txt` in your chosen virtual environment.
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
- Python unit tests with `pytest`.
- Tailwind CSS rebuild from `app/web/static/css/input.css` to `app/web/static/css/output.css`.
- Frontend dependency audit with `npm audit --audit-level=moderate`.

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
- atomic bulk updates, rejection of mixed unauthorised selections, and admin owner assignment.

For UI changes, manually confirm the compact table at a desktop width, select multiple manageable
podcasts, and verify that choosing an override enables only that group's controls.

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

The helper copies the source database into a temporary data directory, runs the normal startup migration path on the copy, and does not modify the source database.

## Branch And Pull Request Check

GitHub Actions runs `npm run verify` on pull requests and pushes to `dev` and `master`. Pull requests normally target `dev`; `master` is reserved for explicitly approved production promotions. The workflow sets `DATA_DIR` to a temporary Linux runner path so tests do not depend on `/data` being writable.

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

Only run the production release path after the tested Dev revision has received explicit promotion approval and has been merged into `master`. The release helper requires a clean `master` checkout, reads the version from `package.json`, validates that it is `MAJOR.MINOR.PATCH`, runs verification, and builds two tags:

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
