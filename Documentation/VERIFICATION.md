# Verification

Verification is the minimum set of repeatable checks to run before merging, tagging, or publishing a Docker image.

## Setup

Install Node dependencies before running frontend-related checks:

```bash
npm ci
```

Python dependencies are normally installed through Docker. For local Python work, install `requirements.txt` in your chosen virtual environment.

## Standard Check

Run:

```bash
npm run verify
```

This currently performs:

- Python syntax compilation for `app/` and `scripts/`.
- Python unit tests with `pytest`.
- Tailwind CSS rebuild from `app/web/static/css/input.css` to `app/web/static/css/output.css`.

## Docker Check

Run this before a version increment or release publish:

```bash
npm run verify:docker
```

This runs the standard check and builds a local image tagged `podcast-ad-remover:verify`.

## Release Publish Check

The release helper reads the version from `package.json`, validates that it is `MAJOR.MINOR.PATCH`, runs verification, and builds two tags:

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

## Current Gaps

- Python test coverage is intentionally small and should be expanded before broad processor refactors.
- `npm audit --audit-level=moderate` is useful, but it is not part of the default gate until the current dependency advisories are resolved.
- There is no automated migration test against a realistic copy of an existing `podcasts.db`; add this before making destructive or rename-style schema changes.
