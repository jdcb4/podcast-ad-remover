# Next Dev pass qualification — 2026-10-03

Implementation: `ee3723c`, `b6dc680`, `5125168`, `fc5e51b` on local `dev`.

## Automated checks

- Full `npm run verify` passed during integration. Final `npm run verify:docker` repeated the complete suite: **707 passed, 6 skipped**, Python syntax, Tailwind rebuild and both dependency audits passed. The suite emits an existing Starlette/AnyIO deprecation warning; Tailwind notes an outdated Browserslist dataset.
- Local image `podcast-ad-remover:verify` built from `fc5e51b` with manifest-list digest `sha256:afc2af88bc66ceb6dbbfdeffc299e213a3864c47bbe7d14a44824fc57ce763c3`.
- An ephemeral container with networking disabled and `/data` on tmpfs verified fresh thread defaults/schema, actual RSS parsing, archive creation/pause/cancellation, and rendering of the library, podcast, Voice, Prompts and System pages.
- Focused tests cover source matching/aliases, stale previews, duplicate mapping/URL races, rollback, permissions, archive exclusions/ordering/priority/retries/recovery/retention, saved-versus-fresh defaults, credential precedence/removal, draft voice preview and temporary cleanup, discovery-value preservation, and Docker fallback/failure behavior.

## Browser checks

Used an isolated synthetic database at port 8770 with processing disabled. Checked desktop (1280px), phone (390px) and narrow phone (320px), both themes, long titles, keyboard podcast disclosure, prompt restore/preview/copy feedback, subscription actions and readable upgrade history. Confirmed a fixture archive, paused/resumed/cancelled it, and confirmed explicit uncertain feed mappings without duplicating episode identities. Fixture feed discovery and provider test responses did not use paid inference.

The temporary QA server was separate from the existing localhost:8767 deployment. Existing app data and remote environments were not used. No registry publication, GitHub push, production promotion or issue closure was performed. #14 stays open until production promotion. Before publishing V2, obtain Joe's fuller explanation of the deliberate breaking changes as recorded in AGENTS.md.

## Bounds

This qualifies the implementation and local Docker image, not live provider catalogs, paid voice generation, GPU inference, or a publisher's complete historical availability. Archive discovery is limited to RSS and supported explicit pagination; see [PODCAST_OPERATIONS.md](PODCAST_OPERATIONS.md). A subsequent deployment must use its intended verified revision and preserve existing data.
