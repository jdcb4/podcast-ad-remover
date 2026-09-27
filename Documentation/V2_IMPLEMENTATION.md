# V2 implementation record

Current status: implementation and local qualification are complete for the recorded scope. Production upgrade rehearsal, live-provider/account checks, GPU host qualification and publication are still separate acceptance work. The milestone-specific counts below are historical checkpoints, not a claim that every release condition is complete. Current user guidance: [V2_UPGRADE.md](V2_UPGRADE.md).

Implementation authorized on 2026-09-27, on local `dev`. The accepted scope is in
`V2_PROPOSAL.md`. Ownership changes and whisper.cpp are deferred. No production promotion
or image publication is part of this authorization.

## Milestone 1: migration foundations

Additive migration `20260927_0023_v2` introduces speech endpoint fields, a single cut-tone flag,
onboarding state, artwork source selection and a credential-free upgrade report. It selects
the first old model/credential, marks Piper as unconfigured, converts queued jobs to timeline
snapshots and preserves owners, published media and feed identities. Running jobs block upgrade;
stop processing and drain them before migration. Fresh databases offer onboarding, existing
databases do not automatically show it.

The existing migration runner takes an integrity-checked database backup before modifications.
For rollback use the previous immutable image and pre-v2 database backup, plus matching media
recovery point if processing has occurred. A binary-only downgrade after v2 writes is not supported.

Validation: real SQLite migration tests cover backup integrity, repeat startup, queued snapshots,
credential exclusion, preserved ownership/media, fresh onboarding and rejection of running jobs.
The project `.venv` supplies Python 3.11 and the dependencies missing from system Python.

## Milestone 2: processing and speech

The processor now classifies Complete Timeline only. SponsorBlock calls and the local Piper
worker/dependencies are removed. Provider requests select one model and credential with environment
precedence; schema-free retries and key rotation are removed. A small JSON decoder accepts valid
wrapped objects and rejects malformed/ambiguous output. Speech adapters support Gemini, OpenAI,
OpenRouter and a separately configured custom endpoint, with bounded responses and atomic output.
One cut-tone switch controls all cut positions. Existing publication/report readers remain.

Validation: 23 focused migration, provider, speech-adapter and real-audio timeline tests pass.
At this milestone the broad suite still asserted retired legacy/cascade behavior. Those tests were subsequently revised and the full gate passed in milestone 5; this paragraph records the earlier state only.

## Milestone 3 — compact interface and onboarding

Desktop sidebar and mobile navigation replace the header; mobile podcasts use artwork/title/manage rows. Shared settings rows cover AI, defaults, system, notifications and feed appearance. Users now owns access requests, feed tokens and API credentials. Scoped settings saves preserve other sections. Added bounded raster artwork upload and an optional, resumable setup wizard. Ownership policy remains unchanged.

Focused verification: 39 migration/provider/pipeline/interface tests passed on Python 3.11. At that milestone full-suite and browser qualification were still in progress; see milestones 5–7 for later results.

## Milestone 4 — browser-only install configurator

`configurator/` generates Compose or POSIX Docker run instructions, with local cryptographic session-secret generation, shell/YAML escaping, named volumes or Linux bind mounts, optional environment credentials and NVIDIA GPU access. No inputs are persisted or transmitted; CSP denies network requests. Later-configurable options are explained without exposing every runtime setting.

Successful authorized Docker publish scripts dispatch `publish-configurator.yml` with the published revision and immutable tag. The workflow preserves stable/dev pages separately and verifies the image exists before Pages publication. GitHub Pages must be configured for Actions and the publisher must have authenticated `gh` workflow access. Dispatch failure is visible and independently retryable; it does not undo an already published image. No image or Pages publication was performed during implementation.

Verification: 17 configurator and publisher tests passed, including quote/dollar escaping, random-secret generation, GPU output and clearing secrets.

## Milestone 5 — qualification and review fixes

Updated the regression suite for mandatory structured timeline analysis, single model/key operation, API speech, migration conversions, optional podcast guidance and the third mobile rendering. Preserved real-audio pipeline/recovery checks. Provider preview failures now return actionable errors without raw endpoint details. The API rejects retired settings; empty podcast guidance clears correctly.

Independent interface review identified and resolved eight issues: staged wizard review/apply and conflict protection, missing minimal wizard choices, stable configurator secret lifecycle, separate env-file output, PowerShell and collapsed credential configuration, offline packaging, mobile navigation keyboard handling, and final viewport evidence. The reviewer returned **ship for the scored fixes**, not an unrestricted whole-app certification. Desktop/mobile library/settings, wizard and configurator captures plus a light tablet capture were inspected. Browser checks confirmed focus entry/return, Escape dismissal and local-file configurator generation without network requests.

Qualification: `npm run verify:docker` passes Python syntax, 626 tests (5 skipped), CSS build, npm audit and Python dependency audit, then builds `podcast-ad-remover:verify`. Both audits report no known vulnerabilities. An isolated disposable-data container served the setup wizard (HTTP 200) and imported faster-whisper. Generated Compose was parsed and a synthetic quote/dollar credential survived an actual container env-file round trip; PowerShell output passed the PowerShell parser. No live paid provider calls, real installation upgrades, GitHub Pages publication, GPU runtime test or production deployment were performed. Those remain deployment acceptance checks, not claims made by mocked contract tests.

Ownership and whisper.cpp remain deferred. No version bump or production promotion was made. V2 rollback still requires the pre-upgrade database backup and matching image/media recovery point; do not downgrade only the binary.

## Milestone 6 — UI feedback and speech discovery

Commit `038b5af` adds short settings explanations, clearer headings/status, expanded GPU setup, provider-aware voice/model lists and refresh with manual fallbacks. Tasks and Settings navigate separately; Unified Feed sits beside search; theme/password actions and primary navigation have icons. The mobile drawer contains settings/account items, closes from its backdrop/X, and supports focus handling. Fresh-install artwork and cut tones default on; optional spoken controls are disabled until configured while retaining saved intent.

Qualification: 632 tests passed, 5 skipped, audits and Docker build passed. Catalogue tests use mocked provider responses; they do not promise exhaustive live voices for every account.

## Milestone 7 — import and portable agent guidance

Commits `d133363` and `1e48a86` add nested OPML/plain-text import with duplicate preview, selection, sequential progress/stop/retry, safe label rendering and membership reuse. The API import endpoint defaults to dry-run. New podcasts inherit global defaults and wait for ordinary scheduled checks; existing owners/settings remain untouched. Input is capped at 100 entries and 1 MiB; query strings remain significant and redirect aliases are not guaranteed duplicate matches.

The API guide was audited against actual endpoints and settings. OpenAPI now reports dependency-derived scopes, and tests keep the documented inventory aligned. The portable skill is packaged with the canonical guide and included in online/offline configurators. Publication hooks remain conditional on authorized image publication.

Qualification at `1e48a86`: 653 tests passed, 5 skipped; CSS and frontend/Python audits passed; local Docker verification and a commit-tagged Dev build succeeded. The import UI reviewer returned **ship** for the reviewed desktop/mobile extension; design documentation was checked and preserved. No production container was replaced and nothing was pushed or published by this work.

## Documentation and release preparation

The README and active guides now describe the implemented V2 behavior, with historical documents explicitly labeled. [V2_RELEASE_NOTES.md](V2_RELEASE_NOTES.md) contains draft release/commit wording about intentional compatibility breaks, and [V2_UPGRADE.md](V2_UPGRADE.md) is the user-facing migration guide. Before publishing, remind Joe to expand his reasoning and incorporate it into the final announcement. This remains open; writing the draft does not settle it.


## Optional media storage — 2026-09-28

Added MEDIA_DIR with explicit destination enablement, durable stable-URL mappings,
copy/verify migration and separate verified-original cleanup. System → Storage and
the recovery CLI share the same background manifest. The configurator supports a
second mount. Appdata/transcripts and processing remain local; existing layouts
remain supported. Mount identity gates processing, playback and retention. See
[STORAGE.md](STORAGE.md) and the qualification scope in [VERIFICATION.md](VERIFICATION.md).
