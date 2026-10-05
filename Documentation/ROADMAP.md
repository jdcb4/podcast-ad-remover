# Roadmap

This roadmap lists improvement candidates. It is not a release commitment.

## V2 release preparation

- Before publication, remind Joe to expand his reasoning for the deliberate breaking changes and update [V2_RELEASE_NOTES.md](V2_RELEASE_NOTES.md).
- Qualify the exact release candidate against a copied real database and selected live providers; preserve rollback evidence. Publish only after explicit approval.
- Mandatory-owner migration and changed removal/deletion ownership semantics remain deferred. Do not implement them as incidental cleanup.

## Reliability

- Expand Python coverage around full processor lifecycle transitions and service boundaries.
- Expand migration tests so they run against a copied realistic `podcasts.db`.
- Measure long-episode performance and exercise paid-provider/unattended processing with a representative dev workload before production promotion.

## Security

- Tighten admin/API authorization tests further before adding more remote management features.

## Resource Usage

- After the planned v2 changes are complete, investigate whisper.cpp against the current faster-whisper engine (requested by Joe, 2026-09-27). Compare transcription and timestamp quality, processing speed, CPU/GPU compatibility, memory use and Docker integration on representative podcast episodes. This is an evaluation, not an approved engine replacement; keep faster-whisper during the v2 work.
- Add documented concurrency and CPU guidance for small homelab machines.
- Make Whisper model choice, worker limits, cleanup policy, and retry settings easier to reason about from the UI.

## User Experience

- Add LLM detection of promotional episodes outside a podcast's standard episode mix, with an optional setting to exclude them (requested by Joe, 2026-10-05). Examples include standalone trailers, cross-promotions and promotional announcements. Distinguish these from regular episodes containing ad or promo segments, which remain eligible for normal content removal. This is a roadmap item, not an implemented feature.
- Add OPML export for a selected podcast list (requested by Joe, 2026-10-05). Proposed flow: select one or more podcasts in either **My Podcasts** or **Library**, then use an **Export OPML** button in the bulk settings area to download the selected list. Disable the action when nothing is selected. This is a roadmap item, not an implemented feature.
- Refine the implemented optional setup wizard based on real first-install feedback; keep it limited to essential choices.
- Add clearer queue state explanations for failed, rate-limited, ignored, and unprocessed episodes.
- Add optional token-attributed feed/audio access logging if admins need true per-user download analytics. Current stats show per-podcast user-library counts and aggregate plays.
- Add dynamic per-user file serving so each user can keep podcast-specific preferences and receive a personalized episode file generated when their podcast client downloads it.
- Add optional podcast classifications that can drive differentiated defaults for retention, queue order, and feed handling: **finite** shows keep a complete start-to-finish catalogue; **current affairs** keep a recent rolling window; **narrative** shows default to chronological processing from the beginning; and **seasonal** shows support season-aware retention and, where useful, separate RSS feeds per season. Classifications must remain optional, preserve existing settings on upgrade, and allow per-podcast overrides.
- Continue extracting independently tested route/processor responsibilities when a concrete feature needs them; episode cards/JavaScript and shared policy/artifact modules are already separated.

## Implemented V2 features (not a publication claim)

- Compact desktop/mobile navigation and settings, consolidated users/access/tokens, and clear Unified Feed/theme/password actions.
- Complete Timeline only, native structured output, one model/key per provider, and retirement of Piper, SponsorBlock and legacy paths.
- API-only optional speech with provider/model/voice discovery and manual IDs.
- Optional rerunnable setup wizard and private browser-only install configurator, online/offline packaging and release-linked Pages hooks.
- OPML/text import with duplicate preview and per-feed outcomes; audited API reference and portable agent skill.
- One cut-tone switch, uploaded unified artwork, and explicit upgrade conversions/recovery guidance.

## Recently Completed

The approved September assessment is tracked in [ASSESSMENT_IMPLEMENTATION.md](ASSESSMENT_IMPLEMENTATION.md).
It adds fenced attempts and capacity claims, last-good publication, atomic feeds, WAL-safe backups,
request/scratch budgets, worker supervision/readiness, server filtering, shared accessible cards,
reproducible dependencies and recovery/HTTP/real-audio tests. Re-review historical candidates against
that implementation before starting a new change.

- Preserve Library position and filters while starring podcasts.
- Add optional ad-free artwork badging.
- Add reversible content-removal, retention and enhancement inheritance; V2 retires global instruction/workflow choices.
- Add a compact podcast table and atomic, permission-checked bulk settings editing.
- Make the mobile dashboard denser, reduce subscription choices to the direct feed and a generic app workflow, and place the Unified Feed action beside search.
- Bound the current-processing panel and refresh its queue data without reloading the dashboard.
- Switch between My Podcasts and Library in place while retaining filters, layout preference, scroll position, and normal-link fallbacks.
- Show effective global values in inherited podcast controls while retaining and restoring stored overrides; verify current global retention in repository reads, feed discovery, and cleanup.

## Maintainability

- Continue migrating new schema work to explicit migrations; older ad hoc column migrations remain for backward compatibility.
- Split very large route and processor modules when tests are in place.
