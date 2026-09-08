# Roadmap

This roadmap lists improvement candidates. It is not a release commitment.

## Reliability

- Expand Python coverage around full processor lifecycle transitions and service boundaries.
- Expand migration tests so they run against a copied realistic `podcasts.db`.
- Measure long-episode performance and exercise paid-provider/unattended processing with a representative dev workload before production promotion.

## Security

- Tighten admin/API authorization tests further before adding more remote management features.

## Resource Usage

- Add documented concurrency and CPU guidance for small homelab machines.
- Make Whisper model choice, worker limits, cleanup policy, and retry settings easier to reason about from the UI.

## User Experience

- Expand first-run setup into a guided wizard for API keys and recommended defaults; the current System Settings checklist covers admin credentials and URL/feed checks.
- Add clearer queue state explanations for failed, rate-limited, ignored, and unprocessed episodes.
- Add optional token-attributed feed/audio access logging if admins need true per-user download analytics. Current stats show per-podcast user-library counts and aggregate plays.
- Add dynamic per-user file serving so each user can keep podcast-specific preferences and receive a personalized episode file generated when their podcast client downloads it.
- Add optional podcast classifications that can drive differentiated defaults for retention, queue order, and feed handling: **finite** shows keep a complete start-to-finish catalogue; **current affairs** keep a recent rolling window; **narrative** shows default to chronological processing from the beginning; and **seasonal** shows support season-aware retention and, where useful, separate RSS feeds per season. Classifications must remain optional, preserve existing settings on upgrade, and allow per-podcast overrides.
- Continue extracting independently tested route/processor responsibilities when a concrete feature needs them; episode cards/JavaScript and shared policy/artifact modules are already separated.

## Recently Completed

The approved September assessment is tracked in [ASSESSMENT_IMPLEMENTATION.md](ASSESSMENT_IMPLEMENTATION.md).
It adds fenced attempts and capacity claims, last-good publication, atomic feeds, WAL-safe backups,
request/scratch budgets, worker supervision/readiness, server filtering, shared accessible cards,
reproducible dependencies and recovery/HTTP/real-audio tests. Re-review historical candidates against
that implementation before starting a new change.

- Preserve Library position and filters while starring podcasts.
- Add optional ad-free artwork badging.
- Add four explicit, reversible global-setting inheritance groups.
- Add a compact podcast table and atomic, permission-checked bulk settings editing.
- Make the mobile dashboard denser, reduce subscription choices to the direct feed and a generic app workflow, and deemphasize the unified feed.
- Bound the current-processing panel and refresh its queue data without reloading the dashboard.
- Switch between My Podcasts and Library in place while retaining filters, layout preference, scroll position, and normal-link fallbacks.
- Show effective global values in inherited podcast controls while retaining and restoring stored overrides; verify current global retention in repository reads, feed discovery, and cleanup.

## Maintainability

- Continue migrating new schema work to explicit migrations; older ad hoc column migrations remain for backward compatibility.
- Split very large route and processor modules when tests are in place.
