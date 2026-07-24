# Roadmap

This roadmap lists improvement candidates. It is not a release commitment.

## Reliability

- Expand Python coverage around full processor lifecycle transitions and service boundaries.
- Expand migration tests so they run against a copied realistic `podcasts.db`.
- Continue expanding the durable job model with recovery tooling for orphaned work directories and richer worker lease visibility.

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
- Add safer backup/export guidance before upgrades.
- Keep the Library view in place when starring a podcast into a user's personal list: update the relevant card asynchronously (or restore its scroll position) so users can star multiple podcasts while working down the list without being returned to the top.
- Add optional podcast classifications that can drive differentiated defaults for retention, queue order, and feed handling: **finite** shows keep a complete start-to-finish catalogue; **current affairs** keep a recent rolling window; **narrative** shows default to chronological processing from the beginning; and **seasonal** shows support season-aware retention and, where useful, separate RSS feeds per season. Classifications must remain optional, preserve existing settings on upgrade, and allow per-podcast overrides.
- Design a small, recognisable “ad free” logo and optionally watermark podcast artwork with it, so processed podcasts remain identifiable in podcast apps where cover images are the primary navigation cue. Preserve source artwork quality and provide a per-podcast opt-out.
- Split large templates and move inline queue/episode JavaScript into static files.

## Maintainability

- Continue migrating new schema work to explicit migrations; older ad hoc column migrations remain for backward compatibility.
- Split very large route and processor modules when tests are in place.
