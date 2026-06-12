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
- Add safer backup/export guidance before upgrades.
- Split large templates and move inline queue/episode JavaScript into static files.

## Maintainability

- Continue migrating new schema work to explicit migrations; older ad hoc column migrations remain for backward compatibility.
- Split very large route and processor modules when tests are in place.
