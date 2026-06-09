# Roadmap

This roadmap lists improvement candidates. It is not a release commitment.

## Reliability

- Add a real Python test suite around feed parsing, episode status transitions, RSS generation, and audio route authorization.
- Add migration tests that run against a copied `podcasts.db`.
- Continue expanding the durable job model with recovery tooling for orphaned work directories and richer worker lease visibility.

## Security

- Require a generated `SESSION_SECRET_KEY` in production rather than falling back to the development default.
- Consider tokenized feed and audio URLs for deployments exposed beyond a private network.
- Tighten admin/API authorization tests before adding more remote management features.

## Resource Usage

- Add documented concurrency and CPU guidance for small homelab machines.
- Make Whisper model choice, worker limits, cleanup policy, and retry settings easier to reason about from the UI.
- Add per-category storage reporting for original audio, processed audio, transcripts, and models.

## User Experience

- Improve first-run setup for base URL, admin credentials, API keys, and recommended defaults.
- Add clearer queue state explanations for failed, rate-limited, ignored, and unprocessed episodes.
- Add safer backup/export guidance before upgrades.
- Split large templates and move inline queue/episode JavaScript into static files.

## Maintainability

- Continue migrating new schema work to explicit migrations; older ad hoc column migrations remain for backward compatibility.
- Split very large route and processor modules when tests are in place.
- Add CI that runs verification on pull requests.
