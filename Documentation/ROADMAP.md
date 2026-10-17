# Roadmap

Outstanding work, reconciled on 2026-10-17 against release history and the homelab backlog. This roadmap is not a release commitment or authorization to start deferred work. Completed features belong in [CHANGELOG.md](CHANGELOG.md).

## Deferred priorities

- **LLM timeout/retry investigation and historical episode recovery (HL-0065, high priority).** The chunk-seam fix is deployed. Remaining work is to correlate provider/proxy requests, qualify bounded client deadlines and retry behavior, and recover historical failed episodes. Deferred at Joe's request, 2026-10-16; recovery and live timeout changes need separate authorization.
- **Whole-episode promotional filtering.** Add Episode Removal in podcast defaults and a per-podcast settings tab, with a master toggle and category choices. The supplied twelve-category taxonomy initially excludes cross-promotions, trailers, paywall teasers and reruns, while keeping substantive crossovers, bonus episodes and announcements. See [the draft specification and backend options](EPISODE_REMOVAL_DRAFT.md). Backend choice remains open; this is not implemented. Requested 2026-10-05; expanded and deferred 2026-10-16.
- **Personalized per-user audio files.** Let users keep podcast-specific preferences and receive a personalized episode file generated when their podcast client downloads it. Deferred at Joe's request, 2026-10-16.
- **Optional podcast classifications.** Differentiate defaults for retention, queue order and feed handling: **finite** shows keep a complete start-to-finish catalogue; **current affairs** keep a recent rolling window; **narrative** shows default to chronological processing from the beginning; and **seasonal** shows support season-aware retention and, where useful, separate RSS feeds per season. Preserve existing settings on upgrade and allow per-podcast overrides. Deferred at Joe's request, 2026-10-16.
- **Review transcription engine and AMD GPU options (HL-0088).** Standalone faster-whisper/whisper.cpp CPU and GPU benchmarking on MiniJ and JMKtec is complete (HL-0085). Review the saved research before choosing whether to retain the current engine, replace it, support both engines, or enable AMD acceleration. Implementation and further qualification are deferred at Joe's request, 2026-10-17. No engine replacement is approved.
- Mandatory-owner migration and changed removal/deletion ownership semantics remain deferred. Do not implement them as incidental cleanup.

## Reliability

- Expand Python coverage around full processor lifecycle transitions and service boundaries.
- Expand migration tests so they run against a copied realistic `podcasts.db`.
- Extend long-episode and paid-provider/unattended processing qualification with representative workloads for future releases.

## Security

- Tighten admin/API authorization tests further before adding more remote management features.

## Resource Usage

- Add documented concurrency and CPU guidance for small homelab machines.
- Make Whisper model choice, worker limits, cleanup policy, and retry settings easier to reason about from the UI.

## User Experience

- Refine the implemented optional setup wizard based on real first-install feedback; keep it limited to essential choices.
- Add clearer queue state explanations for failed, rate-limited, ignored, and unprocessed episodes.
- Add optional token-attributed feed/audio access logging if admins need true per-user download analytics. Current stats show per-podcast user-library counts and aggregate plays.

## Maintainability

- Continue migrating new schema work to explicit migrations; older ad hoc column migrations remain for backward compatibility.
- Split very large route and processor modules when tests are in place.
- Extract independently tested route/processor responsibilities when a concrete feature needs them.

Before starting historical audit candidates, re-review them against [ASSESSMENT_IMPLEMENTATION.md](ASSESSMENT_IMPLEMENTATION.md) and the current implementation to avoid duplicating completed work.
