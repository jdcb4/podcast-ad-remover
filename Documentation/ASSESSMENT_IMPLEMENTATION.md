# Assessment implementation

Baseline: `cc21a258d966df3c425fdc2f807a7ba9b5c24071` (1.12.0).
Scope: the 25 findings in the 2026-09-05 assessment, approved by Joe.
Production promotion is separate. Existing databases and media remain compatible.
The future-feature rehearsal does not add a new ad-review product workflow.

## Completed batches

- F25: remove unused prepend/download/queue wrappers, obsolete path properties and the Python 3.11 timezone fallback. Legacy data reads remain covered. The full 304-test Python 3.11 gate passed with these removals.

- F21 standalone speech follow-up: enforce the configured timeout and a per-operation request ceiling, and preserve HTTP authentication failure classification. 11 provider/budget tests passed.

- F14/F15/F23 browser follow-up: one Jinja card partial serves initial and paginated episodes; native confirmation dialogs, title disclosures, missing field labels and mobile list expansion are fixed. Removed superseded dropdown markup and conflicting list CSS. Desktop/390 px browser checks confirmed complete expanded text, no horizontal overflow, Cancel/Escape and focus return. F05 cancellation now preserves published audio across queue, legacy and v1 routes. 21 focused HTTP/DOM/filter tests passed.
- F05/F22: a separately spawned two-process test confirms global queue capacity; all 7 processing recovery tests passed.

- F17 lifecycle follow-up: shutdown clears owned child/supervisor state and disabled processing remains healthy after a prior worker stops. 23 health, deletion and custom-provider tests passed together.

- F20/F21: transactional scratch admission, stage free-space checks, durable shared request budget (analysis/summary/remote speech), explicit SDK timeouts, provider-aware retry timing, usage records and verified stage reuse. 17 focused tests passed, including an actual failed-cut retry with one download/transcription/analysis total.

- F17/F24, F20 status scans and F23 shared settings: worker heartbeat, real scheduling, bounded child restarts, readiness endpoint, cgroup memory, off-thread cached storage, selected-provider readiness and initial Whisper setting. 22 operational/security tests passed.

- F12/F14/F15/F16 and F22/F23 UI: server filtering, latest-request handling, native dialog, accessible labels, checked action responses, direct RSS search and extracted episode script. 25 focused tests passed, including shipped JavaScript against real rendered HTML in jsdom. Frontend audit now has zero findings.

- F01/F03/F05/F08 and F16 output duration: ID-based artifact roots, per-attempt staging, claim-token write fences, lease-preserving cancellation, transactional capacity, atomic serialized RSS, durable publication retries and last-good reprocessing. Tests include real audio, late workers, concurrent connections and injected feed/provider failures. Web playback and legacy artifact lookup use recorded paths.


- F09: shared bounded redirect streams with validated IP pinning/TLS SNI; 39 feed, artwork, URL and source tests passed. F20 streaming free-space checks added; stage budgets follow.

- F07/F10: strict finite segment validation and codec-aware MP3 output; 20 checks passed including six real FFmpeg MP3/AAC/Opus cases.

- F06/F11/F19 and F16 listen attribution: shared ownership policy, real SQLite + HTTP regressions, safe range parsing; 44 focused HTTP/auth/library tests passed.

- F13: known placeholder rejection, Compose interpolation and first-run instructions; 12 startup security tests passed.

- F04: online backup helper, pre-DDL snapshots, safe dry runs and recovery runbook; 23 backup/migration/queue tests passed.

- F02: escaped report renderer and script-free sandbox for all legacy report responses; 7 report/SponsorBlock tests passed.

- Filesystem deletion guards: reject podcast roots, nested subscription paths and symlink aliases. Focused maintenance tests cover root and sibling preservation.

## Remaining batches

- F01: stable artifact identity with compatible legacy reads.

- F03–F05, F08: atomic publication, WAL-safe backups, fenced processing attempts, cancellation and reprocessing recovery.
- F06–F07, F09–F13: permissions, strict analysis, redirects, audio formats, membership, delete UI and configuration secrets.
- F14–F17, F19–F21: accessibility, server filtering, metrics, health, HTTP errors, resource budgets and retry policy.
- F18, F22–F25: reproducible dependencies, meaningful regression coverage, smaller shared modules, accurate operational docs and safe removals.

Each coherent batch is committed after its relevant checks. Final validation and limitations are recorded here before completion.
