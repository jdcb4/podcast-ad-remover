# Assessment implementation

Baseline: `cc21a258d966df3c425fdc2f807a7ba9b5c24071` (1.12.0).
Scope: the 25 findings in the 2026-09-05 assessment, approved by Joe.
Production promotion is separate. Existing databases and media remain compatible.
The future-feature rehearsal does not add a new ad-review product workflow.

## Completed batches

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
