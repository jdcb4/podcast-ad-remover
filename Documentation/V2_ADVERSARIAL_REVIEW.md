# Antigravity review disposition — 2026-09-28

Antigravity CLI 1.2.12 reviewed a tracked-file snapshot of dev `f6eb4a3` and the local test deployment at port 8767. It inspected source, templates, CSS/JavaScript and live GET responses. It did **not** run a graphical desktop/mobile browser, and its report's “verified” labels do not all establish a reproduced defect. The original report is retained in the separate local PAR_Adversarial_Review workspace.

## Accepted and fixed

- **Origin validation:** Applied the existing management guard with dashboard authentication disabled (`8841c73`). Also reject explicit opaque/malformed Origin values and headerless cross-site browser requests identified by Fetch Metadata. Headerless local non-browser clients remain compatible. Suppressing Referer alone does not suppress a normal cross-origin POST's Origin header; the report overstated that particular bypass.
- **Publication marker ordering:** Write a new revision marker after the fenced episode commit, not before media staging. The database protects the current revision if a crash prevents marker creation; preserve the previous committed revision marker before replacing its pointer. Simulated stage and commit failures no longer leave false published markers. Previously created markers are not automatically deleted.
- **Deferred episode deletion:** Storage unavailability defers physical cleanup instead of raising an HTTP 500 after marking an episode ignored. The existing maintenance loop retries ignored episodes. Migration still intentionally blocks destructive file cleanup.
- **Feed identity:** Share conservative URL identity with the repository, including historical URL lookup and serialized duplicate checks on creation. Scheme/host case and default ports normalize; path/query case and private query tokens remain significant. Do not use SQL COLLATE NOCASE across the entire URL as the report suggested.
- **UI/maintenance:** Use existing theme link and danger colors for active settings navigation and storage errors. Remove the obsolete mobile-menu handler and the unreachable JSON root route; `/` remains the dashboard and `/health` remains the health endpoint.

## Not accepted as defects

- **Alias restriction:** Deliberate fail-closed storage policy. Docker bind mounts do not inherently become filesystem symlinks inside the container. The listed NAS examples do not prove an unsupported deployment. Mount the real host directory into the configured container path; retain symlink escape protection.
- **Volatile setup drafts:** Deliberate short-lived, single-process onboarding state. Restart/expiry produces an actionable restart message; unsaved API keys are not placed in signed-but-readable cookies. Persisting draft secrets would add storage and encryption complexity. Multiple web workers are not the supported deployment topology.
- **Default local administrative access:** Deliberate trusted-network mode, documented separately from the origin-check defect.

## Larger work for Joe's approval

1. **Production-only storage-patch transition (recommended before production promotion).** Production's separate `EPISODE_ARTIFACTS_DIR` patch is outside the standard V2 layout. Design and test a backup-driven transition preserving transcripts, reports, cache reuse, cleanup behavior and existing feed/audio URLs. Do not silently run a consolidation or alter production mounts.
2. **Crash-orphan reconciliation (worth considering).** A process can terminate after staging a media mapping but before committing publication. Normal cancellation has cleanup guards; abrupt termination deserves a durable reconciliation pass tied to attempt ownership and publication history. Do not implement the report's simplistic mapping-inside-transaction proposal without considering long NAS I/O and old revision URLs.

The offline migration stall hypothesis is operationally plausible when processing is disabled and a previously running job remains. The running processor already has periodic stale-job recovery (not only startup recovery). Document safe recovery; do not automatically steal a job based solely on elapsed migration time.

## Verification

Regression tests exercise cross-origin/opaque-origin requests, staging/commit failure markers, deferred deletion/retry and historical URL equivalence while preserving path case. Full verification and local redeployment are required for the accepted batch. A separate rendered UI check covers the changed theme/navigation controls; it does not retroactively make Antigravity's review a visual audit.
