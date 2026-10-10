# Dev review and 2.0 launch readiness — 4 October 2026

**Historical review.** Use the [11 October promotion review](V2_PROMOTION_REVIEW.md)
for current release gates. The October 8 seam fix supersedes the merger finding
below; Joe's expanded cascade rationale is now incorporated and awaits copy review.
The Pages blocker was rechecked and remains unresolved.

Reviewed `dev` at `d1b42cf1bf063b5066d33097de0162b23663b9eb`, matching `origin/dev` after fetch. Scope: legacy processing/configuration, V2 migration and onboarding, static install generation, Pages publication, and current launch documentation. This is a targeted readiness review, not an exhaustive security audit or production upgrade qualification.

**Outcome:** launch documentation is prepared for review and the local installer works. The hosted installer is blocked. The known transcription chunk-join defect remains. Joe's expanded breaking-change explanation is still required before final release copy/publication.

## Findings

### P1 — Hosted installer cannot deploy with the legacy environment rules

The [Pages run for current Dev](https://github.com/jdcb4/podcast-ad-remover/actions/runs/37177117509) failed before executing any steps. Its annotation says: `Branch "dev" is not allowed to deploy to github-pages due to environment protection rules.` Read-only GitHub API inspection found the allowed branches are `master` and `experimental/local-llm-transcript-chunking`. Direct HTTP checks found the landing page returns 200 but still serves the old Local LLM Ad Detection Evaluation report; `/dev/` returns 404. The root is not yet an installer landing page.

`scripts/publish_configurator.py` dispatches `.github/workflows/publish-configurator.yml` from `dev` for both channels, so both channels face this restriction. The repository's `master` → `main` rename and the move to `dev` have not been reflected in this environment policy. The Pages API's old source-branch field also says `master`, but build type is `workflow`; the explicit environment rejection is the verified cause, not an inference from that field.

**Action:** review and explicitly allow the intended `dev` workflow branch in the `github-pages` environment, retaining revision ancestry and immutable-image validation. Then publish the authorized candidate and verify the landing page, correct channel metadata, offline ZIP and agent ZIP. No repository permission or deployment settings were changed in this review. README links to the intended wizard but labels its current unavailability and supplies the working local path.

### P1 — The transcription merger still drops or repeats words at chunk joins

`app/core/ai_services.py`, `Transcriber._transcribe_chunked`, retains a segment solely when its start falls inside the half-overlap window. It does not reconcile text across the seam or account for segments spanning it. This affects the shared CPU/GPU path and can reduce classification/cut quality on long episodes.

Reproduced against the actual current method with synthetic model segments, without downloading models or audio:

- At the first 1190-second join, a first-chunk segment spanning 1180–1188 seconds is retained. A second-chunk segment spanning global 1185–1195 seconds is discarded, including its unique continuation after 1188.
- A first-chunk segment spanning 1188–1196 and a second-chunk segment spanning global 1191–1198 are both retained. Identical wording appears twice with overlapping timestamps.

The [real-podcast qualification](CUDA_LONGFORM_2026-09-24.md) previously demonstrated the same issue. The code still uses that algorithm. **Action:** treat seam reconciliation and listening qualification as a separate audio change; do not claim clean/word-perfect boundaries for 2.0. README and release notes now expose this limitation. No speculative merger fix is included in a documentation preparation task.

### P2 — Active docs advertised a retired setting

`Documentation/Environment_Variables.md` described an editable short-island threshold with zero disabling it. `Documentation/VERIFICATION.md` instructed testers to save/reload zero, and release notes called island cuts optional. Current `resolve_subscription_row()` fixes new processing at 10 seconds; the API accepts only 10 for that compatibility field. Existing queued snapshots can retain older frozen choices.

**Resolved:** align those guides and the release breaking-change table with runtime behavior. The lower-level timeline parser and historical columns remain intentionally compatible with frozen jobs; their presence is not evidence of a live user setting.

### P2 — Fresh-install output could be mistaken for an upgrade recipe

The installer generates a new session secret, defaults to `podcast-data`, and has no way to detect an existing deployment. Compose prefixes named volumes with its project identity; Docker run uses the literal volume name. Moving the Compose directory or switching formats can therefore appear to produce an empty library even though old data remains in another volume. Replacing the env file also changes the session secret.

**Resolved at the documentation/UI scope:** explicitly label the wizard as a fresh-install generator, explain volume identity and preserve-secret requirements, and link existing users to the backup-driven upgrade guide. The generated installation behavior is unchanged.

## Compatibility paths that should remain

- `v2_migration.py` rejects running jobs, selects the first model/credential, converts queued jobs, records changes without secrets, and leaves published media/ownership intact. Backup/idempotency tests pass. Retained schema fields support historical reads and rollback preparation; they are not invitations to delete user data.
- Provider resolution supports one selection, environment precedence and isolated custom credentials. Old names such as `ai_model_cascade` survive as storage compatibility names, without restoring model fallback.
- Legacy artifact paths and old transcript parsing remain compatibility readers. Removing them casually would make old publications or cached work unavailable.
- In-app setup uses expiring server-memory drafts, review/apply, cancellation and concurrent-setting conflict checks. The supported single-web-process topology is unchanged. Tests cover these flows; no real provider calls were made.
- The approved build-only frontend advisory exception remains visible in the gate. This review does not broaden it or describe npm dependencies as vulnerability-free.

## Installer usability review

The static installer preserves its existing compact dark interface. Native labels/selects, disclosure groups, visible keyboard focus, a status region and separate credential downloads support the task. It has no runtime network requests or persistent browser storage; CSP blocks connections. The optional credentials panel starts closed, and no speech service is enabled by generation.

| Dimension | Score / 4 | Evidence and limits |
| --- | --- | --- |
| Accessibility | 3 | Native controls, labels and focus; no assistive-technology certification performed. |
| Performance | 4 | Small static assets, no libraries or service calls in the generator. |
| Responsive layout | 3 | Desktop and 390px mobile inspected; generated output wraps with no horizontal overflow in the checked mobile state. |
| Theming | 2 | Intentional separate dark palette; limited tokens and no light-theme option. |
| Implementation integrity | 4 | Specific install task, browser-only generation, honest channel metadata and fresh-install guidance. |
| Total | **16 / 20** | Good within this bounded review. |

The mechanical design detector reported five palette advisories against the main app's DESIGN.md, not new defects. DESIGN.md explicitly distinguishes the standalone configurator palette; the existing design was preserved. Theming is a maintenance opportunity, not a launch blocker. No additional visual redesign is proposed.

## Verification for this change

- `npm run verify:docker` passed, including the standard verification gate: **714 passed, 6 skipped**, Python syntax, Tailwind build, dependency audits and local Docker image build. Python 3.11.15; no runtime dependency advisories reported. The explicit `braces` build-only exception was reported.
- Actual `docker compose config` validated generated output. Isolated Compose and Docker run probes verified that quotes, dollar signs and backticks survive the separate env file. Probes ran with no application startup, network or data mounts. Compose's serialized config escapes dollar signs; the container environment round trip is the decisive check.
- PowerShell parsed the generated script successfully. Existing tests cover shell quoting, stable/rotated session secrets, clearing state, offline package contents and separate media mounts.
- Desktop Compose and mobile PowerShell generation exercised in the browser. Mobile horizontal overflow check passed. Preview state was cleared and the default Compose form left open for Joe.
- Rebuilt offline configurator and portable agent package; generated archives are ignored deliverables, not committed source artifacts.

These checks do not validate a real provider/account, a production-data migration, all podcast players or the GPU host matrix. Local image building does not publish a Docker tag or Pages deployment.

## Before announcing 2.0

1. Resolve Pages deployment and remove temporary unavailable notices only after live verification.
2. Record Joe's expanded rationale and review of the final [release copy](V2_RELEASE_NOTES.md). The reminder was raised in this review and remains open.
3. Decide whether the known chunk-join limitation is accepted for launch or repaired and requalified first.
4. Rehearse the intended installation's database/media/image recovery point and chosen providers. The earlier review's [deployment-specific storage transition](V2_ADVERSARIAL_REVIEW.md) remains an explicit qualification item if that installation still uses the nonstandard patch; current production state was not inspected here.
5. Follow [VERSIONING.md](VERSIONING.md) for explicit promotion authorization, aligned version files, exact-candidate verification and publication. The package remains 1.16.0 during preparation.
