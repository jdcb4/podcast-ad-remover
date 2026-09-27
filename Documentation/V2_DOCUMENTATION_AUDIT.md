# V2 documentation reconciliation â€” 2026-09-27

Scope: revise current user/developer guidance against the implemented V2 code and confirmed user decisions; prepare the GitHub README and draft release/commit explanation. This is documentation preparation, not production publication or a claim of a real installation upgrade.

## Current guidance reviewed and updated

| Surface | Resolution |
|---------|------------|
| README | Replaced accumulated 1.x/V2 fragments with a coherent V2 overview, deliberate-breaking-change rationale, local test install, channel distinctions and links. Removed outdated screenshots from the current UI presentation. |
| V2_UPGRADE / RECOVERY | Added a user-facing conversion/action table, running-job precondition, data-preservation limits and matched image/database/media rollback. Removed active instructions to select Legacy again. |
| V2_RELEASE_NOTES / VERSIONING / GIT_WORKFLOW / AGENTS | Prepared draft release and commit copy; recorded Joe's explicit request to expand his reasoning before publication. Kept version bump and promotion separately authorized. |
| Architecture / Data_Flow / COMPLETE_TIMELINE | Documented single workflow/model/key, current inheritance, guidance, API speech, import, onboarding, consolidated access and unified artwork/description behavior. |
| Environment_Variables / env.example / config field descriptions | Removed multi-key failover/SponsorBlock examples, clarified environment precedence versus startup seeds, discovery limits, quota behavior and retention semantics. |
| Deployment / Unraid guide and XML | Corrected settings destinations and stale version text; distinguished production tags from unpublished V2; preserved persistent storage and install secrecy. |
| API / Agent_Skill / portable skill | Preserved audited endpoint/permission contracts; linked V2 compatibility, included linked upgrade/recovery references in the portable package and distinguished installed from packaged version. |
| VERIFICATION | Replaced retired Legacy/cascade/Piper checks with V2 checks, corrected ARM64/dependency commands and mobile/feed expectations; added documentation/publication preflight. |
| CHANGELOG / DECISIONS / ROADMAP / V2_PROPOSAL / V2_IMPLEMENTATION | Separated implemented work from deferred ownership/whisper.cpp and from release acceptance; dated old decisions and retained proposal history with later changes called out. |
| PROJECT_INDEX / NAMING / CONTRIBUTING / SECURITY / PRODUCT | Aligned terminology, data paths, authorization/privacy boundaries, new modules and documentation authority. |
| RESOURCE_AUDIT | Replaced current guidance with V2 resource controls; preserved original measurements in a separate historical record. No unmeasured image-size/RAM reduction claimed. |
| WARNING_TONES | One Insert tone at content cuts switch, fixed assets, fresh-on default and mixed-position migration. |
| CUDA | Updated navigation; retained explicit experimental support and hardware qualification limits. |
| DESIGN.md / design sidecar | Reviewed as incumbent visual records; no visual-system change is required for this documentation task. |

## Historical evidence preserved

Dated production releases, Dev rollout, assessment, old design decisions, June audit, local-LLM research and the resource measurements remain historical records. Scope banners distinguish them from active instructions. Previous changelog entries remain versioned history rather than being rewritten as V2 behavior.

CUDA hardware reports and their CSV describe the actual dated hardware runs, not blanket current-host certification. The local-LLM HTML/JSON preserve their experiment's original results; their companion Markdown marks the research historical. Tone sample HTML is a development audition gallery, not a current app setting. Screenshot PNGs remain untouched with a directory scope note. LICENSE is unchanged. Compose manifests use supported variables and API-only image behavior; the production example now reads the documented private `.env` values and requires a session secret; production examples still intentionally select the production channel, with the V2 distinction stated in the guides.

## Validation and boundaries

Validation passed: local Markdown file targets and heading anchors, Unraid XML, environment-example keys against configuration fields, production Compose parsing, and portable/offline package link closure. `npm run verify` passed: 653 tests, 5 skipped; Python syntax, CSS build and frontend/Python dependency audits succeeded. The source skill API link is supplied by the package builder and was checked in the generated archive. This task performs no live host deployment, provider call, database migration on user data, Docker push, Pages publish or production version bump.

**Open before publication:** ask Joe to expand his reasoning, incorporate it into final public copy and record his review. The current explanation uses only the three reasons he supplied; no additional personal motivations or guarantees are invented.
