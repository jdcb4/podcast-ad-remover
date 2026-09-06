# Assessment implementation

Completed 2026-09-06 on `codex/assessment-improvements`, based on `dev` at
`cc21a258d966df3c425fdc2f807a7ba9b5c24071` (1.12.0). Joe approved implementation of all
25 findings from the 2026-09-05 assessment. Application version remains 1.12.0; changes
are unreleased. Production promotion and deployment remain separate.

Subsequent integration, persistent Dev deployment and live qualification are tracked
in [the Dev rollout record](DEV_ROLLOUT_2026-09-06.md), including two additional fixes
found during real processing. The results below describe the original implementation gate.

## Finding coverage

| Finding | Implemented change | Main verification |
| --- | --- | --- |
| F01 Storage identity/deletion | Strict root/sibling/symlink guards; new ID-based episode roots and attempt directories; contained legacy reads. No bulk data rename. | Root/collision/symlink fixtures and legacy artifact/cleanup tests. |
| F02 Report injection | Extracted escaped renderer; script-free sandbox also covers previously generated reports. | Malicious feed/model text plus HTTP CSP regression tests. |
| F03 Publication failure | Guarded publication pointers, durable pending flag, locked atomic RSS replacement and publication-only recovery CLI. | Real FFmpeg pipeline with feed/replace failures; previous XML/audio survives and republishing succeeds. |
| F04 Backups/restore | SQLite online backup includes WAL, verifies integrity and refuses overwrites; pre-DDL backups and corrected dry run. | Uncheckpointed-WAL fixtures and actual dev database snapshot migrated in an isolated Linux container. |
| F05 Ownership/concurrency | Claim-token write fences, retained cancellation leases, attempt-owned cleanup, transactional global capacity and shared manual runner. Cancellation preserves published audio. | Late-worker/cancel/retry tests, HTTP cancellation and separate-process competing claims. |
| F06 Permissions | Shared owner/admin policy across browser, legacy API and v1 processing/removal paths. | Real SQLite/HTTP non-owner denial with unchanged data, plus existing API scope tests. |
| F07 Invalid analysis | Strict shape, finite timestamp/label validation, refusal/truncation errors and bounded schema repair. | Invalid response fixtures fail; valid empty arrays still publish no-cut output. |
| F08 Reprocessing | Stage replacements independently, retain old URLs/GUID until success and verify transcript/source provenance. | Failed replacement preserves playable old publication; cache mismatch and stage reuse tests. |
| F09 Restricted networking | Validate each redirect before connecting; pin validated public IP with TLS hostname checking; bounded hops and no environment proxies in restricted mode. | Mock transport/DNS tests prove private redirects are not contacted. |
| F10 Output codec | Copy no-cut MP3 sources; encode other input codecs to MP3. | Six short real FFmpeg MP3/AAC/Opus fixtures and Linux AAC smoke. |
| F11 Library membership | Membership count moved to the proper repository; real repository response works after commit. | HTTP add/remove/idempotency and form redirect tests. |
| F12 Episode action wiring | Single actions use the actual card/action contract and inspect response outcomes. | Shipped JavaScript against server-rendered HTML; absent obsolete delete-button ID is covered. |
| F13 Compose/secrets | Environment interpolation and persistent-secret setup; known example placeholders rejected. | Startup guards, sentinel propagation and both Compose configurations validated. |
| F14 Accessibility | Native action/confirmation dialogs, title/settings buttons, episode/field labels, visible focus and busy/status feedback. Mobile expansion repaired. | Desktop and 390 px browser keyboard, Cancel/Escape/focus return and full-text/no-overflow checks; rendered DOM tests. |
| F15 Truthful search/actions | Server-side filters/counts before pagination, latest-request ordering, retained results/failed selections and direct RSS entry. | Beyond-page-one and 550-result tests, delayed response ordering, forbidden action feedback and browser search. |
| F16 Duration/metrics | Measured output RSS duration, full-path listen identity, one browser listen event, true counts and accurate storage labeling. | Codec/RSS duration, Range/listen identity and dashboard tests. Storage headline covers current episode outputs; Queue shows all storage categories. |
| F17 Operations | Persisted heartbeat and real deadlines, supervised child restarts, readiness endpoint/Docker probe, cgroup-aware memory and cached off-thread storage scans. | Disabled/healthy/stale/dead-child tests, full-suite lifecycle checks, Linux startup and browser disabled-worker display. |
| F18 Dependencies | Universal Python 3.11 constraints, pinned bootstrap tools and Python/Deno bases, npm fixes, Python audits and container smoke. | Full gate, final-image inventory audit and pip consistency all pass. |
| F19 HTTP failures | Deliberate JSON 403 from middleware and controlled malformed Range responses. | Real middleware/audio HTTP tests including valid partial playback. |
| F20 Resources | Transactional scratch reservations, streaming/stage free-space checks and expiry of abandoned unpublished attempts. | Competing reservations and injected free-space/stage failure tests. |
| F21 Retry/cost controls | Durable job-wide request budget, explicit SDK timeouts/no hidden retries, provider-aware backoff, permanent auth/billing failures, usage records and reusable stages. Standalone speech also bounded. | Provider failure/Retry-After/budget fixtures; failed-cut retry performs download/transcription/analysis only once; speech auth stops after one call. |
| F22 Test effectiveness | Regression coverage at HTTP, real database/files, process, rendered DOM, codec and pipeline boundaries; Linux smoke and target-DB rehearsal. | 304 tests pass under Python 3.11, plus the container and browser evidence below. |
| F23 Maintainability | Shared settings/permissions, small artifact/report/publication/health/budget modules, static episode JS and one Jinja card renderer. | Existing contracts and new integration regressions; no frontend framework, broker or database replacement. |
| F24 Documentation/setup | Current architecture/data flow/recovery/security/verification guidance; selected-provider readiness including keyless custom endpoints; first-run Whisper seed; historical audit separated. | Fresh container startup, custom endpoint tests, real migration rehearsal and local documentation-link checks. |
| F25 Safe removals | Removed unused prepend/download/old-queue wrappers, deprecated unused path properties, timezone fallback, hidden dropdowns and conflicting list CSS. | Call-site review and full verification; legacy data/artifact compatibility retained. |

## Reviewable batches

The branch history separates containment/report safety, WAL recovery, configuration,
permissions/HTTP, analysis/codecs, redirect policy, processing/publication, UI, operations,
resource budgets, lifecycle fixes, browser/process follow-up, speech retry behavior,
unused wrappers, dependencies and documentation. Each code batch was checked before commit;
`git log dev..codex/assessment-improvements` lists them for independent review.

Keep SQLite, Jinja/Tailwind, shared podcast ownership, source/provider adapters, settings
inheritance, optional local transcription/TTS and explicit release promotion. These match
the scale of the application. `filelock` is now an explicit runtime dependency because
RSS writers must serialize across processes; it was already transitive. `jsdom` and
`pip-audit` are development tools with concrete regression/security value. No runtime
capability was removed and no speculative frontend/backend framework was added.

The future-feature rehearsal (review/edit cuts before publishing) remains a future product
feature. The staging, provenance and permission changes remove identified blockers without
adding an unrequested approval UI or waveform dependency.

## Final verification

- `npm run verify:docker`: **passed**, using local Python 3.11 and Docker on JMKtec because
  the local Docker daemon is unavailable. **304 tests passed**; syntax, CSS build, npm audit,
  Python audit and Linux image build passed. Both Compose files parse; version files align at 1.12.0.
- Final tested image: `podcast-ad-remover:verify`, image ID
  `sha256:22b7337f01700352953631a8344a7e1f1978eda4be2030b1e4e81749874230a6`,
  474,224,158 bytes as reported by Docker. It is a local verification image, not a published release tag.
- Full image Python inventory, including Piper and pip/setuptools: **no known vulnerabilities**;
  `pip check`: **passed**. The first image audit caught inherited bootstrap advisories, which were fixed
  and rechecked. No advisories were suppressed.
- Offline disposable container: native imports, fresh migrations, SQLite backup, publication recovery,
  real AAC conversion, app startup, health, dashboard and static assets **passed**.
- Native runtime: Piper import and real Whisper **tiny** CPU inference on a two-second tone **passed**
  in an isolated container capped at two CPUs/1 GB. This checks runtime compatibility, not speech quality.
- Existing dev database: an online integrity-checked snapshot successfully applied all 14 migrations
  in an isolated container. The original dev database still has 12 migrations and passes `quick_check`;
  its container remains `dev-30d4a70`, running with zero restarts. A read-only bind rehearsal first failed
  because SQLite could not initialize its shared-memory access; the corrected online snapshot was taken
  inside the existing app and all task-created snapshot copies were removed.
- Browser: disposable local data at `127.0.0.1:8188`, with automatic processing disabled and no paid keys.
  Checked desktop/390 px list/grid controls, far-page search, empty podcast, queue-disabled state,
  labels, full description expansion, Cancel/Escape and return focus. No horizontal overflow at 390 px.
- Local documentation links and `git diff --check` passed. Tailwind still reports old Browserslist
  metadata despite the installed metadata matching the registry's latest version; it does not fail the build.
  Starlette emits an upstream AnyIO deprecation warning; no tests fail from it.

## Boundaries and remaining operational checks

At the initial implementation gate, this branch had not been deployed, pushed, promoted,
version-bumped or published. The subsequent Dev rollout is recorded separately above;
production promotion is still unapproved. The implementation preserves `/data` compatibility,
but deployment still needs the documented matching media backup and immutable-image rollback plan.
The database rehearsal is not a destructive live restore or a full production failover exercise.

Paid provider billing/quotas and live ad-removal/voice quality, long-episode load/CPU/RAM behavior,
YouTube/SponsorBlock upstream behavior, notification delivery, screen-reader/WCAG certification,
Lighthouse/Core Web Vitals and a complete OS/native-binary vulnerability scan were not verified.
Deterministic tests deliberately stub external services; no paid provider request was made.

Scratch reservations are conservative estimates, not filesystem quotas. Other host writers can
still exhaust storage. Published historical revisions remain until episode retention/deletion;
monitor the Queue storage totals. Provider token counts only include usage returned by the provider
and are not an invoice. HomeLabRef history/project notes were updated and its structural/secret-pattern validation passed;
the previously recorded unrelated semantic catalog issue remains outside this project.

The main UI still permits inline event handlers in CSP; reports have the
stricter sandbox. Root-in-container compatibility and the single-web-worker topology remain explicit.
