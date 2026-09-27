# Podcast Ad Remover v2 proposal

Status: **proposal for Joe's review; not an implementation or release authorization**.
Prepared 2026-09-27 against `dev` commit `22f0aaa` (application version 1.16.0).

## Direction

Make everyday podcast management fast on a phone, make configuration compact and consistent,
and reduce processing to one supported path: local transcription, complete timeline
classification through one structured-output model, deterministic content cuts, and optional
API-generated speech.

This is a reasonable 2.0 boundary: local speech synthesis, legacy processing, model fallback
lists and several settings disappear. Existing podcasts, media, feed addresses, episode GUIDs,
users and valid tokens should survive. A major version permits feature changes; it does not
justify resetting user data. Version numbering and production promotion remain separate decisions.

Joe requested a proposal first. No application behavior changes are included with this document.
Updated 2026-09-27 with confirmed decisions, deferred ownership changes, SponsorBlock removal,
the static installation configurator and the in-app setup wizard.

## Authority and reference interpretation

Joe's message defines scope. The three supplied handoff files are design references, not
independent instructions or finalized requirements:

- `podcast_navigation_layouts.html`: desktop sidebar, compact browsing, mobile bottom navigation.
- `admin_settings_redesign.html`: compact settings rows and consolidated administration.
- `ui_redesign_spec_handoff.md`: useful synthesis, with several illustrative or conflicting details.

All three came from the supplied `81edf1a9-53d5-4339-b86d-19d163d86b34` handoff directory.
The proposal uses their structure without adopting their example accounts, metrics, model lists,
hostnames or unsupported features as product truth.

| Handoff detail | Proposed interpretation |
| --- | --- |
| Remote/OpenAI transcription engines | Out of scope. Keep current local transcription and GPU setup. |
| TTS choices omit Gemini | Keep Gemini; add API alternatives. |
| Three selectable cut sounds | Do not introduce sound selection. One switch, existing fixed Wooden notes. |
| Retention default of three episodes | Do not silently change existing retention values or defaults. Preserve all current retention dimensions. |
| Exactly one heading of a prescribed HTML level | One visible page title; meaningful semantic section headings/legends where useful, without repeated labels. |
| Tiny prototype type and controls | Preserve density, but retain readable text and at least 44px mobile action targets. |
| Admin button in both mobile header and footer | One primary mobile Settings destination; no redundant admin shortcut. |
| Green Live feed badge | Only show status derived from real feed state; omit decorative status claims. |
| Minimum 1400px artwork upload | Proposal: recommend podcast-sized square artwork, validate actual image content and resource limits; do not impose that prototype constraint without a product reason. |

## Confirmed decisions

Joe confirmed the following on 2026-09-27. These are accepted requirements for the planned
implementation, not evidence that the features have been implemented. Ownership changes are deferred.

| Decision | Accepted direction | Consequence |
| --- | --- | --- |
| Initial speech providers | Gemini, OpenAI, OpenRouter, Custom OpenAI-compatible | Broad choice without separate integrations for every vendor. Direct ElevenLabs can follow if its account voices are needed. |
| Former Piper installations | Mark speech as needing configuration; never silently select a paid service | Continue core processing without optional speech, expose an actionable warning and retain the requested speech preferences. |
| Ownership changes | Deferred | Preserve current ownership and membership behavior; no mandatory-owner migration in this scope. |
| Single cut tone migration | Enable if any previous position switch was enabled | Mixed settings become enabled at all applicable cut positions; disclose this in the upgrade report. |
| Podcast-specific classification guidance | Keep an optional field in advanced podcast settings; always apply it when nonempty | Remove the global inclusion toggle, not useful show-specific guidance. Global legacy free-form instructions are retired. |
| Model credentials | One active credential per configured provider, with environment precedence; no automatic key rotation | Simplifies endpoint configuration. Existing multi-key installations keep their first configured key and receive a migration notice. |

Environment precedence and removal of automatic credential rotation are explicitly accepted.
For each provider, use the first nonempty environment credential when present, otherwise the first
saved credential. Retain a saved credential as inactive while an environment value overrides it;
never display or log either value in migration reports. Explain the precedence change on upgrade.
Bounded retries of the **same** endpoint/model for transient failures remain useful and are not a cascade.

## Navigation and browsing

Use one application shell. On wide desktop screens, a roughly 240px sidebar contains Add podcast,
My Podcasts, Library, Tasks, Settings for administrators, and a compact personal feed action.
Account, theme and logout actions sit at its foot. Keep light and dark themes. Do not duplicate
this sidebar with a second permanent admin rail: entering Settings expands its section links
within the same navigation, with a clear route back to browsing.

The main area starts with the page title and a compact search/filter/sort bar. Remove the large
statistics cards and permanent add-podcast search block. Operational detail belongs in Tasks
and System; a small processing count can remain in navigation. Add podcast opens a focused
dialog for directory search or RSS/YouTube URL, including results, errors and duplicate handling.
Existing URLs remain usable through existing routes or redirects.

Desktop concept:

```text
Sidebar                 My Podcasts                       [List / Grid]
  + Add podcast         [Filter podcasts........] [Filter] [Sort]
  My Podcasts           [cover] Podcast title    ...desktop details... [Manage]
  Library               [cover] Podcast title    ...desktop details... [Manage]
  Tasks (2)
  Settings              Podcast content starts here, without a hero or stats row.

  My feed [Copy] [Apps]
  Account / Theme
```

Mobile concept:

```text
Menu                    My Podcasts
[Filter podcasts...........] [Filter / Sort]
[cover] Podcast title that can wrap                  [Manage]
[cover] Another podcast                              [Manage]

My Podcasts | Library | Add | Feed | Settings
```

On mobile, always render the simple list regardless of the saved desktop display mode. Each
row has artwork, title and one action icon. No checkboxes, bulk toolbar, counts, owner labels,
retention columns or secondary metadata. Tapping the title opens the podcast; the action opens
management for an owner/admin, or available subscription actions for other users. Accessible
names include the podcast title. Library add/remove actions remain reachable without pretending
non-owners can edit settings. Desktop retains bulk management and its permission checks.

The mobile drawer contains Tasks, account/theme controls and secondary links. Settings is only
present for administrators; omit it for other users rather than showing a disabled destination.
Feed copying uses the current user's authorized feed URL with visible success/failure feedback
and a selectable URL fallback. Never substitute an administrator's token. Allow bottom safe-area
space and keep the bar clear of dialogs, keyboards and Save actions. At intermediate widths,
use a drawer rather than squeezing the content to keep a desktop rail.

Preserve filter/sort/search state through My Podcasts/Library switches, browser Back and refresh.
Keep ordinary links/forms functional when progressive enhancement fails. Scope counts and
empty states must reflect the real user's library.

## Shared settings system

Create shared Jinja components and CSS for setting rows, switches, numeric inputs with units,
selects, text/secret inputs, disclosure sections, inline errors, dialogs and action menus.
Use the reference's restrained purple accent, not its decorative wrappers or simulated browser.

- Desktop: label on the left, naturally sized control on the right. Numbers about 6–8rem;
  ordinary text/selects about 18–24rem; URLs and model slugs may need more room.
- Mobile: keep short toggles inline when they fit; stack longer controls and labels. Compact
  does not mean smaller tap targets or horizontally scrolling forms.
- One switch style for binary settings. Selection checkboxes remain checkboxes where they
  select desktop rows; they are not configuration switches.
- Use accessible info buttons for optional help, operable by touch and keyboard. Essential
  errors, units, credential requirements and destructive consequences remain visible.
- One Save action per page form, disabled until changes exist; make saved, saving and failed
  states explicit. Prompt definition edits can have local Save/Cancel actions.
- Collapsed sections containing a validation error open automatically. Dialogs restore focus,
  support Escape and have a visible close action. No hover-only essential controls.

## Page-by-page specification

| Page | Primary content | Disclosed or secondary content | Remove or relocate |
| --- | --- | --- | --- |
| Transcription | Model, device, precision; compact actual GPU/CPU status with Set up/Test action | Setup progress, diagnostics and CPU/GPU precision overrides if needed | Repeated title, summary card; no new remote engine |
| Voice | Provider, model, voice; short explicit Preview action | Credential status/edit; custom base URL only when relevant | Piper, voice downloads and cascade selector |
| Text analysis | Provider, credential status/edit, one model | Custom base URL; advanced request/quota limits where still necessary | Cascade shuttle, multi-model ordering and fallback controls |
| Podcast defaults | Remove: Ads, Promos, Intros, Outros, Non-editorial audio. Enhancements: description rewrite, spoken summary, spoken title, artwork badge. One cut-tone switch | Retention section; Advanced timing with retained islands, default 10 seconds | Workflow choice, whitelist/blacklist, editorial-audio removal, global custom ad instructions |
| Unified feed | Name, include podcast name in episode title, artwork source and current preview | URL field or upload control, according to source | Feed address, editable description, page-wide restore defaults |
| Prompt rules | Compact category rows, definition excerpt, Default/Custom state, Edit | Inline full definition editor with Save/Cancel/Use default; summary instructions row | Legacy tab, inclusion toggle, output-format/schema controls |
| System | Compact labeled numeric rows for existing operational settings | Resources, networking and advanced limits in a few useful disclosures | Setup checklist, unified URL and API-token management |
| Users & access | Compact user rows; Add user; pending requests only when present | User edit dialog, API tokens, feed tokens, access settings, active sessions and login history | Separate navigation destinations for access requests and feed tokens |
| Notifications | Existing channels and event choices using shared rows | Channel-specific fields when needed | Repeated headings and redundant descriptions |
| Logs | Existing log viewer and useful controls | Existing detail as needed | No functional redesign |

### Details that matter

**Transcription:** the current form exposes model, requested device, CPU precision and advanced
GPU precision. Present model/device/active precision as the three primary controls; preserve a
valid CPU fallback precision when GPU is selected. Actual device may differ from requested device.
Keep that difference visible. A hardware probe must not be labelled a speed benchmark unless
it actually measures inference. Retain existing experimental GPU qualification limits.

**Providers:** environment-supplied credentials show a source badge without exposing their value.
Saved credentials show Configured plus Replace/Remove; do not refill password fields with secrets.
Changing provider must not accidentally clear another provider's settings. Model selection is a
searchable select with a manual slug option for custom endpoints or incomplete discovery.
Voice choices depend on the chosen model; do not assume one universal voice list. Unsupported
combinations fail validation before they are used for jobs. Preview is an explicit potentially
billable action, not a request automatically made whenever a dropdown changes.

**Retention:** retain latest-episode count, automatic retention days and manual-download retention
where supported; do not collapse them into the handoff's single count dropdown. Put global
defaults in one Retention section and preserve podcast overrides. Distinguish operational cleanup
intervals on System from podcast retention policy rather than duplicating a setting on both pages.

**Prompt rules:** retain all seven classification definitions, including editorial non-speech.
Removing its cut toggle does not remove its category. Add one Preview prompt action showing the
assembled instructions and schema, clearly identified as a global preview rather than an actual
episode request. Podcast-specific guidance stays in advanced podcast settings under the confirmed
decision above. No hidden compatibility switch or automatic prompt rewrite.

**Artwork:** use an explicit source selector: Default, URL, Upload. Display the effective artwork
beside it. Saving a different source changes what is published; retained old uploads do not
silently override a new URL. Validate image decoding, byte/pixel limits and type; re-encode
accepted uploads, store outside executable/static source paths under `/data`, and publish atomically.
A failed upload preserves the old artwork. Use stable public URLs suitable for feed clients.
The fixed feed description uses the existing default, "All your ad-free podcasts in one place."
Existing custom names remain unchanged.

**Users and tokens:** user rows show identity, role and Edit, with a compact token count/status
where useful. Edit contains password, role and account deletion controls. Preserve existing
ownership actions without adding the deferred transfer-on-deletion policy.
Access requests are a short actionable section only when pending. API tokens are application
access credentials, distinct from provider API keys and RSS feed tokens. Keep their scopes,
expiry, revocation and user association explicit in the editor. Show secrets only at creation
where the existing token design permits. Moving controls must not broaden permissions or rotate
valid tokens. Existing access-request/feed-access URLs can redirect to the matching hub section.

**Notifications:** retain Apprise-compatible behavior rather than restricting channels to the
prototype's four examples. Use consistent controls, but retain provider-specific required fields
and event selection. Never expose notification credentials in status text.

## Processing simplification

### Current code findings

- `app/core/ai_services.py` still implements legacy detection, whitelist prompt additions,
  normalization of legacy segment responses, model lists, key rotation and local Piper TTS.
- OpenAI-compatible and Anthropic adapters can retry without a native schema when `auto` mode
  receives a schema-support error. The OpenAI-compatible adapter caches this capability result.
- Complete Timeline's `timeline.parse_response` already uses normal JSON parsing plus strict
  category, range and complete-coverage checks. It does not currently strip Markdown fences.
- Classification may make one validation-correction request; a separate summary-only repair
  preserves valid classification when the summary fails. These are distinct from syntax cleanup.
- `processor.py` treats missing job snapshots as Legacy. Removing a settings dropdown alone
  would leave old queued jobs on the legacy execution path.
- The Dockerfile installs `requirements-tts.txt` by default, which includes `piper-tts`, and
  creates `/data/models/piper`. `tts_worker.py` and voice download code support that path.

### Proposed output contract

Always request native structured output and run application validation. Remove `auto/json/strict`
settings, capability downgrade caches and schema-free retries. OpenRouter text routing must
require schema parameter support. Custom endpoints must implement the supported structured-output
request format; a server that rejects it receives a clear configuration error.

Keep one small shared JSON decoder:

1. Trim whitespace and an optional leading BOM.
2. Try standard JSON decoding; optionally unwrap one surrounding JSON Markdown fence.
3. If prose surrounds the response, accept exactly one complete JSON object using the standard
   decoder, then apply the exact same validation. Reject ambiguous multiple objects.
4. Reject malformed/truncated JSON, duplicate keys, non-finite numbers, wrong field types,
   unknown categories and missing/overlapping timeline coverage. Never interpret a failure as
   an empty list of cuts.

Do not repair quotes, commas, timestamps or missing fields, use Python literal evaluation, or
introduce a JSON-repair dependency. Refusal/truncation metadata is checked before cleanup.
Recommendation: retain at most one same-model correction for a semantic coverage error and the
existing bounded summary-only retry. Neither changes endpoint or removes the schema requirement.
Transport retries respect cooldowns and request budgets. Unsupported-schema errors fail promptly.

Remove the legacy detection runtime and its prompt/response normalizers. Do not remove unrelated
historical media-path or transcript readers just because their names include "legacy"; existing
artifacts still need to be readable. Complete Timeline remains the only new processing workflow.

### Speech provider recommendation

| Provider | Proposed role | Adapter considerations |
| --- | --- | --- |
| Gemini | Preserve existing API users | Keep its native speech generation and explicit audio-format handling. |
| OpenAI | Direct API option | Speech endpoint, one model and supported voice. |
| OpenRouter | Broad model choice through one service | Documented OpenAI-compatible speech endpoint; speech-model discovery, model-specific voices. |
| Custom | User-managed gateway | Explicit OpenAI-compatible speech contract, base URL and isolated optional credential. |
| ElevenLabs direct | Candidate follow-up | Dedicated conversion endpoint and account voice IDs; useful if gateway coverage is insufficient. |
| Azure Speech | Defer direct integration | Regional speech endpoints and voice discovery; requires its own configuration contract. |
| Amazon Polly | Defer direct integration | Dedicated SynthesizeSpeech API and AWS authentication. |
| Google Cloud Text-to-Speech | Defer direct integration | Separate synthesis API and cloud authorization; not interchangeable with the existing Gemini integration. |

OpenRouter now documents `/api/v1/audio/speech` and speech-only model discovery. This supports
the proposed shared adapter, but does not establish that every vendor voice/model is available
or behaves identically. Verify actual combinations during implementation; do not hard-code
the prototype's old model list. [OpenRouter speech documentation](https://openrouter.ai/docs/guides/overview/multimodal/tts).

Direct services provide their own documented speech APIs:
[OpenAI speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech),
[Gemini speech generation](https://ai.google.dev/gemini-api/docs/speech-generation),
[ElevenLabs conversion API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).
The wider survey also checked [Azure Speech](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech),
[Amazon Polly](https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html) and
[Google Cloud Text-to-Speech](https://docs.cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize).
These are capability references, not a quality benchmark. No paid provider calls were made for
this proposal. Cloud-specific integrations beyond these candidates would add authentication
and configuration surface and are not recommended for the initial implementation.

Remove Piper worker/download/validation code, build arguments, requirements and **exclusive**
transitive dependencies. Check shared uses before dropping packages: transcription remains local.
Keep FFmpeg. Stop creating Piper directories in new images, but do not delete existing model files
or speech audio from `/data`. Normalize provider output through the existing audio pipeline;
do not assume raw PCM always has Gemini's sample rate or channel layout.

## SponsorBlock removal

Accepted additional scope: remove the SponsorBlock option and integration entirely. Remove
`SPONSORBLOCK_ENABLED` from supported configuration, Compose/templates, current installation docs,
the client module and processor calls. Remove external-cut merging where it exists solely for
SponsorBlock; retain generic interval logic used by timeline editing. YouTube downloading,
yt-dlp, its JavaScript support and Deno remain in scope as supported features.

The current client imports standard-library modules and shared `httpx`; no standalone SponsorBlock
package appears in `requirements.txt`. Remove only dependencies proven exclusive to removed code,
including the separate Piper audit. Keep historical attribution/provenance where existing reports
or retained data require it, without advertising SponsorBlock as a current feature.

New and reprocessed episodes must not reuse cached SponsorBlock cuts. Update cache/snapshot
versioning accordingly. Leave already published audio and historical reports unchanged. Remove
the environment option from generated installs; an old supplied variable must not reactivate it.

## Browser-only installation configurator

Accepted additional scope: a static page deployed automatically to GitHub Pages with application
deployments. It generates Docker run commands or Docker Compose files through a short guided
form. It does not install anything itself or communicate with a running Podcast Ad Remover server.

### Minimal flow

| Step | Inputs | Result or guidance |
| --- | --- | --- |
| Install format | Compose (recommended) or Docker run; shell for commands (POSIX or PowerShell) | Correct output syntax for the selected shell; Compose YAML remains portable. |
| Storage and address | Persistent host directory or named volume, host port, reachable application URL | Mount at `/data`, map the selected host port to internal 8000; explain that clients must reach the URL. |
| Access and hardware | Direct HTTP or existing HTTPS proxy; CPU default, optional NVIDIA GPU | Set secure cookies consistently; only offer proxy-header trust with the trusted-proxy explanation. GPU requires existing host support; do not install drivers. |
| Review and download | Selected image version, generated secret, optional provider credential fields | Preview/copy/download `compose.yaml` and `.env`, or a run command with an env file. Show next steps: start container, open app, run setup wizard. |

Generate a persistent session secret using browser cryptographic randomness, once per configuration;
editing the host port must not regenerate it. Allow explicit regeneration. Keep credentials out of
the displayed command line by using a downloaded env file. API keys are optional and collapsed by
default: say "Configure later in the app". Do not require a text or speech provider to produce a
valid install. If supplied here, explain that environment credentials override app-saved credentials
and changing them requires updating/recreating the container.

Every field is marked where relevant as Required for install, Optional, or Configure later.
Leave models, voices, content removal, retention, notifications, users/tokens and feed artwork to
the app. Speech is optional. CPU works without CUDA setup. Advanced timeouts and resource caps
are not part of the short wizard; link to the matching version's reference instead of reproducing
every environment variable. Do not emit retired SponsorBlock/Piper/cascade variables.

### Local-only contract

- All form state, validation, secret generation and file generation run in the browser. No form
  submissions, analytics, telemetry, remote fonts/scripts, provider requests or update checks.
- GitHub serves the static assets and receives normal page requests; entered configuration is
  never sent to GitHub or any other service. State this accurately rather than claiming the
  initial page download has no network activity.
- Bundle assets and version-specific metadata with the page. Provide a downloadable offline
  bundle that works without network access, including when opened locally without a server.
- Keep secrets and input values in memory only by default: no localStorage, cookies, URL query
  strings/fragments or logs. Clear/reset removes them. Files are saved only on explicit action.
- No server reachability tests or API-key tests. Model discovery belongs to the installed app.
  External documentation links are user-initiated navigation and contain no configuration data.
- Generate valid YAML, dotenv values and shell-specific commands from structured values; test
  spaces, quotes, dollar signs, backticks, newlines and Windows paths. Reject unsafe/unrepresentable
  values rather than injecting them into commands. Render previews as text, not HTML.
- Output uses the published image, a persistent `/data` mount and a restart policy. Do not copy
  the development Compose file's source bind mount or local image build into production installs.

### Publication and version alignment

Current repository automation has a verification workflow; image publishing is performed by
release scripts. Add Pages publication as an explicit automatic follow-on to a successful image
publication, using the exact published commit/image reference. A push to `dev` or `main` alone
does not prove that its image exists and must not advertise a newly deployed version.

Recommended layout: stable configurator at the site root, a clearly labelled `/dev/` configurator,
and versioned stable snapshots. Each page identifies its target version and emits an exact version
or immutable dev revision by default. Dev publication updates its own page without changing the
stable default. Build the full site artifact from published-version records so updating one channel
does not erase the other; serialize publication and reject stale jobs that would roll a channel back.

Use a GitHub Actions Pages workflow with the Pages artifact/deploy actions and narrowly scoped
permissions. Configure the repository's Pages source and deployment environment during implementation.
If image publishing stays local, the publishing helper dispatches this workflow only after successful
publication with a verified commit and image identity. This is automation of an already authorized
publication, not permission for the configurator workflow to promote or publish Docker images.
Pages failures are visible and retryable for that same release; they do not imply the image failed
or require rebuilding it. No credentials or generated user configurations enter the site artifact.
[GitHub custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## In-app first-install setup wizard

Accepted additional scope: a short optional general-configuration wizard shown for a fresh install,
with Dismiss/Set up later and a permanent "Run setup wizard" action in Settings. This replaces the
static setup checklist; it does not reintroduce it on System. Use the shared compact controls and
mobile layout. The installed app may contact its configured providers when the user explicitly
requests a test; the browser-only restriction above applies to the external configurator.

| Step | Minimum choices | Deferred detail |
| --- | --- | --- |
| General | Reachable public URL; require login and administrator setup through existing bootstrap rules | Proxy/network tuning, extra users, tokens and feed customization |
| Text analysis | One provider, credential if missing, one model; custom URL only for Custom | Quotas, budgets and prompt definitions |
| Transcription | Local model and CPU/GPU choice, CPU selected initially | Precision tuning; link to existing GPU setup/test flow when selected |
| Podcast defaults | Ads/promos/intros/outros/non-editorial audio switches; a simple retention preset with explicit values | Advanced timing, detailed retention overrides, notifications and artwork |
| Review | Summary of changes, Apply and finish, optional Add first podcast | Optional speech is a clearly labelled Settings link, not a required provider setup step |

Use existing default retention values in presets; never imply a preset changes unrelated settings.
Keep all optional speech features off on fresh installs unless deliberately enabled later. Recognize
environment credentials without displaying them and explain where to change them. Allow skipping
AI setup: browsing remains available, while processing readiness clearly identifies what is missing.
Do not make paid test calls automatically on Next, dropdown changes or Finish.

Persist wizard status in the database: not started, dismissed or completed, plus a wizard version.
Fresh databases start not started; existing installations migrate to not auto-showing the wizard,
while retaining the manual entry point. Upgrades must not mistake an empty library for a fresh
installation. Dismissal persists across restarts and does not reset settings or enable features.

Re-running loads current effective values. Apply only explicitly changed fields through the same
validation/settings services used by normal pages; do not reset hidden settings or save stale
defaults over newer edits. Review before saving, preserve form values on error, and make Cancel
discard unapplied edits. Separate side effects such as GPU setup or provider tests into labelled
explicit actions; cancelling cannot pretend those already-completed actions were undone.

Keep the wizard administrator-only after bootstrap, with existing origin/CSRF protections and
secure first-account creation. Dismissing optional onboarding does not bypass required account
bootstrap when authentication is enabled. Do not create the deferred fallback-owner identity.

## Ownership changes deferred

Joe deferred ownership changes on 2026-09-27. This supersedes the original request to assign all
unowned podcasts to an administrator for this implementation scope. Preserve existing behavior,
including nullable owners and the relationship between ownership and My Podcasts membership.
Do not introduce a fallback administrator, change the auth-disabled identity, migrate owners,
or add automatic ownership transfer on user deletion as part of v2.

The initial review found that auth-disabled requests use synthetic admin ID 0, membership removal
clears ownership, explicit reassignment permits NULL, and user deletion does not transfer podcasts.
These findings remain context for a future ownership task, not instructions to change them now.
The new setup wizard must use existing account/bootstrap rules rather than depend on that deferred work.

## Upgrade and rollback contract

Use the existing integrity-checked online backup/migration framework. Rehearse on a copy before
deploying. This proposal does not authorize touching any live database or deploying an image.

| Existing state | Proposed migration |
| --- | --- |
| Valid scalar model or ordered cascade | Keep the scalar or first nonempty configured model. Record discarded alternatives in the upgrade report, not runtime fallback settings. |
| Gemini speech configured | Keep provider, first model, voice and credential source. |
| Piper speech configured | Speech needs setup; do not repurpose text credentials into automatic paid speech. Retain feature intent and continue core processing without optional speech, with an explicit warning. |
| Legacy/whitelist processing preference | All future work uses Complete Timeline. Preserve compatible removal and retention choices. |
| Editorial non-speech removal enabled | Set effective behavior to retain this category. Report the change. |
| Old global custom detection prompt/instructions | Archive in backup/export for reference; do not inject into the new classification contract. |
| Deprecated `append_summary` umbrella | Resolve current effective text/audio features and map to explicit modern flags, preserving inheritance behavior. |
| Three tone switches | Accepted OR migration to one enabled flag; keep fixed tone assets. |
| Custom unified description | Use the fixed description for future feed publication; old value remains recoverable from backup. |
| Existing feed tokens and URLs | Preserve their values and authorization behavior. UI relocation requires no rotation. |
| Ownership and membership | Unchanged; mandatory ownership migration deferred. |
| SponsorBlock enabled or cached segments present | Disable further integration; exclude cached external cuts from new/reprocessed jobs. Preserve historical reports and already published audio. |
| Retention, overrides and published episodes | Preserve. Do not mass-reprocess or delete media on upgrade. |

Queue transition is explicit: quiesce processing before upgrade and do not rewrite actively
leased jobs. Completed episodes, old reports and their provenance remain historical records.
For queued/retryable jobs with Legacy or missing snapshots, create a versioned v2 timeline
snapshot from effective migrated settings and mark the transition in job history. Reuse valid
download/transcript artifacts, but invalidate legacy classification results. Existing Complete
Timeline queued jobs also need a controlled snapshot upgrade where model lists, removed options
or prompt settings differ; do not silently reinterpret a frozen snapshot. Failed historical jobs
remain failed until retried. New cache keys must separate changed prompt/schema/settings versions.

Migrations should be idempotent and produce an actionable report without credentials: model
selection, speech setup needed, retired prompts, tone changes and queued-job
conversions. Retain obsolete database columns temporarily if that avoids risky table rebuilds;
v2 must not continue reading them as live configuration. Return actionable validation errors for
removed API fields rather than silently honoring deprecated behavior. Document these API breaks.

Rollback means restoring the matching pre-v2 database backup and previous immutable image, with
the corresponding media recovery point if processing has occurred. Do not promise safe binary-only
downgrade after v2 writes. Preserve original media and backups; never reset `/data` to make migration
succeed. The release checklist must explain which post-upgrade changes a restore would lose.

## Implementation order and acceptance

Each phase is a coherent verified commit or small set of commits on a branch based on `dev`.
No production version bump, SemVer tag, image publication or promotion is implied.

1. **Migration foundations:** v2 settings, credential precedence and snapshot upgrade;
   migration rehearsal and rollback notes. Preserve existing ownership behavior.
2. **Processing cleanup:** timeline-only execution, single model, structured output contract,
   speech adapters, Piper/SponsorBlock removal, dependency cleanup and single cut tone.
   Align API models and readiness checks.
3. **Shared shell and browsing:** sidebar, mobile list/bottom bar, Add dialog, feed actions and reusable
   settings controls. Preserve routes, membership behavior, desktop bulk actions and themes.
4. **Settings and access:** convert each page, add artwork upload, consolidate users/tokens/requests,
   keep permissions and remove retired controls from podcast forms as well as admin forms.
5. **Installation and onboarding:** browser-only configurator, versioned install metadata,
   release-linked Pages publication and dismissible/re-runnable in-app setup wizard.
6. **Qualification and documentation:** end-to-end upgrade, provider/audio tests, desktop/mobile
   review and Docker verification; publish a v2 upgrade guide before any approved release.

Required acceptance evidence:

- Real SQLite upgrade fixtures: fresh install, auth disabled/no users, multiple admins, orphaned
  ownership, old schema, mixed inheritance and queued jobs; backup integrity and repeat migration.
- Ownership and authorization regression tests: preserve existing behavior during UI consolidation;
  ordinary users cannot gain unauthorized control. No new mandatory-owner assertions.
- Provider tests: one chosen model, mandatory schema parameters, no compatibility downgrade,
  minimal JSON acceptance/rejection, refusals, truncation, incomplete coverage and request limits.
- Speech contract tests for every shipped adapter, model/voice validation, credential isolation,
  audio conversion and failed responses; separately qualify real provider combinations with explicit
  test credentials and bounded requests during implementation.
- Audio fixtures: no cuts/no tone, start/middle/end cuts, merged cuts, retained islands and an
  entirely removed episode; editorial non-speech has no selectable direct removal policy.
- Browser checks at 360px, 390px, tablet and desktop in light/dark themes: mobile row simplicity,
  keyboard/touch dialogs, bottom-bar clearance, no overflow, long titles, no-results and errors.
- Rendered DOM and interaction tests: permission-based actions, saved/unsaved states, token scopes,
  field preservation, consolidated redirects, upload validation and filter/navigation persistence.
- RSS checks: stable GUIDs, enclosure addresses, authorized feeds, description/title choices,
  upload/URL/default artwork and failure preservation.
- Configurator tests: generated Compose parses and validates with `docker compose config`;
  shell output safely handles supported platforms and hostile special characters; no retired
  variables, source mounts or accidental credential transmission. Network interception confirms
  no requests on input/generation; offline bundle works and sensitive state is not persisted.
- Publication tests: stable/dev isolation, exact release alignment, complete site preservation,
  stale-job prevention and retry after Pages failure. No automatic production promotion.
- Setup wizard tests: fresh versus existing database, dismissal across restarts, re-run with current
  values, environment precedence, admin/bootstrap authorization, partial configuration, cancellation
  and preservation of unrelated settings. Include mobile/keyboard access and failed validations.
- SponsorBlock removal tests: no outbound lookup even with an old environment flag or cached
  segments; reprocessing uses only the current timeline policy, while old publications remain intact.
- `npm run verify`, then `npm run verify:docker` and isolated container smoke for implementation.
  Confirm the image excludes Piper and its exclusive dependencies; measure rather than promise
  an image-size improvement. No live-data mounts for these checks.

Update Architecture, Complete Timeline, Environment Variables, API, Deployment, Verification,
Warning Tones, Recovery, Project Index and Changelog alongside the implementation. Preserve
historical docs as history rather than rewriting their former behavior as though it never existed.

## Limits of this proposal

### Deferred transcription investigation

After the other planned changes are complete, investigate whisper.cpp as a possible alternative
to faster-whisper (Joe's request, 2026-09-27). Keep faster-whisper for the current implementation.
Compare transcription accuracy and timestamp suitability for cuts, speed, memory use, CPU/GPU
support and container integration using representative podcast audio. Record findings before
deciding whether to replace or add an engine. This investigation is not a v2 completion gate.

The handoff HTML and current templates were inspected as source; this is not a rendered visual
prototype or browser-QA result. Provider capabilities were checked against documentation, not
paid live inference. Ownership changes are deferred; queue transition details remain proposed. The
current application remains unchanged while these decisions are reviewed.

### Proposal verification

`git diff --check` passed. The required `npm run verify` was attempted on 2026-09-27
using the available local Python 3.13 environment (the supported application baseline is 3.11).
Python syntax passed; pytest reported 595 passed, 5 skipped and 2 failed:

- `test_anthropic_uses_native_format_then_explicit_compatibility_without_tuning`: the local
  environment has no `anthropic` module.
- `test_audio_tracking_uses_complete_path_and_invalid_range_is_client_error`: a malformed
  Range request returned HTTP 200 instead of the expected 400/416; cause not investigated here.

Verification stopped at pytest, so the CSS build and dependency audits did not run. These are
baseline findings with no application or test edits in this proposal; no complete-gate pass is
claimed. Resolve/recheck in the supported environment before implementation qualification.
