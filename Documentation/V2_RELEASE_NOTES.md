# V2 release and commit notes — draft

**Status:** prepared for review, not published release copy. V2 is implemented on `dev`; `package.json` still carries 1.16.0. No 2.0 tag, production promotion or publication is authorized by preparing these notes.

## Maintainer reminder before publication

**Ask Joe to explain his reasoning in more detail before publishing V2.** He explicitly requested this reminder. The explanation below is a first draft based on his stated reasons, not his final release message. Add his expanded reasoning, confirm the wording with him, then mark this reminder resolved with the date. It remains open until that happens.

This reminder also appears in AGENTS.md and the release checklist so it is encountered when publication is being prepared, rather than on an arbitrary date.

## Proposed GitHub release title

Podcast Ad Remover 2.0 — a simpler tool, with deliberate breaking changes

## Proposed release body

Podcast Ad Remover prepares podcast feeds with fewer interruptions, ready for your usual player. Add or import subscriptions, choose which content to remove, and let PAR download, transcribe and process new episodes. Optional summaries, spoken titles, a Unified Feed and separate audio storage help you manage a growing library.

We are deliberately making breaking changes in V2. After spending more time building and using Podcast Ad Remover, we have a clearer view of which features are useful for the long term and which were mostly being kept for compatibility.

We hope these changes will not affect anyone's day-to-day use. However, some installations may rely on the options we are removing, so we want to explain the decision rather than suggest this is a completely compatible upgrade.

Some legacy features no longer seem to have enough long-term value. Others add more complexity to configuration, maintenance and testing than their benefit justifies. Increasingly affordable remote inference also changes the tradeoff for some of the local-generation and fallback machinery we previously carried. We want PAR to be easier to understand and maintain, with fewer overlapping ways to do the same job.

Local transcription is staying. You can still use a self-hosted custom analysis endpoint if it supports structured outputs. Remote speech is optional, and upgrading will not silently enable a paid replacement for local speech.

### What is new

- Optional separate processed-audio storage with a resumable System → Storage migration, stable playback URLs and explicit verified-original cleanup.

- A compact desktop and mobile interface, clearer settings and consolidated Users & access.
- A short, optional setup wizard that can be dismissed or run again later.
- A browser-only Docker/Compose configurator for POSIX or PowerShell, including offline download and private local generation of environment files.
- Complete Timeline classification as the single processing path, followed by deterministic category removal and a fixed 10-second short-island policy for new processing.
- Reviewed RSS source replacement and durable whole-show processing with pause/resume/cancel and explicit archive retention.
- API speech through Gemini, OpenAI, OpenRouter or a custom endpoint, with one model and voice, metadata refresh and manual IDs where needed.
- OPML/text feed import with duplicate preview, per-feed selection, progress, stop and retry. Pocket Casts exports are supported.
- Unified-feed artwork upload, a single RSS subscription action and consistent light/dark navigation.
- An updated API reference, enforced-scope metadata in OpenAPI, a dry-run import endpoint and a portable PAR agent skill bundled with the configurator.

### Deliberate breaking changes

| Previous feature | V2 behavior / action |
|------------------|----------------------|
| Local Piper speech and optional TTS image layer | Removed. Former Piper installs become unconfigured for speech; core processing continues. Choose an API speech endpoint explicitly if wanted. |
| Text/speech model cascades and key rotation | One selected model and credential. Migration keeps the first configured value; environment credentials take precedence. Arrange fallback outside PAR if needed. |
| Legacy whitelist/blacklist workflow | Removed. All new processing uses Complete Timeline; queued/retryable old jobs are converted. |
| Models without native structured output / compatibility settings | Unsupported. Valid complete JSON objects can be unwrapped, but malformed output is not repaired and no schema-free retry occurs. |
| SponsorBlock | Removed, including its option. Public YouTube source support remains. |
| Global free-form ad instructions and inclusion toggle | Retired. Category definitions remain editable; podcast-specific guidance is retained and applied when nonempty. |
| Direct removal of editorial non-speech | Removed. Editorial material is kept unless an independent qualifying short-island cut applies. |
| Separate beginning/middle/end tone switches | One switch. If any old position was enabled, all applicable cut positions are enabled after migration. |
| Editable unified-feed description | Replaced by a fixed description. Name, podcast-title prefix, external artwork and uploaded artwork remain. |
| Configurable retained-island threshold | Fixed at 10 seconds for new processing. Old stored overrides are ignored; already queued snapshots retain their frozen policy. |
| Legacy API settings | `/api/v1` remains, but retired values/unknown PATCH fields are rejected. Update scripts against the installed OpenAPI schema. |

Fresh installations enable ad-free artwork and Insert tone at content cuts; other enhancements start off. Existing saved preferences are preserved except the documented conversions. Requested optional speech preferences remain saved while speech is unconfigured.

### Upgrade carefully

Back up the database and matching media, record the previous immutable image, and drain running jobs before switching versions. Startup creates an integrity-checked database backup and records an upgrade report under System. Review provider/model/credential choices, prompt guidance, speech and tone behavior after migration.

The migration is designed to preserve podcasts, memberships, existing ownership, users/tokens, feed identities and published media. It does not automatically reprocess published audio. Normal processing, explicit reprocessing and retention still change media afterward. **A data-preserving migration does not mean the old configuration behavior is preserved.**

Rollback requires the previous image and pre-upgrade database plus matching media recovery point. Do not run an older image against a database already written by V2. Follow [V2_UPGRADE.md](V2_UPGRADE.md) and [RECOVERY.md](RECOVERY.md).

Mandatory-owner changes and a whisper.cpp evaluation are deferred. faster-whisper remains the transcription engine. Experimental GPU qualification limits still apply.

### Validation and remaining release work

See the [4 October launch review](V2_LAUNCH_REVIEW.md) for current verification and unresolved findings. The [web installer](https://jdcb4.github.io/podcast-ad-remover/) is implemented but its Pages deployment is blocked by stale environment branch rules; the [local preview](INSTALL_WIZARD.md) is available. Do not announce the hosted wizard as live until that gate succeeds.

The existing CPU/GPU transcription chunk merger can omit or repeat words at chunk joins. GPU support remains experimental. See the [recorded audio-quality evidence](CUDA_LONGFORM_2026-09-24.md); no word-perfect cutting claim is made.

Re-run release verification for the exact approved candidate, rehearse the target installation and complete the maintainer explanation before publication. Automated checks do not certify every provider/account, production-data upgrade or GPU host.

## Suggested release commit message

```text
feat!: simplify Podcast Ad Remover for V2

Redesign podcast management and settings, add onboarding and install
configuration, expand API speech, and add reviewed OPML/text imports
and a portable agent skill.

Deliberately retire legacy paths that no longer justify their long-term
complexity, including options made less valuable to us by increasingly
affordable remote inference. Preserve user data through documented
migration and make configuration changes explicit.

BREAKING CHANGE: remove local Piper speech, model/key cascades,
Legacy whitelist/blacklist processing, schema-free analysis,
SponsorBlock and retired global settings. Convert queued jobs to
Complete Timeline. Former Piper users must configure API speech if
they want spoken features; no paid service is selected automatically.
An enabled old tone position enables all applicable cut positions.
Rollback requires the pre-upgrade database and matching image/media.
```

Use this as release/squash copy only if that workflow is explicitly chosen. Do not squash or rewrite the repository's normal fast-forward production promotion. The approved revision and release authorization are separate from this draft message.
