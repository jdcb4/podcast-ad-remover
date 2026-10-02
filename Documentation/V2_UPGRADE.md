# V2 upgrade and compatibility guide

This guide describes the implemented V2 code on `dev`, not an announcement that 2.0 or a matching `latest` image has been published. Check the exact image revision you intend to install. The [release notes](V2_RELEASE_NOTES.md) explain why this is deliberately a breaking-change boundary.

## Before upgrading

1. Record the previous immutable image tag/digest, installation environment, persistent `/data` location and session secret. Keep credentials private.
2. Stop new processing activity and let running jobs finish or cancel and wait for worker acknowledgement. V2 migration refuses a database with jobs still marked `running`. Merely starting with `PROCESSOR_ENABLED=false` does not erase stale running claims; resolve them with the existing version first.
3. Take an integrity-checked online database snapshot and a matching media recovery point using [RECOVERY.md](RECOVERY.md). Startup's automatic database backup does not replace a full backup of media or an off-device copy.
4. Rehearse migrations against a copy with `scripts/migration_dry_run.py`. Test an isolated copy with `PROCESSOR_ENABLED=false`, its own volume and port, and the intended immutable image. Never run old and new applications against the same writable data directory.
5. After reviewing the rehearsal, stop the old application before attaching the real data to the new image. Preserve the original data and backup for rollback.

No command in this guide authorizes an agent to stop a user's deployment or publish an image; coordinate those actions with the operator.

## What changes during migration

Migration `20260927_0023_v2` runs through the normal backup-aware migration runner. It retains historical columns needed by the upgrade/recovery path; their presence does not make retired features supported.

| Area | Conversion and follow-up |
|------|--------------------------|
| Models | Select the first nonempty configured model for each provider and the former Gemini speech model list. Review that selected model in Settings. |
| Credentials | Retain the first saved Gemini key; use the first nonempty environment key ahead of saved credentials. No automatic rotation. Remove obsolete extra keys from your installation configuration after reviewing the active one. Migration reports exclude credentials. |
| Speech | Existing Gemini speech remains selected. Former Piper/local speech becomes unconfigured. Requested spoken features stay saved, core processing continues, and an actionable warning asks you to configure Voice. No paid replacement is chosen. |
| Timeline | All processing uses Complete Timeline. Queued, retry-scheduled and rate-limited jobs are converted; compatible frozen timeline rules/options are retained and normalized. Running jobs must be drained. |
| Instructions | Global free-form detection instructions and whitelist behavior are retired. Nonempty podcast-specific classification guidance still applies; review old guidance for conflicts with category definitions. |
| Editorial audio | Direct editorial non-speech removal is disabled. The advanced retained-island rule remains independent (default 10 seconds; zero disables it). |
| Legacy summary umbrella | Old `append_summary` intent is converted into the supported description/audio-summary flags and the umbrella is disabled. Review effective inherited feature settings. |
| Cut tones | Enabled if any previous beginning/middle/end switch was on. Mixed old settings now produce tones at all applicable cut positions; System's upgrade report records the conversion. |
| Unified feed | Description becomes a fixed default. Name and episode-title prefix remain; an existing external artwork URL selects the URL artwork source. Upload is now another option. |
| Onboarding | Fresh installations offer the wizard. Existing installations do not automatically enter it; run it later from System. |
| Ownership | Unchanged. The proposed mandatory-admin-owner migration and changed membership/removal semantics are deferred. Unowned podcasts can still exist. |

Fresh databases enable the artwork badge and cut tones. Other enhancements start off. Existing configurations are not reset to these fresh-install defaults.

## What is preserved

The migration does not reset `/data`, rewrite existing audio, remove old Piper model files, change podcast ownership, delete memberships, rotate user/feed/API tokens, or change published feed identities. It does not automatically reprocess completed episodes. Normal retention and subsequent processing can still replace or remove media, so preserve the matching recovery point.

Local transcription remains faster-whisper. FFmpeg, CTranslate2 and shared dependencies such as ONNX Runtime remain needed; Piper removal does not remove the transcription stack. Public YouTube support still uses yt-dlp/Deno, without SponsorBlock.

## Review after startup

- Read the **System** upgrade report, then confirm the application URL and client feed links.
- Check **Text analysis** provider/model/key precedence and native structured-output support. There is no schema-free or different-model fallback.
- Configure **Voice** only if wanted. Refresh metadata or enter supported IDs; an explicit speech preview can incur charges. Unconfigured spoken controls are disabled without losing saved intent.
- Review **Podcast defaults**, per-podcast overrides, prompt rules, retained-island timing and the single cut-tone switch. Compare a representative report and listen at edits before broad reprocessing.
- Verify **Users & access** login, feed tokens and API tokens. API stays at `/api/v1`, but retired PATCH values are rejected; review [API.md](API.md).
- Remove `SPONSORBLOCK_ENABLED`, obsolete Piper configuration and `INSTALL_TTS` build arguments from deployment files. They no longer configure supported behavior.
- Verify playback, individual/unified feeds, artwork, queue/worker health and your actual provider. Existing published audio should remain available during failed replacement attempts.

## Rollback

Stop the new application, preserve its current data for diagnosis, and restore the pre-upgrade database into a separate recovery directory together with its matching media tree. Do not combine that database with newer WAL/SHM files. Start the recorded prior immutable image against the recovered copy, initially with processing disabled, and verify login, feeds and playback before switching traffic.

**An image-only downgrade is unsupported after V2 database writes.** See [RECOVERY.md](RECOVERY.md) for the full procedure. A major-version number permits deliberate feature changes, not loss of the user's database or media.
# Source and archive controls

See [PODCAST_OPERATIONS.md](PODCAST_OPERATIONS.md) for the additive migration, automatic pre-upgrade database backup and rollback requirements. Existing retention values and thread counts (including zero) are preserved; only fresh databases start with three Whisper CPU threads.
