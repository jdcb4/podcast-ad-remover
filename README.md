# Podcast Ad Remover

**Less skipping. More listening. Your usual podcast player.**

Podcast Ad Remover is a self-hosted app that downloads episodes, removes the ads and other content you choose, and publishes replacement RSS feeds. Subscribe once in your podcast player and let PAR handle new episodes.

Transcription runs locally with faster-whisper. Your chosen AI model classifies the episode timeline, then FFmpeg cuts the selected categories. You control the removal preferences and where analysis runs.

> **2.0 preview:** this README describes `dev`. The Docker `latest` tag is the production channel and does not imply V2 availability. Existing users should read the [upgrade guide](Documentation/V2_UPGRADE.md) first.

## Why use PAR?

- **Keep your player:** subscribe to individual shows or combine them in one Unified Feed.
- **Choose what to skip:** ads, promos, intros, outros and non-editorial non-speech, with defaults and per-show preferences.
- **Name feeds your way:** optional global prefix and suffix under Podcast defaults → Podcast titles, with a live preview.
- **Bring your subscriptions:** search for podcasts, add RSS feeds or public YouTube channels/playlists, and import OPML exports such as Pocket Casts.
- **Make episodes easier to browse:** optional rewritten descriptions, spoken titles and summaries, artwork badges and cut tones.
- **Run your own library:** local transcription, retention controls, processing reports, retries and optional notifications. Finished audio can live on a separate disk or NAS.
- **Manage it your way:** desktop/mobile interface, light/dark mode, shared library access and an optional scoped API with a [portable agent skill](Documentation/Agent_Skill.md).

PAR suits listeners willing to run Docker and provide an analysis model. Transcription needs CPU/RAM; remote analysis and optional speech may incur charges. AI can misclassify content and cut timing is imperfect. The existing transcription chunk merger can omit or repeat words at joins; see the [known limitation](Documentation/CUDA_LONGFORM_2026-09-24.md).

## Install

Start with the **[web setup wizard](https://jdcb4.github.io/podcast-ad-remover/)**. It generates Docker Compose or Docker run files for Linux/macOS or PowerShell, entirely in your browser. Credentials are not uploaded or saved in browser storage.

**Preview availability:** on 4 October 2026, the Pages homepage still shows an old LLM evaluation report and the Dev wizard returns 404 because deployment is blocked. Until it is published, use the local wizard below. The [installer guide](Documentation/INSTALL_WIZARD.md) covers both options and publication requirements.

1. Install Docker; generated Compose files require Compose 2.30 or newer.
2. Choose your storage, port and an application URL reachable by your podcast player.
3. Download the install file **and** `install.env` into the same folder, then run the command shown.
4. Open PAR. Its optional in-app setup wizard guides you through analysis, transcription and removal preferences. Add or import podcasts, then subscribe to their replacement feeds.

Keep the data volume and session secret across upgrades. The installer generates a **fresh installation**; existing users should preserve their current configuration and follow [Upgrade and rollback](Documentation/V2_UPGRADE.md).

### Preview the wizard locally

From this checkout:

```bash
python scripts/build_configurator.py
```

Open `configurator/index.html`, or serve only that folder:

```bash
python -m http.server 8778 --bind 127.0.0.1 --directory configurator
```

Visit `http://127.0.0.1:8778`. The source preview selects the rolling `:dev` image; it does not build or install this checkout. For an exact source test, follow [local V2 installation](Documentation/INSTALL_WIZARD.md#test-the-current-checkout).

See [Docker deployment](Documentation/Deployment.md), [Unraid](Documentation/Unraid_Deployment.md), [configuration](Documentation/Environment_Variables.md) and [separate audio storage](Documentation/STORAGE.md). Fresh installs use three Whisper CPU threads; Docker CPU/memory limits are separate controls.

## Why 2.0 makes breaking changes

PAR has accumulated overlapping options. We are retiring features whose usefulness no longer justifies their configuration, testing and maintenance cost, and focusing on one clear processing path. More affordable remote inference has also changed the value of some local-generation and fallback features for us.

V2 removes local Piper speech, model/key cascades, Legacy whitelist/blacklist processing, schema-free analysis and SponsorBlock. It simplifies prompt, feed-description and cut-tone settings, and fixes short retained gaps between cuts at 10 seconds for new jobs. These are deliberate behavior changes, even though the migration is designed to preserve podcasts, feed identities and published media.

**Local transcription stays.** A self-hosted analysis endpoint remains supported when it accepts native structured outputs. Speech is optional; upgrading from Piper never silently selects a paid replacement. Mandatory podcast ownership and a transcription-engine replacement are deferred.

Read the [draft 2.0 release notes](Documentation/V2_RELEASE_NOTES.md) for the full change list and rationale, and the [migration checklist](Documentation/V2_UPGRADE.md) for required actions. The maintainer's expanded explanation is still pending review before publication.

## Models, access and privacy

| Task | Options |
| --- | --- |
| Transcription | Local faster-whisper on CPU; [experimental NVIDIA GPU support](Documentation/CUDA.md). |
| Analysis | Gemini, OpenAI, Anthropic, OpenRouter or a custom OpenAI-compatible endpoint with native structured outputs. |
| Optional speech | Gemini, OpenAI, OpenRouter or a custom compatible speech endpoint. |

Choose one model and credential per task; there is no automatic provider fallback or key rotation. The analysis provider receives transcript/episode context; speech receives the text to synthesize. A self-hosted custom endpoint can keep analysis on your infrastructure.

Dashboard login, public subscription browsing, feed/audio protection and API access are separate settings. Protected feed URLs are bearer secrets. Keep them, provider keys and backups private. See [Security](SECURITY.md).

YouTube support covers public channels and explicit playlists, not individual video subscriptions or private/member content. Use sources you are permitted to download and process.

Admin pause controls preserve subscriptions and audio while queued processing waits. My Pods has a personal Unified Feed; Library has the global feed. Stats shows current holdings and recorded processing history. See [processing controls and statistics](Documentation/PROCESSING_CONTROLS.md).

## Documentation and development

- [Documentation index](Documentation/PROJECT_INDEX.md) · [2.0 readiness review](Documentation/V2_LAUNCH_REVIEW.md)
- [Complete Timeline](Documentation/COMPLETE_TIMELINE.md) · [Podcast/archive operations](Documentation/PODCAST_OPERATIONS.md)
- [API](Documentation/API.md) · [Architecture](Documentation/Architecture.md) · [Recovery](Documentation/RECOVERY.md)
- [Changelog](Documentation/CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Verification](Documentation/VERIFICATION.md)

Python 3.11, FastAPI/Jinja, SQLite, FFmpeg and Tailwind. Work integrates into `dev`; `main` is production. Run `npm ci` then `npm run verify`; release qualification also requires `npm run verify:docker`. [Versioning](Documentation/VERSIONING.md) covers explicit release approval and image publication.

[MIT license](LICENSE)
