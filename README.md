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

PAR suits listeners willing to run Docker and provide an analysis model. Transcription needs CPU/RAM; remote analysis and optional speech may incur charges. AI can misclassify content and cut timing is imperfect. Overlapping transcription chunks now preserve uncertain speech together; this can leave coarser cut boundaries and repeated transcript wording. See [timeline boundaries](Documentation/COMPLETE_TIMELINE.md#boundaries-and-processing-order).

## Upgrading to 2.0: what changes and why

**Read this before updating an existing installation.** V2 preserves your library through a database migration, but deliberately removes some features and changes how new episodes are processed. We are treating this as a major release because preserving data does not preserve every configuration or workflow.

PAR has accumulated overlapping ways to process episodes, select models and generate speech. Maintaining those paths adds configuration, testing and support work. V2 focuses that effort on local transcription, reliable timeline classification, audio processing and podcast feeds.

### One model instead of an internal cascade

Sufficiently capable paid models have become inexpensive enough that using one selected model is now a practical option for many installations. Pointing PAR at an affordable model is simpler than maintaining its own multi-model cascade for a benefit that has become more limited. Actual cost still depends on your model, episode volume and provider.

If you want a three-tier cascade, account rotation or more elaborate routing, dedicated tools such as [9Router](https://github.com/decolua/9router), [OmniRoute](https://github.com/diegosouzapw/OmniRoute) and [LiteLLM](https://docs.litellm.ai/docs/proxy/reliability) already address that problem. Maintaining a separate routing system inside PAR duplicates that work. You can configure a compatible gateway as PAR's custom OpenAI-compatible analysis endpoint, with routing managed there. The gateway and every model it selects must support PAR's native structured-output requests; these examples are not a certification of every gateway/model combination.

PAR itself now uses one selected model and credential per task, with no model/key/provider fallback. The upgrade keeps the first configured model and saved credential; environment credentials take precedence. **Review that selection after upgrading**, especially if you previously relied on a free-tier cascade. An exhausted model waits or reports a quota/error state instead of silently selecting another.

### Local speech synthesis is removed; local transcription stays

V2 removes bundled Piper text-to-speech and its dedicated dependencies. Optional spoken titles and audio summaries now use Gemini, OpenAI, OpenRouter or a compatible speech API. This reduces the local speech installation and maintenance paths; it does not remove faster-whisper, FFmpeg or their shared dependencies.

**If you used Piper, new spoken additions will stop until you explicitly configure Voice.** Your requested speech preferences remain saved, core ad removal continues, and existing published audio is preserved. No paid speech service is enabled automatically. You can leave speech off, or use a compatible self-hosted speech endpoint if you want synthesis on your own infrastructure. PAR no longer provides a bundled offline TTS engine.

### Other changes that can affect your setup

| Change | Why and what to review |
| --- | --- |
| Complete Timeline replaces Legacy whitelist/blacklist processing | One classification path separates what the model identifies from what you choose to cut. Queued/retryable Legacy jobs are converted; review removal categories and show-specific guidance. Published episodes are not automatically reprocessed. |
| Native structured output is required | Invalid classifications must not become apparently successful ad-free episodes. Models/endpoints without schema support are unsupported; valid JSON may be unwrapped, but malformed output is not repaired or retried without a schema. |
| SponsorBlock is removed | Removes another source of cut decisions alongside the single timeline workflow. Public YouTube channels and playlists remain supported. |
| Global free-form instructions and direct editorial non-speech removal are retired | Keep classification definitions and removal choices consistent. Category definitions remain editable and nonempty show-specific guidance still applies; review instructions that previously told the model what to cut. |
| One cut-tone switch replaces three position switches | Simplifies controls. If any old beginning/middle/end switch was on, tones are enabled at all applicable cut positions after migration. |
| Short retained islands between cuts use a fixed 10-second threshold | Uses one processing policy for new jobs. Old saved thresholds are ignored; existing queued snapshots retain their frozen choices. This may affect edits when you next process or reprocess an episode. |
| Unified-feed description becomes fixed | Simplifies feed presentation settings. Feed name, episode-title prefix and artwork remain configurable. |
| Retired API settings are rejected | `/api/v1` remains, but scripts sending removed settings or unknown PATCH fields need updating against the installed OpenAPI schema. |
| Individual feed titles and YouTube download windows change | The title suffix defaults to `(ad free)` instead of `(Ad-Free)` and is configurable. YouTube follows the same latest-episode window as RSS, replacing its separate five-video initial cap. Review retention/window choices; old queued work is not cancelled. |

Users, memberships, existing ownership, feed identities and published media are preserved by the migration. Old Piper voice files are retained. Normal retention and subsequent processing still change media. Before upgrading, drain running jobs and keep a matching database, media and previous-image recovery point; **an image-only downgrade to 1.x is unsupported after V2 writes**.

Read the [upgrade checklist](Documentation/V2_UPGRADE.md) and [draft release notes](Documentation/V2_RELEASE_NOTES.md) before switching images. Separate audio storage remains optional. Mandatory podcast ownership and a transcription-engine replacement are deferred.

## Install

During the preview, use the **local setup wizard below**, or follow [Docker deployment](Documentation/Deployment.md). The wizard generates Docker Compose or Docker run files for Linux/macOS or PowerShell, entirely in your browser. Credentials are not uploaded or saved in browser storage.

**Hosted wizard availability, checked 11 October 2026:** the [Pages homepage](https://jdcb4.github.io/podcast-ad-remover/) still shows an old LLM evaluation report and the Dev wizard returns 404 because deployment is blocked. The [installer guide](Documentation/INSTALL_WIZARD.md) covers local use and publication requirements.

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

- [Documentation index](Documentation/PROJECT_INDEX.md) · [2.0 promotion review](Documentation/V2_PROMOTION_REVIEW.md)
- [Complete Timeline](Documentation/COMPLETE_TIMELINE.md) · [Podcast/archive operations](Documentation/PODCAST_OPERATIONS.md)
- [API](Documentation/API.md) · [Architecture](Documentation/Architecture.md) · [Recovery](Documentation/RECOVERY.md)
- [Changelog](Documentation/CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Verification](Documentation/VERIFICATION.md)

Python 3.11, FastAPI/Jinja, SQLite, FFmpeg and Tailwind. Work integrates into `dev`; `main` is production. Run `npm ci` then `npm run verify`; release qualification also requires `npm run verify:docker`. [Versioning](Documentation/VERSIONING.md) covers explicit release approval and image publication.

[MIT license](LICENSE)
