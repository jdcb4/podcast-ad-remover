# Podcast Ad Remover

### Resource limits

Fresh installs use three Whisper CPU threads; upgrades preserve your saved value.
This limits transcription threads, not the whole container, and does not reserve
a CPU core for other applications. Consider setting Docker CPU and memory limits
according to how much resource you want peak transcription to use. Leave enough
RAM for your chosen Whisper model, concurrent work and the web interface; a memory
limit that is too low can cause Docker to kill processing. Smaller models and
supported int8 precision can reduce demand. See [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/).

Self-hosted podcast processing that removes selected content and publishes replacement RSS feeds for your usual podcast player.

PAR downloads episodes, transcribes them locally with **faster-whisper**, asks a structured-output model to classify the complete timeline, and uses FFmpeg to remove the categories you choose. Optional API speech can add a generated title or summary.

> **V2 development:** this README describes the V2 implementation on `dev`. A production 2.0 release has not been authorized or published by this work. Do not assume the Docker `latest` tag includes these changes. Read the [upgrade guide](Documentation/V2_UPGRADE.md) before testing an existing installation.

## Why V2 includes breaking changes

We are deliberately making breaking changes in V2. After working on PAR for a while, we have a clearer idea of which features earn their place. Some legacy options no longer seem to offer lasting value, some make the tool harder to understand and maintain, and some are less useful to us now that remote inference has become more affordable.

We hope these removals will not affect anyone's day-to-day use, but we cannot assume nobody relies on them. This is a deliberate cleanup, with the changes and migration consequences documented below. Preserving podcasts, media and feed identities remains a priority; preserving every old operating mode does not.

Local transcription stays. A custom, locally hosted analysis endpoint is still supported when it accepts native structured-output requests. Remote speech is optional, and an upgrade never silently moves a former local-speech installation onto a paid service.

See the [draft release notes and breaking-change list](Documentation/V2_RELEASE_NOTES.md) and [migration checklist](Documentation/V2_UPGRADE.md).

## What PAR does

- **Manage a shared library:** search for podcasts, add RSS feeds or public YouTube channels/playlists, and keep a personal My Podcasts list. Existing ownership rules remain unchanged.
- **Import subscriptions:** upload Pocket Casts or other OPML exports, upload a text file, or paste one feed URL per line. Review duplicates, select entries, and retry failures. Existing library shows are reused without changing their settings or owner.
- **Choose what to remove:** ads, promos, intros, outros and non-editorial non-speech. Complete Timeline classification keeps ordinary content and uncertain material; an advanced short-island rule controls very short retained gaps between cuts.
- **Listen in your preferred app:** individual podcast feeds and a Unified Feed, with one RSS subscription action beside search on My Podcasts and Library.
- **Add optional enhancements:** rewritten descriptions, generated spoken summaries/titles, an ad-free artwork badge, and a tone at content cuts. Fresh installs enable the artwork badge and cut tones; the other enhancements start off.
- **Use a compact interface:** desktop sidebar, simple mobile podcast rows, consistent settings, light/dark mode, and account/access/token administration together under Users & access.
- **Set up gradually:** an optional first-install wizard, rerunnable from System, and a browser-only Docker/Compose configurator with offline download.
- **Track processing:** durable jobs, retries, queue status, last-good publication preservation, retention controls, logs and optional Apprise notifications.
- **Connect an agent:** an optional scoped REST API plus a [portable agent skill](Documentation/Agent_Skill.md).

Classification and transcription can make mistakes. Review reports and listen to edited seams when tuning removal preferences. PAR's cut boundaries depend on transcript timing; they are not guaranteed word-perfect.

## Get started

### Install configuration

The [static configurator source](configurator/) generates Compose or Docker run files for POSIX shells or PowerShell. Released configurators are published to separate stable/dev GitHub Pages paths after authorized image publication. The generated files and credentials stay in your browser; nothing is sent to a server or saved in browser storage.

To use the source offline, run `python scripts/build_configurator.py`, then open `configurator/index.html`. Alternatively, download the offline ZIP from a published configurator. Download both the install file and its separate `install.env` into the same directory. Generated Compose uses raw env-file values and requires Docker Compose 2.30 or newer.

### Test V2 from this checkout

Build the checked-out code locally:

```bash
docker build -t podcast-ad-remover:v2-local .
```

Copy `env.example` to a private `.env`. Generate `SESSION_SECRET_KEY` once, save it in that file, and set `BASE_URL` to an address your podcast clients can reach:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
docker run -d --name podcast-ad-remover-v2 -p 8000:8000 --mount source=podcast-ad-remover-v2-data,target=/data --env-file .env podcast-ad-remover:v2-local
```

This example uses a **new test volume**. Never mount the same writable data directory into two running PAR instances. For an existing installation, rehearse the upgrade against a backup and follow [V2_UPGRADE.md](Documentation/V2_UPGRADE.md).

Open `http://localhost:8000`. Use or dismiss the setup wizard, configure **Settings → Text analysis**, then add or import podcasts. Provider keys are optional at container startup; analysis must be configured before episodes can be processed. Speech can be configured later in **Settings → Voice**.

Keep the data volume and session secret across upgrades. See [Deployment](Documentation/Deployment.md), [Unraid](Documentation/Unraid_Deployment.md), and [environment/runtime configuration](Documentation/Environment_Variables.md) for production-style examples, networking, GPU access and configuration precedence.

### Image channels

| Image tag | Purpose |
|-----------|---------|
| `jdcb4/podcast-ad-remover:dev-<git-sha>` | An exact published Dev candidate; record the revision you test. |
| `jdcb4/podcast-ad-remover:dev` | The most recently published Dev build, not necessarily the latest source commit. |
| `jdcb4/podcast-ad-remover:<version>` | An explicitly published production release. |
| `jdcb4/podcast-ad-remover:latest` | The production channel; never updated by a Dev publish. |

A local build or Git commit does not publish any of these tags. Production promotion remains a separate, explicit maintainer decision.

## Models and speech

| Task | Supported configuration |
|------|-------------------------|
| Transcription | Local faster-whisper, CPU by default; [optional experimental NVIDIA GPU setup](Documentation/CUDA.md). |
| Text analysis | Gemini, OpenAI, Anthropic, OpenRouter, or a custom OpenAI-compatible endpoint with native structured outputs. |
| Optional speech | Gemini, OpenAI, OpenRouter, or a custom OpenAI-compatible speech endpoint. |

Choose **one provider and model per task**, and one voice for speech. There is no automatic model cascade, provider fallback or key rotation. Retries and quota waiting on the same configured endpoint are distinct from fallback.

Environment keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) take precedence over saved credentials. Custom endpoints use their own credentials. Voice has model/voice lists, metadata refresh and manual IDs where a provider does not expose a complete catalogue. Availability depends on the provider and account; refreshing metadata does not generate speech.

Local Piper speech is removed. Former Piper installations continue core processing without optional speech, retain their requested speech preferences and show configuration guidance. API charges may apply if you subsequently configure speech. See [configuration details](Documentation/Environment_Variables.md).

## Feeds, access and privacy

Dashboard login, public read-only `/subscribe`, feed/audio protection and the agent API are separate choices. Configure users, access requests, feed tokens and API tokens in **Settings → Users & access**. The API is off by default.

A protected feed URL is a bearer secret. Keep it private and revoke its token if exposed. Keep provider keys, notification URLs, environment files and database backups private too. Transcription stays local, but the selected analysis provider receives transcript/episode context, and a speech provider receives text requested for synthesis. A self-hosted custom endpoint can keep analysis on your own infrastructure.

YouTube support is limited to public channels and explicit playlists. It does not support individual video subscriptions, Shorts/streams as dedicated sources, private/member content or login cookies. SponsorBlock is removed; yt-dlp and Deno remain. Use sources you are permitted to download and process.

## Separate audio storage

Optionally mount finished audio on a NAS or media disk with `MEDIA_DIR`. Keep appdata and temporary processing on local storage. **Settings → System → Storage** checks the mount, previews existing audio, runs a resumable verified migration and offers separate original-copy cleanup. Existing installs can keep their layout; public episode URLs stay unchanged. See [storage and migration](Documentation/STORAGE.md).

## Upgrade from 1.x

The main removals are local Piper speech, model/key cascades, Legacy whitelist/blacklist processing, schema-free analysis, SponsorBlock, global free-form detection instructions, editable unified-feed descriptions and separate cut-position switches.

The migration creates a database backup, converts queued jobs to Complete Timeline, chooses the first configured model/credential and records an upgrade report in System. An enabled old cut-position switch enables the new switch for all applicable positions. Existing published audio is not automatically reprocessed. Podcast-specific guidance remains available and always applies when nonempty.

**Drain running jobs before upgrading. Rollback requires the previous immutable image plus the pre-upgrade database and matching media recovery point; downgrading only the image is not supported.** Read [the full upgrade guide](Documentation/V2_UPGRADE.md), including the action table and preservation limits.

Mandatory podcast ownership and any switch to whisper.cpp are deferred. V2 keeps the current ownership behavior and faster-whisper engine.

## Documentation

- [Project and documentation index](Documentation/PROJECT_INDEX.md)
- [V2 release notes](Documentation/V2_RELEASE_NOTES.md) · [Upgrade and rollback](Documentation/V2_UPGRADE.md)
- [Configuration](Documentation/Environment_Variables.md) · [Deployment](Documentation/Deployment.md)
- [Complete Timeline rules](Documentation/COMPLETE_TIMELINE.md) · [Cut tones](Documentation/WARNING_TONES.md)
- [API reference](Documentation/API.md) · [Agent skill](Documentation/Agent_Skill.md)
- [Architecture](Documentation/Architecture.md) · [Data flow](Documentation/Data_Flow.md)
- [Security](SECURITY.md) · [Recovery](Documentation/RECOVERY.md)
- [Changelog](Documentation/CHANGELOG.md) · [Decisions](Documentation/DECISIONS.md) · [Roadmap](Documentation/ROADMAP.md)

Older screenshots, benchmarks and dated release records describe the versions they captured. They are historical evidence, not screenshots or qualification claims for the current V2 interface.

## Development

Python 3.11, FastAPI/Jinja, SQLite, FFmpeg and Tailwind. Normal integration happens on `dev`; `main` is production.

```bash
npm ci
npm run verify
npm run verify:docker
```

See [Contributing](CONTRIBUTING.md), [Verification](Documentation/VERIFICATION.md), [Git workflow](Documentation/GIT_WORKFLOW.md) and [Versioning](Documentation/VERSIONING.md). Publishing images or promoting a release requires explicit authorization.

## License

[MIT](LICENSE)
