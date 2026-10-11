# Podcast Ad Remover

**Less skipping. More listening. Your usual podcast player.**

Podcast Ad Remover is a self-hosted app that downloads podcast episodes, removes ads and other content you choose, and publishes replacement RSS feeds. Subscribe in your usual podcast player and let PAR handle new episodes.

Transcription runs locally with faster-whisper. Your chosen AI model classifies the episode timeline, and PAR cuts the categories you want to skip. Choose your model, removal preferences and where your library lives.

> This README describes V2 on `dev`; the Docker `latest` tag remains the production release channel. Existing users should read the [V2 release notes](Documentation/V2_RELEASE_NOTES.md) and [upgrade guide](Documentation/V2_UPGRADE.md) before updating.

## What you can do

- **Keep your podcast player.** Subscribe to individual shows, combine My Pods into a personal Unified Feed, or browse the shared Library.
- **Choose what to remove.** Set global or per-show preferences for ads, promos, intros, outros and non-editorial non-speech. Review transcripts and processing reports when you want to inspect an edit.
- **Transcribe faster with NVIDIA CUDA.** A compatible NVIDIA GPU can make local transcription significantly faster than CPU processing. Use **Transcription → Set up / retest GPU acceleration** after granting Docker GPU access. CPU remains supported; CUDA is optional and experimental. See [GPU setup and requirements](Documentation/CUDA.md).
- **Bring your subscriptions.** Search for podcasts, add RSS feeds or public YouTube channels/playlists, and import OPML exports such as Pocket Casts.
- **Make episodes easier to browse.** Add optional descriptions, spoken titles and summaries, artwork badges and cut tones; customize feed names with a prefix or suffix.
- **Control your library.** Set retention, pause individual feeds or all processing, retry episodes and process a show's archive. Finished audio can live on a separate disk or NAS.
- **See what is happening.** Track processing and library statistics, use the desktop/mobile interface and light/dark mode, and optionally enable notifications or the scoped API with a [portable agent skill](Documentation/Agent_Skill.md).

PAR needs Docker, storage for your library, and a suitable analysis model. Remote analysis and optional speech can incur charges. AI classification and cut timing are imperfect; reports and per-show settings help you review the results.

## Install

### Docker

Docker is the recommended installation method. Download [docker-compose.prod.yml](docker-compose.prod.yml) and [env.example](env.example) into the same folder, save `env.example` as `.env`, then set a persistent random `SESSION_SECRET_KEY` and a `BASE_URL` reachable by your podcast player.

```bash
docker compose -f docker-compose.prod.yml up -d
```

This uses the published image and stores application data in `./data`. See [Docker deployment](Documentation/Deployment.md) for Docker run, configuration and reverse-proxy options, or the [Unraid guide](Documentation/Unraid_Deployment.md).

### Build from source

Clone the repository, copy `env.example` to `.env` and set the same session secret and application URL. Build and start the checkout with Docker:

```bash
git clone https://github.com/jdcb4/podcast-ad-remover.git
cd podcast-ad-remover
docker compose up -d --build
```

The source Compose file mounts the checkout into the container for development. The repository's default `dev` branch contains development code; choose a release tag for released source. See [contributor guidance](CONTRIBUTING.md).

### Optional setup wizard

The **[setup wizard](https://jdcb4.github.io/podcast-ad-remover/)** simplifies the Docker installation by generating Compose or Docker run files for your storage, URL and optional NVIDIA GPU. It runs entirely in your browser: credentials are not uploaded or saved in browser storage. Generated Compose files require Compose 2.30 or newer.

You can also **[download the single HTML wizard](https://github.com/jdcb4/podcast-ad-remover/raw/refs/heads/dev/configurator/install.html)** and open it in your browser, including offline. No local server or build tools are needed. Check the image/channel displayed, save the generated install file and `install.env` together, then run the command shown.

After installation, open PAR at your configured URL, complete or dismiss the optional in-app setup, add podcasts and subscribe to their replacement feeds. **For upgrades, preserve your data volume, configuration and session secret** and follow the [upgrade guide](Documentation/V2_UPGRADE.md). See also [configuration](Documentation/Environment_Variables.md) and [separate audio storage](Documentation/STORAGE.md).

## Why V2 simplifies things

We want PAR to focus on preparing better podcast feeds, with fewer overlapping options to configure, test and maintain. V2 makes three substantial changes:

- **Local text-to-speech is removed.** Optional spoken features use a compatible speech API instead of bundled Piper, reducing a separate local synthesis installation and maintenance path. Local transcription stays, and no paid speech service is enabled automatically.
- **The model cascade is removed.** Capable paid models have become affordable enough for many users to select one directly. If you want tiered routing or fallback, dedicated tools such as [9Router](https://github.com/decolua/9router), [OmniRoute](https://github.com/diegosouzapw/OmniRoute) and [LiteLLM](https://docs.litellm.ai/docs/proxy/reliability) can manage it outside PAR through a compatible endpoint. Maintaining another router inside this app adds limited value.
- **Complete Timeline replaces blacklist/whitelist processing.** One classification system identifies the episode's content, while your category preferences determine what gets removed. This separates classification from cutting and gives us one workflow to improve.

These changes can affect existing configurations even though the migration preserves your library and published audio. Read the **[V2 release notes](Documentation/V2_RELEASE_NOTES.md)** for the rationale and upgrade impacts, the [changelog](Documentation/CHANGELOG.md) for the detailed changes, and the [upgrade guide](Documentation/V2_UPGRADE.md) before switching images.

## Models, access and privacy

| Task | Options |
| --- | --- |
| Transcription | Local faster-whisper on CPU or an optional compatible NVIDIA GPU. |
| Analysis | Gemini, OpenAI, Anthropic, OpenRouter or a custom OpenAI-compatible endpoint with native structured outputs. |
| Optional speech | Gemini, OpenAI, OpenRouter or a custom compatible speech endpoint. |

Choose one model and credential per task. Custom analysis endpoints and external routers must support native structured-output requests throughout the route. Analysis receives transcript/episode context; speech receives the text to synthesize. Compatible self-hosted endpoints can keep those tasks on your own infrastructure.

Dashboard login, public subscription browsing, feed/audio protection and API access are separate settings. Protected feed URLs are bearer secrets. Keep them, provider keys and backups private. See [Security](SECURITY.md).

YouTube support covers public channels and explicit playlists. Use sources you are permitted to download and process.

## Documentation and development

- [Documentation index](Documentation/PROJECT_INDEX.md) · [V2 release notes](Documentation/V2_RELEASE_NOTES.md)
- [Complete Timeline](Documentation/COMPLETE_TIMELINE.md) · [Podcast/archive operations](Documentation/PODCAST_OPERATIONS.md) · [Processing controls and statistics](Documentation/PROCESSING_CONTROLS.md)
- [API](Documentation/API.md) · [Architecture](Documentation/Architecture.md) · [Recovery](Documentation/RECOVERY.md)
- [Changelog](Documentation/CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Verification](Documentation/VERIFICATION.md)

Python 3.11, FastAPI/Jinja, SQLite, FFmpeg and Tailwind. Work integrates into `dev`; `main` is production. Run `npm ci` then `npm run verify`; release qualification also requires `npm run verify:docker`. [Versioning](Documentation/VERSIONING.md) covers release approval and image publication.

[MIT license](LICENSE)
