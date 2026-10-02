# Environment and runtime configuration (v2)

## Installation variables

The image listens on port 8000 and stores its database and media under `/data`. Preserve the volume and session secret across upgrades. `configurator/` provides a browser-only Docker/Compose builder; it sends no inputs to a server. The optional in-app setup wizard covers the application URL, text analysis, transcription and default removal/retention choices, and can be dismissed or rerun from System.

Provider keys are optional at startup. Configure analysis before processing. Environment credentials take precedence over a saved key. Exactly one credential and model are used per provider; no key rotation or model fallback occurs. Custom endpoints never inherit cloud credentials.

| Environment variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini text analysis and speech |
| `OPENAI_API_KEY` | OpenAI text analysis and speech |
| `ANTHROPIC_API_KEY` | Anthropic text analysis |
| `OPENROUTER_API_KEY` | OpenRouter text analysis and speech |

If an old Gemini environment variable contains comma-separated keys, only its first nonempty key is used. Remove the unused keys from the installation configuration.

## Separate audio storage

`MEDIA_DIR` is an optional container path for processed audio. Leave it unset for the existing layout; configure a mount and enable it in System → Storage before new processing. It must not overlap the existing podcast tree. [Storage guide](STORAGE.md) covers migration, outages and backups. Scratch files and models stay under `DATA_DIR`.

## Optional / Defaults

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory for internal data (DB, temp) | `/data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SESSION_SECRET_KEY` | Session signing key. Set a unique value before enabling dashboard or feed authentication. | `super-secret-session-key-change-me` |
| `PROCESSOR_ENABLED` | Start the background feed polling and episode-processing process. Set `false` for an isolated web-only clone that must still run startup and database migrations. | `true` |
| `CHECK_INTERVAL_MINUTES` | How often to check for new episodes | `60` |
| `WHISPER_MODEL` | First-startup Whisper model seed; existing database settings take precedence | `base` |
| `HOST` | Legacy startup setting; Docker binding is controlled by its Uvicorn command | `0.0.0.0` |
| `PORT` | URL auto-detection hint; Docker still listens on 8000 unless its command changes | `8000` |
| `BASE_URL` | Public URL for the RSS feeds | `http://localhost:8000` |
| `COOKIE_SECURE` | Set session cookies as HTTPS-only. Use `true` behind HTTPS. | `false` |
| `TRUST_PROXY_HEADERS` | Trust `CF-Connecting-IP`, `X-Forwarded-For`, and `X-Real-IP` for login rate limits, IP allowlists, and listen tracking. Only enable behind a reverse proxy that strips client-supplied copies of these headers. | `false` |
| `MAX_FEED_BYTES` | Maximum RSS feed fetch size in bytes. | `10485760` |
| `MAX_DOWNLOAD_BYTES` | Maximum episode download size in bytes. | `1572864000` |
| `MIN_FREE_SPACE_BYTES` | Minimum free disk space to preserve before/during downloads. | `1073741824` |
| `FFMPEG_TIMEOUT_SECONDS` | Maximum time allowed for one FFmpeg or FFprobe operation before the episode fails/retries instead of holding the queue indefinitely. | `7200` |
| `ALLOW_PRIVATE_FEEDS` | Allow feeds/enclosures resolving to private or loopback IP ranges. Keep `true` for LAN/self-hosted feeds; set `false` for hardened public deployments. | `true` |
| `CUDA_SETUP` | Request first-start optional NVIDIA runtime setup. Host drivers and container GPU access must already work. | `false` |
| `MAX_PROVIDER_CALLS_PER_JOB` | Shared analysis/speech request budget per processing attempt | `12` |
| `PROVIDER_TIMEOUT_SECONDS` | Per-provider request timeout, seconds | `120` |
| `LOG_MAX_BYTES` | Rotating log size | `10485760` |
| `LOG_BACKUP_COUNT` | Rotating log backups | `5` |
| `ENVIRONMENT` | Runtime environment label | `production` |


In Docker, set `BASE_URL` or the System Settings public application URL to a host/LAN URL that podcast clients can reach. Fresh Docker installs no longer auto-save the container's internal IP address.


## Configure later in Settings

- **Transcription:** faster-whisper model, CPU/GPU device and supported precision; GPU setup/test. CPU/float32 remains the default. See [CUDA.md](CUDA.md).
- **Text analysis:** provider, one model, credential and optional custom base URL. Model suggestions are choices, never a cascade. Structured schema requests are mandatory. A complete JSON object may be unwrapped from prose or a code fence, but invalid JSON is not repaired.
- **Voice:** Gemini, OpenAI, OpenRouter or a custom OpenAI-compatible speech endpoint; one model and voice. Cloud providers share their saved/environment credential with text analysis. Custom speech has its own endpoint/key. Speech is optional and may incur API charges. Piper installations need explicit reconfiguration; core processing continues without optional speech.
- **Podcast defaults:** ads, promos, intros, outros, non-editorial non-speech, optional summaries/title intros/artwork badge, one cut tone switch, retention, advanced short-island threshold (10 seconds; zero disables).
- **Unified feed:** name, podcast-name prefix, and bundled/external/uploaded artwork. Description is fixed. Raster uploads support PNG/JPEG/WebP up to 5 MB and 4096 pixels, re-encoded as JPEG.
- **Prompt rules:** category definitions and summary instructions; defaults can be restored per rule. Podcast-specific guidance is always applied when nonempty under advanced podcast settings.
- **System:** concurrent downloads, check interval, retention, thread limits (three Whisper CPU threads on fresh installs; existing values preserved) and download redirect limit (0–50; zero disallows redirects).
- **Users & access:** dashboard/feed authentication, IP allowlist, users, access requests, feed tokens and API tokens/limits. Existing ownership behavior is unchanged.
- **Notifications:** Apprise destinations and event toggles. **Logs:** existing viewer.

These Settings pages are database-backed. Variables such as `WHISPER_MODEL` seed new database settings; changing them does not overwrite an existing saved choice. Runtime environment settings such as `PROCESSOR_ENABLED`, network trust and hard download limits remain installation-level controls. Provider credentials explicitly use environment precedence. Do not assume every environment value either seeds once or overrides every saved setting. The retired `SPONSORBLOCK_ENABLED`, Piper settings and `INSTALL_TTS` build argument are unsupported. YouTube RSS extraction still uses yt-dlp and Deno.

## Speech catalog discovery

Voice provides model/voice dropdowns, manual IDs and an explicit Refresh models & voices action. Refresh only reads metadata; it never synthesizes speech or saves the form. Entered keys override saved keys for that request, while cloud environment keys retain precedence. Errors preserve selections.

Gemini pages through its models and prebuilt voices, retaining the 30 documented studio voices if discovery is unavailable. OpenAI discovers TTS models and uses documented built-in voices, filtered by model. OpenRouter discovers speech-output models but has no universal voice catalog: documented OpenAI/Gemini voices are listed for those model families, and other models require a provider voice ID. Custom endpoints use models and, if implemented, audio/voices discovery. Voice creation and cloning are outside this feature.

References: [Gemini voices](https://ai.google.dev/api/voices), [OpenAI speech](https://developers.openai.com/api/docs/guides/text-to-speech), [OpenRouter speech discovery](https://openrouter.ai/docs/guides/overview/multimodal/tts).

New databases enable Ad-free artwork and Insert tone at content cuts; the other enhancements start off. Existing saved preferences are unchanged. Unavailable spoken controls retain their requested values when other settings are saved.

## Migration

[The V2 upgrade guide](V2_UPGRADE.md) describes the pre-upgrade backup, queued-job conversion and per-install migration report. Do not downgrade the executable alone after v2 writes. Restore a matching pre-upgrade database and previous immutable image; preserve the media recovery point.

## Retention and inheritance

Content removal, retention and enhancements can inherit global defaults per podcast. Podcast-specific guidance is always applied when nonempty; there is no global guidance-inclusion switch or useful workflow selection. Retired inheritance/database fields may still appear in API compatibility models.

Automatic episode retention is count-based (`retention_limit`); zero retains no automatically processed episodes and prevents automatic downloads. Manual downloads use `manual_retention_days`; zero makes completed manual output immediately eligible for cleanup. `retention_days` remains a stored compatibility setting but does not drive the current automatic cleanup path. Do not treat a zero retention value as “keep forever.”

## Gemini quota handling

Settings → Text analysis → Quota handling optionally enables PAR's shared Gemini free-tier accounting. Counters and reservations live in SQLite across jobs, and daily resets follow Pacific time. It applies to the one selected model; an exhausted model waits or produces an actionable quota state instead of moving through a cascade. Model limits are the application's configured estimates, not an authoritative statement about your account's current allowance. Unknown models and provider-reported limits must be handled according to the actual account response. See `app/core/gemini_quota.py` and the queue usage display; provider-call budgets and per-request timeouts still apply independently.

## Import, installer and agent settings

Import has no new Docker variables: use Add podcast → Import or the API's dry-run endpoint. The browser form accepts OPML/text uploads and pasted feeds. Configure API access and limits under Users & access; the [API reference](API.md) documents request limits and [Agent_Skill.md](Agent_Skill.md) explains the portable skill. The setup wizard intentionally omits advanced settings, and import does not overwrite existing podcast settings or ownership.
