# Environment Variables

Environment variables provide startup defaults; many runtime settings are stored in SQLite and configured in the Admin UI.

## AI Provider Keys (Optional)
Configure the selected provider before processing. Cloud providers need their own API key; a custom local OpenAI-compatible endpoint can be keyless. The dashboard and library do not require an AI key to start. Keys can come from environment variables or the Admin UI.

**Note:** Settings in the **Admin UI** take priority over Environment Variables.
Gemini direct access uses Google's OpenAI-compatible endpoint through the OpenAI Python SDK.

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API Key |
| `OPENAI_API_KEY` | OpenAI API Key |
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `OPENROUTER_API_KEY` | OpenRouter API Key |

Custom OpenAI-compatible endpoint settings are database-backed and configured under **Admin > AI
Settings > Text Analysis**. They are intentionally not aliases for `OPENAI_API_KEY`, so a saved OpenAI
cloud credential is never forwarded to a custom endpoint.

## Configured Gemini defaults and provider quotas

The default text-analysis provider is Gemini. Each provider tries the following models in order:

### Gemini

1. `gemini-3.8-flash`
2. `gemini-3.7-flash`
3. `gemini-3.6-flash`
4. `gemini-3.5-flash`
5. `gemini-3-flash-preview`
6. `gemini-3.5-flash-lite`
7. `gemini-3.1-flash-lite`

### OpenAI

1. `gpt-6-astra`
2. `gpt-5.6-sol`
3. `gpt-5.6-terra`
4. `gpt-5.6-luna`

### Anthropic

1. `claude-fable-5-1`
2. `claude-opus-5`
3. `claude-sonnet-5`
4. `claude-haiku-4-5-20251001`

### OpenRouter

1. `openai/gpt-5.6-terra`
2. `openai/gpt-5.6-luna`
3. `anthropic/claude-sonnet-5`
4. `anthropic/claude-haiku-4.5`
5. `google/gemini-3.8-flash`
6. `google/gemini-3.5-flash-lite`
7. `tencent/hy4-preview`
8. `tencent/hy3`
9. `z-ai/glm-5.3-flash`
10. `z-ai/glm-5.3`
11. `deepseek/deepseek-v4.1-flash`
12. `deepseek/deepseek-v4-pro`

Custom endpoints have no default model. These are configured application defaults, not a guarantee of provider availability. Saved custom lists take precedence. Migration `20260913_0017_model_defaults` updates only recognized older shipped defaults, including equivalent compact JSON lists. Gemini TTS retains its existing cascade.

Quotas depend on model, project and billing tier. Check your active project limits in
[Google AI Studio via the rate-limit guide](https://ai.google.dev/gemini-api/docs/rate-limits).
The app does not assume a fixed free-tier allowance.

Gemini TTS is optional and uses the same saved Gemini API keys. The default TTS cascade is:

1. `gemini-3.1-flash-tts-preview`
2. `gemini-2.5-flash-preview-tts`

Available Gemini TTS voices are `Orus` (default), `Enceladus`, and `Laomedeia`.

Speech generation shares the configured provider request budget. Check active speech quotas
in the same provider console before enabling it.

## Optional / Defaults

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory for internal data (DB, temp) | `/data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SESSION_SECRET_KEY` | Session signing key. Set a unique value before enabling dashboard or feed authentication. | `super-secret-session-key-change-me` |
| `PROCESSOR_ENABLED` | Start the background feed polling and episode-processing process. Set `false` for an isolated web-only clone that must still run startup and database migrations. | `true` |
| `SPONSORBLOCK_ENABLED` | Read SponsorBlock timestamps for YouTube episodes and merge them with LLM detections. Disabled by default; review the SponsorBlock CC BY-NC-SA 4.0 API/data licence before enabling. | `false` |
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

In Docker, set `BASE_URL` or the System Settings public application URL to a host/LAN URL that podcast clients can reach. Fresh Docker installs no longer auto-save the container's internal IP address.

## Runtime Settings Stored In The Database

These are configured from the Admin UI rather than environment variables:

Podcast defaults for processing workflow, content removal, retention, default features, and custom instructions are also
stored in `app_settings`. They are resolved at read time for subscriptions whose corresponding
inheritance toggle is enabled. `default_watermark_artwork` is off by default.

| Setting | Description | Default |
|---------|-------------|---------|
| `default_processing_workflow` | Workflow for inheriting/new podcasts: `legacy` or `complete_timeline`. Existing podcasts remain pinned to Legacy on migration. | `legacy` |
| `default_remove_editorial_non_speech` | Complete Timeline: remove contextual music examples/illustrative audio. | `0` |
| `default_remove_non_editorial_non_speech` | Complete Timeline: remove contextual ad jingles/non-editorial gaps. | `1` |
| `default_minimum_retained_seconds` | Complete Timeline: bridge retained islands strictly shorter than this between two selected cuts; `0` disables. Range 0–600. | `10` |
| `timeline_definitions` | JSON object containing category definition overrides; blank/omitted categories use defaults. Edit in AI Prompt Rules. | empty |
| `timeline_summary_instructions` | Combined summary instructions, separate from the saved Legacy template. | built-in 2–3 sentence rules |
| `timeline_output_mode` | `auto`, `strict` (require schema), or `json` (validated compatibility). Provider cascade stays within the selected provider. | `auto` |
| `whisper_cpu_threads` | Faster-Whisper CPU thread cap. `0` uses the library default. | `0` |
| `ffmpeg_threads` | FFmpeg thread cap. `0` lets FFmpeg choose automatically. | `0` |
| `unload_whisper_after_job` | Unload the local Whisper model after the queue empties to reduce idle RAM. | `0` |
| `ai_api_enabled` | Enable the token-protected AI-facing REST API under `/api/v1`. | `0` |
| `ai_api_default_requests_per_minute` | Default per-token minute limit for authenticated AI API requests. | `60` |
| `ai_api_default_requests_per_day` | Default per-token daily limit for authenticated AI API requests. | `1000` |
| `ai_api_unauth_requests_per_minute` | Per-IP minute limit for missing or invalid AI API token attempts. | `10` |
| `tts_provider` | TTS engine for spoken title intros and audio summaries: `piper` or `gemini`. | `piper` |
| `gemini_tts_voice` | Gemini TTS voice when `tts_provider=gemini`. | `Orus` |
| `gemini_tts_model_cascade` | JSON array of Gemini TTS models to try in order. | `["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]` |
| `custom_llm_base_url` | Explicit HTTP(S) base URL for an opt-in OpenAI-compatible endpoint. | empty |
| `custom_llm_api_key` | Optional credential used only for the custom endpoint. Keyless endpoints are supported. | empty |
| `custom_llm_model` | JSON array of arbitrary model slugs for the custom endpoint. | `[]` |
| `notifications_enabled` | Enable Apprise-backed admin notifications. | `0` |
| `notification_urls` | Newline-separated Apprise URLs. Treat values as secrets because they can contain tokens or webhooks. | empty |
| `notify_access_requests` | Send notification when a user requests dashboard access. | `1` |
| `notify_new_podcasts` | Send notification when a new global podcast is added. | `1` |
| `notify_episode_downloads` | Send notification when an episode finishes processing and is available in feeds. | `1` |
| `notify_breaking_errors` | Send notification for max-retry processing failures and top-level worker errors. | `1` |

Compose interpolates session/provider values from `.env` or the shell. A persistent random
`SESSION_SECRET_KEY` is required by Compose; known example placeholders cannot enable authentication.
Generate it once with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

`WHISPER_MODEL` seeds the database on first startup. Existing database settings override
the environment; change the model in Settings for an existing installation.

`MAX_PROVIDER_CALLS_PER_JOB` defaults to 12 (1–100), shared across automatic retries,
analysis, schema repair, summaries and remote speech. SDK automatic retries are disabled.
`PROVIDER_TIMEOUT_SECONDS` defaults to 120 (5–600). An operator-triggered new job gets a
new budget. Authentication/billing failures and exhausted budgets require intervention.

## Removal warning tones

Global Subscription Settings provides independent on/off switches and sound choices for start,
middle and end removals. All switches default off; the initial sound choice is Soft chime.
`warning_tone_start`, `warning_tone_middle`, and `warning_tone_end` control insertion;
`warning_tone_start_style`, `warning_tone_middle_style`, and `warning_tone_end_style` select
`soft`, `warm`, `clear`, `bell`, `sonar`, or `wooden` independently.
These global audio settings apply to all podcasts when processing begins, including reprocessing.
They do not alter existing published files automatically. See [Warning tones](WARNING_TONES.md).
