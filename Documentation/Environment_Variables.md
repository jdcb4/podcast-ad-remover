# Environment Variables

The application is configured via environment variables.

## AI Provider Keys (Optional)
The application requires at least one API key to function (Gemini, OpenAI, Anthropic, or OpenRouter). You can set these via Environment Variables (recommended for Docker) or via the Admin UI.

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

## Gemini Model Defaults And Free-Tier Limits

The default direct Gemini cascade is:

1. `gemini-3.5-flash`
2. `gemini-3-flash`
3. `gemini-3.1-flash-lite`
4. `gemini-2.5-flash`
5. `gemini-2.5-flash-lite`

The OpenRouter Gemini cascade uses the same order with the `google/` prefix. The app tries configured models in order and moves to the next model if a request fails or is rate-limited.

Current Gemini free-tier limits recorded for these defaults:

| Model | Category | RPM | TPM | RPD |
|-------|----------|-----|-----|-----|
| Gemini 2.5 Flash | Text-out models | 5 | 250K | 20 |
| Gemini 3 Flash | Text-out models | 5 | 250K | 20 |
| Gemini 2.5 Flash Lite | Text-out models | 10 | 250K | 20 |
| Gemini 3.1 Flash Lite | Text-out models | 15 | 250K | 500 |
| Gemini 3.5 Flash | Text-out models | 5 | 250K | 20 |

Gemini TTS is optional and uses the same saved Gemini API keys. The default TTS cascade is:

1. `gemini-3.1-flash-tts-preview`
2. `gemini-2.5-flash-preview-tts`

Available Gemini TTS voices are `Orus` (default), `Enceladus`, and `Laomedeia`.

Current free-tier limits recorded for Gemini TTS:

| Model | Category | RPM | TPM | RPD |
|-------|----------|-----|-----|-----|
| Gemini 3.1 Flash TTS | Text-to-speech models | 3 | 10K | 10 |
| Gemini 2.5 Flash TTS | Text-to-speech models | 3 | 10K | 10 |

## Optional / Defaults

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory for internal data (DB, temp) | `/data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SESSION_SECRET_KEY` | Session signing key. Set a unique value before enabling dashboard or feed authentication. | `super-secret-session-key-change-me` |
| `CHECK_INTERVAL_MINUTES` | How often to check for new episodes | `60` |
| `WHISPER_MODEL` | Whisper model size | `base` |
| `HOST` | Host to bind to | `0.0.0.0` |
| `PORT` | Port to bind to | `8000` |
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

| Setting | Description | Default |
|---------|-------------|---------|
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
| `ad_chunking_enabled` | Enable context-budget chunking for ad detection. Cloud/default behavior remains unchunked while off. | `0` |
| `ad_chunk_context_tokens` | Declared context window used by the conservative prompt planner. | `8192` |
| `ad_chunk_overlap_seconds` | Maximum timestamp overlap requested around chunk boundaries. | `30` |
| `ad_chunk_max_chunks` | Safety cap on requests for one detection attempt. | `32` |
| `ad_include_reasons` | Ask the model for a brief explanation per detected segment. | `1` |
| `notifications_enabled` | Enable Apprise-backed admin notifications. | `0` |
| `notification_urls` | Newline-separated Apprise URLs. Treat values as secrets because they can contain tokens or webhooks. | empty |
| `notify_access_requests` | Send notification when a user requests dashboard access. | `1` |
| `notify_new_podcasts` | Send notification when a new global podcast is added. | `1` |
| `notify_episode_downloads` | Send notification when an episode finishes processing and is available in feeds. | `1` |
| `notify_breaking_errors` | Send notification for max-retry processing failures and top-level worker errors. | `1` |

When `ad_chunking_enabled=1`, AI description rewrites and audio summaries are disabled because their
current implementation requires one full-transcript request. Saved podcast preferences are retained.
