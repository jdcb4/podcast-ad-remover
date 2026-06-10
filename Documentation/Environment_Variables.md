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

## Optional / Defaults

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory for internal data (DB, temp) | `/data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CHECK_INTERVAL_MINUTES` | How often to check for new episodes | `60` |
| `WHISPER_MODEL` | Whisper model size | `base` |
| `HOST` | Host to bind to | `0.0.0.0` |
| `PORT` | Port to bind to | `8000` |
| `BASE_URL` | Public URL for the RSS feeds | `http://localhost:8000` |
| `COOKIE_SECURE` | Set session cookies as HTTPS-only. Use `true` behind HTTPS. | `false` |
| `MAX_FEED_BYTES` | Maximum RSS feed fetch size in bytes. | `10485760` |
| `MAX_DOWNLOAD_BYTES` | Maximum episode download size in bytes. | `1572864000` |
| `MIN_FREE_SPACE_BYTES` | Minimum free disk space to preserve before/during downloads. | `1073741824` |
| `ALLOW_PRIVATE_FEEDS` | Allow feeds/enclosures resolving to private or loopback IP ranges. Keep `true` for LAN/self-hosted feeds; set `false` for hardened public deployments. | `true` |

In Docker, set `BASE_URL` or the System Settings public application URL to a host/LAN URL that podcast clients can reach. Fresh Docker installs no longer auto-save the container's internal IP address.

## Runtime Settings Stored In The Database

These are configured from the Admin UI rather than environment variables:

| Setting | Description | Default |
|---------|-------------|---------|
| `whisper_cpu_threads` | Faster-Whisper CPU thread cap. `0` uses the library default. | `0` |
| `ffmpeg_threads` | FFmpeg thread cap. `0` lets FFmpeg choose automatically. | `0` |
| `unload_whisper_after_job` | Unload the local Whisper model after the queue empties to reduce idle RAM. | `0` |
