# AI API User Guide

Podcast Ad Remover exposes an optional REST API for AI agents, custom GPT actions, scripts, and automation clients under `/api/v1`.

Use this API when you want an assistant to inspect podcasts, find episodes, read transcripts/reports, add feeds, or trigger processing without using the browser UI.

## Quick Start

1. Open **Admin > System Settings > AI API**.
2. Enable **AI API**.
3. Set default rate limits.
4. Create or choose the dashboard user the API token should act as.
5. Create an API token and assign it to that user.
6. Copy the token immediately. The full token is shown only once.

Example request:

```bash
curl -H "Authorization: Bearer par_REPLACE_ME" \
  http://localhost:8000/api/v1/subscriptions
```

## Authentication

Most endpoints require:

```http
Authorization: Bearer par_...
```

API tokens are different from feed tokens:

- API tokens control management and automation actions.
- Feed tokens only allow podcast clients to read protected feeds/audio.
- API token hashes are stored in SQLite; the full token is never stored.
- New API tokens must be linked to a dashboard user. API management actions are treated as that user.

Linked-token access rules:

- Tokens linked to normal users can browse the full podcast library, matching the dashboard's global library view.
- Tokens linked to normal users can change settings or trigger processing only for podcasts owned by that user.
- Tokens linked to admin users can see and manage the global library.
- The `admin` scope only exposes system status when the token is linked to an admin user.
- Legacy unlinked tokens keep the older global behavior until you migrate or revoke them.

Available scopes:

| Scope | Use this when the client should |
|-------|---------------------------------|
| `read` | Read podcasts, episodes, queue state, transcripts, reports, and search results. |
| `write` | Add subscriptions and update podcast settings. |
| `process` | Trigger feed checks, downloads, reprocessing, cancellation, and ignore actions. |
| `admin` | Read instance-level system status. |

For a general AI assistant, a practical token is usually `read`, `write`, and `process`. Add `admin` only when the assistant should inspect instance health.

## Linking Existing API Tokens

Use `scripts/link_api_token_user.py` to migrate an existing token to a dashboard user without recreating the token.

List current API tokens:

```bash
python scripts/link_api_token_user.py --list --db-path /data/db/podcasts.db
```

Link by token id and username:

```bash
python scripts/link_api_token_user.py --token-id 1 --username alice --db-path /data/db/podcasts.db
```

Link by token prefix and user id:

```bash
python scripts/link_api_token_user.py --token-prefix par_abc12345 --user-id 2 --db-path /data/db/podcasts.db
```

In Docker, run the same command inside the app container:

```bash
docker exec -it podcast-ad-remover python scripts/link_api_token_user.py --list
docker exec -it podcast-ad-remover python scripts/link_api_token_user.py --token-id 1 --username alice
```

The script only updates the `user_id` on an active API token. It does not need or reveal the full bearer token.

## Rate Limits

Rate limits are SQLite-backed and survive app restarts.

- Authenticated requests are limited by token.
- Missing or invalid token attempts are limited by client IP.
- `429` responses include `Retry-After`.

Example rate-limit response:

```json
{
  "detail": "Rate limit exceeded"
}
```

When `TRUST_PROXY_HEADERS=true`, client IP detection uses trusted reverse-proxy headers. Only enable that setting behind a proxy that strips client-supplied forwarded headers.

## Common Workflows

### Let an AI answer "what podcasts do I have?"

Call:

```http
GET /api/v1/subscriptions
```

Required scope: `read`

Outcome: returns the global podcast library.

### Let an AI summarize available episodes for a podcast

Call:

```http
GET /api/v1/subscriptions/{subscription_id}/episodes?limit=20
```

Then, for any completed episode that has artifacts:

```http
GET /api/v1/episodes/{episode_id}/transcript
GET /api/v1/episodes/{episode_id}/report
```

Required scope: `read`

Outcome: the assistant can inspect episode metadata, transcript text, and ad-removal report data.

### Add a new podcast from an AI chat

Call:

```http
POST /api/v1/subscriptions
Content-Type: application/json

{
  "feed_url": "https://example.com/podcast/feed.xml",
  "initial_count": 5
}
```

Required scope: `write`

Outcome: the app parses the feed, creates or reuses the global subscription, and checks the newest episodes.

### Ask an AI to refresh a show

Call:

```http
POST /api/v1/subscriptions/{subscription_id}/check
```

Required scope: `process`

Outcome: the app checks the feed for new episodes and queues normal processing work.

### Ask an AI to reprocess an episode

Call:

```http
POST /api/v1/episodes/{episode_id}/reprocess?skip_transcription=false
```

Required scope: `process`

Outcome: the app versions the current output, resets the episode, and queues processing.

Use `skip_transcription=true` only when an existing transcript should be reused.

## Discovery Endpoints

### Check API health

```http
GET /api/v1/health
```

Auth: none

Use this to confirm the app is reachable and whether the API is enabled.

Example response:

```json
{
  "status": "healthy",
  "enabled": true
}
```

### Read API capabilities

```http
GET /api/v1/capabilities
```

Auth: none, but the API must be enabled.

Use this to discover supported scopes and configured default limits.

Example response:

```json
{
  "name": "Podcast Ad Remover API",
  "version": "v1",
  "auth": "bearer",
  "scopes": ["read", "write", "process", "admin"],
  "rate_limits": {
    "default_requests_per_minute": 60,
    "default_requests_per_day": 1000,
    "unauth_requests_per_minute": 10
  }
}
```

### Read the OpenAPI schema

```http
GET /api/v1/openapi.json
```

Auth: none, but the API must be enabled.

Use this for clients that can import an OpenAPI schema.

## Read Endpoints

### Get system status

```http
GET /api/v1/system/status
```

Required scope: `admin`

Use this when an assistant needs operational health such as queue, storage, feed check, or system status summaries.

Example response shape:

```json
{
  "disk": {},
  "memory": {},
  "queue": {},
  "feeds": {}
}
```

The exact keys come from the app's operation status helper and may grow as the admin queue dashboard evolves.

### Get queue status

```http
GET /api/v1/queue
```

Required scope: `read`

Use this to answer "what is processing?", "what failed?", or "what recently completed?"

Example response shape:

```json
{
  "queue": [
    {
      "id": 42,
      "title": "Episode title",
      "status": "processing",
      "job_status": "running",
      "processing_step": "transcribing",
      "progress": 35
    }
  ],
  "recently_processed": [],
  "operation_status": {}
}
```

### List subscriptions

```http
GET /api/v1/subscriptions
```

Required scope: `read`

Use this to list podcasts. Linked user tokens, linked admin tokens, and legacy unlinked tokens all return the global library.

Example response:

```json
[
  {
    "id": 1,
    "feed_url": "https://example.com/feed.xml",
    "title": "Example Show",
    "description": "Podcast description",
    "slug": "example-show",
    "image_url": "https://example.com/art.jpg",
    "is_active": true,
    "created_at": "2026-06-17T10:00:00",
    "last_checked_at": "2026-06-17T11:00:00",
    "remove_ads": true,
    "remove_promos": true,
    "remove_intros": false,
    "remove_outros": false,
    "custom_instructions": null,
    "append_summary": false,
    "append_title_intro": false,
    "ai_rewrite_description": false,
    "ai_audio_summary": false,
    "owner_user_id": null,
    "retention_days": 30,
    "manual_retention_days": 14,
    "retention_limit": 1
  }
]
```

### Get one subscription

```http
GET /api/v1/subscriptions/{subscription_id}
```

Required scope: `read`

Use this when the assistant already knows the subscription id and needs settings or metadata.

Response: one subscription object, same shape as `GET /subscriptions`.

### List episodes for a subscription

```http
GET /api/v1/subscriptions/{subscription_id}/episodes?limit=20&offset=0&search=topic
```

Required scope: `read`

Use this to page through episodes or search episode titles for a podcast.

Query parameters:

| Name | Default | Notes |
|------|---------|-------|
| `limit` | `20` | Clamped to `1..100`. |
| `offset` | `0` | Negative values are treated as `0`. |
| `search` | none | Optional title search. |

Example response:

```json
{
  "episodes": [
    {
      "id": 10,
      "subscription_id": 1,
      "guid": "episode-guid",
      "title": "Episode title",
      "pub_date": "2026-06-17T00:00:00",
      "original_url": "https://example.com/audio.mp3",
      "duration": 3600,
      "status": "completed",
      "processed_at": "2026-06-17T01:00:00",
      "processing_step": "completed",
      "progress": 100,
      "ai_summary": "This episode includes...",
      "file_size": 12345678,
      "listen_count": 2
    }
  ],
  "total": 1,
  "offset": 0,
  "limit": 20,
  "search": "topic",
  "has_more": false
}
```

### Get one episode

```http
GET /api/v1/episodes/{episode_id}
```

Required scope: `read`

Use this for detailed episode metadata, including podcast title and slug.

Example response shape:

```json
{
  "id": 10,
  "subscription_id": 1,
  "podcast_title": "Example Show",
  "subscription_slug": "example-show",
  "title": "Episode title",
  "status": "completed",
  "processing_step": "completed",
  "progress": 100,
  "transcript_path": "/data/podcasts/example-show/episode/transcript.json",
  "report_path": "/data/podcasts/example-show/episode/report.json"
}
```

### Get an episode transcript

```http
GET /api/v1/episodes/{episode_id}/transcript
```

Required scope: `read`

Use this when the assistant needs to answer questions about episode content.

Example response shape:

```json
{
  "episode_id": 10,
  "transcript": {
    "segments": [
      {
        "start": 0.0,
        "end": 4.2,
        "text": "Welcome to the episode."
      }
    ]
  }
}
```

If no transcript file exists, the response is `404`.

### Get an episode ad-removal report

```http
GET /api/v1/episodes/{episode_id}/report
```

Required scope: `read`

Use this to inspect what was removed, kept, or detected during processing.

Example JSON report response:

```json
{
  "episode_id": 10,
  "content_type": "application/json",
  "report": {
    "segments": [
      {
        "start": 120.0,
        "end": 150.0,
        "label": "Ad",
        "reason": "Sponsor read"
      }
    ]
  }
}
```

If only an HTML report exists, `content_type` is `text/html` and `report` is an HTML string.

## Search Endpoint

### Search for podcasts

```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "show name or topic"
}
```

Required scope: `read`

Use this before adding a new podcast when the user provides a show name instead of an RSS URL.

Example response shape:

```json
[
  {
    "title": "Example Show",
    "feed_url": "https://example.com/feed.xml",
    "image_url": "https://example.com/art.jpg",
    "description": "Podcast description"
  }
]
```

## Write Endpoints

### Add a subscription

```http
POST /api/v1/subscriptions
Content-Type: application/json

{
  "feed_url": "https://example.com/feed.xml",
  "initial_count": 5
}
```

Required scope: `write`

Use this when a user asks an assistant to add a podcast by RSS URL.

Request fields:

| Field | Required | Notes |
|-------|----------|-------|
| `feed_url` | yes | Must be HTTP or HTTPS. |
| `initial_count` | no | Number of newest episodes to check initially. Default `5`, allowed `0..50`. |

Outcome:

- If the feed already exists, the existing subscription is returned.
- If the token is linked to a user, the existing show is added to that user's library.
- If it is new, the feed is parsed, saved, a new-podcast notification is sent, and initial feed checking starts.
- Linked normal-user tokens can later manage the subscription only if they own it.

Response: a subscription object.

### Update subscription settings

```http
PATCH /api/v1/subscriptions/{subscription_id}/settings
Content-Type: application/json

{
  "remove_ads": true,
  "remove_promos": true,
  "remove_intros": false,
  "remove_outros": false,
  "custom_instructions": "Also remove local event promos.",
  "append_summary": true,
  "append_title_intro": false,
  "ai_rewrite_description": false,
  "ai_audio_summary": false,
  "retention_days": 30,
  "manual_retention_days": 14,
  "retention_limit": 1
}
```

Required scope: `write`

Use this when a user asks an assistant to change how a podcast is processed.

Linked normal-user tokens can update only podcasts owned by that user. Linked admin tokens and legacy unlinked tokens can update any podcast.

All fields are optional. Omitted fields keep their current values.

Response:

```json
{
  "status": "updated",
  "id": 1,
  "detail": null
}
```

The app also schedules cleanup and a feed check in the background.

## Processing Endpoints

### Check a subscription for new episodes

```http
POST /api/v1/subscriptions/{subscription_id}/check
```

Required scope: `process`

Use this when a user asks "refresh this show" or "check for new episodes."

Linked normal-user tokens can trigger processing only for podcasts owned by that user. The same ownership rule applies to episode download, reprocess, cancel, and ignore actions.

Response:

```json
{
  "status": "check_triggered",
  "id": 1,
  "detail": null
}
```

### Queue a manual episode download

```http
POST /api/v1/episodes/{episode_id}/download
```

Required scope: `process`

Use this when a user wants a specific episode downloaded and processed now.

Outcome: marks the episode as a manual download and queues it by setting status to `pending`.

Response:

```json
{
  "status": "download_queued",
  "id": 10,
  "detail": null
}
```

### Reprocess an episode

```http
POST /api/v1/episodes/{episode_id}/reprocess?skip_transcription=false
```

Required scope: `process`

Use this after changing podcast settings or when an episode should be processed again.

Query parameters:

| Name | Default | Notes |
|------|---------|-------|
| `skip_transcription` | `false` | Set `true` to reuse an existing transcript when available. |

Outcome: versions the existing output, resets processing state, and queues the episode.

Response:

```json
{
  "status": "reprocess_queued",
  "id": 10,
  "detail": null
}
```

If the episode is already processing:

```json
{
  "status": "ignored",
  "id": 10,
  "detail": "already_processing"
}
```

### Cancel queued or running work

```http
POST /api/v1/episodes/{episode_id}/cancel
```

Required scope: `process`

Use this when a user wants to stop queued/running work without deleting or ignoring the episode.

Outcome: active jobs are cancelled and the episode returns to `unprocessed`.

Response:

```json
{
  "status": "cancelled",
  "id": 10,
  "detail": null
}
```

### Ignore an episode

```http
POST /api/v1/episodes/{episode_id}/ignore
```

Required scope: `process`

Use this when a user does not want an episode to appear in processed output.

Outcome: the app runs the same ignore/delete episode cleanup path used by the UI.

Response:

```json
{
  "status": "ignored",
  "id": 10,
  "detail": null
}
```

## Error Responses

Common errors:

| Status | Meaning |
|--------|---------|
| `400` | Invalid input, such as a non-HTTP feed URL. |
| `401` | Missing or invalid bearer token. |
| `403` | Token is valid but lacks the required scope. |
| `404` | API disabled, subscription not found, episode not found, or artifact not found. |
| `429` | Rate limit exceeded. Check `Retry-After`. |

Example:

```json
{
  "detail": "Insufficient API token scope"
}
```

## Endpoint Summary

| Method | Path | Scope | Main use |
|--------|------|-------|----------|
| `GET` | `/api/v1/health` | none | Check app/API reachability. |
| `GET` | `/api/v1/capabilities` | none | Discover scopes and default limits. |
| `GET` | `/api/v1/openapi.json` | none | Import schema into API clients. |
| `GET` | `/api/v1/system/status` | `admin` | Read instance-level health. |
| `GET` | `/api/v1/queue` | `read` | Inspect active/recent processing. |
| `GET` | `/api/v1/subscriptions` | `read` | List podcasts. |
| `GET` | `/api/v1/subscriptions/{id}` | `read` | Read one podcast's metadata/settings. |
| `GET` | `/api/v1/subscriptions/{id}/episodes` | `read` | Page/search a podcast's episodes. |
| `GET` | `/api/v1/episodes/{id}` | `read` | Read one episode's metadata/state. |
| `GET` | `/api/v1/episodes/{id}/transcript` | `read` | Read transcript JSON. |
| `GET` | `/api/v1/episodes/{id}/report` | `read` | Read ad-removal report JSON or HTML. |
| `POST` | `/api/v1/search` | `read` | Search podcast directories. |
| `POST` | `/api/v1/subscriptions` | `write` | Add or attach an RSS subscription. |
| `PATCH` | `/api/v1/subscriptions/{id}/settings` | `write` | Change processing settings. |
| `POST` | `/api/v1/subscriptions/{id}/check` | `process` | Check a feed for new episodes. |
| `POST` | `/api/v1/episodes/{id}/download` | `process` | Queue a manual episode download. |
| `POST` | `/api/v1/episodes/{id}/reprocess` | `process` | Reprocess an episode. |
| `POST` | `/api/v1/episodes/{id}/cancel` | `process` | Cancel queued/running work. |
| `POST` | `/api/v1/episodes/{id}/ignore` | `process` | Ignore an episode and clean artifacts. |

Hard global podcast deletion is intentionally not exposed in API v1.
