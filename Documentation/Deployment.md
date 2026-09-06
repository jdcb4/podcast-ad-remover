# Deployment

Docker is the recommended deployment path.

## Docker Run

```bash
docker run -d \
  --name podcast-ad-remover \
  -p 8000:8000 \
  -v ./data:/data \
  -e GEMINI_API_KEY=your_key_here \
  -e BASE_URL=http://your-server-ip:8000 \
  jdcb4/podcast-ad-remover:latest
```

For a production install, also set a unique `SESSION_SECRET_KEY`.
If users access the app through HTTPS behind a reverse proxy, set `COOKIE_SECURE=true`.
Only set `TRUST_PROXY_HEADERS=true` when that proxy strips any client-supplied forwarding headers before passing requests to the app.
For authenticated management access behind a reverse proxy, set `BASE_URL` or the System Settings public application URL to the browser-facing origin so same-origin checks accept legitimate form submissions.

## Docker Compose

Use `docker-compose.prod.yml` and the published image when running a normal install:

```yaml
services:
  app:
    image: jdcb4/podcast-ad-remover:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY:-}
      - BASE_URL=http://your-server-ip:8000
      - SESSION_SECRET_KEY=${SESSION_SECRET_KEY:?Set a persistent random secret}
      - LOG_LEVEL=INFO
```

Copy `env.example` to `.env`, set the public URL and selected provider, and save a random
`SESSION_SECRET_KEY` once. For example, generate it with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`. Reuse it on upgrades.

Start it with:

```bash
docker compose -f docker-compose.prod.yml up -d
```

The repository `docker-compose.yml` is intended for local source builds and development. It bind-mounts the source tree into `/app`; do not use that file for a stable install unless you intentionally want live source edits inside the container.

## Local Or OpenAI-Compatible LLM Networking

Configure local text analysis from **Admin > AI Settings > Text Analysis**. The API base URL must use
HTTP or HTTPS, must not contain credentials, a query string, or a fragment, and normally includes the
service's `/v1` compatibility path.

Inside the Podcast Ad Remover container, `localhost` refers to that container—not the Docker host.
Choose the address that matches the LLM deployment:

- another service in the same Compose project: `http://ollama:11434/v1`;
- Docker Desktop host service: `http://host.docker.internal:11434/v1`;
- a LAN or Tailscale service: its trusted hostname or address, for example `http://llm-host:11434/v1`.

On native Linux Docker, `host.docker.internal` may require this Compose entry:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

The custom provider is deliberately separate from OpenAI. It uses only its own optional credential,
so selecting a local URL cannot send a saved OpenAI cloud key to that endpoint. Restrict local model
servers with host firewall and network controls; do not expose an unauthenticated generation endpoint
to the public internet.

## Unraid

The Unraid user-template XML lives at `Documentation/unraid/podcast-ad-remover.xml`. It uses the published `jdcb4/podcast-ad-remover:latest` image, maps `/data` to `/mnt/user/appdata/podcast-ad-remover`, and exposes port `8000`.

## Data Volume

Mount `/data` and follow [RECOVERY.md](RECOVERY.md) for an online database snapshot, matching media backup and isolated restore rehearsal before upgrades.

Important paths:

- `/data/db/podcasts.db`: SQLite database.
- `/data/podcasts/`: podcast and episode artifacts.
- `/data/feeds/`: generated RSS files.
- `/data/artwork/`: cached artwork generated when the ad-free badge is enabled.
- `/data/models/`: downloaded local model files.
- `/data/app.log`: application log.

Do not delete `/data` unless you intentionally want to remove the app database and downloaded podcasts.

The inheritance and artwork migrations are additive and the normal startup migration path creates a
database backup under `/data/backups/` first. Existing podcast setting values are not erased. If an
upgrade must be rolled back, stop the container and restore the matching pre-migration database backup
before starting the older image; cached files in `/data/artwork/` can remain because older releases do
not read them.

## Notifications

Admin notifications are optional and off by default. Configure them from **Admin > Notifications** after the app is running.

The app embeds the Apprise Python library, so no extra container is required for most setups. Add one Apprise URL per line and use the test button before relying on alerts. Examples of supported targets include:

- ntfy for simple mobile/web push, including self-hosted ntfy servers;
- Gotify for a self-hosted notification server;
- Pushover for low-maintenance hosted push;
- Discord, Telegram, Slack, email/SMTP, and webhook targets.

Notification URLs can contain bearer tokens, webhook IDs, usernames, or passwords. Treat them as deployment secrets and avoid sharing screenshots of the Notifications page.

## Building From Source

```bash
docker compose up -d --build
```

For a local image without Compose:

```bash
docker build -t podcast-ad-remover:local .
```

The image pins yt-dlp plus its matching EJS scripts and copies Deno from a pinned multi-architecture
image stage. YouTube extraction does not self-update or download remote components at runtime.
`SPONSORBLOCK_ENABLED` remains `false` unless the operator deliberately enables it after reviewing
the SponsorBlock API/data licence.

## Development Image Channel

Committed builds from the `dev` branch use two tags in the normal Docker Hub repository:

- `jdcb4/podcast-ad-remover:dev` follows the newest published Dev build.
- `jdcb4/podcast-ad-remover:dev-<git-sha>` identifies an exact build and should be recorded when testing.

Build locally from a clean `dev` checkout with:

```bash
npm run docker:dev
```

Publish both Dev tags with:

```bash
npm run docker:dev:publish
```

These commands do not update production SemVer tags or `latest`. Persistent Dev deployment configuration is intentionally handled separately from the image publishing workflow.

The default image includes Piper TTS and is intended primarily for `linux/amd64`. Experimental Apple Silicon / ARM64 builds can skip Piper TTS:

```bash
npm run docker:experimental:arm64 -- --push
```

This path targets `linux/arm64`, tags the image as `jdcb4/podcast-ad-remover:experimental-arm64`, and sets `INSTALL_TTS=0`. Piper is unavailable in that image, but spoken summaries and title intros can still be tested by selecting Gemini TTS and configuring a Gemini API key. Podcast download, transcription, ad detection, cutting, feed generation, and the web UI remain the intended test surface.

## Release Publishing

Production releases are promoted from a tested Dev revision only after explicit approval. The release helper runs only from a clean `main` checkout.

Before upgrading an existing install with important data, dry-run database migrations against a copy:

```bash
npm run db:migration-dry-run -- --db-path /path/to/data/db/podcasts.db
```

The command copies the database to a temporary data directory and runs the normal startup migration path there without modifying the source database.

Before publishing:

```bash
npm run verify:docker
```

To publish the current version from `package.json`:

```bash
npm run docker:publish
```

This pushes both:

- `jdcb4/podcast-ad-remover:<version>`
- `jdcb4/podcast-ad-remover:latest`

## Worker readiness and supervision

Run one Uvicorn web worker per data directory. It supervises a dedicated processor child
with restart delays bounded to 5–60 seconds. The child writes a heartbeat every 10 seconds;
90 seconds without a heartbeat is stale. The minimal `/health` endpoint returns 503 for a
missing/stale enabled processor or an unavailable database, and 200 for a healthy worker
or explicitly disabled automatic processing. Dashboard authentication does not redirect
health probes; an IP allowlist still applies, so include the container loopback probe.
Docker uses this endpoint with a 120-second startup grace period. Docker health status
alone does not restart a container; the parent handles child restarts, while the configured
container restart policy handles process exit.

With `PROCESSOR_ENABLED=false`, scheduled discovery is disabled. Explicit manual actions
use a shared, bounded in-process runner. The queue page states that automatic processing
is disabled. Memory figures prefer cgroup limits and label host fallback; storage figures
are cached for 30 seconds. Feed-check deadlines are recorded by the actual scheduler.
