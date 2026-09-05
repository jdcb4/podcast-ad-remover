# Security posture and reporting

Podcast Ad Remover targets a personal Docker/homelab installation. Dashboard login,
feed protection and the public Subscribe page are separate settings. With dashboard
authentication disabled, reachable clients have management access; place the service only
on a network you intend to trust. One Uvicorn worker per data directory is the supported topology.

## Authentication and permissions

- Dashboard passwords use bcrypt. Sessions are signed, not an encrypted secret store.
- Set a persistent, randomly generated `SESSION_SECRET_KEY` before enabling dashboard or
  feed authentication. Startup and settings forms reject known example placeholders.
- `COOKIE_SECURE=true` makes cookies HTTPS-only. `ENVIRONMENT=production` disables debug
  and interactive API docs; it does **not** automatically enable secure cookies or login.
- SameSite=Lax cookies and same-origin checks protect authenticated management mutations.
  Configure the public application URL to match the browser-facing origin.
- Owners/admins manage podcast settings and episode processing/removal. Other members have
  read/library/discovery access; deleting the global podcast is admin-only. Pure ownership
  rules live in `app/core/permissions.py` and are exercised through real HTTP routes.
- Feed and optional v1 API tokens are stored as hashes. Feed links are bearer secrets;
  revoke compromised tokens in Feed Access. Session-key rotation does not revoke those tokens.
- The v1 API is disabled by default and uses separate scoped credentials and SQLite request
  counters. Login throttling is process-local (5 failed attempts per IP over 15 minutes).

## Network and browser boundaries

`TRUST_PROXY_HEADERS` is off by default. Enable it only behind a proxy that strips incoming
client-supplied forwarding headers. The global IP allowlist runs before public-route bypasses;
include the container loopback address if Docker health checks must pass an allowlist.

`ALLOW_PRIVATE_FEEDS=true` intentionally permits trusted LAN/self-hosted sources. Restricted
mode validates every feed/artwork/RSS media redirect before connecting, rejects non-public DNS
answers and pins the validated address while preserving TLS hostname verification. Environment
proxies are disabled in that mode. YouTube extraction and explicitly configured provider endpoints
are separate integrations; this is not a general network sandbox.

The main UI CSP still allows inline handlers. Template nonce attributes alone do not make it a
nonce-enforced policy. Escaping and route permissions remain essential. Report values are escaped,
and every `/artifacts/report/` response, including old reports, gets a script-free opaque sandbox.
Other headers include nosniff, DENY framing, restricted device permissions, referrer policy and
HSTS. HSTS is sent for proxy deployments; HTTP clients do not gain TLS merely from that header.

## Secrets, data and recovery

Provider settings may be stored in SQLite or supplied through the environment. Custom endpoints
use only their own optional credential. Protect the database, `.env`, backups, provider settings,
notification URLs and feed URLs. Logs can contain upstream error details or private source URLs;
review and redact them before sharing. Do not claim that every handled error is free of internal
information merely because production debug is off.

Use the [recovery runbook](Documentation/RECOVERY.md). Online backups include committed WAL data
and are integrity-checked; media needs its own matching backup. Worker claims fence writes and
cleanup to owned attempts, and failed reprocessing retains the last published audio. These measures
reduce corruption risk but do not replace off-device backups or an actual restore rehearsal.

For public exposure, configure HTTPS/login, the appropriate feed policy, proxy trust and a stable
secret before opening access. The image currently uses the default root user for compatibility with
existing volume permissions. Non-root operation requires a tested ownership/mount plan; it is not
silently imposed on existing installs.

## Verification and incident handling

Run `npm run verify` for regression tests plus frontend/Python audits, and the disposable container
smoke from [VERIFICATION.md](Documentation/VERIFICATION.md) before deployment. Audits cover known
package advisories at run time, not a security guarantee or a full OS/native-binary scan.

For an incident, restrict access at the existing network boundary, preserve logs/data for diagnosis,
identify affected accounts/tokens, rotate the relevant credentials and restore a verified image/data
pair if necessary. Coordinate disruptive service or network changes with the operator.

Report security issues privately to the repository owner using the repository's available private
contact or security-reporting channel. Include affected revision, reproduction steps and impact;
do not post working secrets, private feed URLs, production databases or user transcripts publicly.
