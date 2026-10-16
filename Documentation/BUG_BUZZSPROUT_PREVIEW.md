# Buzzsprout feed denial and masked preview errors

Recorded and investigated: 2026-10-16. Status: fixes deployed to Dev; production
deployment outstanding. Joe will correct stale metadata manually; recovery is removed
from project work and acceptance criteria at his request.
Homelab follow-ups: HL-0066 (feed client), HL-0083 (preview errors).

## Reported production evidence

Production V2 subscription **80**, **If Books Could Kill**, displays **Loading...**.
Saved URL: `https://rss.buzzsprout.com/2040953.rss`.
Publisher-advertised URL: `https://feeds.buzzsprout.com/2040953.rss`, redirecting to the saved URL.
Both are legitimate addresses for the same feed.

Joe supplied an investigation reproduced inside production: default python-httpx
User-Agent receives HTTP 403 HTML; changing only User-Agent to
`PodcastAdRemover/2.0 (+RSS reader)` receives HTTP 200 XML, title If Books Could Kill,
and 80 entries. The advertised URL also succeeds through the validated redirect helper
with that header. Publisher internal blocking policy is unknown.

Five preview requests returned HTTP 409 on 15 October UTC. Read-only SQLite service
probes reproduced two underlying messages: saved URL gives “Choose a different feed
URL”; advertised alias gives “Could not discover complete available RSS history…”
after HTTP 403. The UI instead displays `Unexpected token '<' ... is not valid JSON`.
Historical request bodies were not retained, so individual attempts cannot be assigned
an exact rejection. These live observations are supplied evidence, not new live probes
performed during this repository investigation. The canonical operational record is
the homelab `runbooks/services/podcast-ad-remover.md`, 16 October diagnosis.

## Repository findings

Inspected local dev revision `25f1a93` without running application operations:

- `app/core/feed.py`, `FeedManager._fetch_feed`: HTTPX client has no explicit User-Agent.
  It retains a 30-second network timeout, initial/final URL validation, bounded
  redirect handling and declared/streamed feed size limits. A client-level header
  would also apply to requests made by `stream_get`; no validation bypass is needed.
- `app/web/static/js/podcast-operations.js`, `request`: POST sets Content-Type JSON
  but no Accept JSON; all responses are parsed unconditionally with `response.json()`.
- `app/web/podcast_operations.py`, `call`: service ValueError becomes HTTP 409.
  `app/web/error_handlers.py` returns HTML for that exception unless Accept includes
  application/json. JSON errors use message/error fields, while the shipped client
  only checks detail, so adding Accept alone would still hide the specific message.
- The 6 October fix exists as commit `32d58f69d1b0b2d5bdbee0b82eba3581223de2d2`
  on local branch `codex/fix-podcast-operation-errors`. It requests JSON, handles
  detail/message/error and guards redirected/session and non-JSON responses, with
  regression tests and documentation. It is not an ancestor of current HEAD; among
  locally known branches only that fix branch contains it. This reconciles the
  runbook's “repository fix, not deployed” statement with its absence from current
  dev. Remote refs and deployed artifact provenance were not freshly checked.
- Successful ordinary RSS polling does **not** refresh subscription title/image/
  description. `RSSAdapter.discover` returns only parsed episode entries;
  `Processor.check_feeds` inserts/backfills episodes and records check success.
  `SubscriptionRepository.record_check_success` updates check timestamps/errors and
  truncation only. A User-Agent fix alone therefore will not repair Loading....
- A successful `podcast_operations.preview` is **not read-only**: it reconciles
  batches and persists a preview. Do not use it against production for verification.
  Similarly, `check_feeds` can queue episodes. Use isolated fixtures or a read-only
  database replacement for rejection-only probes.

## Proposed next work

1. Qualify a descriptive RSS-reader User-Agent in the shared feed client, preserving
   timeouts, size limits, URL validation, DNS/network protections and redirect limits.
2. Review and integrate the existing 6 October fix with its tests, rather than
   recreating it without accounting for the unmerged branch.
3. Verify using isolated databases and mocked responses; bounded anonymous feed GETs
   may additionally qualify both official URLs without running discovery/polling jobs.

## Acceptance criteria

- Production-equivalent feed client retrieves both official URLs successfully.
- Same-URL and feed-discovery rejections display meaningful messages without JSON
  parsing failures; detail/message/error, HTML and session responses are handled safely.
- Redirect/network protections, timeouts and feed size limits remain intact.
- Verification does not queue episodes or change production subscriptions/jobs.

The initial investigation changed no application code or production state.

## Implementation, 16 October 2026

Joe subsequently authorized both fixes, requesting a neutral RSS client identity.
The shared feed client now uses `PAR/2.0 (+RSS reader)`; no expanded product name
or repository URL is included in that header. Applied the JavaScript, regression
tests and operations documentation from commit `32d58f6` onto the current branch.
URL validation, redirect/DNS protections, size limits and timeouts remain intact.

Anonymous GET probes from the workstation using the actual changed FeedManager
and a disposable data directory retrieved both official URLs successfully. Each
returned valid XML titled If Books Could Kill with 80 entries, ending at the saved
rss URL. This qualifies the exact neutral header from the workstation; the changed
client has not been deployed or re-probed inside production.

Regression coverage exercises header-dependent rejection/success, both URL forms,
restricted-mode DNS pinning and blocked private redirects, same-URL/denied-feed
HTTP 409 JSON responses, and JavaScript message/session/non-JSON handling. Database
checks use temporary fixtures and assert rejected previews leave feeds/jobs intact.
Production configuration, subscriptions, jobs and metadata remain unchanged. No
production preview/polling or provider calls were run. Joe will correct Loading...
metadata manually; no recovery implementation or agent action remains planned.

Verification passed with the existing project Python 3.11 environment on PATH:
`npm run verify` (795 tests passed, 8 skipped; syntax, CSS and dependency audits
passed with the documented frontend build-only exception). The 31 focused feed,
redirect and operation regressions passed. An initial run using default system
Python 3.13 failed because that environment lacks Anthropic and has older serving
dependencies; no application workaround or dependency migration was introduced.
