# Production release 1.14.0 — 21 September 2026

Joe explicitly requested Dev promotion, a new production image, and the production
upgrade. **1.14.0 is deployed at [pod.jboxtv.com](https://pod.jboxtv.com/).**

Release source: `214c9cd2a2f23424b247db7d3169f09b53268ac1`, fast-forwarded from
`dev` to `main`. [PR #34](https://github.com/jdcb4/podcast-ad-remover/pull/34)
updated both package versions and the changelog; application code is the previously
deployed Dev revision `101c3f1`. Later release-record documentation does not change
the image identity.

```text
jdcb4/podcast-ad-remover:1.14.0
sha256:9162be91da61cec9ee1ba80f40664182e85472f7fabefdf08651a5547d761606
```

`latest` was published at the same digest. Production is pinned to the exact
version and digest. CasaOS retains its saved configuration and `latest` catalogue
entry. The persistent Dev instance remains on `dev-101c3f1`; the rolling registry
Dev image is the release candidate `dev-214c9cd`, digest
`sha256:3f05f43053f83ccb06eb0fa53899f6eaec594e2e2ff43e5e9228bc4691b8b0d1`.

## Verification

- `npm run verify:docker` passed: 559 tests passed and five platform-dependent tests
  skipped on Windows, Python syntax, Tailwind, frontend/Python audits, and the Linux
  image build. An initial browser-date subprocess timeout passed on a complete retry.
- [Dev CI](https://github.com/jdcb4/podcast-ad-remover/actions/runs/35540299571)
  passed all 564 Linux tests and both audits before promotion.
- The clean `main` production helper repeated verification before publishing.
  [Main CI](https://github.com/jdcb4/podcast-ad-remover/actions/runs/35540425747)
  passed. Candidate and production images passed offline native-import, database,
  backup, publication, AAC conversion, startup and asset smoke checks. Dependency
  consistency passed. All 136 application files match between the candidate and
  production image after normalizing line endings.
- A production snapshot migrated from 14 to 21 recorded migrations. All existing
  subscription, episode, user and job fields were compared and preserved. Only
  recognized OpenAI/Anthropic model defaults changed among existing global settings;
  custom Gemini choices were preserved. Existing podcasts remain on Legacy, and the
  new removal tones and Gemini free-tier option remain disabled.
- The final backup was restored into separate, network-disabled containers using
  both 1.13.0 and 1.14.0, with automatic processing disabled. Both passed integrity,
  entity counts, all 235 audio-file paths/sizes/publication identifiers, authenticated
  pages, individual/unified RSS, and a byte-matched non-first-byte HTTP 206 audio check.

## Backup and cutover

Production had no active jobs at the stop gate. It stopped at 22:03:54 UTC on
20 September and the replacement started at 22:10:05 UTC, approximately **6m11s**
apart (08:03–08:10 on 21 September in Sydney).

Backup set: `release-1.14.0-20260921`. Private locations and configuration are in
`C:\HomeLabRef\projects\index.md`, not this repository. The archive contains the
matching media, feeds, models and database, excluding older backup sets and SQLite
WAL/SHM sidecars. Restoration explicitly installs the finalized SQLite snapshot.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `data.tar` | 15,332,679,680 | `29bff269b3d54c9c4e9eb682fbaa2a4cdeb2faf13eff84e1f6798ab95eb627e0` |
| `cutover.db` | 51,572,736 | `9fcb8d1abca41cce1dc41fb74baedc61b6eb297c27fbb41f735d6142bf392631` |

All ten original backup/configuration/evidence files matched the off-host SHA-256
checksums before the new image touched production data. The stopped 1.13.0 container
and image are retained for recovery. Do not start that container against migrated
live data: follow [RECOVERY.md](RECOVERY.md), restoring matching media/database into
a fresh directory with the matching old image and processing disabled.

## Production result

At initial verification, production retained 81 subscriptions, 24,928 episodes,
four users, 1,489 historical jobs and all 235 audio files. The 15 historical failed
jobs were not manually retried. Database integrity passed with 21 migrations.

The existing environment, session/provider values, data mount, port 8000, bridge
network, restart policy and 8 GiB memory / 16 GiB memory-plus-swap limits match the
pre-upgrade container. Automatic processing resumed with a current worker heartbeat
and feed polling. Docker health is healthy, with zero restarts and no OOM event.

Public and Tailscale health checks passed. Authenticated public/internal dashboard,
queue, subscriptions, login, subscribe page, JavaScript, individual/unified RSS and
byte-matched seeking passed. A public probe using Python's default User-Agent was
denied with 403; the browser User-Agent probe passed all checks.

Human listening and a podcast-client subscription check were not repeated. The
unrelated Codex Hub endpoint still refuses connections (existing HomeLabRef HL-0016);
direct production checks passed. No monitoring configuration was changed.
