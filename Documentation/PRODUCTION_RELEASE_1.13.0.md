# Production release 1.13.0 — 6 September 2026

Joe approved promotion after the [Dev qualification](DEV_ROLLOUT_2026-09-06.md).
**1.13.0 is deployed at [pod.jboxtv.com](https://pod.jboxtv.com/).**

Release source: `576a58808702429ee858f5930edd4d1fb80049b0`, promoted from `dev` to
`master` by fast-forward. Both version manifests and the changelog identify 1.13.0.
Later documentation commits do not change the deployed image.

```text
jdcb4/podcast-ad-remover:1.13.0
sha256:86a6781406544de31f2db595d0624e96e61c7ba4efed1d671c2cc612a2772422
```

`latest` was published at the same digest. The production container uses the exact
version and digest; CasaOS retains its existing `latest` catalogue entry and unchanged
saved configuration. The immutable version is the deployment identity.

## Release verification

- `npm run verify:docker` passed on the versioned Dev candidate: **314 tests**, Python
  syntax, Tailwind, frontend/Python audits and the image build. No known dependency
  vulnerabilities were reported. [Candidate CI passed](https://github.com/jdcb4/podcast-ad-remover/actions/runs/34005964787).
- Published and tested `dev-576a588`, digest
  `sha256:6d0b780032d1add9c192d26199c0361b608049a86023d5049c0a8007e595feed`.
  Its isolated production-data rehearsal migrated twelve schema versions to fourteen
  while preserving 80 subscriptions, 24,165 episode records, four users, 1,285 job
  records and all 230 then-published audio files. Startup, login/subscription routes,
  individual/unified RSS and a byte-for-byte audio range check passed without networking
  or automatic processing.
- The clean `master` release helper repeated all 314 tests and audits before publishing.
  [Production-branch CI passed](https://github.com/jdcb4/podcast-ad-remover/actions/runs/34006378796).
  Both candidate and release images passed the offline native-import, migration,
  backup, publication, AAC-to-MP3, startup and asset smoke checks.
- A comparison of all 107 files under `/app` found runtime files and modes identical
  between candidate and release. Only the two package manifests changed LF/CRLF line
  endings during branch checkout; their normalized bytes and parsed JSON match.
  Dependency/base layers, command, environment and health check match. This explains
  the different image digests without an application-code change.

## Cutover and backup

A scheduled production job began during preparation. Job 1286 / episode 27113 was
allowed to finish on 1.12.0 at 02:37:16 UTC. The stop gate required an empty active
queue before freezing the final snapshot, which therefore includes that publication.

The old container stopped at 02:38:32 UTC and the replacement started at 02:45:36 UTC,
about **7m04s** apart. The maintenance window included copying the final archive to
the workstation and rehearsing restoration before the new image touched production data.

Backup set: `release-1.13.0-20260906`. Exact private paths are recorded in
`C:\HomeLabRef\projects\index.md`; private configuration and credentials are not in Git.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `data.tar` | 14,023,956,480 | `5e3232fae21ed3901520ad34d95296514b66b1cca934c6066a0a2418114b4d0f` |
| `cutover.db` | 49,778,688 | `c7697a7e46dd652160dc6c421f44aa829b740d176b4df77fdd33b7f581ef7db3` |

The archive contains matching media, models, feeds and the finalized SQLite snapshot,
excluding older backup sets. The database snapshot includes committed WAL contents and
passed integrity checking. Both large artifacts match the off-host copy; eleven
configuration, metadata and database files also passed checksum comparisons.

An actual extraction of `data.tar` into a disposable **1.12.0** container passed
database integrity, twelve migration records, all **231** audio files, login/subscription
routes, both feeds and a non-first-byte HTTP 206 audio response matching disk. It had
no live data mount, host port or network access. Its container and temporary media copy
were removed after verification. The older stopped 1.11.0 container was retired under
the documented rollback-retention policy; the 1.12.0 container and image remain available.

## Production result

- Container `3ebc4df1b60a796a8c0d73b84f0712bb49c55735f7bece985b89984ddce36a50`
  is running and Docker-healthy, with zero restarts and no OOM event at verification.
- The `/DATA/PodcastFiles:/data` mount, port 8000, bridge network, restart policy,
  8 GiB memory / 16 GiB memory-plus-swap limits, session secret and provider environment
  are preserved. Every existing database-backed application setting matches the
  pre-upgrade snapshot. These include two concurrent jobs, automatic Whisper/FFmpeg
  thread selection and a sixty-minute feed interval. The new request budget defaults
  to twelve provider calls per job across retries.
- Production automatic processing remains enabled. Its worker heartbeat advances;
  the first feed check completed at 02:48:26 UTC and recorded the next deadline at
  03:48:26 UTC. At the post-cycle check there were no active jobs or pending publications.
- Database integrity passed with fourteen migrations, 80 subscriptions, 24,166 episodes,
  four users and 1,286 historical jobs. Five pre-existing failed episodes remain.
- All **230 currently published files** are present. The first normal retention pass
  removed episode 26935 from *The Ancients*: the newly finished episode had temporarily
  raised that podcast to four completed episodes, while its existing explicit limit is
  three. The three newer episodes remain published; the removed episode is in the full
  pre-upgrade backup. This accounts for the difference from the 231-file backup.
- Authenticated production login, dashboard, queue and subscription API passed.
  Public and Tailscale health returned 200. Public login, subscription UI, JavaScript,
  individual RSS and unified RSS passed. The latest enclosure's `bytes=1024-2047`
  response was HTTP 206 with 1,024 bytes matching the file on disk.
- Persistent Dev is unchanged at `dev-b78ccb1` and still displays 1.12.0. Rolling
  `dev` now identifies the versioned `dev-576a588` candidate. The released application
  changes are present in both; the persistent Dev version manifest was not redeployed.

## Recovery and remaining limits

Use [RECOVERY.md](RECOVERY.md) and the backup's `manifest.json`. Restore `data.tar`
into a fresh directory, pair it with the retained immutable 1.12.0 image, start with
automatic processing disabled, and validate before switching traffic. Preserve the
upgraded data for diagnosis. Do not simply start the retained rollback container against
the migrated live database. The restore rehearsal proves this particular backup/image pair.

The backup directory retains `candidate-rehearsal.json`, `rollback-rehearsal.json`,
`release-content-comparison.json`, `deployment.json`, `initial-production-health.json`,
`production-verification.json` and checksum receipts. Use `docker exec` for live SQLite
checks so root-owned WAL sidecars are accessible without changing host permissions.

Human listening and the usual podcast-client check were not repeated during promotion.
No additional paid test episode was started. HomeLabRef reports both Podcast services
up; its known unrelated central-listener warnings (HL-0016) and semantic-catalog
validation failure (HL-0020) remain. Structural and secret-pattern validation passes.
