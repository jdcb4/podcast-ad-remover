# Production release 1.15.0 — 24 September 2026

Joe explicitly requested promotion of Dev, publication of the production Docker
image, and the production upgrade.

**1.15.0 is deployed at [pod.jboxtv.com](https://pod.jboxtv.com/).**

Release source: `1161f61ad43f0281cbe1b2c5c1c19f6056389080`, fast-forwarded from
`dev` to `main`. [PR #37](https://github.com/jdcb4/podcast-ad-remover/pull/37)
aligns the package versions and changelog. Application changes are the previously
deployed Dev revision `75fea70`: configurable redirects and browser-local light/dark
mode. CUDA and compute-type configuration are not included.

```text
jdcb4/podcast-ad-remover:1.15.0
sha256:87e2c847755cc2b552501680f5eda108f5e8a92646b8d96e9155572fca33ae58
```

`latest` was published at the same digest. The qualified candidate is
`dev-1161f61`, digest
`sha256:d04667508e69132c0d37f2e7724159d3cffa8603441f1702f81ce47bbd38fef9`.
The persistent Dev service remains on `dev-75fea70`; its application behavior is
unchanged by the version-only candidate preparation.

## Verification

- `npm run verify:docker` passed: 575 tests passed, five POSIX-only tests skipped on
  Windows, Python syntax, Tailwind, both dependency audits, and image build. The
  initial run hit the existing 15-second browser-date subprocess timeout; a full
  rerun passed without code changes.
- Required [Dev CI](https://github.com/jdcb4/podcast-ad-remover/actions/runs/35927043227)
  passed before promotion. [Main CI](https://github.com/jdcb4/podcast-ad-remover/actions/runs/35927239284)
  passed. The clean-main production helper repeated local verification before publishing.
- Offline container smoke passed native imports, database migrations and backup,
  publication recovery, AAC conversion, startup, health, dashboard and assets.
  Dependency consistency passed. All 137 application files match between the
  candidate and production image after normalizing line endings.
- Migration rehearsal preserved every existing subscription, episode, user, job
  and app-settings field. Schema records increase from 21 to 22; the new redirect
  limit is 8. No existing saved preferences are rewritten.
- All ten original backup files passed off-host size/SHA-256 verification. Actual
  full restores under both 1.14.0 and 1.15.0 passed integrity, all 235 audio files,
  authenticated pages, RSS and byte-matched seeking, with networking and processing disabled.
- After deployment, all 1,523 media/feed files match their backup hashes. Public
  authenticated pages, Admin System settings, theme/CSS assets, RSS and audio ranges
  pass. Browser light mode survives reload; dark mode was restored after checking.

## Deployed state

Production uses the version/digest above. Environment and secrets, data mount,
ports, restart policy, Docker network and memory/swap limits are preserved. CasaOS
retains its saved configuration and catalogue entry. The background processor is
enabled with a current heartbeat. Docker reports healthy, zero restarts and no OOM.
Initial counts remain 81 subscriptions, 24,979 episodes, four users and 1,539 jobs
(1,491 completed, 26 cancelled and 22 historical failed); no historical jobs were retried.
The upgrade adds one schema record, for 22 total.

The old service stopped at 22:16:55 UTC and 1.15.0 started at 22:23:25 UTC on
23 September (24 September Sydney), approximately 6m30s apart. The stopped
`podcast-ad-remover-rollback-1.14.0-20260924` and its image are retained.
Persistent Dev was not restarted or changed.

CodexHub port 3015 remains unavailable, matching existing HomeLabRef backlog
HL-0016; direct production and Dev health checks pass. Human listening and a
separate podcast-client test were not repeated for this release.

## Backup and rollback

The idle production service was stopped gracefully for a consistent full backup.
Private configuration, online/final SQLite snapshots, matching media archive and
verification evidence are stored at:

```text
/DATA/PodcastFiles/backups/release-1.15.0-20260924
C:\Users\joedo\.codex\artifacts\podcast-prod-1.15.0-20260924\backup
```

The on-host directory is mode 700; the workstation artifact directory has a
restricted ACL. Private container inspection/configuration files contain secrets
and must not be committed or shared.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `data.tar` | 15,278,008,320 | `fc410dd3c09b00f0122d23cbeaa9bc46488867f45fed538f8a89af8cdc3845ef` |
| `cutover.db` | 51,765,248 | `68e82a561524844099dcf00c679cd6c4bb92d747624fb5bcb2dbcbfb0376b6f3` |

For rollback, restore the matching archive and final database into a fresh data
directory with the saved configuration and immutable 1.14.0 image
`sha256:9162be91da61cec9ee1ba80f40664182e85472f7fabefdf08651a5547d761606`.
Start with processing disabled, check pages, feeds and seeking, then switch traffic
and restore normal processing. Do not run the old container against migrated live
data. Preserve the current data for diagnosis. See [recovery](RECOVERY.md).
