# Backup, migration rehearsal and recovery

Create an online database snapshot with Python's SQLite backup API. This includes
committed WAL transactions; copying only `podcasts.db` while the app runs is unsafe.

```sh
python scripts/backup_database.py --output /data/backups/podcasts-manual-YYYYMMDD.db
python scripts/migration_dry_run.py --db-path /data/backups/podcasts-manual-YYYYMMDD.db
```

Use a new destination each time. Both commands refuse overwriting an existing file
or the source. Startup takes a snapshot before any schema changes when migrations
are pending. Snapshots are integrity-checked before they become available.

Also back up `/data/podcasts`, artwork, configuration and the session secret to
private storage. A database snapshot alone cannot recover missing audio. For a
consistent full backup, stop processing during the media copy or use filesystem
snapshots. Keep at least one copy on a different device and test restoration.

To recover: stop the application, preserve the current data directory for diagnosis,
restore the chosen database into a fresh data directory's `db/podcasts.db`, and
restore the matching media tree. Do not combine a restored database with the old
`-wal` or `-shm` files. Start the matching immutable application image against that
copy, with `PROCESSOR_ENABLED=false`, then check login, episode playback, feeds and
the migration dry run before switching traffic. Re-enable processing when satisfied.

For a rollback after an upgrade, use the previous image and the pre-upgrade snapshot
with its corresponding media backup. Downgrading only the executable is not a
database rollback. No recovery command automatically stops services or replaces data.

Migration `20260910_0015_timestamp_provenance` adds `users.last_login_is_utc` and
`episodes.processed_at_is_utc`, both defaulting to `0`. It leaves all existing timestamp
values, media paths, and publication identifiers unchanged. Earlier releases did not record
the server timezone used for these two fields, so the UI labels old naive values "timezone
unknown" and the API preserves them. The next successful login or episode completion records
a new UTC value and sets its flag to `1`. Requeueing or failed processing does not relabel history.
Existing PR #21 databases follow the same conservative upgrade policy; no offset is guessed.

Startup takes an integrity-checked snapshot before this additive migration. Rehearse it with
`migration_dry_run.py` as above. Roll back with the previous image, pre-upgrade database snapshot,
and matching media backup; do not run an older timestamp writer against the upgraded database,
because it would not maintain the provenance flags. There is no automatic conversion of history.

Migration `20260824_0013_unified_feed_preferences` adds the unified feed's title, description,
episode-title prefix flag and artwork URL to `app_settings`, preserving the previous defaults.
The original migration identifier is retained so installations already running PR #20 keep their
custom preferences when upgrading. Existing `dev` databases receive the same additive columns.
Use the previous image with the pre-upgrade snapshot and matching media backup for rollback;
do not drop columns or rewrite the live database as part of a downgrade.

Migration `20260905_0013_processing_recovery` is additive (claim cancellation flag,
published duration/GUID, publication retry flag and worker-status table). Restore the
pre-migration snapshot with the previous image to roll back. New media directories are
not renamed over legacy data. Failed attempts cannot replace an existing publication.
A completed episode with `publication_pending=1` retains playable audio; the processor
retries RSS publication on its next cycle. With processing disabled, run this inside the application image with the existing `DATA_DIR`:

```sh
python scripts/publish_pending_feeds.py
```

It retries only feed publication, prints the remaining count and exits nonzero if work remains.
It does not run transcription/ad detection or remove the prior audio.

Migration `20260905_0014_processing_budgets` adds request counts, scratch reservations,
working-directory references and provider usage records. It uses the same pre-migration
backup/rollback process. Usage records contain provider/model names, timing and token
counts, never prompts, keys or transcript text. They show calls and reported tokens, not
an inferred invoice amount. Remote failures may omit token counts.

Automatic retries copy finalized source/cache files into a fresh owned stage. Transcripts
require matching source SHA-256 and Whisper model; analysis also requires the same
transcript, prompts, selected provider/models and removal options. Abandoned unpublished
attempts older than 48 hours are removed unless an active job still references them.
Published revisions remain available until episode deletion/retention removes their root.
