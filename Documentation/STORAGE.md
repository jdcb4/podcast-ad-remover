# Separate processed-audio storage

PAR can keep finished audio on a dedicated disk/NAS while SQLite, transcripts,
reports, artwork, models and temporary processing files stay under `/data`.
Leaving `MEDIA_DIR` unset preserves the existing single-volume layout. This is
optional; upgrading does not move audio automatically.

## Configure the mount

Mount an **empty dedicated directory** into the container and set `MEDIA_DIR` to
its container path. Mount network shares on the Docker host first. For example:

```yaml
services:
  app:
    environment:
      MEDIA_DIR: /media
    volumes:
      - /srv/par-appdata:/data
      - type: bind
        source: /mnt/nas/par-audio
        target: /media
        bind:
          create_host_path: false
```

This is a fragment to merge into your existing Compose service, not a complete
installation. Keep its other settings and current appdata mount. The static
configurator can generate the complete configuration with a separate audio mount.
Do not point media at appdata itself or the existing `/data/podcasts` tree.
A local directory such as `/data/media` also works if you want logical separation
without a second mount. Do not run two PAR installations against the same data.

After restarting, open **Settings → System → Storage** and choose **Check and
enable destination**. New processing waits until a configured destination has
been enabled. Enabling creates a volume identity marker; subsequent writes,
playback and cleanup require that identity. The application never creates a
missing media root or silently falls back to local storage. The host must still
ensure its NAS is mounted before Docker starts; directory existence alone cannot
prove that a network filesystem is mounted.

## Migrate from the UI

1. Check the destination and preview file count, size and free space.
2. Start migration. New episode claims and automatic retention pause; current jobs
   finish first. A database snapshot is saved under `/data/backups`.
3. The background operation copies one published file at a time, checks size and
   SHA-256, finalizes the media copy and commits its location. Originals stay.
4. Close the browser if needed. Progress is durable and resumes after application
   restart. Pause/End take effect after the current file. Pause or an error keeps
   processing paused; Resume retries the pending item. End retains completed
   mappings/copies and permits processing again if media is available.
5. Check playback, then explicitly choose **Delete verified local copies**. This
   rechecks both copies before removing each local audio file. Transcripts and
   reports are untouched. Keep a backup before cleanup.

Current audio, prior immutable published attempts and older `processed.mp3`
layouts are included. Missing published sources or conflicting destination files
stop the operation for review. Retry inputs are not bulk-migrated. Public audio
URLs and episode/feed GUIDs do not change; the database maintains storage-relative
URL mappings. Old and migrated audio can coexist indefinitely.

Progress is per file. Large files can take time without incrementing the counter.
Manual deletion during migration may remain pending until the operation ends;
retention and deletion remove registered media as well as local episode artifacts.

## New processing and outages

Download, transcription and FFmpeg work stay local. Finished output is copied to a
temporary file on media storage, verified and renamed there before the episode
commit. The local finished copy is then removed; if cleanup fails it is retained
for a later verified cleanup. Original downloads are still removed after success.
The SSD must have enough scratch space for concurrent jobs, including temporary
originals and outputs. This change does not move scratch files or models.

If the media mount is missing or has the wrong identity, new processing and
automatic retention pause. Requests for mapped audio return HTTP 503, not a false
404 or a local fallback. Restore the original mount/identity. An interrupted copy
can be retried; it never publishes a partial file or overwrites a differing file.
NAS latency and filesystem durability affect transfer/playback; qualify your
actual host and filesystem before moving the only copy of valuable audio.

## CLI and recovery

Run these inside the configured application environment after database startup:

```bash
python scripts/migrate_media.py status
python scripts/migrate_media.py enable
python scripts/migrate_media.py preview
python scripts/migrate_media.py copy --yes
python scripts/migrate_media.py run
python scripts/migrate_media.py pause
python scripts/migrate_media.py resume
python scripts/migrate_media.py cancel
python scripts/migrate_media.py cleanup --yes
```

The web background worker normally runs queued operations. `run` is the equivalent
for an offline administrator; it shares the same local lock and manifest. It may
wait for running jobs to drain. These commands never configure Docker mounts.

Back up appdata with the SQLite backup procedure in [RECOVERY.md](RECOVERY.md).
Exclude reproducible model caches and temporary retry inputs only if you accept
recreating that work. Back up media separately, including `.par-media-volume` and
its `audio/` tree. Appdata alone preserves configuration/transcripts, **not audio**.
Restore matching appdata and media snapshots; the media host path may change while
the configured container path and identity remain consistent.

Schema migration `20260928_0024_media_storage` adds mapping/state/manifest tables
and uses the normal pre-migration backup. Existing episode paths remain logical
identifiers for compatibility. To roll back to an older application, restore its
pre-upgrade database and matching original audio tree. Once originals are cleaned
up, the old image cannot read moved files without restoring them. Do not remove
the media configuration after migrating and expect playback to work automatically.
