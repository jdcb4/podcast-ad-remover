# Podcast source and archive controls

Open **Podcast settings** on an individual show. Only its owner or an administrator can change its source or process available history. These actions support active RSS subscriptions, not YouTube sources.

## Change source feed

Preview uses the normal bounded download and redirect/network policies. It follows explicit `rel="next"` feed pagination, not publisher websites. It matches exact GUIDs (including reviewed aliases), then unique unchanged enclosure URLs. Suggested title/date matches require review. Choose an existing episode or explicitly accept a new episode for every unmatched entry. Two incoming entries cannot share an episode.

Confirmation atomically updates the source URL and records incoming GUID aliases while preserving episode IDs, published GUIDs, slug, media, settings and memberships. It creates no processing jobs. New entries subsequently follow the normal discovery/download window. Queued/running work or an unfinished archive batch blocks replacement. Duplicate URLs, stale previews and conflicts leave the old source untouched.

## Process whole show

Preview reports available, completed, already active, eligible and deliberately excluded episodes. “Whole show” means history exposed through RSS and supported explicit pagination; it cannot recover episodes omitted by the publisher. Discovery fails without queueing if invalid/incomplete or over 50 pages, 10,000 episodes, 50 MiB combined feed data or a 120-second between-page deadline (an in-flight request may take its normal timeout).

Confirm creates a durable, ordered batch of eligible episodes and enables **Keep whole show**. Effective retention/download-window values are frozen before disabling inheritance: the existing discovery window is not widened. The override prevents automatic count/age deletion of processed episodes, including manual downloads, but deliberate deletion still works. Switching it off later resumes ordinary cleanup.

Jobs start oldest first, missing publication dates last, with stable GUID tie-breaking. Parallel jobs can finish out of order. Archive work has lower priority than ordinary/manual work. Failures use normal retry limits and do not hold later episodes indefinitely. Completed episodes, active jobs and deliberately ignored/deleted episodes are skipped.

Pause stops new starts. Resume continues the same batch. Cancel stops remaining starts without deleting completed files or reverting retention. Already-running episodes finish even after pause/cancel. Failed running work in a cancelled batch is not retried by that batch. State and membership survive restarts; Tasks identifies archive work. One active/paused batch is allowed per show. A later preview includes remaining eligible work; normal automatic settings govern future episodes.

## Migration and rollback

Migration `20261002_0025_podcast_operations` additively introduces source aliases, expiring previews, archive batches/job associations and `keep_whole_show` (false for existing shows). Existing retention values are not reinterpreted. The migration runner creates an integrity-checked pre-upgrade SQLite backup under `/data/backups` before applying pending formal migrations.

Before downgrading, stop processing and restore the pre-upgrade database with its matching application revision, following [RECOVERY.md](RECOVERY.md). Older workers do not understand archive pause, priority or retention. Database rollback discards operations since that backup; preserve current data and new audio for recovery.

See [API.md](API.md) for equivalent versioned operations.
