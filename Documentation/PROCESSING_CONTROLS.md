# Processing controls, collection feeds and statistics

## Pause controls

Administrators can toggle **Pause feed** in Podcast Settings > Manage, beneath Process whole show. The same tri-state action (unchanged/pause/resume) is an independent subsection of bulk Content Removal. Owners who are not administrators cannot change either pause control.

**Settings > System > Pause all processing** stops claims across all podcasts. Both controls persist across restarts. Running episodes finish; queued, retry, manual and archive work waits. Discovery checks skip paused podcasts (or all podcasts during global pause). Automatic count and manual-age episode cleanup is blocked while paused, including a transaction guard against a pause after cleanup selected its candidates. Manual removal and deliberate podcast deletion remain available. Existing audio and feeds remain available. Resume restores ordinary discovery, queueing and retention settings; accumulated queued work may start and overdue retention may run.

The desktop admin sidebar places Processing paused above the color-mode control, linking to System without interrupting expanded Settings navigation. My Pods and Library headings show a contrasting processing-paused suffix on desktop and mobile. Feed pause icons appear in mobile rows, desktop tables, grid/art views and podcast details, without modifying published feed metadata. Tasks and Settings remain admin-only; Stats is available to all signed-in users. With dashboard login disabled the installation retains its existing implicit-admin behavior.

## Unified feeds

- `/feed/unified.xml` remains the global Library feed, irrespective of ownership.
- `/feed/users/{user_id}/unified.xml` contains local, non-ignored audio from that user's current collection (`user_subscriptions`), irrespective of ownership.
- My Pods copies the personal feed; Library copies the global feed, including after an enhanced view switch. `/account/feed?view=mine` is the default personal link helper; `view=library` selects global.

Personal XML is regenerated under the publication lock on every fetch, so changes in membership, publication, deletion and feed presentation are reflected without a separate scheduler. It uses the global feed title/artwork preferences. Unknown users return 404. When feed authentication is enabled, a personal feed requires that user's active feed token or that user's existing Basic/legacy query credentials. Another user's credentials cannot retrieve it. Audio enclosure access uses the same validated credentials. With feed authentication disabled, personal feeds follow the existing public-feed policy. Dashboard sessions alone are not podcast-client authentication.

## Stats

The page defaults to My Pods and All time. Matching visible button selectors switch between My Pods/Library (Library is admin-only) and Week/Month/Year/All time. All authorized totals are rendered on the initial page response; selector changes reveal preloaded sections without requests. Current holdings follows collection selection; Historical processing and admin-only Resources used follow both selectors and display calendar date ranges. Individual metrics use compact bordered components. Explanations and measurement limits are documented here rather than on the page.

Current holdings count subscriptions and retained published episodes in the selected collection, with output audio hours and source-minus-output hours saved. Missing duration pairs are explicitly excluded from savings; time saved measures removed audio, not listening activity. New successful processing records the probed source duration. Admins see My Pods and Library; regular users see only their own holdings and history. Without login there is no personal identity, so My Pods represents the shared installation.

History counts each episode once, using its first recorded successful processing time. Reprocessing updates its measurements without creating another episode count. Collection membership is captured at completion; removing a podcast later does not remove its prior personal statistics. Joining a collection later does not retroactively earn its processing history. History survives media cleanup and podcast deletion. The processing date, not source publication date, determines the UTC calendar week (Monday start), month or year. Counts of podcasts in history mean distinct podcasts processed in that period.

Admin usage shows persisted provider call attempts for the selected collection and reported input/output tokens. Provider records include classification, summary and speech calls; missing token measurements are not estimated. Recorded transcription seconds accumulate successful published attempts for episodes whose timings were recorded after this change. Failed/interrupted transcription and historical transcription times are unavailable. No currency cost estimate is inferred. Personal resource attribution uses retained job-to-episode links and recorded completion membership or current membership at the call time; calls whose jobs were deleted cannot be assigned to personal history. Global resource totals still include those calls.

## Migration, backup and rollback

Migration `20261010_0027_pause_statistics` adds default-off pause columns and independent `processing_history` / `processing_history_users` tables. The existing migration runner takes a WAL-safe database backup before changes. No existing media, podcast settings or feed identities are reset.

The migration imports retained episodes with known processing dates and attaches current memberships only when their recorded added-at date precedes processing. Previously removed media and past membership intervals cannot be reconstructed. Imported history may be incomplete and older timestamps may have the previous server timezone; these limits remain documented in this guide. Minimal early episode schemas import no invented history when required columns are absent. Re-running initialization does not duplicate imported records.

For rollback, retain the previous immutable image and pre-migration database backup. Restore the matching database backup with the app stopped as described in RECOVERY.md; do not drop columns or clear media in place. Older code will not honor the new pause controls, so keep processing disabled during rollback if a pause must remain effective.
