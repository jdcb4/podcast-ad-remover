# Dev rollout — 6 September 2026

## Current status

The assessment changes are merged and pushed to `dev`, and deployed at
https://poddev.jboxtv.com/. These changes are unreleased; the application still reports **1.12.0**.
Production, `master`, SemVer tags and `latest` have not changed.

The automated Dev qualification is complete. The short AAC, full AFL and 5½-hour
history processing checks passed, including publication, audio integrity, resource
observations and recovery. Joe's podcast-client check and subjective listening at
cut boundaries remain outstanding. This is not production approval.

Deployed revision: `b78ccb1fe52018dc8fab2225f13131b4d694f547`.

```text
jdcb4/podcast-ad-remover:dev-b78ccb1
sha256:0620381ab1663df486fe3efc476a0a27075852d1e2bf463a9487a660c6aaaa30
```

Both the immutable Dev tag and rolling `dev` were published. Compose pins the
immutable tag and digest. Scheduled feed processing remains disabled on persistent
Dev. Explicit jobs use one concurrent slot, two Whisper CPU threads, two container
CPUs, the existing 2 GiB memory / 4 GiB memory-plus-swap limits, and at most four
provider requests per job. The copied library is not being scheduled wholesale.

## Integration and verification

- The original 17 assessment commits were fast-forwarded into `dev`, followed by
  whitespace-only review cleanup `6689c1c`.
- Live qualification found and fixed two further defects in independent commits:
  `4a2be53` honors explicit RSS API initial-download counts, including zero;
  `b78ccb1` preserves published counts, previews and listen totals during replacement
  processing. These add ten regression cases.
- **314 tests passed**, plus Python syntax, Tailwind, npm audit and Python audit.
  No known dependency vulnerabilities were reported. The final image built and
  published, and [GitHub verification passed](https://github.com/jdcb4/podcast-ad-remover/actions/runs/34000456723).
- The first deployment image passed an offline container smoke for native imports,
  migrations, backup, real AAC conversion, app startup, health, dashboard and assets.
  Later fixes affect API queue selection and display queries; the processing engine
  and dependencies are unchanged from the exercised image.
- Authenticated login, dashboard, queue and subscription API passed. Public login,
  subscription page, JavaScript, individual RSS and unified RSS passed. Both Dev
  login endpoints are classified up by HomeLabRef.
- Browser inspection verified the public subscription page and corrected counts
  while a published episode was reprocessing. Four real podcasts and fourteen
  published episodes remain after removal of the temporary fixtures.
- SQLite integrity passed; all fourteen previous audio files survived the upgrade,
  and both new migrations applied, giving fourteen migrations total.

## Backup and rollback

Dev was stopped for a consistent full backup before its first upgrade. This contains
matching media, an integrity-checked SQLite snapshot, Compose configuration and a
private copy of the existing secret environment. Copies on JMKtec and the workstation
have matching checksums. Secrets and backup contents are not in this repository.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `data.tar` | 2,249,205,760 | `507bcd5a15e282f806b07081302bac2d7c4257630bfc9df2c1161f5c1b36555c` |
| `podcasts.db` | 50,900,992 | `4591f5c3acea772a32c1e3c106ed7d4657e09d0127521394c70e2bff5b94cba3` |

The previous `dev-30d4a70` image successfully booted a disposable restore of this
database and media: integrity, all fourteen files, health, subscription UI and an
audio range request passed. No live data was replaced by that rehearsal. Additional
online database snapshots precede the two follow-up image updates.

The full rollback point is `deploy-6689c1c-20260906`. Exact private locations and
the previous image identity are recorded in `C:\HomeLabRef\projects\index.md`
and the backup's `manifest.json`. Follow [RECOVERY.md](RECOVERY.md): restore matching
database/media into a separate directory with the previous image, validate them,
and preserve upgraded data for diagnosis. A code-only downgrade is not a rollback.

## Live exercises

| Exercise | Evidence |
| --- | --- |
| Cancel running and queued jobs | Running AFL and queued history jobs cancelled; the running claim was acknowledged and released. Previous publications still served HTTP 206 audio ranges. |
| Reprocess AFL episode 26614 | Completed on its first attempt after deliberate cancellation. Source about 31m40s; measured output 1,590.648163 seconds (26m31s). Four removal intervals reflect the configured content whitelist, including intro/outro removal. |
| Publish replacement | New MP3 and feed available; publication pending cleared. RSS duration 1,591 seconds and new enclosure HTTP 206 for bytes 1024–2047. Previous file retained. |
| No-cut AAC | A 120-second excerpt of already processed history narration passed real transcription and Gemini analysis. Zero removal intervals; valid MP3 of 120.032653 seconds. |
| Zero initial downloads | Fixed API discovered the fixture episode as unprocessed and queued zero jobs. Positive and negative counts also have real-database HTTP regressions. |
| Unattended scheduler and recovery | Disposable instance automatically discovered and processed one bounded AAC fixture. One request and completed attempt, zero cuts. Killing only its processor child caused replacement and healthy readiness in 3.02 seconds, with no duplicate provider call. |
| Long history episode 11683 | Job 934 completed on its first attempt on the final image. MP3 measured 19,368.437551 seconds (5h22m48s), 371,000,559 bytes; full FFmpeg decode reported no errors. Individual and unified RSS published the new GUID/enclosure and duration 19,368 seconds. |

The long episode completed at **01:27:34 UTC**. Its report contains three finite,
ordered removal intervals: 0–48.16, 618.9–731.3 and 19,528.96–19,764.035813 seconds,
totaling 395.635813 seconds (about 6m36s). Under the existing content whitelist and
intro/outro settings, these cover an opening advertisement, the show introduction,
and closing credits/support messages. This validates the configured processing
path; it does not establish whether every cut matches Joe's listening preference.
The report's final bound comes from the source duration measured during processing;
the original download was removed by normal successful-job cleanup.

Post-completion checks at 01:38 UTC found zero active jobs, zero active reserved
bytes, zero pending publications, and SQLite `quick_check=ok`. Small public
`bytes=1024-2047` requests returned HTTP 206 for both replacement and previous audio,
with returned bytes matching the files on disk. All fourteen pre-upgrade audio files
remain present. Public and Tailscale health returned 200; Dev has zero restarts/OOM
events. Production's container ID and image ID still match the pre-rollout manifest.

The unattended rehearsal used a fresh database, a copied Whisper model and one
loopback feed, with no production mount or host port. Its container exited and was
removed. Temporary subscriptions on persistent Dev were removed through its API.

## Resources and provider usage

The successful AFL job ran from 23:57:01 to 00:08:06 UTC, about eleven minutes.
Its container's recorded memory peak was 1,659,719,680 bytes (about 1.55 GiB),
with no OOM event or unplanned restart. The long job ran from 00:10:38 to 01:27:34 UTC,
**76m56s**, with one provider request and no retry. Its historical scratch reservation
was 2,523,904,000 bytes; there are now no active reservations. Completed jobs retain
reservation sizes for history, without consuming queue capacity.

The final image's 152 samples span 00:10:45–01:27:46 UTC. Container CPU usage increased
by 8,621.17 CPU-seconds (143.69 CPU-minutes), averaging 1.87 cores over that interval.
Memory peaked at the 2 GiB cgroup ceiling, with 881 recorded `memory.events max` events,
zero `oom`/`oom_kill` events and zero restarts. Cgroup usage includes filesystem cache,
not only Python memory. Host free disk started at 128,140,259,328 bytes, bottomed at
126,550,851,584 and ended at 127,828,647,936. These are whole-filesystem observations;
other host writers and cleanup prevent attributing the entire change to this job.

| Successful request | Model | Input tokens | Output tokens | Request time |
| --- | --- | ---: | ---: | ---: |
| AFL episode | `gemini-3.5-flash` | 20,027 | 407 | 18.496 s |
| Persistent Dev no-cut sample | `gemini-3.5-flash` | 769 | 69 | 15.631 s |
| Isolated unattended sample | `gemini-3.5-flash` | 649 | 1 | 16.886 s |
| Long history episode | `gemini-3.5-flash` | 100,711 | 317 | 17.351 s |

Google's published standard paid rates are US$1.50 per million input tokens and
US$9 per million output tokens, implying approximately **US$0.1539 for the long run**
and **US$0.1904 for all four successful qualification requests** from these returned
counts (122,156 input and 794 output tokens). This is an estimate, not an account invoice;
actual charges depend on account tier and billing records. [Google pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash).

## Remaining checks and limits

- Confirm subscription/listening in Joe's usual podcast client. Its name was
  requested. Direct media navigation was blocked by the browser tool; server-side
  range checks passed. No workaround was used for that browser block.
- Subjective listening quality is not proven by successful processing and valid
  timestamps alone. Human listening at cut boundaries is still useful.
- External health returned 200 from the workstation. A default Python urllib
  public-origin request from JMKtec returned 403, while direct Tailscale health was
  200. No network/security configuration was changed.
- HomeLabRef structural/secret-pattern validation passes. Its complete validator
  still stops on the existing unrelated `chook-defence-lab` criticality enum issue
  (HL-0020). Existing central-monitoring listener issues remain under HL-0016;
  direct Podcast Ad Remover checks pass.

The bounded read-only sampler exited after recording completion at 01:27:46 UTC.
Its `processing-samples.jsonl`, `processing-result.json` and summarized
`long-job-resources.json` remain beside the rollout backup. The matching
`long-job-validation.json` records feed, range, report and database checks.
No further jobs were started. The completion follow-up is paused after this record
is committed; the human listening/client check remains separate.
