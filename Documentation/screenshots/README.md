# Application screenshots

The root README uses these current V2 captures, taken on 11 October 2026 from the
real application at revision `35c0e13` with a disposable demo library:

| File | Feature |
| --- | --- |
| `v2-library-mobile.jpg` | My Pods and podcast artwork. |
| `v2-playback-mobile.jpg` | Episode summary, active playback, seeking and separate download. |
| `v2-report-mobile.jpg` | Timestamped classifications and Keep/Remove decisions, with an ad expanded to show its reason and transcript. |

All three are JPEG captures from a 390 × 844 phone-sized viewport, displayed as
linked 200-pixel thumbnails in the README. Names, artwork, summaries, report
evidence and silent audio are synthetic. The report uses the app's real renderer,
timestamp normalization and cut-preference logic; no model was called.
No live library, account,
credential or protected feed URL is included.

## Refreshing the captures

From the repository root, with development dependencies installed, run
`python scripts/preview_readme.py` and open `http://127.0.0.1:8767/?view=mine`.
The helper uses its own temporary data directory and demo session, binds only to
loopback, rejects write requests and starts no processor or model calls. Stop it
normally to remove the demo data. Do not use this helper as a deployed server.

Use a 390 × 844 viewport. Capture My Pods, then the first episode card during
playback with its seek bar visible. Open **More → View Report**, expand
**38.00–55.00s · Ad · Remove**, and frame the report from **Cut preferences** /
**Transcript timestamp normalization** through the classification rows. Leave the
other rows collapsed so Keep and Remove decisions remain visible together.
Review images for readable controls and private information
before committing; retain descriptive alt text in the README.

## Historical images

`dashboard.png`, `episodes.png`, `public-subscribe.png` and `admin-queue.png` show
older interfaces. They remain historical evidence and are not presented as V2.
