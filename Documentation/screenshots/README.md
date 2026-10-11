# Application screenshots

The root README uses these current V2 captures, taken on 11 October 2026 from the
real application at revision `35c0e13` with a disposable demo library:

| File | Feature |
| --- | --- |
| `v2-app-overview.jpg` | README composite: library, playback and report side by side. |
| `v2-library-mobile.jpg` | My Pods and podcast artwork. |
| `v2-playback-mobile.jpg` | Episode summary, active playback, seeking and separate download. |
| `v2-report-mobile.jpg` | Timestamped classifications and Keep/Remove decisions, with an ad expanded to show its reason and transcript. |

The three source JPEGs were captured from a 390 × 844 phone-sized viewport.
The README uses one linked 680-pixel-wide composite, keeping the views in a single
row even on narrow screens. The 1274 × 948 composite is about 120 KB; click to enlarge.
Each screen has a contrasting border and a heading describing its feature.
Names, artwork, summaries, report
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

To rebuild the composite, size each source to 390 × 844 pixels and add a 4-pixel
border (`#9080ae`). Place them in library/playback/report order with 24-pixel
gutters, 16-pixel outer margins and a dark background (`#0a0a0f`). Reserve a
64-pixel heading band above the screens, using 26-pixel bold white text:
**Your podcast library**, **Play, pause & download**, **See what was removed**.
Save as `v2-app-overview.jpg` at JPEG quality 88. Keep the source captures for
individual inspection; the README loads only the composite.

## Historical images

`dashboard.png`, `episodes.png`, `public-subscribe.png` and `admin-queue.png` show
older interfaces. They remain historical evidence and are not presented as V2.
