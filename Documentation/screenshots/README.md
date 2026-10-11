# Application screenshots

The root README uses these current V2 captures, taken on 11 October 2026 from the
real application at revision `35c0e13` with a disposable demo library:

| File | Feature |
| --- | --- |
| `v2-library.webp` | My Pods, cover artwork and RSS subscription actions. |
| `v2-playback.webp` | Episode summaries, active playback, seeking and separate downloads. |
| `v2-content-removal.jpg` | Podcast defaults and removal categories. |

Compressed WebP/JPEG keeps download sizes modest; the README displays linked 280-pixel thumbnails.
Names, artwork, summaries and silent audio are synthetic. No live library, account,
credential or protected feed URL is included.

## Refreshing the captures

From the repository root, with development dependencies installed, run
`python scripts/preview_readme.py` and open `http://127.0.0.1:8767/?view=mine`.
The helper uses its own temporary data directory and demo session, binds only to
loopback, rejects write requests and starts no processor or model calls. Stop it
normally to remove the demo data. Do not use this helper as a deployed server.

Capture the library in grid view, a podcast's episode cards during playback, and
**Settings → Podcast defaults**. The current captures use desktop viewports around
1200 pixels wide. Review images for readable controls and private information
before committing; retain descriptive alt text in the README.

## Historical images

`dashboard.png`, `episodes.png`, `public-subscribe.png` and `admin-queue.png` show
older interfaces. They remain historical evidence and are not presented as V2.
