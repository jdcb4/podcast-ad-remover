# V2 promotion record — 11 October 2026

**Release authorized: 2.0.0.** Joe accepted the reviewed README, release rationale
and screenshots and instructed “Go ahead with the 2.0 release.” The wording
reminder is resolved. Package files are aligned at 2.0.0.

## Release contents

See the [release overview](V2_RELEASE_NOTES.md#release-overview),
[changelog](CHANGELOG.md#200---2026-10-11), [upgrade guide](V2_UPGRADE.md) and
[README](../README.md). Local Piper speech, internal model/key cascades and Legacy
classification are deliberately retired; local transcription remains, including
optional experimental CUDA acceleration. The README uses one labeled, bordered
phone screenshot composite (119594 bytes) with demonstration data.

## Qualification and publication

The runtime application at revision `35c0e13e2f3f7e84ce58ce1857f13373eb67af85`
was previously qualified and deployed to Dev and production at Joe's request:
785 tests / 8 skips, native/offline container smoke, database integrity and
preservation, worker/media health, authenticated pages, reports/transcripts, RSS,
inline playback, downloads, byte-checked seeking and multi-hostname artwork.
Private operational receipts remain in the homelab reference.

The 2.0.0 candidate additionally includes the reviewed documentation/screenshots
and version metadata. Its exact commit and image are qualified before production
publication: local/Docker checks, required GitHub CI, offline container smoke and
Dev deployment checks. Production source is fast-forwarded to that same approved
commit; versioned image and `latest`, Git tag and GitHub release are then published.
Final qualification evidence and image measurements are recorded in the release
and deployment receipts. No user data reset or image-only downgrade is permitted.

## Installer gate resolved

The `github-pages` environment now explicitly allows workflow branch `dev`;
existing protection and channel/revision/immutable-image checks remain. Successful
[run38127114250](https://github.com/jdcb4/podcast-ad-remover/actions/runs/38127114250)
published the previously qualified Dev image. Homepage, Dev installer and exact
revision/image metadata were checked successfully on 11 October 2026. Stable
installer publication accompanies 2.0.0 and must be verified before announcement,
including single-HTML/offline downloads and preservation of the Dev channel.

## Size baseline and final measurement

The earlier measured `dev-bbced60` image was 47.8 MB smaller in uncompressed
layers (3.4%) and 25.3 MB smaller in compressed registry layers (5.3%) than 1.16.0.
Those measurements predate 2.0.0 and are retained as historical evidence in
[RESOURCE_AUDIT.md](RESOURCE_AUDIT.md#release-image-comparison--11-october-2026).
Re-measure the exact published 2.0.0 manifest/layers; do not relabel old measurements.
Piper/phonemizer accounted for 46.2 MB, while shared transcription dependencies
remain required. Downloaded models, optional CUDA libraries and user data are
outside the base image. No large RAM or word-perfect cutting claim is made.

## Verification limits

Synthetic regression tests do not replace a new real-audio CPU/GPU benchmark for
the October 8 seam merger. Ambiguous overlaps can retain repeated wording and
coarser cut boundaries. CUDA remains experimental. Qualification of the maintainer's
installations does not certify every provider, browser, operating system or GPU.
