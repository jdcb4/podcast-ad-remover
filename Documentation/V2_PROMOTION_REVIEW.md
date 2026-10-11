# V2 promotion preparation — 11 October 2026

**Status: ready for maintainer documentation review, not approved for publication.**
Proposed release: **2.0.0**. Package files still say 1.16.0. This documentation
refresh does not change `main`, SemVer/`latest`, deployment policy or live services.

## Review first

- [README: current features and screenshots](../README.md).
- [README: what changes and why](../README.md#why-v2-simplifies-things).
- [Draft public release body](V2_RELEASE_NOTES.md#proposed-release-body).
- [Upgrade checklist and conversion table](V2_UPGRADE.md).
- [Measured image sizes](RESOURCE_AUDIT.md#release-image-comparison--11-october-2026).

Joe's expanded rationale is incorporated: capable paid models make one selected
model practical for many users, while 9Router, OmniRoute and LiteLLM provide
external routing for those who need it. The README emphasizes the current app,
optional CUDA acceleration and the principles behind the three main V2 changes.
Detailed compatibility conversions remain in the changelog, release notes and
upgrade guide. Installation starts with Docker/source, followed by the optional
hosted or single-HTML wizard. Current phone-sized screenshots use disposable demonstration
data rather than a private library.

Joe expressed satisfaction and requested this final tidy-up on 11 October.
The explanation reminder has been raised; record acceptance of final release
wording before publication rather than asking him to repeat the rationale.

## Application evidence

The tested application revision is `35c0e13e2f3f7e84ce58ce1857f13373eb67af85`,
including [PR #54](https://github.com/jdcb4/podcast-ad-remover/pull/54) and
[PR #55](https://github.com/jdcb4/podcast-ad-remover/pull/55). Its image is
`dev-35c0e13`, immutable digest
`sha256:abe063c8b8fc98864e9d643775419c678e138c8e0158781007ba0038bc0fd39b`.
Required merged-dev CI [38118175834](https://github.com/jdcb4/podcast-ad-remover/actions/runs/38118175834)
passed. Exact-image qualification passed 785 tests / 8 skips, syntax, CSS and
dependency audits with the documented build-only frontend exception. Offline
container smoke covered native imports, database/backup, audio conversion,
startup, health and assets.

That exact image runs in the maintainer's Dev and production instances at his
request. These installation updates did not publish V2 to `latest` or promote
source `main`. Cutover checks covered database integrity and row preservation,
unchanged configuration/mounts and sibling containers, worker/media readiness,
authenticated pages, reports/transcripts and individual/personal/global RSS.
Production counts at cutover: 4 users, 91 subscriptions, 27506 episodes, 1853 jobs.
Backup coverage passed; monitoring reported 22 up with no down/warn/unknown checks.

Player qualification checked rendered controls, inline playback, separate
attachment downloads and byte-checked HTTP 206 seeking. Dev playback checks
covered three HTTPS origins; production playback checks covered its public HTTPS
origin. Artwork was byte-checked across seven Dev and six production addresses,
with same-origin UI URLs, absolute RSS artwork and unchanged CSP. Sanitized
operational receipts and recovery details remain in the private homelab reference.
This qualifies those installations, not every provider, browser, operating system,
podcast player or GPU host.

## Release findings

1. **Classification boundaries:** the October 8 seam fix retains crossing speech
   and connected overlap alternatives, including cached nested overlaps, without
   rewriting source transcripts. Ambiguous items can have coarser cut boundaries
   and repeated wording. Synthetic regression tests do not replace a new
   real-audio CPU/GPU benchmark. See [timeline boundaries](COMPLETE_TIMELINE.md#boundaries-and-processing-order).
2. **Hosted installer publication is blocked:** checks on 11 October found the
   Pages root still serves the old evaluation report and `/dev/` returns 404.
   Environment rules permit retired branches rather than `dev`; the latest
   [publication run 38118318989](https://github.com/jdcb4/podcast-ad-remover/actions/runs/38118318989)
   was rejected by that protection. Both channel workflows dispatch from `dev`.
   At Joe's request, launch README copy assumes working hosting; resolve and verify
   this gate before publication. The committed single-HTML download already works
   independently of Pages.
3. **Image measurements are an earlier baseline:** the measured V2 image
   `dev-bbced60` (digest `b5bc458…`) was 47.8 MB smaller in uncompressed layers
   (3.4%) and 25.3 MB smaller in compressed registry layers (5.3%) than 1.16.0.
   Docker Desktop reported 73.1 MB less combined stored content (3.9%).
   These are net differences, not isolated Piper savings. Piper/phonemizer package
   files accounted for 46.2 MB; ONNX Runtime remains required by faster-whisper.
   Downloaded models, optional CUDA libraries and user data are outside the image.
   These measurements predate the current player build and must not be relabeled
   as its size.

## Remaining promotion steps

1. Review the updated README and final release wording; record acceptance. Resolve
   the installer environment/publication issue, then verify hosted channels,
   single-HTML downloads and displayed immutable image metadata.
2. Confirm 2.0.0 and authorize cutting the versioned candidate. Align package files
   and dated changelog on `dev`; run required local/Docker checks, publish and
   qualify its exact immutable Dev image. Re-measure that final candidate.
3. Obtain explicit approval for that candidate's release. Fast-forward `main` to
   the approved revision, then verify/publish 2.0.0 and `latest` under
   [VERSIONING.md](VERSIONING.md). Verify release metadata, installer assets and
   recovery points; keep experimental CUDA limits visible.

## Verification of this documentation refresh

The refresh updates release evidence and adds current README captures plus a
disposable, loopback-only screenshot helper. Application runtime code and
dependencies are unchanged. `npm run verify:docker` passed: 785 tests, 8 skips,
syntax, CSS, dependency audits with the documented build-only exception, and a
local Docker build. All 63 checked local documentation/image links resolve;
`git diff --check` passed. Demo pages/artwork returned HTTP 200 and writes were
rejected with HTTP 405. Three JPEGs captured at a 390 × 844 viewport total 91297
bytes. The README displays one linked 680-pixel-wide side-by-side composite
(119594 bytes), with clear borders and feature headings, so the phone views cannot
wrap into a tall stack. The report shows timestamped
Keep/Remove classifications with one ad expanded to reveal its reason and text.
The final versioned candidate still requires release qualification.
No paid inference, live migration, image push or deployment is part of this refresh.
