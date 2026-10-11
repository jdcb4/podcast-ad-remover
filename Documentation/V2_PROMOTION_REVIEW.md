# V2 promotion preparation — 11 October 2026

**Status: ready for maintainer documentation review, not approved for publication.**
Proposed release: **2.0.0**. Package files still say 1.16.0. This preparation does
not change `main`, SemVer/`latest`, repository deployment policy or live services.

## Review first

- [README: what changes and why](../README.md#why-v2-simplifies-things).
- [Draft public release body](V2_RELEASE_NOTES.md#proposed-release-body).
- [Upgrade checklist and conversion table](V2_UPGRADE.md).
- [Measured image sizes](RESOURCE_AUDIT.md#release-image-comparison--11-october-2026).

Joe's expanded rationale is recorded: affordable capable paid models make one
model practical for many users, and 9Router, OmniRoute and LiteLLM provide external
routing for those who need it. The README focuses on current features, CUDA acceleration and the principles behind
the three main V2 changes. Detailed compatibility conversions live in the changelog,
separate V2 release notes and upgrade guide. It links the hosted wizard and a
self-contained HTML download; users do not need to build or serve the wizard.
The explanation reminder has been raised; final wording still needs Joe's review.

## Application evidence

The tested application revision is `bbced603103a159c35565621eaf3e98dc46bbe79`
(PR #53), image `dev-bbced60`, immutable digest
`sha256:b5bc458c9d93c4b3cf84edf91ea4ab6c010c48df50843bdd1b929b5c4972edaa`.
Required merged-dev CI [38092782179](https://github.com/jdcb4/podcast-ad-remover/actions/runs/38092782179)
passed. That build passed 780 tests / 8 skips, syntax, CSS and audits, with the
existing documented build-only frontend exception. Offline container smoke covered
native imports, database/backup, audio conversion, startup, health and assets.

The exact image was deployed to the maintainer's Dev and production instances at
his request. These instance updates did not publish V2 to `latest`. Production
qualification included a network-disabled startup on a copied database,
preservation of existing rows, an integrity-checked cutover snapshot, unchanged
runtime configuration and sibling containers, healthy worker/media, login/logout,
dashboard/settings/Stats, reports/transcripts, individual/personal/global RSS,
two byte-checked HTTP206 audio ranges and artwork across six public/LAN/Tailscale
addresses. Counts at cutover:4 users,91 subscriptions,27504 episodes,1850 jobs.
Backup coverage passed; monitoring reported22up and no down/warn/unknown checks.
Sanitized operational receipts and recovery details are in the private homelab
reference, not in this public repository. This qualifies those installations,
not every provider, operating system, podcast player or GPU host.

## Findings from this review

1. **Rationale incorporated; copy review pending.** The earlier reminder to obtain
   Joe's explanation now records what he supplied rather than asking him again.
2. **Old seam finding superseded.** The October 8 implementation retains segments
   intersecting ownership windows and groups connected cross-chunk overlaps.
   Cached nested overlaps are normalized without rewriting source transcripts.
   README/release notes now explain conservative/coarser boundaries and possible
   repeated wording, rather than claiming the old segment-start-only algorithm
   is still running. Synthetic regression coverage does not substitute for a new
   real-audio CPU/GPU benchmark. See [timeline boundaries](COMPLETE_TIMELINE.md#boundaries-and-processing-order).
3. **Hosted installer still blocked.** Fresh read-only checks on11October found
   the Pages root serves the old evaluation report (HTTP200), while `/dev/` is404.
   Allowed environment branches remain `master` and the retired experimental
   branch. [Run38093480028](https://github.com/jdcb4/podcast-ad-remover/actions/runs/38093480028)
   failed on that policy. Both stable and Dev installer publication dispatch from
   `dev`. At Joe's request the launch README assumes working hosting; this dated
   operational finding remains a publication gate. A single downloadable HTML
   wizard now works without adjacent assets or a server. Repository-policy changes
   and Pages publication remain separate actions.
4. **Image reduction measured.** Compared with1.16.0, the tested V2 image is
   47.8MB smaller in uncompressed layers (3.4%) and25.3MB smaller in compressed
   registry layers (5.3%). Docker Desktop reports73.1MB less combined stored content
   (3.9%), which includes both representations. This is the net release difference,
   not an isolated Piper saving. Piper/phonemizer package files account for46.2MB;
   shared ONNX Runtime remains required by faster-whisper.
   Downloaded models, CUDA libraries and user data are outside the image.

## Remaining promotion steps

1. Joe reviews the README and release wording; record acceptance. Resolve the
   installer policy/publication issue before launch. The README is intentionally
   written for working hosting; verify the hosted and single-HTML downloads before
   publishing that copy.
2. Confirm2.0.0 and authorize cutting the versioned candidate. Align package files
   and dated changelog, commit on `dev`, run the required local/Docker checks,
   publish the exact Dev candidate and qualify its immutable image. Re-measure
   that final candidate; this review's image measurements predate docs/version changes.
3. Obtain explicit approval for that candidate's release. Promote the approved
   revision to `main` by fast-forward, then verify/publish2.0.0 and `latest` under
   [VERSIONING.md](VERSIONING.md). Verify release metadata, installer assets and
   the chosen installation/recovery point. Keep experimental CUDA limits visible.

## Verification of this documentation change

`npm run verify:docker` passed with Python3.11:780tests,8skips, Python syntax,
Tailwind rebuild, frontend audit with the existing documented build-only exception,
Python audit with no known vulnerabilities, and a local Docker build. The suite
reported the existing AnyIO deprecation warning; Tailwind reported stale Browserslist
metadata. Local file/heading checks covered83Markdown links without errors;
`git diff --check` passed. Documentation-only edits made after the application
checks do not change runtime code or dependencies. The final versioned candidate
still requires its own release qualification.
No new model calls, paid inference, live data migration, image push or deployment
is part of this documentation review.

### README and single-file wizard follow-up

At Joe's request, the README now emphasizes current app features and CUDA speed,
with a short principles section for Piper removal, cascade removal and Complete
Timeline. Detailed compatibility conversions are consolidated in the changelog;
the separate V2 release notes remain linked. Installation copy assumes working
Pages hosting and offers one downloadable HTML file, without user-facing local
build/server steps.

The configurator builder embeds CSS, scripts and channel metadata into `install.html`,
with exact CSP hashes and network connections still blocked. Publication regenerates
it for the channel's immutable image. The committed source copy clearly selects Dev.
Automated checks exercise the standalone file after deleting adjacent assets,
including GPU Compose generation, separate environment download, secret preservation,
explicit rotation and clearing. Hash checks and committed-output drift checks pass.
The browser client's URL policy prevented a native `file://` preview; automated
file-page tests passed, but no native-browser file preview is claimed.

`npm run verify` passed:781 tests,8 skips, syntax, CSS and dependency audits with
the existing documented frontend exception. Local Markdown links and
`git diff --check` passed. The application runtime and dependencies are unchanged;
the final versioned release candidate still needs the normal Docker qualification.
