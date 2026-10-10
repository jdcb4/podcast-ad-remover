# V2 promotion preparation — 11 October 2026

**Status: ready for maintainer documentation review, not approved for publication.**
Proposed release: **2.0.0**. Package files still say 1.16.0. This preparation does
not change `main`, SemVer/`latest`, repository deployment policy or live services.

## Review first

- [README: what changes and why](../README.md#upgrading-to-20-what-changes-and-why).
- [Draft public release body](V2_RELEASE_NOTES.md#proposed-release-body).
- [Upgrade checklist and conversion table](V2_UPGRADE.md).
- [Measured image sizes](RESOURCE_AUDIT.md#release-image-comparison--11-october-2026).

Joe's expanded rationale is recorded: affordable capable paid models make one
model practical for many users, and 9Router, OmniRoute and LiteLLM provide external
routing for those who need it. The README explains local Piper removal separately,
including loss of bundled offline speech, explicit API configuration, preserved
speech preferences and continued local transcription. It distinguishes feature
removals from less disruptive changes such as tone conversion and feed titles.
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
   `dev`; do not advertise the hosted wizard as available. Local generation works.
   Repository-policy changes and Pages publication remain separate actions.
4. **Image reduction measured.** Compared with1.16.0, the tested V2 image is
   73.1MB smaller unpacked (3.9%) and25.3MB smaller in compressed registry layers
   (5.3%). This is the net release difference, not an isolated Piper saving.
   Downloaded models, CUDA libraries and user data are outside the image.

## Remaining promotion steps

1. Joe reviews the README and release wording; record acceptance. Resolve the
   installer policy/publication issue before presenting the wizard as a launch
   feature, or explicitly agree to launch with local installation only and make
   that the primary README installation route.
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
