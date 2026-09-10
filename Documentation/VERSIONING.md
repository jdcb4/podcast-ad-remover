# Versioning

Podcast Ad Remover uses Semantic Versioning in the form `MAJOR.MINOR.PATCH`.

The previous public release label `1.3` is treated as `1.3.0` from this point forward. Future releases should always include all three components.

## Version Source

- `package.json` is the primary version source.
- The root package entry in `package-lock.json` must match `package.json`.
- Docker releases are tagged with both the exact version and `latest`.

Example for version `1.3.0`:

```text
jdcb4/podcast-ad-remover:1.3.0
jdcb4/podcast-ad-remover:latest
```

## Branches And Image Channels

- `dev` is the GitHub default and primary integration branch for completed development work.
- `main` is the production branch and receives application changes only during an explicitly approved release promotion.
- Normal feature and fix branches start from and merge back into `dev`.
- Both long-lived branches require the GitHub Actions `verify` check. See [Git workflow](GIT_WORKFLOW.md) for protection, worktrees and cleanup.

Dev images stay in the same Docker Hub repository but use non-release tags:

```text
jdcb4/podcast-ad-remover:dev
jdcb4/podcast-ad-remover:dev-<git-sha>
```

`dev` is a convenient rolling tag for the persistent development instance. `dev-<git-sha>` is immutable in normal use and identifies exactly which commit was tested. Dev builds never publish a SemVer tag or update `latest`. The application version may continue to show the last production version while work is accumulating on `dev`; the image revision tag is the Dev build identity.

## Bump Rules

- `PATCH`: bug fixes, documentation fixes, small internal improvements, and dependency updates that do not change user-visible behavior.
- `MINOR`: new features, new settings, new UI flows, additive API changes, and backward-compatible database migrations.
- `MAJOR`: breaking configuration changes, incompatible API changes, destructive storage changes, or database changes that cannot migrate existing installs automatically.

Because existing installs may have many downloaded podcasts, database and `/data` compatibility should be treated as release-critical.

For the next release (1.14.0), the maintainer has explicitly accepted the timestamp
value-format change within `/api/v1` as a scoped exception to the major-version rule.
Retain the new UTC representation and document the parser impact in the release notes.
See the 2026-09-10 decision in [DECISIONS.md](DECISIONS.md). This does not change the
versioning rules for other incompatible changes or authorize production promotion.

## Release Checklist

During normal development, publish and test clean committed Dev builds as needed:

```bash
npm run docker:dev:publish
```

When Joe asks to prepare a release candidate:

1. Decide the next SemVer number.
2. Update `package.json`, `package-lock.json`, and `Documentation/CHANGELOG.md` on `dev`.
3. Run local verification on `dev`:

```bash
npm run verify
```

4. Run Docker verification:

```bash
npm run verify:docker
```

5. Commit the release candidate, publish its Dev image, and test that exact `dev-<git-sha>` build.

Do not continue until Joe explicitly approves production promotion of the tested candidate.

6. Confirm the approved `dev` revision has a successful GitHub Actions `verify` check. Fast-forward `main` to that exact revision without adding unrelated changes; do not squash or rebase a production promotion. If `dev` has advanced since qualification, use the recorded approved commit, not its newer tip. Stop and reconcile any divergence before publishing.
7. From a clean `main` checkout, repeat required release verification and publish:

```bash
npm run docker:publish
```

This builds and pushes `jdcb4/podcast-ad-remover:<version>` and `jdcb4/podcast-ad-remover:latest`.
The release helper refuses to run outside a clean `main` checkout. Return the primary working checkout to `dev` after completing the release.
