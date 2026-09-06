# Git workflow

## Branches and checks

| Ref | Purpose | Lifetime |
| --- | --- | --- |
| `dev` | GitHub default; normal integration and development builds | Permanent |
| `main` | Production source; explicitly approved promotions of tested Dev revisions | Permanent |
| `codex/<topic>`, `feature/<topic>`, `fix/<topic>` | One active change, based on and merged into `dev` | Delete after merging |
| `archive/<topic>` tag | Preserve concluded, unmerged research without keeping an active branch | Retained for reference |

Both permanent branches block deletion and force pushes and require the GitHub Actions `verify`
check. Pull requests must be current with their target branch before merging. GitHub automatically
deletes merged pull-request branches; `dev` is the default and both permanent branches are protected.
The existing administrator bypass remains available for deliberate recovery. Passing checks do not
replace Joe's explicit production-promotion approval, and routine agents must not bypass the rules.

GitHub does not deploy images automatically. See [VERSIONING.md](VERSIONING.md) for testing the
exact Dev image, approval, fast-forward promotion to `main`, versioning and Docker publication.
Repository-maintenance changes to `main` also require explicit authorization; a branch rename is
not an application release.

## Normal work

Start with a clean checkout, update `dev`, and create one branch for the change:

```bash
git fetch origin --prune
git switch dev
git pull --ff-only
git switch -c codex/example-change
```

Run `npm run verify`, commit the coherent change, push its branch, and open a pull request targeting
`dev`. Wait for the required `verify` check. After merging, return the main checkout to `dev`, pull
with `--ff-only`, and remove the merged local branch. Keep concurrent work on separate branches.
Do not reuse a merged branch for a different task.

A separate worktree is optional for simultaneous changes:

```bash
git worktree add ../podcast-example-change -b codex/example-change dev
```

Use this instead of the preceding branch-creation command, with a new branch name and directory.
Before removing a worktree, inspect its `git status --short --untracked-files=all`, review ignored
local files with `git status --short --ignored`, and preserve anything valuable. Confirm its commits
are merged, then use `git worktree remove <path>` without `--force`. Remove its branch only after
the worktree has been removed. Do not delete a directory to unregister a worktree.

## Periodic cleanup

```bash
git fetch origin --prune
git worktree list --porcelain
git worktree prune --dry-run --verbose
git branch -vv
git branch --merged dev
```

Protect `dev` and `main` from cleanup. For each other branch, check for open pull requests or active
tasks and use `git merge-base --is-ancestor <branch> dev` to prove ancestry before deleting it.
`git branch -d <branch>` removes a merged local branch; `git push origin --delete <branch>` removes
a reviewed, merged remote branch. Recheck the remote tip before deletion if other agents may be
working on it. Age alone and a missing upstream do not prove that work is disposable.

Squash-merged branches may not pass the ancestry check. Inspect their pull request and diff before
removal. Preserve unique concluded work under an annotated `archive/<topic>` tag and verify that
the pushed tag resolves to the original branch tip before removing the branch. Never publish an
archive as a SemVer tag or Docker release. Prune worktree metadata only after confirming the old
directory is gone and has no recoverable work.

## Updating clones that still use master

The production branch was renamed on 2026-09-06. In an older clone, first inspect local changes and
branch divergence. If a local `master` exists and `main` does not, rename it without resetting it:

```bash
git branch -m master main
git fetch origin --prune
git branch --set-upstream-to=origin/main main
git remote set-head origin -a
git switch dev
git branch --set-upstream-to=origin/dev dev
git pull --ff-only
```

If `dev` does not exist locally, use `git switch --track origin/dev` instead. If both `master` and
`main` already exist, compare them and preserve unique commits before removing either. Do not
recreate `master` on the remote.

Update external scripts and saved raw-file URLs from `/master/` to `/main/`. In particular, older
Unraid templates may retain the old `TemplateURL` or icon URL; the checked-in template uses `main`.
GitHub redirects ordinary branch URLs but does not redirect raw-file URLs or old `git pull`
targets. See [GitHub's branch-rename guidance](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/renaming-a-branch).

## Cleanup record: 2026-09-06

The audit started from clean `dev` at `98978437ef6da0ab7a051ade450bb25fc4fe0682`, identical to
production `master`. There was one registered worktree and no stale/prunable worktree metadata.
Twelve leftover local branches were proven to be ancestors of `dev` and removed:

```text
DocumentationUpdate
agent/custom-llm-endpoint-main
audit
audit-work
codex/assessment-improvements
codex/bulk-subscription-delete
codex/dev-rollout-fixes
codex/fix-bulk-delete-submit
codex/queue-stall-resilience
codex/version-bulk-js-asset
feature/subscription-management-improvements
feature/youtube-sources
```

The only unmerged owned branch was the explicitly concluded local-LLM experiment, with 14 unique
commits. Its complete history is preserved by annotated tag `archive/local-llm-transcript-chunking`
at `96fa4f51646538c025f5870229657953b256ad88`; its local and remote branch were removed. The research
decision and results remain in [LOCAL_LLM_EVALUATION.md](LOCAL_LLM_EVALUATION.md). A verified Git
bundle plus original refs and repository settings were also saved under the local Git directory's
`maintenance-backups/git-structure-20260906-190237/` before cleanup.

Five open contributor pull requests (#9, #19, #20, #21 and #22) were preserved in their forks;
#9 and #19 were retargeted to `dev`. Their code was not merged or discarded by this cleanup.
Only `dev` and `main` remain as owned branches after the maintenance change is integrated.

The previous rule protected a nonexistent `main` branch while `master` and `dev` were unprotected.
The repaired rule covers both permanent branches and requires `verify` from GitHub Actions.
The repository default is now `dev`, and merged pull-request branch deletion is enabled.
The existing production 1.13.0 image and data are unaffected by this Git maintenance.

Local verification passed `npm run verify`: 318 Python tests, Python syntax, the CSS build and both
dependency audits. The release-guard tests accept clean `main` and reject `dev`, the old `master`,
feature branches, detached HEAD and dirty `main`. GitHub verification is required before integration.
No Docker build or deployment is needed for this repository configuration change. Other clones and
external consumers of the old raw URLs were not inspected; use the migration instructions above.
