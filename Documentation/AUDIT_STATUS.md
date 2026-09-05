# Audit status

Updated 2026-09-06. The 2026-09-05 whole-project assessment was approved for implementation.
Work is on `codex/assessment-improvements`, based on `dev` at `cc21a25` (1.12.0).

[Assessment implementation](ASSESSMENT_IMPLEMENTATION.md) is the current finding-by-finding
record, with verification results and deployment limits. [CHANGELOG.md](CHANGELOG.md) describes
unreleased user-visible behavior; [RECOVERY.md](RECOVERY.md) is the upgrade/restore procedure.
Production promotion, version changes and release tags remain separate decisions.

The [June 2026 audit status](history/AUDIT_STATUS_2026-06-11.md) and root `AUDIT.md` are historical
evidence. Their branch names, test counts and deferred items do not describe the current checkout.
The active design choices remain SQLite/local storage, FastAPI/Jinja, source/provider adapters,
shared podcast ownership and explicit dev-to-production promotion.
