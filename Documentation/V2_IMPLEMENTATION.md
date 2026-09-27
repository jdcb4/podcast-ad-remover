# V2 implementation record

Implementation authorized on 2026-09-27, on local `dev`. The accepted scope is in
`V2_PROPOSAL.md`. Ownership changes and whisper.cpp are deferred. No production promotion
or image publication is part of this authorization.

## Milestone 1: migration foundations

Additive migration `20260927_0023_v2` introduces speech endpoint fields, a single cut-tone flag,
onboarding state, artwork source selection and a credential-free upgrade report. It selects
the first old model/credential, marks Piper as unconfigured, converts queued jobs to timeline
snapshots and preserves owners, published media and feed identities. Running jobs block upgrade;
stop processing and drain them before migration. Fresh databases offer onboarding, existing
databases do not automatically show it.

The existing migration runner takes an integrity-checked database backup before modifications.
For rollback use the previous immutable image and pre-v2 database backup, plus matching media
recovery point if processing has occurred. A binary-only downgrade after v2 writes is not supported.

Validation: real SQLite migration tests cover backup integrity, repeat startup, queued snapshots,
credential exclusion, preserved ownership/media, fresh onboarding and rejection of running jobs.
The project `.venv` supplies Python 3.11 and the dependencies missing from system Python.

## Milestone 2: processing and speech

The processor now classifies Complete Timeline only. SponsorBlock calls and the local Piper
worker/dependencies are removed. Provider requests select one model and credential with environment
precedence; schema-free retries and key rotation are removed. A small JSON decoder accepts valid
wrapped objects and rejects malformed/ambiguous output. Speech adapters support Gemini, OpenAI,
OpenRouter and a separately configured custom endpoint, with bounded responses and atomic output.
One cut-tone switch controls all cut positions. Existing publication/report readers remain.

Validation: 23 focused migration, provider, speech-adapter and real-audio timeline tests pass.
The broad suite exposed tests asserting retired legacy/cascade behavior; those and the UI contract
tests are being revised with the interface milestone. This is not yet a full verification pass.
