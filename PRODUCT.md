# Product
<!-- impeccable:product-schema 1 -->

## Platform
web

## Users
Self-hosting administrators and podcast listeners managing shared podcasts and personal subscriptions.

## Product Purpose
Download podcasts and YouTube audio, transcribe locally, classify the complete episode, remove selected
content and publish replacement feeds. Keep installation and routine management approachable.

## Capabilities and Constraints
FastAPI/Jinja, SQLite and Docker. Preserve media, feed identities and existing ownership behavior.
The accepted design history is in Documentation/V2_PROPOSAL.md; current behavior and upgrade limits are in Documentation/V2_IMPLEMENTATION.md and Documentation/V2_UPGRADE.md. V2 deliberately retires legacy complexity; it is not yet declared a production release. Local transcription remains
faster-whisper; speech generation is API-only. Settings must work on phones and desktops.

## Product Principles
Prioritize podcast actions over metrics. Show the minimum useful settings first. Use consistent
controls, concise labels and optional help. Keep secrets out of rendered values and logs.

Import OPML/text with duplicate preview and per-feed outcomes. Provide optional staged onboarding, local-only install-file generation and a portable API agent skill. Fresh artwork and cut-tone defaults are on; other enhancements are off. Ownership changes and whisper.cpp remain deferred. Before V2 publication, remind Joe to expand his rationale.
