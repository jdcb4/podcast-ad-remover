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
The accepted v2 scope is recorded in Documentation/V2_PROPOSAL.md. Local transcription remains
faster-whisper; speech generation is API-only. Settings must work on phones and desktops.

## Product Principles
Prioritize podcast actions over metrics. Show the minimum useful settings first. Use consistent
controls, concise labels and optional help. Keep secrets out of rendered values and logs.
