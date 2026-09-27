# Complete Timeline processing (v2)

V2 classifies the complete episode once, then applies removal choices in code. There is no Legacy workflow or whitelist mode. Configure defaults under Podcast defaults; owners/admins can override the content-removal group per podcast. Published media is not automatically reprocessed.

## Categories and definitions

**Prompt rules** offers collapsible category definitions with per-rule defaults and a preview of the assembled prompt. Optional podcast-specific guidance is under advanced podcast settings and is always applied when nonempty.

| Category | Default meaning | Initial removal choice |
| --- | --- | --- |
| Ad | External commercial messages, offers, sponsor reads and calls to purchase. Check for interruptions within the discussion. | Existing Ads preference |
| Promo | Distinct pitches for shows, memberships, merchandise, books or events. Substantive readings and interviews remain Content. | Existing Promos preference |
| Intro | Generic greetings, identification and opening housekeeping. Substantive setup, explanations, demonstrations and cold opens remain Content. | Existing Intros preference |
| Outro | Generic sign-offs, credits, routine like/subscribe reminders and upcoming-episode notices. Substantive conclusions, summaries and reflections remain Content. | Existing Outros preference |
| EditorialNonSpeech | Context-supported music samples, illustrative recordings or performances, including samples with transcribed lyrics. | Keep |
| NonEditorialNonSpeech | Context-supported ad jingles, non-editorial bumpers or dead air. Missing text alone is insufficient evidence. | Remove |
| Content | Substantive material, ordinary pauses and every uncertain classification. | Keep |

No transcript text does **not** prove silence. The app inserts explicit leading, internal and
trailing GAP items using the measured source duration. The model receives nearby speech and
episode metadata, not the audio. An introduction to a musical example supports keeping a gap;
a sponsor segue followed by a gap and sales pitch supports a non-editorial classification.
Unexplained gaps remain Content. This version does not supply acoustic silence detection.

Legacy custom instructions are retained and included as additional classification guidance
when a podcast opts in. Review instructions that say what to cut: express those preferences
through definitions and category checkboxes. The fixed rules tell the model to classify every
item, protect uncertain/mixed boundaries, and leave cut selection to the app.

## Boundaries and processing order

The model returns inclusive `first_id` / `last_id` ranges covering every numbered item exactly
once. The app maps these IDs to the original segment/gap boundaries, without accepting invented
model timestamps. Transcript overlap seams are normalized at their midpoint when both segments
remain valid; the report records that adjustment and the source transcript is unchanged.
Unresolvable timestamps fail processing. A mixed speech item that cannot be split safely stays
Content. Cut precision is therefore limited by Whisper's segment timing; it is not word alignment.

1. Classify the complete timeline and generate the summary.
2. Select removal intervals from the podcast's category choices. If already enabled for YouTube,
   include the selected SponsorBlock categories as separate external evidence.
3. Merge overlapping/adjacent removal intervals. Remove a retained island only when it lies
   between two such intervals and is **strictly shorter** than the threshold.
4. Cut and stitch the remaining audio with FFmpeg.

At a 10-second threshold, `Ad → 8 seconds of Content → Intro` bridges the Content only when both
Ads and Intros are selected for removal. Exactly 10 seconds is kept. Zero disables bridging.
Leading and trailing retained material is not an island. This rule deliberately overrides even
Content or editorial classifications for qualifying islands; the report identifies those extra
cuts separately. It does not modify the original model labels.

Editorial non-speech is never directly selected for removal. The advanced short-island policy remains independent.

## Summary and reports

One response contains both `segments` and `summary`. The default summary is 2–3 sentences,
starts exactly with **This episode includes**, and excludes the podcast name, full episode title,
publication date, ads and housekeeping. Historical dates discussed in the episode are allowed.
Admins can edit the summary instructions independently of the category definitions.

The combined summary appears in the report. Existing description-rewrite and spoken-summary
switches still control RSS description replacement and TTS. Enabling Complete Timeline alone
does not turn either feature on. The Legacy `append_summary` umbrella flag retains its meaning.
If summary formatting fails, one summary-only repair is attempted without discarding valid
classifications. If it still fails, the report records the error, cuts can complete, and there is
no invented summary. A later reprocess can retry the summary while reusing the classification.

The human report shows final cuts, all model classifications with reasons/transcript context,
Keep/Remove outcomes, the combined summary, provider/model/format, and separate short-island
cuts. JSON reports retain their existing root `segments` as the final removal intervals, adding
`workflow`, `analysis` and `edit_policy` for Complete Timeline.

## Output handling and provider routing

**AI Prompt Rules → Output handling and effective prompt preview** shows the assembled system
instructions, JSON schema and resolved model selections, including unsaved definition edits.
It can include a podcast's effective custom instructions. Preview does not call a model or expose
API keys. The numbered transcript is supplied as a separate source-data message at processing time.

Native JSON schema output is required. A complete valid JSON object can be unwrapped from prose or a code fence; malformed syntax, duplicate keys, nonfinite values and incomplete coverage are rejected.

Native requests use OpenAI-compatible `response_format.json_schema` for OpenAI, Gemini,
OpenRouter and custom endpoints. OpenRouter requests require endpoints that accept the schema
parameter. Anthropic uses `output_config.format`. See the official
[OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs),
[Gemini](https://ai.google.dev/gemini-api/docs/openai#structured-output),
[OpenRouter](https://openrouter.ai/docs/guides/features/structured-outputs), and
[Anthropic](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) contracts.

One selected model and credential are used. Unsupported schemas, outages, authentication errors, truncation and refusals never trigger model/key/format fallback. Temperature and reasoning parameters remain omitted. Request budgets and timeouts apply to classification and summary repair. Invalid classification is never treated as a clean episode.

## Migration, cached results and rollback

V2 migration `20260927_0023_v2` creates timeline snapshots for queued/retryable legacy jobs and normalizes existing timeline snapshots. Running jobs must be drained first. Credentials are loaded only at execution, never written to snapshots. The upgrade report identifies conversions. New jobs freeze the prompt, schema, selected model and removal policy; retries retain that snapshot.

Classification caches require matching source SHA-256, transcript, measured duration, prompt,
schema and provider configuration. Changing only removal choices or the island threshold can
reuse matching classification on reprocess. A changed download, including dynamically inserted
ads, invalidates reuse. Previous published audio stays available until its replacement completes.
Changing workflow does not move or rewrite existing reports, feed URLs, GUIDs or audio.

Rollback requires the previous immutable image, matching pre-upgrade database and media recovery point. Follow [RECOVERY.md](RECOVERY.md) and [V2_IMPLEMENTATION.md](V2_IMPLEMENTATION.md); do not perform a binary-only downgrade after v2 writes.
