# Complete Timeline processing

Complete Timeline is an opt-in alternative to Legacy ad detection. It classifies the whole
episode once, then applies the podcast's removal choices in code. Existing installations,
podcasts, custom Legacy prompts and queued jobs keep their previous behaviour on upgrade.

## Enable it for a podcast

1. Open the podcast's **Processing Settings**.
2. Leave **Use global settings** off for Processing workflow and choose **Complete Timeline**.
3. Choose the categories to remove under Content Removal. Disable that group's inheritance
   if this podcast needs different choices from the global defaults.
4. Set the short-island threshold: **10 seconds** by default, **0** to disable it.
5. Review custom instructions, then save. Changes apply to newly queued jobs. Published audio
   and jobs already in the queue are not automatically reprocessed or converted.

Owners and admins can change a podcast's settings. The podcast, audio and replacement feed are
shared by everyone subscribed to that podcast; these are not individual listener preferences.
Use the existing reprocess action when you want to replace an already completed episode.

Admins can choose Complete Timeline as the default in **Global Subscription Settings**. New
podcasts inherit that default. Podcasts that existed before migration remain explicitly on
Legacy until an owner/admin changes their workflow or enables workflow inheritance. Changing
the default does not opt those existing podcasts in.

## Categories and definitions

Admins edit the seven definitions in **AI Prompt Rules → Complete Timeline**. Blank values or
**Use default definition** select the built-in definition when saved. Category identifiers and
the coverage contract are fixed; definitions refine what belongs in each category.

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

Legacy whitelist mode remains available for Legacy jobs. Complete Timeline uses category choices
and explicit gaps, so the global whitelist switch has no effect on this workflow.

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

| Mode | Behaviour |
| --- | --- |
| Auto (default) | Request native JSON Schema. Only an explicit unsupported-format response permits a retry as JSON text, which is still strictly validated. OpenAI-compatible capability results are cached per endpoint/provider/model for one hour. |
| Require schema | Keep schema enforcement throughout the selected provider's cascade. Fail if none of its configured models can provide it. |
| JSON compatibility | Send the schema as instructions, without requiring native schema support. Still validate every label, range and complete coverage. |

Native requests use OpenAI-compatible `response_format.json_schema` for OpenAI, Gemini,
OpenRouter and custom endpoints. OpenRouter requests require endpoints that accept the schema
parameter. Anthropic uses `output_config.format`. See the official
[OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs),
[Gemini](https://ai.google.dev/gemini-api/docs/openai#structured-output),
[OpenRouter](https://openrouter.ai/docs/guides/features/structured-outputs), and
[Anthropic](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) contracts.

Capability is determined by actual requests, not guesses from model names or extra paid probes.
Outages, authentication errors, invalid schemas, truncation and refusals do not trigger a format
downgrade. Failures use only the configured cascade within the selected provider; Gemini does
not fall back to OpenRouter in production. Temperature and reasoning parameters remain omitted
so provider defaults apply. Existing timeouts, request budgets and the Anthropic output-token
limit still apply. An incomplete classification gets at most one application-level repair;
all model/key/format attempts share the durable job request budget. Invalid analysis cannot
be treated as a successful no-cut result.

## Migration, cached results and rollback

Migration `20260910_0016_complete_timeline_opt_in` is additive. It adds workflow, non-speech,
threshold and prompt settings plus `jobs.processing_snapshot`. Existing podcasts get
`processing_workflow=legacy`, `inherit_processing_workflow=0`. Existing jobs retain a NULL
snapshot, explicitly interpreted as Legacy even if the global default later changes.
Legacy prompt overrides and the whitelist flag are not rewritten.

New Complete Timeline jobs freeze the effective prompt, resolved model cascade, output mode,
schema version and removal policy when queued. Credentials are loaded at execution time and
are never stored in the snapshot. Automatic retries and re-enqueuing an active job retain its
snapshot. Description/TTS/retention and other existing features continue resolving as before;
they are not part of this new classification snapshot. Unsupported snapshot/schema versions
fail with an actionable error rather than silently using different rules.

Classification caches require matching source SHA-256, transcript, measured duration, prompt,
schema and provider configuration. Changing only removal choices or the island threshold can
reuse matching classification on reprocess. A changed download, including dynamically inserted
ads, invalidates reuse. Previous published audio stays available until its replacement completes.
Changing workflow does not move or rewrite existing reports, feed URLs, GUIDs or audio.

To opt out, select Legacy for newly queued jobs. Keep the upgraded application: switching modes
does not require a database downgrade. To roll back the application itself, use the previous
immutable image with the pre-upgrade database snapshot and matching media backup, following
[RECOVERY.md](RECOVERY.md). Do not point an older binary at the upgraded database: it cannot
honour Complete Timeline job snapshots. Startup creates a WAL-safe, integrity-checked snapshot
before migration, and the migration dry-run command can rehearse the upgrade first.
