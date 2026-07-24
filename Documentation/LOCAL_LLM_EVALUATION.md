# Local LLM Evaluation

Evaluation date: 2026-07-24.

## Purpose

This evaluation validates the opt-in custom OpenAI-compatible provider and context-aware
ad-detection chunking against local-class and constrained-context models. Gemini remains the
recommended/default provider; the goal is to establish whether smaller models can be used safely,
not to change the cloud-first product direction.

## Safety And Corpus

Production data on JMKtec was read through SQLite read-only mode. Evaluation did not modify the
production database, settings, media, or container.

Three completed production episodes were selected to cover materially different prompt sizes:

| Episode ID | Case | Transcript size | Historical remove time |
|---|---|---:|---:|
| 19667 | Very short promo/message | 527 characters | 19.36 seconds |
| 22272 | Context-boundary episode | 32,892 characters | 147.60 seconds |
| 26476 | Long episode | 88,034 characters | 218.35 seconds |

Historical production report intervals were used as the reference. They are useful regression
evidence but are not a hand-adjudicated ground truth, so interval scores should be interpreted as
relative comparisons rather than final model benchmarks.

All test models were pinned individually through OpenRouter so a cascade could not hide the model
that completed a request. Whitelist mode matched production. Reasons were disabled. Chunked runs
used bounded 30-second overlap and a declared 8K context for the constrained Qwen/Phi comparison.

## Transcript-Level Results

Scores are time-interval precision, recall, and F1 against the historical production remove windows.
“Error” means the complete attempt was rejected; no partial result was accepted as “no ads.”

| Model/configuration | Episode 19667 F1 | Episode 22272 F1 | Episode 26476 F1 | Outcome |
|---|---:|---:|---:|---|
| Gemini 3.1 Flash Lite, single request | 1.00 | 0.81 | 0.82 | Completed all |
| Gemini 3.1 Flash Lite, chunked | 1.00 | 0.67 | 0.90 | Completed 1/5/13 chunks |
| Qwen 2.5 7B, declared 8K, temperature 0 | 0.00 | 0.27 | Error | Long run rejected after chunk 10 returned truncated JSON |
| Phi-4 14B, declared 8K, temperature 0 | 0.00 | 0.48 | 0.11 | Completed 1/5/13 chunks |
| Gemma 3n 4B, 32K | 0.00 | Error | Error | Later chunks reached the output limit and were rejected |
| Gemma 2 27B, native 8K stress | Error | Error | Error | OpenRouter upstream was rate-limited; failure propagated safely |

Representative OpenRouter costs for successful complete runs were low:

- Gemini single-request boundary episode: approximately USD 0.0033.
- Gemini single-request long episode: approximately USD 0.0090.
- Qwen 8K boundary episode: approximately USD 0.0044.
- Phi-4 8K boundary episode: approximately USD 0.0010.
- Phi-4 8K long episode: approximately USD 0.0029.

The functional conclusion is stronger than the quality conclusion:

- Chunk coverage, request caps, global timestamps, strict parsing, error propagation, usage
  aggregation, and safe failure metadata worked.
- Gemini remained materially more accurate on this small corpus.
- Qwen and Phi both missed the short all-promo episode.
- Qwen and Phi are suitable as explicit experimental choices, not recommended defaults.
- Gemma 3n's response behavior is not compatible enough with this whitelist workload at the current
  bounded output allowance.

## End-To-End Results

The approximately 26-minute source episode was re-downloaded and transcribed from audio in separate,
isolated Docker data volumes. Each run then used the custom OpenAI-compatible provider pointed at
OpenRouter, an 8K declared context, five detection chunks, whitelist inversion, FFmpeg cutting,
report generation, and RSS feed generation.

| Model | Status | Chunks | Remove windows/time | Processed duration | Provider usage/cost |
|---|---|---:|---:|---:|---|
| Phi-4 | Completed | 5/5 | 4 / 29.92 seconds | 1,578.63 seconds | 12,160 prompt + 1,082 completion tokens; USD 0.0010 |
| Qwen 2.5 7B | Completed | 5/5 | 4 / 93.76 seconds | 1,514.79 seconds | 14,360 prompt + 431 completion tokens; USD 0.0044 |

For both runs:

- FFprobe accepted the processed MP3.
- The transcript, JSON report, HTML report, podcast feed, and unified feed were created.
- Report metadata identified the custom provider, actual model, complete chunk count, token usage,
  finish reasons, and complete analysis.
- Summary preferences were deliberately enabled on the test subscription, but the episode summary
  remained null and neither `summary.txt` nor `summary.mp3` was created.
- The report recorded `summary_features_disabled=true`.

The large difference between Phi and Qwen cut time reinforces that local model selection materially
affects output quality even when the integration and chunking behavior are correct.

## Recommendation

Ship the feature as advanced, opt-in support while retaining Gemini and cloud cascades as the primary
defaults. The UI should continue to explain context sizing, Docker networking, summary disabling, and
the risk that smaller models may miss promotions or produce materially different cut windows.

Before recommending a specific local model, expand the corpus and manually adjudicate content/ad
boundaries. The current historical-reference comparison is sufficient to validate the feature and to
show that the tested smaller models do not yet match Gemini quality.

