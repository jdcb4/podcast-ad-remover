# Local LLM Evaluation

Last updated: 2026-07-24.

## Purpose

This is the living evaluation record for the opt-in custom OpenAI-compatible provider and
context-aware ad-detection chunking. It measures whether local-class models can complete the
production whitelist workflow safely and how their detected removal windows compare with the
standard cloud model.

Gemini remains the recommended and default provider. Local OpenAI-compatible endpoints are an
advanced option; the benchmark does not change the product's cloud-first direction.

**Outcome:** transcript chunking for smaller/local models is concluded as an unsuccessful product
experiment. The implementation is retained on the experimental branch for inspection, but there is
no current intention to continue developing it or merge it into the production application. The
separately useful configurable OpenAI-compatible endpoint is being retained without chunking.

The detailed, self-contained comparison is available in
[`LOCAL_LLM_EVALUATION_REPORT.html`](LOCAL_LLM_EVALUATION_REPORT.html). It includes run and quality
pass/fail status, per-episode timelines, every detected removal window, overlap with each reference
window, failures, token use, cost, and latency. The machine-readable counterpart is
[`LOCAL_LLM_EVALUATION_RESULTS.json`](LOCAL_LLM_EVALUATION_RESULTS.json).

## Safety And Corpus

Production data on JMKtec was read through SQLite read-only mode. Evaluation did not modify the
production database, settings, media, or container. The working corpus remained outside the
repository and was removed after the run.

Three completed production episodes cover materially different prompt sizes:

| Episode ID | Case | Duration | Transcript segments | Historical remove time |
|---|---|---:|---:|---:|
| 19667 | Very short promo/message | 19 seconds | 7 | 19.36 seconds |
| 22272 | Context-boundary episode | 26:48 | 314 | 147.60 seconds |
| 26476 | Long episode | 1:11:52 | 943 | 218.35 seconds |

Historical production report intervals are regression references, not hand-adjudicated ground
truth. They can compare model behavior and expose regressions, but they cannot establish that every
reference boundary is objectively correct.

Committed result artifacts deliberately exclude:

- Transcript text and episode titles.
- Feed, media, and production file paths.
- API keys and other credentials.
- Raw model responses.

They retain only episode IDs, case labels, timing intervals, aggregate transcript sizes, metrics,
safe error diagnostics, token usage, estimated cost, and latency.

## Current Benchmark

### Method

The 2026-07-24 matrix used the OpenRouter catalogue and the `OPENROUTER_API_KEY` from the local
`.env`. Models were pinned individually, so a fallback cascade could not hide which model handled a
request.

All local/open-model candidates used:

- Whitelist detection, matching production behavior.
- An 8,192-token declared context.
- Context-aware chunks with 30 seconds of overlap and a 32-chunk maximum.
- Temperature 0 and no per-segment reasons.
- The application's strict complete-failure behavior: a malformed chunk invalidates the whole
  episode result rather than being accepted as “no ads.”
- Summary generation disabled because chunking was enabled.

Gemini 3.1 Flash Lite was tested twice: its normal single-request cloud behavior and the same 8K
chunking constraints used for the local-class matrix.

Quality passes require a complete run and time-interval F1 of at least 0.70 against the historical
remove windows. Execution pass and quality pass are reported separately.

### Results

| Model/configuration | Size | Short F1 | Boundary F1 | Long F1 | Execution | Quality passes | Estimated cost | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Flash Lite, standard | Undisclosed | **1.000** | 0.652 | **0.907** | 3/3 | 2/3 | $0.0122 | 7.2s |
| Gemini 3.1 Flash Lite, 8K chunked | Undisclosed | **1.000** | **0.793** | **0.890** | 3/3 | **3/3** | $0.0160 | 27.4s |
| Gemma 3n E4B, 8K | 4B effective | 0.000 | Error | Error | 1/3 | 0/3 | $0.0013 | 119.8s |
| Qwen 2.5 7B Instruct, 8K | 7B | 0.000 | 0.255 | 0.323 | 3/3 | 0/3 | $0.0026 | 67.7s |
| Llama 3.1 8B Instruct, 8K | 8B | Error | Error | Error | 0/3 | 0/3 | $0.0015 | 23.4s |
| Phi-4, 8K | 14B | 0.000 | 0.482 | Error | 2/3 | 0/3 | $0.0027 | 58.1s |
| Mistral Small 3.1, 8K | 24B | Error | Error | Error | 0/3 | 0/3 | $0.0000 | 6.2s |
| Gemma 3, 8K | 27B | 0.000 | 0.377 | 0.573 | 3/3 | 0/3 | $0.0075 | 186.1s |
| Llama 3.3 Instruct, 8K | 70B | 0.000 | Error | **0.764** | 2/3 | 1/3 | $0.0070 | 42.8s |
| Qwen 2.5 Instruct, 8K | 72B | 0.000 | 0.480 | 0.443 | 3/3 | 0/3 | $0.0212 | 55.8s |

Across the complete matrix:

- 20 of 30 episode runs completed.
- 6 of 30 episode/model comparisons passed the quality gate.
- Chunked Gemini was the only configuration to pass all three quality gates.
- Llama 3.3 70B was the only local/open-model candidate to pass a larger episode, scoring 0.764
  on the long case.
- The total estimated OpenRouter cost was USD 0.0720.

### Interpretation

Model size alone did not predict reliable application behavior. Qwen 7B, Gemma 27B, and Qwen 72B
completed every episode but remained below the quality threshold. Llama 8B and Mistral 24B failed
structured response parsing on every case. Llama 70B produced the strongest open-model long-episode
result but still missed the short promo and failed the final boundary chunk.

The short case is a useful warning for whitelist mode: every local/open-model candidate that
completed it returned no removal window, while both Gemini configurations matched the 19.36-second
reference. Larger parameter counts did not fix that failure.

Chunking itself was not merely an accommodation for small context windows. On this run, the
chunked Gemini control improved the boundary case from 0.652 to 0.793 while retaining a strong long
score. It cost more requests, time, and tokens, and it introduced a 0.56-second extra removal in the
long case. The HTML report makes those boundary differences visible.

The functional safety behavior held:

- Complete chunk counts and globally aligned timestamps were recorded.
- Malformed or truncated JSON propagated as a failure with the failed chunk number.
- Partial output was never accepted as a complete “no ads” result.
- Provider usage and finish reasons were aggregated.
- No summary output was requested while chunking was enabled.

The current recommendation remains:

- Keep Gemini/cloud providers as the default.
- Present local endpoints and small models as advanced, explicitly experimental options.
- Do not recommend one local model based on this three-episode corpus.
- Manually adjudicate a larger, more diverse corpus before defining a production acceptance gate.
- Prefer models with reliable schema-constrained output support when local providers expose it.

## Earlier End-To-End Processing Validation

Before the expanded transcript matrix, the approximately 26-minute source episode was downloaded
and transcribed in separate, isolated Docker data volumes. Each run then used the custom
OpenAI-compatible provider through OpenRouter, an 8K declared context, five detection chunks,
whitelist inversion, FFmpeg cutting, report generation, and RSS feed generation.

| Model | Status | Chunks | Remove windows/time | Processed duration | Provider usage/cost |
|---|---|---:|---:|---:|---|
| Phi-4 | Completed | 5/5 | 4 / 29.92 seconds | 1,578.63 seconds | 12,160 prompt + 1,082 completion tokens; $0.0010 |
| Qwen 2.5 7B | Completed | 5/5 | 4 / 93.76 seconds | 1,514.79 seconds | 14,360 prompt + 431 completion tokens; $0.0044 |

For both runs, FFprobe accepted the processed MP3; transcript, JSON/HTML report, and feeds were
created; report metadata identified the exact model and complete chunk count; and summary artifacts
remained absent despite deliberately enabled summary preferences. This validates the end-to-end
integration, not the detection quality of either model.

## Reproducing And Updating The Report

The evaluator is [`scripts/evaluate_local_llms.py`](../scripts/evaluate_local_llms.py), and the
versioned model configuration is
[`scripts/local_llm_model_matrix.json`](../scripts/local_llm_model_matrix.json).

The corpus JSON must remain outside the repository. Its top level contains `episodes`; each episode
contains:

- `id`, `case`, and `duration`.
- A normal application `transcript` object with `segments`.
- `reference_segments` containing `start`, `end`, and `label`.

Run the complete matrix from PowerShell:

```powershell
python scripts/evaluate_local_llms.py `
  --corpus "$env:TEMP\podcast_llm_eval_corpus.json"
```

Add or rerun selected configurations while preserving other results:

```powershell
python scripts/evaluate_local_llms.py `
  --corpus "$env:TEMP\podcast_llm_eval_corpus.json" `
  --resume `
  --only "llama-3.3-70b-instruct-8k,qwen-2.5-72b-instruct-8k"
```

The evaluator rewrites the sanitized JSON and self-contained HTML after each model configuration.
To add future results:

1. Confirm the live model ID, native context, and pricing with the provider.
2. Add a uniquely named run to `scripts/local_llm_model_matrix.json`.
3. Run it with `--resume --only <run-id>`.
4. Review the HTML timelines and exact removal windows, not only average F1.
5. Update the current-results table and interpretation in this document.
6. Confirm that transcripts, titles, paths, credentials, and raw responses remain absent before
   committing the generated artifacts.

Provider availability and pricing change over time. The matrix records the catalogue check date;
costs are estimates based on that snapshot and token usage returned by OpenRouter.
