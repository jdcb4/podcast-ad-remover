# Real-podcast CUDA comparison — 24 September 2026

## Source and scope

This follow-up to the [RTX A4000 qualification](CUDA_RUNPOD_2026-09-24.md) uses the
complete **LINUX Unplugged #600: Everyone, Everywhere, All at Once**, published by
Jupiter Broadcasting on 2 February 2025. The [episode page](https://linuxunplugged.com/600)
lists a 68-minute, 49-second duration and a [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
The downloaded MP3 measures **4,129.224 seconds**, contains multiple speakers and sponsor
segments, and was not shortened or looped for the main runs.

Source SHA-256: `3dcc9e35486603a8e9bda964ede4bd6f0413a1a9e83ab7c1766b77374ca2132c`.

The unchanged candidate image is `jdcb4/podcast-ad-remover:cuda-17cbaff`, digest
`sha256:b57f844da757d06c25a6ef9a99feb84d64acf0a2f459d4a6b28b1d1a53f7b234`.
Runpod Secure Cloud supplied an **NVIDIA L4, 23,034 MiB**, driver **580.159.04**, with
12 vCPUs and 100 GB allocated RAM at US$0.49/hour. The cheaper A4000 allocation was
unavailable. This adds a second GPU architecture to the earlier runtime qualification.

## Method

Each main run processed the same complete MP3 through the application's `Transcriber`.
The existing audio normalization, 1,200-second chunks, 20-second overlaps, merge rules and
beam size of five were retained. Model downloads and settings-validation probes were outside
the timer. Fresh-process model loading, audio preparation, inference and merging were inside
the timer. All runs used four CPU threads and two FFmpeg threads. This is a controlled
configuration comparison, not a claim about the best possible CPU tuning.
Configurations ran sequentially on one shared cloud host, once each except the explicit
base/float16 repeat. They were not randomized and do not provide statistical confidence
intervals or a general hardware benchmark.

Actual execution metadata was checked against the requested device/precision. Resource
samples were collected once per second. GPU memory values are observed sampled peaks;
process-tree RSS sums include shared mappings and are not unique physical-memory measurements.

Word differences use case-normalized token Levenshtein distance against CPU/base/float32.
They measure **disagreement, not accuracy**: no verified human reference transcript was
available. Differences in punctuation and case are ignored, while wording/tokenization
differences remain. A repeated GPU/base/float16 run checks variation with the same settings.

Three independent 80-second context decodes per selected configuration supplement review
of joins at 19:50, 39:30 and 59:10. Changing context can change recognition, so those excerpts
are diagnostic aids rather than ground truth. No paid LLM ad-classification calls, audio
cuts or feed publications were made; this tests CUDA transcription and its shared chunk path.

## Results

All **nine full runs across eight configurations**, plus nine independent boundary decodes,
completed on their requested device/precision. There were no CUDA failures or CPU fallbacks.
Every main run had valid, ordered segment timestamps and removed its temporary audio.
Those execution checks do not establish transcription accuracy or nonoverlapping segments.

| Configuration | Whole-episode time | Sampled peak GPU MiB | Word difference vs CPU/base/float32 |
|---|---:|---:|---:|
| CPU base float32, 4 threads | 7:21 | 1 | Baseline |
| GPU base float16 | 1:14 | 500 | 3.99% |
| CPU base int8, 4 threads | 6:15 | 0 | 4.85% |
| GPU base float32 | 1:36 | 782 | 0.13% |
| GPU base int8_float16 | 1:26 | 404 | 4.71% |
| GPU small float16 | 2:03 | 1,044 | 9.27% |
| GPU medium float16 | 4:37 | 2,964 | 9.94% |
| GPU medium int8_float16 | 4:14 | 2,100 | 9.29% |
| GPU base float16, repeat | 1:16 | 500 | 3.99% |

Exact timings, additional memory measurements and structural checks are in
[the CSV](CUDA_LONGFORM_2026-09-24.csv). The two base/float16 runs produced **zero normalized
word edits** against one another. GPU/base/float16 was about 6.0 times faster than the
four-thread CPU/base/float32 run on this host. GPU/base/float32 was about 4.6 times faster
and closest in wording to that CPU baseline; closeness is not proof of accuracy.

Medium/int8_float16 reduced sampled GPU memory by about 29% relative to medium/float16
and took about 8% less time. Base/int8_float16 used less GPU memory than base/float16 but
was slower. Quantization is therefore not an automatic speed improvement for every model.
Model-size differences in the table also change recognition behavior and must not be
interpreted as precision-only effects. Initial optional CUDA setup took 28.36 seconds.

## Outputs

The user-facing results package includes an offline HTML report with original-audio playback,
side-by-side searchable transcripts, join selectors, full TXT/JSON/SRT/VTT outputs for every
run, timing/resource CSV/JSON, independent boundary decodes, source and hardware metadata,
test/report scripts and SHA-256 checksums. Audio and generated transcripts remain outside
the source repository. Preserve the source attribution and license when sharing them.

## Interpretation

Successful execution is separate from transcript quality. The real episode reproduces
the shared merge problem found in the synthetic qualification:

- **19:50:** base and small full-episode outputs omit wording recovered by all three
  independent context decodes, including the phrase about Chris having sweet memories.
  Their timestamps leave approximately 3.5–3.8 seconds between segments at that join.
  Medium/float16 retains that wording; medium/int8_float16 repeats nine matching words.
- **39:30:** every configuration repeats wording across the join, with 1.6–4.08 seconds
  of timestamp overlap. CPU/base/float32 and GPU/base/float16 each repeat 19 matching
  normalized words. The independent context decodes contain the sentence once.
- **59:10:** seam deltas are near zero (roughly -0.16 to +0.04 seconds); that alone does
  not certify accurate or complete recognition.

The existing merger keeps segments based on their start falling inside a half-overlap
window. Segments spanning that cut can be discarded or retained alongside an overlapping
segment from the next chunk. This logic is shared by CPU and GPU and was unchanged for
this test. Repairing and retesting it is a prerequisite to claiming clean chunk boundaries;
changing CUDA precision does not fix it.

Intro/outro behavior also varies: base runs emit repeated farewell text near the end,
while small and medium/float16 stop around 67:28. The source audio and ending-window view
are included for listening review. A last timestamp close to the file duration is not,
by itself, a reliable completeness or quality metric.

This test does not qualify Windows/WSL2 or Unraid host setup, all NVIDIA hardware, or the
full ad-removal pipeline. The CUDA candidate remains experimental.

## Cost, cleanup and verification

The allocation lasted approximately 37 minutes. The observed balance fell from
US$9.8466714582 to US$9.5446210692: **US$0.302050389**, approximately **US$0.30** for this
follow-up. This includes setup, model downloads, all full runs and boundary diagnostics.
The balance was checked immediately after deletion; later billing adjustments are possible.

The result archive was downloaded and its SHA-256 matched the server copy before deletion.
Runpod then confirmed the test Pod was absent, no Pods remained and the network-volume
list was empty. Its attached disposable volume was removed with the Pod. The cleanup
watchdog observed deletion and exited. Dev and production were untouched.

Output verification checked nine complete transcripts, nine boundary decodes, actual
device/precision metadata, subtitle cue counts, source/artifact checksums, ZIP CRCs and
the report's embedded JSON/JavaScript. Browser checks covered the comparison selectors,
search and join windows. `npm run verify` passed with 597 tests, 5 skipped, CSS compilation
and clean dependency audits. An initial run hit an unrelated 15-second Node DOM-test timeout;
the unchanged suite passed on rerun. Only documentation and a metrics CSV were added to Git.
