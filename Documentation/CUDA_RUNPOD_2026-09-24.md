# Runpod CUDA qualification — 24 September 2026

## Scope and environment

Tested the published application image `jdcb4/podcast-ad-remover:cuda-17cbaff`, revision
`17cbaff6e6abe0fa69221abd566ac8f8d3a698b7`, pinned to registry digest
`sha256:b57f844da757d06c25a6ef9a99feb84d64acf0a2f459d4a6b28b1d1a53f7b234`.

The disposable Runpod Secure Cloud Pod used one NVIDIA RTX A4000 (16,376 MiB), driver
580.159.04, Linux x86-64, CTranslate2 4.8.2 and the application's pinned CUDA 12 libraries
(cuBLAS 12.8.4.1, cuDNN 9.10.2.21). The driver advertised CUDA 13.0 support; the application
successfully used its CUDA 12 runtime. GPU rental was US$0.25/hour, with a 15 GB container
disk and a 30 GB persistent `/data` volume. The authorized budget was US$10.

The application image and dependencies were not replaced with a preconfigured PyTorch/CUDA
image. OpenSSH was installed solely for test access; the web server listened on loopback.
Scheduled processing was disabled, and no production data or AI-provider credentials were
copied. No paid LLM calls were made. CPU transcription used four threads and FFmpeg two.

The speech fixture was the `jfk.flac` sample from
[faster-whisper's tests](https://github.com/SYSTRAN/faster-whisper/tree/master/tests/data).
A repeated 1,208-second version exercised the real 1,200-second chunk threshold and overlap
merge. This is a functional regression fixture, not a diverse podcast accuracy benchmark.

## Verified behavior

| Check | Result |
|---|---|
| Fresh CPU startup | CPU/float32; no installed NVIDIA Python packages or downloaded runtime |
| First-start `CUDA_SETUP=true` | Downloaded 1,301,105,388 bytes, verified hashes, loaded the model and completed real inference; GPU activated in 52.06 seconds |
| Manual setup/retest | Activated GPU and reused verified cached libraries without replacement |
| Supported precision choices | Base and medium models completed actual speech inference for all seven advertised modes: `bfloat16`, `float16`, `float32`, `int8`, `int8_bfloat16`, `int8_float16`, `int8_float32` |
| Invalid CPU/GPU precision | HTTP 400; saved transcription settings unchanged |
| Real GPU out of memory | Reserved 15,741 MiB using the CUDA driver API; native GPU inference failed with OOM and exactly one CPU/float32 transcription succeeded |
| Failed GPU model change | Under the same real memory pressure, a proposed larger model failed validation and prior saved settings were retained |
| Worker crash | Killed the real GPU subprocess; exactly one GPU attempt and one CPU retry |
| Cancellation | Real worker stopped; cancellation propagated with zero CPU retries |
| Hung worker | Suspended the real worker; shortened the test deadline to four seconds, verified termination and one CPU retry (the production limit remains two hours) |
| GPU memory release | Saving CPU closed the idle cached GPU worker |
| Pod restart | Persistent libraries and the saved CPU choice survived an actual Runpod Pod restart; setting `CUDA_SETUP=true` did not override that choice |
| Saved GPU with flag off | Application restart revalidated the saved GPU configuration with `CUDA_SETUP=false` |
| Concurrent setup | Second setup request returned HTTP 409; a later CPU save took precedence over the in-progress GPU test |
| Corrupt runtime recovery | Flipped a byte in cached cuBLAS; the installer detected it, preserved the damaged directory, downloaded a verified replacement and passed real inference again |
| Low disk space | An isolated data directory on a real 64 MiB filesystem was rejected before any wheel download |
| Interrupted download | Killed the real installer after 1 MiB downloaded; retry removed abandoned staging and installed a hash-verified bundle in 400.06 seconds |
| Cached worker configuration changes | Changed base/float16 to base/float32 to small/float16; each change replaced the worker and actual inference reported the new model/precision |
| Unrelated settings | Saving the voice-settings section retained all transcription preferences |

The web health endpoint remained available during setup and after injected failures. GPU and
CPU execution metadata recorded the actual device, precision and model. GPU failure retry
counts were observed around real inference; provider/publication stages were deliberately
not invoked in this cloud test.

## Interpretation and remaining qualification

Every base-model precision produced the same normalized words as CPU/float32 on the short
fixture, with punctuation differences. Short-clip timings varied with model loading, caching
and shared cloud resources; they should not be treated as a general benchmark.
Base-model GPU memory samples ranged from 299 to 565 MiB across the tested precisions/runs.
These are sampled observations, not guaranteed peak-memory requirements for larger models or audio.

The repeated 1,208-second fixture completed in 67.413 seconds on GPU/base/float16 and
312.364 seconds on CPU/base/float32. However, outputs were not equivalent: GPU returned
133 segments through 1,206.3 seconds; CPU returned 164 through 1,199.12 seconds. The CPU
result failed the harness's end-of-audio coverage assertion. Do not treat these timings as
an accuracy-equivalent speed comparison or mark the complete quality matrix passed.

The shared chunk merger retains segments based on their **start** falling within its half-overlap
window. A segment starting before the cut can therefore be discarded even when it extends
past it. Reprocessing the final 28-second chunk directly on CPU produced one segment from
0 to 28 seconds: after adding the 1,180-second offset, the existing merger rejects it because
its start precedes the 1,190-second merge boundary. Its decoded text also contained repetition.
That implementation is unchanged by this CUDA branch. Boundary coverage and real podcast/ad-cut comparisons
remain an explicit follow-up before supported release.

A separate fixture placed one speech clip after 1,197 seconds of silence. Both devices
returned speech after 1,200 seconds and cleaned up temporary files, confirming that both
can execute the second chunk. However, both also hallucinated words during silence and
this narrow check did not establish complete speech retention. It does not cancel the
repeated-fixture failure or qualify transcript/ad-boundary accuracy.

The test harness initially used a 180-second corruption-recovery wait. A slow download
exceeded that wait; continued observation confirmed the same application setup completed
without another request. The application's download deadline is one hour. A request sent
before a previous setup released its lock correctly returned 409; subsequent tests waited
for setup completion.

This evidence qualifies the tested Linux container/GPU combination. It does not establish
Windows Docker Desktop/WSL2 or Unraid host setup, old-driver compatibility, all NVIDIA
architectures, or real-world ad-boundary accuracy. Those remain part of the broader matrix in
[CUDA.md](CUDA.md). The candidate remains experimental; no production promotion is implied.

## Cost, cleanup and local verification

The account balance fell from US$10.00 to US$9.8466714582: approximately **US$0.15** spent
over a 36-minute allocation, including the restart and recovery downloads. This is the
observed balance immediately after deletion; later billing adjustments are possible.

Reports, transcripts and GPU samples were exported locally before deleting the test Pod.
The Runpod API then confirmed the Pod was absent, the Pod list was empty and no network
volumes remained. The cleanup watchdog also exited after observing deletion. No cloud
test resource was retained. No Dev or production deployment was changed.

`npm run verify`, using the project's `.venv`, passed: **597 tests, 5 skipped**, CSS build
and both dependency audits clean. An initial invocation accidentally used the system
Python environment and failed on missing/mismatched dependencies; the project environment
passed without code changes. This qualification update changes documentation only.
