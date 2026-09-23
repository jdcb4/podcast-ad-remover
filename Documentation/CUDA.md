# Optional NVIDIA GPU transcription (experimental)

CPU remains the default, using float32 precision. The regular image contains no NVIDIA
CUDA runtime packages. GPU setup downloads optional libraries only when requested and
keeps them in the existing persistent `/data` mount. No separate manager is needed.

This implementation is experimental until the hardware checks below pass. CPU verification
and mocked GPU tests cannot establish driver compatibility, GPU memory use or transcription quality.

## What belongs where

- **Host:** NVIDIA GPU driver and working Docker GPU access. The application cannot install
  the host driver or grant a running container access to a GPU.
- **Container:** the normal application and its existing CTranslate2 transcription engine.
- **Persistent storage:** optional cuBLAS/cuDNN libraries in `/data/runtimes/cuda/<bundle-id>`;
  Whisper models remain in the existing model cache.

Linux x86-64 NVIDIA containers are the initial target, including Unraid and Windows Docker
Desktop with its WSL2 Linux backend. Native Windows Python, macOS, AMD GPUs and ARM64 CUDA
are not covered. Windows users install the NVIDIA Windows driver with WSL support, update
WSL and enable the WSL2 backend; they do not install a Linux display driver inside the container.
See [Docker Desktop GPU prerequisites](https://docs.docker.com/desktop/features/gpu/) and
[NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Start with GPU setup enabled

Use the standalone `docker-compose.cuda.yml` example with a separate test data volume.
Set a persistent random `SESSION_SECRET_KEY` in `.env`, then run:

```bash
docker compose -f docker-compose.cuda.yml up -d
```

The example uses `:experimental-cuda`, requests one NVIDIA GPU and sets `CUDA_SETUP=true`.
Its UI is bound to localhost port 8011. Set `CUDA_IMAGE` to the published immutable candidate
tag when recording test results. The normal CPU Compose file remains usable without GPU access.
GPU reservations follow [Docker's Compose GPU documentation](https://docs.docker.com/compose/how-tos/gpu-support/).

`CUDA_SETUP=true` starts background setup on first application startup; pulling an image alone
does not execute setup. The web UI and CPU processing remain available while setup runs.
The flag does not override a later manually saved CPU choice. A saved GPU choice is revalidated
after restart even if the flag has subsequently been removed. Failed initial setup can be retried
from the UI or on restart while the initial flag still applies.

For Unraid, first configure host NVIDIA support and container GPU access using the appropriate
Unraid/NVIDIA instructions. Select the experimental image and add the `CUDA_SETUP` environment
variable if automatic setup is desired. The supplied template defaults it to false; adding this
variable alone does not configure GPU passthrough. Keep the existing `/data` mapping.

If Docker itself rejects a GPU reservation, the container cannot start: remove the GPU request
to run on CPU or repair host GPU access. Application CPU fallback applies once the container runs.

## Manual setup and settings

Open **Admin > AI Settings > Transcription** and select **Set up / retest GPU**. This tests the
currently saved model, downloads the libraries if necessary and activates GPU only after successful
inference. Refresh the page after success to see saved fields and supported GPU precision choices.

CPU and GPU have separate precision settings. CPU choices come from CTranslate2 on the current
machine; GPU choices come from the isolated GPU probe. Unsupported submitted values are rejected
server-side. GPU model/precision changes are tested before being saved; the prior settings survive
a failed test. Larger models may fail because the GPU lacks memory even when its precision is supported.
GPU thread changes also require retesting. Saving CPU takes precedence over an in-progress setup.

Status shows setup progress, errors and the last execution device/precision. GPU errors disable GPU
use until retest/restart validation and display a warning. The failed transcription is retried once
on CPU/float32; provider analysis and feed publication are not repeated by this retry. Cancellation
stops the worker without starting a CPU retry. CPU fallback can itself fail and uses ordinary job
error handling. A GPU operation or wait has a two-hour limit.

## Download, integrity and recovery

The reviewed bundle in `app/core/cuda_bundle.json` pins CTranslate2 4.8.2, cuBLAS 12.8.4.1
and cuDNN 9.10.2.21. NVIDIA wheels are fetched from fixed PyPI URLs, checked against exact sizes
and SHA-256 hashes, and extracted without executing package installers. About 1.3 GB is downloaded;
setup requires at least 6 GiB free for extraction. Missing Whisper models require additional space
and downloads. NVIDIA package licenses remain in the extracted distributions.

Setup is serialized across processes. Downloads are staged and atomically installed. Interrupted
staging folders are removed on the next setup attempt. Installed file hashes are checked on reuse;
damaged bundles are preserved with an `.incomplete-<timestamp>` suffix before replacement. Old
versioned bundles are retained for rollback, so upgrades may require additional disk space.

Select CPU before maintenance. Retry setup after correcting disk space, connectivity, permissions
or host GPU access. A new image/bundle version triggers validation and, if needed, a new download.
CPU users need no CUDA downloads, and cached libraries are not added to the main Python environment.

Migration `20260924_0022_transcription_runtime` adds settings and a runtime-status table through the
existing backup-aware migration runner. Existing installations remain CPU/float32. Keep the standard
pre-upgrade database/media backup; see [RECOVERY.md](RECOVERY.md). These additions are ignored by
older application versions, which continue their CPU behavior. Optional runtime caches need not be
restored with a database backup: they can be downloaded again. Never delete the whole `/data` mount.

## Hardware qualification before supported release

Record exact image digest, host OS, driver, GPU/VRAM, Docker/toolkit version, model, precision,
setup/download timings, peak VRAM and transcript results for both Linux NVIDIA Docker/Unraid and
Windows Docker Desktop/WSL2. Test on disposable data with a short licensed speech recording:

1. Fresh CPU startup and upgraded CPU data: no runtime download, same transcript behavior.
2. First-start flag and manual setup: libraries persist across container replacement; repeated setup
   reuses them. Test interrupted downloads, low disk space and damaged cached libraries.
3. All offered precisions for small and larger models: compare CPU/GPU text and word timestamps,
   normal and chunked audio, and downstream ad boundaries. Measure time and memory.
4. Missing/old driver, absent GPU access, unsupported precision and out-of-memory: clear diagnosis,
   retained settings, one CPU retry and no duplicate provider calls or publication.
5. Cancellation, worker crash, stalled inference, idle unload, restart and concurrent setup requests:
   no stuck queue, competing model allocations or orphan GPU worker.
6. Save CPU during setup and restart with the flag still set: CPU remains selected. Test GPU
   model changes, cached transcript provenance and explicit reuse of an existing transcript.

Automated coverage lives in `test_transcription_settings.py`, `test_cuda_installer.py`,
`test_cuda_setup.py` and `test_cuda_worker_lifecycle.py`. Real GPU qualification is still required.
