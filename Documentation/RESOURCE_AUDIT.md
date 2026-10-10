# Resource use and measurement (V2)

Local transcription and audio remain the main resource costs. PAR uses faster-whisper/CTranslate2, FFmpeg, local SQLite/media storage and network calls to the selected analysis/optional speech endpoints. Piper and its exclusive phonemizer dependency are removed; ONNX Runtime, NumPy and FFmpeg remain shared requirements of the transcription/audio stack. Do not equate Piper removal with removing all inference dependencies.

Existing downloaded Piper voices under `/data/models` are not deleted during upgrade. The historical assessment measured roughly 46 MB of unpacked exclusive packages and 114 MB for one downloaded voice; those package estimates do not isolate the image change measured below. No idle-RAM or performance saving has been measured here.

## Release image comparison — 11 October 2026

Measured the published Linux/amd64 images for 1.16.0 and the tested V2 application
revision `bbced603103a159c35565621eaf3e98dc46bbe79`, before the release-documentation
changes. Decimal MB/GB are used below; these are not GiB/MiB.

| Measurement | 1.16.0 | V2 `dev-bbced60` | Reduction |
| --- | ---: | ---: | ---: |
| Uncompressed layers (`docker history --human=false`, summed) | 1,385,287,680 bytes (1.385 GB) | 1,337,516,032 bytes (1.338 GB) | 47,771,648 bytes (47.8 MB), 3.4% |
| Compressed registry layers (download payload) | 474,451,231 bytes (474.5 MB) | 449,155,654 bytes (449.2 MB) | 25,295,577 bytes (25.3 MB), 5.3% |
| Docker Desktop reported storage (`docker image inspect .Size`) | 1,859,750,071 bytes (1.860 GB) | 1,786,682,589 bytes (1.787 GB) | 73,067,482 bytes (73.1 MB), 3.9% |

Immutable image digests:

- 1.16.0: `sha256:939bea0a4f6ac4e918e48e952e61bcd82f8a4611103bdde7d6b0222a9e73fc16`.
- V2 tested build: `sha256:b5bc458c9d93c4b3cf84edf91ea4ab6c010c48df50843bdd1b929b5c4972edaa`.

The compressed values sum `layers[].size` from the Linux/amd64 manifest retrieved
with `docker buildx imagetools inspect --raw`, selecting that platform from the
image index first. They exclude small manifests/configs and any cache reuse.
The uncompressed values sum byte sizes from `docker history --human=false`.
Docker Desktop's containerd image store reports compressed plus uncompressed
content and metadata in `.Size` for these images; calling that value unpacked
size would double-count the compressed payload. None of these values is exclusive
disk use: shared layers and the storage implementation affect actual host storage.
Downloaded Whisper/Piper voices, optional CUDA runtime, database and media under
`/data` or `/media` are excluded from both measurements. No CUDA libraries are
bundled in the normal image. These totals cover all changes since 1.16.0 and do
not isolate Piper removal; no RAM saving was measured. Re-measure the final
versioned candidate before publication.

### Why removing Piper saves about 48 MB, not hundreds

Network-disabled disposable containers confirmed Piper and its phonemizer are
absent from V2. In1.16.0 their installed distribution files total46,225,480bytes:
Piper180,571bytes and its phonemizer46,044,909bytes. The Python installation layer
shrinks from679,301,120 to631,336,960bytes (48.0MB); the net uncompressed-layer
reduction is47.8MB after other source/layer changes. This is consistent with
removing the two exclusive packages, rather than leaving Piper hidden in a layer.

The larger components remain: FFmpeg's system installation layer is about457MB,
Deno about95.6MB, and the Python installation layer about631MB. Installed package
files include CTranslate2 about140MB, PyAV about107MB, NumPy about71MB and ONNX
Runtime about67MB. These package totals are descriptive, not additive layer
accounting. Both images use faster-whisper1.2.1, whose installed metadata declares
ONNX Runtime as a dependency; removing Piper cannot also remove that shared runtime.
The default Piper voice was about114MB downloaded separately under `/data/models`,
not bundled. Upgrade retains old voices rather than deleting user files.

## Current controls

- System exposes processing concurrency and Whisper/FFmpeg thread limits. One job can use several native threads; tune for the actual host rather than equating a job count with CPU cores.
- Model choice, CPU precision and optional GPU setup are in Transcription. CPU/float32 remains the default. [CUDA.md](CUDA.md) records experimental host requirements and qualification limits.
- `unload_whisper_after_job` releases the local model when the queue empties, trading lower idle RAM for a later reload. Use actual logs/measurements before choosing limits.
- Scratch reservations, free-space checks, maximum download size and provider request/time limits bound work. They do not reserve disk space against unrelated host writers.
- Automatic retention keeps a count of completed automatic episodes; manual retention is time-based. Media, backups and model/runtime caches all live under `/data` and need capacity planning.
- API-only speech moves synthesis to the selected endpoint. It can incur fees and network latency; it does not remove local transcription or FFmpeg costs.
- Both amd64 and experimental ARM64 images use API-only speech. There is no Piper/no-TTS image variant or `INSTALL_TTS` switch. whisper.cpp is a deferred evaluation, not a resource optimization already shipped.

## Measure the exact installation

Use read-only checks on the authorized host, recording the image revision, model/precision, CPU/GPU, active-job count and episode duration with the result:

```bash
docker stats <container-name> --no-stream
docker image inspect <immutable-image-tag> --format '{{.Size}}'
docker exec <container-name> sh -lc 'du -h -d 2 /data'
```

Compare idle and active workload separately. Queue memory reporting prefers cgroup limits; provider token/call totals are usage records, not invoice estimates. The same episode and settings are needed for meaningful before/after performance comparisons. Do not run paid inference, change deployment settings or remove cached files merely to prepare documentation.

## Historical evidence

[The preserved June–September resource audit](history/RESOURCE_AUDIT_2026-06_to_09.md) contains original image/host measurements, command recipes and the Piper-removal package assessment. Its old image variants, dependency choices, active-host claims and recommendations describe that time. The [RTX A4000](CUDA_RUNPOD_2026-09-24.md) and [L4 long-form](CUDA_LONGFORM_2026-09-24.md) reports remain bounded hardware evidence; they do not qualify every host or a future transcription engine.
