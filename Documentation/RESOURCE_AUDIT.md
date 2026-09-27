# Resource use and measurement (V2)

Local transcription and audio remain the main resource costs. PAR uses faster-whisper/CTranslate2, FFmpeg, local SQLite/media storage and network calls to the selected analysis/optional speech endpoints. Piper and its exclusive phonemizer dependency are removed; ONNX Runtime, NumPy and FFmpeg remain shared requirements of the transcription/audio stack. Do not equate Piper removal with removing all inference dependencies.

Existing downloaded Piper voices under `/data/models` are not deleted during upgrade. The historical assessment measured roughly 46 MB of unpacked exclusive packages and 114 MB for one downloaded voice, but these are not measured V2 image or RAM savings. No new image-size, idle-RAM or performance claim is made by this documentation update.

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
