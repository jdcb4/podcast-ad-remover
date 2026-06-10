# Resource Audit

Date: 2026-06-10
Branch: `audit-work`

## Summary

The Docker image was larger than necessary. The previous image was about 3.25 GB. A local optimized build is about 1.6 GB, mainly by removing the unused PyTorch install and keeping local development artifacts out of the image.

The app itself is small. The heavy parts are:

- FFmpeg and its Debian runtime libraries.
- `faster-whisper`, CTranslate2, PyAV, tokenizers, and NumPy.
- Piper TTS, ONNX Runtime, and phonemizer libraries.
- Downloaded Whisper and Piper models under `/data/models`.
- Processed podcast audio under `/data/podcasts`.

Idle memory in a smoke container was about 200 MB before loading Whisper or Piper models. Processing memory and CPU will be much higher while transcribing, detecting ads, cutting audio, or generating TTS.

## Findings

### Image Size

Measured locally:

```text
jdcb4/podcast-ad-remover:1.3.1     3.25 GB
podcast-ad-remover:resource-audit  1.60 GB
```

The old image included:

- Explicit CPU PyTorch install: about 1.03 GB layer.
- Full `torch` package in site-packages: about 754 MB.
- `torchvision` and `torchaudio`, neither imported by the application.
- Local `node_modules` copied into `/app`: about 23 MB.
- Git and wget installed in the runtime image.

The optimized image keeps:

- FFmpeg.
- `faster-whisper` and CTranslate2.
- Piper TTS and ONNX Runtime.
- AI provider SDKs.
- FastAPI/Jinja/SQLite runtime dependencies.

### Runtime Disk Use

Disk use is dominated by persistent `/data`, not the application code:

- `/data/db/podcasts.db`: SQLite metadata.
- `/data/podcasts`: processed audio, transcripts, reports, generated summaries.
- `/data/feeds`: generated RSS files.
- `/data/models`: downloaded Whisper and Piper models.
- `/data/backups`: migration backups.

The largest ongoing growth risk is retained processed audio. The second largest is model storage, especially if users switch between multiple Whisper model sizes.

### Runtime CPU and Memory

Expected hotspots:

- Whisper/faster-whisper transcription.
- FFmpeg decoding, cutting, and concatenation.
- LLM provider requests, mostly waiting on network/API rather than CPU.
- Piper TTS when title intros or audio summaries are enabled.

The app already has a `concurrent_downloads` setting, but that is really processing concurrency. One concurrent job can still fan out into FFmpeg and Whisper internal threads. Small machines should usually use `concurrent_downloads=1`.

## Changes Made

- Removed explicit `torch`, `torchvision`, and `torchaudio` install from `Dockerfile`.
- Changed apt install to `--no-install-recommends` and removed runtime `git`/`wget`.
- Added `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1`.
- Added `.dockerignore` entries for `node_modules`, tests, local agent files, and virtualenvs.
- Moved pytest from production `requirements.txt` to `requirements-dev.txt`.

## Recommended Next Steps

1. Add optional image variants:
   - `standard`: current full transcription plus Piper support.
   - `no-tts`: remove Piper and ONNX Runtime for users who do not use audio summaries or title intros.
   - potentially `external-transcription`: for users who do not want local Whisper.
2. Add UI visibility for `/data` storage by category:
   - processed audio
   - transcripts/reports
   - feeds
   - models
   - backups
3. Add controls for:
   - minimum free disk space
   - maximum episode download size
   - Whisper thread count
   - FFmpeg thread count
4. Make cleanup safer and more visible:
   - show retained episode count per podcast
   - show estimated disk reclaimed before deleting files
   - expose stale `.work` cleanup after staged processing is introduced

## Commands For A Real Container Audit

Run these on the PC hosting the live Docker container.

Find the container name:

```bash
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
```

Check image size:

```bash
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep podcast
```

Check live CPU, memory, and block I/O:

```bash
docker stats <container-name> --no-stream
```

Check persistent data size:

```bash
docker exec <container-name> sh -lc 'du -h -d 2 /data | sort -h'
```

Check model sizes:

```bash
docker exec <container-name> sh -lc 'du -h -d 2 /data/models | sort -h'
```

Check podcast storage sizes:

```bash
docker exec <container-name> sh -lc 'du -h -d 2 /data/podcasts | sort -h | tail -30'
```

Check database and backup sizes:

```bash
docker exec <container-name> sh -lc 'du -h /data/db /data/backups 2>/dev/null || true'
```

Check idle process memory inside the container:

```bash
docker exec <container-name> sh -lc 'ps -o pid,ppid,rss,pcpu,comm,args | sort -k3 -n'
```

If a job is actively processing, run this several times a minute:

```bash
docker stats <container-name> --no-stream
docker exec <container-name> sh -lc 'du -h -d 2 /data | sort -h | tail -30'
```

Send back the outputs if deeper tuning is needed.
