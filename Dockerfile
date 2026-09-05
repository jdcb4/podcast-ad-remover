FROM denoland/deno:2.9.5@sha256:b429777c3dcff34a6488f365a1537db1640b2d48379b60f5e6206be034472463 AS deno

FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ARG INSTALL_TTS=1

# Keep Python quiet and avoid writing .pyc files into the container layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TTS_ENABLED=${INSTALL_TTS}

# Install runtime system dependencies. FFmpeg is required for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp uses Deno for YouTube's JavaScript challenges. Both yt-dlp and its
# matching EJS scripts are explicitly pinned, so no runtime component download
# or self-update is needed.
COPY --from=deno /usr/bin/deno /usr/local/bin/deno

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-tts.txt requirements-build.txt constraints.txt ./
RUN python -m pip install --no-cache-dir --upgrade -r requirements-build.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_TTS" = "1" ]; then pip install --no-cache-dir -r requirements-tts.txt; fi

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/db /data/podcasts /data/feeds /data/models/piper

# Expose port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
