FROM python:3.11-slim

ARG INSTALL_TTS=1

# Keep Python quiet and avoid writing .pyc files into the container layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TTS_ENABLED=${INSTALL_TTS}

# Install runtime system dependencies. FFmpeg is required for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-tts.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_TTS" = "1" ]; then pip install --no-cache-dir -r requirements-tts.txt; fi

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/db /data/podcasts /data/feeds /data/models/piper

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
