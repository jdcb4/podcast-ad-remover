FROM denoland/deno:2.9.5 AS deno

FROM python:3.11-slim

ARG INSTALL_TTS=1

# Keep Python quiet and avoid writing .pyc files into the container layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TTS_ENABLED=${INSTALL_TTS} \
    TORCH_CUDA_ARCH_LIST="" \
    CUDA_VISIBLE_DEVICES=""

# Install runtime system dependencies. FFmpeg is required for audio processing.
# Added 'patchelf' as a modern replacement for 'execstack' to modify shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp uses Deno for YouTube's JavaScript challenges. Both yt-dlp and its
# matching EJS scripts are explicitly pinned, so no runtime component download
# or self-update is needed.
COPY --from=deno /usr/bin/deno /usr/local/bin/deno

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-tts.txt constraints-cpu.txt ./

# Install CPU-only PyTorch and all requirements using constraints to prevent CUDA dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -c constraints-cpu.txt torch torchvision torchaudio
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -c constraints-cpu.txt -r requirements.txt \
    && if [ "$INSTALL_TTS" = "1" ]; then pip install --no-cache-dir -r requirements-tts.txt; fi

# FIX: Patch whisperx and pyannote to use 'token' instead of the deprecated 'use_auth_token' to support modern huggingface_hub
RUN find /usr/local/lib/python3.11/site-packages/whisperx /usr/local/lib/python3.11/site-packages/pyannote -type f -name "*.py" -exec sed -i 's/use_auth_token/token/g' {} + || true

# FIX: Find any installed ctranslate2 shared libraries and clear their executable stack flag
RUN find /usr/local -name "libctranslate2*.so*" -exec patchelf --clear-execstack {} \;

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/db /data/podcasts /data/feeds /data/models/piper

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
