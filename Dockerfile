FROM python:3.11-slim

# Install system dependencies (FFmpeg is required)
RUN apt-get update && apt-get install -y ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch (Save ~3GB)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/db /data/downloads /data/transcripts /data/models /public/feeds /public/audio /data/models/piper

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
