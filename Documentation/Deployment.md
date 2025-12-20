# Deployment

## Docker Compose (Recommended)

```yaml
## Docker Compose (Recommended)

```yaml
services:
  app:
    build: .
    image: podcast-ad-remover
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data      # App data (DB, downloads, transcripts, feeds, audio)
    environment:
      # Optional: Pre-configure API Key (or set in Admin UI)
      - GEMINI_API_KEY=your_key_here
      # Optional: Config
      - BASE_URL=http://your-server-ip:8000
```

## Manual Docker Run

```bash
docker build -t podcast-ad-remover .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -v $(pwd)/public:/public \
  -e GEMINI_API_KEY=your_key_here \
  podcast-ad-remover
```

## Data Volumes
1. **`/data` (Internal)**:
    - `db/`: Database file.
    - `downloads/`: Temporary raw downloads.
    - `transcripts/`: Intermediate JSON transcripts.

2. **`/public` (External)**:
    - `feeds/`: RSS XML files.
    - `audio/`: Cleaned MP3 files.
    - `index.html`: Landing page.

