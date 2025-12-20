# Architecture

## Overview
Podcast Ad Remover is a monolithic, Dockerized web application designed to automatically remove ads from podcasts. It subscribes to RSS feeds, downloads episodes, processes them using AI (Whisper + Gemini), and republishes them via a new ad-free RSS feed.

## Core Principles
- **Simplicity**: Single service, minimal dependencies.
- **Efficiency**: Lightweight base image, efficient resource usage.
- **Modularity**: Clear separation between web, core logic, and infrastructure.

## Technology Stack
- **Language**: Python 3.11+
- **Web Framework**: FastAPI (for performance, async support, and auto-docs)
- **Database**: SQLite (lightweight, file-based)
- **AI Services**:
    - **Transcription**: OpenAI Whisper (local execution)
    - **Ad Detection**: Google Gemini API (cloud service)
- **Audio Processing**: FFmpeg
- **Scheduling**: Python `asyncio` background tasks (simple periodic loop)

## System Components

### 1. Web Server (FastAPI)
- Serves the Web UI (HTML/HTMX/Jinja2).
- Provides REST API for management.
- Serves static files (processed audio and generated RSS feeds).

### 2. Core Logic (`app/core`)
- **Feed Manager**: Handles RSS subscription and parsing.
- **Processor**: Orchestrates the download -> transcribe -> detect -> cut pipeline.
- **Feed Generator**: Re-generates RSS XML for processed episodes.

### 3. Infrastructure (`app/infra`)
- **Database**: SQLite connection and repositories.
- **Storage**: Manages file paths for downloads and processed files.
- **Scheduler**: A simple async loop that triggers the processor periodically.

## Directory Structure
```
/
├── app/
│   ├── __init__.py
│   ├── main.py           # Entrypoint
│   ├── api/              # API Routes
│   ├── web/              # Web UI Routes & Templates
│   ├── core/             # Business Logic
│   │   ├── processor.py
│   │   ├── feed_manager.py
│   │   └── ai_services.py
│   └── infra/            # DB, Config, Storage
│       ├── database.py
│       └── config.py
├── Documentation/        # Project Documentation
├── Dockerfile
└── requirements.txt
```

## Data Storage

The application uses two distinct storage locations which should be mounted as volumes:

### 1. Internal Data (`/data`)
Used for application state and temporary processing files.
- **/data/db/podcasts.db**: SQLite database.
- **/data/downloads/**: Temporary raw audio files.
- **/data/transcripts/**: Intermediate transcript files.

### 2. Public Output (`/public`)
Used for files that need to be served to podcast players/users.
- **/public/feeds/**: Generated RSS XML feeds.
- **/public/audio/**: Processed ad-free audio files (organized by podcast slug).
- **/public/index.html**: Landing page.
