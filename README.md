# Podcast Ad Remover (AGPAR)
This is an app that downloads podcasts, uses AI to remove ads, and then creates a new feed that you can consume them from. 
 I mainly vibe-coded for my own personal use and it seems to be working pretty well so I thought I would share it. I'm sure there are heaps of things wrong with it, but for running on my own homelab it has worked OK. 

The Audio transcription happens locally using Whisper which can be fairly processing ehavy, but I run it on the CPU on a little N100 machine and it stil manages to work at something like 3x real speed. I do reccomend pinning it to some subset of your CPU cores because othwerwise you may well freeze the PC.

The ad detection uses LLMs. I use Gemini because the free tier is very generous and enough to very easuily manage the API calls this app makes. I only built in support for other models out of curiosity and I haven't tested them rigorously at all.

![Podcast Ad Remover UI](app/web/static/img/favicon.png)

## Features

-   **Automatic Ad Detection**: Uses state-of-the-art LLMs (Google Gemini, OpenAI GPT-4, Anthropic Claude, or OpenRouter) to intelligently identify ads, sponsor reads, and promotional segments.
-   **Audio Processing**: Uses **Whisper** for accurate transcription and **FFmpeg** for precise audio cutting.
-   **Seamless Playback**: Generates custom RSS feeds for every subscription. Add them to your favorite podcast player (Apple Podcasts, Pocket Casts, etc.) to listen ad-free.
-   **Smart Enhancements**: 
    -   **Intro/Outro Removal**: Option to trim standard podcast intros/outros.
    -   **AI Summaries**: Generate and append spoken AI summaries to the start of episodes.
    -   **Custom "Title Intros"**: Adds a "You're listening to..." intro for context.
-   **Robust Management**:
    -   Processing queue with pause/resume/cancel/retry capabilities.
    -   Manual "Reprocess" option to try different settings.
    -   Full admin dashboard for managing subscriptions and viewing logs.

## Quick Start

### Docker Run (Fastest)

Run the application directly from Docker Hub without cloning the repository:

```bash
docker run -d \
  --name podcast-ad-remover \
  -p 8000:8000 \
  -v ./data:/data \
  -e GEMINI_API_KEY=your_api_key \
  jdcb4/podcast-ad-remover:latest
```

### Docker Compose (Build from Source)

1.  Clone the repository.
2.  Create a `.env` file (see `env.example`):
    ```bash
    cp env.example .env
    ```
3.  Run with Docker Compose:
    ```bash
    docker-compose up -d --build
    ```
4.  Access the web interface at `http://localhost:8000`.
5.  Navigate to **Settings > AI Models** to configure your API key.

### Getting a Free Gemini API Key

This application recommends using Google Gemini (Flash model) as it is currently free and very fast.

1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Sign in with your Google account.
3.  Click **Create API key**.
4.  Search for "Google Cloud Platform" project or create a new one if prompted.
5.  Copy the generated key (starts with `AIza...`).
6.  Paste this key into the application settings.

### Unraid

A dedicated Unraid template is included (`podcast-ad-remover.xml`). Add the template to your Docker templates URL or copy it to your flash drive. See [Documentation/Unraid_Deployment.md](Documentation/Unraid_Deployment.md) for details.

## Documentation

-   [Architecture](Documentation/Architecture.md)
-   [Data Flow](Documentation/Data_Flow.md)
-   [Deployment](Documentation/Deployment.md)
-   [Environment Variables](Documentation/Environment_Variables.md)

## License

MIT License
