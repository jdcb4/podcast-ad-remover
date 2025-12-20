# Design Decisions

## 1. Framework: FastAPI vs Flask
**Decision**: Use **FastAPI**.
- **Reason**:
    - Native `async` support is better for handling concurrent I/O (downloads, AI API calls) without blocking the main thread, even in a single process.
    - Built-in Pydantic validation ensures data integrity.
    - Automatic Swagger UI is excellent for testing and API documentation.
    - Performance is generally higher.

## 2. Scheduler: Custom Async Loop vs APScheduler
**Decision**: Use a **Simple Async Loop**.
- **Reason**:
    - APScheduler is powerful but adds complexity (executors, job stores).
    - Our requirement is simple: "Check every X minutes".
    - A simple `while True: await sleep(3600)` loop in a background task is sufficient, easier to debug, and uses fewer resources.

## 3. Database: SQLite
**Decision**: **SQLite**.
- **Reason**:
    - Zero configuration.
    - Single file storage makes backups easy.
    - Sufficient performance for a single-user application.
    - WAL mode allows concurrent readers/writers if needed.

## 4. Frontend: Server-Side Rendered (Jinja2) + HTMX
**Decision**: **Jinja2 Templates + HTMX**.
- **Reason**:
    - Avoids the complexity of a separate build step (React/Vue).
    - Keeps the project as a single deployable unit.
    - HTMX provides a "modern" feel (SPA-like updates) with very little JavaScript code.

## 5. AI Abstraction
**Decision**: Create an abstract `AdDetector` and `Transcriber` interface.
- **Reason**:
    - Allows swapping Whisper for other STT services later.
    - Allows swapping Gemini for GPT-4 or Claude if needed.
    - Makes testing easier (can mock the AI services).
