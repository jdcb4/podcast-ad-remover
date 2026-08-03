from types import SimpleNamespace

import pytest

from app.web import router as web_router


@pytest.mark.asyncio
async def test_dashboard_queue_returns_a_small_safe_partial_refresh_payload(monkeypatch):
    monkeypatch.setattr(
        web_router,
        "ep_repo",
        SimpleNamespace(
            get_queue=lambda: [
                {
                    "id": 7,
                    "title": "Episode",
                    "podcast_title": "Podcast",
                    "status": "processing",
                    "processing_step": "Transcribing",
                    "progress": 150,
                    "original_url": "https://example.com/secret-source.mp3",
                }
            ]
        ),
    )

    result = await web_router.dashboard_queue()

    assert result == {
        "items": [
            {
                "id": 7,
                "title": "Episode",
                "podcast_title": "Podcast",
                "status": "processing",
                "processing_step": "Transcribing",
                "progress": 100,
            }
        ],
        "total": 1,
    }
