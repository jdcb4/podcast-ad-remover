"""Serve the current UI with disposable demo data for documentation screenshots.

Run from the repository root: python scripts/preview_readme.py
This creates its own temporary data directory, binds only to loopback and never
starts processing or calls a model. Close the server to delete the demo data.
"""
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from types import SimpleNamespace
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="par-readme-demo-") as directory:
        os.environ.update(DATA_DIR=directory, PROCESSOR_ENABLED="false",
                          BASE_URL="http://127.0.0.1:8767")
        from fastapi import FastAPI, Request
        from fastapi.staticfiles import StaticFiles
        from starlette.responses import Response
        from starlette.middleware.sessions import SessionMiddleware
        import uvicorn
        from app.core.config import settings
        from app.infra.database import init_db, get_db_connection
        from app.web import router as web
        from app.api.audio_routes import router as audio_router
        from app.core.reports import render_ad_report
        from app.core import timeline

        for folder in (Path(settings.DB_PATH).parent, Path(settings.PODCASTS_DIR),
                       Path(settings.FEEDS_DIR), Path(settings.MODELS_DIR)):
            folder.mkdir(parents=True, exist_ok=True)
        init_db()
        artwork = Path(directory) / "demo-artwork"
        artwork.mkdir()
        shows = [
            ("Field Notes", "field-notes", "Stories of science, nature and everyday discoveries.", "#194b43", "FIELD", "NOTES"),
            ("Signal & Story", "signal-story", "Thoughtful conversations about technology and the people behind it.", "#403361", "SIGNAL", "& STORY"),
            ("The Long View", "long-view", "History, ideas and the connections that shape our world.", "#784329", "THE", "LONG VIEW"),
        ]
        titles = ["The hidden life of urban forests", "Why cities need quiet spaces", "What we can learn from a river"]
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users(id,username,password_hash,is_admin) VALUES(1,'demo','preview-login-disabled',1)")
            conn.execute("""UPDATE app_settings SET auth_enabled=1,onboarding_status='completed',
                active_ai_provider='custom',custom_llm_base_url='http://127.0.0.1:9999/v1',
                custom_llm_model='demo-model',app_external_url='http://127.0.0.1:8767' WHERE id=1""")
            for show_id, (title, slug, description, color, line1, line2) in enumerate(shows, 1):
                (artwork / f"{slug}.svg").write_text(
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">'
                    f'<rect width="400" height="400" fill="{color}"/>'
                    '<circle cx="320" cy="80" r="95" fill="white" opacity=".08"/>'
                    f'<text x="32" y="190" fill="#fff" font-family="sans-serif" font-size="42" font-weight="700">{line1}</text>'
                    f'<text x="32" y="244" fill="#fff" font-family="sans-serif" font-size="42" font-weight="700">{line2.replace("&", "&amp;")}</text>'
                    '<path d="M32 300h96" stroke="white" stroke-width="5"/>'
                    '</svg>', encoding="utf-8")
                conn.execute("""INSERT INTO subscriptions(id,feed_url,title,slug,description,image_url,owner_user_id)
                    VALUES(?,?,?,?,?,?,1)""", (show_id, f"https://example.com/{slug}.xml", title, slug,
                    description, f"/demo-artwork/{slug}.svg"))
                conn.execute("INSERT INTO user_subscriptions(user_id,subscription_id) VALUES(1,?)", (show_id,))
                for index in range(3):
                    path = Path(settings.PODCASTS_DIR) / slug / str(index) / "processed.wav"
                    path.parent.mkdir(parents=True)
                    with wave.open(str(path), "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(8000)
                        audio.writeframes(b"\x00\x00" * 8000 * 120)
                    episode_title = titles[index] if show_id == 1 else [
                        "Building technology that lasts", "A different way to tell the story", "Ideas worth keeping"
                    ][index]
                    summary = "A conversation about green spaces, biodiversity and making room for nature in our cities." if show_id == 1 else "Thoughtful perspectives, practical examples and ideas to take into your week."
                    conn.execute("""INSERT INTO episodes(subscription_id,guid,title,original_url,pub_date,
                        duration,status,local_filename,description,ai_summary,file_size,listen_count)
                        VALUES(?,?,?,?,?,?,'completed',?,?,?,?,?)""", (show_id, f"demo-{show_id}-{index}",
                        episode_title, "https://example.com/episode.wav", f"2026-10-{10-index:02d}T08:00:00",
                        120, str(path), "A new episode with stories, interviews and ideas to explore.",
                        summary, path.stat().st_size, 12-index*3))
            conn.commit()

        # Synthetic classification evidence rendered by the production report code.
        examples = [
            (0, 12, "Intro", "Welcome to Field Notes. Today we explore urban forests.", "Show identification and opening housekeeping."),
            (12, 38, "Content", "Trees cool our streets and create habitats for wildlife.", "Substantive discussion of urban biodiversity."),
            (38, 55, "Ad", "This episode is sponsored by Example Coffee. Use code FOREST for 20% off your first order.", "Sponsor endorsement with a discount code and a call to purchase."),
            (55, 91, "Content", "Planting a mix of native species helps a woodland withstand disease.", "The interview returns to the episode's subject."),
            (91, 104, "Promo", "Join the Field Notes membership for bonus episodes each month.", "A distinct pitch for the podcast's paid membership."),
            (104, 120, "Outro", "Thanks for listening. We will see you next week.", "Routine closing sign-off."),
        ]
        transcript = {"segments": [{"start": start, "end": end, "text": text}
                                  for start, end, label, text, reason in examples]}
        transcript["segments"][0]["end"] = 12.2
        transcript["segments"][1]["start"] = 11.8
        units, notes = timeline.prepare_timeline(transcript, 120)
        rows = [{"first_id": i, "last_id": i, "start": start, "end": end,
                 "label": label, "text": text, "reason": reason}
                for i, (start, end, label, text, reason) in enumerate(examples, 1)]
        policy = timeline.apply_preferences(rows, {"remove_ads": True, "remove_promos": True})
        analysis = {"timeline": units, "segments": rows, "summary":
                    "This episode includes urban biodiversity and the benefits of native trees.",
                    "normalization_notes": notes,
                    "provider": "custom", "model": "demo-model", "output_mode": "strict",
                    "prompt_version": timeline.PROMPT_VERSION}
        report = Path(directory) / "demo-report.html"
        report.write_text(render_ad_report(SimpleNamespace(id=1, title=titles[0], guid="demo-1-0"),
                          policy["segments"], analysis=analysis, edit_policy=policy), encoding="utf-8")
        transcript_path = Path(directory) / "demo-transcript.json"
        transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
        with get_db_connection() as conn:
            conn.execute("UPDATE episodes SET report_path=?, transcript_path=? WHERE id=1",
                         (str(report), str(transcript_path)))
            conn.commit()

        app = FastAPI()

        @app.middleware("http")
        async def demo_session(request: Request, call_next):
            if request.method not in {"GET", "HEAD"}:
                return Response(status_code=405)
            request.session["user_id"] = 1
            return await call_next(request)

        app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32))
        app.include_router(web.router)
        app.include_router(audio_router)
        app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
        app.mount("/demo-artwork", StaticFiles(directory=artwork), name="demo-artwork")
        print("Demo UI: http://127.0.0.1:8767/?view=mine", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=8767)


if __name__ == "__main__":
    main()
