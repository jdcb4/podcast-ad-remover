from fastapi import APIRouter, Request, Form, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.feed import FeedManager
from app.core.models import SubscriptionCreate
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")
sub_repo = SubscriptionRepository()
ep_repo = EpisodeRepository()

# Helper to get settings
def get_global_settings():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        if row:
            return dict(row)
    return {}

@router.get("/admin", response_class=RedirectResponse)
async def admin_root():
    return RedirectResponse(url="/admin/system")

@router.get("/settings", response_class=RedirectResponse)
async def view_settings_redirect():
    return RedirectResponse(url="/admin/system")

# --- Admin: System ---
@router.get("/admin/system", response_class=HTMLResponse)
async def admin_system(request: Request):
    return templates.TemplateResponse("admin/system.html", {
        "request": request,
        "settings": get_global_settings(),
        "active_tab": "system"
    })

@router.post("/admin/system/update")
async def update_system_settings(
    request: Request,
    concurrent_downloads: int = Form(3),
    retention_days: int = Form(30)
):
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings 
            SET concurrent_downloads = ?,
                retention_days = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (concurrent_downloads, retention_days))
        conn.commit()
    return RedirectResponse(url="/admin/system", status_code=303)

# --- Admin: AI ---
@router.get("/admin/ai", response_class=HTMLResponse)
async def admin_ai(request: Request):
    from app.core.config import settings
    
    # helper to check which env vars are set
    env_keys = {
        "GEMINI_API_KEY": bool(settings.GEMINI_API_KEY),
        "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
        "ANTHROPIC_API_KEY": bool(settings.ANTHROPIC_API_KEY),
        "OPENROUTER_API_KEY": bool(settings.OPENROUTER_API_KEY)
    }

    return templates.TemplateResponse("admin/ai.html", {
        "request": request,
        "settings": get_global_settings(),
        "active_tab": "ai",
        "env_keys": env_keys
    })

@router.post("/admin/ai/update")
async def update_ai_settings(
    request: Request,
    whisper_model: str = Form("base"),
    ai_model_cascade: str = Form(...),
    piper_model: str = Form("en_GB-cori-high.onnx"),
    active_ai_provider: str = Form("gemini"),
    openai_api_key: str = Form(None),
    anthropic_api_key: str = Form(None),
    openrouter_api_key: str = Form(None),
    openai_model: str = Form("gpt-4o"),
    anthropic_model: str = Form("claude-3-5-sonnet"),
    openrouter_model: str = Form("google/gemini-2.0-flash-001")
):
    from app.infra.database import get_db_connection
    import json
    try:
        json.loads(ai_model_cascade)
    except:
        ai_model_cascade = '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]'

    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings 
            SET whisper_model = ?,
                ai_model_cascade = ?,
                piper_model = ?,
                active_ai_provider = ?,
                openai_api_key = ?,
                anthropic_api_key = ?,
                openrouter_api_key = ?,
                openai_model = ?,
                anthropic_model = ?,
                openrouter_model = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (
            whisper_model, ai_model_cascade, piper_model, active_ai_provider,
            openai_api_key, anthropic_api_key, openrouter_api_key,
            openai_model, anthropic_model, openrouter_model
        ))
        conn.commit()
    return RedirectResponse(url="/admin/ai", status_code=303)

@router.post("/admin/ai/test")
async def test_ai_connection(
    provider: str = Form(...),
    api_key: str = Form(None),
    model: str = Form(None)
):
    try:
        from app.core.ai_services import AdDetector
        detector = AdDetector()
        
        # Create provider slightly differently depending on type to pass correct args
        # But our factory method handles it if we pass inputs
        # We need to map form inputs to factory args
        # The factory takes (provider_type, api_key, model, openrouter_key)
        # We passed api_key as generic.
        
        prov_instance = detector.create_provider(provider, api_key=api_key, model=model)
        result = prov_instance.test_connection()
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/admin/ai/refresh/{provider}")
async def refresh_models(provider: str):
    try:
        from app.core.ai_services import AdDetector
        detector = AdDetector()
        # Create provider using saved settings (implies user must save key first usually, 
        # but we could allow passing key in query param if we wanted to be fancy. 
        # For now, rely on saved settings for Auth to keep it simple).
        prov_instance = detector.create_provider(provider) 
        models = prov_instance.list_models()
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}

# --- Admin: Prompts ---
@router.get("/admin/prompts", response_class=HTMLResponse)
async def admin_prompts(request: Request):
    return templates.TemplateResponse("admin/prompts.html", {
        "request": request,
        "settings": get_global_settings(),
        "active_tab": "prompts"
    })

# --- Admin: Queue ---
@router.get("/admin/queue", response_class=HTMLResponse)
async def admin_queue(request: Request):
    queue = ep_repo.get_queue()
    return templates.TemplateResponse("admin/queue.html", {
        "request": request,
        "queue": queue,
        "active_tab": "queue"
    })

@router.post("/admin/queue/cancel/{episode_id}")
async def cancel_episode(episode_id: int):
    # Set status to unprocessed -> Processor will detect this change and abort
    ep_repo.reset_status(episode_id)
    return RedirectResponse(url="/admin/queue", status_code=303)

@router.post("/admin/queue/retry/{episode_id}")
async def retry_episode(episode_id: int, background_tasks: BackgroundTasks):
    from app.core.processor import Processor
    processor = Processor()
    
    # Check if already processing?
    status = ep_repo.get_status(episode_id)
    if status == 'processing':
         return RedirectResponse(url="/admin/queue", status_code=303)
         
    # Force to pending
    background_tasks.add_task(processor.process_episode, episode_id)
    return RedirectResponse(url="/admin/queue", status_code=303)

# --- Admin: Logs ---
@router.get("/admin/logs", response_class=HTMLResponse)
async def admin_logs(request: Request, lines: int = 1000, level: str = "ALL"):
    from app.core.config import settings
    log_path = os.path.join(settings.DATA_DIR, "app.log")
    logs = ""
    
    if os.path.exists(log_path):
        try:
            # Read relevant lines
            # For simplicity, read last N bytes then filter lines
            # Reading 1MB roughly
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 1024 * 1024)) # 1MB
                raw_logs = f.read()
                
            log_lines = raw_logs.splitlines()
            
            # Simple Filter
            filtered = []
            for line in log_lines:
                if level != "ALL" and level not in line:
                    continue
                filtered.append(line)
                
            # Take last N
            logs = "\n".join(filtered[-lines:])
            
        except Exception as e:
            logs = f"Error reading logs: {e}"
    else:
        logs = "Log file not found."

    return templates.TemplateResponse("admin/logs.html", {
        "request": request,
        "logs": logs,
        "active_tab": "logs",
        "current_lines": lines,
        "current_level": level
    })

# --- Admin: Access ---
@router.get("/admin/access", response_class=HTMLResponse)
async def admin_access(request: Request):
    return templates.TemplateResponse("admin/access.html", {
        "request": request,
        "active_tab": "access"
    })

# Helper to render index with consistent data
def _render_index(request: Request, error: str = None):
    subs = sub_repo.get_all()
    
    # Calculate stats
    total_podcasts = len(subs)
    total_episodes = 0
    total_duration = 0 # seconds
    total_size = 0 # bytes
    
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        rows = conn.execute("SELECT duration, file_size FROM episodes WHERE status = 'completed'").fetchall()
        total_episodes = len(rows)
        for row in rows:
            if row['duration']: total_duration += row['duration']
            if row['file_size']: total_size += row['file_size']
            
    stats = {
        "podcasts": total_podcasts,
        "episodes": total_episodes,
        "hours": round(total_duration / 3600, 1),
        "size_gb": round(total_size / (1024 * 1024 * 1024), 2)
    }

    def generate_links(sub):
        base_url = str(request.base_url).rstrip("/")
        rss_url = f"{base_url}/feeds/{sub.slug}.xml"
        clean_url = rss_url.replace('https://', '').replace('http://', '')
        return {
            "rss": rss_url,
            "apple": f"podcast://{clean_url}",
            "pocket_casts": f"pktc://subscribe/{clean_url}",
            "overcast": f"overcast://x-callback-url/add?url={rss_url}",
            "castbox": f"castbox://subscribe?url={rss_url}",
            "podcast_addict": f"podcastaddict://subscribe/{rss_url}"
        }

    # Check for configuration warning
    from app.core.ai_services import AdDetector
    detector = AdDetector()
    config_warning = not detector.has_valid_config()

    subs_with_links = []
    for sub in subs:
        subs_with_links.append({
            "sub": sub,
            "links": generate_links(sub)
        })

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "subscriptions": subs_with_links, 
        "stats": stats,
        "error": error,
        "config_warning": config_warning
    })

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render_index(request)

from app.core.processor import Processor

@router.post("/add", response_class=HTMLResponse)
async def add_subscription(request: Request, feed_url: str = Form(...), initial_count: int = Form(1)):
    try:
        # Check if exists
        existing = sub_repo.get_by_url(feed_url)
        if existing:
            return _render_index(request, error="Subscription already exists")
            
        title, slug, image_url = FeedManager.parse_feed(feed_url)
        sub_create = SubscriptionCreate(feed_url=feed_url)
        new_sub = sub_repo.create(sub_create, title, slug, image_url)
        
        # Trigger initial check
        proc = Processor()
        await proc.check_feeds(subscription_id=new_sub.id, limit=initial_count)
        await proc.process_queue()
        
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
         # Need stats here too
        subs = sub_repo.get_all()
        stats = {"podcasts": len(subs), "episodes": 0, "hours": 0, "size_gb": 0}
        # ... rudimentary stats ...
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "subscriptions": [], # Fallback
            "stats": stats,
            "error": str(e)
        })

@router.get("/subscriptions/{id}", response_class=HTMLResponse)
async def view_subscription(request: Request, id: int):
    sub = sub_repo.get_by_id(id)
    if not sub:
        return RedirectResponse(url="/")
        
    # Get episodes (we need to add this to repo)
    # For now, raw SQL query or add method to repo
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        episodes = conn.execute("SELECT * FROM episodes WHERE subscription_id = ? ORDER BY pub_date DESC", (id,)).fetchall()
    
    def format_duration(seconds: int) -> str:
        if not seconds:
            return "-"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # Generate Links
    # Generate Links (Dynamic)
    base_url = str(request.base_url).rstrip("/")
    rss_url = f"{base_url}/feeds/{sub.slug}.xml"
    clean_url = rss_url.replace('https://', '').replace('http://', '')
    
    links = {
        "rss": rss_url,
        "apple": f"podcast://{clean_url}",
        "pocket_casts": f"pktc://subscribe/{clean_url}",
        "overcast": f"overcast://x-callback-url/add?url={rss_url}",
        "castbox": f"castbox://subscribe?url={rss_url}",
        "podcast_addict": f"podcastaddict://subscribe/{rss_url}"
    }

    return templates.TemplateResponse("episodes.html", {
        "request": request, 
        "subscription": sub, 
        "episodes": episodes,
        "links": links,
        "basename": lambda p: p.split('/')[-1] if p else '',
        "format_duration": format_duration
    })

@router.post("/subscriptions/{id}/settings")
async def update_settings(
    id: int,
    remove_ads: bool = Form(False),
    remove_promos: bool = Form(False),
    remove_intros: bool = Form(False),
    remove_outros: bool = Form(False),
    custom_instructions: str = Form(None),
    append_summary: bool = Form(False),
    append_title_intro: bool = Form(False)
):
    sub_repo.update_settings(
        id, 
        remove_ads, 
        remove_promos, 
        remove_intros, 
        remove_outros, 
        custom_instructions,
        append_summary,
        append_title_intro
    )
    return RedirectResponse(url=f"/subscriptions/{id}", status_code=303)

from fastapi.responses import FileResponse

@router.get("/artifacts/transcript/{id}")
async def get_transcript(id: int):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    
    with get_db_connection() as conn:
        row = conn.execute("SELECT transcript_path FROM episodes WHERE id = ?", (id,)).fetchone()
        if row and row['transcript_path']:
            path = row['transcript_path']
            # Try exact path
            if os.path.exists(path):
                return FileResponse(path)
            
            # Fallback: Try basename in configured directory
            # Handle potential Windows paths in DB
            basename = os.path.basename(path.replace('\\', '/'))
            fallback_path = os.path.join(settings.TRANSCRIPTS_DIR, basename)
            if os.path.exists(fallback_path):
                return FileResponse(fallback_path)
                
    raise HTTPException(status_code=404, detail="Transcript not found")

@router.get("/artifacts/report/{id}")
async def get_report(id: int):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    
    with get_db_connection() as conn:
        row = conn.execute("SELECT report_path, ad_report_path FROM episodes WHERE id = ?", (id,)).fetchone()
        
        # Helper to check path with fallback
        def check_path(path):
            if not path: return None
            if os.path.exists(path): return path
            basename = os.path.basename(path.replace('\\', '/'))
            fallback = os.path.join(settings.TRANSCRIPTS_DIR, basename) # Reports are also in transcripts dir usually? 
            # Actually reports might be in transcripts dir based on processor.py
            if os.path.exists(fallback): return fallback
            return None

        # Prefer HTML report
        if row and row['report_path']:
            valid_path = check_path(row['report_path'])
            if valid_path: return FileResponse(valid_path)

        # Fallback to JSON
        if row and row['ad_report_path']:
            valid_path = check_path(row['ad_report_path'])
            if valid_path: return FileResponse(valid_path)
            
    raise HTTPException(status_code=404, detail="Report not found")
