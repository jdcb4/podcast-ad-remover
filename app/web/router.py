from fastapi import APIRouter, Request, Form, Depends, BackgroundTasks, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.feed import FeedManager
from app.core.models import SubscriptionCreate
from app.web.auth import get_current_user, require_auth, require_admin, log_login_attempt, SESSION_USER_KEY
from app.web.auth_utils import hash_password, verify_password, generate_secure_password, get_client_ip
from app.infra.database import get_db_connection
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Add simple markdown filter
def simple_markdown(text):
    """Convert basic markdown to HTML: **bold**, bullets (*, -, •, 1.)"""
    if not text:
        return ""
    import re
    # Handle both Unix and Windows line endings
    lines = text.replace('\r\n', '\n').split('\n')
    result = []
    in_list = False
    list_items = []
    
    def apply_bold(t):
        import re
        # Convert **text** to <strong>text</strong> (Bold)
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        # Convert *text* to <strong>text</strong> (User requested single star bold)
        t = re.sub(r'\*(.+?)\*', r'<strong>\1</strong>', t)
        return t

    def flush_list():
        nonlocal in_list, list_items, result
        if in_list and list_items:
            # Inline styles as baseline, tailwind classes for styling
            list_html = '<ul class="list-disc ml-8 space-y-2 mb-4 text-gray-800" style="list-style-type: disc; margin-left: 2rem; margin-bottom: 1rem;">\n' + \
                        '\n'.join(list_items) + '\n</ul>'
            result.append(list_html)
            list_items = []
            in_list = False

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if in_list:
                flush_list()
            continue
            
        # Match bullets: *, -, • or "1." followed by any whitespace (\s+)
        bullet_match = re.match(r'^([\*\-\•]|\d+\.)\s+(.+)$', stripped_line)
        
        if bullet_match:
            if not in_list:
                in_list = True
            content = bullet_match.group(2).strip()
            # Apply bolding only to the content of the list item
            content = apply_bold(content)
            list_items.append(f'<li class="mb-1">{content}</li>')
        else:
            if in_list:
                flush_list()
            # Apply bolding to the paragraph text
            processed_line = apply_bold(stripped_line)
            result.append(f'<p class="mb-2 text-gray-800 leading-relaxed">{processed_line}</p>')
            
    if in_list:
        flush_list()
        
    return '\n'.join(result)

templates.env.filters['simple_markdown'] = simple_markdown

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

from app.core.utils import get_app_base_url



def generate_rss_links(request: Request, sub, global_settings: dict, user_obj=None):
    """Consolidated logic for generating RSS links with optional auth injection."""
    base_url = get_app_base_url(global_settings, request)
    
    rss_url = f"{base_url}/feeds/{sub.slug}.xml"
    
    # Inject Auth if Enabled
    auth_enabled_val = global_settings.get('enable_feed_auth')
    is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False
    
    if is_auth_enabled:
        if global_settings.get('auth_enabled'):
            # Integrated Auth: Use same credentials as dashboard
            auth_user = user_obj.username if user_obj else "admin"
            auth_pass = request.session.get("user_pass", "")
        else:
            # Standalone Auth: Use manually provided feed credentials
            auth_user = global_settings.get('feed_auth_username')
            auth_pass = global_settings.get('feed_auth_password')
            
        if auth_user and auth_pass:
            import base64
            token = base64.b64encode(f"{auth_user}:{auth_pass}".encode()).decode()
            separator = "&" if "?" in rss_url else "?"
            rss_url = f"{rss_url}{separator}auth={token}"


    clean_url = rss_url.replace('https://', '').replace('http://', '')
    return {
        "rss": rss_url,
        "apple": rss_url,  # Method 1: Direct HTTPS URL for manual "Follow a Show by URL"
        "pocket_casts": f"pktc://subscribe/{rss_url}",
        "overcast": f"overcast://x-callback-url/add?url={rss_url}",
        "castbox": f"castbox://subscribe?url={rss_url}",
        "podcast_addict": f"podcastaddict://subscribe/{rss_url}"
    }

# Helper to get pending access requests count for sidebar badge
def get_pending_requests_count():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM access_requests WHERE status = 'pending'").fetchone()
        return result[0] if result else 0

# --- Authentication Routes ---
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Display login page or first-time setup."""
    with get_db_connection() as conn:
        settings = conn.execute("SELECT auth_enabled, initial_password FROM app_settings WHERE id = 1").fetchone()
        user_count = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()['count']
    
    # Check if this is first launch
    first_launch = user_count == 0
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "first_launch": first_launch,
        "initial_password": settings['initial_password'] if settings else None,
        "auth_enabled": settings['auth_enabled'] if settings else False
    })

@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login submission."""
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")
    
    with get_db_connection() as conn:
        user_row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    
    if not user_row or not verify_password(password, user_row['password_hash']):
        log_login_attempt(username, client_ip, False, user_agent)
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password",
            "first_launch": False,
            "auth_enabled": True
        })
    
    # Successful login
    log_login_attempt(username, client_ip, True, user_agent)
    
    # Update last login
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_row['id']))
        conn.commit()
    
    # Set session
    request.session[SESSION_USER_KEY] = user_row['id']
    request.session["user_pass"] = password
    
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

@router.get("/logout")
async def logout(request: Request):
    """Handle logout."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@router.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, user: dict = Depends(require_auth)):
    """Display password change page."""
    with get_db_connection() as conn:
        settings = conn.execute("SELECT require_password_change FROM app_settings WHERE id = 1").fetchone()
    
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "user": user,
        "required": settings['require_password_change'] if settings else False
    })

@router.post("/change-password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user: dict = Depends(require_auth)
):
    """Handle password change submission."""
    if new_password != confirm_password:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "user": user,
            "error": "Passwords do not match"
        })
    
    # Verify current password
    with get_db_connection() as conn:
        user_row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()
    
    if not verify_password(current_password, user_row['password_hash']):
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "user": user,
            "error": "Current password is incorrect"
        })
    
    # Update password
    new_hash = hash_password(new_password)
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user.id))
        conn.execute("UPDATE app_settings SET require_password_change = 0, initial_password = NULL WHERE id = 1")
        conn.commit()
    
    # Update session password
    request.session["user_pass"] = new_password
    
    return RedirectResponse(url="/admin/system?password_changed=1", status_code=status.HTTP_302_FOUND)

@router.get("/request-access", response_class=HTMLResponse)
async def request_access_page(request: Request):
    """Display access request form."""
    return templates.TemplateResponse("request_access.html", {"request": request})

@router.post("/submit-access-request")
async def submit_access_request(
    request: Request,
    username: str = Form(...),
    email: str = Form(None),
    reason: str = Form(None)
):
    """Handle access request submission."""
    client_ip = get_client_ip(request)
    
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO access_requests (username, email, reason, ip_address) VALUES (?, ?, ?, ?)",
            (username, email, reason, client_ip)
        )
        conn.commit()
    
    return templates.TemplateResponse("request_access.html", {
        "request": request,
        "success": "Your access request has been submitted. You will be notified when it is reviewed."
    })

@router.get("/admin", response_class=RedirectResponse)
async def admin_root():
    return RedirectResponse(url="/admin/system")

@router.get("/settings", response_class=RedirectResponse)
async def view_settings_redirect():
    return RedirectResponse(url="/admin/system")

# --- Admin: System ---
@router.get("/admin/system", response_class=HTMLResponse)
async def admin_system(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse("admin/system.html", {
        "request": request,
        "user": user,
        "settings": get_global_settings(),
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "system"
    })

@router.post("/admin/system/update")
async def update_system_settings(
    request: Request,
    concurrent_downloads: int = Form(3),
    retention_days: int = Form(30),
    check_interval_minutes: int = Form(60),
    app_external_url: str = Form(None),
    auth_enabled: bool = Form(False),
    ip_allowlist: str = Form(None),
    enable_feed_auth: bool = Form(False),
    feed_auth_username: str = Form(None),
    feed_auth_password: str = Form(None),
    redirect_to: str = Form(None)
):
    from app.infra.database import get_db_connection
    
    # Feed password is stored as plaintext to allow injection into links
    # (Feed auth is a simpler security tier than app login)
    hashed_feed_password = feed_auth_password

    with get_db_connection() as conn:
        # Get current settings
        current_settings = conn.execute("SELECT auth_enabled, feed_auth_password FROM app_settings WHERE id = 1").fetchone()
        
        # Get current feed password if not changing
        if enable_feed_auth and not feed_auth_password:
            hashed_feed_password = current_settings['feed_auth_password'] if current_settings else None

        # Backend Validation: If feed auth is enabled standalone, ensure we have credentials
        if enable_feed_auth and not auth_enabled:
            # Check if we have username AND (new password OR existing password)
            if not feed_auth_username or not (feed_auth_password or hashed_feed_password):
                # Force disable feed auth if credentials missing
                enable_feed_auth = False
        
        # Check if auth is being enabled for the first time
        if auth_enabled and (not current_settings or not current_settings['auth_enabled']):
            # Check if ANY admin user exists (regardless of username)
            # This prevents re-creating 'admin' if the user renamed their account
            admin_exists = conn.execute("SELECT COUNT(*) as count FROM users WHERE is_admin = 1").fetchone()['count']
            
            if not admin_exists:
                # Create admin user with random password
                initial_password = generate_secure_password()
                password_hash = hash_password(initial_password)
                
                conn.execute(
                    "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                    ("admin", password_hash, 1)
                )
                
                # Store initial password and set require_password_change
                conn.execute(
                    "UPDATE app_settings SET initial_password = ?, require_password_change = 1 WHERE id = 1",
                    (initial_password,)
                )
        
        # Check if app_external_url is changing
        old_url = conn.execute("SELECT app_external_url FROM app_settings WHERE id = 1").fetchone()
        url_changed = old_url and old_url['app_external_url'] != app_external_url
        
        # Update settings
        conn.execute("""
            UPDATE app_settings SET concurrent_downloads = ?,
                retention_days = ?,
                check_interval_minutes = ?,
                app_external_url = ?,
                auth_enabled = ?,
                ip_allowlist = ?,
                enable_feed_auth = ?,
                feed_auth_username = ?,
                feed_auth_password = COALESCE(?, feed_auth_password),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (concurrent_downloads, retention_days, check_interval_minutes, app_external_url,
              1 if auth_enabled else 0, ip_allowlist,
              1 if enable_feed_auth else 0, 
              feed_auth_username if feed_auth_username else None, 
              hashed_feed_password if hashed_feed_password else None))
        conn.commit()
    
    # Regenerate all feeds if the URL changed
    if url_changed:
        try:
            from app.core.rss_gen import RSSGenerator
            logger.info(f"Public URL changed to '{app_external_url}', regenerating all RSS feeds...")
            rss_gen = RSSGenerator()
            
            # Get all subscriptions
            from app.infra.repository import SubscriptionRepository
            sub_repo = SubscriptionRepository()
            subs = sub_repo.get_all()
            
            for sub in subs:
                rss_gen.generate_feed(sub.id)
            rss_gen.generate_unified_feed()
            
            logger.info(f"Successfully regenerated {len(subs) + 1} feeds with new URL.")
        except Exception as e:
            logger.error(f"Failed to regenerate feeds after URL change: {e}")
    
    url = redirect_to if redirect_to else "/admin/system?success=System+settings+updated"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

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

    user = get_current_user(request)

    return templates.TemplateResponse("admin/ai.html", {
        "request": request,
        "user": user,
        "settings": get_global_settings(),
        "pending_requests_count": get_pending_requests_count(),
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
    gemini_api_key: str = Form(None),
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
                gemini_api_key = ?,
                openai_model = ?,
                anthropic_model = ?,
                openrouter_model = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (
            whisper_model, ai_model_cascade, piper_model, active_ai_provider,
            openai_api_key, anthropic_api_key, openrouter_api_key, gemini_api_key,
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
    # Default prompts from ai_services.py
    default_prompts = {
        "ad_base": """Identify segments in the transcript that match the Targets.
Targets: {targets}
{custom_instr}
Return a JSON array of objects with "start", "end", "label" (Ad/Promo/Intro/Outro), and "reason" (brief explanation).
Example: [{"start": 0.0, "end": 10.0, "label": "Ad", "reason": "Sponsor read for XYZ"}]""",
        "sponsor": "Sponsor messages, ad reads, promotional segments",
        "promo": "Cross-promotions, plugs for other shows or content",
        "summary": "Summarize the key points of this podcast episode in 3-5 bullet points."
    }
    
    user = get_current_user(request)

    return templates.TemplateResponse("admin/prompts.html", {
        "request": request,
        "user": user,
        "settings": get_global_settings(),
        "default_prompts": default_prompts,
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "prompts"
    })

@router.post("/admin/prompts")
async def save_prompts(request: Request):
    user = get_current_user(request)
    if not user or not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    form = await request.form()
    
    # Required variables for validation
    required_vars = {
        'ad_prompt_base': ['{targets}', '{custom_instr}'],
        'summary_prompt_template': ['{transcript_context}']
    }
    
    # Validate required variables
    for field, vars_needed in required_vars.items():
        value = form.get(field, '')
        for var in vars_needed:
            if var not in value:
                raise HTTPException(status_code=400, detail=f"{field} must include {var}")
    
    # Save to database
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings SET
                ad_prompt_base = ?,
                ad_target_sponsor = ?,
                ad_target_promo = ?,
                summary_prompt_template = ?
            WHERE id = 1
        """, (
            form.get('ad_prompt_base'),
            form.get('ad_target_sponsor'),
            form.get('ad_target_promo'),
            form.get('summary_prompt_template')
        ))
        conn.commit()
    
    return {"status": "success"}

@router.post("/admin/prompts/reset")
async def reset_prompts(request: Request):
    user = get_current_user(request)
    if not user or not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Default prompts
    defaults = {
        'summary': """You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
The summary must:
1. NOT mention the podcast name, episode title, or date.
2. Start immediately with "This episode includes".
3. Briefly summarize key topics.
Transcript Context: {transcript_context}""",
        'ad_base': """Identify segments in the transcript that match the Targets.
Targets: {targets}
{custom_instr}
Return a JSON array of objects with "start", "end", "label" (Ad/Promo/Intro/Outro), and "reason" (brief explanation).
Example: [{"start": 0.0, "end": 10.0, "label": "Ad", "reason": "Sponsor read for XYZ"}]""",
        'sponsor': 'Sponsor messages, ad reads, promotional segments',
        'promo': 'Cross-promotions, plugs for other shows or content'
    }
    
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings SET
                summary_prompt_template = ?,
                ad_prompt_base = ?,
                ad_target_sponsor = ?,
                ad_target_promo = ?
            WHERE id = 1
        """, (defaults['summary'], defaults['ad_base'], defaults['sponsor'], defaults['promo']))
        conn.commit()
    
    return {"status": "success"}

# --- Admin: Queue ---
@router.get("/admin/queue", response_class=HTMLResponse)
async def admin_queue(request: Request):
    user = get_current_user(request)
    queue = ep_repo.get_queue()
    return templates.TemplateResponse("admin/queue.html", {
        "request": request,
        "user": user,
        "queue": queue,
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "queue"
    })

@router.post("/admin/queue/cancel/{episode_id}")
async def cancel_episode(episode_id: int):
    # Soft delete an episode (marks as ignored, cleans up files)
    from app.core.processor import Processor
    proc = Processor()
    await proc.delete_episode(episode_id)
    return RedirectResponse(url="/admin/queue", status_code=303)

@router.post("/admin/queue/retry/{episode_id}")
async def retry_episode(episode_id: int):
    # Check if already processing?
    status = ep_repo.get_status(episode_id)
    if status == 'processing':
         return RedirectResponse(url="/admin/queue", status_code=303)
         
    # Force to pending (Background processor will pick it up)
    ep_repo.update_status(episode_id, "pending")
    return RedirectResponse(url="/admin/queue", status_code=303)

@router.post("/episodes/{episode_id}/download")
async def manual_download_episode(episode_id: int, request: Request):
    # Update DB to pending
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET is_manual_download=1, status='pending' WHERE id=?", (episode_id,))
        conn.commit()
    
    # Background processor will see 'pending' and pick it up (polls every 10s)
    
    return RedirectResponse(url=request.headers.get("referer") or "/", status_code=303)


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

    
    user = get_current_user(request)

    return templates.TemplateResponse("admin/logs.html", {
        "request": request,
        "user": user,
        "logs": logs,
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "logs",
        "current_lines": lines,
        "current_level": level
    })

# --- Admin: Access ---

@router.get("/subscribe/apple", response_class=HTMLResponse)
async def apple_subscribe_page(request: Request, url: str):
    """Render the Apple Podcasts subscription instruction page."""
    return templates.TemplateResponse("apple_subscribe.html", {
        "request": request,
        "feed_url": url
    })

@router.get("/admin/access-requests", response_class=HTMLResponse)
async def admin_access(request: Request):
    from app.infra.database import get_db_connection
    from datetime import datetime, timedelta
    
    # Load settings and user
    settings = get_global_settings()
    user = get_current_user(request)
    
    # Load pending access requests
    with get_db_connection() as conn:
        pending_requests = conn.execute(
            "SELECT * FROM access_requests WHERE status = 'pending' ORDER BY requested_at DESC"
        ).fetchall()
        
        # Load login history for last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        login_history = conn.execute(
            """SELECT * FROM login_attempts 
               WHERE timestamp > ? 
               ORDER BY timestamp DESC 
               LIMIT 100""",
            (thirty_days_ago,)
        ).fetchall()
        
        # Load active users
        active_users = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
    
    return templates.TemplateResponse("admin/access_requests.html", {
        "request": request,
        "user": user,
        "active_tab": "access",
        "settings": settings,
        "app_base_url": get_app_base_url(request, settings),
        "pending_requests": [dict(row) for row in pending_requests],
        "active_users": [dict(row) for row in active_users],
        "login_history": [dict(row) for row in login_history],
        "pending_requests_count": get_pending_requests_count(),
    })

@router.post("/admin/users/delete/{user_id}")
async def delete_user(request: Request, user_id: int):
    # Check admin
    user = require_admin(request)
    if user.id == user_id:
        return RedirectResponse(
            url="/admin/access?error=Cannot delete your own account", 
            status_code=status.HTTP_303_SEE_OTHER
        )

    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    
    return RedirectResponse(
        url="/admin/access?success=User deleted successfully", 
        status_code=status.HTTP_303_SEE_OTHER
    )

# --- Admin: Approve Access Request ---
@router.post("/admin/access-requests/{request_id}/approve")
async def approve_access_request(request: Request, request_id: int):
    from app.infra.database import get_db_connection
    from app.web.auth_utils import hash_password, generate_secure_password
    
    with get_db_connection() as conn:
        # Get the access request
        access_req = conn.execute(
            "SELECT * FROM access_requests WHERE id = ?", (request_id,)
        ).fetchone()
        
        if not access_req:
            return RedirectResponse(url="/admin/access?error=Request+not+found", status_code=303)
        
        # Check if username already exists
        existing_user = conn.execute(
            "SELECT id FROM users WHERE username = ?", (access_req['username'],)
        ).fetchone()
        
        if existing_user:
            # Update request status to denied with reason
            conn.execute(
                "UPDATE access_requests SET status = 'denied', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (request_id,)
            )
            conn.commit()
            return RedirectResponse(url="/admin/access?error=Username+already+exists", status_code=303)
        
        # Generate random password for the new user
        temp_password = generate_secure_password()
        password_hash = hash_password(temp_password)
        
        # Create the new user
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            (access_req['username'], password_hash)
        )
        
        # Update access request status
        conn.execute(
            "UPDATE access_requests SET status = 'approved', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        
        logger.info(f"AUTH - Access request approved: {access_req['username']} - Temp password generated")
        
    # Redirect back to access page with success message
    return RedirectResponse(url=f"/admin/access?approved={access_req['username']}&password={temp_password}", status_code=303)

# --- Admin: Deny Access Request ---
@router.post("/admin/access-requests/{request_id}/deny")
async def deny_access_request(request: Request, request_id: int):
    from app.infra.database import get_db_connection
    
    with get_db_connection() as conn:
        # Get the access request to log username
        access_req = conn.execute(
            "SELECT username FROM access_requests WHERE id = ?", (request_id,)
        ).fetchone()
        
        if not access_req:
            return RedirectResponse(url="/admin/access?error=Request+not+found", status_code=303)
        
        # Update access request status to denied
        conn.execute(
            "UPDATE access_requests SET status = 'denied', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        
        logger.info(f"AUTH - Access request denied: {access_req['username']}")
        
    return RedirectResponse(url="/admin/access?denied=1", status_code=303)

# --- Admin: Update User Username ---
@router.post("/admin/users/{user_id}/username")
async def update_user_username(request: Request, user_id: int, username: str = Form(...)):
    require_admin(request)
    with get_db_connection() as conn:
        # Check if username already exists
        existing = conn.execute("SELECT id FROM users WHERE username = ? AND id != ?", (username, user_id)).fetchone()
        if existing:
            return RedirectResponse(url="/admin/access?error=Username already exists", status_code=303)
            
        conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
        conn.commit()
    return RedirectResponse(url="/admin/access", status_code=303)

# --- Admin: Update Request Username ---
@router.post("/admin/access-requests/{request_id}/username")
async def update_request_username(request: Request, request_id: int, username: str = Form(...)):
    require_admin(request)
    with get_db_connection() as conn:
        conn.execute("UPDATE access_requests SET username = ? WHERE id = ?", (username, request_id))
        conn.commit()
    return RedirectResponse(url="/admin/access", status_code=303)

# Helper to render index with consistent data
def _render_index(request: Request, error: str = None):
    from app.infra.repository import SubscriptionRepository
    sub_repo = SubscriptionRepository()
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

    user = get_current_user(request)
    
    subs_with_links = []
    global_settings = get_global_settings()
    for sub in subs:
        # Get completed episodes for this subscription
        with get_db_connection() as conn:
            episodes = conn.execute(
                """SELECT title, pub_date as published_date, status FROM episodes 
                   WHERE subscription_id = ? AND status = 'completed'
                   ORDER BY pub_date DESC LIMIT 10""",
                (sub.id,)
            ).fetchall()
            
            # Get latest episode with AI summary
            latest_ep = conn.execute(
                """SELECT id, title, description, ai_summary FROM episodes 
                   WHERE subscription_id = ? AND status = 'completed'
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()
        
        latest_summary = None
        latest_description = None
        if latest_ep:
            latest_summary = latest_ep['ai_summary']
            latest_description = latest_ep['description']
        
        subs_with_links.append({
            "sub": sub,
            "links": generate_rss_links(request, sub, global_settings, user),
            "episodes": [dict(ep) for ep in episodes],
            "episode_count": len(episodes),
            "latest_ai_summary": latest_summary,
            "latest_description": latest_description
        })

    # Get queue data for dashboard display
    queue = ep_repo.get_queue()

    user = get_current_user(request)
    
    # Determine if AI is configured (DB Overrides/Augments Env)
    from app.core.config import settings
    config_warning = not any([
        settings.GEMINI_API_KEY,
        settings.OPENAI_API_KEY,
        settings.ANTHROPIC_API_KEY,
        settings.OPENROUTER_API_KEY,
        global_settings.get('gemini_api_key'),
        global_settings.get('openai_api_key'),
        global_settings.get('anthropic_api_key'),
        global_settings.get('openrouter_api_key')
    ])

    # Generate Unified Links if subscriptions exist
    unified_links = None
    if subs:
        # Determine Base URL using consolidated logic
        base_url = get_app_base_url(global_settings, request)
        
        rss_url = f"{base_url}/feed/unified.xml"
        
        # Inject Auth if Enabled
        auth_enabled_val = global_settings.get('enable_feed_auth')
        is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False
        
        if is_auth_enabled:
            # Determine credentials
            if global_settings.get('auth_enabled'):
                auth_user = user.username if user else "admin"
                auth_pass = request.session.get("user_pass", "")
            else:
                auth_user = global_settings.get('feed_auth_username')
                auth_pass = global_settings.get('feed_auth_password')
                
            if auth_user and auth_pass:
                import base64
                token = base64.b64encode(f"{auth_user}:{auth_pass}".encode()).decode()
                separator = "&" if "?" in rss_url else "?"
                rss_url = f"{rss_url}{separator}auth={token}"


        clean_url = rss_url.replace('https://', '').replace('http://', '')
        unified_links = {
            "rss": rss_url,
            "direct": rss_url,
            "apple": rss_url,  # Method 1: Direct HTTPS URL
            "pocket_casts": f"pktc://subscribe/{rss_url}",
            "overcast": f"overcast://x-callback-url/add?url={rss_url}",
            "castbox": f"castbox://subscribe?url={rss_url}",
            "podcast_addict": f"podcastaddict://subscribe/{rss_url}"
        }

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": user,
        "subscriptions": subs_with_links, 
        "stats": stats,
        "error": error,
        "config_warning": config_warning,
        "queue": queue,
        "unified_links": unified_links,
        "settings": global_settings
    })

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render_index(request)

from app.core.processor import Processor

# --- Admin: Global Subscription Settings ---
@router.get("/admin/global-subscription-settings", response_class=HTMLResponse)
async def admin_global_subscription_settings(request: Request):
    user = get_current_user(request)
    
    with get_db_connection() as conn:
        settings_row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        
    return templates.TemplateResponse("admin/global_subscription_settings.html", {
        "request": request,
        "user": user,
        "settings": settings_row,
        "active_tab": "global_subs"
    })

@router.post("/admin/global-subscription-settings/update")
async def update_global_subscription_settings(
    request: Request,
    default_remove_ads: bool = Form(False),
    default_remove_promos: bool = Form(False),
    default_remove_intros: bool = Form(False),
    default_remove_outros: bool = Form(False),
    default_ai_rewrite_description: bool = Form(False),
    default_ai_audio_summary: bool = Form(False),
    default_append_title_intro: bool = Form(False),
    default_retention_limit: int = Form(1),
    default_retention_days: int = Form(30),
    default_manual_retention_days: int = Form(14),
    default_custom_instructions: str = Form(None)
):
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings 
            SET default_remove_ads = ?, 
                default_remove_promos = ?, 
                default_remove_intros = ?, 
                default_remove_outros = ?, 
                default_ai_rewrite_description = ?,
                default_ai_audio_summary = ?, 
                default_append_title_intro = ?,
                default_retention_limit = ?,
                default_retention_days = ?,
                default_manual_retention_days = ?,
                default_custom_instructions = ?
            WHERE id = 1
        """, (
            default_remove_ads, default_remove_promos, default_remove_intros, default_remove_outros,
            default_ai_rewrite_description, default_ai_audio_summary, default_append_title_intro,
            default_retention_limit, default_retention_days, default_manual_retention_days,
            default_custom_instructions
        ))
        conn.commit()
        
    return RedirectResponse(url="/admin/global-subscription-settings?success=Settings updated", status_code=303)


@router.post("/add", response_class=HTMLResponse)
async def add_subscription(request: Request, background_tasks: BackgroundTasks, feed_url: str = Form(...), initial_count: int = Form(1)):
    try:
        # Check if exists (quick DB check)
        existing = sub_repo.get_by_url(feed_url)
        if existing:
            return _render_index(request, error="Subscription already exists")
        
        # Create subscription with placeholder data
        # Fetch global defaults first
        with get_db_connection() as conn:
            app_settings = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
            
        retention_limit = app_settings['default_retention_limit'] if app_settings['default_retention_limit'] is not None else initial_count
        
        # Override initial_count if user didn't explicitly change it? 
        # Actually user input 'initial_count' (which is technically retention limit in UI) should take precedence if exposed in UI,
        # but the request asks "when a user first hits a podcast, this would be what it defaults to".
        # The UI 'initial_count' field might be "Keep Latest 1". 
        # If the user selects something specific in the add form, that should prob win. 
        # If default is 1, and user sends 1, effectively it's the same.
        # But wait, usually 'Add Subscription' form is simple URL input. The 'initial_count' is likely hidden or default '1'.
        # Let's use the global default for retention if provided, or fallback to the form default.
        if 'default_retention_limit' in app_settings.keys() and app_settings['default_retention_limit'] is not None:
             retention_limit = app_settings['default_retention_limit']
        
        sub_create = SubscriptionCreate(feed_url=feed_url)
        new_sub = sub_repo.create(sub_create, "Loading...", f"loading-{int(__import__('time').time())}", None, "Fetching feed information...", retention_limit=retention_limit)
        
        # Apply other global defaults immediately
        sub_repo.update_settings(
            new_sub.id,
            remove_ads=bool(app_settings['default_remove_ads']),
            remove_promos=bool(app_settings['default_remove_promos']),
            remove_intros=bool(app_settings['default_remove_intros']),
            remove_outros=bool(app_settings['default_remove_outros']),
            custom_instructions=app_settings['default_custom_instructions'],
            append_summary=bool(app_settings['default_ai_audio_summary']), # Mapped correctly? Yes
            append_title_intro=bool(app_settings['default_append_title_intro']),
            ai_rewrite_description=bool(app_settings['default_ai_rewrite_description']),
            ai_audio_summary=bool(app_settings['default_ai_audio_summary']),
            retention_days=app_settings['default_retention_days'] or 30,
            manual_retention_days=app_settings['default_manual_retention_days'] or 14,
            retention_limit=retention_limit
        )

        
        # All heavy lifting happens in background
        async def setup_subscription(sub_id: int, url: str, limit: int):
            from app.core.processor import Processor
            from app.core.feed import FeedManager
            from app.infra.database import get_db_connection
            
            try:
                # Parse feed (network call)
                title, slug, image_url, description = FeedManager.parse_feed(url)
                
                # Update subscription with real data
                # Keep the settings we just set! Only update metadata.
                with get_db_connection() as conn:
                    conn.execute("""
                        UPDATE subscriptions 
                        SET title = ?, slug = ?, image_url = ?, description = ?
                        WHERE id = ?
                    """, (title, slug, image_url, description, sub_id))
                    conn.commit()
                
                # Now check feeds and process queue
                proc = Processor()
                await proc.check_feeds(subscription_id=sub_id, limit=limit)
                await proc.process_queue()
                
            except Exception as e:
                logger.error(f"Error setting up subscription {sub_id}: {e}")
        
        background_tasks.add_task(setup_subscription, new_sub.id, feed_url, retention_limit)
        
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        return _render_index(request, error=str(e))

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
    global_settings = get_global_settings()
    
    # Get current user for nav bar and links
    from app.web.auth import get_current_user
    user = get_current_user(request)

    links = generate_rss_links(request, sub, global_settings, user)

    return templates.TemplateResponse("episodes.html", {
        "request": request, 
        "user": user,
        "subscription": sub, 
        "episodes": episodes,
        "links": links,
        "basename": lambda p: p.split('/')[-1] if p else '',
        "format_duration": format_duration
    })

@router.post("/subscriptions/{id}/settings")
async def update_settings(
    id: int,
    background_tasks: BackgroundTasks,
    remove_ads: bool = Form(False),
    remove_promos: bool = Form(False),
    remove_intros: bool = Form(False),
    remove_outros: bool = Form(False),
    custom_instructions: str = Form(None),
    append_summary: bool = Form(False),
    append_title_intro: bool = Form(False),
    ai_rewrite_description: bool = Form(False),
    ai_audio_summary: bool = Form(False),
    retention_days: int = Form(30),
    manual_retention_days: int = Form(14),
    retention_limit: int = Form(1)
):
    sub_repo.update_settings(
        id, 
        remove_ads, 
        remove_promos, 
        remove_intros, 
        remove_outros, 
        custom_instructions, 
        append_summary, 
        append_title_intro,
        ai_rewrite_description,
        ai_audio_summary,
        retention_days,
        manual_retention_days,
        retention_limit
    )
    
    # Trigger processing if any ads/promos settings were changed
    from app.core.processor import Processor
    proc = Processor()
    
    async def post_update_tasks(sub_id):
        await proc.cleanup_old_episodes()
        await proc.check_feeds(sub_id)
        await proc.process_queue()

    background_tasks.add_task(post_update_tasks, id)
    return RedirectResponse(url=f"/subscriptions/{id}", status_code=303)

from fastapi.responses import FileResponse

@router.get("/artifacts/transcript/{id}")
async def get_transcript(id: int):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT e.guid, s.slug, e.transcript_path 
               FROM episodes e 
               JOIN subscriptions s ON e.subscription_id = s.id 
               WHERE e.id = ?""", 
            (id,)
        ).fetchone()
        
        if row:
            # Try new hierarchical structure first
            episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
            new_path = os.path.join(
                settings.get_episode_dir(row['slug'], episode_slug),
                "transcript.json"
            )
            if os.path.exists(new_path):
                return FileResponse(new_path)
            
            # Fallback to old path for backward compatibility
            if row['transcript_path'] and os.path.exists(row['transcript_path']):
                return FileResponse(row['transcript_path'])
                
    raise HTTPException(status_code=404, detail="Transcript not found")

@router.get("/artifacts/report/{id}")
async def get_report(id: int):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT e.guid, s.slug, e.report_path, e.ad_report_path 
               FROM episodes e 
               JOIN subscriptions s ON e.subscription_id = s.id 
               WHERE e.id = ?""", 
            (id,)
        ).fetchone()
        
        if row:
            episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
            episode_dir = settings.get_episode_dir(row['slug'], episode_slug)
            
            # Try new hierarchical structure first (prefer HTML)
            html_path = os.path.join(episode_dir, "report.html")
            if os.path.exists(html_path):
                return FileResponse(html_path)
            
            json_path = os.path.join(episode_dir, "report.json")
            if os.path.exists(json_path):
                return FileResponse(json_path)
            
            # Fallback to old paths for backward compatibility
            if row['report_path'] and os.path.exists(row['report_path']):
                return FileResponse(row['report_path'])
            if row['ad_report_path'] and os.path.exists(row['ad_report_path']):
                return FileResponse(row['ad_report_path'])
            
    raise HTTPException(status_code=404, detail="Report not found")

@router.get("/feeds/{slug}.xml")
async def get_individual_feed(slug: str, request: Request):
    """Serve individual podcast RSS feed with optional token injection for audio URLs."""
    from app.infra.repository import SubscriptionRepository
    from app.core.rss_gen import RSSGenerator
    from app.core.config import settings as app_settings
    from fastapi.responses import FileResponse, Response
    import base64
    import re
    
    sub_repo = SubscriptionRepository()
    sub = sub_repo.get_by_slug(slug)
    if not sub:
        raise HTTPException(status_code=404, detail="Feed not found")
        
    file_path = os.path.join(app_settings.FEEDS_DIR, f"{slug}.xml")
    if not os.path.exists(file_path):
        gen = RSSGenerator()
        gen.generate_feed(sub.id)
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Feed generation failed")
    
    # Check if we should inject auth tokens
    settings = get_global_settings()
    auth_enabled_val = settings.get('enable_feed_auth')
    is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False
    
    # Extract credentials from request (header or query param)
    username = None
    password = None
    
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Basic '):
        try:
            encoded = auth_header.split(' ')[1]
            decoded = base64.b64decode(encoded).decode('utf-8')
            username, password = decoded.split(':', 1)
        except:
            pass
    else:
        auth_param = request.query_params.get('auth')
        if auth_param:
            try:
                decoded = base64.b64decode(auth_param).decode('utf-8')
                username, password = decoded.split(':', 1)
            except:
                pass
    
    # If auth is enabled and we have credentials, inject token into audio URLs
    if is_auth_enabled and username and password:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            
            def inject_auth(match):
                url = match.group(2)
                separator = "&" if "?" in url else "?"
                return f'{match.group(1)}{url}{separator}auth={token}'

            xml_content = re.sub(r'(enclosure\s+url=")(https?://[^"]+)', inject_auth, xml_content)
            
            return Response(content=xml_content, media_type="application/xml")
        except Exception as e:
            logger.error(f"Error injecting auth into feed {slug}: {e}")
    
    return FileResponse(file_path, media_type="application/xml")

@router.get("/feed/unified")
@router.get("/feed/unified.xml")
async def get_unified_feed(request: Request):
    """Serve the unified RSS feed with optional authentication."""
    # Check Auth if enabled
    settings = get_global_settings()
    auth_enabled_val = settings.get('enable_feed_auth')
    is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False

    if is_auth_enabled:
        # Check Basic Auth header
        import base64
        auth_header = request.headers.get('Authorization')
        authorized = False
        
        if auth_header and auth_header.startswith('Basic '):
            try:
                encoded_creds = auth_header.split(' ')[1]
                decoded_creds = base64.b64decode(encoded_creds).decode('utf-8')
                username, password = decoded_creds.split(':', 1)
                
                if settings.get('auth_enabled'):
                     # Validate against app users
                    from app.infra.database import get_db_connection
                    from app.web.auth_utils import verify_password
                    with get_db_connection() as conn:
                        user_row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
                        if user_row and verify_password(password, user_row['password_hash']):
                            authorized = True
                else:
                    # Validate against standalone settings
                    expected_user = settings.get('feed_auth_username')
                    expected_pass = settings.get('feed_auth_password')
                    if username == expected_user and password == expected_pass:
                        authorized = True
            except Exception:
                pass
        
        if not authorized:
            headers = {"WWW-Authenticate": 'Basic realm="Podcast Ad Remover"'}
            raise HTTPException(status_code=401, detail="Unauthorized", headers=headers)

    from fastapi.responses import FileResponse, Response
    from app.core.config import settings as app_settings
    
    file_path = os.path.join(app_settings.FEEDS_DIR, "unified.xml")
    if not os.path.exists(file_path):
        # Generate on demand if missing
        from app.core.rss_gen import RSSGenerator
        gen = RSSGenerator()
        gen.generate_unified_feed()
    
    # If auth is enabled, inject credentials into the XML on the fly
    if is_auth_enabled and authorized:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            
            # Inject credentials into enclosure URLs
            # username and password are available from the Basic Auth block
            if username and password:
                # Use token auth: append ?auth=base64(user:pass) to audio URLs
                import re
                token = base64.b64encode(f"{username}:{password}".encode()).decode()
                
                def inject_auth(match):
                    url = match.group(2)
                    separator = "&" if "?" in url else "?"
                    return f'{match.group(1)}{url}{separator}auth={token}'

                # Find enclosure url="...", capture the prefix and the URL
                xml_content = re.sub(r'(enclosure\s+url=")(https?://[^"]+)', inject_auth, xml_content)

                
            return Response(content=xml_content, media_type="application/xml")
        except Exception as e:
            logger.error(f"Error injecting credentials into unified feed: {e}")
            # Fallback to static file if injection fails
    
    return FileResponse(file_path, media_type="application/xml")
