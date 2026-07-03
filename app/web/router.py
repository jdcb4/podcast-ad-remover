from fastapi import APIRouter, Request, Form, Depends, BackgroundTasks, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.infra.repository import ApiTokenRepository, SubscriptionRepository, EpisodeRepository, FeedTokenRepository
from app.core.feed import FeedManager
from app.core.models import SubscriptionCreate
from app.core.system_status import get_operation_status
from app.core.url_utils import validate_http_url
from app.core.notifications import (
    EVENT_ACCESS_REQUEST,
    EVENT_BREAKING_ERROR,
    EVENT_NEW_PODCAST,
    send_notification,
    send_notification_async,
    send_test_notification,
)
from app.web.auth import get_current_user, require_auth, require_admin, log_login_attempt, SESSION_USER_KEY
from app.web.auth_utils import hash_password, verify_feed_password, verify_password, generate_secure_password, get_client_ip
from app.web.rate_limiter import login_rate_limiter, check_rate_limit
from app.web.subscription_links import build_subscribe_instruction_context, build_subscription_links
from app.web.template_filters import compact_datetime
from app.web.template_filters import clean_description as safe_clean_description
from app.web.template_filters import simple_markdown as safe_simple_markdown
from app.infra.database import get_db_connection
from app.core.config import is_default_session_secret, settings as runtime_settings
from datetime import datetime
import os
import logging
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)

router = APIRouter()
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Helper to get CSP nonce from request
def get_csp_nonce(request: Request) -> str:
    """Extract CSP nonce from request state (set by SecurityHeadersMiddleware)"""
    return getattr(request.state, 'csp_nonce', '')


templates.env.filters['simple_markdown'] = safe_simple_markdown
templates.env.filters['clean_description'] = safe_clean_description
templates.env.filters['compact_datetime'] = compact_datetime

sub_repo = SubscriptionRepository()
ep_repo = EpisodeRepository()
feed_token_repo = FeedTokenRepository()
api_token_repo = ApiTokenRepository()


def _append_feed_access_to_enclosures(xml_content: str, param_name: str, token_value: str) -> str:
    """Append a feed access query parameter to enclosure URLs in RSS XML."""
    encoded_value = quote(token_value, safe="")

    def inject_auth(match):
        url = match.group(2)
        separator = "&amp;" if "?" in url else "?"
        return f'{match.group(1)}{url}{separator}{param_name}={encoded_value}'

    return re.sub(r'(enclosure\s+url=")(https?://[^"]+)', inject_auth, xml_content)


def _safe_local_redirect(target: str | None, fallback: str) -> str:
    """Allow redirects only to local app paths."""
    if not target:
        return fallback
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        return fallback
    return target

# Helper to get settings
def get_global_settings():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        if row:
            return dict(row)
    return {}


def _real_user_id(user) -> int | None:
    user_id = getattr(user, "id", None)
    return user_id if user_id and user_id > 0 else None


def _can_manage_subscription(user, sub) -> bool:
    if not user:
        return False
    if getattr(user, "is_admin", False):
        return True
    user_id = _real_user_id(user)
    return bool(user_id and getattr(sub, "owner_user_id", None) == user_id)

from app.core.utils import get_app_base_url


def generate_rss_links(request: Request, sub, global_settings: dict, user_obj=None, include_auth_token: bool = True):
    """Consolidated logic for generating RSS links with optional auth injection."""
    base_url = get_app_base_url(global_settings, request)
    
    rss_url = f"{base_url}/feeds/{sub.slug}.xml"
    
    # Inject Auth if Enabled
    auth_enabled_val = global_settings.get('enable_feed_auth')
    is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False
    
    if is_auth_enabled and include_auth_token:
        token = get_or_create_feed_token(request, user_obj)
        if token:
            separator = "&" if "?" in rss_url else "?"
            rss_url = f"{rss_url}{separator}token={token}"

    return build_subscription_links(rss_url)

# Helper to get pending access requests count for sidebar badge
def get_pending_requests_count():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM access_requests WHERE status = 'pending'").fetchone()
        return result[0] if result else 0


def get_setup_status(request: Request, global_settings: dict) -> dict:
    with get_db_connection() as conn:
        admin_count = conn.execute("SELECT COUNT(*) AS count FROM users WHERE is_admin = 1").fetchone()["count"]
        subscription_count = conn.execute("SELECT COUNT(*) AS count FROM subscriptions").fetchone()["count"]

    base_url = get_app_base_url(global_settings, request)
    feed_auth_enabled = str(global_settings.get("enable_feed_auth")).lower() in ("1", "true", "yes", "on")
    public_subscribe_enabled = str(global_settings.get("public_subscribe_page_enabled")).lower() in ("1", "true", "yes", "on")
    auth_enabled = bool(global_settings.get("auth_enabled"))
    security_warnings = []
    if is_default_session_secret():
        security_warnings.append(
            "SESSION_SECRET_KEY is still using the default value. "
            "Set a unique SESSION_SECRET_KEY before enabling dashboard or feed authentication."
        )
    if (auth_enabled or feed_auth_enabled) and base_url.lower().startswith("https://") and not runtime_settings.COOKIE_SECURE:
        security_warnings.append(
            "COOKIE_SECURE is false while authentication is enabled on an HTTPS base URL. "
            "Set COOKIE_SECURE=true when users access this app through HTTPS."
        )

    return {
        "admin_count": admin_count,
        "has_admin": admin_count > 0,
        "auth_enabled": auth_enabled,
        "base_url": base_url,
        "health_url": f"{base_url}/health",
        "subscribe_url": f"{base_url}/subscribe",
        "unified_feed_url": f"{base_url}/feed/unified.xml",
        "subscription_count": subscription_count,
        "feed_auth_enabled": feed_auth_enabled,
        "public_subscribe_enabled": public_subscribe_enabled,
        "security_warnings": security_warnings,
    }

# --- Authentication Routes ---
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Display login page or first-time setup."""
    with get_db_connection() as conn:
        settings = conn.execute(
            "SELECT auth_enabled, initial_password, public_subscribe_page_enabled FROM app_settings WHERE id = 1"
        ).fetchone()
        user_count = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()['count']
    
    # Check if this is first launch
    first_launch = user_count == 0
    
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "first_launch": first_launch,
            "initial_password": settings['initial_password'] if settings else None,
            "auth_enabled": settings['auth_enabled'] if settings else False,
            "public_subscribe_page_enabled": settings['public_subscribe_page_enabled'] if settings else True,
        }
    )

@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login submission with rate limiting protection."""
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")
    public_subscribe_page_enabled = True
    with get_db_connection() as conn:
        settings_row = conn.execute(
            "SELECT public_subscribe_page_enabled FROM app_settings WHERE id = 1"
        ).fetchone()
        if settings_row:
            public_subscribe_page_enabled = bool(settings_row["public_subscribe_page_enabled"])
    
    # Check rate limit before processing login
    try:
        check_rate_limit(client_ip)
    except HTTPException as e:
        # Return user-friendly error page instead of raw exception
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "csp_nonce": get_csp_nonce(request),
                "error": e.detail,
                "first_launch": False,
                "auth_enabled": True,
                "rate_limited": True,
                "public_subscribe_page_enabled": public_subscribe_page_enabled,
            },
            status_code=e.status_code
        )
    
    with get_db_connection() as conn:
        user_row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    
    if not user_row or not verify_password(password, user_row['password_hash']):
        # Record failed attempt and check if now locked
        is_locked = login_rate_limiter.record_failed_attempt(client_ip)
        log_login_attempt(username, client_ip, False, user_agent)
        
        error_msg = "Invalid username or password"
        if is_locked:
            error_msg = f"Too many failed login attempts. Your IP has been locked for {login_rate_limiter.lockout_seconds // 60} minutes."
        
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "csp_nonce": get_csp_nonce(request),
                "error": error_msg,
                "first_launch": False,
                "auth_enabled": True,
                "rate_limited": is_locked,
                "public_subscribe_page_enabled": public_subscribe_page_enabled,
            }
        )
    
    # Successful login - clear rate limiting for this IP
    login_rate_limiter.record_successful_login(client_ip)
    log_login_attempt(username, client_ip, True, user_agent)
    
    # Update last login
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_row['id']))
        conn.commit()
    
    # Set session
    request.session.pop("user_pass", None)
    request.session[SESSION_USER_KEY] = user_row['id']
    
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
    
    return templates.TemplateResponse(
        request=request,
        name="change_password.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "required": settings['require_password_change'] if settings else False
        }
    )

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
        return templates.TemplateResponse(
            request=request,
            name="change_password.html",
            context={
                "csp_nonce": get_csp_nonce(request),
                "user": user,
                "error": "Passwords do not match"
            }
        )
    
    # Verify current password
    with get_db_connection() as conn:
        user_row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()
    
    if not verify_password(current_password, user_row['password_hash']):
        return templates.TemplateResponse(
            request=request,
            name="change_password.html",
            context={
                "csp_nonce": get_csp_nonce(request),
                "user": user,
                "error": "Current password is incorrect"
            }
        )
    
    # Update password
    new_hash = hash_password(new_password)
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user.id))
        conn.execute("UPDATE app_settings SET require_password_change = 0, initial_password = NULL WHERE id = 1")
        conn.commit()
    
    request.session.pop("user_pass", None)
    
    return RedirectResponse(url="/admin/system?password_changed=1", status_code=status.HTTP_302_FOUND)

@router.get("/request-access", response_class=HTMLResponse)
async def request_access_page(request: Request):
    """Display access request form."""
    return templates.TemplateResponse(
        request=request,
        name="request_access.html",
        context={"csp_nonce": get_csp_nonce(request)}
    )

@router.post("/submit-access-request")
async def submit_access_request(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    email: str = Form(None),
    reason: str = Form(None)
):
    """Handle access request submission."""
    client_ip = get_client_ip(request)
    username = username.strip()

    def render_error(message: str):
        return templates.TemplateResponse(
            request=request,
            name="request_access.html",
            context={
                "csp_nonce": get_csp_nonce(request),
                "error": message,
                "username": username,
                "email": email,
                "reason": reason,
            },
            status_code=400,
        )

    if len(username) < 3:
        return render_error("Username must be at least 3 characters.")
    if password != confirm_password:
        return render_error("Passwords do not match.")
    if len(password) < 8:
        return render_error("Password must be at least 8 characters.")
    
    with get_db_connection() as conn:
        existing_user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing_user:
            return render_error("Username already exists.")

        existing_request = conn.execute(
            "SELECT id FROM access_requests WHERE username = ? AND status = 'pending'",
            (username,),
        ).fetchone()
        if existing_request:
            return render_error("There is already a pending request for that username.")

        conn.execute(
            "INSERT INTO access_requests (username, email, reason, password_hash, ip_address) VALUES (?, ?, ?, ?, ?)",
            (username, email, reason, hash_password(password), client_ip)
        )
        conn.commit()

    send_notification(
        EVENT_ACCESS_REQUEST,
        "Podcast Ad Remover access request",
        f"{username} requested dashboard access.",
        severity="info",
    )
    
    return templates.TemplateResponse(
        request=request,
        name="request_access.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "success": "Your access request has been submitted. You can use your chosen password after an administrator approves the request."
        }
    )

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
    global_settings = get_global_settings()
    new_api_token = request.session.pop("new_api_token", None)
    return templates.TemplateResponse(
        request=request,
        name="admin/system.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "settings": global_settings,
            "active_api_tokens": api_token_repo.list_active(),
            "active_users": _active_users(),
            "new_api_token": new_api_token,
            "setup_status": get_setup_status(request, global_settings),
            "pending_requests_count": get_pending_requests_count(),
            "active_tab": "system"
        }
    )


@router.post("/admin/setup/admin-user")
async def create_setup_admin_user(
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    user = Depends(require_admin)
):
    if is_default_session_secret():
        return RedirectResponse(
            url="/admin/system?error=Set+SESSION_SECRET_KEY+before+enabling+login",
            status_code=303,
        )

    if password != confirm_password:
        return RedirectResponse(url="/admin/system?error=Passwords+do+not+match", status_code=303)
    if len(password) < 8:
        return RedirectResponse(url="/admin/system?error=Password+must+be+at+least+8+characters", status_code=303)

    with get_db_connection() as conn:
        existing_admin = conn.execute("SELECT COUNT(*) AS count FROM users WHERE is_admin = 1").fetchone()["count"]
        if existing_admin:
            return RedirectResponse(url="/admin/system?error=Admin+user+already+exists", status_code=303)

        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
            (username, hash_password(password)),
        )
        conn.execute("""
            UPDATE app_settings
            SET auth_enabled = 1,
                require_password_change = 0,
                initial_password = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """)
        conn.commit()

    return RedirectResponse(url="/login?success=Admin+account+created", status_code=303)

@router.post("/admin/system/update")
async def update_system_settings(
    request: Request,
    concurrent_downloads: int = Form(2),
    retention_days: int = Form(30),
    check_interval_minutes: int = Form(60),
    whisper_cpu_threads: int = Form(0),
    ffmpeg_threads: int = Form(0),
    unload_whisper_after_job: bool = Form(False),
    ai_api_enabled: bool = Form(False),
    ai_api_default_requests_per_minute: int = Form(60),
    ai_api_default_requests_per_day: int = Form(1000),
    ai_api_unauth_requests_per_minute: int = Form(10),
    app_external_url: str = Form(None),
    auth_enabled: bool = Form(False),
    ip_allowlist: str = Form(None),
    enable_feed_auth: bool = Form(False),
    feed_auth_username: str = Form(None),
    feed_auth_password: str = Form(None),
    public_subscribe_page_enabled: bool = Form(False),
    whitelist_mode: str = Form(None),
    redirect_to: str = Form(None),
    admin_user = Depends(require_admin)
):
    from app.infra.database import get_db_connection

    if (auth_enabled or enable_feed_auth) and is_default_session_secret():
        url = _safe_local_redirect(redirect_to, "/admin/system")
        separator = "&" if "?" in url else "?"
        return RedirectResponse(
            url=f"{url}{separator}error=Set+SESSION_SECRET_KEY+before+enabling+dashboard+or+feed+authentication",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    
    # Standalone feed password is retained for Basic Auth compatibility.
    # Generated podcast links now prefer feed tokens instead.
    hashed_feed_password = hash_password(feed_auth_password) if feed_auth_password else None

    with get_db_connection() as conn:
        # Get current settings
        current_settings = conn.execute(
            "SELECT auth_enabled, feed_auth_username, feed_auth_password, whitelist_mode FROM app_settings WHERE id = 1"
        ).fetchone()
        
        # Get current feed password if not changing
        if enable_feed_auth and not feed_auth_password:
            hashed_feed_password = current_settings['feed_auth_password'] if current_settings else None
        effective_feed_username = feed_auth_username or (current_settings['feed_auth_username'] if current_settings else None)

        # Backend validation: if feed auth is enabled without dashboard auth, standalone credentials are required.
        if enable_feed_auth and not auth_enabled:
            if not effective_feed_username or not (feed_auth_password or hashed_feed_password):
                url = _safe_local_redirect(redirect_to, "/admin/system")
                separator = "&" if "?" in url else "?"
                return RedirectResponse(
                    url=f"{url}{separator}error=Standalone+feed+authentication+requires+a+username+and+password",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
        
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
        
        current_whitelist_mode = current_settings["whitelist_mode"] if current_settings and "whitelist_mode" in current_settings.keys() else 0
        next_whitelist_mode = (
            current_whitelist_mode
            if whitelist_mode is None
            else 1 if str(whitelist_mode).lower() in {"1", "true", "yes", "on"} else 0
        )

        # Update settings
        whisper_cpu_threads = max(0, min(64, whisper_cpu_threads or 0))
        ffmpeg_threads = max(0, min(64, ffmpeg_threads or 0))
        ai_api_default_requests_per_minute = max(1, min(10000, ai_api_default_requests_per_minute or 60))
        ai_api_default_requests_per_day = max(1, min(1000000, ai_api_default_requests_per_day or 1000))
        ai_api_unauth_requests_per_minute = max(1, min(1000, ai_api_unauth_requests_per_minute or 10))

        conn.execute("""
            UPDATE app_settings SET concurrent_downloads = ?,
                retention_days = ?,
                check_interval_minutes = ?,
                whisper_cpu_threads = ?,
                ffmpeg_threads = ?,
                unload_whisper_after_job = ?,
                ai_api_enabled = ?,
                ai_api_default_requests_per_minute = ?,
                ai_api_default_requests_per_day = ?,
                ai_api_unauth_requests_per_minute = ?,
                app_external_url = ?,
                auth_enabled = ?,
                ip_allowlist = ?,
                enable_feed_auth = ?,
                feed_auth_username = ?,
                feed_auth_password = COALESCE(?, feed_auth_password),
                public_subscribe_page_enabled = ?,
                whitelist_mode = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (concurrent_downloads, retention_days, check_interval_minutes,
              whisper_cpu_threads, ffmpeg_threads, 1 if unload_whisper_after_job else 0,
              1 if ai_api_enabled else 0,
              ai_api_default_requests_per_minute,
              ai_api_default_requests_per_day,
              ai_api_unauth_requests_per_minute,
              app_external_url,
              1 if auth_enabled else 0, ip_allowlist,
              1 if enable_feed_auth else 0, 
              effective_feed_username if enable_feed_auth else (feed_auth_username if feed_auth_username else None),
              hashed_feed_password if hashed_feed_password else None,
              1 if public_subscribe_page_enabled else 0,
              next_whitelist_mode))
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
    
    url = _safe_local_redirect(redirect_to, "/admin/system?success=System+settings+updated")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/admin/api-tokens")
async def create_api_token(
    request: Request,
    name: str = Form(...),
    user_id: int = Form(...),
    scopes: list[str] = Form(["read"]),
    requests_per_minute: int = Form(None),
    requests_per_day: int = Form(None),
    admin_user = Depends(require_admin),
):
    with get_db_connection() as conn:
        token_user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if not token_user:
            return RedirectResponse(url="/admin/system?error=Select+a+valid+user+for+the+API+token", status_code=303)

    allowed_scopes = ApiTokenRepository.VALID_SCOPES
    selected_scopes = [scope for scope in scopes if scope in allowed_scopes]
    if not selected_scopes:
        selected_scopes = ["read"]

    token = api_token_repo.create(
        name=name,
        scopes=selected_scopes,
        user_id=user_id,
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day,
    )
    request.session["new_api_token"] = token
    return RedirectResponse(url="/admin/system?success=API+token+created", status_code=303)


@router.post("/admin/api-tokens/{token_id}/revoke")
async def revoke_api_token(token_id: int, admin_user = Depends(require_admin)):
    revoked = api_token_repo.revoke_by_id(token_id)
    if not revoked:
        return RedirectResponse(url="/admin/system?error=API+token+not+found", status_code=303)
    return RedirectResponse(url="/admin/system?success=API+token+revoked", status_code=303)


def _render_admin_ai(request: Request, ai_section: str):
    from app.core.config import settings
    
    # helper to check which env vars are set
    env_keys = {
        "GEMINI_API_KEY": bool(settings.GEMINI_API_KEY),
        "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
        "ANTHROPIC_API_KEY": bool(settings.ANTHROPIC_API_KEY),
        "OPENROUTER_API_KEY": bool(settings.OPENROUTER_API_KEY)
    }

    user = get_current_user(request)

    return templates.TemplateResponse(
        request=request,
        name="admin/ai.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "settings": get_global_settings(),
            "pending_requests_count": get_pending_requests_count(),
            "active_tab": ai_section,
            "ai_section": ai_section,
            "env_keys": env_keys
        }
    )


# --- Admin: AI ---
@router.get("/admin/ai", response_class=RedirectResponse)
async def admin_ai_redirect():
    return RedirectResponse(url="/admin/ai/text-analysis")

@router.get("/admin/ai/transcription", response_class=HTMLResponse)
async def admin_ai_transcription(request: Request):
    return _render_admin_ai(request, "ai_transcription")

@router.get("/admin/ai/voice", response_class=HTMLResponse)
async def admin_ai_voice(request: Request):
    return _render_admin_ai(request, "ai_voice")

@router.get("/admin/ai/text-analysis", response_class=HTMLResponse)
async def admin_ai_text_analysis(request: Request):
    return _render_admin_ai(request, "ai_text")

@router.post("/admin/ai/update")
async def update_ai_settings(
    request: Request,
    whisper_model: str = Form("base"),
    whisper_compute_type: str = Form("float32"),
    ai_model_cascade: str = Form(...),
    piper_model: str = Form("en_GB-cori-high.onnx"),
    tts_provider: str = Form("piper"),
    gemini_tts_voice: str = Form("Orus"),
    gemini_tts_model_cascade: str = Form('["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]'),
    active_ai_provider: str = Form("gemini"),
    openai_api_key: str = Form(None),
    openai_base_url: str = Form(None),
    anthropic_api_key: str = Form(None),
    openrouter_api_key: str = Form(None),
    gemini_api_keys: str = Form(None),
    openai_model: str = Form("gpt-4o"),
    anthropic_model: str = Form("claude-3-5-sonnet"),
    openrouter_model: str = Form('["google/gemini-3.5-flash", "google/gemini-3-flash", "google/gemini-3.1-flash-lite", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]'),
    chunk_num_chunks: int = Form(10),
    chunk_overlap_percent: int = Form(25),
    include_reason: int = Form(None),
    redirect_to: str = Form("/admin/ai/text-analysis"),
    admin_user = Depends(require_admin)
):
    from app.infra.database import get_db_connection
    import json
    try:
        json.loads(ai_model_cascade)
    except:
        ai_model_cascade = '["gemini-3.5-flash", "gemini-3-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]'

    try:
        parsed_tts_models = json.loads(gemini_tts_model_cascade)
        if not isinstance(parsed_tts_models, list):
            raise ValueError("Gemini TTS cascade must be a list")
    except:
        gemini_tts_model_cascade = '["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]'

    if tts_provider not in {"piper", "gemini"}:
        tts_provider = "piper"
    if gemini_tts_voice not in {"Orus", "Enceladus", "Laomedeia"}:
        gemini_tts_voice = "Orus"
    if whisper_compute_type not in {"int8", "int8_float16", "float16", "bfloat16", "float32"}:
        whisper_compute_type = "float32"

    # Validate chunking parameters
    if chunk_num_chunks < 2 or chunk_num_chunks > 50:
        chunk_num_chunks = 10
    if chunk_overlap_percent < 0 or chunk_overlap_percent > 75:
        chunk_overlap_percent = 25

    # Handle checkbox: if not sent (None), it's unchecked (0). If sent, it's checked (1)
    if include_reason is None:
        include_reason = 0
    else:
        include_reason = 1
    
    # Validate gemini_api_keys is valid JSON array
    if gemini_api_keys:
        try:
            parsed_keys = json.loads(gemini_api_keys)
            if not isinstance(parsed_keys, list):
                gemini_api_keys = "[]"
        except:
            gemini_api_keys = "[]"
    else:
        gemini_api_keys = "[]"

    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings
            SET whisper_model = ?,
                whisper_compute_type = ?,
                ai_model_cascade = ?,
                piper_model = ?,
                tts_provider = ?,
                gemini_tts_voice = ?,
                gemini_tts_model_cascade = ?,
                active_ai_provider = ?,
                openai_api_key = ?,
                openai_base_url = ?,
                anthropic_api_key = ?,
                openrouter_api_key = ?,
                gemini_api_keys = ?,
                openai_model = ?,
                anthropic_model = ?,
                openrouter_model = ?,
                chunk_num_chunks = ?,
                chunk_overlap_percent = ?,
                include_reason = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (
            whisper_model, whisper_compute_type, ai_model_cascade, piper_model,
            tts_provider, gemini_tts_voice, gemini_tts_model_cascade,
            active_ai_provider,
            openai_api_key, openai_base_url, anthropic_api_key, openrouter_api_key, gemini_api_keys,
            openai_model, anthropic_model, openrouter_model,
            chunk_num_chunks, chunk_overlap_percent, include_reason
        ))
        conn.commit()
    return RedirectResponse(url=_safe_local_redirect(redirect_to, "/admin/ai/text-analysis"), status_code=303)

@router.post("/admin/ai/test")
async def test_ai_connection(
    provider: str = Form(...),
    api_key: str = Form(None),
    model: str = Form(None),
    admin_user = Depends(require_admin)
):
    try:
        from app.core.ai_services import AdDetector
        detector = AdDetector()
        
        prov_instance = detector.create_provider(provider, api_key=api_key, model=model)
        result = prov_instance.test_connection()

        # Return the dictionary returned by test_connection() directly!
        # It is already formatted as {"status": "ok", "response": "Hello!"}
        # or {"status": "error", "error": "message"}
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/admin/ai/refresh/{provider}")
async def refresh_models(
    provider: str,
    api_key: str = None,
    base_url: str = None,
    admin_user = Depends(require_admin)
):
    try:
        from app.core.ai_services import AdDetector, OpenAIProvider
        if provider == "gemini_tts":
            return {"models": AdDetector.DEFAULT_GEMINI_TTS_MODELS}
        detector = AdDetector()
        # Create provider using provided credentials or fall back to saved settings
        prov_instance = detector.create_provider(provider, api_key=api_key)
        # Override base_url and model_prefixes if provided (for OpenAI with custom endpoint)
        if base_url and provider == 'openai' and isinstance(prov_instance, OpenAIProvider):
            # Reinitialize client with custom base_url and disable model prefix filtering
            import openai
            key = prov_instance.api_keys[0] if prov_instance.api_keys else api_key
            # Some local LLMs don't require API keys, use empty string if none provided
            if not key:
                key = "sk-placeholder"  # OpenAI SDK requires a non-empty key
            prov_instance.client = openai.OpenAI(api_key=key, base_url=base_url)
            prov_instance.model_prefixes = None  # Disable filtering for local LLMs
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

    return templates.TemplateResponse(
        request=request,
        name="admin/prompts.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "settings": get_global_settings(),
            "default_prompts": default_prompts,
            "pending_requests_count": get_pending_requests_count(),
            "active_tab": "prompts"
        }
    )

@router.post("/admin/prompts")
async def save_prompts(request: Request, admin_user = Depends(require_admin)):
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
async def reset_prompts(request: Request, admin_user = Depends(require_admin)):
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
    recently_processed = ep_repo.get_recently_processed(days=3)
    operation_status = get_operation_status()
    return templates.TemplateResponse(
        request=request,
        name="admin/queue.html",
        context={
            "user": user,
            "queue": queue,
            "recently_processed": recently_processed,
            "operation_status": operation_status,
            "pending_requests_count": get_pending_requests_count(),
            "active_tab": "queue",
            "csp_nonce": get_csp_nonce(request),
        }
    )

@router.get("/api/queue/status")
async def api_queue_status(user = Depends(require_auth)):
    return {
        "queue": ep_repo.get_queue(),
        "recently_processed": ep_repo.get_recently_processed(days=3),
        "operation_status": get_operation_status(),
    }


def get_or_create_feed_token(request: Request, user_obj=None) -> str:
    """Return the current session's feed token, creating one if needed."""
    session_token = request.session.get("feed_token")
    if session_token and feed_token_repo.validate(session_token):
        return session_token

    user_id = user_obj.id if user_obj and user_obj.id and user_obj.id > 0 else None
    token = feed_token_repo.create(user_id=user_id, name="Generated subscription links")
    request.session["feed_token"] = token
    return token


@router.post("/admin/feed-token/regenerate")
async def regenerate_feed_token(request: Request, user = Depends(require_admin)):
    current_token = request.session.get("feed_token")
    if current_token:
        feed_token_repo.revoke(current_token)
    new_token = feed_token_repo.create(user_id=user.id if user and user.id and user.id > 0 else None, name="Generated subscription links")
    request.session["feed_token"] = new_token
    return RedirectResponse(url="/admin/feed-access?success=Feed+token+regenerated", status_code=303)


@router.post("/admin/feed-token/{token_id}/revoke")
async def revoke_feed_token(token_id: int, admin_user = Depends(require_admin)):
    revoked = feed_token_repo.revoke_by_id(token_id)
    if not revoked:
        return RedirectResponse(url="/admin/feed-access?error=Feed+token+not+found", status_code=303)
    return RedirectResponse(url="/admin/feed-access?success=Feed+token+revoked", status_code=303)

@router.post("/admin/queue/cancel/{episode_id}")
async def cancel_episode(episode_id: int, admin_user = Depends(require_admin)):
    from app.infra.repository import JobRepository

    JobRepository().cancel_active_for_episode(episode_id)
    ep_repo.reset_status(episode_id)
    return RedirectResponse(url="/admin/queue?success=Queued+job+cancelled", status_code=303)

@router.post("/admin/queue/retry/{episode_id}")
async def retry_episode(episode_id: int, admin_user = Depends(require_admin)):
    # Check if already processing?
    status = ep_repo.get_status(episode_id)
    if status == 'processing':
         return RedirectResponse(url="/admin/queue", status_code=303)
         
    # Force to pending (Background processor will pick it up)
    from app.core.processor import Processor
    proc = Processor()
    await proc.version_episode(episode_id)
    ep_repo.update_status(episode_id, "pending")
    await proc.process_queue()
    return RedirectResponse(url="/admin/queue", status_code=303)

@router.post("/api/episodes/{episode_id}/reprocess")
async def api_reprocess_episode(episode_id: int, skip_transcription: bool = False, user = Depends(require_auth)):
    import json
    logger.info(f"Reprocess request for {episode_id} with skip_transcription={skip_transcription}")
    
    # API version of retry - force status to pending
    current_status = ep_repo.get_status(episode_id)
    if current_status == 'processing':
         return {"status": "ignored", "reason": "already_processing"}
    
    # Set processing flags (like subscriptions.py does)
    flags = {'skip_transcription': skip_transcription}
    flags_json = json.dumps(flags)
    
    # Reset status with flags so processor respects skip_transcription
    from app.core.processor import Processor
    proc = Processor()
    await proc.version_episode(episode_id)
    ep_repo.reset_status(episode_id, processing_flags=flags_json)
    ep_repo.update_status(episode_id, "pending")
    await proc.process_queue()
    return {"status": "ok"}

@router.post("/api/episodes/{episode_id}/ignore")
async def api_ignore_episode(episode_id: int, user = Depends(require_auth)):
    # API version of cancel/delete - soft delete
    from app.core.processor import Processor
    proc = Processor()
    await proc.delete_episode(episode_id)
    return {"status": "ok"}

@router.post("/episodes/{episode_id}/download")
async def manual_download_episode(episode_id: int, request: Request, user = Depends(require_auth)):
    # Update DB to pending
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET is_manual_download=1 WHERE id=?", (episode_id,))
        conn.commit()

    ep_repo.update_status(episode_id, "pending")

    from app.core.processor import Processor
    proc = Processor()
    await proc.process_queue()

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
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
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

    return templates.TemplateResponse(
        request=request,
        name="admin/logs.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "logs": logs,
            "pending_requests_count": get_pending_requests_count(),
            "active_tab": "logs",
            "current_lines": lines,
            "current_level": level
        }
    )

# --- Admin: Access ---

@router.get("/subscribe/apple", response_class=HTMLResponse)
async def apple_subscribe_page(request: Request, url: str):
    """Render the Apple Podcasts subscription instruction page."""
    context = build_subscribe_instruction_context("apple", url)
    return templates.TemplateResponse(
        request=request,
        name="subscribe_instructions.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "client": context,
        },
    )


@router.get("/subscribe/{client_key}", response_class=HTMLResponse)
async def subscribe_instruction_page(request: Request, client_key: str, url: str):
    context = build_subscribe_instruction_context(client_key, url)
    if not context:
        raise HTTPException(status_code=404, detail="Subscription client not found")
    return templates.TemplateResponse(
        request=request,
        name="subscribe_instructions.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "client": context,
        },
    )


def _admin_context(request: Request, active_tab: str) -> dict:
    settings = get_global_settings()
    return {
        "csp_nonce": get_csp_nonce(request),
        "user": get_current_user(request),
        "active_tab": active_tab,
        "settings": settings,
        "app_base_url": get_app_base_url(settings, request),
        "pending_requests_count": get_pending_requests_count(),
    }


def _recent_login_history() -> list[dict]:
    from datetime import datetime, timedelta

    with get_db_connection() as conn:
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        rows = conn.execute(
            """SELECT * FROM login_attempts 
               WHERE timestamp > ? 
               ORDER BY timestamp DESC 
               LIMIT 100""",
            (thirty_days_ago,)
        ).fetchall()
        return [dict(row) for row in rows]


def _active_users() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def _pending_access_requests() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM access_requests WHERE status = 'pending' ORDER BY requested_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


@router.get("/admin/access", response_class=RedirectResponse)
async def admin_access_legacy_redirect():
    return RedirectResponse(url="/admin/access-requests", status_code=status.HTTP_302_FOUND)


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request):
    context = _admin_context(request, "users")
    context.update({
        "active_users": _active_users(),
        "login_history": _recent_login_history(),
    })

    return templates.TemplateResponse(
        request=request,
        name="admin/users.html",
        context=context,
    )


@router.post("/admin/users")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    is_admin: bool = Form(False),
    admin_user = Depends(require_admin),
):
    cleaned_username = username.strip()
    if len(cleaned_username) < 3:
        return RedirectResponse(url="/admin/users?error=Username+must+be+at+least+3+characters", status_code=303)
    if password != confirm_password:
        return RedirectResponse(url="/admin/users?error=Passwords+do+not+match", status_code=303)
    if len(password) < 8:
        return RedirectResponse(url="/admin/users?error=Password+must+be+at+least+8+characters", status_code=303)

    with get_db_connection() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (cleaned_username,)).fetchone()
        if existing:
            return RedirectResponse(url="/admin/users?error=Username+already+exists", status_code=303)

        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (cleaned_username, hash_password(password), 1 if is_admin else 0),
        )
        conn.commit()

    return RedirectResponse(url=f"/admin/users?success=User+created+for+{quote(cleaned_username)}", status_code=303)


@router.get("/admin/access-requests", response_class=HTMLResponse)
async def admin_access_requests(request: Request):
    context = _admin_context(request, "access_requests")
    context.update({
        "pending_requests": _pending_access_requests(),
    })
    
    return templates.TemplateResponse(
        request=request,
        name="admin/access_requests.html",
        context=context,
    )


@router.get("/admin/feed-access", response_class=HTMLResponse)
async def admin_feed_access(request: Request):
    context = _admin_context(request, "feed_access")
    context.update({
        "active_feed_tokens": feed_token_repo.list_active(),
    })

    return templates.TemplateResponse(
        request=request,
        name="admin/feed_access.html",
        context=context,
    )


@router.get("/admin/notifications", response_class=HTMLResponse)
async def admin_notifications(request: Request, admin_user = Depends(require_admin)):
    return templates.TemplateResponse(
        request=request,
        name="admin/notifications.html",
        context=_admin_context(request, "notifications"),
    )


@router.post("/admin/notifications/update")
async def update_notification_settings(
    notifications_enabled: bool = Form(False),
    notification_urls: str = Form(""),
    notify_access_requests: bool = Form(False),
    notify_new_podcasts: bool = Form(False),
    notify_episode_downloads: bool = Form(False),
    notify_breaking_errors: bool = Form(False),
    admin_user = Depends(require_admin),
):
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET notifications_enabled = ?,
                notification_urls = ?,
                notify_access_requests = ?,
                notify_new_podcasts = ?,
                notify_episode_downloads = ?,
                notify_breaking_errors = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (
                1 if notifications_enabled else 0,
                notification_urls.strip(),
                1 if notify_access_requests else 0,
                1 if notify_new_podcasts else 0,
                1 if notify_episode_downloads else 0,
                1 if notify_breaking_errors else 0,
            ),
        )
        conn.commit()

    return RedirectResponse(url="/admin/notifications?success=Notification+settings+updated", status_code=303)


@router.post("/admin/notifications/test")
async def test_notification_settings(admin_user = Depends(require_admin)):
    sent = send_test_notification()
    if sent:
        return RedirectResponse(url="/admin/notifications?success=Test+notification+sent", status_code=303)
    return RedirectResponse(
        url="/admin/notifications?error=Test+notification+could+not+be+sent.+Check+that+notifications+are+enabled+and+the+Apprise+URL+is+valid.",
        status_code=303,
    )

@router.post("/admin/users/{user_id}/password")
async def admin_change_user_password(
    request: Request, 
    user_id: int, 
    password: str = Form(...),
    admin_user: dict = Depends(require_admin)
):
    """Admin route to force change a user's password."""
    # Prevent changing own password via this route (use /change-password instead)
    if user_id == admin_user.id:
        return RedirectResponse(
            url="/admin/users?error=Use+My+Profile+to+change+your+own+password",
            status_code=status.HTTP_303_SEE_OTHER
        )

    # Hash new password
    new_hash = hash_password(password)
    
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        # Check if user exists
        user = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return RedirectResponse(
                url="/admin/users?error=User+not+found",
                status_code=status.HTTP_303_SEE_OTHER
            )
            
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", 
            (new_hash, user_id)
        )
        conn.commit()
    
    return RedirectResponse(
        url=f"/admin/users?success=Password+updated+for+{user['username']}",
        status_code=status.HTTP_303_SEE_OTHER
    )

@router.delete("/admin/users/{user_id}")
async def delete_user(user_id: int, request: Request, user: dict = Depends(require_admin)):
    # Check admin
    if user.id == user_id:
        return RedirectResponse(
            url="/admin/users?error=Cannot+delete+your+own+account",
            status_code=status.HTTP_303_SEE_OTHER
        )

    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    
    return RedirectResponse(
        url="/admin/users?success=User+deleted+successfully",
        status_code=status.HTTP_303_SEE_OTHER
    )


@router.post("/admin/users/{user_id}/delete")
async def delete_user_post(user_id: int, request: Request, user: dict = Depends(require_admin)):
    return await delete_user(user_id, request, user)

# --- Admin: Approve Access Request ---
@router.post("/admin/access-requests/{request_id}/approve")
async def approve_access_request(request: Request, request_id: int, admin_user = Depends(require_admin)):
    from app.infra.database import get_db_connection
    
    with get_db_connection() as conn:
        # Get the access request
        access_req = conn.execute(
            "SELECT * FROM access_requests WHERE id = ?", (request_id,)
        ).fetchone()
        
        if not access_req:
            return RedirectResponse(url="/admin/access-requests?error=Request+not+found", status_code=303)

        if not access_req["password_hash"]:
            return RedirectResponse(
                url="/admin/access-requests?error=Request+was+submitted+before+password+capture.+Ask+the+user+to+submit+a+new+request.",
                status_code=303,
            )
        
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
            return RedirectResponse(url="/admin/access-requests?error=Username+already+exists", status_code=303)
        
        # Create the new user
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            (access_req['username'], access_req["password_hash"])
        )
        
        # Update access request status
        conn.execute(
            "UPDATE access_requests SET status = 'approved', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        
        logger.info(f"AUTH - Access request approved: {access_req['username']}")
        
    # Redirect back to access page with success message
    return RedirectResponse(url=f"/admin/access-requests?approved={access_req['username']}", status_code=303)

# --- Admin: Deny Access Request ---
@router.post("/admin/access-requests/{request_id}/deny")
async def deny_access_request(request: Request, request_id: int, admin_user = Depends(require_admin)):
    from app.infra.database import get_db_connection
    
    with get_db_connection() as conn:
        # Get the access request to log username
        access_req = conn.execute(
            "SELECT username FROM access_requests WHERE id = ?", (request_id,)
        ).fetchone()
        
        if not access_req:
            return RedirectResponse(url="/admin/access-requests?error=Request+not+found", status_code=303)
        
        # Update access request status to denied
        conn.execute(
            "UPDATE access_requests SET status = 'denied', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        
        logger.info(f"AUTH - Access request denied: {access_req['username']}")
        
    return RedirectResponse(url="/admin/access-requests?denied=1", status_code=303)

# --- Admin: Update User Username ---
@router.post("/admin/users/{user_id}/username")
async def update_user_username(
    request: Request,
    user_id: int,
    username: str = Form(...),
    admin_user = Depends(require_admin),
):
    with get_db_connection() as conn:
        # Check if username already exists
        existing = conn.execute("SELECT id FROM users WHERE username = ? AND id != ?", (username, user_id)).fetchone()
        if existing:
            return RedirectResponse(url="/admin/users?error=Username+already+exists", status_code=303)
            
        conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
        conn.commit()
    return RedirectResponse(url="/admin/users", status_code=303)

# --- Admin: Update Request Username ---
@router.post("/admin/access-requests/{request_id}/username")
async def update_request_username(
    request: Request,
    request_id: int,
    username: str = Form(...),
    admin_user = Depends(require_admin),
):
    with get_db_connection() as conn:
        conn.execute("UPDATE access_requests SET username = ? WHERE id = ?", (username, request_id))
        conn.commit()
    return RedirectResponse(url="/admin/access-requests", status_code=303)

# Helper to render index with consistent data
def _render_index(request: Request, error: str = None):
    from app.infra.repository import SubscriptionRepository
    sub_repo = SubscriptionRepository()
    user = get_current_user(request)
    user_id = _real_user_id(user)
    requested_view = request.query_params.get("view") if hasattr(request, "query_params") else None
    library_view = requested_view if requested_view in ("mine", "library") else ("mine" if user_id else "library")
    subs = sub_repo.get_all(user_id=user_id, only_user=library_view == "mine")
    all_subs = sub_repo.get_all()
    
    # Calculate stats
    total_podcasts = len(all_subs)
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
                """SELECT id, title, description, ai_summary, pub_date FROM episodes 
                   WHERE subscription_id = ? AND status = 'completed'
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()
            
            # Get the latest episode date (any status) for filtering/sorting
            latest_any_ep = conn.execute(
                """SELECT pub_date FROM episodes 
                   WHERE subscription_id = ?
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()
            
            # Count processing/pending episodes for this subscription
            processing_row = conn.execute(
                """SELECT COUNT(*) as count FROM episodes 
                   WHERE subscription_id = ? AND status IN ('processing', 'pending')""",
                (sub.id,)
            ).fetchone()
            processing_count = processing_row['count'] if processing_row else 0
            user_library_row = conn.execute(
                "SELECT COUNT(*) as count FROM user_subscriptions WHERE subscription_id = ?",
                (sub.id,),
            ).fetchone()
            user_library_count = user_library_row["count"] if user_library_row else 0
        
        latest_summary = None
        latest_description = None
        latest_episode_date = None
        if latest_ep:
            latest_summary = latest_ep['ai_summary']
            latest_description = latest_ep['description']
        if latest_any_ep and latest_any_ep['pub_date']:
            # Convert to ISO format string for safe JS parsing
            d = latest_any_ep['pub_date']
            if hasattr(d, 'isoformat'):
                latest_episode_date = d.isoformat()
            else:
                latest_episode_date = str(d)
        
        subs_with_links.append({
            "sub": sub,
            "links": generate_rss_links(request, sub, global_settings, user),
            "episodes": [dict(ep) for ep in episodes],
            "episode_count": len(episodes),
            "processing_count": processing_count,
            "total_listens": ep_repo.get_subscription_listen_count(sub.id),
            "latest_ai_summary": latest_summary,
            "latest_description": latest_description,
            "latest_episode_date": latest_episode_date,
            "in_user_library": sub_repo.is_in_user_library(user_id, sub.id) if user_id else True,
            "can_manage": _can_manage_subscription(user, sub),
            "owner_username": sub_repo.get_owner_username(sub.id),
            "user_library_count": user_library_count,
        })

    # Get queue data for dashboard display
    queue = ep_repo.get_queue()

    # Determine if AI is configured (DB Overrides/Augments Env)
    from app.core.config import settings

    # Check if the DB has a non-empty list of Gemini keys
    db_gemini_keys = global_settings.get('gemini_api_keys')
    has_db_gemini = db_gemini_keys and db_gemini_keys != "[]" and db_gemini_keys != "null"

    config_warning = not any([
        settings.GEMINI_API_KEY,
        settings.OPENAI_API_KEY,
        settings.ANTHROPIC_API_KEY,
        settings.OPENROUTER_API_KEY,
        has_db_gemini,  # Correctly check the plural database list
        global_settings.get('openai_api_key'),
        global_settings.get('anthropic_api_key'),
        global_settings.get('openrouter_api_key')
    ])

    # Generate Unified Links if subscriptions exist
    unified_links = None
    if all_subs:
        # Determine Base URL using consolidated logic
        base_url = get_app_base_url(global_settings, request)
        
        rss_url = f"{base_url}/feed/unified.xml"
        
        # Inject Auth if Enabled
        auth_enabled_val = global_settings.get('enable_feed_auth')
        is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False
        
        if is_auth_enabled:
            token = get_or_create_feed_token(request, user)
            if token:
                separator = "&" if "?" in rss_url else "?"
                rss_url = f"{rss_url}{separator}token={token}"


        unified_links = build_subscription_links(rss_url)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "request": request,
            "user": user,
            "subscriptions": subs_with_links,
            "library_view": library_view,
            "my_podcast_count": len(sub_repo.get_all(user_id=user_id, only_user=True)) if user_id else len(all_subs),
            "library_podcast_count": len(all_subs),
            "stats": stats,
            "error": error,
            "success": request.query_params.get("success") if hasattr(request, "query_params") else None,
            "config_warning": config_warning,
            "queue": queue,
            "unified_links": unified_links,
            "settings": global_settings
        }
    )

def _build_public_subscribe_context(request: Request, global_settings: dict):
    subs = sub_repo.get_all()
    public_links = []
    for sub in subs:
        with get_db_connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as count
                   FROM episodes
                   WHERE subscription_id = ? AND status = 'completed'""",
                (sub.id,)
            ).fetchone()
            latest = conn.execute(
                """SELECT title, pub_date
                   FROM episodes
                   WHERE subscription_id = ? AND status = 'completed'
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()

        public_links.append({
            "sub": sub,
            "links": generate_rss_links(request, sub, global_settings, include_auth_token=False),
            "episode_count": row["count"] if row else 0,
            "latest_episode": dict(latest) if latest else None,
        })

    unified_links = None
    if subs:
        base_url = get_app_base_url(global_settings, request)
        rss_url = f"{base_url}/feed/unified.xml"
        unified_links = build_subscription_links(rss_url)

    feed_auth_enabled = str(global_settings.get("enable_feed_auth")).lower() in ("1", "true", "yes", "on")

    return {
        "request": request,
        "csp_nonce": get_csp_nonce(request),
        "user": get_current_user(request) if request.session.get(SESSION_USER_KEY) else None,
        "subscriptions": public_links,
        "unified_links": unified_links,
        "feed_auth_enabled": feed_auth_enabled,
        "settings": global_settings,
    }

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render_index(request)

@router.get("/subscribe", response_class=HTMLResponse)
async def public_subscribe(request: Request):
    global_settings = get_global_settings()
    if not global_settings.get("public_subscribe_page_enabled"):
        raise HTTPException(status_code=404, detail="Subscribe page is disabled")

    return templates.TemplateResponse(
        request=request,
        name="public_subscribe.html",
        context=_build_public_subscribe_context(request, global_settings)
    )

from app.core.processor import Processor

# --- Admin: Global Subscription Settings ---
@router.get("/admin/global-subscription-settings", response_class=HTMLResponse)
async def admin_global_subscription_settings(request: Request):
    user = get_current_user(request)
    
    with get_db_connection() as conn:
        settings_row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        
    return templates.TemplateResponse(
        request=request,
        name="admin/global_subscription_settings.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "settings": settings_row,
            "active_tab": "global_subs"
        }
    )

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
    default_custom_instructions: str = Form(None),
    default_download_order: str = Form("newest"),
    whitelist_mode: bool = Form(False),
    admin_user = Depends(require_admin)
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
                default_custom_instructions = ?,
                default_download_order = ?,
                whitelist_mode = ?
            WHERE id = 1
        """, (
            default_remove_ads, default_remove_promos, default_remove_intros, default_remove_outros,
            default_ai_rewrite_description, default_ai_audio_summary, default_append_title_intro,
            default_retention_limit, default_retention_days, default_manual_retention_days,
            default_custom_instructions, default_download_order, 1 if whitelist_mode else 0
        ))
        conn.commit()
        
    return RedirectResponse(url="/admin/global-subscription-settings?success=Settings updated", status_code=303)


@router.post("/add", response_class=HTMLResponse)
async def add_subscription(
    request: Request,
    background_tasks: BackgroundTasks,
    feed_url: str = Form(...),
    initial_count: int = Form(1),
    user = Depends(require_auth),
):
    try:
        print(f"Adding feed: feed_url: {feed_url}")
        validate_http_url(feed_url, allow_private=runtime_settings.ALLOW_PRIVATE_FEEDS)
        print(f"Validated url: {feed_url}")
        # Check if exists (quick DB check)
        existing = sub_repo.get_by_url(feed_url)
        if existing:
            print(f"Feed already exists in DB: {existing.id}")
            added = sub_repo.add_to_user_library(_real_user_id(user), existing.id)
            message = "Podcast added to My Podcasts from the library" if added else "Podcast is already in My Podcasts"
            return RedirectResponse(url=f"/?view=mine&success={quote(message)}", status_code=303)
        
        # Create subscription with placeholder data
        # Fetch global defaults first
        print(f"Feed doesn't already exists in DB.")
        with get_db_connection() as conn:
            app_settings = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()

        print(f"app_settings: {app_settings}")

        # Use user-provided initial_count (from UI dropdown) as retention limit
        # The UI defaults this dropdown to the global default setting already.
        retention_limit = initial_count

        print(f"About to create Subscription")

        sub_create = SubscriptionCreate(feed_url=feed_url)
        new_sub = sub_repo.create(
            sub_create,
            "Loading...",
            f"loading-{int(__import__('time').time())}",
            None,
            "Fetching feed information...",
            retention_limit=retention_limit,
            owner_user_id=_real_user_id(user),
        )

        print(f"Created new sub: {new_sub}")

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
            feed_url=feed_url,
            retention_days=app_settings['default_retention_days'] or 30,
            manual_retention_days=app_settings['default_manual_retention_days'] or 14,
            retention_limit=retention_limit,
            download_order=app_settings.get('default_download_order', 'newest')
        )

        
        # All heavy lifting happens in background
        print(f"About to setup subscription async: new_sub.id: {new_sub.id}")
        async def setup_subscription(sub_id: int, url: str, limit: int):
            from app.core.processor import Processor
            from app.core.feed import FeedManager
            from app.infra.database import get_db_connection
            
            try:
                print(f"About to parse the feed: url:{url}")
                # Parse feed (network call)
                title, slug, image_url, description = FeedManager.parse_feed(url)

                print(f"Parsed feed: title:{title}, slug:{slug}, image_url:{image_url}, description:{description}")

                # Update subscription with real data
                # Keep the settings we just set! Only update metadata.
                with get_db_connection() as conn:
                    print(f"About to add feed details to DB")
                    conn.execute("""
                        UPDATE subscriptions 
                        SET title = ?, slug = ?, image_url = ?, description = ?
                        WHERE id = ?
                    """, (title, slug, image_url, description, sub_id))
                    conn.commit()

                print(f"About to send notification")

                await send_notification_async(
                    EVENT_NEW_PODCAST,
                    "Podcast added",
                    f"{title} was added to the global podcast library.",
                    severity="success",
                )

                print(f"About to check feed and process queue")

                # Now check feeds and process queue
                proc = Processor()
                await proc.check_feeds(subscription_id=sub_id, limit=limit)
                await proc.process_queue()
                
            except Exception as e:
                logger.error(f"Error setting up subscription {sub_id}: {e}")
                await send_notification_async(
                    EVENT_BREAKING_ERROR,
                    "Podcast setup failed",
                    f"A new podcast could not be set up: {e}",
                    severity="error",
                )
        
        background_tasks.add_task(setup_subscription, new_sub.id, feed_url, retention_limit)
        
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        return _render_index(request, error=str(e))


@router.post("/subscriptions/{id}/library")
async def update_user_library_membership(
    request: Request,
    id: int,
    action: str = Form("toggle"),
    redirect_to: str = Form(None),
    user = Depends(require_auth),
):
    sub = sub_repo.get_by_id(id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")

    user_id = _real_user_id(user)
    if not user_id:
        return RedirectResponse(url=_safe_local_redirect(redirect_to, "/"), status_code=303)

    is_member = sub_repo.is_in_user_library(user_id, id)
    if action == "add" or (action == "toggle" and not is_member):
        changed = sub_repo.add_to_user_library(user_id, id)
        message = "Podcast added to My Podcasts" if changed else "Podcast is already in My Podcasts"
    elif action == "remove" or (action == "toggle" and is_member):
        changed = sub_repo.remove_from_user_library(user_id, id)
        message = "Podcast removed from My Podcasts" if changed else "Podcast was not in My Podcasts"
    else:
        message = "No library change made"

    target = _safe_local_redirect(redirect_to, "/")
    separator = "&" if "?" in target else "?"
    return RedirectResponse(url=f"{target}{separator}success={quote(message)}", status_code=303)


@router.post("/subscriptions/{id}/owner")
async def update_subscription_owner(
    id: int,
    owner_user_id: str = Form(""),
    admin_user = Depends(require_admin),
):
    owner_user_id = owner_user_id.strip()
    new_owner_id = None
    if owner_user_id:
        try:
            new_owner_id = int(owner_user_id)
        except ValueError:
            return RedirectResponse(url=f"/subscriptions/{id}?error=Invalid+owner", status_code=303)
        if new_owner_id <= 0:
            return RedirectResponse(url=f"/subscriptions/{id}?error=Invalid+owner", status_code=303)

    updated = sub_repo.set_owner(id, new_owner_id)
    if not updated:
        return RedirectResponse(url=f"/subscriptions/{id}?error=Owner+or+podcast+not+found", status_code=303)

    message = "Podcast owner updated" if new_owner_id else "Podcast owner cleared"
    return RedirectResponse(url=f"/subscriptions/{id}?success={quote(message)}", status_code=303)


@router.get("/subscriptions/{id}", response_class=HTMLResponse)
async def view_subscription(request: Request, id: int):
    sub = sub_repo.get_by_id(id)
    if not sub:
        return RedirectResponse(url="/")
    
    # Initial page size for lazy loading
    INITIAL_PAGE_SIZE = 20
    
    # Get first batch of episodes using pagination
    episodes = ep_repo.get_by_subscription_paginated(id, limit=INITIAL_PAGE_SIZE, offset=0)
    total_episodes = ep_repo.count_by_subscription(id)
    has_more = total_episodes > INITIAL_PAGE_SIZE
    
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
    user_id = _real_user_id(user)

    links = generate_rss_links(request, sub, global_settings, user)
    
    # Get total listen count for this subscription
    total_listens = ep_repo.get_subscription_listen_count(sub.id)
    with get_db_connection() as conn:
        user_library_row = conn.execute(
            "SELECT COUNT(*) as count FROM user_subscriptions WHERE subscription_id = ?",
            (sub.id,),
        ).fetchone()
        user_library_count = user_library_row["count"] if user_library_row else 0

    return templates.TemplateResponse(
        request=request,
        name="episodes.html",
        context={
            "request": request, # Required for some template helpers
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "subscription": sub,
            "episodes": episodes,
            "links": links,
            "is_in_user_library": sub_repo.is_in_user_library(user_id, sub.id) if user_id else True,
            "can_manage_subscription": _can_manage_subscription(user, sub),
            "owner_username": sub_repo.get_owner_username(sub.id),
            "user_library_count": user_library_count,
            "active_users": _active_users() if user and user.is_admin else [],
            "basename": lambda p: p.split('/')[-1] if p else '',
            "format_duration": format_duration,
            "total_listens": total_listens,
            "total_episodes": total_episodes,
            "has_more": has_more,
            "page_size": INITIAL_PAGE_SIZE,
            "settings": global_settings
        }
    )

@router.get("/api/subscriptions/{id}/episodes")
async def get_subscription_episodes_api(id: int, limit: int = 20, offset: int = 0, search: str = None):
    """Return episodes for a subscription as JSON for lazy loading. Supports search by title."""
    sub = sub_repo.get_by_id(id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Pass search to repository methods
    episodes = ep_repo.get_by_subscription_paginated(id, limit=limit, offset=offset, search=search)
    total = ep_repo.count_by_subscription(id, search=search)
    
    # Convert sqlite rows to dicts
    episodes_data = []
    for ep in episodes:
        ep_dict = dict(ep)
        # Ensure pub_date is a string for JSON serialization
        if ep_dict.get('pub_date') and hasattr(ep_dict['pub_date'], 'isoformat'):
            ep_dict['pub_date'] = ep_dict['pub_date'].isoformat()
        episodes_data.append(ep_dict)
    
    return {
        "episodes": episodes_data,
        "total": total,
        "offset": offset,
        "limit": limit,
        "search": search,
        "has_more": offset + len(episodes) < total,
        "subscription_slug": sub.slug
    }

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
    feed_url: str = Form(None),
    retention_days: int = Form(30),
    manual_retention_days: int = Form(14),
    retention_limit: int = Form(1),
    download_order: str = Form("newest"),
    user = Depends(require_auth),
):
    sub = sub_repo.get_by_id(id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if not _can_manage_subscription(user, sub):
        raise HTTPException(status_code=403, detail="Only admins and the podcast owner can change podcast settings")

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
        feed_url,
        retention_days,
        manual_retention_days,
        retention_limit,
        download_order
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

@router.get("/episodes/{id}/transcript")
async def view_transcript(id: int, request: Request):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    import json
    
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT e.id, e.title, e.pub_date, e.duration, e.guid, s.slug as subscription_slug, e.transcript_path 
               FROM episodes e 
               JOIN subscriptions s ON e.subscription_id = s.id 
               WHERE e.id = ?""", 
            (id,)
        ).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Episode not found")
            
        transcript_path = row['transcript_path']
        
        # Check standard paths if not recorded in DB or file missing
        if not transcript_path or not os.path.exists(transcript_path):
             episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
             potential_path = os.path.join(
                settings.get_episode_dir(row['subscription_slug'], episode_slug),
                "transcript.json"
            )
             if os.path.exists(potential_path):
                 transcript_path = potential_path
        
        if not transcript_path or not os.path.exists(transcript_path):
             raise HTTPException(status_code=404, detail="Transcript file not found")
             
        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading transcript: {str(e)}")
    
    def format_duration(seconds: int) -> str:
        if not seconds:
            return "-"
        seconds = int(seconds)  # Convert to int to handle floats
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    return templates.TemplateResponse(
        request=request,
        name="transcript.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "episode": row,
            "transcript_data": data,
            "format_duration": format_duration
        }
    )

@router.get("/artifacts/transcript/{id}")
async def get_transcript_json(id: int):
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
    
    # Extract credentials from request (header or legacy query param), or preferred bearer token.
    username = None
    password = None
    audio_token = None
    request_token = request.query_params.get('token')
    if request_token and feed_token_repo.validate(request_token):
        audio_token = request_token
    
    auth_header = request.headers.get('Authorization')
    if not audio_token and auth_header and auth_header.startswith('Basic '):
        try:
            encoded = auth_header.split(' ')[1]
            decoded = base64.b64decode(encoded).decode('utf-8')
            username, password = decoded.split(':', 1)
        except:
            pass
    elif not audio_token:
        auth_param = request.query_params.get('auth')
        if auth_param:
            try:
                decoded = base64.b64decode(auth_param).decode('utf-8')
                username, password = decoded.split(':', 1)
            except:
                pass
    
    # Set no-cache headers
    cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    # If auth is enabled and we have token/credentials, inject access into audio URLs.
    if is_auth_enabled and (audio_token or (username and password)):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            
            if not audio_token:
                audio_token = base64.b64encode(f"{username}:{password}".encode()).decode()
                param_name = "auth"
            else:
                param_name = "token"
            
            xml_content = _append_feed_access_to_enclosures(xml_content, param_name, audio_token)
            
            return Response(content=xml_content, media_type="application/xml", headers=cache_headers)
        except Exception as e:
            logger.error(f"Error injecting auth into feed {slug}: {e}")
    
    return FileResponse(file_path, media_type="application/xml", headers=cache_headers)

@router.get("/feed/unified")
@router.get("/feed/unified.xml")
async def get_unified_feed(request: Request):
    """Serve the unified RSS feed with optional authentication."""
    # Check Auth if enabled
    settings = get_global_settings()
    auth_enabled_val = settings.get('enable_feed_auth')
    is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False

    if is_auth_enabled:
        # Check preferred bearer token, Basic Auth header, or legacy auth query param.
        import base64
        auth_header = request.headers.get('Authorization')
        auth_token = request.query_params.get('auth')
        feed_token = request.query_params.get('token')
        authorized = False
        username = None
        password = None
        audio_token = None

        if feed_token and feed_token_repo.validate(feed_token):
            authorized = True
            audio_token = feed_token
        
        encoded_creds = None
        if not authorized and auth_header and auth_header.startswith('Basic '):
            encoded_creds = auth_header.split(' ')[1]
        elif not authorized and auth_token:
            encoded_creds = auth_token
            
        if encoded_creds:
            try:
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
                    if username == expected_user and verify_feed_password(password, expected_pass):
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
    
    # Set no-cache headers
    cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }

    # If auth is enabled, inject credentials into the XML on the fly
    if is_auth_enabled and authorized:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            
            # Inject feed access into enclosure URLs so podcast clients can download audio.
            if audio_token or (username and password):
                if audio_token:
                    token_param = "token"
                    token_value = audio_token
                else:
                    token_param = "auth"
                    token_value = base64.b64encode(f"{username}:{password}".encode()).decode()

                xml_content = _append_feed_access_to_enclosures(xml_content, token_param, token_value)

                
            return Response(content=xml_content, media_type="application/xml", headers=cache_headers)
        except Exception as e:
            logger.error(f"Error injecting credentials into unified feed: {e}")
            # Fallback to static file if injection fails
    
    return FileResponse(file_path, media_type="application/xml", headers=cache_headers)
