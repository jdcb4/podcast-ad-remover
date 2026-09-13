from app.core.model_defaults import MODEL_DEFAULTS
from app.core.artifacts import artifact_path
from app.core.permissions import can_manage_subscription as _can_manage_subscription
from app.web.permissions import require_episode_management
from fastapi import APIRouter, Request, Form, Depends, BackgroundTasks, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from app.infra.repository import ApiTokenRepository, SubscriptionRepository, EpisodeRepository, FeedTokenRepository
from app.core.feed import FeedManager
from app.core.models import SubscriptionCreate
from app.core.queue_status import get_queue_payload
from app.core.system_status import get_operation_status
from app.core.time_utils import now_utc, utc_iso, with_utc_timestamps
from app.core.url_utils import validate_http_url
from app.core.sources import resolve_source
from app.core.notifications import (
    EVENT_ACCESS_REQUEST,
    EVENT_BREAKING_ERROR,
    EVENT_NEW_PODCAST,
    send_notification,
    send_notification_async,
    send_test_notification,
)
from app.web.auth import get_current_user, require_auth, require_admin, log_login_attempt, SESSION_USER_KEY
from app.web.auth_utils import hash_password, verify_feed_password, verify_password, generate_secure_password, get_client_ip, get_request_base_origin
from app.web.rate_limiter import login_rate_limiter, check_rate_limit
from app.web.subscription_links import build_subscribe_instruction_context, build_subscription_links
from app.web.static_assets import configure_static_asset_versioning
from app.web.template_filters import compact_datetime, format_duration, local_time, utc_isoformat
from app.web.template_filters import clean_description as safe_clean_description
from app.web.template_filters import simple_markdown as safe_simple_markdown
from app.infra.database import get_db_connection
from app.core.config import is_default_session_secret, settings as runtime_settings
from app.core.artwork import ArtworkWatermarker, effective_artwork_url
from app.core.unified_feed import (
    DEFAULT_UNIFIED_FEED_DESCRIPTION,
    DEFAULT_UNIFIED_FEED_TITLE,
    normalize_unified_feed_settings,
    resolve_unified_feed_artwork_preview,
    resolve_unified_feed_settings,
)
import asyncio
import os
import logging
import re
from urllib.parse import quote
from pathlib import Path

logger = logging.getLogger(__name__)

from app.web.timeline_rules import router as timeline_rules_router

router = APIRouter()
router.include_router(timeline_rules_router)
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Helper to get CSP nonce from request
def get_csp_nonce(request: Request) -> str:
    """Extract CSP nonce from request state (set by SecurityHeadersMiddleware)"""
    return getattr(request.state, 'csp_nonce', '')


templates.env.filters['simple_markdown'] = safe_simple_markdown
templates.env.filters['clean_description'] = safe_clean_description
templates.env.filters['compact_datetime'] = compact_datetime
templates.env.filters['local_time'] = local_time
templates.env.filters['utc_isoformat'] = utc_isoformat
configure_static_asset_versioning(templates)

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
from app.core.utils import get_global_settings


def _real_user_id(user) -> int | None:
    user_id = getattr(user, "id", None)
    return user_id if user_id and user_id > 0 else None




def _reconcile_artwork_and_feeds(subscription_ids: list[int] | None = None) -> None:
    from app.core.rss_gen import RSSGenerator

    service = ArtworkWatermarker()
    subscriptions = sub_repo.get_all()
    selected = [sub for sub in subscriptions if subscription_ids is None or sub.id in subscription_ids]
    for sub in selected:
        try:
            service.reconcile(sub.id)
        except Exception as exc:
            logger.warning("Artwork reconciliation failed for subscription %s: %s", sub.id, exc)

    generator = RSSGenerator()
    for sub in selected:
        generator.generate_feed(sub.id)
    generator.generate_unified_feed()

from app.core.utils import get_app_base_url


def _build_rss_links(request: Request, feed_path: str, global_settings: dict, user_obj=None, include_auth_token: bool = True):
    """Consolidated logic for generating RSS links with optional auth injection."""
    base_url = get_app_base_url(global_settings, request)
    
    rss_url = f"{base_url}{feed_path}"
    
    # Inject Auth if Enabled
    auth_enabled_val = global_settings.get('enable_feed_auth')
    is_auth_enabled = str(auth_enabled_val).lower() in ('1', 'true', 'yes', 'on') if auth_enabled_val is not None else False
    
    if is_auth_enabled and include_auth_token:
        token = get_or_create_feed_token(request, user_obj)
        if token:
            separator = "&" if "?" in rss_url else "?"
            rss_url = f"{rss_url}{separator}token={token}"

    return build_subscription_links(rss_url)


def generate_rss_links(request: Request, sub, global_settings: dict, user_obj=None, include_auth_token: bool = True):
    return _build_rss_links(
        request, f"/feeds/{sub.slug}.xml", global_settings, user_obj, include_auth_token,
    )

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
        conn.execute("UPDATE users SET last_login = ?, last_login_is_utc = 1 WHERE id = ?", (now_utc(), user_row['id']))
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
            "model_defaults": MODEL_DEFAULTS,
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
    section: str = Form("ai_text"),
    whisper_model: str = Form(None),
    ai_model_cascade: str = Form(None),
    piper_model: str = Form(None),
    tts_provider: str = Form(None),
    gemini_tts_voice: str = Form(None),
    gemini_tts_model_cascade: str = Form(None),
    active_ai_provider: str = Form(None),
    openai_api_key: str = Form(None),
    anthropic_api_key: str = Form(None),
    openrouter_api_key: str = Form(None),
    gemini_api_keys: str = Form(None),
    openai_model: str = Form(None),
    anthropic_model: str = Form(None),
    openrouter_model: str = Form(None),
    custom_llm_base_url: str = Form(None),
    custom_llm_api_key: str = Form(None),
    custom_llm_model: str = Form(None),
    clear_openai_api_key: bool = Form(False),
    clear_anthropic_api_key: bool = Form(False),
    clear_openrouter_api_key: bool = Form(False),
    clear_custom_llm_api_key: bool = Form(False),
    clear_gemini_api_keys: bool = Form(False),
    redirect_to: str = Form("/admin/ai/text-analysis"),
    admin_user = Depends(require_admin)
):
    from app.infra.database import get_db_connection
    from app.core.ai_services import AdDetector, normalize_openai_base_url
    import json

    def normalized_models(value: str | None, current: str, default: list[str]) -> str:
        if value is None:
            return current or json.dumps(default)
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                parsed = [str(parsed)]
        except json.JSONDecodeError:
            parsed = [value]
        parsed = [str(item).strip() for item in parsed if str(item).strip()]
        if len(parsed) > 10:
            raise HTTPException(status_code=400, detail="A model cascade can contain at most 10 models.")
        return json.dumps(parsed)

    def updated_secret(submitted: str | None, clear: bool, current: str | None) -> str | None:
        if clear:
            return None
        if submitted and submitted.strip():
            return submitted.strip()
        return current

    with get_db_connection() as conn:
        current_row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        current = dict(current_row) if current_row else {}

        if section == "ai_transcription":
            selected_whisper = whisper_model if whisper_model in {"tiny", "base", "small", "medium", "large"} else "base"
            conn.execute(
                "UPDATE app_settings SET whisper_model = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (selected_whisper,),
            )
        elif section == "ai_voice":
            selected_tts_provider = tts_provider if tts_provider in {"piper", "gemini"} else "piper"
            selected_voice = (
                gemini_tts_voice
                if gemini_tts_voice in AdDetector.GEMINI_TTS_VOICES
                else "Orus"
            )
            selected_tts_models = normalized_models(
                gemini_tts_model_cascade,
                current.get("gemini_tts_model_cascade"),
                AdDetector.DEFAULT_GEMINI_TTS_MODELS,
            )
            conn.execute(
                """
                UPDATE app_settings
                SET piper_model = ?,
                    tts_provider = ?,
                    gemini_tts_voice = ?,
                    gemini_tts_model_cascade = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (
                    piper_model or current.get("piper_model") or "en_GB-cori-high.onnx",
                    selected_tts_provider,
                    selected_voice,
                    selected_tts_models,
                ),
            )
        elif section == "ai_text":
            selected_provider = active_ai_provider or current.get("active_ai_provider") or "gemini"
            if selected_provider not in {"gemini", "openai", "anthropic", "openrouter", "custom"}:
                raise HTTPException(status_code=400, detail="Unsupported text-analysis provider.")

            selected_gemini_models = normalized_models(
                ai_model_cascade,
                current.get("ai_model_cascade"),
                AdDetector.DEFAULT_GEMINI_MODELS,
            )
            selected_openai_models = normalized_models(
                openai_model, current.get("openai_model"), MODEL_DEFAULTS["openai"]
            )
            selected_anthropic_models = normalized_models(
                anthropic_model,
                current.get("anthropic_model"),
                MODEL_DEFAULTS["anthropic"],
            )
            selected_openrouter_models = normalized_models(
                openrouter_model,
                current.get("openrouter_model"),
                AdDetector.DEFAULT_OPENROUTER_MODELS,
            )
            selected_custom_models = normalized_models(
                custom_llm_model, current.get("custom_llm_model"), []
            )

            selected_gemini_keys = current.get("gemini_api_keys") or "[]"
            if clear_gemini_api_keys:
                selected_gemini_keys = "[]"
            elif gemini_api_keys:
                try:
                    parsed_keys = json.loads(gemini_api_keys)
                except json.JSONDecodeError as exc:
                    raise HTTPException(status_code=400, detail="Gemini API keys must be a JSON array.") from exc
                parsed_keys = [
                    str(key).strip() for key in parsed_keys
                    if isinstance(key, str) and key.strip()
                ]
                if parsed_keys:
                    selected_gemini_keys = json.dumps(parsed_keys)

            try:
                selected_base_url = (
                    normalize_openai_base_url(custom_llm_base_url)
                    if custom_llm_base_url and custom_llm_base_url.strip()
                    else current.get("custom_llm_base_url")
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if selected_provider == "custom":
                if not selected_base_url:
                    raise HTTPException(status_code=400, detail="Custom API base URL is required.")
                if not json.loads(selected_custom_models):
                    raise HTTPException(status_code=400, detail="At least one custom model slug is required.")

            conn.execute(
                """
                UPDATE app_settings
                SET active_ai_provider = ?,
                    ai_model_cascade = ?,
                    openai_api_key = ?,
                    anthropic_api_key = ?,
                    openrouter_api_key = ?,
                    gemini_api_keys = ?,
                    openai_model = ?,
                    anthropic_model = ?,
                    openrouter_model = ?,
                    custom_llm_base_url = ?,
                    custom_llm_api_key = ?,
                    custom_llm_model = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (
                    selected_provider,
                    selected_gemini_models,
                    updated_secret(openai_api_key, clear_openai_api_key, current.get("openai_api_key")),
                    updated_secret(anthropic_api_key, clear_anthropic_api_key, current.get("anthropic_api_key")),
                    updated_secret(openrouter_api_key, clear_openrouter_api_key, current.get("openrouter_api_key")),
                    selected_gemini_keys,
                    selected_openai_models,
                    selected_anthropic_models,
                    selected_openrouter_models,
                    selected_base_url,
                    updated_secret(custom_llm_api_key, clear_custom_llm_api_key, current.get("custom_llm_api_key")),
                    selected_custom_models,
                ),
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported AI settings section.")
        conn.commit()
    return RedirectResponse(url=_safe_local_redirect(redirect_to, "/admin/ai/text-analysis"), status_code=303)

@router.post("/admin/ai/test")
async def test_ai_connection(
    provider: str = Form(...),
    api_key: str = Form(None),
    model: str = Form(None),
    base_url: str = Form(None),
    admin_user = Depends(require_admin)
):
    try:
        from app.core.ai_services import AdDetector
        detector = AdDetector()
        
        prov_instance = detector.create_provider(
            provider, api_key=api_key, model=model, base_url=base_url
        )
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
async def refresh_models(provider: str, admin_user = Depends(require_admin)):
    try:
        from app.core.ai_services import AdDetector
        if provider == "gemini_tts":
            return {"models": AdDetector.DEFAULT_GEMINI_TTS_MODELS}
        detector = AdDetector()
        # Create provider using saved settings (implies user must save key first usually, 
        # but we could allow passing key in query param if we wanted to be fancy. 
        # For now, rely on saved settings for Auth to keep it simple).
        prov_instance = detector.create_provider(provider) 
        models = prov_instance.list_models()
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}


@router.post("/admin/ai/refresh")
async def refresh_models_from_form(
    provider: str = Form(...),
    api_key: str = Form(None),
    base_url: str = Form(None),
    admin_user = Depends(require_admin),
):
    try:
        from app.core.ai_services import AdDetector
        if provider == "gemini_tts":
            return {"models": AdDetector.DEFAULT_GEMINI_TTS_MODELS}
        detector = AdDetector()
        instance = detector.create_provider(
            provider,
            api_key=api_key,
            base_url=base_url,
        )
        return {"models": instance.list_models()}
    except Exception as e:
        return {"error": str(e)}

# --- Admin: Prompts ---
@router.get("/admin/prompts", response_class=HTMLResponse)
async def admin_prompts(request: Request):
    from app.core.prompt_defaults import LEGACY_DEFAULTS
    from app.core import timeline
    current = get_global_settings()
    default_prompts = LEGACY_DEFAULTS

    user = get_current_user(request)

    return templates.TemplateResponse(
        request=request,
        name="admin/prompts.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": user,
            "settings": current,
            "timeline_definitions": timeline.definitions(current),
            "timeline_defaults": timeline.DEFINITIONS,
            "timeline_labels": timeline.LABEL_NAMES,
            "timeline_prompt_template": timeline.PROMPT,
            "timeline_summary_default": timeline.SUMMARY_DEFAULT,
            "preview_subscriptions": sub_repo.get_all(),
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
                summary_prompt_template = ?,
                ad_target_intro = COALESCE(?, ad_target_intro),
                ad_target_outro = COALESCE(?, ad_target_outro)
            WHERE id = 1
        """, (
            form.get('ad_prompt_base'),
            form.get('ad_target_sponsor'),
            form.get('ad_target_promo'),
            form.get('summary_prompt_template'),
            form.get('ad_target_intro'),
            form.get('ad_target_outro')
        ))
        conn.commit()
    
    return {"status": "success"}

@router.post("/admin/prompts/reset")
async def reset_prompts(request: Request, admin_user = Depends(require_admin)):
    from app.core.prompt_defaults import LEGACY_DEFAULTS
    defaults = LEGACY_DEFAULTS

    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings SET
                summary_prompt_template = ?,
                ad_prompt_base = ?,
                ad_target_sponsor = ?,
                ad_target_promo = ?,
                ad_target_intro = ?,
                ad_target_outro = ?
            WHERE id = 1
        """, (defaults['summary'], defaults['ad_base'], defaults['sponsor'], defaults['promo'], defaults['intro'], defaults['outro']))
        conn.commit()
    
    return {"status": "success"}

# --- Admin: Queue ---
@router.get("/admin/queue", response_class=HTMLResponse)
async def admin_queue(request: Request):
    user = get_current_user(request)
    queue = ep_repo.get_queue()
    recently_processed = ep_repo.get_recently_processed(days=3)
    operation_status = await asyncio.to_thread(get_operation_status)
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
    return await asyncio.to_thread(get_queue_payload)


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
    require_episode_management(user, episode_id)
    import json
    logger.info(f"Reprocess request for {episode_id} with skip_transcription={skip_transcription}")
    
    # API version of retry - force status to pending
    current_status = ep_repo.get_status(episode_id)
    if current_status == 'processing':
         raise HTTPException(409, "Episode is already processing")
    
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
    require_episode_management(user, episode_id)
    # API version of cancel/delete - soft delete
    from app.core.processor import Processor
    proc = Processor()
    await proc.delete_episode(episode_id)
    return {"status": "ok"}

@router.post("/episodes/{episode_id}/download")
async def manual_download_episode(episode_id: int, request: Request, user = Depends(require_auth)):
    require_episode_management(user, episode_id)
    # Update DB to pending
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET is_manual_download=1 WHERE id=?", (episode_id,))
        conn.commit()

    ep_repo.update_status(episode_id, "pending")

    from app.core.processor import Processor
    proc = Processor()
    await proc.process_queue()

    if 'application/json' in request.headers.get('accept', ''):
        return JSONResponse({'status': 'queued'})
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
    from datetime import timedelta

    with get_db_connection() as conn:
        thirty_days_ago = (now_utc() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
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
        rows = conn.execute("SELECT COALESCE(output_duration,duration) AS duration, file_size FROM episodes WHERE local_filename IS NOT NULL AND status != 'ignored'").fetchall()
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
    artwork_base_url = get_app_base_url(global_settings, request)
    for sub in subs:
        # Get completed episodes for this subscription
        with get_db_connection() as conn:
            episodes = conn.execute(
                """SELECT title, pub_date as published_date, status FROM episodes 
                   WHERE subscription_id = ? AND local_filename IS NOT NULL AND status != 'ignored'
                   ORDER BY pub_date DESC LIMIT 10""",
                (sub.id,)
            ).fetchall()
            
            # Get latest episode with AI summary
            latest_ep = conn.execute(
                """SELECT id, title, description, ai_summary, pub_date FROM episodes 
                   WHERE subscription_id = ? AND local_filename IS NOT NULL AND status != 'ignored'
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
        latest_episode_date = ""
        if latest_ep:
            latest_summary = latest_ep['ai_summary']
            latest_description = latest_ep['description']
        if latest_any_ep and latest_any_ep['pub_date']:
            # Z-suffixed UTC ISO so browser-side new Date() parses it correctly
            latest_episode_date = utc_iso(latest_any_ep['pub_date']) or str(latest_any_ep['pub_date'])
        
        subs_with_links.append({
            "sub": sub,
            "links": generate_rss_links(request, sub, global_settings, user),
            "episodes": [dict(ep) for ep in episodes],
            "episode_count": ep_repo.count_by_subscription(sub.id),
            "processing_count": processing_count,
            "total_listens": ep_repo.get_subscription_listen_count(sub.id),
            "latest_ai_summary": latest_summary,
            "latest_description": latest_description,
            "latest_episode_date": latest_episode_date,
            "in_user_library": sub_repo.is_in_user_library(user_id, sub.id) if user_id else True,
            "can_manage": _can_manage_subscription(user, sub),
            "owner_username": sub_repo.get_owner_username(sub.id),
            "user_library_count": user_library_count,
            "artwork_url": effective_artwork_url(sub, artwork_base_url),
        })

    # Get queue data for dashboard display
    queue = ep_repo.get_queue()

    from app.core.provider_readiness import provider_configuration_error
    config_warning = provider_configuration_error(global_settings)

    # Generate Unified Links if subscriptions exist
    unified_links = None
    if all_subs:
        unified_links = _build_rss_links(request, "/feed/unified.xml", global_settings, user)

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
            "settings": global_settings,
            "active_users": _active_users() if user and user.is_admin else [],
        }
    )

def _build_public_subscribe_context(request: Request, global_settings: dict):
    subs = sub_repo.get_all()
    artwork_base_url = get_app_base_url(global_settings, request)
    public_links = []
    for sub in subs:
        with get_db_connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as count
                   FROM episodes
                   WHERE subscription_id = ? AND local_filename IS NOT NULL AND status != 'ignored'""",
                (sub.id,)
            ).fetchone()
            latest = conn.execute(
                """SELECT title, pub_date
                   FROM episodes
                   WHERE subscription_id = ? AND local_filename IS NOT NULL AND status != 'ignored'
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()

        public_links.append({
            "sub": sub,
            "links": generate_rss_links(request, sub, global_settings, include_auth_token=False),
            "episode_count": row["count"] if row else 0,
            "latest_episode": dict(latest) if latest else None,
            "artwork_url": effective_artwork_url(sub, artwork_base_url),
        })

    unified_links = None
    if subs:
        unified_links = _build_rss_links(
            request, "/feed/unified.xml", global_settings, include_auth_token=False,
        )

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


@router.get("/api/dashboard/queue")
async def dashboard_queue():
    """Return the small, user-facing processing queue payload for partial refreshes."""
    items = []
    for episode in ep_repo.get_queue():
        items.append({
            "id": episode.get("id"),
            "title": episode.get("title") or "Untitled episode",
            "podcast_title": episode.get("podcast_title") or "Unknown podcast",
            "status": episode.get("status") or "pending",
            "processing_step": episode.get("processing_step"),
            "progress": max(0, min(100, int(episode.get("progress") or 0))),
        })
    return {"items": items, "total": len(items)}


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

# --- Admin: Podcast Preferences ---
@router.get("/admin/unified-feed", response_class=HTMLResponse)
async def admin_unified_feed(
    request: Request,
    admin_user=Depends(require_admin),
):
    global_settings = get_global_settings()
    base_url = get_app_base_url(global_settings, request)
    feed_settings = resolve_unified_feed_settings(global_settings, base_url)
    links = _build_rss_links(request, "/feed/unified.xml", global_settings, admin_user)

    return templates.TemplateResponse(
        request=request,
        name="admin/unified_feed.html",
        context={
            "csp_nonce": get_csp_nonce(request),
            "user": admin_user,
            "settings": feed_settings,
            "feed_url": links["direct"],
            "artwork_preview_url": resolve_unified_feed_artwork_preview(
                feed_settings["custom_artwork_url"], get_request_base_origin(request),
            ),
            "default_title": DEFAULT_UNIFIED_FEED_TITLE,
            "default_description": DEFAULT_UNIFIED_FEED_DESCRIPTION,
            "active_tab": "unified_feed",
        },
    )


@router.post("/admin/unified-feed/update")
async def update_unified_feed_settings(
    background_tasks: BackgroundTasks,
    unified_feed_title: str = Form(...),
    unified_feed_description: str = Form(...),
    unified_feed_include_podcast_name: bool = Form(False),
    unified_feed_artwork_url: str = Form(""),
    admin_user=Depends(require_admin),
):
    try:
        normalized = normalize_unified_feed_settings(
            unified_feed_title,
            unified_feed_description,
            unified_feed_include_podcast_name,
            unified_feed_artwork_url,
        )
    except ValueError as exc:
        return RedirectResponse(
            url=f"/admin/unified-feed?error={quote(str(exc))}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET unified_feed_title = ?,
                unified_feed_description = ?,
                unified_feed_include_podcast_name = ?,
                unified_feed_artwork_url = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (
                normalized["title"],
                normalized["description"],
                1 if normalized["include_podcast_name"] else 0,
                normalized["custom_artwork_url"],
            ),
        )
        conn.commit()

    from app.core.rss_gen import RSSGenerator

    background_tasks.add_task(RSSGenerator().generate_unified_feed)
    return RedirectResponse(
        url="/admin/unified-feed?success=Unified+feed+settings+updated",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/admin/unified-feed/reset")
async def reset_unified_feed_settings(
    background_tasks: BackgroundTasks,
    admin_user=Depends(require_admin),
):
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE app_settings
            SET unified_feed_title = ?,
                unified_feed_description = ?,
                unified_feed_include_podcast_name = 1,
                unified_feed_artwork_url = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (DEFAULT_UNIFIED_FEED_TITLE, DEFAULT_UNIFIED_FEED_DESCRIPTION),
        )
        conn.commit()

    from app.core.rss_gen import RSSGenerator

    background_tasks.add_task(RSSGenerator().generate_unified_feed)
    return RedirectResponse(
        url="/admin/unified-feed?success=Unified+feed+defaults+restored",
        status_code=status.HTTP_303_SEE_OTHER,
    )


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
            "active_tab": "global_subs",
        }
    )

@router.post("/admin/global-subscription-settings/update")
async def update_global_subscription_settings(
    request: Request,
    background_tasks: BackgroundTasks,
    default_remove_ads: bool = Form(False),
    default_remove_promos: bool = Form(False),
    default_remove_intros: bool = Form(False),
    default_remove_outros: bool = Form(False),
    default_ai_rewrite_description: bool = Form(False),
    default_ai_audio_summary: bool = Form(False),
    default_append_title_intro: bool = Form(False),
    default_watermark_artwork: bool = Form(False),
    default_retention_limit: int = Form(1),
    default_retention_days: int = Form(30),
    default_manual_retention_days: int = Form(14),
    default_custom_instructions: str = Form(None),
    whitelist_mode: bool = Form(False),
    default_processing_workflow: str | None = Form(None),
    default_remove_editorial_non_speech: bool | None = Form(None),
    default_remove_non_editorial_non_speech: bool | None = Form(None),
    default_minimum_retained_seconds: float | None = Form(None),
    timeline_settings_present: bool = Form(False),
    warning_tones_present: bool = Form(False),
    warning_tone_start: bool = Form(False),
    warning_tone_middle: bool = Form(False),
    warning_tone_end: bool = Form(False),
    admin_user = Depends(require_admin)
):
    from app.core.timeline import WORKFLOWS, threshold
    if default_processing_workflow is not None and default_processing_workflow not in WORKFLOWS:
        raise HTTPException(400, 'Unknown processing workflow')
    if default_minimum_retained_seconds is not None:
        try:
            default_minimum_retained_seconds = threshold(default_minimum_retained_seconds)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    if timeline_settings_present:
        default_remove_editorial_non_speech = bool(default_remove_editorial_non_speech)
        default_remove_non_editorial_non_speech = bool(default_remove_non_editorial_non_speech)
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
                default_watermark_artwork = ?,
                default_retention_limit = ?,
                default_retention_days = ?,
                default_manual_retention_days = ?,
                default_custom_instructions = ?,
                whitelist_mode = ?,
                default_processing_workflow = COALESCE(?, default_processing_workflow),
                default_remove_editorial_non_speech = COALESCE(?, default_remove_editorial_non_speech),
                default_remove_non_editorial_non_speech = COALESCE(?, default_remove_non_editorial_non_speech),
                default_minimum_retained_seconds = COALESCE(?, default_minimum_retained_seconds)
            WHERE id = 1
        """, (
            default_remove_ads, default_remove_promos, default_remove_intros, default_remove_outros,
            default_ai_rewrite_description, default_ai_audio_summary, default_append_title_intro,
            default_watermark_artwork,
            default_retention_limit, default_retention_days, default_manual_retention_days,
            default_custom_instructions, 1 if whitelist_mode else 0,
            default_processing_workflow, default_remove_editorial_non_speech,
            default_remove_non_editorial_non_speech, default_minimum_retained_seconds
        ))
        if warning_tones_present:
            conn.execute("""UPDATE app_settings SET warning_tone_start = ?, warning_tone_middle = ?,
                         warning_tone_end = ? WHERE id = 1""",
                         (warning_tone_start, warning_tone_middle, warning_tone_end))
        conn.commit()

    background_tasks.add_task(_reconcile_artwork_and_feeds)
        
    return RedirectResponse(url="/admin/global-subscription-settings?success=Settings updated", status_code=303)


@router.post("/add", response_class=HTMLResponse)
async def add_subscription(
    request: Request,
    background_tasks: BackgroundTasks,
    feed_url: str = Form(...),
    initial_count: str = Form("inherit"),
    user = Depends(require_auth),
):
    try:
        source = await asyncio.to_thread(resolve_source, feed_url)
        feed_url = source.canonical_url

        existing = (
            sub_repo.get_by_source_identity(source.source_type, source.external_id)
            or sub_repo.get_by_url(feed_url)
        )
        if existing:
            added = sub_repo.add_to_user_library(_real_user_id(user), existing.id)
            message = "Podcast added to My Podcasts from the library" if added else "Podcast is already in My Podcasts"
            return RedirectResponse(url=f"/?view=mine&success={quote(message)}", status_code=303)
        
        with get_db_connection() as conn:
            app_settings = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()

        inherit_retention = str(initial_count).strip().lower() == "inherit"
        default_retention_limit = app_settings["default_retention_limit"]
        if default_retention_limit is None:
            default_retention_limit = 1
        if source.source_type.startswith("youtube_"):
            retention_limit = default_retention_limit
            inherit_retention = True
            if str(initial_count).strip().lower() == "inherit":
                initial_limit = min(max(int(default_retention_limit), 0), 5)
            else:
                try:
                    initial_limit = int(initial_count)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid initial video count") from exc
                if initial_limit not in {0, 1, 3, 5}:
                    raise ValueError("YouTube initial import must be 0, 1, 3, or 5 videos")
        elif inherit_retention:
            retention_limit = default_retention_limit
            initial_limit = retention_limit
        else:
            try:
                retention_limit = int(initial_count)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid initial episode count") from exc
            if retention_limit < 0:
                raise ValueError("Initial episode count cannot be negative")
            initial_limit = retention_limit

        slug = sub_repo.available_slug(source.slug, source.external_id)
        sub_create = SubscriptionCreate(feed_url=feed_url)
        new_sub = sub_repo.create(
            sub_create,
            source.title,
            slug,
            source.image_url,
            source.description,
            retention_limit=retention_limit,
            owner_user_id=_real_user_id(user),
            inherit_retention=inherit_retention,
            source_type=source.source_type,
            source_external_id=source.external_id,
        )
        
        # Apply other global defaults immediately
        sub_repo.update_settings(
            new_sub.id,
            remove_ads=bool(app_settings['default_remove_ads']),
            remove_promos=bool(app_settings['default_remove_promos']),
            remove_intros=bool(app_settings['default_remove_intros']),
            remove_outros=bool(app_settings['default_remove_outros']),
            custom_instructions=app_settings['default_custom_instructions'],
            append_summary=False,
            append_title_intro=bool(app_settings['default_append_title_intro']),
            ai_rewrite_description=bool(app_settings['default_ai_rewrite_description']),
            ai_audio_summary=bool(app_settings['default_ai_audio_summary']),
            retention_days=app_settings['default_retention_days'] or 30,
            manual_retention_days=app_settings['default_manual_retention_days'] or 14,
            retention_limit=retention_limit
        )

        
        async def setup_subscription(sub_id: int, title: str, limit: int):
            from app.core.processor import Processor
            try:
                try:
                    ArtworkWatermarker().reconcile(sub_id)
                except Exception as exc:
                    logger.warning("Initial artwork generation failed for subscription %s: %s", sub_id, exc)

                await send_notification_async(
                    EVENT_NEW_PODCAST,
                    "Podcast added",
                    f"{title} was added to the global podcast library.",
                    severity="success",
                )
                
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
        
        background_tasks.add_task(setup_subscription, new_sub.id, source.title, initial_limit)
        
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

    in_user_library = sub_repo.is_in_user_library(user_id, id)
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(
            {
                "status": "updated",
                "subscription_id": id,
                "changed": changed if "changed" in locals() else False,
                "in_user_library": in_user_library,
                "user_library_count": sub_repo.count_user_library_members(id),
                "message": message,
            }
        )

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


@router.post("/subscriptions/bulk-settings")
async def bulk_update_subscription_settings(
    background_tasks: BackgroundTasks,
    subscription_ids: list[int] = Form(...),
    content_mode: str = Form("unchanged"),
    remove_ads: bool = Form(False),
    remove_promos: bool = Form(False),
    remove_intros: bool = Form(False),
    remove_outros: bool = Form(False),
    retention_mode: str = Form("unchanged"),
    retention_limit: int = Form(1),
    retention_days: int = Form(30),
    manual_retention_days: int = Form(14),
    features_mode: str = Form("unchanged"),
    ai_rewrite_description: bool = Form(False),
    ai_audio_summary: bool = Form(False),
    append_title_intro: bool = Form(False),
    watermark_artwork: bool = Form(False),
    instructions_mode: str = Form("unchanged"),
    custom_instructions: str = Form(""),
    owner_mode: str = Form("unchanged"),
    owner_user_id: str = Form(""),
    user = Depends(require_auth),
):
    ids = list(dict.fromkeys(subscription_ids))
    if not ids:
        raise HTTPException(status_code=400, detail="Select at least one podcast")
    if len(ids) > 500:
        raise HTTPException(status_code=400, detail="Bulk updates are limited to 500 podcasts")

    valid_group_modes = {"unchanged", "inherit", "override"}
    if any(
        mode not in valid_group_modes
        for mode in (content_mode, retention_mode, features_mode, instructions_mode)
    ):
        raise HTTPException(status_code=400, detail="Invalid bulk settings mode")
    if owner_mode not in {"unchanged", "set", "clear"}:
        raise HTTPException(status_code=400, detail="Invalid owner update mode")

    subscriptions = [sub_repo.get_by_id(subscription_id) for subscription_id in ids]
    if any(sub is None for sub in subscriptions):
        raise HTTPException(status_code=404, detail="One or more podcasts no longer exist")
    if any(not _can_manage_subscription(user, sub) for sub in subscriptions):
        raise HTTPException(
            status_code=403,
            detail="The selection contains a podcast you are not allowed to manage",
        )
    if owner_mode != "unchanged" and not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Only administrators can reassign owners")

    owner_id = None
    if owner_mode == "set":
        try:
            owner_id = int(owner_user_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Select a valid owner") from exc
        if owner_id <= 0:
            raise HTTPException(status_code=400, detail="Select a valid owner")

    custom_instructions = custom_instructions.strip()
    if instructions_mode == "override" and not custom_instructions:
        raise HTTPException(
            status_code=400,
            detail="Enter custom instructions or select global instructions",
        )
    if retention_mode == "override" and (
        retention_limit < 0 or retention_days < 1 or manual_retention_days < 1
    ):
        raise HTTPException(status_code=400, detail="Retention values are outside the allowed range")

    assignments = []
    values = []

    def set_value(column: str, value):
        assignments.append(f"{column} = ?")
        values.append(value)

    if content_mode == "inherit":
        set_value("inherit_content_removal", 1)
    elif content_mode == "override":
        set_value("inherit_content_removal", 0)
        for column, value in (
            ("remove_ads", remove_ads),
            ("remove_promos", remove_promos),
            ("remove_intros", remove_intros),
            ("remove_outros", remove_outros),
        ):
            set_value(column, int(value))

    if retention_mode == "inherit":
        set_value("inherit_retention", 1)
    elif retention_mode == "override":
        set_value("inherit_retention", 0)
        set_value("retention_limit", retention_limit)
        set_value("retention_days", retention_days)
        set_value("manual_retention_days", manual_retention_days)

    if features_mode == "inherit":
        set_value("inherit_default_features", 1)
    elif features_mode == "override":
        set_value("inherit_default_features", 0)
        for column, value in (
            ("ai_rewrite_description", ai_rewrite_description),
            ("ai_audio_summary", ai_audio_summary),
            ("append_title_intro", append_title_intro),
            ("watermark_artwork", watermark_artwork),
        ):
            set_value(column, int(value))

    if instructions_mode == "inherit":
        set_value("inherit_custom_instructions", 1)
    elif instructions_mode == "override":
        set_value("inherit_custom_instructions", 0)
        set_value("custom_instructions", custom_instructions)

    if owner_mode == "set":
        set_value("owner_user_id", owner_id)
    elif owner_mode == "clear":
        set_value("owner_user_id", None)

    if not assignments:
        return RedirectResponse(
            url="/?view=library&success=" + quote("No bulk changes selected"),
            status_code=303,
        )

    placeholders = ",".join("?" for _ in ids)
    with get_db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if owner_mode == "set":
            owner = conn.execute("SELECT id FROM users WHERE id = ?", (owner_id,)).fetchone()
            if not owner:
                conn.rollback()
                raise HTTPException(status_code=400, detail="Selected owner no longer exists")
        conn.execute(
            f"UPDATE subscriptions SET {', '.join(assignments)} WHERE id IN ({placeholders})",
            (*values, *ids),
        )
        if owner_mode == "set":
            conn.executemany(
                """
                INSERT OR IGNORE INTO user_subscriptions (user_id, subscription_id)
                VALUES (?, ?)
                """,
                [(owner_id, subscription_id) for subscription_id in ids],
            )
        conn.commit()

    async def post_bulk_update_tasks():
        await asyncio.to_thread(_reconcile_artwork_and_feeds, ids)
        processor = Processor()
        await processor.cleanup_old_episodes()
        for subscription_id in ids:
            await processor.check_feeds(subscription_id)

    background_tasks.add_task(post_bulk_update_tasks)
    return RedirectResponse(
        url="/?view=library&success="
        + quote(f"Updated settings for {len(ids)} podcast{'s' if len(ids) != 1 else ''}"),
        status_code=303,
    )


@router.post("/subscriptions/bulk-delete")
async def bulk_delete_subscriptions(
    background_tasks: BackgroundTasks,
    subscription_ids: list[int] = Form(...),
    delete_confirmation: str = Form(""),
    admin_user = Depends(require_admin),
):
    """Start the normal durable deletion workflow for selected subscriptions."""
    ids = list(dict.fromkeys(subscription_ids))
    if not ids:
        raise HTTPException(status_code=400, detail="Select at least one podcast")
    if len(ids) > 500:
        raise HTTPException(status_code=400, detail="Bulk deletion is limited to 500 podcasts")
    if delete_confirmation != "delete":
        raise HTTPException(status_code=400, detail="Confirm the permanent bulk deletion")

    subscriptions = [sub_repo.get_by_id(subscription_id) for subscription_id in ids]
    if any(sub is None for sub in subscriptions):
        raise HTTPException(status_code=404, detail="One or more podcasts no longer exist")

    processor = Processor()
    for subscription_id in ids:
        await asyncio.to_thread(processor.sub_repo.begin_deletion, subscription_id)

    async def finish_bulk_deletion():
        for subscription_id in ids:
            try:
                await processor.delete_subscription(subscription_id)
            except Exception:
                logger.exception(
                    "Background cleanup failed for bulk-deleted subscription %s; "
                    "the durable deletion state will be retried",
                    subscription_id,
                )

    background_tasks.add_task(finish_bulk_deletion)
    return RedirectResponse(
        url="/?view=library&success="
        + quote(
            f"Deletion started for {len(ids)} podcast{'s' if len(ids) != 1 else ''}"
        ),
        status_code=303,
    )


@router.get("/episodes/{episode_id}/audio")
async def play_episode(episode_id: int):
    from pathlib import Path
    from app.core.config import settings as app_settings
    episode = ep_repo.get_by_id(episode_id)
    if not episode or episode.status == 'ignored' or not episode.local_filename:
        raise HTTPException(404, 'Published audio not found')
    path = Path(episode.local_filename).resolve()
    root = Path(app_settings.PODCASTS_DIR).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, 'Published audio not found')
    return RedirectResponse('/audio/' + quote(path.relative_to(root).as_posix(), safe='/'), status_code=307)


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
            "settings": global_settings,
            "artwork_url": effective_artwork_url(sub, get_app_base_url(global_settings, request)),
        }
    )

@router.get("/api/subscriptions/{id}/episodes")
async def get_subscription_episodes_api(request: Request, id: int, limit: int = 20, offset: int = 0, search: str = None, filter: str = 'all', user = Depends(require_auth)):
    """Return episodes for a subscription as JSON for lazy loading. Supports search by title."""
    sub = sub_repo.get_by_id(id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    if limit < 1 or limit > 100 or offset < 0 or filter not in {'all','completed','manual','not-downloaded','ignored','played'}:
        raise HTTPException(422, 'Invalid episode pagination or filter')
    # Filter before pagination, with the same predicates for the count.
    episodes = ep_repo.get_by_subscription_paginated(id, limit=limit, offset=offset, search=search, filter=filter)
    total = ep_repo.count_by_subscription(id, search=search, filter=filter)
    
    # Convert sqlite rows to dicts, every timestamp Z-suffixed for the browser
    episodes_data = [with_utc_timestamps(ep) for ep in episodes]
    
    return {
        "episodes": episodes_data,
        "html": templates.get_template('_episode_cards.html').render(
            request=request, episodes=episodes, format_duration=format_duration,
            can_manage_subscription=_can_manage_subscription(user, sub),
        ),
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
    watermark_artwork: bool = Form(False),
    retention_days: int = Form(30),
    manual_retention_days: int = Form(14),
    retention_limit: int = Form(1),
    inherit_content_removal: bool = Form(False),
    inherit_retention: bool = Form(False),
    inherit_default_features: bool = Form(False),
    inherit_custom_instructions: bool = Form(False),
    processing_workflow: str | None = Form(None),
    inherit_processing_workflow: bool | None = Form(None),
    remove_editorial_non_speech: bool | None = Form(None),
    remove_non_editorial_non_speech: bool | None = Form(None),
    minimum_retained_seconds: float | None = Form(None),
    timeline_settings_present: bool = Form(False),
    user = Depends(require_auth),
):
    sub = sub_repo.get_by_id(id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if not _can_manage_subscription(user, sub):
        raise HTTPException(status_code=403, detail="Only admins and the podcast owner can change podcast settings")

    stored = sub.setting_overrides
    from app.core.timeline import WORKFLOWS, threshold
    if processing_workflow is not None and inherit_processing_workflow is None:
        inherit_processing_workflow = False
    if processing_workflow is not None and processing_workflow not in WORKFLOWS:
        raise HTTPException(400, 'Unknown processing workflow')
    if minimum_retained_seconds is not None:
        try:
            minimum_retained_seconds = threshold(minimum_retained_seconds)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    if timeline_settings_present:
        inherit_processing_workflow = bool(inherit_processing_workflow)
        if inherit_processing_workflow:
            processing_workflow = stored.get('processing_workflow')
        if inherit_content_removal:
            remove_editorial_non_speech = stored.get('remove_editorial_non_speech')
            remove_non_editorial_non_speech = stored.get('remove_non_editorial_non_speech')
            minimum_retained_seconds = stored.get('minimum_retained_seconds')
        else:
            remove_editorial_non_speech = bool(remove_editorial_non_speech)
            remove_non_editorial_non_speech = bool(remove_non_editorial_non_speech)
    if inherit_content_removal:
        remove_ads = bool(stored.get("remove_ads"))
        remove_promos = bool(stored.get("remove_promos"))
        remove_intros = bool(stored.get("remove_intros"))
        remove_outros = bool(stored.get("remove_outros"))
    if inherit_retention:
        retention_days = stored.get("retention_days") if stored.get("retention_days") is not None else 30
        manual_retention_days = (
            stored.get("manual_retention_days")
            if stored.get("manual_retention_days") is not None
            else 14
        )
        retention_limit = (
            stored.get("retention_limit")
            if stored.get("retention_limit") is not None
            else 1
        )
    if inherit_default_features:
        append_summary = bool(stored.get("append_summary"))
        append_title_intro = bool(stored.get("append_title_intro"))
        ai_rewrite_description = bool(stored.get("ai_rewrite_description"))
        ai_audio_summary = bool(stored.get("ai_audio_summary"))
        watermark_artwork = bool(stored.get("watermark_artwork"))
    if inherit_custom_instructions:
        custom_instructions = stored.get("custom_instructions")
    else:
        custom_instructions = (custom_instructions or "").strip()
        if not custom_instructions:
            return RedirectResponse(
                url=f"/subscriptions/{id}?error={quote('Enter custom instructions or use the global instructions')}",
                status_code=303,
            )

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
        retention_limit,
        inherit_content_removal=inherit_content_removal,
        inherit_retention=inherit_retention,
        inherit_default_features=inherit_default_features,
        inherit_custom_instructions=inherit_custom_instructions,
        watermark_artwork=watermark_artwork,
        processing_workflow=processing_workflow,
        inherit_processing_workflow=inherit_processing_workflow,
        remove_editorial_non_speech=remove_editorial_non_speech,
        remove_non_editorial_non_speech=remove_non_editorial_non_speech,
        minimum_retained_seconds=minimum_retained_seconds,
    )
    
    # Trigger processing if any ads/promos settings were changed
    from app.core.processor import Processor
    proc = Processor()
    
    async def post_update_tasks(sub_id):
        await asyncio.to_thread(_reconcile_artwork_and_feeds, [sub_id])
        await proc.cleanup_old_episodes()
        await proc.check_feeds(sub_id)
        await proc.process_queue()

    background_tasks.add_task(post_update_tasks, id)
    return RedirectResponse(url=f"/subscriptions/{id}", status_code=303)

@router.get("/artwork/{subscription_id}.jpg")
# Feed XML cached by a client can still reference the pre-JPEG .png URL. Serving
# the current file under both paths keeps those clients from getting a 404.
@router.get("/artwork/{subscription_id}.png")
async def serve_watermarked_artwork(subscription_id: int):
    sub = sub_repo.get_by_id(subscription_id)
    if not sub or not sub.watermark_artwork or not sub.watermarked_image_path:
        raise HTTPException(status_code=404, detail="Artwork not found")

    path = Path(sub.watermarked_image_path).resolve()
    root = Path(runtime_settings.ARTWORK_DIR).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Artwork not found")

    return FileResponse(
        path,
        media_type="image/jpeg" if path.suffix == ".jpg" else "image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )

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
            
        transcript_path = artifact_path(dict(row), 'transcript_path', 'transcript.json')
        if not transcript_path or not os.path.exists(transcript_path):
             raise HTTPException(status_code=404, detail="Transcript file not found")
             
        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading transcript: {str(e)}")
    

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
            path = artifact_path(dict(row), 'transcript_path', 'transcript.json')
            if path:
                return FileResponse(path, media_type='application/json')

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
            for column, name in [('report_path', 'report.html'), ('ad_report_path', 'report.json')]:
                path = artifact_path(dict(row), column, name)
                if path:
                    return FileResponse(path)

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
