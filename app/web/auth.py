from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from typing import Optional
import logging
from datetime import datetime

from app.infra.database import get_db_connection
from app.web.auth_utils import get_client_ip, is_ip_allowed, verify_password
from app.core.models import User

logger = logging.getLogger(__name__)

# Session key
SESSION_USER_KEY = "user_id"

def get_current_user(request: Request) -> Optional[User]:
    """Get the currently logged-in user from session."""
    user_id = request.session.get(SESSION_USER_KEY)
    
    # Check if auth is disabled globally - treat everyone as admin
    try:
        with get_db_connection() as conn:
            settings = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()
            if settings and not settings['auth_enabled']:
                return User(
                    id=0, 
                    username="admin", 
                    password_hash="", 
                    is_admin=True, 
                    created_at=datetime.now(), 
                    last_login=datetime.now()
                )
    except Exception as e:
        logger.error(f"Error checking auth settings: {e}")

    if not user_id:
        return None
    
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return User.model_validate(dict(row))
    return None

def require_auth(request: Request) -> User:
    """Dependency that requires authentication."""
    # First check if auth is enabled globally
    with get_db_connection() as conn:
        settings = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()
        
    # If settings exist and auth is disabled, return a dummy admin user
    if settings and not settings['auth_enabled']:
        return User(
            id=0, 
            username="admin", 
            password_hash="", 
            is_admin=True, 
            created_at=datetime.now(), 
            last_login=datetime.now()
        )

    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user

def require_admin(request: Request) -> User:
    """Dependency that requires admin privileges."""
    user = require_auth(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user

async def auth_middleware(request: Request, call_next):
    """
    Middleware to handle authentication and IP allowlisting.
    """
    path = request.url.path
    
    # Skip auth/IP check for specific paths
    # static: always public
    # feeds/audio: public to world (IP check skipped), but might be protected by Feed Auth elsewhere
    if path in ["/login", "/request-access", "/submit-access-request"] or \
       path.startswith("/static/") or \
       path.startswith("/feeds/") or \
       path.startswith("/feed/") or \
       path.startswith("/audio/"):
        return await call_next(request)
    
    # Check if auth is enabled
    with get_db_connection() as conn:
        settings = conn.execute("SELECT auth_enabled, ip_allowlist FROM app_settings WHERE id = 1").fetchone()
    
    if not settings:
        return await call_next(request)

    # 1. GLOBAL IP CHECK (High Priority)
    # If an allowlist is set, it applies to EVERYTHING (Admin, Feeds, Audio, Dashboard)
    if settings['ip_allowlist']:
        client_ip = get_client_ip(request)
        if not is_ip_allowed(client_ip, settings['ip_allowlist']):
            logger.warning(f"AUTH - IP blocked: {client_ip} - Path: {path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied from your IP address"
            )

    # 2. USER AUTHENTICATION CHECK
    # Only if auth is enabled
    if settings['auth_enabled']:
        # Dashboard and Admin routes require user auth
        user = get_current_user(request)
        if not user:
            # Log the attempt
            client_ip = get_client_ip(request)
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO login_attempts (username, ip_address, success, user_agent) VALUES (?, ?, ?, ?)",
                    (None, client_ip, 0, request.headers.get("user-agent", ""))
                )
                conn.commit()
            
            logger.info(f"AUTH - Unauthorized access attempt: {client_ip} - Path: {path}")
            return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        
        # Check if password change is required
        with get_db_connection() as conn:
            settings_row = conn.execute("SELECT require_password_change FROM app_settings WHERE id = 1").fetchone()
            if settings_row and settings_row['require_password_change'] and path != "/change-password":
                return RedirectResponse(url="/change-password", status_code=status.HTTP_302_FOUND)
                
        # 3. ADMIN PRIVILEGE CHECK
        # Protect /admin routes from non-admin users
        if path.startswith("/admin") and not user.is_admin:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
    
    return await call_next(request)

def log_login_attempt(username: str, ip_address: str, success: bool, user_agent: str):
    """Log a login attempt to the database."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO login_attempts (username, ip_address, success, user_agent) VALUES (?, ?, ?, ?)",
            (username, ip_address, 1 if success else 0, user_agent)
        )
        conn.commit()
    
    status_str = "SUCCESS" if success else "FAILED"
    logger.info(f"AUTH - Login {status_str}: {username} from {ip_address}")
