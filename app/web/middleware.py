from fastapi import Request, HTTPException, status
from fastapi.responses import Response
import bcrypt
import base64

async def feed_auth_middleware(request: Request, call_next):
    """
    Middleware to protect /feeds/* and /audio/* routes with HTTP Basic Auth.
    - If user auth is enabled: uses user credentials
    - If user auth is disabled: uses global feed credentials
    """
    path = request.url.path
    
    # Only protect feeds and audio routes
    if not (path.startswith('/feeds/') or path.startswith('/audio/')):
        return await call_next(request)
    
    # Check if feed auth is enabled
    from app.web.router import get_global_settings
    settings = get_global_settings()
    
    # Determine if we should enforce auth
    if not settings.get('enable_feed_auth'):
        return await call_next(request)
    
    # Allow access if already logged in via session
    from app.web.auth import SESSION_USER_KEY
    if request.session.get(SESSION_USER_KEY):
        return await call_next(request)
    
    # Check for Authorization header
    auth_header = request.headers.get('Authorization')
    encoded_credentials = None
    
    if auth_header and auth_header.startswith('Basic '):
        encoded_credentials = auth_header.split(' ')[1]
    else:
        # Fallback: Check for ?auth= query parameter
        encoded_credentials = request.query_params.get('auth')
    
    if not encoded_credentials:
        return Response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={'WWW-Authenticate': 'Basic realm="Podcast Feeds"'}
        )
    
    # Decode credentials
    try:
        decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
        username, password = decoded_credentials.split(':', 1)
    except Exception:
        return Response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={'WWW-Authenticate': 'Basic realm="Podcast Feeds"'}
        )
    
    # Determine which credentials to check
    if settings.get('auth_enabled'):
        # User auth is enabled - check against users table
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            user_row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
        
        if not user_row or not bcrypt.checkpw(password.encode('utf-8'), user_row['password_hash'].encode('utf-8')):
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={'WWW-Authenticate': 'Basic realm="Podcast Feeds"'}
            )
    else:
        # User auth is disabled - check against global feed credentials
        expected_username = settings.get('feed_auth_username')
        expected_password_hash = settings.get('feed_auth_password')
        
        if not expected_username or not expected_password_hash:
            # Feed auth enabled but not configured - allow access (or maybe block? safe to allow if not configured)
            return await call_next(request)
        
        if username != expected_username:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={'WWW-Authenticate': 'Basic realm="Podcast Feeds"'}
            )
        
        # Verify password (plaintext for global feed credentials)
        if password != expected_password_hash:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={'WWW-Authenticate': 'Basic realm="Podcast Feeds"'}
            )
    
    # Authentication successful
    return await call_next(request)
