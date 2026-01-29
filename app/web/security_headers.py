"""
Security headers middleware for protecting against common web vulnerabilities.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all HTTP responses to protect against:
    - XSS attacks (Content-Security-Policy with nonce-based approach)
    - Clickjacking (X-Frame-Options)
    - MIME-sniffing (X-Content-Type-Options)
    - Protocol downgrade attacks (Strict-Transport-Security)
    - Information leakage (Referrer-Policy)
    - Unauthorized feature access (Permissions-Policy)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate nonce for backward compatibility with templates that reference it
        # Note: Not used in CSP due to incompatibility with inline event handlers
        import secrets
        request.state.csp_nonce = secrets.token_urlsafe(16)
        
        response = await call_next(request)
        
        # Content Security Policy - Defense against XSS and injection attacks
        # Note: Using 'unsafe-inline' for scripts to support 70+ inline event handlers (onclick, etc.)
        # Nonce-based CSP is incompatible with inline event handlers (nonces don't apply to them,
        # and their presence causes 'unsafe-inline' to be ignored per CSP spec)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://static.cloudflareinsights.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "style-src-elem 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data: blob: https:; "
            "media-src 'self' blob:; "
            "font-src 'self' data: https://fonts.gstatic.com; "
            "connect-src 'self' https://cloudflareinsights.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # Strict Transport Security - Enforces HTTPS connections
        # Applied unconditionally to support reverse proxy/load balancer deployments
        # where the app sees HTTP but clients connect via HTTPS
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        
        # X-Frame-Options - Prevents clickjacking by blocking iframe embedding
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options - Prevents MIME-sniffing attacks
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Referrer-Policy - Controls referrer information leakage
        # strict-origin-when-cross-origin provides good balance between privacy and functionality
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy - Controls browser feature access (formerly Feature-Policy)
        # Restrictive by default, only allow necessary features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )
        
        # X-XSS-Protection - Legacy header for older browsers
        # Modern browsers use CSP instead, but this provides backward compatibility
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response
