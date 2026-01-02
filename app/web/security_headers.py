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
        # Generate cryptographic nonce for this request
        # Used for Content-Security-Policy to allow specific inline scripts
        import secrets
        csp_nonce = secrets.token_urlsafe(16)
        
        # Store nonce in request state so templates can access it
        request.state.csp_nonce = csp_nonce
        
        response = await call_next(request)
        
        # Content Security Policy - Defense against XSS and injection attacks
        # Uses nonce-based approach to allow specific inline scripts while blocking arbitrary code execution
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{csp_nonce}' 'unsafe-hashes' https://cdnjs.cloudflare.com https://static.cloudflareinsights.com; "  # unsafe-hashes allows inline event handlers
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "style-src-elem 'self' 'unsafe-inline' https://fonts.googleapis.com; "  # Explicit style-src-elem
            "img-src 'self' data: blob: https:; "
            "media-src 'self' blob:; "
            "font-src 'self' data: https://fonts.gstatic.com; "
            "connect-src 'self' https://cloudflareinsights.com; "  # Allow Cloudflare beacon
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
