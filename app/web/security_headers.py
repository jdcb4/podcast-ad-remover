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
    - XSS attacks (Content-Security-Policy)
    - Clickjacking (X-Frame-Options)
    - MIME-sniffing (X-Content-Type-Options)
    - Protocol downgrade attacks (Strict-Transport-Security)
    - Information leakage (Referrer-Policy)
    - Unauthorized feature access (Permissions-Policy)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Content Security Policy - Defense against XSS and injection attacks
        # Configured to allow self-hosted content and inline styles/scripts where needed
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "  # Allow inline scripts for dynamic content
            "style-src 'self' 'unsafe-inline'; "  # Allow inline styles
            "img-src 'self' data: blob:; "  # Allow images from self, data URIs, and blobs
            "media-src 'self' blob:; "  # Allow audio/video from self and blobs
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "  # Prevent embedding in iframes (redundant with X-Frame-Options)
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # Strict Transport Security - Enforces HTTPS connections
        # Only add if the request was made over HTTPS to avoid browser errors
        if request.url.scheme == "https":
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
