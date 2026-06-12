import bcrypt
import ipaddress
import secrets
import string
from typing import Optional
from urllib.parse import urlparse

from app.core.config import settings

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def is_bcrypt_hash(value: str | None) -> bool:
    return bool(value and value.startswith(("$2a$", "$2b$", "$2y$")))

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False

def verify_feed_password(password: str, stored_password: str | None) -> bool:
    """Verify standalone feed Basic Auth password against bcrypt or legacy plaintext."""
    if not stored_password:
        return False
    if is_bcrypt_hash(stored_password):
        return verify_password(password, stored_password)
    return password == stored_password

def generate_secure_password(length: int = 16) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    # Ensure at least one of each type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*")
    ]
    # Fill the rest
    password += [secrets.choice(alphabet) for _ in range(length - 4)]
    # Shuffle
    secrets.SystemRandom().shuffle(password)
    return ''.join(password)

def get_client_ip(request) -> str:
    """Extract client IP from request, considering proxies."""
    if settings.TRUST_PROXY_HEADERS:
        # Only trust these when a reverse proxy strips client-supplied copies.
        if "CF-Connecting-IP" in request.headers:
            return request.headers["CF-Connecting-IP"]
        if "X-Forwarded-For" in request.headers:
            return request.headers["X-Forwarded-For"].split(",")[0].strip()
        if "X-Real-IP" in request.headers:
            return request.headers["X-Real-IP"]
    # Fallback to direct client
    return request.client.host if request.client else "unknown"

def _header_value(headers, name: str) -> str | None:
    return headers.get(name) or headers.get(name.lower()) or headers.get(name.upper())

def _normalize_origin(value: str | None) -> str | None:
    if not value:
        return None

    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.hostname:
        return None

    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    port = parsed.port

    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        return f"{scheme}://{hostname}:{port}"
    return f"{scheme}://{hostname}"

def get_request_base_origin(request) -> str | None:
    """Return the effective request origin, optionally honoring trusted proxy headers."""
    scheme = getattr(request.url, "scheme", "http")
    host = _header_value(request.headers, "host") or getattr(request.url, "netloc", None)

    if settings.TRUST_PROXY_HEADERS:
        forwarded_proto = _header_value(request.headers, "x-forwarded-proto")
        forwarded_host = _header_value(request.headers, "x-forwarded-host")
        if forwarded_proto:
            scheme = forwarded_proto.split(",")[0].strip()
        if forwarded_host:
            host = forwarded_host.split(",")[0].strip()

    if not host:
        return None
    return _normalize_origin(f"{scheme}://{host}")

def is_same_origin_request(request, app_external_url: str | None = None) -> bool:
    """
    Validate browser-sent Origin/Referer headers for unsafe authenticated requests.

    Missing Origin/Referer is allowed for compatibility with simple local clients and
    older form submissions. Mismatched values are rejected when present.
    """
    if getattr(request, "method", "GET").upper() not in UNSAFE_METHODS:
        return True

    supplied_origin = _normalize_origin(_header_value(request.headers, "origin"))
    if not supplied_origin:
        supplied_origin = _normalize_origin(_header_value(request.headers, "referer"))
    if not supplied_origin:
        return True

    allowed_origins = {
        origin
        for origin in (
            get_request_base_origin(request),
            _normalize_origin(app_external_url),
        )
        if origin
    }

    return supplied_origin in allowed_origins

def is_ip_allowed(ip: str, allowlist: Optional[str]) -> bool:
    """Check if IP is in the allowlist."""
    if not allowlist or not allowlist.strip():
        return True  # No allowlist means all IPs allowed
    
    allowed_entries = [entry.strip() for entry in allowlist.split(",") if entry.strip()]

    try:
        client_ip = ipaddress.ip_address(ip)
    except ValueError:
        return ip in allowed_entries

    for entry in allowed_entries:
        try:
            if "/" in entry:
                if client_ip in ipaddress.ip_network(entry, strict=False):
                    return True
            elif client_ip == ipaddress.ip_address(entry):
                return True
        except ValueError:
            if ip == entry:
                return True

    return False
