import bcrypt
import secrets
import string
from typing import Optional

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False

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
    # Check for Cloudflare headers first
    if "CF-Connecting-IP" in request.headers:
        return request.headers["CF-Connecting-IP"]
    # Check for standard proxy headers
    if "X-Forwarded-For" in request.headers:
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    if "X-Real-IP" in request.headers:
        return request.headers["X-Real-IP"]
    # Fallback to direct client
    return request.client.host if request.client else "unknown"

def is_ip_allowed(ip: str, allowlist: Optional[str]) -> bool:
    """Check if IP is in the allowlist."""
    if not allowlist or not allowlist.strip():
        return True  # No allowlist means all IPs allowed
    
    allowed_ips = [ip.strip() for ip in allowlist.split(",") if ip.strip()]
    
    # Simple exact match for now (can be extended to support CIDR)
    return ip in allowed_ips
