import re
import os

DEFAULT_BASE_URL = "http://localhost:8000"


def is_running_in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.environ.get("container") is not None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and remove illegal characters.
    """
    if not filename:
        return "unknown"
    
    # Remove any directory separators
    filename = os.path.basename(filename)
    
    # Replace common traversal patterns and illegal characters
    # Keep alphanumeric, dots, dashes, and underscores
    sanitized = re.sub(r'[^\w\.\-\ ]', '_', filename)
    
    # Prevent hidden files or relative paths
    while sanitized.startswith(('.', '_')):
        sanitized = sanitized[1:]
        
    if not sanitized:
        return "unnamed_file"
        
    return sanitized

def sanitize_path_segment(segment: str) -> str:
    """
    Sanitize a string to be used as a safe directory name or path segment.
    """
    if not segment:
        return "unknown"
        
    # Remove any dots that could be used for traversal
    segment = segment.replace('..', '')
    
    # Replace slashes and other dangerous characters
    sanitized = re.sub(r'[^\w\-\ ]', '_', segment)
    
    # Trim and normalize
    sanitized = sanitized.strip().replace(' ', '_')
    
    if not sanitized:
        return "unnamed_segment"
        
    return sanitized

def get_lan_ip():
    import socket
    try:
        # Use a dummy connection to a public IP to find the preferred interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def get_app_base_url(global_settings: dict, request=None) -> str:
    """Consolidated logic for getting the application base URL."""
    from app.core.config import settings

    external_url = global_settings.get("app_external_url")
    
    if external_url and external_url.strip():
        return external_url.rstrip("/")

    if request:
        return str(request.base_url).rstrip("/")

    if settings.BASE_URL and settings.BASE_URL != DEFAULT_BASE_URL:
        return settings.BASE_URL.rstrip("/")

    if is_running_in_container():
        return settings.BASE_URL.rstrip("/")

    # Bare-metal fallback only. In Docker this is usually the container IP.
    lan_ip = get_lan_ip()
    
    if lan_ip and lan_ip != "localhost":
        return f"http://{lan_ip}:{settings.PORT}"
    
    return settings.BASE_URL.rstrip("/")

def get_global_settings():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        # Check if table exists first (bootstrap safety)
        try:
            row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
            if row:
                return dict(row)
        except Exception:
            pass
    return {}
