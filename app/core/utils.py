import re
import os

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
    
    # If no external URL is configured, PREFER the LAN IP address.
    lan_ip = get_lan_ip()
    
    if lan_ip and lan_ip != "localhost":
        return f"http://{lan_ip}:{settings.PORT}"

    # Fallback to request URL or settings BASE_URL
    if request:
        return str(request.base_url).rstrip("/")
    
    return settings.BASE_URL.rstrip("/")
