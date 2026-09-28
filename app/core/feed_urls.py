"""Conservative feed URL identity shared by all subscription entry points."""
import re
from urllib.parse import urlsplit, urlunsplit


def feed_key(value: str) -> str:
    """Conservative URL identity: never discard query tokens or change path case."""
    value = value.strip()
    if len(value) > 4096 or re.search(r"[\x00-\x20\x7f]", value):
        raise ValueError("Use a feed URL without spaces or control characters (maximum 4,096 characters).")
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("Use a complete http:// or https:// feed URL.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs with embedded usernames or passwords are not supported.")
    host = parts.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port and (parts.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", parts.query, ""))

