"""Global presentation of individual RSS channel titles; source names stay intact."""
import unicodedata

MAX_TITLE_AFFIX_LENGTH = 40
DEFAULT_TITLE_PREFIX = "PAR - "
DEFAULT_TITLE_SUFFIX = "(ad free)"


def validate_title_affix(value: str, label: str, enabled: bool) -> str:
    # Preserve intentional spacing (including the default prefix's final space).
    if len(value) > MAX_TITLE_AFFIX_LENGTH:
        raise ValueError(f"{label} must be {MAX_TITLE_AFFIX_LENGTH} characters or fewer")
    if any(unicodedata.category(char).startswith("C") or char in "<>\r\n\t\u2028\u2029" for char in value):
        raise ValueError(f"{label} must be a single line of plain text without markup or control characters")
    if enabled and not value.strip():
        raise ValueError(f"{label} is required when enabled")
    return value


def podcast_feed_title(title: str, values: dict) -> str:
    prefix = values.get("podcast_title_prefix", DEFAULT_TITLE_PREFIX)
    suffix = values.get("podcast_title_suffix", DEFAULT_TITLE_SUFFIX)
    if values.get("podcast_title_prefix_enabled", False) and prefix:
        title = prefix + ("" if prefix[-1].isspace() else " ") + title
    if values.get("podcast_title_suffix_enabled", True) and suffix:
        title += ("" if suffix[0].isspace() else " ") + suffix
    return title
