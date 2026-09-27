"""Bounded, raster-only unified-feed artwork stored by content hash."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import tempfile

from PIL import Image, UnidentifiedImageError
from app.core.config import settings

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def save_artwork(payload: bytes) -> str:
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a PNG, JPEG or WebP image under 5 MB")
    try:
        with Image.open(BytesIO(payload)) as original:
            if original.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Artwork must be PNG, JPEG or WebP")
            if not (64 <= original.width <= 4096 and 64 <= original.height <= 4096):
                raise ValueError("Artwork dimensions must be between 64 and 4096 pixels")
            original.load()
            image = original.convert("RGB")
            image.thumbnail((1600, 1600))
            output = BytesIO()
            image.save(output, format="JPEG", quality=90)
            data = output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not decode artwork") from exc
    name = sha256(data).hexdigest() + ".jpg"
    directory = Path(settings.ARTWORK_DIR) / "unified"
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(temporary, directory / name)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return name
