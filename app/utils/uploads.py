"""Shared multipart image upload parsing."""

import base64
import io
import re
from typing import BinaryIO

from fastapi import HTTPException, UploadFile, status

from app.config import Settings

_DATA_URL_PREFIX = re.compile(
    r"^data:(image/(?:jpeg|jpg|png|webp));base64,",
    re.IGNORECASE,
)


def validate_mime(content_type: str | None, settings: Settings) -> str:
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image type: {mime or 'unknown'}. "
                f"Allowed: {', '.join(settings.allowed_mime_types)}"
            ),
        )
    return mime


def sniff_image_mime(raw: bytes) -> str | None:
    if len(raw) >= 3 and raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _finalize_image_bytes(
    raw: bytes,
    filename: str,
    content_type: str | None,
    settings: Settings,
    *,
    enforce_size_limit: bool = True,
) -> tuple[BinaryIO, str, bytes]:
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image data is empty",
        )
    if enforce_size_limit and len(raw) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image exceeds {settings.max_image_size_mb}MB limit",
        )

    mime = (content_type or "").split(";")[0].strip().lower() or None
    if mime == "image/jpg":
        mime = "image/jpeg"
    if not mime:
        mime = sniff_image_mime(raw)
    if not mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not detect image type; use JPEG, PNG, or WebP",
        )
    validate_mime(mime, settings)
    return io.BytesIO(raw), filename, raw


def parse_base64_image(
    image_base64: str,
    settings: Settings,
) -> tuple[BinaryIO, str, bytes]:
    """Decode base64 or data-URL image payload."""
    payload = image_base64.strip()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_base64 is empty",
        )

    content_type: str | None = None
    match = _DATA_URL_PREFIX.match(payload)
    if match:
        content_type = match.group(1).lower()
        if content_type == "image/jpg":
            content_type = "image/jpeg"
        payload = payload[match.end() :]

    payload = "".join(payload.split())
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data",
        ) from exc

    ext = "jpg"
    if content_type == "image/png":
        ext = "png"
    elif content_type == "image/webp":
        ext = "webp"
    return _finalize_image_bytes(
        raw,
        f"image.{ext}",
        content_type,
        settings,
        enforce_size_limit=False,
    )


async def parse_uploaded_image(
    upload: UploadFile,
    settings: Settings,
) -> tuple[BinaryIO, str, bytes]:
    """Parse exactly one image file upload."""
    validate_mime(upload.content_type, settings)
    raw = await upload.read()
    return _finalize_image_bytes(
        raw,
        upload.filename or "image.jpg",
        upload.content_type,
        settings,
    )


async def resolve_image_input(
    image: UploadFile | None,
    image_base64: str | None,
    settings: Settings,
) -> tuple[BinaryIO, str, bytes]:
    """Accept either a file upload or base64 string (not both)."""
    has_b64 = bool(image_base64 and image_base64.strip())

    file_bytes: bytes | None = None
    if image is not None:
        file_bytes = await image.read()

    has_file = bool(file_bytes)

    if has_file and has_b64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either image (file) or image_base64, not both",
        )
    if not has_file and not has_b64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image (file) or image_base64 is required",
        )

    if has_b64:
        return parse_base64_image(image_base64 or "", settings)

    assert file_bytes is not None and image is not None
    return _finalize_image_bytes(
        file_bytes,
        image.filename or "image.jpg",
        image.content_type,
        settings,
    )


SINGLE_IMAGE_OPENAPI = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "format": "binary",
                            "description": (
                                "Single asset photo (JPEG, PNG, or WebP). "
                                "Provide this OR image_base64."
                            ),
                        },
                        "image_base64": {
                            "type": "string",
                            "description": (
                                "Base64-encoded image bytes or data URL "
                                "(e.g. data:image/jpeg;base64,...). No length limit for now. "
                                "Provide this OR image."
                            ),
                        },
                        "locale": {
                            "type": "string",
                            "default": "en",
                            "description": "Output language for Gemini response",
                        },
                    },
                }
            }
        }
    }
}
