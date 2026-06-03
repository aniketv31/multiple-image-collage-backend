"""Shared multipart image upload parsing."""

import io
from typing import BinaryIO

from fastapi import HTTPException, UploadFile, status

from app.config import Settings


def parse_angle_labels(angles: str | None) -> list[str] | None:
    if not angles:
        return None
    return [label.strip() for label in angles.split(",") if label.strip()]


def validate_mime(content_type: str | None, settings: Settings) -> str:
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image type: {mime or 'unknown'}. "
                f"Allowed: {', '.join(settings.allowed_mime_types)}"
            ),
        )
    return mime


async def parse_uploaded_images(
    images: list[UploadFile],
    settings: Settings,
) -> list[tuple[BinaryIO, str, bytes]]:
    if len(images) < settings.min_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least {settings.min_images} images are required",
        )
    if len(images) > settings.max_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {settings.max_images} images allowed",
        )

    parsed: list[tuple[BinaryIO, str, bytes]] = []
    for idx, upload in enumerate(images):
        validate_mime(upload.content_type, settings)
        raw = await upload.read()
        if len(raw) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image {idx + 1} exceeds {settings.max_image_size_mb}MB limit",
            )
        if not raw:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image {idx + 1} is empty",
            )
        parsed.append(
            (io.BytesIO(raw), upload.filename or f"image_{idx + 1}.jpg", raw)
        )
    return parsed


MULTI_IMAGE_OPENAPI = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["images"],
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": (
                                "Upload 2-10 images. In Swagger UI click **Add item** "
                                "for each image, or select multiple files if your browser supports it."
                            ),
                        },
                        "angles": {
                            "type": "string",
                            "description": "Optional comma-separated labels, e.g. Front,Back,Left,Right",
                            "example": "Front,Back,Left,Right,Top",
                        },
                    },
                }
            }
        }
    }
}
