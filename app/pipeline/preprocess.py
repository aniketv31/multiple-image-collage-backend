"""Image preprocessing: EXIF fix, resize, format validation."""

import io
from typing import BinaryIO

from PIL import Image, ImageOps

from app.config import Settings
from app.models.pipeline import ProcessedImage
from app.pipeline.quality import score_blur, score_image_quality


def _resize_pil(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def preprocess_single_image(
    file: tuple[BinaryIO, str, bytes],
    settings: Settings,
) -> tuple[ProcessedImage, list[str]]:
    """Load and preprocess one uploaded image for Gemini."""
    _file_obj, filename, raw_bytes = file
    warnings: list[str] = []

    try:
        pil = Image.open(io.BytesIO(raw_bytes))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image '{filename}': {exc}") from exc

    pil = _resize_pil(pil, settings.max_preprocess_edge_px)
    blur = score_blur(pil)

    if blur < 50:
        warnings.append("image_blurry")

    processed = ProcessedImage(
        index=0,
        label="Photo",
        original_bytes=raw_bytes,
        pil_image=pil,
        blur_score=blur,
        quality_warnings=list(warnings),
    )
    return processed, warnings


def compute_quality_score(image: ProcessedImage) -> float:
    return score_image_quality(image.blur_score)
