"""Image preprocessing: EXIF fix, resize, format validation."""

import io
from typing import BinaryIO

from PIL import Image, ImageOps

from app.config import Settings
from app.models.pipeline import ProcessedImage
from app.pipeline.quality import score_blur


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
    index: int = 0,
) -> ProcessedImage:
    """Load and preprocess one uploaded image (EXIF fix, RGB, resize)."""
    _file_obj, filename, raw_bytes = file
    try:
        pil = Image.open(io.BytesIO(raw_bytes))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image '{filename}': {exc}") from exc

    pil = _resize_pil(pil, settings.max_preprocess_edge_px)
    blur = score_blur(pil)

    warnings: list[str] = []
    if blur < 50:
        warnings.append(f"image_{index + 1}_blurry")

    return ProcessedImage(
        index=index,
        label=f"Image {index + 1}",
        original_bytes=raw_bytes,
        pil_image=pil,
        blur_score=blur,
        quality_warnings=warnings,
    )


def preprocess_images(
    files: list[tuple[BinaryIO, str, bytes]],
    settings: Settings,
) -> list[ProcessedImage]:
    """Preprocess a list of uploaded images."""
    return [preprocess_single_image(f, settings, idx) for idx, f in enumerate(files)]
