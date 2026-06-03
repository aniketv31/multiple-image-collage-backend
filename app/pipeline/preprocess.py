"""Image preprocessing: EXIF fix, resize, format validation."""

import io
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import Settings
from app.models.pipeline import ProcessedImage
from app.pipeline.quality import score_blur, score_image_quality


def _default_label(index: int, angle: str | None) -> str:
    if angle:
        return angle.strip().title()
    return f"View {index + 1}"


def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resize_pil(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def preprocess_images(
    files: list[tuple[BinaryIO, str, bytes]],
    angles: list[str] | None,
    settings: Settings,
) -> tuple[list[ProcessedImage], list[str]]:
    """Load and preprocess uploaded images."""
    warnings: list[str] = []
    processed: list[ProcessedImage] = []

    for idx, (_file_obj, filename, raw_bytes) in enumerate(files):
        angle = angles[idx] if angles and idx < len(angles) else None
        label = _default_label(idx, angle)

        try:
            pil = Image.open(io.BytesIO(raw_bytes))
            pil = ImageOps.exif_transpose(pil)
            pil = pil.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not decode image '{filename}': {exc}") from exc

        pil_stitch = _resize_pil(pil, settings.max_stitch_edge_px)
        cv_image = _pil_to_cv2(pil_stitch)
        blur = score_blur(cv_image)

        img_warnings: list[str] = []
        if blur < 50:
            img_warnings.append(f"image_{idx + 1}_blurry")
            warnings.append(f"image_{idx + 1}_blurry")

        processed.append(
            ProcessedImage(
                index=idx,
                label=label,
                original_bytes=raw_bytes,
                pil_image=pil,
                cv_image=cv_image,
                blur_score=blur,
                quality_warnings=img_warnings,
            )
        )

    if len(processed) >= 2:
        exposures = [float(np.mean(img.cv_image)) for img in processed]
        if max(exposures) - min(exposures) > 60:
            warnings.append("lighting_inconsistent")

    return processed, warnings


def compute_quality_score(images: list[ProcessedImage]) -> float:
    blur_scores = [score_blur(img.cv_image) for img in images]
    return score_image_quality(blur_scores)
