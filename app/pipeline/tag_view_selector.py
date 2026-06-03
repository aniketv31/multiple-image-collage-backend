"""Select which images are best candidates for tag/barcode OCR."""

from enum import Enum

import cv2
import numpy as np

from app.models.pipeline import ProcessedImage

TAG_ANGLE_KEYWORDS = frozenset({"tag", "barcode", "label", "sticker", "serial"})


class TagDetectionMethod(str, Enum):
    EXPLICIT_INDEX = "explicit_index"
    ANGLE_HINT = "angle_hint"
    AUTO_DETECT = "auto_detect"


def _label_suggests_tag(label: str) -> bool:
    return label.strip().lower() in TAG_ANGLE_KEYWORDS or any(
        kw in label.strip().lower() for kw in TAG_ANGLE_KEYWORDS
    )


def _edge_density_score(cv_image: np.ndarray) -> float:
    """Higher score suggests barcode-like high-frequency patterns."""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin_y, margin_x = h // 4, w // 4
    center = gray[margin_y : h - margin_y, margin_x : w - margin_x]
    edges = cv2.Canny(center, 50, 150)
    return float(np.mean(edges > 0))


def _location_prior(label: str) -> float:
    lower = label.strip().lower()
    if lower in ("back", "bottom", "tag", "barcode", "label"):
        return 0.3
    if lower in ("rear", "underside", "serial"):
        return 0.25
    if lower in ("left", "right", "side"):
        return 0.28
    if "left" in lower or "right" in lower or "side" in lower:
        return 0.22
    return 0.0


def select_tag_candidates(
    images: list[ProcessedImage],
    tag_image_index: int | None = None,
    max_candidates: int = 3,
) -> tuple[list[ProcessedImage], TagDetectionMethod, list[str]]:
    """
    Priority: explicit index > angle hint > auto-detect heuristic.
    Returns (candidates, method, labels).
    """
    if tag_image_index is not None:
        if 0 <= tag_image_index < len(images):
            img = images[tag_image_index]
            return [img], TagDetectionMethod.EXPLICIT_INDEX, [img.label]
        return [], TagDetectionMethod.EXPLICIT_INDEX, []

    angle_matches = [img for img in images if _label_suggests_tag(img.label)]
    if angle_matches:
        selected = angle_matches[:max_candidates]
        return selected, TagDetectionMethod.ANGLE_HINT, [img.label for img in selected]

    scored = []
    for img in images:
        score = _edge_density_score(img.cv_image) + _location_prior(img.label)
        scored.append((score, img))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [img for _, img in scored[:max_candidates]]
    if not top:
        top = images[:1]

    return top, TagDetectionMethod.AUTO_DETECT, [img.label for img in top]
