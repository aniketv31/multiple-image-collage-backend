"""Tests for tag/barcode candidate image selection."""

import io

import numpy as np
from PIL import Image

from app.models.pipeline import ProcessedImage
from app.pipeline.tag_view_selector import TagDetectionMethod, select_tag_candidates


def _processed(index: int, label: str) -> ProcessedImage:
    pil = Image.new("RGB", (200, 200), color=(index * 40, 80, 120))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    raw = buf.getvalue()
    cv = np.array(pil.convert("RGB"))[:, :, ::-1].copy()
    return ProcessedImage(
        index=index,
        label=label,
        original_bytes=raw,
        pil_image=pil,
        cv_image=cv,
        blur_score=100.0,
    )


def test_explicit_index_selects_one_image():
    images = [_processed(0, "Front"), _processed(1, "Back"), _processed(2, "Tag")]
    candidates, method, labels = select_tag_candidates(images, tag_image_index=1)
    assert method == TagDetectionMethod.EXPLICIT_INDEX
    assert len(candidates) == 1
    assert candidates[0].label == "Back"
    assert labels == ["Back"]


def test_angle_hint_prefers_tag_label():
    images = [_processed(0, "Front"), _processed(1, "Barcode"), _processed(2, "Left")]
    candidates, method, labels = select_tag_candidates(images)
    assert method == TagDetectionMethod.ANGLE_HINT
    assert candidates[0].label == "Barcode"
    assert "Barcode" in labels


def test_auto_detect_returns_up_to_three():
    images = [_processed(i, f"View{i}") for i in range(5)]
    candidates, method, labels = select_tag_candidates(images)
    assert method == TagDetectionMethod.AUTO_DETECT
    assert 1 <= len(candidates) <= 3
    assert len(labels) == len(candidates)
