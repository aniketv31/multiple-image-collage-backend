"""Tests for collage composer and angle groups."""

from app.config import Settings
from app.pipeline.angle_groups import (
    CollageSlot,
    assign_collage_slots,
    is_stitch_eligible,
)
from app.pipeline.collage_composer import compose_collage_for_panorama
from app.pipeline.preprocess import preprocess_images
from tests.conftest import make_test_image


def _make_labeled_processed():
    files = []
    angles = ["Front", "Front", "Left", "Tag"]
    for i, angle in enumerate(angles):
        raw = make_test_image((50 + i * 30, 80, 120), size=(900, 700))
        files.append((f"{angle.lower()}.jpg", "image/jpeg", raw))
    processed, _ = preprocess_images(files, angles, Settings())
    return processed


def test_side_and_tag_not_stitch_eligible():
    assert is_stitch_eligible("Front") is True
    assert is_stitch_eligible("Left") is False
    assert is_stitch_eligible("Tag") is False


def test_assign_collage_slots_tag_index():
    processed = _make_labeled_processed()
    assignments = assign_collage_slots(processed, tag_image_index=3)
    tag_slots = [a for a in assignments if a.slot == CollageSlot.TAG]
    assert len(tag_slots) == 1
    assert tag_slots[0].image.index == 3


def test_panorama_collage_under_byte_budget():
    processed = _make_labeled_processed()
    settings = Settings()
    image = compose_collage_for_panorama(processed, settings)
    from app.pipeline.collage_composer import _estimate_jpeg_bytes

    assert _estimate_jpeg_bytes(image) <= settings.max_gemini_composite_bytes
    assert image.width > 200
