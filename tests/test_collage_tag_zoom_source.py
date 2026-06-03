"""Tests for TAG ZOOM sourcing from Left/side slot."""

from app.config import Settings
from app.pipeline.angle_groups import assign_collage_slots
from app.pipeline.collage_composer import resolve_tag_image_for_zoom
from app.pipeline.preprocess import preprocess_images
from tests.conftest import make_test_image


def _make_labeled_processed():
    files = []
    angles = ["Front", "Left", "Right"]
    for i, angle in enumerate(angles):
        color = (200, 50, 50) if angle == "Front" else (50, 200, 50) if angle == "Left" else (50, 50, 200)
        raw = make_test_image(color, size=(900, 700))
        files.append((f"{angle.lower()}.jpg", "image/jpeg", raw))
    processed, _ = preprocess_images(files, angles, Settings())
    return processed


def test_resolve_tag_image_prefers_left_without_tag_label():
    processed = _make_labeled_processed()
    assignments = assign_collage_slots(processed)
    tag_img = resolve_tag_image_for_zoom(processed, assignments, None)
    assert tag_img is not None
    assert tag_img.label == "Left"


def test_resolve_tag_image_explicit_index():
    processed = _make_labeled_processed()
    assignments = assign_collage_slots(processed)
    tag_img = resolve_tag_image_for_zoom(processed, assignments, tag_image_index=2)
    assert tag_img.label == "Right"


def test_compose_includes_tag_zoom_source_label():
    from app.pipeline.collage_composer import compose_collage_with_tag_zoom

    processed = _make_labeled_processed()
    _, _, _, source = compose_collage_with_tag_zoom(processed, Settings())
    assert source == "Left"
