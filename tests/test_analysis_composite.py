"""Tests for analysis composite builder (collage + TAG ZOOM)."""

import pytest

from app.config import Settings
from app.pipeline.analysis_composite import compose_analysis_composite
from app.pipeline.preprocess import preprocess_images
from tests.conftest import make_test_image


def _make_processed(count: int, size: tuple[int, int] = (800, 600)):
    files = []
    angles = []
    labels = ["Front", "Left", "Tag", "Right", "Back", "View 6", "View 7", "View 8", "View 9", "View 10"]
    for i in range(count):
        raw = make_test_image((i * 20, 40, 80), size=size)
        files.append((f"img{i}.jpg", "image/jpeg", raw))
        angles.append(labels[i] if i < len(labels) else f"View {i + 1}")
    processed, _ = preprocess_images(files, angles, Settings())
    return processed


@pytest.mark.parametrize("count", [7, 8, 10])
def test_composite_size_budget(count):
    processed = _make_processed(count)
    settings = Settings()
    result = compose_analysis_composite(processed, settings)
    assert result.jpeg_bytes <= settings.max_gemini_composite_bytes
    assert result.image.width <= settings.max_composite_width_px
    assert result.image.height > 0
    assert result.cell_size >= settings.analysis_min_cell_px
    assert result.composite_layout == "collage_with_tag_zoom"


def test_tag_image_index_drives_zoom_row():
    processed = _make_processed(3)
    settings = Settings()
    result = compose_analysis_composite(processed, settings, tag_image_index=1)
    assert result.tag_labels == ["Left"]
    assert result.tag_detection_method.value == "explicit_index"
    assert result.tag_zoom_row_px >= settings.tag_zoom_row_min_px


def test_collage_taller_with_tag_zoom_than_grid_only():
    processed = _make_processed(4)
    settings = Settings()
    result = compose_analysis_composite(processed, settings, tag_image_index=2)
    assert result.image.height > 400
