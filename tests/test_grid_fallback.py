"""Tests for labeled grid composer."""

import io

from app.config import Settings
from app.pipeline.grid_composer import compose_labeled_grid
from app.pipeline.preprocess import preprocess_images
from tests.conftest import make_test_image


def test_grid_compose_dimensions():
    settings = Settings()
    raw_images = [make_test_image((i * 40, 80, 120)) for i in range(4)]
    files = [(io.BytesIO(r), f"img_{i}.jpg", r) for i, r in enumerate(raw_images)]
    processed, _ = preprocess_images(files, ["Front", "Back", "Left", "Right"], settings)

    grid = compose_labeled_grid(processed, settings)
    assert grid.width > 0
    assert grid.height > 0
    assert grid.width <= settings.max_output_width_px


def test_grid_compose_respects_max_width():
    settings = Settings(max_output_width_px=800)
    raw_images = [make_test_image((100, 100, 100), size=(1200, 900)) for _ in range(6)]
    files = [(io.BytesIO(r), f"img_{i}.jpg", r) for i, r in enumerate(raw_images)]
    processed, _ = preprocess_images(files, None, settings)

    grid = compose_labeled_grid(processed, settings)
    assert grid.width <= 800
