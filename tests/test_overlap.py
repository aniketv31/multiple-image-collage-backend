"""Tests for overlap graph and connected components."""

import io

import pytest
from PIL import Image

from app.config import Settings
from app.pipeline.overlap import build_overlap_graph, find_connected_components
from app.pipeline.preprocess import preprocess_images


def _make_files(image_bytes_list: list[bytes]) -> list[tuple]:
    files = []
    for i, raw in enumerate(image_bytes_list):
        files.append((io.BytesIO(raw), f"img_{i}.jpg", raw))
    return files


def test_overlap_graph_empty_for_unrelated_images(multi_color_images):
    settings = Settings()
    files = _make_files(multi_color_images)
    processed, _ = preprocess_images(files, None, settings)
    edges = build_overlap_graph(processed)
    assert len(edges) == 0


def test_connected_components_single_nodes(multi_color_images):
    settings = Settings()
    files = _make_files(multi_color_images)
    processed, _ = preprocess_images(files, None, settings)
    clusters = find_connected_components(len(processed), [])
    assert len(clusters) == len(processed)


def test_preprocess_rejects_invalid():
    settings = Settings()
    with pytest.raises(ValueError):
        preprocess_images([(io.BytesIO(b"not-an-image"), "bad.jpg", b"not-an-image")], None, settings)
