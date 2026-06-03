"""Tests for tag ROI enhancement (prep only)."""

from app.config import Settings
from app.pipeline.preprocess import preprocess_images
from app.pipeline.tag_enhance import TagZoomPanels, build_tag_zoom_panels, extract_tag_roi
from tests.conftest import make_test_image


def test_extract_tag_roi_non_empty():
    raw = make_test_image((200, 180, 160), size=(600, 400))
    from PIL import Image
    import io

    pil = Image.open(io.BytesIO(raw))
    roi = extract_tag_roi(pil)
    assert roi.width > 0 and roi.height > 0


def test_build_tag_zoom_panels_three_outputs():
    raw = make_test_image((100, 100, 100), size=(800, 600))
    files = [("tag.jpg", "image/jpeg", raw)]
    processed, _ = preprocess_images(files, ["Tag"], Settings())
    panels = build_tag_zoom_panels(processed[0], Settings())
    assert isinstance(panels, TagZoomPanels)
    assert panels.raw.size[0] > 0
    assert panels.enhanced.size[0] > 0
    assert panels.upright.size[0] > 0


def test_upright_panel_wider_than_tall_for_portrait_side():
    """Portrait side crop should become landscape after upright rotate."""
    raw = make_test_image((120, 120, 120), size=(400, 800))
    files = [("left.jpg", "image/jpeg", raw)]
    processed, _ = preprocess_images(files, ["Left"], Settings())
    panels = build_tag_zoom_panels(processed[0], Settings())
    assert panels.upright.width >= panels.upright.height
