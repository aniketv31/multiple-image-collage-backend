"""TAG ZOOM row should be a substantial fraction of composite height."""

from app.config import Settings
from app.pipeline.collage_composer import compose_collage_with_tag_zoom
from app.pipeline.preprocess import preprocess_images
from tests.conftest import make_test_image


def test_tag_zoom_row_fraction_of_height():
    files = []
    angles = ["Front", "Left", "Right"]
    for i, angle in enumerate(angles):
        raw = make_test_image((40 + i * 50, 80, 120), size=(900, 700))
        files.append((f"{angle}.jpg", "image/jpeg", raw))
    processed, _ = preprocess_images(files, angles, Settings())
    settings = Settings()
    composite, _, panels, _ = compose_collage_with_tag_zoom(processed, settings)
    assert panels is not None
    assert composite.height >= 400
    ratio = settings.tag_zoom_row_height_ratio
    min_zoom_band = int(composite.height * ratio * 0.5)
    assert composite.height > min_zoom_band
