"""High-Resolution Analysis Composite (HRC) — collage + TAG ZOOM for single Gemini pass."""

from dataclasses import dataclass

from PIL import Image

from app.config import Settings
from app.models.pipeline import ProcessedImage
from app.pipeline.collage_composer import compose_collage_with_tag_zoom
from app.pipeline.image_utils import image_to_bytes
from app.pipeline.tag_view_selector import TagDetectionMethod, select_tag_candidates


@dataclass
class AnalysisCompositeResult:
    image: Image.Image
    tag_labels: list[str]
    tag_detection_method: TagDetectionMethod
    cell_size: int
    jpeg_bytes: int
    tag_zoom_row_px: int = 0
    composite_layout: str = "collage_with_tag_zoom"
    tag_zoom_source_label: str | None = None


def compose_analysis_composite(
    images: list[ProcessedImage],
    settings: Settings,
    tag_image_index: int | None = None,
) -> AnalysisCompositeResult:
    tag_candidates, tag_method, tag_labels = select_tag_candidates(
        images, tag_image_index=tag_image_index, max_candidates=3
    )
    composite, params, _, source_label = compose_collage_with_tag_zoom(
        images, settings, tag_image_index=tag_image_index
    )
    jpeg_bytes = len(image_to_bytes(composite, fmt="JPEG"))

    return AnalysisCompositeResult(
        image=composite,
        tag_labels=tag_labels,
        tag_detection_method=tag_method,
        cell_size=params.base_cell_px,
        jpeg_bytes=jpeg_bytes,
        tag_zoom_row_px=params.tag_zoom_row_px,
        composite_layout="collage_with_tag_zoom",
        tag_zoom_source_label=source_label,
    )
