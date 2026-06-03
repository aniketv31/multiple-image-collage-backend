"""Internal pipeline data structures."""

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class ProcessedImage:
    index: int
    label: str
    original_bytes: bytes
    pil_image: Image.Image
    blur_score: float
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class UnifiedViewResult:
    image: Image.Image
    method: str
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    unified_view: UnifiedViewResult
    image_quality_score: float
    quality_warnings: list[str]
