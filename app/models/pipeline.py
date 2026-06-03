"""Internal pipeline data structures."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class ProcessedImage:
    index: int
    label: str
    original_bytes: bytes
    pil_image: Image.Image
    cv_image: np.ndarray
    blur_score: float
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class OverlapEdge:
    i: int
    j: int
    inlier_count: int
    inlier_ratio: float


@dataclass
class StitchClusterResult:
    indices: list[int]
    stitched: np.ndarray | None
    confidence: float
    success: bool


@dataclass
class UnifiedViewResult:
    image: Image.Image
    method: str
    stitching_confidence: float
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    unified_view: UnifiedViewResult
    image_quality_score: float
    quality_warnings: list[str]
