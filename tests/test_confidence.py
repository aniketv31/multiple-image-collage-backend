"""Tests for confidence aggregation."""

from app.config import Settings
from app.models.pipeline import PipelineResult, UnifiedViewResult
from app.models.responses import GeminiExtractionResult
from app.utils.confidence import aggregate_confidence
from PIL import Image


def _pipeline(stitch_conf: float = 0.0) -> PipelineResult:
    img = Image.new("RGB", (800, 600), (128, 128, 128))
    return PipelineResult(
        unified_view=UnifiedViewResult(
            image=img,
            method="labeled_grid",
            stitching_confidence=stitch_conf,
        ),
        image_quality_score=0.8,
        quality_warnings=[],
    )


def test_aggregate_confidence_review_flag():
    settings = Settings(review_confidence_threshold=0.65, field_confidence_threshold=0.5)
    gemini = GeminiExtractionResult(
        detectedAsset="Compressor",
        confidence_asset_name=0.3,
        confidence_asset_condition=0.3,
        confidence_asset_description=0.3,
        confidence_asset_tag_number=0.3,
    )
    scores, review = aggregate_confidence(gemini, _pipeline(), settings)
    assert review is True
    assert scores.overall < 0.65


def test_aggregate_confidence_high_scores():
    settings = Settings()
    gemini = GeminiExtractionResult(
        detectedAsset="Compressor",
        confidence_asset_name=0.9,
        confidence_asset_condition=0.85,
        confidence_asset_description=0.88,
        confidence_asset_tag_number=0.92,
    )
    scores, review = aggregate_confidence(gemini, _pipeline(0.7), settings)
    assert review is False
    assert scores.overall >= 0.65
