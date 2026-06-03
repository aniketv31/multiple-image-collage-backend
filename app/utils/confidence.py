"""Weighted confidence score aggregation."""

from app.config import Settings
from app.models.pipeline import PipelineResult
from app.models.responses import ConfidenceScores, GeminiExtractionResult


def aggregate_confidence(
    gemini: GeminiExtractionResult,
    pipeline: PipelineResult,
    settings: Settings,
) -> tuple[ConfidenceScores, bool]:
    gemini_scores = {
        "asset_name": gemini.confidence_asset_name,
        "asset_condition": gemini.confidence_asset_condition,
        "asset_description": gemini.confidence_asset_description,
        "asset_tag_number": gemini.confidence_asset_tag_number,
    }
    gemini_mean = sum(gemini_scores.values()) / len(gemini_scores)

    stitching = pipeline.unified_view.stitching_confidence

    overall = (
        0.45 * gemini_mean
        + 0.25 * pipeline.image_quality_score
        + 0.30 * stitching
    )
    overall = max(0.0, min(1.0, overall))

    scores = ConfidenceScores(
        overall=round(overall, 3),
        asset_name=round(gemini_scores["asset_name"], 3),
        asset_condition=round(gemini_scores["asset_condition"], 3),
        asset_description=round(gemini_scores["asset_description"], 3),
        asset_tag_number=round(gemini_scores["asset_tag_number"], 3),
    )

    review_required = overall < settings.review_confidence_threshold or any(
        v < settings.field_confidence_threshold for v in gemini_scores.values()
    )
    return scores, review_required
