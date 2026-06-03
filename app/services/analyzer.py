"""End-to-end asset analysis pipeline orchestration."""

import time
import uuid

import structlog

from app.config import Settings
from app.models.pipeline import PipelineResult, UnifiedViewResult
from app.models.responses import (
    AnalyzeResponse,
    AnalyzeUnifiedView,
    PanoramaResponse,
    UnifiedViewInfo,
    UnifiedViewMethod,
    ValidationFields,
)
from app.pipeline.analysis_composite import compose_analysis_composite
from app.pipeline.image_utils import hash_image_set, image_to_base64
from app.pipeline.preprocess import compute_quality_score, preprocess_images
from app.pipeline.unified_view import build_unified_view
from app.services.field_merger import composite_to_gemini_result, to_asset_fields
from app.services.gemini import GeminiService, _normalize_barcode_position
from app.utils.confidence import aggregate_confidence

logger = structlog.get_logger()


class AssetAnalysisService:
    def __init__(
        self,
        settings: Settings,
        gemini: GeminiService,
    ):
        self.settings = settings
        self.gemini = gemini
        self._cache: dict[str, AnalyzeResponse] = {}

    async def create_panorama(
        self,
        files: list[tuple],
        angles: list[str] | None = None,
        include_image_base64: bool = True,
        layout: str | None = None,
    ) -> PanoramaResponse:
        """Build unified panorama/grid image only (no Gemini call)."""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        processed, quality_warnings = preprocess_images(files, angles, self.settings)
        unified = build_unified_view(
            processed, self.settings, quality_warnings, layout=layout
        )

        image_base64 = image_to_base64(unified.image) if include_image_base64 else None

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        method = UnifiedViewMethod(unified.method)

        logger.info(
            "panorama_created",
            request_id=request_id,
            method=unified.method,
            image_count=len(processed),
            elapsed_ms=elapsed_ms,
        )

        return PanoramaResponse(
            request_id=request_id,
            status="success",
            processing_time_ms=elapsed_ms,
            unified_view=UnifiedViewInfo(
                method=method,
                image_url=None,
                image_base64=image_base64,
                width=unified.image.width,
                height=unified.image.height,
                stitching_confidence=unified.stitching_confidence,
            ),
            quality_warnings=list(set(quality_warnings + unified.quality_warnings)),
            image_count=len(processed),
        )

    async def analyze(
        self,
        files: list[tuple],
        angles: list[str] | None = None,
        locale: str = "en",
        use_cache: bool = True,
        tag_image_index: int | None = None,
        user_asset_name: str | None = None,
        user_description: str | None = None,
    ) -> AnalyzeResponse:
        request_id = str(uuid.uuid4())
        raw_bytes_list = [raw for _, _, raw in files]

        if use_cache and not user_asset_name:
            cache_key = hash_image_set(raw_bytes_list)
            cached = self._cache.get(cache_key)
            if cached:
                return cached.model_copy(update={"request_id": request_id})

        start = time.perf_counter()

        processed, quality_warnings = preprocess_images(files, angles, self.settings)
        quality_score = compute_quality_score(processed)

        composite = compose_analysis_composite(
            processed,
            self.settings,
            tag_image_index=tag_image_index,
        )

        unified = UnifiedViewResult(
            image=composite.image,
            method=UnifiedViewMethod.ANALYSIS_COMPOSITE.value,
            stitching_confidence=1.0,
        )

        pipeline_result = PipelineResult(
            unified_view=unified,
            image_quality_score=quality_score,
            quality_warnings=quality_warnings,
        )

        composite_result = await self.gemini.extract_from_composite(
            composite.image,
            locale=locale,
            user_asset_name=user_asset_name,
            user_description=user_description,
        )

        raw_tag_number = composite_result.detectedtagnumber
        gemini_result = composite_to_gemini_result(
            composite_result, self.settings
        )

        if gemini_result.detectedtagnumber and str(gemini_result.detectedtagnumber).upper() == "UNREADABLE":
            gemini_result.imageReadability = "E"
            gemini_result.confidence_asset_tag_number = min(
                gemini_result.confidence_asset_tag_number, 0.2
            )

        _, review_required = aggregate_confidence(
            gemini_result, pipeline_result, self.settings
        )

        validation: ValidationFields | None = None
        if user_asset_name and composite_result.namedescriptionmatch is not None:
            validation = ValidationFields(
                namedescriptionmatch=composite_result.namedescriptionmatch,
                namedescriptionmatchpercent=composite_result.namedescriptionmatchpercent,
                reasoning=composite_result.reasoning,
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        response = AnalyzeResponse(
            request_id=request_id,
            status="success",
            processing_time_ms=elapsed_ms,
            unified_view=AnalyzeUnifiedView(
                method=UnifiedViewMethod.ANALYSIS_COMPOSITE,
                width=composite.image.width,
                height=composite.image.height,
                stitching_confidence=1.0,
            ),
            asset=to_asset_fields(gemini_result, self.settings),
            tag_detection_reasoning=gemini_result.tag_detection_reasoning,
            barcodeposition=_normalize_barcode_position(gemini_result.barcodeposition),
            image_readability=gemini_result.imageReadability,
            detected_tag_number_raw=raw_tag_number,
            tag_zoom_source_label=composite.tag_zoom_source_label,
            validation=validation,
            visible_labels=list(composite_result.visible_labels),
        )

        if use_cache and not user_asset_name:
            cache_key = hash_image_set(raw_bytes_list)
            self._cache[cache_key] = response

        logger.info(
            "analysis_complete",
            request_id=request_id,
            method=UnifiedViewMethod.ANALYSIS_COMPOSITE.value,
            review_required=review_required,
            tag_detection_method=composite.tag_detection_method.value,
            gemini_calls=1,
            composite_layout=composite.composite_layout,
            jpeg_bytes=composite.jpeg_bytes,
            tag_zoom_row_px=composite.tag_zoom_row_px,
            tag_zoom_source_label=composite.tag_zoom_source_label,
            detectedtagnumber_raw=raw_tag_number,
            detectedtagnumber_normalized=gemini_result.detectedtagnumber,
            elapsed_ms=elapsed_ms,
        )
        return response
