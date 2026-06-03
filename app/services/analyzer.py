"""End-to-end asset analysis pipeline orchestration."""

import time
import uuid

import structlog

from app.config import Settings
from app.models.pipeline import PipelineResult, UnifiedViewResult
from app.models.responses import (
    AnalyzeResponse,
    AnalyzeUnifiedView,
    UnifiedViewMethod,
)
from app.pipeline.image_utils import hash_image_set
from app.pipeline.preprocess import compute_quality_score, preprocess_single_image
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

    async def analyze(
        self,
        file: tuple,
        locale: str = "en",
        use_cache: bool = True,
    ) -> AnalyzeResponse:
        request_id = str(uuid.uuid4())
        _file_obj, _filename, raw_bytes = file

        if use_cache:
            cache_key = hash_image_set([raw_bytes])
            cached = self._cache.get(cache_key)
            if cached:
                return cached.model_copy(update={"request_id": request_id})

        start = time.perf_counter()

        processed, quality_warnings = preprocess_single_image(file, self.settings)
        quality_score = compute_quality_score(processed)

        pil_for_gemini = processed.pil_image

        unified = UnifiedViewResult(
            image=pil_for_gemini,
            method=UnifiedViewMethod.DIRECT_IMAGE.value,
            quality_warnings=quality_warnings,
        )

        pipeline_result = PipelineResult(
            unified_view=unified,
            image_quality_score=quality_score,
            quality_warnings=quality_warnings,
        )

        gemini_result_raw = await self.gemini.extract_from_image(
            pil_for_gemini,
            locale=locale,
        )

        raw_tag_number = gemini_result_raw.detectedtagnumber
        gemini_result = composite_to_gemini_result(
            gemini_result_raw, self.settings
        )

        if gemini_result.detectedtagnumber and str(gemini_result.detectedtagnumber).upper() == "UNREADABLE":
            gemini_result.imageReadability = "E"
            gemini_result.confidence_asset_tag_number = min(
                gemini_result.confidence_asset_tag_number, 0.2
            )

        _, review_required = aggregate_confidence(
            gemini_result, pipeline_result, self.settings
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        response = AnalyzeResponse(
            request_id=request_id,
            status="success",
            processing_time_ms=elapsed_ms,
            unified_view=AnalyzeUnifiedView(
                method=UnifiedViewMethod.DIRECT_IMAGE,
                width=pil_for_gemini.width,
                height=pil_for_gemini.height,
            ),
            asset=to_asset_fields(gemini_result, self.settings),
            tag_detection_reasoning=gemini_result.tag_detection_reasoning,
            barcodeposition=_normalize_barcode_position(gemini_result.barcodeposition),
            image_readability=gemini_result.imageReadability,
            detected_tag_number_raw=raw_tag_number,
            visible_labels=list(gemini_result_raw.visible_labels),
        )

        if use_cache:
            cache_key = hash_image_set([raw_bytes])
            self._cache[cache_key] = response

        logger.info(
            "analysis_complete",
            request_id=request_id,
            method=UnifiedViewMethod.DIRECT_IMAGE.value,
            review_required=review_required,
            detectedtagnumber_raw=raw_tag_number,
            detectedtagnumber_normalized=gemini_result.detectedtagnumber,
            elapsed_ms=elapsed_ms,
        )
        return response
