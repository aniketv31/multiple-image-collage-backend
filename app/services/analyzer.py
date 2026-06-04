"""End-to-end asset analysis: preprocess -> Gemini -> cost, for both methods."""

import base64
import time
import uuid
from typing import BinaryIO

import structlog

from app.config import Settings
from app.models.responses import (
    AnalyzeResponse,
    AssetDetails,
    ConditionReport,
    ConfidenceScores,
    DamageSeverityCounts,
    LLMAnalysisResult,
    MoneyRange,
    UnifiedViewMethod,
    Valuation,
    ValuationAmount,
)
from app.pipeline.collage_composer import build_collage
from app.pipeline.image_utils import fit_images_to_budget, image_to_bytes
from app.pipeline.preprocess import preprocess_images
from app.services.cost import compute_cost
from app.services.field_merger import _clean_list, to_asset_details
from app.services.fx import get_usd_to_inr
from app.services.gemini import GeminiService
from app.services.condition_mapper import (
    build_damage_items,
    damage_needs_review,
    stickers_need_review,
)
from app.services.placement_mapper import (
    build_identifiers,
    identifiers_need_review,
    merge_sticker_sources,
    stickers_image_index_need_review,
)

logger = structlog.get_logger()

UploadTuple = tuple[BinaryIO, str, bytes]


class AssetAnalysisService:
    def __init__(self, settings: Settings, gemini: GeminiService):
        self.settings = settings
        self.gemini = gemini

    async def analyze(
        self,
        files: list[UploadTuple],
        method: UnifiedViewMethod,
        locale: str = "en",
    ) -> AnalyzeResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        processed = preprocess_images(files, self.settings)
        images = [p.pil_image for p in processed]

        # Resize all images if their combined payload exceeds the budget
        budget = self.settings.max_gemini_payload_bytes
        images = fit_images_to_budget(images, max_total_bytes=budget)

        collage_base64: str | None = None
        if method == UnifiedViewMethod.COLLAGE:
            collage = build_collage(images)
            # Re-check: the collage itself may exceed the budget
            [collage] = fit_images_to_budget([collage], max_total_bytes=budget)
            gemini_images = [collage]
            media_resolution = self.settings.media_resolution_collage
            collage_b64 = base64.b64encode(image_to_bytes(collage)).decode("ascii")
            collage_base64 = f"data:image/jpeg;base64,{collage_b64}"
        else:
            gemini_images = images
            media_resolution = self.settings.media_resolution_multi

        image_labels = [p.label for p in processed]
        llm, usage = await self.gemini.analyze_images(
            gemini_images,
            media_resolution=media_resolution,
            locale=locale,
            image_labels=image_labels if method == UnifiedViewMethod.MULTI_IMAGE else None,
            total_images=len(processed),
        )

        llm = merge_sticker_sources(llm, images_analyzed=len(processed))

        fx = await get_usd_to_inr(self.settings)
        cost = compute_cost(usage, fx, self.settings)

        asset: AssetDetails = to_asset_details(llm, self.settings)
        condition = self._build_condition(llm, len(processed))
        identifiers = build_identifiers(
            llm, asset.asset_tag_number, images_analyzed=len(processed)
        )
        confidence = self._build_confidence(llm)
        valuation = self._build_valuation(llm, fx.rate)
        review_required = (
            confidence.overall < self.settings.review_confidence_threshold
            or not identifiers.tag_readable
            or identifiers_need_review(llm, asset.asset_tag_number, len(processed))
            or stickers_need_review(llm)
            or stickers_image_index_need_review(llm.stickers, len(processed))
            or damage_needs_review(llm)
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        response = AnalyzeResponse(
            collage_base64=collage_base64,
            request_id=request_id,
            status="success",
            processing_time_ms=elapsed_ms,
            analysis_method=method,
            images_analyzed=len(processed),
            review_required=review_required,
            asset=asset,
            condition=condition,
            identifiers=identifiers,
            valuation=valuation,
            confidence=confidence,
            token_usage=usage,
            cost=cost,
        )

        logger.info(
            "analysis_complete",
            request_id=request_id,
            method=method.value,
            images_analyzed=len(processed),
            asset_name=asset.name,
            damage_count=condition.damage_count,
            review_required=review_required,
            input_tokens=usage.input_tokens,
            image_tokens=usage.image_tokens,
            text_tokens=usage.text_tokens,
            estimated_image_tokens=usage.estimated_image_tokens,
            output_tokens=usage.output_tokens,
            total_cost_usd=cost.total_cost_usd,
            total_cost_inr=cost.total_cost_inr,
            fx_source=cost.fx_source,
            detectedtagnumber_raw=llm.asset_tag_number,
            detectedtagnumber_normalized=asset.asset_tag_number,
            elapsed_ms=elapsed_ms,
        )
        return response

    @staticmethod
    def _build_condition(llm: LLMAnalysisResult, images_analyzed: int) -> ConditionReport:
        items = build_damage_items(llm, images_analyzed)
        counts = DamageSeverityCounts()
        for item in items:
            sev = (item.severity or "").strip().lower()
            if sev == "minor":
                counts.minor += 1
            elif sev == "moderate":
                counts.moderate += 1
            elif sev == "severe":
                counts.severe += 1

        score = llm.condition_score
        if isinstance(score, int):
            if 1 <= score <= 10:
                score = score * 10
            score = max(0, min(100, score))
        else:
            score = None

        return ConditionReport(
            grade=llm.condition_grade,
            overall_score=score,
            summary=llm.condition_summary,
            cosmetic_condition=llm.cosmetic_condition,
            structural_condition=llm.structural_condition,
            functional_status=llm.functional_status,
            cleanliness=llm.cleanliness,
            wear_level=llm.wear_level,
            usability=llm.usability,
            repair_recommendation=llm.repair_recommendation,
            estimated_remaining_life=llm.estimated_remaining_life,
            missing_parts=_clean_list(llm.missing_parts),
            functional_issues=_clean_list(llm.functional_issues),
            positive_aspects=_clean_list(llm.positive_aspects),
            has_damage=bool(items),
            damage_count=len(items),
            damage_by_severity=counts,
            damage_items=items,
        )

    @staticmethod
    def _build_confidence(llm: LLMAnalysisResult) -> ConfidenceScores:
        parts = [
            llm.confidence_asset_name,
            llm.confidence_asset_condition,
            llm.confidence_asset_description,
            llm.confidence_asset_tag_number,
        ]
        overall = round(sum(parts) / len(parts), 3) if parts else 0.0
        return ConfidenceScores(
            overall=overall,
            asset_name=round(llm.confidence_asset_name, 3),
            asset_condition=round(llm.confidence_asset_condition, 3),
            asset_description=round(llm.confidence_asset_description, 3),
            asset_tag_number=round(llm.confidence_asset_tag_number, 3),
            valuation=round(llm.valuation_confidence, 3),
        )

    def _build_valuation(self, llm: LLMAnalysisResult, usd_to_inr: float) -> Valuation:
        def amount(usd_min, usd_max) -> ValuationAmount:
            return ValuationAmount(
                usd=MoneyRange(min=usd_min, max=usd_max),
                inr=MoneyRange(
                    min=round(usd_min * usd_to_inr, 2) if usd_min is not None else None,
                    max=round(usd_max * usd_to_inr, 2) if usd_max is not None else None,
                ),
            )

        return Valuation(
            as_is=amount(llm.estimated_value_usd_min, llm.estimated_value_usd_max),
            like_new_reference=amount(llm.like_new_value_usd_min, llm.like_new_value_usd_max),
            currency_note=f"INR converted at 1 USD = {usd_to_inr:g} INR.",
            confidence=llm.valuation_confidence,
            assumptions=llm.valuation_assumptions,
        )
