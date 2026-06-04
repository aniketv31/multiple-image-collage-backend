"""Pydantic response and domain models (clean, grouped JSON for UI)."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UnifiedViewMethod(str, Enum):
    COLLAGE = "collage"
    MULTI_IMAGE = "multi_image"


# --------------------------------------------------------------------------- #
# Asset details
# --------------------------------------------------------------------------- #
class AssetDetails(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    estimated_dimensions: Optional[str] = None
    estimated_age: Optional[str] = None
    quantity: int = 1
    serial_number: Optional[str] = None
    asset_tag_number: Optional[str] = None
    specifications: list[str] = Field(default_factory=list)
    accessories: list[str] = Field(default_factory=list)
    distinguishing_features: list[str] = Field(default_factory=list)
    description: Optional[str] = None


# --------------------------------------------------------------------------- #
# Placement (shared by damage, barcode, stickers)
# --------------------------------------------------------------------------- #
class PlacementInfo(BaseModel):
    asset_location: Optional[str] = None
    horizontal: Optional[str] = None
    vertical: Optional[str] = None
    seen_in_image: Optional[int] = None
    in_frame_position: Optional[str] = None
    description: Optional[str] = None


# --------------------------------------------------------------------------- #
# Condition / damage
# --------------------------------------------------------------------------- #
class DamageItem(BaseModel):
    location: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    seen_in_image: Optional[int] = None
    detail: Optional[str] = None
    affects_function: Optional[bool] = None
    repair_action: Optional[str] = None
    placement: Optional[PlacementInfo] = None


class DamageSeverityCounts(BaseModel):
    minor: int = 0
    moderate: int = 0
    severe: int = 0


class ConditionReport(BaseModel):
    grade: Optional[str] = None
    overall_score: Optional[int] = Field(default=None, ge=0, le=100)
    summary: Optional[str] = None
    cosmetic_condition: Optional[str] = None
    structural_condition: Optional[str] = None
    functional_status: Optional[str] = None
    cleanliness: Optional[str] = None
    wear_level: Optional[str] = None
    usability: Optional[str] = None
    repair_recommendation: Optional[str] = None
    estimated_remaining_life: Optional[str] = None
    missing_parts: list[str] = Field(default_factory=list)
    functional_issues: list[str] = Field(default_factory=list)
    positive_aspects: list[str] = Field(default_factory=list)
    has_damage: bool = False
    damage_count: int = 0
    damage_by_severity: DamageSeverityCounts = Field(default_factory=DamageSeverityCounts)
    damage_items: list[DamageItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Identifiers (tags / labels / barcode)
# --------------------------------------------------------------------------- #
class BarcodeDetails(BaseModel):
    present: bool = False
    readable: bool = False
    placement: Optional[PlacementInfo] = None
    detection_reasoning: Optional[str] = None


class StickerItem(BaseModel):
    label_text: str
    sticker_type: Optional[str] = None
    placement: Optional[PlacementInfo] = None


class Identifiers(BaseModel):
    asset_tag_number: Optional[str] = None
    asset_tag_number_raw: Optional[str] = None
    tag_readable: bool = False
    tag_position: Optional[str] = None
    tag_detection_reasoning: Optional[str] = None
    visible_labels: list[str] = Field(default_factory=list)
    barcode: BarcodeDetails = Field(default_factory=BarcodeDetails)
    stickers: list[StickerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #
class MoneyRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class ValuationAmount(BaseModel):
    usd: MoneyRange = Field(default_factory=MoneyRange)
    inr: MoneyRange = Field(default_factory=MoneyRange)


class Valuation(BaseModel):
    as_is: ValuationAmount = Field(default_factory=ValuationAmount)
    like_new_reference: ValuationAmount = Field(default_factory=ValuationAmount)
    currency_note: str = "INR values use a fixed rate of 1 USD = configured rate."
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    assumptions: Optional[str] = None
    disclaimer: str = (
        "AI-generated estimate from images only; not a certified appraisal. "
        "Verify before financial decisions."
    )


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
class ConfidenceScores(BaseModel):
    overall: float = 0.0
    asset_name: float = 0.0
    asset_condition: float = 0.0
    asset_description: float = 0.0
    asset_tag_number: float = 0.0
    valuation: float = 0.0


# --------------------------------------------------------------------------- #
# Cost / tokens
# --------------------------------------------------------------------------- #
class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Breakdown of input_tokens (image_tokens + text_tokens == input_tokens)
    image_tokens: int = 0
    text_tokens: int = 0
    images_sent_to_gemini: int = 0
    per_image_token_budget: int = 0
    estimated_image_tokens: int = 0


class CostInfo(BaseModel):
    model: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    usd_to_inr: float
    total_cost_inr: float
    fx_source: str
    fx_is_fallback: bool
    fx_as_of: Optional[str] = None


# --------------------------------------------------------------------------- #
# Top-level response
# --------------------------------------------------------------------------- #
class AnalyzeResponse(BaseModel):
    collage_base64: Optional[str] = None
    request_id: str
    status: str = "success"
    processing_time_ms: int
    analysis_method: UnifiedViewMethod
    images_analyzed: int
    review_required: bool = False
    asset: AssetDetails
    condition: ConditionReport
    identifiers: Identifiers
    valuation: Valuation
    confidence: ConfidenceScores
    token_usage: TokenUsage
    cost: CostInfo


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool


# --------------------------------------------------------------------------- #
# LLM structured-output schema (passed to Gemini as response_schema)
# Keep flat and simple types so the JSON schema stays clean.
# --------------------------------------------------------------------------- #
class LLMDamageItem(BaseModel):
    location: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    seen_in_image: Optional[int] = None
    horizontal: Optional[str] = None
    vertical: Optional[str] = None
    in_frame_position: Optional[str] = None
    detail: Optional[str] = None
    affects_function: Optional[bool] = None
    repair_action: Optional[str] = None


class LLMStickerItem(BaseModel):
    label_text: Optional[str] = None
    sticker_type: Optional[str] = None
    asset_location: Optional[str] = None
    horizontal: Optional[str] = None
    vertical: Optional[str] = None
    seen_in_image: Optional[int] = None
    in_frame_position: Optional[str] = None


class LLMAnalysisResult(BaseModel):
    # Identity
    asset_name: Optional[str] = None
    category: Optional[str] = None
    asset_type: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    estimated_dimensions: Optional[str] = None
    estimated_age: Optional[str] = None
    quantity: Optional[int] = None
    serial_number: Optional[str] = None
    specifications: list[str] = Field(default_factory=list)
    accessories: list[str] = Field(default_factory=list)
    distinguishing_features: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    # Identifiers (before condition — less likely to be truncated)
    asset_tag_number: Optional[str] = None
    tag_readable: Optional[bool] = None
    tag_detection_reasoning: Optional[str] = None
    barcode_present: Optional[bool] = None
    barcode_asset_location: Optional[str] = None
    barcode_horizontal: Optional[str] = None
    barcode_vertical: Optional[str] = None
    barcode_seen_in_image: Optional[int] = None
    barcode_in_frame_position: Optional[str] = None
    barcode_position: Optional[str] = None
    stickers: list[LLMStickerItem] = Field(default_factory=list)
    visible_labels: list[str] = Field(default_factory=list)
    damage_items: list[LLMDamageItem] = Field(default_factory=list)

    # Condition (detailed)
    condition_summary: Optional[str] = None
    condition_grade: Optional[str] = None
    condition_score: Optional[int] = None
    cosmetic_condition: Optional[str] = None
    structural_condition: Optional[str] = None
    functional_status: Optional[str] = None
    cleanliness: Optional[str] = None
    wear_level: Optional[str] = None
    usability: Optional[str] = None
    repair_recommendation: Optional[str] = None
    estimated_remaining_life: Optional[str] = None
    missing_parts: list[str] = Field(default_factory=list)
    functional_issues: list[str] = Field(default_factory=list)
    positive_aspects: list[str] = Field(default_factory=list)

    # Confidence
    confidence_asset_name: float = 0.0
    confidence_asset_condition: float = 0.0
    confidence_asset_description: float = 0.0
    confidence_asset_tag_number: float = 0.0

    # Valuation (USD)
    estimated_value_usd_min: Optional[float] = None
    estimated_value_usd_max: Optional[float] = None
    like_new_value_usd_min: Optional[float] = None
    like_new_value_usd_max: Optional[float] = None
    valuation_confidence: float = 0.0
    valuation_assumptions: Optional[str] = None
