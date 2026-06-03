"""Pydantic response and domain models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UnifiedViewMethod(str, Enum):
    DIRECT_IMAGE = "direct_image"


class AssetFields(BaseModel):
    asset_name: Optional[str] = None
    asset_condition: Optional[str] = None
    asset_description: Optional[str] = None
    asset_tag_number: Optional[str] = None


class ValidationFields(BaseModel):
    namedescriptionmatch: Optional[str] = None
    namedescriptionmatchpercent: Optional[int] = Field(default=None, ge=0, le=100)
    reasoning: Optional[str] = None


class AnalyzeUnifiedView(BaseModel):
    method: UnifiedViewMethod
    width: int
    height: int


class AssetAnalysisResult(BaseModel):
    """Gemini output from multi-image asset understanding pass."""

    detectedAsset: Optional[str] = None
    imageAnalysis: Optional[str] = None
    damage_assessment: Optional[str] = None
    confidence_asset_name: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_condition: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_description: float = Field(ge=0.0, le=1.0, default=0.0)


class TagOcrResult(BaseModel):
    """Gemini output from tag/barcode OCR pass."""

    detectedtagnumber: Optional[str] = None
    tag_detection_reasoning: Optional[str] = None
    barcodeposition: Optional[dict | str] = None
    visible_labels: list[str] = Field(default_factory=list)
    imageReadability: Optional[str] = None
    confidence_asset_tag_number: float = Field(ge=0.0, le=1.0, default=0.0)


class ConfidenceScores(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    asset_name: float = Field(ge=0.0, le=1.0)
    asset_condition: float = Field(ge=0.0, le=1.0)
    asset_description: float = Field(ge=0.0, le=1.0)
    asset_tag_number: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str = "success"
    processing_time_ms: int
    unified_view: AnalyzeUnifiedView
    asset: AssetFields
    tag_detection_reasoning: Optional[str] = None
    barcodeposition: Optional[str] = None
    image_readability: Optional[str] = None
    detected_tag_number_raw: Optional[str] = None
    validation: Optional[ValidationFields] = None
    visible_labels: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool


class GeminiExtractionResult(BaseModel):
    """Schema aligned with SingleValidation phase-1 Gemini output."""

    imageAnalysis: Optional[str] = None
    imageReadability: Optional[str] = None
    detectedAsset: Optional[str] = None
    detectedtagnumber: Optional[str] = None
    tag_detection_reasoning: Optional[str] = None
    barcodeposition: Optional[dict | str] = None
    damage_assessment: Optional[str] = None
    confidence_asset_name: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_condition: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_description: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_tag_number: float = Field(ge=0.0, le=1.0, default=0.0)


class CompositeAnalysisResult(BaseModel):
    """Unified Gemini output from single image call."""

    detectedAsset: Optional[str] = None
    imageAnalysis: Optional[str] = None
    damage_assessment: Optional[str] = None
    confidence_asset_name: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_condition: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence_asset_description: float = Field(ge=0.0, le=1.0, default=0.0)
    detectedtagnumber: Optional[str] = None
    tag_detection_reasoning: Optional[str] = None
    barcodeposition: Optional[dict | str] = None
    visible_labels: list[str] = Field(default_factory=list)
    imageReadability: Optional[str] = None
    confidence_asset_tag_number: float = Field(ge=0.0, le=1.0, default=0.0)
    namedescriptionmatch: Optional[str] = None
    namedescriptionmatchpercent: Optional[int] = Field(default=None, ge=0, le=100)
    reasoning: Optional[str] = None
