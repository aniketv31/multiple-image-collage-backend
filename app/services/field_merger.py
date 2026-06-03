"""Merge asset analysis and tag OCR Gemini results into unified output."""



import re



from app.config import Settings, get_settings

from app.models.responses import (

    AssetAnalysisResult,

    AssetFields,

    CompositeAnalysisResult,

    GeminiExtractionResult,

    TagOcrResult,

)



_TAG_DIGITS_RE = re.compile(r"^[0-9]+$")





def _normalize_tag_number(

    tag: str | None,

    settings: Settings | None = None,

) -> str | None:

    if tag is None:

        return None

    cleaned = str(tag).strip().strip('"\'')

    if not cleaned or cleaned.upper() == "UNREADABLE":

        return None



    digits_only = re.sub(r"\D", "", cleaned)

    if digits_only and digits_only != cleaned.replace(" ", ""):

        cleaned = digits_only



    if not _TAG_DIGITS_RE.match(cleaned):

        return None



    settings = settings or get_settings()

    min_len = 8

    max_len = 20

    if len(cleaned) < min_len or len(cleaned) > max_len:

        return None



    return cleaned





def composite_to_gemini_result(

    result: CompositeAnalysisResult,

    settings: Settings | None = None,

) -> GeminiExtractionResult:

    settings = settings or get_settings()

    description = result.imageAnalysis or ""

    if result.visible_labels:

        label_snippet = ", ".join(result.visible_labels[:5])

        if label_snippet and label_snippet.lower() not in description.lower():

            description = f"{description.rstrip('.')}. Visible labels: {label_snippet}."



    normalized_tag = _normalize_tag_number(result.detectedtagnumber, settings)

    tag_conf = result.confidence_asset_tag_number

    if result.detectedtagnumber and str(result.detectedtagnumber).upper() == "UNREADABLE":

        tag_conf = min(tag_conf, 0.2)

    elif normalized_tag is None and result.detectedtagnumber:

        tag_conf = min(tag_conf, 0.2)



    readability = result.imageReadability

    if readability is None:

        readability = "Y" if normalized_tag else "E"



    return GeminiExtractionResult(

        imageAnalysis=description,

        imageReadability=readability,

        detectedAsset=result.detectedAsset,

        detectedtagnumber=normalized_tag,

        tag_detection_reasoning=result.tag_detection_reasoning,

        barcodeposition=result.barcodeposition,

        damage_assessment=result.damage_assessment,

        confidence_asset_name=result.confidence_asset_name,

        confidence_asset_condition=result.confidence_asset_condition,

        confidence_asset_description=result.confidence_asset_description,

        confidence_asset_tag_number=tag_conf if normalized_tag else min(tag_conf, 0.2),

    )





def merge_analysis_results(

    asset: AssetAnalysisResult,

    tag: TagOcrResult,

    settings: Settings | None = None,

) -> GeminiExtractionResult:

    settings = settings or get_settings()

    description = asset.imageAnalysis or ""

    if tag.visible_labels:

        label_snippet = ", ".join(tag.visible_labels[:5])

        if label_snippet and label_snippet.lower() not in description.lower():

            description = f"{description.rstrip('.')}. Visible labels: {label_snippet}."



    normalized_tag = _normalize_tag_number(tag.detectedtagnumber, settings)

    tag_conf = tag.confidence_asset_tag_number

    if tag.detectedtagnumber and str(tag.detectedtagnumber).upper() == "UNREADABLE":

        tag_conf = min(tag_conf, 0.2)



    readability = tag.imageReadability

    if readability is None:

        readability = "Y" if normalized_tag else "E"



    return GeminiExtractionResult(

        imageAnalysis=description,

        imageReadability=readability,

        detectedAsset=asset.detectedAsset,

        detectedtagnumber=normalized_tag,

        tag_detection_reasoning=tag.tag_detection_reasoning,

        barcodeposition=tag.barcodeposition,

        damage_assessment=asset.damage_assessment,

        confidence_asset_name=asset.confidence_asset_name,

        confidence_asset_condition=asset.confidence_asset_condition,

        confidence_asset_description=asset.confidence_asset_description,

        confidence_asset_tag_number=tag_conf if normalized_tag else min(tag_conf, 0.2),

    )





def to_asset_fields(

    merged: GeminiExtractionResult,

    settings: Settings | None = None,

) -> AssetFields:

    settings = settings or get_settings()

    return AssetFields(

        asset_name=merged.detectedAsset,

        asset_condition=merged.damage_assessment,

        asset_description=merged.imageAnalysis,

        asset_tag_number=_normalize_tag_number(merged.detectedtagnumber, settings),

    )


