"""Tests for merging asset and tag Gemini results."""



from app.models.responses import AssetAnalysisResult, TagOcrResult

from app.services.field_merger import merge_analysis_results, to_asset_fields





def test_merge_analysis_results_combines_fields():

    asset = AssetAnalysisResult(

        detectedAsset="Dell Laptop",

        imageAnalysis="Black laptop on desk.",

        damage_assessment="Good",

        confidence_asset_name=0.9,

        confidence_asset_condition=0.8,

        confidence_asset_description=0.85,

    )

    tag = TagOcrResult(

        detectedtagnumber="1234567890123456",

        tag_detection_reasoning="Clear barcode on back panel.",

        barcodeposition={"position": "Back center"},

        visible_labels=["Intel Core i7", "16GB RAM"],

        imageReadability="Y",

        confidence_asset_tag_number=0.75,

    )



    merged = merge_analysis_results(asset, tag)



    assert merged.detectedAsset == "Dell Laptop"

    assert merged.detectedtagnumber == "1234567890123456"

    assert merged.damage_assessment == "Good"

    assert "Visible labels:" in (merged.imageAnalysis or "")

    assert "Intel Core i7" in (merged.imageAnalysis or "")





def test_merge_unreadable_tag_lowers_confidence():

    tag = TagOcrResult(

        detectedtagnumber="UNREADABLE",

        confidence_asset_tag_number=0.9,

    )

    merged = merge_analysis_results(AssetAnalysisResult(), tag)

    assert merged.confidence_asset_tag_number <= 0.2





def test_to_asset_fields_normalizes_tag():

    from app.models.responses import GeminiExtractionResult



    merged = GeminiExtractionResult(

        detectedAsset="Chair",

        damage_assessment="Fair",

        imageAnalysis="Office chair.",

        detectedtagnumber="UNREADABLE",

    )

    fields = to_asset_fields(merged)

    assert fields.asset_name == "Chair"

    assert fields.asset_tag_number is None


