"""Tests for Gemini prompt loading."""

from app.prompts.loader import get_analysis_prompt


def test_analysis_prompt_core_fields():
    prompt = get_analysis_prompt()
    assert "ONE OR MORE photographs" in prompt
    assert "USE ALL IMAGES TOGETHER" in prompt
    assert "asset_name" in prompt
    assert "specifications" in prompt
    assert "accessories" in prompt
    assert "damage_items" in prompt
    assert "condition_grade" in prompt
    assert "condition_score" in prompt
    assert "cosmetic_condition" in prompt
    assert "structural_condition" in prompt
    assert "functional_status" in prompt
    assert "repair_recommendation" in prompt
    assert "asset_tag_number" in prompt
    assert "estimated_value_usd_min" in prompt
    assert "like_new_value_usd_min" in prompt
    assert "valuation_confidence" in prompt
    assert "STICKERS (exhaustive" in prompt
    assert "DAMAGE (exhaustive" in prompt
    assert "NEVER merge multiple labels" in prompt


def test_analysis_prompt_validator_framing():
    prompt = get_analysis_prompt()
    assert "asset validator" in prompt.lower()
    # Must work whether images arrive separately or as a collage.
    assert "separate images" in prompt
    assert "collage" in prompt
