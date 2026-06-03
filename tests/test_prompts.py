"""Tests for Gemini prompt loading."""

from app.prompts.loader import get_analysis_prompt


def test_analysis_prompt_core_fields():
    prompt = get_analysis_prompt()
    assert "ONE photograph" in prompt
    assert "detectedtagnumber" in prompt
    assert "detectedAsset" in prompt
    assert "damage_assessment" in prompt
    assert "visible_labels" in prompt
    assert "TAG ZOOM" not in prompt


def test_analysis_prompt_with_validation():
    prompt = get_analysis_prompt("Dell Laptop", "Office laptop")
    assert "Dell Laptop" in prompt
    assert "namedescriptionmatch" in prompt
    assert "VALIDATION" in prompt


def test_analysis_prompt_without_validation():
    prompt = get_analysis_prompt(None, None)
    assert "ONE photograph" in prompt
    assert "User's Asset Name" not in prompt
