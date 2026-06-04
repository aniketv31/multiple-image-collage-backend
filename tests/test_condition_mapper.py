"""Unit tests for damage mapping (single-call)."""

from app.models.responses import LLMAnalysisResult, LLMDamageItem
from app.services.condition_mapper import (
    build_damage_items,
    damage_needs_review,
    merge_damage_sources,
)


def test_build_damage_with_placement():
    llm = LLMAnalysisResult(
        damage_items=[
            LLMDamageItem(
                location="top lid rear-left",
                type="dent",
                severity="moderate",
                seen_in_image=2,
                horizontal="left",
                in_frame_position="upper-left",
                detail="~1cm dent on corner.",
                affects_function=False,
                repair_action="Buff panel.",
            ),
        ],
    )
    items = build_damage_items(llm, images_analyzed=3)
    assert len(items) == 1
    assert items[0].placement is not None
    assert items[0].placement.seen_in_image == 2
    assert items[0].placement.horizontal == "left"


def test_merge_damage_from_functional_issues():
    llm = LLMAnalysisResult(
        damage_items=[],
        functional_issues=["bent hinge restricts opening"],
    )
    items = merge_damage_sources(llm, max_images=2)
    assert len(items) == 1
    assert items[0].type == "functional"


def test_damage_needs_review():
    llm = LLMAnalysisResult(
        damage_items=[],
        condition_summary="Visible scratch on the lid and a small dent.",
    )
    assert damage_needs_review(llm) is True

    llm_ok = LLMAnalysisResult(
        damage_items=[LLMDamageItem(location="lid", type="scratch", severity="minor")],
        condition_summary="Scratch on lid.",
    )
    assert damage_needs_review(llm_ok) is False
