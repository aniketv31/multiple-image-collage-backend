"""API integration tests."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.responses import CompositeAnalysisResult
from tests.conftest import make_test_image


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gemini_configured" in data


def test_analyze_requires_minimum_images(client):
    app = create_app()
    test_client = TestClient(app)
    img = make_test_image((100, 100, 100))
    response = test_client.post(
        "/v1/assets/analyze",
        files=[("images", ("a.jpg", img, "image/jpeg"))],
    )
    assert response.status_code == 400


def test_analyze_success_mocked(client):
    settings = Settings(gemini_api_key="fake-key")
    mock_result = CompositeAnalysisResult(
        imageAnalysis="Blue industrial equipment with minor wear visible.",
        detectedAsset="Test Asset",
        damage_assessment="Good condition with minor cosmetic wear.",
        detectedtagnumber="1234567890123456",
        imageReadability="Y",
        tag_detection_reasoning="View Front. Rotated 0°. Counted 6 digits. High confidence.",
        barcodeposition={"position": "Front panel, center-right of Front view"},
        visible_labels=["R32 Eco-Friendly"],
        confidence_asset_name=0.9,
        confidence_asset_condition=0.8,
        confidence_asset_description=0.85,
        confidence_asset_tag_number=0.7,
    )

    with (
        patch("app.api.v1.assets.get_settings", return_value=settings),
        patch(
            "app.services.gemini.GeminiService.extract_from_composite",
            new=AsyncMock(return_value=mock_result),
        ) as mock_extract,
    ):
        app = create_app()
        test_client = TestClient(app)
        images = [make_test_image((i * 30, 60, 90)) for i in range(3)]
        files = [("images", (f"img{i}.jpg", img, "image/jpeg")) for i, img in enumerate(images)]
        response = test_client.post(
            "/v1/assets/analyze",
            data={"angles": "Front,Back,Left"},
            files=files,
        )

    assert response.status_code == 200
    mock_extract.assert_awaited_once()
    data = response.json()
    assert data["status"] == "success"
    assert data["asset"]["asset_name"] == "Test Asset"
    assert data["unified_view"]["method"] == "analysis_composite"
    assert "image_base64" not in data["unified_view"]
    assert "image_url" not in data["unified_view"]
    assert "confidence" not in data
    assert "quality_warnings" not in data
    assert "review_required" not in data
    assert data["image_readability"] == "Y"
    assert data["detected_tag_number_raw"] == "1234567890123456"
    assert data["asset"]["asset_tag_number"] == "1234567890123456"
    assert "analysis_sources" not in data
    assert data["visible_labels"] == ["R32 Eco-Friendly"]


def test_panorama_success(client):
    app = create_app()
    test_client = TestClient(app)
    images = [make_test_image((i * 30, 60, 90)) for i in range(3)]
    files = [("images", (f"img{i}.jpg", img, "image/jpeg")) for i, img in enumerate(images)]
    response = test_client.post(
        "/v1/assets/panorama",
        data={"angles": "Front,Back,Left", "include_image_base64": "true"},
        files=files,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["image_count"] == 3
    assert data["unified_view"]["method"] in (
        "collage_contact_sheet",
        "labeled_grid",
        "hybrid",
        "stitched_panorama",
    )
    assert data["unified_view"]["image_base64"] is not None
    assert data["unified_view"]["image_url"] is None
