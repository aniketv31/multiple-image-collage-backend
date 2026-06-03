"""API integration tests."""

import base64
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


def test_analyze_requires_image(client):
    app = create_app()
    test_client = TestClient(app)
    response = test_client.post("/v1/assets/analyze", data={})
    assert response.status_code == 400
    assert "image_base64" in response.json()["detail"]


def test_analyze_success_mocked(client):
    settings = Settings(gemini_api_key="fake-key")
    mock_result = CompositeAnalysisResult(
        imageAnalysis="Blue industrial equipment with minor wear visible.",
        detectedAsset="Test Asset",
        damage_assessment="Good condition with minor cosmetic wear.",
        detectedtagnumber="1234567890123456",
        imageReadability="Y",
        tag_detection_reasoning="Single photo. Rotated 0°. Counted 6 digits. High confidence.",
        barcodeposition={"position": "Front panel, center-right"},
        visible_labels=["R32 Eco-Friendly"],
        confidence_asset_name=0.9,
        confidence_asset_condition=0.8,
        confidence_asset_description=0.85,
        confidence_asset_tag_number=0.7,
    )

    with (
        patch("app.api.v1.assets.get_settings", return_value=settings),
        patch(
            "app.services.gemini.GeminiService.extract_from_image",
            new=AsyncMock(return_value=mock_result),
        ) as mock_extract,
    ):
        app = create_app()
        test_client = TestClient(app)
        img = make_test_image((90, 120, 150))
        response = test_client.post(
            "/v1/assets/analyze",
            files=[("image", ("photo.jpg", img, "image/jpeg"))],
        )

    assert response.status_code == 200
    mock_extract.assert_awaited_once()
    data = response.json()
    assert data["status"] == "success"
    assert data["asset"]["asset_name"] == "Test Asset"
    assert data["unified_view"]["method"] == "direct_image"
    assert "image_base64" not in data["unified_view"]
    assert "image_url" not in data["unified_view"]
    assert "confidence" not in data
    assert "quality_warnings" not in data
    assert "review_required" not in data
    assert "tag_zoom_source_label" not in data
    assert data["image_readability"] == "Y"
    assert data["detected_tag_number_raw"] == "1234567890123456"
    assert data["asset"]["asset_tag_number"] == "1234567890123456"
    assert "analysis_sources" not in data
    assert data["visible_labels"] == ["R32 Eco-Friendly"]


def test_analyze_success_base64_mocked(client):
    settings = Settings(gemini_api_key="fake-key")
    mock_result = CompositeAnalysisResult(
        detectedAsset="Base64 Asset",
        imageReadability="Y",
        confidence_asset_name=0.8,
        confidence_asset_condition=0.8,
        confidence_asset_description=0.8,
        confidence_asset_tag_number=0.8,
    )

    img = make_test_image((10, 20, 30))
    b64 = base64.b64encode(img).decode("ascii")

    with (
        patch("app.api.v1.assets.get_settings", return_value=settings),
        patch(
            "app.services.gemini.GeminiService.extract_from_image",
            new=AsyncMock(return_value=mock_result),
        ) as mock_extract,
    ):
        app = create_app()
        test_client = TestClient(app)
        response = test_client.post(
            "/v1/assets/analyze",
            data={"image_base64": f"data:image/jpeg;base64,{b64}"},
        )

    assert response.status_code == 200
    mock_extract.assert_awaited_once()
    assert response.json()["asset"]["asset_name"] == "Base64 Asset"
