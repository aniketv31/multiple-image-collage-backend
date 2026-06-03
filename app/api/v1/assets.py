"""Asset analysis API endpoints."""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.models.responses import AnalyzeResponse, HealthResponse, PanoramaResponse
from app.services.analyzer import AssetAnalysisService
from app.services.gemini import GeminiService
from app.services.metrics import REQUEST_COUNT, REQUEST_LATENCY, STITCH_METHOD
from app.services.rate_limiter import RateLimiter
from app.utils.timing import timer
from app.utils.uploads import MULTI_IMAGE_OPENAPI, parse_angle_labels, parse_uploaded_images

router = APIRouter()
logger = structlog.get_logger()

_rate_limiter: RateLimiter | None = None
_analyzer: AssetAnalysisService | None = None


def get_rate_limiter(settings: Settings = Depends(get_settings)) -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    return _rate_limiter


def get_analyzer(settings: Settings = Depends(get_settings)) -> AssetAnalysisService:
    global _analyzer
    if _analyzer is None:
        _analyzer = AssetAnalysisService(
            settings=settings,
            gemini=GeminiService(settings),
        )
    return _analyzer


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    gemini = GeminiService(settings)
    return HealthResponse(status="ok", gemini_configured=gemini.is_configured())


@router.post(
    "/assets/panorama",
    response_model=PanoramaResponse,
    tags=["Panorama"],
    summary="Create unified panorama from multiple images",
    description=(
        "Upload 2–10 images and receive a single unified composite (stitched panorama, "
        "labeled grid, or hybrid). Use this endpoint to preview how images are combined "
        "before running full Gemini analysis."
    ),
    openapi_extra={
        **MULTI_IMAGE_OPENAPI,
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["images"],
                        "properties": {
                            **MULTI_IMAGE_OPENAPI["requestBody"]["content"][
                                "multipart/form-data"
                            ]["schema"]["properties"],
                            "include_image_base64": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include base64 JPEG in response for inline preview",
                            },
                        },
                    }
                }
            }
        },
    },
)
async def create_panorama(
    images: Annotated[
        list[UploadFile],
        File(
            description=(
                "Asset photos (2–10). In Swagger: click **Add item** and choose one file per row."
            ),
        ),
    ],
    angles: Annotated[
        str | None,
        Form(description="Comma-separated angle labels, e.g. Front,Back,Left,Right"),
    ] = None,
    include_image_base64: Annotated[
        bool,
        Form(description="Return base64 image in JSON for Swagger preview"),
    ] = True,
    layout: Annotated[
        str | None,
        Form(
            description=(
                "Composite layout: collage (default), stitch, or grid. "
                "Analyze always uses collage+TAG ZOOM; panorama preview only here."
            ),
        ),
    ] = None,
    settings: Settings = Depends(get_settings),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analyzer: AssetAnalysisService = Depends(get_analyzer),
) -> PanoramaResponse:
    rate_limiter.check("poc")
    parsed_files = await parse_uploaded_images(images, settings)
    angle_list = parse_angle_labels(angles)

    with timer() as elapsed:
        try:
            result = await analyzer.create_panorama(
                files=parsed_files,
                angles=angle_list,
                include_image_base64=include_image_base64,
                layout=layout,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    STITCH_METHOD.labels(method=result.unified_view.method.value).inc()
    REQUEST_LATENCY.observe(elapsed[0] / 1000)
    return result


@router.post(
    "/assets/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze asset from multiple images (Gemini)",
    description=(
        "Upload 2–10 images. All photos are combined into one high-resolution analysis "
        "contact sheet (labeled grid + tag zoom row) before a single Gemini call extracts "
        "asset name, condition, description, tag number, and optional name validation."
    ),
    openapi_extra={
        **MULTI_IMAGE_OPENAPI,
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["images"],
                        "properties": {
                            **MULTI_IMAGE_OPENAPI["requestBody"]["content"][
                                "multipart/form-data"
                            ]["schema"]["properties"],
                            "locale": {
                                "type": "string",
                                "default": "en",
                                "description": "Output language for Gemini response",
                            },
                            "tag_image_index": {
                                "type": "integer",
                                "description": (
                                    "0-based index of the image that best shows the asset tag/barcode"
                                ),
                            },
                        },
                    }
                }
            }
        },
    },
)
async def analyze_assets(
    images: Annotated[
        list[UploadFile],
        File(
            description=(
                "Asset photos (2–10). In Swagger: click **Add item** and choose one file per row."
            ),
        ),
    ],
    angles: Annotated[
        str | None,
        Form(description="Comma-separated angle labels, e.g. Front,Back,Left,Right"),
    ] = None,
    locale: Annotated[str, Form(description="Output language")] = "en",
    asset_name: Annotated[
        str | None,
        Form(description="Optional user-provided asset name for validation"),
    ] = None,
    description: Annotated[
        str | None,
        Form(description="Optional user-provided description for validation"),
    ] = None,
    tag_image_index: Annotated[
        int | None,
        Form(
            description=(
                "0-based index of the upload whose tag/barcode should be used for OCR "
                "(overrides automatic tag-view selection)"
            ),
        ),
    ] = None,
    settings: Settings = Depends(get_settings),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analyzer: AssetAnalysisService = Depends(get_analyzer),
) -> AnalyzeResponse:
    rate_limiter.check("poc")
    parsed_files = await parse_uploaded_images(images, settings)
    angle_list = parse_angle_labels(angles)

    with timer() as elapsed:
        try:
            result = await analyzer.analyze(
                files=parsed_files,
                angles=angle_list,
                locale=locale,
                tag_image_index=tag_image_index,
                user_asset_name=asset_name,
                user_description=description,
            )
        except ValueError as exc:
            REQUEST_COUNT.labels(status="400").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except RuntimeError as exc:
            REQUEST_COUNT.labels(status="503").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
                headers={"Retry-After": "30"},
            ) from exc

    REQUEST_COUNT.labels(status="200").inc()
    REQUEST_LATENCY.observe(elapsed[0] / 1000)
    STITCH_METHOD.labels(method=result.unified_view.method.value).inc()
    return result


_job_store: dict[str, dict] = {}


@router.post(
    "/assets/analyze/async",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Analysis"],
    summary="Analyze asset asynchronously",
)
async def analyze_assets_async(
    images: Annotated[list[UploadFile], File(description="Asset photos (2–10)")],
    angles: Annotated[str | None, Form()] = None,
    locale: Annotated[str, Form()] = "en",
    settings: Settings = Depends(get_settings),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analyzer: AssetAnalysisService = Depends(get_analyzer),
) -> dict:
    rate_limiter.check("poc")
    job_id = str(uuid.uuid4())
    parsed_files = await parse_uploaded_images(images, settings)
    angle_list = parse_angle_labels(angles)

    _job_store[job_id] = {"status": "processing", "result": None}

    import asyncio

    async def _run():
        try:
            result = await analyzer.analyze(
                files=parsed_files,
                angles=angle_list,
                locale=locale,
            )
            _job_store[job_id] = {"status": "completed", "result": result.model_dump()}
        except Exception as exc:
            _job_store[job_id] = {"status": "failed", "error": str(exc)}

    asyncio.create_task(_run())
    return {
        "request_id": job_id,
        "status": "processing",
        "poll_url": f"/v1/assets/analyze/{job_id}",
    }


@router.get("/assets/analyze/{request_id}", tags=["Analysis"])
async def get_analysis_job(request_id: str) -> dict:
    job = _job_store.get(request_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return {"request_id": request_id, **job}
