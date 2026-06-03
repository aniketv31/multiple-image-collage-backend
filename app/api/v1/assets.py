"""Asset analysis API endpoints."""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.models.responses import AnalyzeResponse, HealthResponse
from app.services.analyzer import AssetAnalysisService
from app.services.gemini import GeminiService
from app.services.metrics import ANALYSIS_METHOD, REQUEST_COUNT, REQUEST_LATENCY
from app.services.rate_limiter import RateLimiter
from app.utils.timing import timer
from app.utils.uploads import SINGLE_IMAGE_OPENAPI, resolve_image_input

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
    "/assets/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze asset from one image (Gemini)",
    description=(
        "Send one photo as a file upload (`image`) or as base64 (`image_base64`, "
        "including data URLs). The image is preprocessed and sent to Gemini for "
        "asset identification, condition, description, and tag/barcode."
    ),
    openapi_extra=SINGLE_IMAGE_OPENAPI,
)
async def analyze_assets(
    image: Annotated[
        UploadFile | None,
        File(description="Single asset photo (JPEG, PNG, or WebP). Omit if using image_base64."),
    ] = None,
    image_base64: Annotated[
        str | None,
        Form(
            description=(
                "Base64 image or data URL (e.g. data:image/jpeg;base64,...). "
                "Omit if using image file upload."
            ),
        ),
    ] = None,
    locale: Annotated[str, Form(description="Output language")] = "en",
    settings: Settings = Depends(get_settings),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analyzer: AssetAnalysisService = Depends(get_analyzer),
) -> AnalyzeResponse:
    rate_limiter.check("poc")
    parsed_file = await resolve_image_input(image, image_base64, settings)

    with timer() as elapsed:
        try:
            result = await analyzer.analyze(file=parsed_file, locale=locale)
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
    ANALYSIS_METHOD.labels(method=result.unified_view.method.value).inc()
    return result


_job_store: dict[str, dict] = {}


@router.post(
    "/assets/analyze/async",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Analysis"],
    summary="Analyze asset asynchronously",
)
async def analyze_assets_async(
    image: Annotated[
        UploadFile | None,
        File(description="Single asset photo. Omit if using image_base64."),
    ] = None,
    image_base64: Annotated[str | None, Form()] = None,
    locale: Annotated[str, Form()] = "en",
    settings: Settings = Depends(get_settings),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analyzer: AssetAnalysisService = Depends(get_analyzer),
) -> dict:
    rate_limiter.check("poc")
    job_id = str(uuid.uuid4())
    parsed_file = await resolve_image_input(image, image_base64, settings)

    _job_store[job_id] = {"status": "processing", "result": None}

    import asyncio

    async def _run():
        try:
            result = await analyzer.analyze(file=parsed_file, locale=locale)
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
