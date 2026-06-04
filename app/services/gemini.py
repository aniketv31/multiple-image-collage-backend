"""Gemini client — single call over one or many images, with token usage."""

import asyncio
from typing import Any

import structlog
from google import genai
from google.genai import types
from PIL import Image

from app.config import Settings
from app.models.responses import LLMAnalysisResult, TokenUsage
from app.pipeline.image_utils import image_to_bytes
from app.prompts.loader import get_analysis_prompt

logger = structlog.get_logger()

_MEDIA_RESOLUTION_MAP = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_HIGH",
}

MEDIA_RESOLUTION_IMAGE_TOKENS = {
    "low": 280,
    "medium": 560,
    "high": 1120,
    "ultra_high": 2240,
    "unspecified": 1120,
}


def _resolve_media_resolution(name: str) -> str:
    return _MEDIA_RESOLUTION_MAP.get((name or "high").strip().lower(), "MEDIA_RESOLUTION_HIGH")


def _per_image_token_budget(name: str) -> int:
    return MEDIA_RESOLUTION_IMAGE_TOKENS.get((name or "high").strip().lower(), 1120)


def _build_image_parts(
    images: list[Image.Image],
    prompt: str,
    image_labels: list[str] | None = None,
) -> list[types.Part]:
    parts: list[types.Part] = [types.Part.from_text(text=prompt)]
    n = len(images)
    for i, img in enumerate(images):
        if n > 1:
            label = (
                image_labels[i]
                if image_labels and i < len(image_labels)
                else f"Image {i + 1}"
            )
            parts.append(types.Part.from_text(text=f"--- {label} ({i + 1} of {n}) ---"))
        parts.append(
            types.Part.from_bytes(data=image_to_bytes(img), mime_type="image/jpeg")
        )
    return parts


class GeminiService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self.settings.gemini_api_key)

    async def analyze_images(
        self,
        images: list[Image.Image],
        media_resolution: str,
        locale: str = "en",
        image_labels: list[str] | None = None,
    ) -> tuple[LLMAnalysisResult, TokenUsage]:
        if not self.is_configured():
            raise RuntimeError("Gemini API key is not configured")
        if not images:
            raise ValueError("At least one image is required")

        prompt = get_analysis_prompt()
        if len(images) > 1:
            prompt += (
                f"\n\nThis request has {len(images)} images numbered 1 through {len(images)}. "
                "Use those indices for barcode_seen_in_image, stickers[].seen_in_image, "
                "and damage_items[].seen_in_image."
            )
        if locale != "en":
            prompt += f"\n\nRespond in locale: {locale}."

        parts = _build_image_parts(images, prompt, image_labels)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LLMAnalysisResult,
            temperature=self.settings.gemini_analyze_temperature,
            max_output_tokens=self.settings.gemini_max_output_tokens,
            media_resolution=_resolve_media_resolution(media_resolution),
        )
        return await self._generate(
            parts, config, images_sent=len(images), media_resolution=media_resolution
        )

    async def _generate(
        self,
        parts: list[types.Part],
        config: types.GenerateContentConfig,
        images_sent: int,
        media_resolution: str,
    ) -> tuple[LLMAnalysisResult, TokenUsage]:
        last_error: Exception | None = None
        for attempt in range(self.settings.gemini_max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.settings.gemini_model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                result = getattr(response, "parsed", None)
                if not isinstance(result, LLMAnalysisResult):
                    text = (response.text or "{}").strip()
                    result = LLMAnalysisResult.model_validate_json(text)
                return result, _extract_usage(response, images_sent, media_resolution)
            except Exception as exc:
                last_error = exc
                logger.warning("gemini_request_failed", attempt=attempt, error=str(exc))
                if attempt < self.settings.gemini_max_retries:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(f"Gemini analysis failed after retries: {last_error}") from last_error

    async def health_check(self) -> bool:
        if not self.is_configured():
            return False
        try:
            await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.settings.gemini_model,
                contents="Reply with OK",
                config=types.GenerateContentConfig(max_output_tokens=8),
            )
            return True
        except Exception:
            return False


def _extract_usage(
    response: Any,
    images_sent: int,
    media_resolution: str,
) -> TokenUsage:
    um = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(um, "total_token_count", 0) or 0)

    per_image = _per_image_token_budget(media_resolution)
    estimated_image_tokens = images_sent * per_image

    details = getattr(um, "prompt_tokens_details", None) or []
    actual_image_tokens = 0
    has_details = False
    for detail in details:
        has_details = True
        modality = str(getattr(detail, "modality", "")).upper()
        count = int(getattr(detail, "token_count", 0) or 0)
        if "IMAGE" in modality:
            actual_image_tokens += count

    if has_details:
        image_tokens = actual_image_tokens
    else:
        image_tokens = estimated_image_tokens

    image_tokens = max(0, min(image_tokens, input_tokens))
    text_tokens = max(0, input_tokens - image_tokens)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        images_sent_to_gemini=images_sent,
        per_image_token_budget=per_image,
        estimated_image_tokens=estimated_image_tokens,
    )
