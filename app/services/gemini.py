"""Gemini AI Studio client — single image analysis pipeline."""

import asyncio
from typing import Any

import structlog
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from app.config import Settings
from app.models.responses import CompositeAnalysisResult
from app.pipeline.image_utils import image_to_bytes
from app.prompts.loader import get_analysis_prompt

logger = structlog.get_logger()


def _normalize_barcode_position(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        position = value.get("position")
    else:
        position = value
    if position is None:
        return None
    text = str(position).strip()
    return text or None


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

    async def extract_from_image(
        self,
        image: Image.Image,
        locale: str = "en",
    ) -> CompositeAnalysisResult:
        if not self.is_configured():
            raise RuntimeError("Gemini API key is not configured")

        prompt = get_analysis_prompt()
        if locale != "en":
            prompt += f"\n\nRespond in locale: {locale}."

        parts: list[types.Part] = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(
                data=image_to_bytes(image),
                mime_type="image/jpeg",
            ),
        ]

        return await self._call_structured(
            parts,
            CompositeAnalysisResult,
            temperature=self.settings.gemini_analyze_temperature,
        )

    async def _call_structured(
        self,
        parts: list[types.Part],
        schema_model: type[BaseModel],
        temperature: float = 0.1,
    ) -> BaseModel:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        )

        last_error: Exception | None = None
        for attempt in range(self.settings.gemini_max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.settings.gemini_model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                text = (response.text or "{}").strip()
                return schema_model.model_validate_json(text)
            except Exception as exc:
                last_error = exc
                logger.warning("gemini_request_failed", attempt=attempt, error=str(exc))
                if attempt < self.settings.gemini_max_retries:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError(f"Gemini extraction failed after retries: {last_error}") from last_error

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
