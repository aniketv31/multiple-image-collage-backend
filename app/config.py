"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    min_images: int = 1
    max_images: int = 10
    max_image_size_mb: int = 15
    max_preprocess_edge_px: int = 2048
    gemini_analyze_temperature: float = 0.0
    gemini_max_output_tokens: int = 8192

    # Detail vs cost lever (SDK supports low/medium/high; ultra_high not available -> clamps to high)
    media_resolution_collage: str = "high"
    media_resolution_multi: str = "high"

    # Pricing for gemini-3.1-flash-lite (USD per 1M tokens), text/image/video input
    gemini_input_usd_per_1m: float = 0.25
    gemini_output_usd_per_1m: float = 1.50

    # USD->INR. Fixed at 100 (fx_enabled=False uses usd_to_inr_fallback as the rate).
    fx_enabled: bool = False
    fx_api_url: str = "https://api.frankfurter.app/latest"
    fx_cache_ttl_seconds: int = 3600
    fx_timeout_seconds: float = 4.0
    usd_to_inr_fallback: float = 100.0

    rate_limit_per_minute: int = 60
    gemini_max_retries: int = 2
    gemini_timeout_seconds: int = 30

    review_confidence_threshold: float = 0.65
    field_confidence_threshold: float = 0.5

    allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_multipart_part_bytes(self) -> int:
        """Per-part cap for multipart uploads (well above per-file image limit)."""
        return max(self.max_image_size_bytes * 2, 32 * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
