"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"
    max_images: int = 1
    min_images: int = 1
    max_image_size_mb: int = 7
    max_preprocess_edge_px: int = 2048
    gemini_analyze_temperature: float = 0.0

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
        """Multipart part cap for image_base64 (no practical limit for now)."""
        # Starlette default is 1MB; keep high until production hardening.
        return 2_147_483_647


@lru_cache
def get_settings() -> Settings:
    return Settings()
