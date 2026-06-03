"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"
    max_images: int = 10
    min_images: int = 2
    max_image_size_mb: int = 7
    max_stitch_edge_px: int = 1600
    max_gemini_edge_px: int = 2048
    max_output_width_px: int = 4096
    max_gemini_composite_bytes: int = 6_500_000
    max_composite_width_px: int = 4096
    analysis_min_cell_px: int = 640
    analysis_max_cell_px: int = 1280
    tag_zoom_scale: float = 2.5

    collage_tag_slot_scale: float = 1.35
    collage_hero_scale: float = 1.2
    tag_zoom_row_min_px: int = 400
    tag_zoom_panel_min_px: int = 1200
    tag_zoom_upscale_max: float = 4.0
    tag_zoom_row_height_ratio: float = 0.45
    tag_zoom_clahe_clip: float = 3.0
    gemini_tag_max_edge_px: int = 2048
    gemini_analyze_temperature: float = 0.0
    panorama_default_layout: str = "collage"
    panorama_tag_thumb_max_edge: int = 480

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
