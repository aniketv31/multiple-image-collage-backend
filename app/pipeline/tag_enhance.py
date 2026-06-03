"""Prepare tag/barcode ROI panels for TAG ZOOM row (image prep only, no OCR)."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import Settings
from app.models.pipeline import ProcessedImage


@dataclass
class TagZoomPanels:
    raw: Image.Image
    enhanced: Image.Image
    upright: Image.Image
    source_label: str


def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    rgb = np.array(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _band_roi_horizontal(gray: np.ndarray) -> tuple[int, int, int, int, float]:
    h, w = gray.shape
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_energy = np.mean(np.abs(sobel_y), axis=1)
    window = max(h // 8, 20)
    best_y = 0
    best_score = -1.0
    for y in range(0, max(h - window, 1)):
        score = float(np.mean(row_energy[y : y + window]))
        if score > best_score:
            best_score = score
            best_y = y
    band_h = max(int(h * 0.35), window)
    top = max(0, best_y - window // 2)
    bottom = min(h, top + band_h)
    margin_x = w // 10
    return margin_x, top, w - margin_x, bottom, best_score


def _band_roi_vertical(gray: np.ndarray) -> tuple[int, int, int, int, float]:
    h, w = gray.shape
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    col_energy = np.mean(np.abs(sobel_x), axis=0)
    window = max(w // 8, 20)
    best_x = 0
    best_score = -1.0
    for x in range(0, max(w - window, 1)):
        score = float(np.mean(col_energy[x : x + window]))
        if score > best_score:
            best_score = score
            best_x = x
    band_w = max(int(w * 0.35), window)
    left = max(0, best_x - window // 2)
    right = min(w, left + band_w)
    margin_y = h // 10
    return left, margin_y, right, h - margin_y, best_score


def extract_tag_roi(pil: Image.Image) -> Image.Image:
    """Crop label/barcode strip using best horizontal or vertical band."""
    bgr = _pil_to_bgr(pil)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    h_box = _band_roi_horizontal(gray)
    v_box = _band_roi_vertical(gray)
    if v_box[4] > h_box[4] * 1.05:
        left, top, right, bottom = v_box[0], v_box[1], v_box[2], v_box[3]
    else:
        left, top, right, bottom = h_box[0], h_box[1], h_box[2], h_box[3]

    if right - left < 50 or bottom - top < 50:
        crop_w = max(int(w * 0.75), min(w, 400))
        crop_h = max(int(h * 0.55), min(h, 300))
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right, bottom = left + crop_w, top + crop_h
    return pil.crop((left, top, right, bottom))


def _upscale_min_edge(pil: Image.Image, min_edge: int, max_scale: float) -> Image.Image:
    shortest = min(pil.width, pil.height)
    if shortest >= min_edge:
        return pil
    scale = min(min_edge / shortest, max_scale)
    return pil.resize(
        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _maybe_denoise(bgr: np.ndarray, blur_score: float | None) -> np.ndarray:
    if blur_score is not None and blur_score < 80:
        return cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    return bgr


def enhance_tag_roi(pil: Image.Image, settings: Settings, blur_score: float | None = None) -> Image.Image:
    """CLAHE + sharpen for uneven lighting."""
    bgr = _pil_to_bgr(pil)
    bgr = _maybe_denoise(bgr, blur_score)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=settings.tag_zoom_clahe_clip,
        tileGridSize=(4, 4),
    )
    l_ch = clahe.apply(l_ch)
    merged = cv2.merge([l_ch, a_ch, b_ch])
    bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(bgr, (0, 0), 2)
    bgr = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)
    return _bgr_to_pil(bgr)


def _to_upright(pil: Image.Image) -> Image.Image:
    """Rotate so the longer dimension is horizontal (typical barcode orientation)."""
    if pil.height > pil.width:
        return pil.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    return pil.copy()


def build_tag_zoom_panels(
    tag_image: ProcessedImage,
    settings: Settings,
) -> TagZoomPanels:
    """Build raw, enhanced, and upright panels from full-resolution upload."""
    pil = ImageOps.exif_transpose(tag_image.pil_image.copy())
    max_edge = settings.gemini_tag_max_edge_px
    longest = max(pil.size)
    if longest > max_edge:
        scale = max_edge / longest
        pil = pil.resize(
            (int(pil.width * scale), int(pil.height * scale)),
            Image.Resampling.LANCZOS,
        )

    raw_roi = extract_tag_roi(pil)
    raw_roi = _upscale_min_edge(
        raw_roi, settings.tag_zoom_panel_min_px, settings.tag_zoom_upscale_max
    )

    upright_roi = _to_upright(raw_roi)
    upright_roi = _upscale_min_edge(
        upright_roi, settings.tag_zoom_panel_min_px, settings.tag_zoom_upscale_max
    )

    blur = tag_image.blur_score
    enhanced_roi = enhance_tag_roi(raw_roi, settings, blur)
    upright_enhanced = enhance_tag_roi(upright_roi, settings, blur)

    return TagZoomPanels(
        raw=raw_roi,
        enhanced=enhanced_roi,
        upright=upright_enhanced,
        source_label=tag_image.label,
    )
