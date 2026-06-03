"""Labeled grid mosaic fallback composer."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.models.pipeline import ProcessedImage


def _grid_dims(count: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return cols, rows


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose_labeled_grid(
    images: list[ProcessedImage],
    settings: Settings,
    cell_size: int = 512,
    padding: int = 8,
    label_height: int = 28,
) -> Image.Image:
    cols, rows = _grid_dims(len(images))
    canvas_w = cols * cell_size + (cols + 1) * padding
    canvas_h = rows * (cell_size + label_height) + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _get_font()

    for slot, img in enumerate(images):
        col = slot % cols
        row = slot // cols
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + label_height + padding)

        thumb = img.pil_image.copy()
        thumb.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        tx = x + (cell_size - thumb.width) // 2
        ty = y + label_height + (cell_size - thumb.height) // 2
        canvas.paste(thumb, (tx, ty))

        draw.rectangle([x, y, x + cell_size, y + label_height], fill=(48, 48, 48))
        draw.text((x + 6, y + 4), img.label, fill=(220, 220, 220), font=font)

    return _cap_width(canvas, settings.max_output_width_px)


def compose_hybrid_canvas(
    main_bgr: np.ndarray,
    orphan_images: list[ProcessedImage],
    main_label: str,
    settings: Settings,
    inset_max_edge: int | None = None,
) -> Image.Image:
    main_rgb = cv2_to_pil(main_bgr)
    main_rgb = _cap_width(main_rgb, settings.max_output_width_px)

    if not orphan_images:
        return main_rgb

    max_edge = inset_max_edge or getattr(
        settings, "panorama_tag_thumb_max_edge", 160
    )
    inset_h = max_edge
    inset_w = max_edge
    inset_cols = min(4, len(orphan_images))
    inset_rows = math.ceil(len(orphan_images) / inset_cols)
    strip_h = inset_rows * (inset_h + 8) + 16
    canvas = Image.new("RGB", (main_rgb.width, main_rgb.height + strip_h), (32, 32, 32))
    canvas.paste(main_rgb, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)
    draw.text((8, 8), main_label, fill=(200, 200, 200), font=font)

    for slot, img in enumerate(orphan_images):
        col = slot % inset_cols
        row = slot // inset_cols
        x = 8 + col * (inset_w + 8)
        y = main_rgb.height + 8 + row * (inset_h + 8)
        thumb = img.pil_image.copy()
        thumb.thumbnail((inset_w, inset_h), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb.height + 2), img.label, fill=(180, 180, 180), font=font)

    return canvas


def cv2_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def _cap_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    new_size = (max_width, int(image.height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)
