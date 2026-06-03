"""Single canvas collage with labeled boxes and TAG ZOOM row for one Gemini pass."""

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.models.pipeline import ProcessedImage
from app.pipeline.angle_groups import CollageSlot, SlotAssignment, assign_collage_slots
from app.pipeline.image_utils import image_to_bytes
from app.pipeline.tag_enhance import TagZoomPanels, build_tag_zoom_panels
from app.pipeline.tag_view_selector import select_tag_candidates

LABEL_HEIGHT = 28
PADDING = 8
BG_COLOR = (32, 32, 32)
HEADER_BG = (48, 48, 48)
TAG_ZOOM_BANNER = "TAG ZOOM (read barcodes here first)"


@dataclass
class CollageComposeParams:
    base_cell_px: int
    hero_scale: float
    tag_slot_scale: float
    tag_zoom_row_px: int


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _estimate_jpeg_bytes(image: Image.Image) -> int:
    return len(image_to_bytes(image, fmt="JPEG"))


def paste_contained(
    canvas: Image.Image,
    pil: Image.Image,
    box_x: int,
    box_y: int,
    box_w: int,
    box_h: int,
    label: str,
    label_height: int = LABEL_HEIGHT,
) -> None:
    """Fit image inside box with uniform scale (contain), centered."""
    draw = ImageDraw.Draw(canvas)
    font = _get_font(max(12, label_height - 8))
    draw.rectangle(
        [box_x, box_y, box_x + box_w, box_y + label_height],
        fill=HEADER_BG,
    )
    draw.text((box_x + 6, box_y + 4), label, fill=(220, 220, 220), font=font)

    inner_y = box_y + label_height
    inner_h = box_h - label_height
    if inner_h <= 0 or box_w <= 0:
        return

    img = pil.copy()
    scale = min(box_w / img.width, inner_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    tx = box_x + (box_w - new_w) // 2
    ty = inner_y + (inner_h - new_h) // 2
    canvas.paste(img, (tx, ty))


def resolve_tag_image_for_zoom(
    images: list[ProcessedImage],
    assignments: list[SlotAssignment],
    tag_image_index: int | None,
) -> ProcessedImage | None:
    """Pick upload for TAG ZOOM: explicit index > TAG slot > LEFT > RIGHT > BACK > heuristic."""
    if tag_image_index is not None and 0 <= tag_image_index < len(images):
        return images[tag_image_index]

    slot_order = (
        CollageSlot.TAG,
        CollageSlot.LEFT,
        CollageSlot.RIGHT,
        CollageSlot.BACK,
    )
    for slot in slot_order:
        match = next((a for a in assignments if a.slot == slot), None)
        if match:
            return match.image

    candidates, _, _ = select_tag_candidates(images, tag_image_index=None, max_candidates=1)
    return candidates[0] if candidates else None


def _slot_display_label(assignment: SlotAssignment) -> str:
    if assignment.slot == CollageSlot.HERO:
        return f"HERO — {assignment.slot_label}"
    if assignment.slot == CollageSlot.TAG:
        return f"Tag — {assignment.slot_label}"
    if assignment.slot == CollageSlot.LEFT:
        return f"Left — {assignment.slot_label}"
    return assignment.slot_label


def _layout_bottom_slots(
    assignments: list[SlotAssignment],
) -> list[SlotAssignment]:
    return [a for a in assignments if a.slot != CollageSlot.HERO]


def _layout_bottom_grid_dims(
    assignments: list[SlotAssignment],
) -> tuple[int, int, list[SlotAssignment], SlotAssignment | None]:
    hero = next((a for a in assignments if a.slot == CollageSlot.HERO), None)
    bottom = _layout_bottom_slots(assignments)
    if not bottom:
        return 0, 0, [], hero
    return 1, len(bottom), bottom, hero


def _build_collage_grid(
    assignments: list[SlotAssignment],
    params: CollageComposeParams,
    settings: Settings,
) -> Image.Image:
    bottom_rows, _, bottom_slots, hero = _layout_bottom_grid_dims(assignments)
    cell = params.base_cell_px
    tag_scale = settings.collage_tag_slot_scale
    padding = PADDING
    label_h = LABEL_HEIGHT

    hero_h = int(cell * params.hero_scale) if hero else 0
    bottom_cell_w = cell
    bottom_cell_h = cell
    tag_w = (
        int(bottom_cell_w * tag_scale)
        if any(s.slot in (CollageSlot.TAG, CollageSlot.LEFT) for s in bottom_slots)
        else bottom_cell_w
    )

    bottom_cols_layout: list[int] = []
    for slot in bottom_slots:
        if slot.slot in (CollageSlot.TAG, CollageSlot.LEFT):
            bottom_cols_layout.append(tag_w)
        else:
            bottom_cols_layout.append(bottom_cell_w)

    bottom_row_w = sum(bottom_cols_layout) + padding * (len(bottom_cols_layout) + 1)
    canvas_w = max(bottom_row_w, cell + 2 * padding)
    if hero:
        canvas_w = max(canvas_w, cell * 2 + padding * 3)

    bottom_total_h = bottom_rows * (bottom_cell_h + label_h) + (bottom_rows + 1) * padding
    canvas_h = bottom_total_h + (hero_h + label_h + padding * 2 if hero else 0)
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)

    y = padding
    if hero:
        hero_w = canvas_w - 2 * padding
        paste_contained(
            canvas,
            hero.image.pil_image,
            padding,
            y,
            hero_w,
            hero_h + label_h,
            _slot_display_label(hero),
            label_h,
        )
        y += hero_h + label_h + padding

    idx = 0
    x = padding
    row_h = bottom_cell_h + label_h
    for slot_assign in bottom_slots:
        w = tag_w if slot_assign.slot in (CollageSlot.TAG, CollageSlot.LEFT) else bottom_cell_w
        if x + w + padding > canvas_w and idx > 0:
            y += row_h + padding
            x = padding
            idx = 0
        paste_contained(
            canvas,
            slot_assign.image.pil_image,
            x,
            y,
            w,
            row_h,
            _slot_display_label(slot_assign),
            label_h,
        )
        x += w + padding
        idx += 1

    return canvas


def _tag_zoom_row_height(grid: Image.Image, tag_zoom_row_px: int, settings: Settings) -> int:
    ratio_h = int(grid.height * settings.tag_zoom_row_height_ratio)
    inner = max(tag_zoom_row_px, ratio_h, settings.tag_zoom_row_min_px)
    return 32 + inner + PADDING * 2


def _append_tag_zoom_row(
    grid: Image.Image,
    panels: TagZoomPanels | None,
    tag_zoom_row_px: int,
    settings: Settings,
    padding: int = PADDING,
) -> Image.Image:
    if panels is None:
        return grid

    label_h = 32
    font = _get_font(16)
    row_inner_h = max(
        tag_zoom_row_px,
        int(grid.height * settings.tag_zoom_row_height_ratio),
        settings.tag_zoom_row_min_px,
    )
    row_h = label_h + row_inner_h + padding * 2

    canvas_w = max(grid.width, settings.max_composite_width_px)
    row = Image.new("RGB", (canvas_w, row_h), (24, 24, 24))
    draw = ImageDraw.Draw(row)
    draw.text((padding, 4), TAG_ZOOM_BANNER, fill=(255, 200, 80), font=font)

    panel_specs = [
        (panels.upright, "TAG ZOOM — upright (read first)"),
        (panels.enhanced, "TAG ZOOM — enhanced"),
        (panels.raw, "TAG ZOOM — raw"),
    ]
    panel_count = len(panel_specs)
    slot_w = (canvas_w - (panel_count + 1) * padding) // panel_count
    y = label_h + padding

    for idx, (panel, sublabel) in enumerate(panel_specs):
        x = padding + idx * (slot_w + padding)
        paste_contained(
            row,
            panel,
            x,
            y,
            slot_w,
            row_inner_h + LABEL_HEIGHT,
            sublabel,
            LABEL_HEIGHT,
        )

    if grid.width != canvas_w:
        grid = grid.resize((canvas_w, grid.height), Image.Resampling.LANCZOS)

    combined = Image.new("RGB", (canvas_w, grid.height + row.height), BG_COLOR)
    combined.paste(grid, (0, 0))
    combined.paste(row, (0, grid.height))
    return combined


def _cap_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    return image.resize(
        (max_width, int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )


def _compose_at_params(
    assignments: list[SlotAssignment],
    tag_panels: TagZoomPanels | None,
    params: CollageComposeParams,
    settings: Settings,
) -> Image.Image:
    grid = _build_collage_grid(assignments, params, settings)
    grid = _cap_width(grid, settings.max_composite_width_px)
    composite = _append_tag_zoom_row(grid, tag_panels, params.tag_zoom_row_px, settings)
    if composite.width > settings.max_composite_width_px:
        composite = _cap_width(composite, settings.max_composite_width_px)
    return composite


def compute_optimal_collage_params(
    assignments: list[SlotAssignment],
    tag_panels: TagZoomPanels | None,
    settings: Settings,
) -> CollageComposeParams:
    """Binary search grid cell size; preserve TAG ZOOM row height."""
    tag_zoom_row_px = settings.tag_zoom_row_min_px if tag_panels else 0
    lo = settings.analysis_min_cell_px
    hi = settings.analysis_max_cell_px
    best = CollageComposeParams(
        base_cell_px=lo,
        hero_scale=settings.collage_hero_scale,
        tag_slot_scale=settings.collage_tag_slot_scale,
        tag_zoom_row_px=tag_zoom_row_px,
    )

    while lo <= hi:
        mid = (lo + hi) // 2
        trial_params = CollageComposeParams(
            base_cell_px=mid,
            hero_scale=settings.collage_hero_scale,
            tag_slot_scale=settings.collage_tag_slot_scale,
            tag_zoom_row_px=tag_zoom_row_px,
        )
        trial = _compose_at_params(assignments, tag_panels, trial_params, settings)
        if _estimate_jpeg_bytes(trial) <= settings.max_gemini_composite_bytes:
            best = trial_params
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def compose_collage_with_tag_zoom(
    images: list[ProcessedImage],
    settings: Settings,
    tag_image_index: int | None = None,
) -> tuple[Image.Image, CollageComposeParams, TagZoomPanels | None, str | None]:
    """Build one contact sheet: collage grid + TAG ZOOM (upright | enhanced | raw)."""
    assignments = assign_collage_slots(images, tag_image_index=tag_image_index)
    tag_source = resolve_tag_image_for_zoom(images, assignments, tag_image_index)

    tag_panels: TagZoomPanels | None = None
    source_label: str | None = None
    if tag_source:
        tag_panels = build_tag_zoom_panels(tag_source, settings)
        source_label = tag_panels.source_label

    params = compute_optimal_collage_params(assignments, tag_panels, settings)
    composite = _compose_at_params(assignments, tag_panels, params, settings)
    return composite, params, tag_panels, source_label


def compose_collage_for_panorama(
    images: list[ProcessedImage],
    settings: Settings,
) -> Image.Image:
    """Collage without TAG ZOOM row (panorama preview)."""
    assignments = assign_collage_slots(images)
    params = CollageComposeParams(
        base_cell_px=min(settings.analysis_max_cell_px, 640),
        hero_scale=settings.collage_hero_scale,
        tag_slot_scale=settings.collage_tag_slot_scale,
        tag_zoom_row_px=0,
    )
    lo, hi = settings.analysis_min_cell_px, settings.analysis_max_cell_px
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        params.base_cell_px = mid
        trial = _compose_at_params(assignments, None, params, settings)
        if _estimate_jpeg_bytes(trial) <= settings.max_gemini_composite_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    params.base_cell_px = best
    return _compose_at_params(assignments, None, params, settings)
