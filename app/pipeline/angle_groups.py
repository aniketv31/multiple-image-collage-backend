"""Map angle labels to collage layout slots and stitch eligibility."""

from dataclasses import dataclass
from enum import Enum

from app.models.pipeline import ProcessedImage

STITCH_GROUP_KEYWORDS = frozenset(
    {
        "front",
        "front-left",
        "front-right",
        "wide",
        "panorama",
        "forward",
    }
)

LAYOUT_TAG_KEYWORDS = frozenset(
    {"tag", "barcode", "label", "sticker", "serial", "qr"}
)

LAYOUT_SIDE_KEYWORDS = frozenset(
    {"left", "right", "back", "bottom", "top", "side", "rear", "underside"}
)

HERO_KEYWORDS = frozenset({"front", "forward", "wide", "panorama"})


class CollageSlot(str, Enum):
    HERO = "hero"
    TAG = "tag"
    LEFT = "left"
    RIGHT = "right"
    BACK = "back"
    CELL = "cell"


@dataclass
class SlotAssignment:
    slot: CollageSlot
    image: ProcessedImage
    slot_label: str


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", "-").replace(" ", "-")


def is_stitch_eligible(label: str) -> bool:
    """Side/tag views must not enter panorama stitch clusters."""
    lower = _normalize_label(label)
    if any(kw in lower for kw in LAYOUT_TAG_KEYWORDS):
        return False
    if any(kw in lower for kw in LAYOUT_SIDE_KEYWORDS):
        return False
    if lower in STITCH_GROUP_KEYWORDS:
        return True
    return any(kw in lower for kw in STITCH_GROUP_KEYWORDS)


def classify_slot(label: str) -> CollageSlot:
    lower = _normalize_label(label)
    if any(kw in lower for kw in LAYOUT_TAG_KEYWORDS):
        return CollageSlot.TAG
    if "left" in lower:
        return CollageSlot.LEFT
    if "right" in lower:
        return CollageSlot.RIGHT
    if "back" in lower or "rear" in lower:
        return CollageSlot.BACK
    if any(kw in lower for kw in HERO_KEYWORDS):
        return CollageSlot.HERO
    return CollageSlot.CELL


def assign_collage_slots(
    images: list[ProcessedImage],
    tag_image_index: int | None = None,
) -> list[SlotAssignment]:
    """
    Assign each image to a collage slot. tag_image_index forces that image to TAG slot.
    """
    slot_for_index: dict[int, CollageSlot] = {}
    preferred: dict[CollageSlot, ProcessedImage] = {}

    if tag_image_index is not None and 0 <= tag_image_index < len(images):
        preferred[CollageSlot.TAG] = images[tag_image_index]
        slot_for_index[tag_image_index] = CollageSlot.TAG

    for img in images:
        if img.index in slot_for_index:
            continue
        slot = classify_slot(img.label)
        if slot not in preferred:
            preferred[slot] = img
            slot_for_index[img.index] = slot

    if CollageSlot.HERO not in preferred and images:
        first = images[0]
        if first.index not in slot_for_index:
            preferred[CollageSlot.HERO] = first
            slot_for_index[first.index] = CollageSlot.HERO

    order = [
        CollageSlot.HERO,
        CollageSlot.LEFT,
        CollageSlot.RIGHT,
        CollageSlot.TAG,
        CollageSlot.BACK,
        CollageSlot.CELL,
    ]
    result: list[SlotAssignment] = []
    used: set[int] = set()

    for slot in order:
        if slot in preferred:
            img = preferred[slot]
            if img.index not in used:
                result.append(
                    SlotAssignment(slot=slot, image=img, slot_label=img.label)
                )
                used.add(img.index)

    for img in images:
        if img.index not in used:
            result.append(
                SlotAssignment(
                    slot=CollageSlot.CELL, image=img, slot_label=img.label
                )
            )
            used.add(img.index)

    return result


def filter_stitch_eligible(images: list[ProcessedImage]) -> list[ProcessedImage]:
    """Images safe to include in cv2 panorama stitch (excludes tag/side angles)."""
    return [img for img in images if is_stitch_eligible(img.label)]
