# API output fields (collage vs multi)

Both `POST /v1/assets/analyze/collage` and `POST /v1/assets/analyze/multi` return the same `AnalyzeResponse` JSON shape defined in [`app/models/responses.py`](../app/models/responses.py).

## One Gemini call

Each analyze request uses a **single** `generate_content` call. Stickers and damage may be enriched from other fields in that same JSON (e.g. `specifications` → `identifiers.stickers`).

## `identifiers`

- Legacy: `asset_tag_number`, `tag_position`, `tag_detection_reasoning`, `visible_labels[]`
- `barcode`: `{ present, readable, placement, detection_reasoning }`
- `stickers[]`: one object per distinct physical label — `{ label_text, sticker_type, placement }`
- `placement`: `{ asset_location, horizontal, vertical, seen_in_image, in_frame_position, description }`. `description` is a detailed natural-language phrase (panel, sub-region, surface, adjacent landmarks, distance from edge, bar orientation, Image N). Populated for the barcode; `identifiers.tag_position` mirrors the same phrase.

## `condition.damage_items[]`

Each defect (additive `placement`):

- `location`, `type`, `severity`, `seen_in_image`, `detail`, `affects_function`, `repair_action`
- `placement`: `{ asset_location, horizontal, vertical, seen_in_image, in_frame_position }`

## `review_required`

Set when confidence is low, tag unreadable, barcode placement missing, label hints without stickers after merge, or damage mentioned in narrative without `damage_items[]`.
