"""Shared image encoding helpers."""

import base64
import hashlib
import io

from PIL import Image


def resize_for_gemini(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def image_to_bytes(pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, quality=90)
    return buf.getvalue()


def image_to_base64(pil_image: Image.Image) -> str:
    return base64.b64encode(image_to_bytes(pil_image)).decode("ascii")


def hash_image_set(image_bytes_list: list[bytes]) -> str:
    h = hashlib.sha256()
    for data in sorted(image_bytes_list, key=len):
        h.update(data)
    return h.hexdigest()
