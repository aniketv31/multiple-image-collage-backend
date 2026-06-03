"""Vercel serverless entry — exposes FastAPI ASGI app."""

import sys
from pathlib import Path

# Project root is parent of api/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402
