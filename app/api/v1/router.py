"""API v1 router."""

from fastapi import APIRouter

from app.api.v1 import assets

router = APIRouter(prefix="/v1")
router.include_router(assets.router, tags=["assets"])
