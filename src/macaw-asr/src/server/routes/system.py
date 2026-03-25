"""System routes: /, /api/version.

SRP: Health check and version info only.
"""

from __future__ import annotations

from fastapi import APIRouter

import macaw_asr
from macaw_asr.api.types import VersionResponse

router = APIRouter(tags=["system"])


@router.get("/")
@router.head("/")
async def health():
    return "macaw-asr is running"


@router.get("/api/version")
async def version():
    return VersionResponse(version=macaw_asr.__version__)
