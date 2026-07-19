from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ueba_server.dto.ueba_dto import CollectRequest
from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api", tags=["telemetry"])


@router.get("/sample")
def sample() -> dict[str, Any]:
    return ueba_service.current_sample()


@router.post("/collect")
def collect(request: CollectRequest) -> dict[str, Any]:
    return ueba_service.start_collection(request)
