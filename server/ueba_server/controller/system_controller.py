from __future__ import annotations

from fastapi import APIRouter

from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return ueba_service.health()


@router.get("/features")
def features() -> dict[str, list[str]]:
    return ueba_service.features()
