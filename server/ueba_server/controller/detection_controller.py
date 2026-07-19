from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ueba_server.dto.ueba_dto import DemoRequest, DetectRequest
from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api", tags=["detection"])


@router.post("/demo")
def demo(request: DemoRequest) -> dict[str, Any]:
    return ueba_service.run_demo(request)


@router.post("/detect/once")
def detect_once(request: DetectRequest) -> dict[str, Any]:
    return ueba_service.detect_once(request)


@router.post("/detect/start")
def detect_start(request: DetectRequest) -> dict[str, Any]:
    return ueba_service.start_detection(request)
