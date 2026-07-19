from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ueba_server.dto.ueba_dto import TrainRequest
from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api", tags=["training"])


@router.post("/train")
def train(request: TrainRequest) -> dict[str, Any]:
    return ueba_service.train(request)
