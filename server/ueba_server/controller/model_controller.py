from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from ueba_server.dto.ueba_dto import TrainRequest
from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status")
def model_status(model_dir: Path = Query(default=Path("models/default"))) -> dict[str, Any]:
    return ueba_service.model_status(model_dir)


@router.post("/train")
def train_model(request: TrainRequest) -> dict[str, Any]:
    return ueba_service.train(request)
