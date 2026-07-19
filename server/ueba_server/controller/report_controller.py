from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/reports")
def reports(report_path: Path = Query(default=Path("reports/anomalies.jsonl"))) -> dict[str, Any]:
    return ueba_service.reports(report_path)
