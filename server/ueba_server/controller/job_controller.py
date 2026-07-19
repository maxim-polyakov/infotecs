from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ueba_server.service.ueba_service import ueba_service


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs() -> dict[str, Any]:
    return ueba_service.list_jobs()


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return ueba_service.get_job(job_id)
