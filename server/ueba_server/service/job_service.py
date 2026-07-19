from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class Job:
    id: str
    name: str
    status: str
    created_at: str
    finished_at: str | None = None
    result: Any = None
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class JobService:
    max_workers: int = 2
    _executor: ThreadPoolExecutor = field(init=False)
    _jobs: dict[str, Job] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def submit(self, name: str, func: Callable[[], Any]) -> Job:
        job = Job(id=str(uuid.uuid4()), name=name, status="queued", created_at=_now())
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            self._set_status(job.id, "running")
            try:
                result = func()
            except Exception as exc:  # pragma: no cover - keeps API jobs observable.
                self._finish(job.id, "failed", error=str(exc), tb=traceback.format_exc())
            else:
                self._finish(job.id, "completed", result=result)

        self._executor.submit(_run)
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            self._jobs[job_id].status = status

    def _finish(self, job_id: str, status: str, result: Any = None, error: str | None = None, tb: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.finished_at = _now()
            job.result = result
            job.error = error
            job.traceback = tb


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


job_service = JobService()
