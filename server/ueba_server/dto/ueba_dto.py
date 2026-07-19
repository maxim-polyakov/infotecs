from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CollectRequest(BaseModel):
    duration_hours: float = Field(default=24.0, gt=0)
    interval_seconds: float = Field(default=5.0, gt=0)
    output_path: Path = Path("data/raw.csv")


class TrainRequest(BaseModel):
    data_path: Path = Path("data/raw.csv")
    model_dir: Path = Path("models/default")
    epochs: int = Field(default=250, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    threshold_quantile: float = Field(default=0.995, gt=0, lt=1)


class DetectRequest(BaseModel):
    model_dir: Path = Path("models/default")
    report_path: Path = Path("reports/anomalies.jsonl")
    interval_seconds: float = Field(default=5.0, gt=0)
    max_samples: int | None = Field(default=None, ge=1)


class DemoRequest(BaseModel):
    output_dir: Path = Path("reports/demo")


class JobResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    finished_at: str | None = None
    result: Any = None
    error: str | None = None
