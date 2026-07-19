from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ueba_prototype.collector import TelemetryCollector, collect_to_csv
from ueba_prototype.demo import run_demo
from ueba_prototype.detector import analyze_current_sample, run_detection_loop
from ueba_prototype.features import FEATURE_COLUMNS, load_csv, rows_to_matrix
from ueba_prototype.model import METADATA_FILE, ModelBundle, train_model
from ueba_prototype.reporter import append_jsonl, read_jsonl

from ueba_server.dto.ueba_dto import CollectRequest, DemoRequest, DetectRequest, TrainRequest

from .job_service import job_service


class UebaService:
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def features(self) -> dict[str, list[str]]:
        return {"features": FEATURE_COLUMNS}

    def model_status(self, model_dir: Path) -> dict[str, Any]:
        metadata_path = model_dir / METADATA_FILE
        if not metadata_path.exists():
            return {"exists": False, "model_dir": str(model_dir)}

        bundle = ModelBundle.load(model_dir)
        return {
            "exists": True,
            "model_dir": str(model_dir),
            "threshold": bundle.threshold,
            "feature_count": len(bundle.feature_columns),
            "training_sample_count": len(bundle.training_errors),
        }

    def current_sample(self) -> dict[str, Any]:
        collector = TelemetryCollector()
        return asdict(collector.sample(interval_seconds=1.0))

    def run_demo(self, request: DemoRequest) -> dict[str, Any]:
        reports = run_demo(request.output_dir)
        return {
            "output_dir": str(request.output_dir),
            "detected_count": len(reports),
            "reports": [report.to_dict() for report in reports],
        }

    def start_collection(self, request: CollectRequest) -> dict[str, Any]:
        duration_seconds = request.duration_hours * 60 * 60
        job = job_service.submit(
            "collect",
            lambda: {
                "samples": collect_to_csv(
                    request.output_path,
                    duration_seconds=duration_seconds,
                    interval_seconds=request.interval_seconds,
                ),
                "output_path": str(request.output_path),
            },
        )
        return job.to_dict()

    def train(self, request: TrainRequest) -> dict[str, Any]:
        if not request.data_path.exists():
            raise HTTPException(status_code=404, detail=f"Training data not found: {request.data_path}")

        rows = load_csv(request.data_path)
        matrix = rows_to_matrix(rows, FEATURE_COLUMNS)
        try:
            bundle = train_model(
                matrix,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                threshold_quantile=request.threshold_quantile,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        bundle.save(request.model_dir)
        return {
            "model_dir": str(request.model_dir),
            "threshold": bundle.threshold,
            "training_sample_count": len(bundle.training_errors),
        }

    def detect_once(self, request: DetectRequest) -> dict[str, Any]:
        self._ensure_model_exists(request.model_dir)
        sample_data, report = analyze_current_sample(request.model_dir)
        if report:
            append_jsonl(request.report_path, report)
        return {
            "sample": asdict(sample_data),
            "is_anomaly": report is not None,
            "report": report.to_dict() if report else None,
            "report_path": str(request.report_path),
        }

    def start_detection(self, request: DetectRequest) -> dict[str, Any]:
        self._ensure_model_exists(request.model_dir)
        job = job_service.submit(
            "detect",
            lambda: {
                "analyzed": run_detection_loop(
                    request.model_dir,
                    request.report_path,
                    interval_seconds=request.interval_seconds,
                    max_samples=request.max_samples,
                ),
                "report_path": str(request.report_path),
            },
        )
        return job.to_dict()

    def reports(self, report_path: Path) -> dict[str, Any]:
        items = read_jsonl(report_path)
        return {"report_path": str(report_path), "count": len(items), "reports": items[-100:]}

    def list_jobs(self) -> dict[str, Any]:
        return {"jobs": job_service.list_jobs()}

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def _ensure_model_exists(self, model_dir: Path) -> None:
        if not (model_dir / METADATA_FILE).exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_dir}")


ueba_service = UebaService()
