from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .collector import TelemetryCollector, TelemetrySample
from .features import sample_to_vector
from .model import ModelBundle
from .reporter import AnomalyReport, append_jsonl, build_report


class Detector:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def analyze_sample(self, sample: TelemetrySample) -> AnomalyReport | None:
        matrix = sample_to_vector(sample, self.bundle.feature_columns)
        errors, normalized, reconstructed = self.bundle.evaluate(matrix)
        score = float(errors[0])
        if score <= self.bundle.threshold:
            return None
        return build_report(
            sample=sample,
            feature_columns=self.bundle.feature_columns,
            normalized_vector=normalized,
            reconstructed_vector=reconstructed,
            anomaly_score=score,
            threshold=self.bundle.threshold,
        )


def analyze_current_sample(model_dir: Path) -> tuple[TelemetrySample, AnomalyReport | None]:
    bundle = ModelBundle.load(model_dir)
    detector = Detector(bundle)
    collector = TelemetryCollector()
    time.sleep(1.0)
    sample = collector.sample(interval_seconds=1.0)
    return sample, detector.analyze_sample(sample)


def run_detection_loop(
    model_dir: Path,
    report_path: Path,
    interval_seconds: float = 5.0,
    max_samples: int | None = None,
    on_report: Callable[[AnomalyReport], None] | None = None,
) -> int:
    bundle = ModelBundle.load(model_dir)
    detector = Detector(bundle)
    collector = TelemetryCollector()
    analyzed = 0

    while max_samples is None or analyzed < max_samples:
        time.sleep(interval_seconds)
        sample = collector.sample(interval_seconds=interval_seconds)
        report = detector.analyze_sample(sample)
        if report:
            append_jsonl(report_path, report)
            if on_report:
                on_report(report)
        analyzed += 1

    return analyzed
