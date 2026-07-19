from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .collector import TelemetrySample
from .features import FEATURE_COLUMNS, rows_to_matrix
from .model import train_model
from .reporter import AnomalyReport, build_report, write_csv


def generate_normal_rows(count: int = 240, seed: int = 7) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for index in range(count):
        rows.append(
            {
                "timestamp": time.time() + index,
                "interval_seconds": 5.0,
                "cpu_percent": float(rng.normal(18, 5)),
                "memory_percent": float(rng.normal(55, 4)),
                "swap_percent": float(max(rng.normal(8, 2), 0)),
                "process_count": float(rng.normal(145, 6)),
                "user_process_count": float(rng.normal(120, 5)),
                "new_process_count": float(max(rng.poisson(1), 0)),
                "thread_count": float(rng.normal(1700, 80)),
                "open_file_count": float(rng.normal(90, 18)),
                "connection_count": float(rng.normal(35, 5)),
                "established_count": float(rng.normal(12, 3)),
                "listen_count": float(rng.normal(18, 2)),
                "remote_ip_count": float(rng.normal(7, 2)),
                "remote_port_count": float(rng.normal(5, 1)),
                "uncommon_remote_port_count": float(max(rng.poisson(0.5), 0)),
                "bytes_sent_per_sec": float(max(rng.normal(9000, 2500), 0)),
                "bytes_recv_per_sec": float(max(rng.normal(18000, 4000), 0)),
                "packets_sent_per_sec": float(max(rng.normal(35, 8), 0)),
                "packets_recv_per_sec": float(max(rng.normal(48, 10), 0)),
            }
        )
    return rows


def generate_anomalous_samples() -> list[TelemetrySample]:
    now = time.time()
    return [
        _sample(now, connection_count=180, established_count=120, remote_ip_count=70, packets_sent_per_sec=420),
        _sample(now + 5, uncommon_remote_port_count=18, remote_port_count=25, bytes_sent_per_sec=260000),
        _sample(now + 10, new_process_count=35, process_count=210, cpu_percent=92, memory_percent=81),
    ]


def run_demo(output_dir: Path) -> list[AnomalyReport]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_rows = generate_normal_rows()
    matrix = rows_to_matrix(normal_rows)
    bundle = train_model(matrix, epochs=300, learning_rate=0.015, threshold_quantile=0.99)
    bundle.save(output_dir / "model")

    reports: list[AnomalyReport] = []
    for sample in generate_anomalous_samples():
        sample_matrix = rows_to_matrix([vars(sample)])
        errors, normalized, reconstructed = bundle.evaluate(sample_matrix)
        if float(errors[0]) > bundle.threshold:
            reports.append(
                build_report(
                    sample=sample,
                    feature_columns=FEATURE_COLUMNS,
                    normalized_vector=normalized,
                    reconstructed_vector=reconstructed,
                    anomaly_score=float(errors[0]),
                    threshold=bundle.threshold,
                )
            )

    write_csv(output_dir / "demo_anomalies.csv", reports)
    return reports


def _sample(timestamp: float, **overrides: float) -> TelemetrySample:
    base = {
        "timestamp": timestamp,
        "hostname": "demo-host",
        "os_name": "DemoOS",
        "interval_seconds": 5.0,
        "cpu_percent": 18.0,
        "memory_percent": 55.0,
        "swap_percent": 8.0,
        "process_count": 145,
        "user_process_count": 120,
        "new_process_count": 1,
        "thread_count": 1700,
        "open_file_count": 90,
        "connection_count": 35,
        "established_count": 12,
        "listen_count": 18,
        "remote_ip_count": 7,
        "remote_port_count": 5,
        "uncommon_remote_port_count": 1,
        "bytes_sent_per_sec": 9000.0,
        "bytes_recv_per_sec": 18000.0,
        "packets_sent_per_sec": 35.0,
        "packets_recv_per_sec": 48.0,
    }
    base.update(overrides)
    return TelemetrySample(**base)
