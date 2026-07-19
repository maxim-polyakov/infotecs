from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SECURITY_HINTS = {
    "connection_count": "unusual number of network connections; possible scanning, C2 activity, or data exfiltration",
    "established_count": "many active outbound sessions; possible malware beaconing or unauthorized remote access",
    "remote_ip_count": "unusual number of remote peers; possible lateral movement or network discovery",
    "remote_port_count": "unusual diversity of remote ports; possible exploitation or tunneling",
    "uncommon_remote_port_count": "rare remote ports observed; possible non-standard C2 channel or policy bypass",
    "bytes_sent_per_sec": "high outbound traffic; possible data exfiltration",
    "bytes_recv_per_sec": "high inbound traffic; possible payload download or remote control",
    "packets_sent_per_sec": "high packet rate; possible scanning or flooding",
    "packets_recv_per_sec": "high inbound packet rate; possible remote interaction",
    "new_process_count": "many new processes; possible script execution, malware activity, or living-off-the-land behavior",
    "process_count": "process count differs from baseline; possible unauthorized software activity",
    "thread_count": "thread activity differs from baseline; possible heavy background workload",
    "cpu_percent": "CPU usage differs from baseline; possible miner, scanner, or intensive malware task",
    "memory_percent": "memory usage differs from baseline; possible resource abuse or suspicious workload",
}


@dataclass
class AnomalyReport:
    timestamp: float
    anomaly_score: float
    threshold: float
    top_features: list[dict[str, float | str]]
    explanations: list[str]
    sample: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "top_features": self.top_features,
            "explanations": self.explanations,
            "sample": self.sample,
        }


def build_report(
    sample: object,
    feature_columns: Sequence[str],
    normalized_vector: np.ndarray,
    reconstructed_vector: np.ndarray,
    anomaly_score: float,
    threshold: float,
    top_n: int = 5,
) -> AnomalyReport:
    sample_dict = _sample_to_dict(sample)
    feature_errors = np.square(reconstructed_vector[0] - normalized_vector[0])
    ranked_indexes = np.argsort(feature_errors)[::-1][:top_n]
    top_features = [
        {"feature": feature_columns[index], "contribution": float(feature_errors[index])}
        for index in ranked_indexes
        if float(feature_errors[index]) > 0.0
    ]
    explanations = explain_features([str(item["feature"]) for item in top_features])

    return AnomalyReport(
        timestamp=float(sample_dict.get("timestamp", 0.0)),
        anomaly_score=float(anomaly_score),
        threshold=float(threshold),
        top_features=top_features,
        explanations=explanations,
        sample=sample_dict,
    )


def explain_features(features: Iterable[str]) -> list[str]:
    explanations: list[str] = []
    for feature in features:
        hint = SECURITY_HINTS.get(feature)
        if hint and hint not in explanations:
            explanations.append(hint)
    if not explanations:
        explanations.append("behavior differs from the trained normal profile")
    return explanations


def append_jsonl(path: Path, report: AnomalyReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    reports: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                reports.append(json.loads(line))
    return reports


def write_csv(path: Path, reports: Sequence[AnomalyReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "anomaly_score", "threshold", "top_features", "explanations"])
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "timestamp": report.timestamp,
                    "anomaly_score": report.anomaly_score,
                    "threshold": report.threshold,
                    "top_features": "; ".join(str(item["feature"]) for item in report.top_features),
                    "explanations": "; ".join(report.explanations),
                }
            )


def _sample_to_dict(sample: object) -> dict[str, Any]:
    if hasattr(sample, "__dict__"):
        return dict(vars(sample))
    return dict(sample)  # type: ignore[arg-type]
