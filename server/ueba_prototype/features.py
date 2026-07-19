from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FEATURE_COLUMNS = [
    "interval_seconds",
    "cpu_percent",
    "memory_percent",
    "swap_percent",
    "process_count",
    "user_process_count",
    "new_process_count",
    "thread_count",
    "open_file_count",
    "connection_count",
    "established_count",
    "listen_count",
    "remote_ip_count",
    "remote_port_count",
    "uncommon_remote_port_count",
    "bytes_sent_per_sec",
    "bytes_recv_per_sec",
    "packets_sent_per_sec",
    "packets_recv_per_sec",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rows_to_matrix(rows: Iterable[dict[str, object]], feature_columns: Sequence[str] = FEATURE_COLUMNS) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([_to_float(row.get(column, 0.0)) for column in feature_columns])

    if not matrix:
        raise ValueError("No rows were provided for feature extraction")

    return np.asarray(matrix, dtype=np.float64)


def sample_to_vector(sample: object, feature_columns: Sequence[str] = FEATURE_COLUMNS) -> np.ndarray:
    if hasattr(sample, "__dict__"):
        row = vars(sample)
    else:
        row = dict(sample)  # type: ignore[arg-type]
    return rows_to_matrix([row], feature_columns=feature_columns)


def _to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "FeatureNormalizer":
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.std

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix * self.std) + self.mean

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "FeatureNormalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float64), std=np.asarray(payload["std"], dtype=np.float64))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureNormalizer":
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
