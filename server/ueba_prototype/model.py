from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FEATURE_COLUMNS, FeatureNormalizer


MODEL_FILE = "autoencoder.npz"
METADATA_FILE = "metadata.json"


@dataclass
class Autoencoder:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(cls, input_dim: int, hidden_dim: int, seed: int = 42) -> "Autoencoder":
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, np.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim, dtype=np.float64)
        w2 = rng.normal(0.0, np.sqrt(2.0 / max(hidden_dim, 1)), size=(hidden_dim, input_dim))
        b2 = np.zeros(input_dim, dtype=np.float64)
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    def reconstruct(self, matrix: np.ndarray) -> np.ndarray:
        hidden = np.tanh(matrix @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def reconstruction_errors(self, matrix: np.ndarray) -> np.ndarray:
        reconstructed = self.reconstruct(matrix)
        return np.mean(np.square(reconstructed - matrix), axis=1)

    def feature_errors(self, vector: np.ndarray) -> np.ndarray:
        reconstructed = self.reconstruct(vector)
        return np.square(reconstructed - vector)[0]

    def train(self, matrix: np.ndarray, epochs: int = 250, learning_rate: float = 0.01) -> list[float]:
        losses: list[float] = []
        sample_count, feature_count = matrix.shape
        denominator = max(sample_count * feature_count, 1)

        for _ in range(epochs):
            hidden = np.tanh(matrix @ self.w1 + self.b1)
            output = hidden @ self.w2 + self.b2
            diff = output - matrix
            loss = float(np.mean(np.square(diff)))
            losses.append(loss)

            grad_output = (2.0 / denominator) * diff
            grad_w2 = hidden.T @ grad_output
            grad_b2 = grad_output.sum(axis=0)
            grad_hidden = (grad_output @ self.w2.T) * (1.0 - np.square(hidden))
            grad_w1 = matrix.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            self.w1 -= learning_rate * grad_w1
            self.b1 -= learning_rate * grad_b1
            self.w2 -= learning_rate * grad_w2
            self.b2 -= learning_rate * grad_b2

        return losses

    def save(self, path: Path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: Path) -> "Autoencoder":
        payload = np.load(path)
        return cls(w1=payload["w1"], b1=payload["b1"], w2=payload["w2"], b2=payload["b2"])


@dataclass
class ThreatClassifier:
    labels: list[str]
    centroids: np.ndarray

    def predict(self, normalized_vector: np.ndarray) -> dict[str, float | str]:
        distances = np.linalg.norm(self.centroids - normalized_vector[0], axis=1)
        similarities = 1.0 / np.maximum(distances, 1e-9)
        probabilities = similarities / float(similarities.sum())
        index = int(np.argmax(probabilities))
        return {"threat_class": self.labels[index], "threat_confidence": float(probabilities[index])}

    def to_json(self) -> dict[str, object]:
        return {"labels": self.labels, "centroids": self.centroids.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ThreatClassifier":
        return cls(
            labels=[str(label) for label in payload["labels"]],
            centroids=np.asarray(payload["centroids"], dtype=np.float64),
        )


@dataclass
class ModelBundle:
    autoencoder: Autoencoder
    normalizer: FeatureNormalizer
    feature_columns: list[str]
    threshold: float
    training_errors: list[float]
    threat_classifier: ThreatClassifier | None = None

    def evaluate(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = self.normalizer.transform(matrix)
        reconstructed = self.autoencoder.reconstruct(normalized)
        errors = np.mean(np.square(reconstructed - normalized), axis=1)
        return errors, normalized, reconstructed

    def classify_threat(self, normalized_vector: np.ndarray) -> dict[str, float | str]:
        if not self.threat_classifier:
            return {"threat_class": "unknown_anomaly", "threat_confidence": 0.0}
        return self.threat_classifier.predict(normalized_vector)

    def is_anomaly(self, matrix: np.ndarray) -> np.ndarray:
        errors, _, _ = self.evaluate(matrix)
        return errors > self.threshold

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.autoencoder.save(model_dir / MODEL_FILE)
        metadata = {
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
            "training_errors": self.training_errors,
            "normalizer": self.normalizer.to_json(),
            "threat_classifier": self.threat_classifier.to_json() if self.threat_classifier else None,
        }
        (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> "ModelBundle":
        metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
        threat_classifier = metadata.get("threat_classifier")
        return cls(
            autoencoder=Autoencoder.load(model_dir / MODEL_FILE),
            normalizer=FeatureNormalizer.from_json(metadata["normalizer"]),
            feature_columns=list(metadata["feature_columns"]),
            threshold=float(metadata["threshold"]),
            training_errors=[float(value) for value in metadata.get("training_errors", [])],
            threat_classifier=ThreatClassifier.from_json(threat_classifier) if threat_classifier else None,
        )


def train_model(
    matrix: np.ndarray,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    hidden_dim: int | None = None,
    epochs: int = 250,
    learning_rate: float = 0.01,
    threshold_quantile: float = 0.995,
    seed: int = 42,
) -> ModelBundle:
    if matrix.ndim != 2:
        raise ValueError("Training matrix must be two-dimensional")
    if matrix.shape[0] < 2:
        raise ValueError("At least two training samples are required")

    normalizer = FeatureNormalizer.fit(matrix)
    normalized = normalizer.transform(matrix)
    hidden_dim = hidden_dim or max(3, min(12, matrix.shape[1] // 2))
    autoencoder = Autoencoder.create(input_dim=matrix.shape[1], hidden_dim=hidden_dim, seed=seed)
    autoencoder.train(normalized, epochs=epochs, learning_rate=learning_rate)
    errors = autoencoder.reconstruction_errors(normalized)
    threshold = float(np.quantile(errors, threshold_quantile))
    threat_classifier = train_threat_classifier(normalized, list(feature_columns), seed=seed)

    return ModelBundle(
        autoencoder=autoencoder,
        normalizer=normalizer,
        feature_columns=list(feature_columns),
        threshold=threshold,
        training_errors=[float(value) for value in errors],
        threat_classifier=threat_classifier,
    )


def train_threat_classifier(normalized_matrix: np.ndarray, feature_columns: list[str], seed: int = 42) -> ThreatClassifier:
    rng = np.random.default_rng(seed)
    sample_count = normalized_matrix.shape[0]
    scenario_offsets = {
        "data_exfiltration": {
            "bytes_sent_per_sec": 7.5,
            "packets_sent_per_sec": 4.5,
            "remote_ip_count": 2.5,
            "remote_port_count": 2.5,
        },
        "c2_or_remote_access": {
            "established_count": 6.5,
            "connection_count": 4.5,
            "remote_ip_count": 4.0,
            "bytes_recv_per_sec": 2.5,
        },
        "port_tunneling_or_policy_bypass": {
            "uncommon_remote_port_count": 7.0,
            "remote_port_count": 5.5,
            "listen_count": 3.0,
        },
        "process_execution_burst": {
            "new_process_count": 7.0,
            "process_count": 4.0,
            "cpu_percent": 4.0,
            "thread_count": 3.0,
        },
        "resource_abuse": {
            "cpu_percent": 7.0,
            "memory_percent": 4.5,
            "thread_count": 4.0,
            "open_file_count": 3.0,
        },
    }
    labels: list[str] = []
    centroids: list[np.ndarray] = []

    for label, offsets in scenario_offsets.items():
        synthetic = normalized_matrix[rng.integers(0, sample_count, size=min(256, max(sample_count, 16)))].copy()
        for feature, offset in offsets.items():
            if feature in feature_columns:
                synthetic[:, feature_columns.index(feature)] += offset
        labels.append(label)
        centroids.append(synthetic.mean(axis=0))

    return ThreatClassifier(labels=labels, centroids=np.vstack(centroids))
