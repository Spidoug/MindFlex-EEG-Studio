from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .parser import EEG_BANDS
from .settings import app_data_dir

# BCI features are extracted from timed RAW epochs. The classifier therefore
# has the same input contract for Bluetooth, Serial/USB and replay/simulation.
FEATURE_NAMES = (
    *tuple(f"log_{name}" for name in EEG_BANDS),
    "rel_theta",
    "rel_alpha",
    "rel_beta",
    "rel_gamma",
    "log_theta_beta",
    "log_alpha_beta",
    "spectral_entropy",
    "spectral_centroid",
    "log_rms",
    "log_line_length",
)
MODEL_SCHEMA = 3
SESSION_SCHEMA = 3


@dataclass(slots=True)
class TrainingSample:
    label: str
    features: list[float]
    timestamp: float = 0.0
    trial_id: str = ""
    epoch_index: int = 0
    epoch_seconds: float = 1.0


@dataclass(slots=True)
class TrainingSession:
    name: str
    samples: list[TrainingSample] = field(default_factory=list)
    owner: str = ""

    @property
    def labels(self) -> list[str]:
        return sorted({sample.label for sample in self.samples})

    def count(self, label: str) -> int:
        return sum(1 for sample in self.samples if sample.label == label)

    def trial_count(self, label: str) -> int:
        ids = {sample.trial_id for sample in self.samples if sample.label == label and sample.trial_id}
        return len(ids)

    @property
    def total_trials(self) -> int:
        return len({sample.trial_id for sample in self.samples if sample.trial_id})

    def add(
        self,
        label: str,
        features: Iterable[float],
        timestamp: float = 0.0,
        *,
        trial_id: str = "",
        epoch_index: int = 0,
        epoch_seconds: float = 1.0,
    ) -> None:
        clean_label = str(label).strip()
        if not clean_label:
            raise ValueError("Label cannot be empty")
        vector = [float(x) for x in features]
        if len(vector) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(vector)}")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Features must be finite numbers")
        ts = float(timestamp)
        if not math.isfinite(ts):
            raise ValueError("Timestamp must be finite")
        seconds = float(epoch_seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Epoch duration must be positive")
        trial = str(trial_id).strip() or f"manual-{len(self.samples) + 1:06d}"
        self.samples.append(
            TrainingSample(
                label=clean_label,
                features=vector,
                timestamp=ts,
                trial_id=trial,
                epoch_index=max(0, int(epoch_index)),
                epoch_seconds=seconds,
            )
        )

    def clear(self) -> None:
        self.samples.clear()

    def trial_vectors(self) -> list[tuple[str, str, np.ndarray]]:
        """Return one mean vector per timed trial for unbiased validation."""
        groups: dict[tuple[str, str], list[list[float]]] = {}
        for sample in self.samples:
            groups.setdefault((sample.label, sample.trial_id), []).append(sample.features)
        result: list[tuple[str, str, np.ndarray]] = []
        for (label, trial_id), vectors in sorted(groups.items()):
            matrix = np.asarray(vectors, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
                continue
            result.append((label, trial_id, matrix.mean(axis=0)))
        return result


class FeatureExtractor:
    MIN_SAMPLES = 256

    @staticmethod
    def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask])) if np.any(mask) else 0.0

    @classmethod
    def from_raw(cls, samples: Iterable[float], sample_rate: float = 512.0) -> np.ndarray:
        """Extract one transport-independent BCI vector from a RAW epoch."""
        x = np.asarray(list(samples), dtype=np.float64)
        if x.ndim != 1 or x.size < cls.MIN_SAMPLES:
            raise ValueError(f"At least {cls.MIN_SAMPLES} RAW samples are required")
        if not np.isfinite(x).all():
            raise ValueError("RAW epoch contains non-finite values")
        x = x - float(np.mean(x))
        spread = float(np.std(x))
        if spread < 1e-6:
            raise ValueError("RAW epoch is flat")

        window = np.hanning(x.size)
        spectrum = np.fft.rfft(x * window)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
        bands = {
            "delta": cls._band_power(freqs, power, 0.5, 4.0),
            "theta": cls._band_power(freqs, power, 4.0, 8.0),
            "low_alpha": cls._band_power(freqs, power, 8.0, 10.0),
            "high_alpha": cls._band_power(freqs, power, 10.0, 13.0),
            "low_beta": cls._band_power(freqs, power, 13.0, 20.0),
            "high_beta": cls._band_power(freqs, power, 20.0, 30.0),
            "low_gamma": cls._band_power(freqs, power, 30.0, 40.0),
            "mid_gamma": cls._band_power(freqs, power, 40.0, 50.0),
        }
        total = sum(bands.values())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("RAW epoch has no usable spectral energy")
        eps = max(1e-12, total * 1e-12)
        theta = bands["theta"]
        alpha = bands["low_alpha"] + bands["high_alpha"]
        beta = bands["low_beta"] + bands["high_beta"]
        gamma = bands["low_gamma"] + bands["mid_gamma"]

        useful_mask = (freqs >= 0.5) & (freqs < 50.0)
        useful_power = power[useful_mask]
        useful_freqs = freqs[useful_mask]
        useful_total = float(np.sum(useful_power))
        probabilities = useful_power / max(useful_total, eps)
        probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        entropy /= math.log(max(2, useful_power.size))
        centroid = float(np.sum(useful_freqs * useful_power) / max(useful_total, eps)) / 50.0
        rms = math.sqrt(float(np.mean(x * x)))
        line_length = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0

        vector = np.asarray(
            [
                *[math.log1p(max(0.0, bands[name])) for name in EEG_BANDS],
                theta / total,
                alpha / total,
                beta / total,
                gamma / total,
                math.log((theta + eps) / (beta + eps)),
                math.log((alpha + eps) / (beta + eps)),
                entropy,
                centroid,
                math.log1p(rms),
                math.log1p(line_length),
            ],
            dtype=np.float64,
        )
        if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise ValueError("BCI feature vector is invalid")
        return vector


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


@dataclass(slots=True)
class CentroidModel:
    labels: list[str] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    centroids: dict[str, list[float]] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return len(self.labels) >= 2 and bool(self.centroids)

    @classmethod
    def train(cls, session: TrainingSession) -> "CentroidModel":
        if len(session.labels) < 2:
            raise ValueError("At least two classes are required")
        x = np.asarray([sample.features for sample in session.samples], dtype=np.float64)
        y = [sample.label for sample in session.samples]
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid training matrix")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        z = (x - mean) / scale
        labels = sorted(set(y))
        centroids: dict[str, list[float]] = {}
        for label in labels:
            indices = [index for index, current in enumerate(y) if current == label]
            centroids[label] = z[indices].mean(axis=0).tolist()
        return cls(labels=labels, mean=mean.tolist(), scale=scale.tolist(), centroids=centroids)

    def predict(self, features: Iterable[float]) -> Prediction:
        if not self.ready:
            raise RuntimeError("Model is not trained")
        vector = np.asarray(list(features), dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != len(FEATURE_NAMES) or not np.isfinite(vector).all():
            raise ValueError(f"Expected {len(FEATURE_NAMES)} finite features")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        z = (vector - mean) / scale
        distances: dict[str, float] = {}
        for label in self.labels:
            centroid = np.asarray(self.centroids[label], dtype=np.float64)
            distances[label] = float(np.linalg.norm(z - centroid))
        logits = {label: -distance for label, distance in distances.items()}
        maximum = max(logits.values())
        exp = {label: math.exp(value - maximum) for label, value in logits.items()}
        total = sum(exp.values()) or 1.0
        scores = {label: value / total for label, value in exp.items()}
        label = max(scores, key=scores.get)
        return Prediction(label=label, confidence=float(scores[label]), scores=scores)

    def to_dict(self) -> dict:
        return {
            "schema": MODEL_SCHEMA,
            "algorithm": "centroid-zscore",
            "feature_names": list(FEATURE_NAMES),
            "labels": self.labels,
            "mean": self.mean,
            "scale": self.scale,
            "centroids": self.centroids,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CentroidModel":
        if payload.get("schema") != MODEL_SCHEMA or payload.get("algorithm") != "centroid-zscore":
            raise ValueError("Unsupported model schema")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Model feature schema does not match this application")
        model = cls(
            labels=[str(value) for value in payload.get("labels", [])],
            mean=[float(value) for value in payload.get("mean", [])],
            scale=[float(value) for value in payload.get("scale", [])],
            centroids={str(key): [float(v) for v in values] for key, values in payload.get("centroids", {}).items()},
        )
        n = len(FEATURE_NAMES)
        if len(model.labels) < 2 or len(set(model.labels)) != len(model.labels):
            raise ValueError("Model labels are invalid")
        if len(model.mean) != n or len(model.scale) != n:
            raise ValueError("Model normalization dimensions are invalid")
        if any(value <= 0 or not math.isfinite(value) for value in model.scale):
            raise ValueError("Model normalization values are invalid")
        if set(model.centroids) != set(model.labels) or any(len(model.centroids[label]) != n for label in model.labels):
            raise ValueError("Model centroids are invalid")
        values = model.mean + model.scale + [v for label in model.labels for v in model.centroids[label]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Model contains non-finite values")
        return model


@dataclass(slots=True)
class ValidationResult:
    total: int
    correct: int
    accuracy: float
    per_label: dict[str, float]


def validate_model(model: CentroidModel, session: TrainingSession) -> ValidationResult:
    # Timed calibration creates several nearby epochs per trial. Validation is
    # therefore reported once per independent trial rather than inflating the
    # score with overlapping epochs.
    trials = session.trial_vectors()
    if not model.ready or not trials:
        return ValidationResult(0, 0, 0.0, {})
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    hits = 0
    for label, _trial_id, vector in trials:
        prediction = model.predict(vector)
        totals[label] = totals.get(label, 0) + 1
        if prediction.label == label:
            hits += 1
            correct[label] = correct.get(label, 0) + 1
    per_label = {label: correct.get(label, 0) / total for label, total in totals.items()}
    return ValidationResult(len(trials), hits, hits / len(trials), per_label)


class ModelStore:
    """Strict persistence for model, calibration session and metadata files."""

    def __init__(self, root: Path | None = None, *, owner_name: str = "") -> None:
        self.root = root or app_data_dir()
        self.owner_name = " ".join(str(owner_name).strip().split())

    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def session_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @staticmethod
    def normalize_name(name: str) -> str:
        safe = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "-_ ").strip()
        safe = " ".join(safe.split())
        if not safe:
            raise ValueError("Name cannot be empty")
        return safe

    def list_models(self) -> list[str]:
        if not self.model_dir.exists():
            return []
        return sorted(path.stem for path in self.model_dir.glob("*.json"))

    def save(self, name: str, model: CentroidModel) -> Path:
        safe = self.normalize_name(name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / f"{safe}.json"
        payload = model.to_dict()
        payload["owner"] = self.owner_name
        self._atomic_json(path, payload)
        return path

    def load(self, name: str) -> CentroidModel:
        safe = self.normalize_name(name)
        path = self.model_dir / f"{safe}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CentroidModel.from_dict(payload)

    def delete_model(self, name: str) -> None:
        safe = self.normalize_name(name)
        path = self.model_dir / f"{safe}.json"
        if path.exists():
            path.unlink()

    def list_sessions(self) -> list[str]:
        if not self.session_dir.exists():
            return []
        return sorted(path.stem for path in self.session_dir.glob("*.json"))

    def save_session(self, name: str, session: TrainingSession) -> Path:
        safe = self.normalize_name(name)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"{safe}.json"
        payload = {
            "schema": SESSION_SCHEMA,
            "sampling": "timed-trials",
            "name": session.name,
            "owner": session.owner or self.owner_name,
            "feature_names": list(FEATURE_NAMES),
            "samples": [asdict(sample) for sample in session.samples],
        }
        self._atomic_json(path, payload)
        return path

    def load_session(self, name: str) -> TrainingSession:
        safe = self.normalize_name(name)
        path = self.session_dir / f"{safe}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SESSION_SCHEMA or payload.get("sampling") != "timed-trials":
            raise ValueError("Unsupported calibration session schema")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Session feature schema does not match this application")
        session = TrainingSession(str(payload.get("name") or name), owner=str(payload.get("owner") or self.owner_name))
        for item in payload.get("samples", []):
            session.add(
                str(item["label"]),
                item["features"],
                float(item.get("timestamp", 0.0)),
                trial_id=str(item.get("trial_id", "")),
                epoch_index=int(item.get("epoch_index", 0)),
                epoch_seconds=float(item.get("epoch_seconds", 1.0)),
            )
        return session

    def delete_session(self, name: str) -> None:
        safe = self.normalize_name(name)
        path = self.session_dir / f"{safe}.json"
        if path.exists():
            path.unlink()

    def save_metadata(self, name: str, payload: dict) -> Path:
        safe = self.normalize_name(name)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        path = self.metadata_dir / f"{safe}.json"
        self._atomic_json(path, {"schema": 1, "owner": self.owner_name, "payload": payload})
        return path

    def load_metadata(self, name: str) -> dict:
        safe = self.normalize_name(name)
        path = self.metadata_dir / f"{safe}.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != 1 or not isinstance(payload.get("payload"), dict):
            raise ValueError("Unsupported metadata schema")
        return dict(payload["payload"])

    def delete_metadata(self, name: str) -> None:
        safe = self.normalize_name(name)
        path = self.metadata_dir / f"{safe}.json"
        if path.exists():
            path.unlink()

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
