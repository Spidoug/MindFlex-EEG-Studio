from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .parser import EEG_BANDS
from .settings import app_data_dir
from .storage import atomic_write_json, read_json_object

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
MODEL_SCHEMA = 1
SESSION_SCHEMA = 1
FEATURE_SCHEMA = "mindflex-raw18-v1"
MODEL_ALGORITHM = "class-balanced-trial-centroid-zscore-v1"
SESSION_SAMPLING = "timed-trials-v1"
BCI_EPOCH_SECONDS = 2.0
MODEL_MAX_BYTES = 2 * 1024 * 1024
SESSION_MAX_BYTES = 128 * 1024 * 1024
METADATA_MAX_BYTES = 4 * 1024 * 1024


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float_list(values: object, *, name: str) -> list[float]:
    if not isinstance(values, list) or any(not _is_json_number(value) for value in values):
        raise ValueError(f"{name} must be a numeric JSON array")
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must contain only finite numbers")
    return parsed


@dataclass(slots=True)
class TrainingSample:
    label: str
    features: list[float]
    timestamp: float = 0.0
    trial_id: str = ""
    epoch_index: int = 0
    epoch_seconds: float = BCI_EPOCH_SECONDS


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
        return len({(sample.label, sample.trial_id) for sample in self.samples if sample.trial_id})

    def add(
        self,
        label: str,
        features: Iterable[float],
        timestamp: float = 0.0,
        *,
        trial_id: str = "",
        epoch_index: int = 0,
        epoch_seconds: float = BCI_EPOCH_SECONDS,
    ) -> None:
        if not isinstance(label, str):
            raise ValueError("Label must be a string")
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("Label cannot be empty")
        if len(clean_label) > 80 or any(ord(ch) < 32 for ch in clean_label):
            raise ValueError("Label is invalid")
        try:
            vector = [float(x) for x in features]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Features must be numeric") from exc
        if len(vector) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(vector)}")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Features must be finite numbers")
        if isinstance(timestamp, bool):
            raise ValueError("Timestamp must be numeric")
        try:
            ts = float(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Timestamp must be numeric") from exc
        if not math.isfinite(ts):
            raise ValueError("Timestamp must be finite")
        if isinstance(epoch_seconds, bool):
            raise ValueError("Epoch duration must be numeric")
        try:
            seconds = float(epoch_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Epoch duration must be numeric") from exc
        if not math.isfinite(seconds) or abs(seconds - BCI_EPOCH_SECONDS) > 1e-9:
            raise ValueError(f"MindFlex model V1 requires {BCI_EPOCH_SECONDS:g}s epochs")
        if not isinstance(trial_id, str):
            raise ValueError("Trial id must be a string")
        trial = trial_id.strip() or f"manual-{len(self.samples) + 1:06d}"
        if len(trial) > 128 or any(ord(ch) < 32 for ch in trial):
            raise ValueError("Trial id is invalid")
        if type(epoch_index) is not int:
            raise ValueError("Epoch index must be an integer")
        index = epoch_index
        if index < 0:
            raise ValueError("Epoch index cannot be negative")
        self.samples.append(
            TrainingSample(
                label=clean_label,
                features=vector,
                timestamp=ts,
                trial_id=trial,
                epoch_index=index,
                epoch_seconds=seconds,
            )
        )

    def clear(self) -> None:
        self.samples.clear()

    def training_fingerprint(self) -> str:
        """Hash the exact trial-balanced inputs consumed by model training."""
        trials = self.trial_vectors()
        if not trials:
            raise ValueError("Training session has no independent trials")
        payload = [
            [label, trial_id, [float(value) for value in vector]]
            for label, trial_id, vector in trials
        ]
        canonical = json.dumps(payload, sort_keys=False, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def trial_vectors(self) -> list[tuple[str, str, np.ndarray]]:
        """Return one strictly validated mean vector per independent timed trial."""
        groups: dict[tuple[str, str], list[list[float]]] = {}
        seen_epochs: set[tuple[str, str, int]] = set()
        for sample in self.samples:
            if not isinstance(sample, TrainingSample):
                raise ValueError("Training session contains an invalid sample object")
            if (
                not isinstance(sample.label, str)
                or not sample.label
                or sample.label != sample.label.strip()
                or len(sample.label) > 80
                or any(ord(ch) < 32 for ch in sample.label)
            ):
                raise ValueError("Training session contains an invalid label")
            if (
                not isinstance(sample.trial_id, str)
                or not sample.trial_id
                or sample.trial_id != sample.trial_id.strip()
                or len(sample.trial_id) > 128
                or any(ord(ch) < 32 for ch in sample.trial_id)
            ):
                raise ValueError("Training session contains an invalid trial id")
            if type(sample.epoch_index) is not int or sample.epoch_index < 0:
                raise ValueError("Training session contains an invalid epoch index")
            key = (sample.label, sample.trial_id, sample.epoch_index)
            if key in seen_epochs:
                raise ValueError("Training session contains a duplicate trial epoch")
            seen_epochs.add(key)
            if isinstance(sample.timestamp, bool) or not isinstance(sample.timestamp, (int, float)):
                raise ValueError("Training session contains an invalid timestamp")
            if not math.isfinite(float(sample.timestamp)):
                raise ValueError("Training session contains a non-finite timestamp")
            if isinstance(sample.epoch_seconds, bool) or not isinstance(sample.epoch_seconds, (int, float)):
                raise ValueError("Training session contains an invalid epoch duration")
            if not math.isfinite(float(sample.epoch_seconds)) or abs(float(sample.epoch_seconds) - BCI_EPOCH_SECONDS) > 1e-9:
                raise ValueError("Training session contains an incompatible epoch duration")
            if not isinstance(sample.features, list) or len(sample.features) != len(FEATURE_NAMES):
                raise ValueError("Training session contains an invalid feature vector")
            if any(not _is_json_number(value) or not math.isfinite(float(value)) for value in sample.features):
                raise ValueError("Training session contains non-finite or non-numeric features")
            groups.setdefault((sample.label, sample.trial_id), []).append(sample.features)
        result: list[tuple[str, str, np.ndarray]] = []
        for (label, trial_id), vectors in sorted(groups.items()):
            matrix = np.asarray(vectors, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
                raise ValueError("Training session contains an invalid trial matrix")
            mean_vector = matrix.mean(axis=0)
            if not np.isfinite(mean_vector).all():
                raise ValueError("Training trial mean is numerically invalid")
            result.append((label, trial_id, mean_vector))
        return result


class FeatureExtractor:
    MIN_SAMPLES = int(round(512.0 * BCI_EPOCH_SECONDS))

    @staticmethod
    def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask])) if np.any(mask) else 0.0

    SAMPLE_RATE = 512.0

    @classmethod
    def from_raw(
        cls,
        samples: Iterable[float],
        sample_rate: float = SAMPLE_RATE,
        *,
        epoch_seconds: float = BCI_EPOCH_SECONDS,
    ) -> np.ndarray:
        """Extract one fixed-duration, transport-independent BCI vector."""
        rate = float(sample_rate)
        seconds = float(epoch_seconds)
        if not math.isfinite(rate) or abs(rate - cls.SAMPLE_RATE) > 1e-9:
            raise ValueError("MindFlex BCI feature extraction requires the fixed 512 Hz RAW stream")
        if not math.isfinite(seconds) or abs(seconds - BCI_EPOCH_SECONDS) > 1e-9:
            raise ValueError(f"MindFlex model V1 requires {BCI_EPOCH_SECONDS:g}s RAW epochs")
        required = int(round(rate * BCI_EPOCH_SECONDS))
        x = np.asarray(list(samples), dtype=np.float64)
        if x.ndim != 1 or x.size < required:
            raise ValueError(f"A complete {seconds:g}s RAW epoch requires {required} samples")
        if x.size > required:
            x = x[-required:]
        if not np.isfinite(x).all():
            raise ValueError("RAW epoch contains non-finite values")
        if np.any(x < -32768.0) or np.any(x > 32767.0):
            raise ValueError("RAW epoch contains values outside the signed 16-bit TGAM range")
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
    training_fingerprint: str = ""

    @property
    def ready(self) -> bool:
        n = len(FEATURE_NAMES)
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            return False
        if any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or len(label) > 80
            or any(ord(ch) < 32 for ch in label)
            for label in self.labels
        ):
            return False
        if len(self.mean) != n or len(self.scale) != n or set(self.centroids) != set(self.labels):
            return False
        if len(self.training_fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in self.training_fingerprint):
            return False
        values: list[object] = [*self.mean, *self.scale]
        for label in self.labels:
            centroid = self.centroids.get(label)
            if not isinstance(centroid, list) or len(centroid) != n:
                return False
            values.extend(centroid)
        return (
            all(_is_json_number(value) and math.isfinite(float(value)) for value in values)
            and all(_is_json_number(value) and math.isfinite(float(value)) and float(value) > 0.0 for value in self.scale)
        )

    @classmethod
    def train(cls, session: TrainingSession) -> "CentroidModel":
        trials = session.trial_vectors()
        labels = sorted({label for label, _trial_id, _vector in trials})
        if len(labels) < 2:
            raise ValueError("At least two classes with independent trials are required")
        x = np.asarray([vector for _label, _trial_id, vector in trials], dtype=np.float64)
        y = [label for label, _trial_id, _vector in trials]
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid trial-balanced training matrix")
        # Equalize class influence in feature scaling. Communication training can
        # accumulate different numbers of trials per command; the scale must not
        # become a hidden class-prior weight. Each class contributes one equal
        # share to the normalization moments, while all trials inside that class
        # contribute equally to its statistics and centroid.
        class_arrays = {
            label: x[[index for index, current in enumerate(y) if current == label]]
            for label in labels
        }
        class_means = np.asarray([class_arrays[label].mean(axis=0) for label in labels], dtype=np.float64)
        mean = class_means.mean(axis=0)
        variance = np.asarray(
            [np.mean((class_arrays[label] - mean) ** 2, axis=0) for label in labels],
            dtype=np.float64,
        ).mean(axis=0)
        scale = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Training normalization is numerically invalid")
        scale[scale < 1e-6] = 1.0
        z = (x - mean) / scale
        if not np.isfinite(z).all():
            raise ValueError("Training normalization produced non-finite values")
        centroids: dict[str, list[float]] = {}
        for label in labels:
            indices = [index for index, current in enumerate(y) if current == label]
            centroid = z[indices].mean(axis=0)
            if not np.isfinite(centroid).all():
                raise ValueError("Training centroid is numerically invalid")
            centroids[label] = centroid.tolist()
        model = cls(
            labels=labels,
            mean=mean.tolist(),
            scale=scale.tolist(),
            centroids=centroids,
            training_fingerprint=session.training_fingerprint(),
        )
        if not model.ready:
            raise ValueError("Training produced an invalid model")
        return model

    def fingerprint(self) -> str:
        if not self.ready:
            raise RuntimeError("Model is not trained")
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def predict(self, features: Iterable[float]) -> Prediction:
        if not self.ready:
            raise RuntimeError("Model is not trained")
        vector = np.asarray(list(features), dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != len(FEATURE_NAMES) or not np.isfinite(vector).all():
            raise ValueError(f"Expected {len(FEATURE_NAMES)} finite features")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        z = (vector - mean) / scale
        if not np.isfinite(z).all():
            raise ValueError("Prediction normalization produced non-finite values")
        distances: dict[str, float] = {}
        for label in self.labels:
            centroid = np.asarray(self.centroids[label], dtype=np.float64)
            distance = float(np.linalg.norm(z - centroid))
            if not math.isfinite(distance):
                raise ValueError("Prediction distance is non-finite")
            distances[label] = distance
        logits = {label: -distance for label, distance in distances.items()}
        maximum = max(logits.values())
        exp = {label: math.exp(value - maximum) for label, value in logits.items()}
        total = sum(exp.values()) or 1.0
        scores = {label: value / total for label, value in exp.items()}
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores.values()):
            raise ValueError("Prediction evidence is numerically invalid")
        label = max(scores, key=scores.get)
        return Prediction(label=label, confidence=float(scores[label]), scores=scores)

    def to_dict(self) -> dict:
        return {
            "schema": MODEL_SCHEMA,
            "algorithm": MODEL_ALGORITHM,
            "feature_schema": FEATURE_SCHEMA,
            "sample_rate_hz": int(FeatureExtractor.SAMPLE_RATE),
            "feature_names": list(FEATURE_NAMES),
            "labels": self.labels,
            "mean": self.mean,
            "scale": self.scale,
            "centroids": self.centroids,
            "training_fingerprint": self.training_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CentroidModel":
        if type(payload.get("schema")) is not int or payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("Unsupported model schema")
        if not isinstance(payload.get("algorithm"), str) or payload.get("algorithm") != MODEL_ALGORITHM:
            raise ValueError("Unsupported model algorithm")
        if not isinstance(payload.get("feature_schema"), str) or payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Model feature contract does not match this application")
        if type(payload.get("sample_rate_hz")) is not int or payload.get("sample_rate_hz") != int(FeatureExtractor.SAMPLE_RATE):
            raise ValueError("Model sample rate does not match this application")
        if not isinstance(payload.get("feature_names"), list) or tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Model feature schema does not match this application")
        labels_raw = payload.get("labels")
        centroids_raw = payload.get("centroids")
        training_fingerprint = payload.get("training_fingerprint")
        if not isinstance(labels_raw, list) or any(not isinstance(value, str) for value in labels_raw):
            raise ValueError("Model labels must be strings")
        if not isinstance(centroids_raw, dict) or any(not isinstance(key, str) for key in centroids_raw):
            raise ValueError("Model centroids are invalid")
        if not isinstance(training_fingerprint, str):
            raise ValueError("Model training fingerprint is invalid")
        mean = _finite_float_list(payload.get("mean"), name="Model mean")
        scale = _finite_float_list(payload.get("scale"), name="Model scale")
        centroids = {
            key: _finite_float_list(values, name=f"Centroid {key}")
            for key, values in centroids_raw.items()
        }
        model = cls(
            labels=list(labels_raw),
            mean=mean,
            scale=scale,
            centroids=centroids,
            training_fingerprint=training_fingerprint,
        )
        n = len(FEATURE_NAMES)
        if (
            len(model.labels) < 2
            or len(set(model.labels)) != len(model.labels)
            or any(
                not label.strip()
                or label != label.strip()
                or len(label) > 80
                or any(ord(ch) < 32 for ch in label)
                for label in model.labels
            )
        ):
            raise ValueError("Model labels are invalid")
        if len(model.mean) != n or len(model.scale) != n:
            raise ValueError("Model normalization dimensions are invalid")
        if len(model.training_fingerprint) != 64 or any(
            ch not in "0123456789abcdef" for ch in model.training_fingerprint
        ):
            raise ValueError("Model training fingerprint is invalid")
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
    trials_per_label: dict[str, int]
    correct_per_label: dict[str, int]
    model_fingerprint: str
    validation_fingerprint: str


def validate_model(model: CentroidModel, session: TrainingSession) -> ValidationResult:
    # Timed calibration creates several nearby epochs per trial. Validation is
    # therefore reported once per independent trial rather than inflating the
    # score with overlapping epochs.
    trials = session.trial_vectors()
    if not model.ready or not trials:
        return ValidationResult(0, 0, 0.0, {}, {}, {}, "", "")
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
    return ValidationResult(
        len(trials),
        hits,
        hits / len(trials),
        per_label,
        dict(totals),
        {label: correct.get(label, 0) for label in totals},
        model.fingerprint(),
        session.training_fingerprint(),
    )


class ModelStore:
    """Strict persistence for model, calibration session and metadata files."""

    def __init__(self, root: Path | None = None, *, owner_name: str = "") -> None:
        self.root = root or app_data_dir()
        self.owner_name = " ".join(unicodedata.normalize("NFKC", str(owner_name)).strip().split())

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
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        safe = " ".join(name.strip().split())
        if not safe:
            raise ValueError("Name cannot be empty")
        if len(safe) > 96:
            raise ValueError("Name is too long")
        if any(not (ch.isalnum() or ch in "-_ ") for ch in safe):
            raise ValueError("Name contains unsupported characters")
        return safe

    def list_models(self) -> list[str]:
        if not self.model_dir.exists():
            return []
        return sorted(path.stem for path in self.model_dir.glob("*.json"))

    def save(self, name: str, model: CentroidModel) -> Path:
        if not model.ready:
            raise ValueError("Cannot persist an untrained or malformed model")
        safe = self.normalize_name(name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / f"{safe}.json"
        payload = model.to_dict()
        payload["owner"] = self.owner_name
        self._atomic_json(path, payload, max_bytes=MODEL_MAX_BYTES)
        return path

    def load(self, name: str) -> CentroidModel:
        safe = self.normalize_name(name)
        path = self.model_dir / f"{safe}.json"
        payload = read_json_object(path, max_bytes=MODEL_MAX_BYTES)
        self._validate_owner(payload)
        expected = set(CentroidModel().to_dict()) | {"owner"}
        if set(payload) != expected:
            raise ValueError("Model JSON fields do not match the Version 1 contract")
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
        if not isinstance(session.name, str):
            raise ValueError("Session name is invalid")
        canonical_session_name = " ".join(session.name.strip().split())
        if not canonical_session_name or session.name != canonical_session_name or len(session.name) > 120:
            raise ValueError("Session name is invalid")
        if not isinstance(session.owner, str):
            raise ValueError("Session owner is invalid")
        if self.owner_name and session.owner and self._canonical_owner(session.owner) != self._canonical_owner(self.owner_name):
            raise ValueError("Session owner does not match the active user")

        # Validate the in-memory object through the same Version 1 contract used
        # on restore. Dataclass fields are mutable and callers must not be able
        # to persist an artifact that this application would immediately reject.
        validated = TrainingSession(session.name, owner=session.owner)
        seen_epochs: set[tuple[str, str, int]] = set()
        for sample in session.samples:
            if not isinstance(sample, TrainingSample):
                raise ValueError("Session contains an invalid sample object")
            if not isinstance(sample.label, str) or sample.label != sample.label.strip():
                raise ValueError("Session label is invalid")
            if not isinstance(sample.trial_id, str) or not sample.trial_id or sample.trial_id != sample.trial_id.strip():
                raise ValueError("Session trial id is invalid")
            if not isinstance(sample.features, list) or any(not _is_json_number(value) for value in sample.features):
                raise ValueError("Session features must be a numeric JSON array")
            if not _is_json_number(sample.timestamp) or not _is_json_number(sample.epoch_seconds):
                raise ValueError("Session timing values must be JSON numbers")
            if type(sample.epoch_index) is not int:
                raise ValueError("Session epoch index must be an integer")
            key = (sample.label, sample.trial_id, sample.epoch_index)
            if key in seen_epochs:
                raise ValueError("Session contains a duplicate trial epoch")
            seen_epochs.add(key)
            validated.add(
                sample.label,
                sample.features,
                float(sample.timestamp),
                trial_id=sample.trial_id,
                epoch_index=sample.epoch_index,
                epoch_seconds=float(sample.epoch_seconds),
            )

        safe = self.normalize_name(name)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"{safe}.json"
        payload = {
            "schema": SESSION_SCHEMA,
            "sampling": SESSION_SAMPLING,
            "feature_schema": FEATURE_SCHEMA,
            "sample_rate_hz": int(FeatureExtractor.SAMPLE_RATE),
            "name": validated.name,
            "owner": validated.owner or self.owner_name,
            "feature_names": list(FEATURE_NAMES),
            "samples": [asdict(sample) for sample in validated.samples],
        }
        self._atomic_json(path, payload, max_bytes=SESSION_MAX_BYTES)
        return path

    def load_session(self, name: str) -> TrainingSession:
        safe = self.normalize_name(name)
        path = self.session_dir / f"{safe}.json"
        payload = read_json_object(path, max_bytes=SESSION_MAX_BYTES)
        self._validate_owner(payload)
        if type(payload.get("schema")) is not int or payload.get("schema") != SESSION_SCHEMA:
            raise ValueError("Unsupported calibration session schema")
        if not isinstance(payload.get("sampling"), str) or payload.get("sampling") != SESSION_SAMPLING:
            raise ValueError("Unsupported calibration sampling strategy")
        if not isinstance(payload.get("feature_schema"), str) or payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Session feature contract does not match this application")
        if type(payload.get("sample_rate_hz")) is not int or payload.get("sample_rate_hz") != int(FeatureExtractor.SAMPLE_RATE):
            raise ValueError("Session sample rate does not match this application")
        if not isinstance(payload.get("feature_names"), list) or tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Session feature schema does not match this application")
        expected_top = {
            "schema", "sampling", "feature_schema", "sample_rate_hz", "name",
            "owner", "feature_names", "samples",
        }
        if set(payload) != expected_top:
            raise ValueError("Session JSON fields do not match the Version 1 contract")
        items = payload.get("samples")
        if not isinstance(items, list):
            raise ValueError("Session samples must be a list")
        session_name = payload.get("name")
        owner = payload.get("owner")
        if not isinstance(session_name, str):
            raise ValueError("Session name is invalid")
        canonical_session_name = " ".join(session_name.strip().split())
        if not canonical_session_name or session_name != canonical_session_name or len(session_name) > 120:
            raise ValueError("Session name is invalid")
        if not isinstance(owner, str):
            raise ValueError("Session owner is invalid")
        session = TrainingSession(session_name, owner=owner)
        sample_fields = {"label", "features", "timestamp", "trial_id", "epoch_index", "epoch_seconds"}
        seen_epochs: set[tuple[str, str, int]] = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != sample_fields:
                raise ValueError("Session sample fields do not match the Version 1 contract")
            label = item.get("label")
            trial_id = item.get("trial_id")
            features = item.get("features")
            timestamp = item.get("timestamp")
            epoch_index_raw = item.get("epoch_index")
            epoch_seconds_raw = item.get("epoch_seconds")
            if (
                not isinstance(label, str)
                or label != label.strip()
                or not isinstance(trial_id, str)
                or trial_id != trial_id.strip()
                or not trial_id
            ):
                raise ValueError("Session label/trial id is invalid")
            if not isinstance(features, list) or any(not _is_json_number(value) for value in features):
                raise ValueError("Session features must be a numeric JSON array")
            if not _is_json_number(timestamp) or not _is_json_number(epoch_seconds_raw):
                raise ValueError("Session timing values must be JSON numbers")
            if not isinstance(epoch_index_raw, int) or isinstance(epoch_index_raw, bool):
                raise ValueError("Session epoch index must be an integer")
            epoch_index = epoch_index_raw
            key = (label.strip(), trial_id.strip(), epoch_index)
            if key in seen_epochs:
                raise ValueError("Session contains a duplicate trial epoch")
            seen_epochs.add(key)
            session.add(
                label,
                features,
                float(timestamp),
                trial_id=trial_id,
                epoch_index=epoch_index,
                epoch_seconds=float(epoch_seconds_raw),
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
        self._atomic_json(
            path, {"schema": 1, "owner": self.owner_name, "payload": payload}, max_bytes=METADATA_MAX_BYTES
        )
        return path

    def load_metadata(self, name: str) -> dict:
        safe = self.normalize_name(name)
        path = self.metadata_dir / f"{safe}.json"
        if not path.exists():
            return {}
        payload = read_json_object(path, max_bytes=METADATA_MAX_BYTES)
        self._validate_owner(payload)
        if set(payload) != {"schema", "owner", "payload"}:
            raise ValueError("Metadata JSON fields do not match the Version 1 contract")
        if type(payload.get("schema")) is not int or payload.get("schema") != 1 or not isinstance(payload.get("payload"), dict):
            raise ValueError("Unsupported metadata schema")
        return dict(payload["payload"])

    def delete_metadata(self, name: str) -> None:
        safe = self.normalize_name(name)
        path = self.metadata_dir / f"{safe}.json"
        if path.exists():
            path.unlink()

    @staticmethod
    def _canonical_owner(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split()).casefold()

    def _validate_owner(self, payload: dict) -> None:
        if not self.owner_name:
            return
        owner = payload.get("owner")
        if not isinstance(owner, str) or self._canonical_owner(owner) != self._canonical_owner(self.owner_name):
            raise ValueError("Persisted artifact owner does not match the active user")

    @staticmethod
    def _atomic_json(path: Path, payload: dict, *, max_bytes: int) -> None:
        atomic_write_json(path, payload, max_bytes=max_bytes)
