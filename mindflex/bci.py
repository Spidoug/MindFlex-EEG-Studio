from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .parser import EEG_BANDS
from .signal_processing import band_power, preprocess_eeg, welch_psd
from .settings import (
    BCI_EPOCH_SAMPLES,
    BCI_MIN_EVIDENCE_OVER_CHANCE,
    BCI_STABILIZATION_DECISIONS,
    BCI_STEP_SAMPLES,
    BCI_WELCH_SEGMENT_SAMPLES,
    BCI_WELCH_STEP_SAMPLES,
    EEG_HIGHPASS_HZ,
    EEG_LOWPASS_HZ,
    MINDFLEX_RAW_SAMPLE_RATE,
    app_data_dir,
)
from .storage import atomic_write_json, read_json_object

# Version 1 has one BCI data path only: the fixed 512 Hz RAW stream. TGAM band
# summaries/eSense remain available for monitoring but never feed the model.
FEATURE_NAMES = (
    *tuple(f"log_{name}" for name in EEG_BANDS),
    "rel_delta",
    "rel_theta",
    "rel_alpha",
    "rel_beta",
    "rel_gamma",
    "log_theta_beta",
    "log_alpha_beta",
    "spectral_entropy",
    "spectral_centroid",
    "log_total_power",
    "engagement_index",
    "spectral_edge_90",
    "spectral_flatness",
    "hjorth_mobility",
    "hjorth_complexity",
    "log_line_length",
)
MODEL_SCHEMA = 1
SESSION_SCHEMA = 1
FEATURE_SCHEMA = "mindflex-v1-raw24-welch"
MODEL_ALGORITHM = "mindflex-v1-balanced-shrinkage-lda"
SESSION_SAMPLING = "mindflex-v1-raw-fixed-grid"
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


@dataclass(frozen=True, slots=True)
class BCIWindow:
    """One exact RAW interval used by every BCI mode."""

    start_sample: int
    end_sample: int
    index: int


def bci_window(boundary_sample: int, index: int) -> BCIWindow:
    if type(boundary_sample) is not int or boundary_sample < 0:
        raise ValueError("BCI boundary must be a non-negative integer sample index")
    if type(index) is not int or index < 0:
        raise ValueError("BCI window index must be a non-negative integer")
    start = boundary_sample + index * BCI_STEP_SAMPLES
    return BCIWindow(start, start + BCI_EPOCH_SAMPLES, index)


def latest_bci_window(total_samples: int, boundary_sample: int = 0) -> BCIWindow | None:
    """Return the newest complete fixed-grid window after ``boundary_sample``."""
    if type(total_samples) is not int or total_samples < 0:
        raise ValueError("RAW total must be a non-negative integer")
    if type(boundary_sample) is not int or boundary_sample < 0 or boundary_sample > total_samples:
        raise ValueError("BCI boundary is outside the current RAW stream")
    available = total_samples - boundary_sample - BCI_EPOCH_SAMPLES
    if available < 0:
        return None
    return bci_window(boundary_sample, available // BCI_STEP_SAMPLES)


@dataclass(slots=True)
class TrainingSample:
    label: str
    features: list[float]
    timestamp: float
    trial_id: str
    epoch_index: int
    raw_start: int
    raw_end: int


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
        return len({sample.trial_id for sample in self.samples if sample.label == label})

    @property
    def total_trials(self) -> int:
        return len({(sample.label, sample.trial_id) for sample in self.samples})

    def add(
        self,
        label: str,
        features: Iterable[float],
        timestamp: float,
        *,
        trial_id: str,
        epoch_index: int,
        raw_start: int,
        raw_end: int,
    ) -> None:
        if not isinstance(label, str):
            raise ValueError("Label must be a string")
        clean_label = label.strip()
        if not clean_label or clean_label != label or len(clean_label) > 80 or any(ord(ch) < 32 for ch in clean_label):
            raise ValueError("Label is invalid")
        try:
            vector = [float(value) for value in features]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Features must be numeric") from exc
        if len(vector) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} finite features")
        if isinstance(timestamp, bool):
            raise ValueError("Timestamp must be numeric")
        try:
            ts = float(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Timestamp must be numeric") from exc
        if not math.isfinite(ts):
            raise ValueError("Timestamp must be finite")
        if not isinstance(trial_id, str) or not trial_id or trial_id != trial_id.strip():
            raise ValueError("Trial id is required")
        if len(trial_id) > 128 or any(ord(ch) < 32 for ch in trial_id):
            raise ValueError("Trial id is invalid")
        if type(epoch_index) is not int or epoch_index < 0:
            raise ValueError("Epoch index must be a non-negative integer")
        if type(raw_start) is not int or type(raw_end) is not int or raw_start < 0:
            raise ValueError("RAW sample bounds must be non-negative integers")
        if raw_end - raw_start != BCI_EPOCH_SAMPLES:
            raise ValueError(f"Every BCI epoch must contain exactly {BCI_EPOCH_SAMPLES} RAW samples")
        self.samples.append(
            TrainingSample(
                label=clean_label,
                features=vector,
                timestamp=ts,
                trial_id=trial_id,
                epoch_index=epoch_index,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        )

    def clear(self) -> None:
        self.samples.clear()

    def _validated_groups(self) -> dict[tuple[str, str], list[TrainingSample]]:
        groups: dict[tuple[str, str], list[TrainingSample]] = {}
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
            if (
                isinstance(sample.timestamp, bool)
                or not isinstance(sample.timestamp, (int, float))
                or not math.isfinite(float(sample.timestamp))
            ):
                raise ValueError("Training session contains an invalid timestamp")
            if type(sample.raw_start) is not int or type(sample.raw_end) is not int or sample.raw_start < 0:
                raise ValueError("Training session contains invalid RAW sample bounds")
            if sample.raw_end - sample.raw_start != BCI_EPOCH_SAMPLES:
                raise ValueError("Training session contains an incompatible RAW epoch")
            if not isinstance(sample.features, list) or len(sample.features) != len(FEATURE_NAMES):
                raise ValueError("Training session contains an invalid feature vector")
            if any(not _is_json_number(value) or not math.isfinite(float(value)) for value in sample.features):
                raise ValueError("Training session contains non-finite or non-numeric features")
            groups.setdefault((sample.label, sample.trial_id), []).append(sample)
        return groups

    def trial_matrices(self) -> list[tuple[str, str, np.ndarray]]:
        """Return every independent trial as its exact validated epoch matrix."""
        groups = self._validated_groups()
        result: list[tuple[str, str, np.ndarray]] = []
        for (label, trial_id), samples in sorted(groups.items()):
            bases = {sample.raw_start - sample.epoch_index * BCI_STEP_SAMPLES for sample in samples}
            if len(bases) != 1:
                raise ValueError("Training trial does not follow the fixed BCI sample grid")
            ordered = sorted(samples, key=lambda sample: sample.epoch_index)
            matrix = np.asarray([sample.features for sample in ordered], dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
                raise ValueError("Training session contains an invalid trial matrix")
            result.append((label, trial_id, matrix))
        return result

    def trial_vectors(self) -> list[tuple[str, str, np.ndarray]]:
        """Return one robust representative vector per independent trial."""
        result: list[tuple[str, str, np.ndarray]] = []
        for label, trial_id, matrix in self.trial_matrices():
            vector = np.median(matrix, axis=0)
            if not np.isfinite(vector).all():
                raise ValueError("Training trial representative is numerically invalid")
            result.append((label, trial_id, vector))
        return result

    def training_fingerprint(self) -> str:
        groups = self._validated_groups()
        if not groups:
            raise ValueError("Training session has no independent trials")
        payload: list[list[object]] = []
        for (label, trial_id), samples in sorted(groups.items()):
            for sample in sorted(samples, key=lambda item: item.epoch_index):
                payload.append([
                    label,
                    trial_id,
                    sample.epoch_index,
                    sample.raw_start,
                    sample.raw_end,
                    [float(value) for value in sample.features],
                ])
        canonical = json.dumps(payload, sort_keys=False, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FeatureExtractor:
    """Robust single-channel time/frequency features from one exact RAW epoch."""

    SAMPLE_RATE = float(MINDFLEX_RAW_SAMPLE_RATE)
    MIN_SAMPLES = BCI_EPOCH_SAMPLES
    WELCH_SEGMENT_SAMPLES = BCI_WELCH_SEGMENT_SAMPLES
    WELCH_STEP_SAMPLES = BCI_WELCH_STEP_SAMPLES
    MIN_ANALYSIS_HZ = EEG_HIGHPASS_HZ
    MAX_ANALYSIS_HZ = EEG_LOWPASS_HZ

    @staticmethod
    def _hjorth(x: np.ndarray) -> tuple[float, float]:
        activity = float(np.var(x))
        dx = np.diff(x)
        ddx = np.diff(dx)
        d_activity = float(np.var(dx)) if dx.size else 0.0
        dd_activity = float(np.var(ddx)) if ddx.size else 0.0
        eps = 1e-12
        mobility = math.sqrt(max(0.0, d_activity) / max(activity, eps))
        derivative_mobility = math.sqrt(max(0.0, dd_activity) / max(d_activity, eps))
        complexity = derivative_mobility / max(mobility, eps)
        if not math.isfinite(mobility) or not math.isfinite(complexity):
            raise ValueError("Hjorth parameters are invalid")
        return mobility, complexity

    @classmethod
    def from_raw(cls, samples: Iterable[float]) -> np.ndarray:
        """Extract the single Version 1 feature vector from exactly one RAW epoch."""
        x = np.asarray(list(samples), dtype=np.float64)
        if x.ndim != 1 or x.size != BCI_EPOCH_SAMPLES:
            raise ValueError(f"BCI input must contain exactly {BCI_EPOCH_SAMPLES} RAW samples")
        if not np.isfinite(x).all():
            raise ValueError("BCI RAW epoch contains non-finite samples")
        x = preprocess_eeg(x)
        freqs, power = welch_psd(
            x,
            sample_rate=cls.SAMPLE_RATE,
            segment_samples=cls.WELCH_SEGMENT_SAMPLES,
            step_samples=cls.WELCH_STEP_SAMPLES,
        )
        analysis_mask = (freqs >= cls.MIN_ANALYSIS_HZ) & (freqs <= cls.MAX_ANALYSIS_HZ)
        spectral = power[analysis_mask]
        spectral_freqs = freqs[analysis_mask]
        if spectral.size < 8:
            raise ValueError("BCI spectrum has insufficient usable bins")
        spacing = float(freqs[1] - freqs[0])
        total = float(np.sum(spectral) * spacing)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("RAW epoch has no usable spectral energy")
        eps = max(1e-18, total * 1e-12)

        bands = {
            "delta": band_power(freqs, power, 0.5, 4.0),
            "theta": band_power(freqs, power, 4.0, 8.0),
            "low_alpha": band_power(freqs, power, 8.0, 10.0),
            "high_alpha": band_power(freqs, power, 10.0, 13.0),
            "low_beta": band_power(freqs, power, 13.0, 20.0),
            "high_beta": band_power(freqs, power, 20.0, 30.0),
            "low_gamma": band_power(freqs, power, 30.0, 40.0),
            "mid_gamma": band_power(freqs, power, 40.0, cls.MAX_ANALYSIS_HZ + 1e-9),
        }
        if set(bands) != set(EEG_BANDS) or any(
            not math.isfinite(value) or value < 0.0 for value in bands.values()
        ):
            raise ValueError("Internal spectral powers are invalid")

        delta = bands["delta"]
        theta = bands["theta"]
        alpha = bands["low_alpha"] + bands["high_alpha"]
        beta = bands["low_beta"] + bands["high_beta"]
        gamma = bands["low_gamma"] + bands["mid_gamma"]
        band_total = delta + theta + alpha + beta + gamma
        if band_total <= 0.0:
            raise ValueError("RAW epoch has no usable band power")

        probabilities = spectral / max(float(np.sum(spectral)), 1e-18)
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log(positive))) / math.log(max(2, spectral.size))
        centroid = float(np.sum(spectral_freqs * probabilities)) / cls.MAX_ANALYSIS_HZ
        cumulative = np.cumsum(spectral)
        edge_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.90, side="left"))
        edge_index = min(max(edge_index, 0), spectral_freqs.size - 1)
        spectral_edge = float(spectral_freqs[edge_index]) / cls.MAX_ANALYSIS_HZ
        spectral_flatness = float(
            math.exp(float(np.mean(np.log(spectral + eps)))) / (float(np.mean(spectral)) + eps)
        )
        spectral_flatness = max(0.0, min(1.0, spectral_flatness))
        mobility, complexity = cls._hjorth(x)
        line_length = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0
        engagement = (beta + gamma + eps) / (theta + alpha + eps)

        vector = np.asarray(
            [
                *[math.log1p(bands[name]) for name in EEG_BANDS],
                delta / band_total,
                theta / band_total,
                alpha / band_total,
                beta / band_total,
                gamma / band_total,
                math.log((theta + eps) / (beta + eps)),
                math.log((alpha + eps) / (beta + eps)),
                entropy,
                centroid,
                math.log1p(total),
                math.log(engagement),
                spectral_edge,
                spectral_flatness,
                mobility,
                complexity,
                math.log1p(line_length),
            ],
            dtype=np.float64,
        )
        if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise ValueError("BCI feature vector is invalid")
        return vector


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    features: tuple[float, ...]
    start_sample: int
    end_sample: int
    index: int


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


def _decision_from_scores(labels: Iterable[str], scores: dict[str, float]) -> Prediction | None:
    """Resolve posterior scores with the single global evidence-over-chance rule."""
    clean = tuple(dict.fromkeys(str(label) for label in labels if str(label)))
    if len(clean) < 2:
        raise ValueError("At least two labels are required")
    row = {label: max(0.0, float(scores.get(label, 0.0))) for label in clean}
    total = sum(row.values())
    if total <= 0.0 or not math.isfinite(total):
        return None
    normalized = {label: value / total for label, value in row.items()}
    label = max(normalized, key=normalized.get)
    chance = 1.0 / len(clean)
    evidence = (float(normalized[label]) - chance) / (1.0 - chance)
    confidence = max(0.0, min(1.0, evidence))
    if confidence < BCI_MIN_EVIDENCE_OVER_CHANCE:
        return None
    return Prediction(label=label, confidence=confidence, scores=normalized)


class PredictionStabilizer:
    """One temporal decision rule shared by every live BCI output."""

    def __init__(self, labels: Iterable[str]) -> None:
        clean = tuple(dict.fromkeys(str(label) for label in labels if str(label)))
        if len(clean) < 2:
            raise ValueError("At least two labels are required")
        self.labels = clean
        self._scores: deque[dict[str, float]] = deque(maxlen=BCI_STABILIZATION_DECISIONS)

    def reset(self) -> None:
        self._scores.clear()

    def push(self, prediction: Prediction | None) -> Prediction | None:
        if prediction is None:
            return None
        row = {label: max(0.0, float(prediction.scores.get(label, 0.0))) for label in self.labels}
        total = sum(row.values())
        if total <= 0.0 or not math.isfinite(total):
            return None
        normalized = {label: value / total for label, value in row.items()}
        self._scores.append(normalized)
        if len(self._scores) < BCI_STABILIZATION_DECISIONS:
            return None
        averaged = {
            label: sum(item[label] for item in self._scores) / len(self._scores)
            for label in self.labels
        }
        return _decision_from_scores(self.labels, averaged)


class TrialDecisionAccumulator:
    """Canonical trial evaluator used by validation and every cue-based test.

    Raw model predictions enter here exactly as they do in live control. The
    accumulator owns the same global stabilizer and resolves the complete trial
    from stabilized posterior evidence. No test is allowed to invent a second
    confidence or voting rule.
    """

    def __init__(self, labels: Iterable[str]) -> None:
        clean = tuple(dict.fromkeys(str(label) for label in labels if str(label)))
        if len(clean) < 2:
            raise ValueError("At least two labels are required")
        self.labels = clean
        self._stabilizer = PredictionStabilizer(clean)
        self._score_sum = {label: 0.0 for label in clean}
        self.raw_observations = 0
        self.stable_observations = 0

    def reset(self) -> None:
        self._stabilizer.reset()
        self._score_sum = {label: 0.0 for label in self.labels}
        self.raw_observations = 0
        self.stable_observations = 0

    def push(self, prediction: Prediction | None) -> Prediction | None:
        if prediction is None:
            return None
        self.raw_observations += 1
        stable = self._stabilizer.push(prediction)
        if stable is None:
            return None
        for label in self.labels:
            self._score_sum[label] += max(0.0, float(stable.scores.get(label, 0.0)))
        self.stable_observations += 1
        return stable

    def resolve(self) -> Prediction | None:
        if self.stable_observations <= 0:
            return None
        averages = {
            label: self._score_sum[label] / self.stable_observations
            for label in self.labels
        }
        return _decision_from_scores(self.labels, averages)


@dataclass(slots=True)
class BCIModel:
    """Class-balanced shrinkage LDA for correlated single-channel EEG features."""

    labels: list[str] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    class_means: dict[str, list[float]] = field(default_factory=dict)
    precision: list[list[float]] = field(default_factory=list)
    shrinkage: float = 0.0
    distance_scale: float = 1.0
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
        if len(self.mean) != n or len(self.scale) != n or set(self.class_means) != set(self.labels):
            return False
        if len(self.precision) != n or any(not isinstance(row, list) or len(row) != n for row in self.precision):
            return False
        if len(self.training_fingerprint) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.training_fingerprint
        ):
            return False
        values: list[object] = [*self.mean, *self.scale, self.shrinkage, self.distance_scale]
        for label in self.labels:
            center = self.class_means.get(label)
            if not isinstance(center, list) or len(center) != n:
                return False
            values.extend(center)
        for row in self.precision:
            values.extend(row)
        if not all(_is_json_number(value) and math.isfinite(float(value)) for value in values):
            return False
        if any(float(value) <= 0.0 for value in self.scale):
            return False
        if not 0.0 <= float(self.shrinkage) <= 1.0 or float(self.distance_scale) <= 0.0:
            return False
        matrix = np.asarray(self.precision, dtype=np.float64)
        return bool(
            np.isfinite(matrix).all()
            and np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-8)
            and np.all(np.diag(matrix) > 0.0)
        )

    @classmethod
    def train(cls, session: TrainingSession) -> "BCIModel":
        trials = session.trial_matrices()
        labels = sorted({label for label, _trial_id, _matrix in trials})
        if len(labels) < 2:
            raise ValueError("At least two classes with independent trials are required")
        grouped: dict[str, list[np.ndarray]] = {label: [] for label in labels}
        for label, _trial_id, matrix in trials:
            grouped[label].append(matrix)
        if any(not grouped[label] for label in labels):
            raise ValueError("Every class requires at least one independent trial")

        # Equal class and equal trial influence. Overlapping epochs improve the
        # estimate inside a trial but never give a longer/more complete trial
        # more weight than another trial.
        raw_class_means: dict[str, np.ndarray] = {}
        for label in labels:
            trial_means = np.asarray([matrix.mean(axis=0) for matrix in grouped[label]], dtype=np.float64)
            raw_class_means[label] = trial_means.mean(axis=0)
        mean = np.asarray([raw_class_means[label] for label in labels], dtype=np.float64).mean(axis=0)

        within_rows: list[np.ndarray] = []
        total_rows: list[np.ndarray] = []
        for label in labels:
            class_within: list[np.ndarray] = []
            class_total: list[np.ndarray] = []
            for matrix in grouped[label]:
                class_within.append(np.mean((matrix - raw_class_means[label]) ** 2, axis=0))
                class_total.append(np.mean((matrix - mean) ** 2, axis=0))
            within_rows.append(np.mean(np.asarray(class_within), axis=0))
            total_rows.append(np.mean(np.asarray(class_total), axis=0))
        within_variance = np.mean(np.asarray(within_rows, dtype=np.float64), axis=0)
        total_variance = np.mean(np.asarray(total_rows, dtype=np.float64), axis=0)
        variance = np.maximum(within_variance, total_variance * 0.03)
        variance = np.maximum(variance, 1e-8)
        scale = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Training normalization is numerically invalid")

        class_means = {label: (raw_class_means[label] - mean) / scale for label in labels}
        covariance_rows: list[np.ndarray] = []
        for label in labels:
            trial_covariances: list[np.ndarray] = []
            center = class_means[label]
            for matrix in grouped[label]:
                z = (matrix - mean) / scale
                residual = z - center
                trial_covariances.append((residual.T @ residual) / max(1, residual.shape[0]))
            covariance_rows.append(np.mean(np.asarray(trial_covariances), axis=0))
        covariance = np.mean(np.asarray(covariance_rows), axis=0)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.isfinite(covariance).all():
            raise ValueError("Training covariance is numerically invalid")

        feature_count = len(FEATURE_NAMES)
        independent_trials = len(trials)
        shrinkage = max(0.25, min(0.85, feature_count / (feature_count + independent_trials)))
        diagonal = np.diag(np.diag(covariance))
        regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        regularized += np.eye(feature_count, dtype=np.float64) * 0.05
        precision = np.linalg.pinv(regularized, hermitian=True)
        precision = 0.5 * (precision + precision.T)
        if not np.isfinite(precision).all() or np.any(np.diag(precision) <= 0.0):
            raise ValueError("Training precision matrix is invalid")

        # Calibrate the distance-to-posterior temperature from independent
        # trial representatives. Weakly separated training data therefore
        # stays uncertain instead of becoming artificially overconfident.
        margins: list[float] = []
        for label, _trial_id, matrix in trials:
            representative = np.median(matrix, axis=0)
            z = (representative - mean) / scale
            distances = {}
            for candidate in labels:
                diff = z - class_means[candidate]
                distances[candidate] = max(0.0, float(diff @ precision @ diff))
            correct_distance = distances[label]
            other_distance = min(value for candidate, value in distances.items() if candidate != label)
            margins.append(other_distance - correct_distance)
        positive = [margin for margin in margins if math.isfinite(margin) and margin > 0.0]
        median_margin = float(np.median(positive)) if positive else 0.0
        distance_scale = max(feature_count * 0.10, median_margin * 0.50, 1e-6)

        model = cls(
            labels=labels,
            mean=mean.tolist(),
            scale=scale.tolist(),
            class_means={label: class_means[label].tolist() for label in labels},
            precision=precision.tolist(),
            shrinkage=float(shrinkage),
            distance_scale=float(distance_scale),
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
        precision = np.asarray(self.precision, dtype=np.float64)
        z = (vector - mean) / scale
        if not np.isfinite(z).all():
            raise ValueError("Prediction normalization produced non-finite values")
        logits: dict[str, float] = {}
        for label in self.labels:
            center = np.asarray(self.class_means[label], dtype=np.float64)
            diff = z - center
            distance = max(0.0, float(diff @ precision @ diff))
            if not math.isfinite(distance):
                raise ValueError("Prediction distance is non-finite")
            logits[label] = -0.5 * distance / self.distance_scale
        maximum = max(logits.values())
        exponentials = {label: math.exp(value - maximum) for label, value in logits.items()}
        total = sum(exponentials.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Prediction evidence is invalid")
        scores = {label: value / total for label, value in exponentials.items()}
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores.values()):
            raise ValueError("Prediction evidence is numerically invalid")
        label = max(scores, key=scores.get)
        chance = 1.0 / len(scores)
        evidence = (float(scores[label]) - chance) / (1.0 - chance)
        return Prediction(label=label, confidence=max(0.0, min(1.0, evidence)), scores=scores)

    def to_dict(self) -> dict:
        return {
            "schema": MODEL_SCHEMA,
            "algorithm": MODEL_ALGORITHM,
            "feature_schema": FEATURE_SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "labels": self.labels,
            "mean": self.mean,
            "scale": self.scale,
            "class_means": self.class_means,
            "precision": self.precision,
            "shrinkage": self.shrinkage,
            "distance_scale": self.distance_scale,
            "training_fingerprint": self.training_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BCIModel":
        expected_top = {
            "schema", "algorithm", "feature_schema", "feature_names", "labels", "mean", "scale",
            "class_means", "precision", "shrinkage", "distance_scale", "training_fingerprint",
        }
        if not isinstance(payload, dict) or set(payload) != expected_top:
            raise ValueError("Model JSON fields do not match the Version 1 format")
        if type(payload.get("schema")) is not int or payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("Unsupported model schema")
        if payload.get("algorithm") != MODEL_ALGORITHM or payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Model algorithm or feature format does not match this application")
        if not isinstance(payload.get("feature_names"), list) or tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Model feature schema does not match this application")
        labels_raw = payload.get("labels")
        means_raw = payload.get("class_means")
        precision_raw = payload.get("precision")
        fingerprint = payload.get("training_fingerprint")
        if not isinstance(labels_raw, list) or any(not isinstance(value, str) for value in labels_raw):
            raise ValueError("Model labels must be strings")
        if not isinstance(means_raw, dict) or any(not isinstance(key, str) for key in means_raw):
            raise ValueError("Model class means are invalid")
        if not isinstance(precision_raw, list):
            raise ValueError("Model precision matrix is invalid")
        if not isinstance(fingerprint, str):
            raise ValueError("Model training fingerprint is invalid")
        mean = _finite_float_list(payload.get("mean"), name="Model mean")
        scale = _finite_float_list(payload.get("scale"), name="Model scale")
        class_means = {
            key: _finite_float_list(values, name=f"Class mean {key}")
            for key, values in means_raw.items()
        }
        precision = [_finite_float_list(row, name="Model precision row") for row in precision_raw]
        shrinkage = payload.get("shrinkage")
        distance_scale = payload.get("distance_scale")
        if not _is_json_number(shrinkage) or not _is_json_number(distance_scale):
            raise ValueError("Model regularization parameters are invalid")
        model = cls(
            labels=list(labels_raw),
            mean=mean,
            scale=scale,
            class_means=class_means,
            precision=precision,
            shrinkage=float(shrinkage),
            distance_scale=float(distance_scale),
            training_fingerprint=fingerprint,
        )
        if not model.ready:
            raise ValueError("Model is malformed")
        return model


@dataclass(slots=True)
class ValidationResult:
    total: int
    correct: int
    decided: int
    accuracy: float
    balanced_accuracy: float
    decision_rate: float
    per_label: dict[str, float]
    trials_per_label: dict[str, int]
    correct_per_label: dict[str, int]
    decided_per_label: dict[str, int]
    model_fingerprint: str
    validation_fingerprint: str


def validate_model(model: BCIModel, session: TrainingSession) -> ValidationResult:
    """Score independent trials through the canonical live decision pipeline."""
    session.trial_vectors()
    if not model.ready or not session.samples:
        return ValidationResult(0, 0, 0, 0.0, 0.0, 0.0, {}, {}, {}, {}, "", "")
    unknown = set(session.labels) - set(model.labels)
    if unknown:
        raise ValueError(f"Validation contains labels outside the model: {', '.join(sorted(unknown))}")

    groups: dict[tuple[str, str], list[TrainingSample]] = {}
    for sample in session.samples:
        groups.setdefault((sample.label, sample.trial_id), []).append(sample)

    totals = {label: 0 for label in model.labels}
    correct = {label: 0 for label in model.labels}
    decided = {label: 0 for label in model.labels}
    hits = 0
    decisions = 0
    for (label, _trial_id), samples in sorted(groups.items()):
        evaluator = TrialDecisionAccumulator(model.labels)
        for sample in sorted(samples, key=lambda item: item.epoch_index):
            evaluator.push(model.predict(sample.features))
        resolved = evaluator.resolve()
        totals[label] += 1
        if resolved is not None:
            decisions += 1
            decided[label] += 1
            if resolved.label == label:
                hits += 1
                correct[label] += 1

    total_trials = sum(totals.values())
    per_label = {
        label: (correct[label] / totals[label]) if totals[label] else 0.0
        for label in model.labels
    }
    active = [per_label[label] for label in model.labels if totals[label] > 0]
    balanced = sum(active) / len(active) if active else 0.0
    return ValidationResult(
        total=total_trials,
        correct=hits,
        decided=decisions,
        accuracy=hits / total_trials if total_trials else 0.0,
        balanced_accuracy=balanced,
        decision_rate=decisions / total_trials if total_trials else 0.0,
        per_label=per_label,
        trials_per_label=totals,
        correct_per_label=correct,
        decided_per_label=decided,
        model_fingerprint=model.fingerprint(),
        validation_fingerprint=session.training_fingerprint(),
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

    def save(self, name: str, model: BCIModel) -> Path:
        if not model.ready:
            raise ValueError("Cannot persist an untrained or malformed model")
        safe = self.normalize_name(name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / f"{safe}.json"
        payload = model.to_dict()
        payload["owner"] = self.owner_name
        self._atomic_json(path, payload, max_bytes=MODEL_MAX_BYTES)
        return path

    def load(self, name: str) -> BCIModel:
        safe = self.normalize_name(name)
        path = self.model_dir / f"{safe}.json"
        payload = read_json_object(path, max_bytes=MODEL_MAX_BYTES)
        self._validate_owner(payload)
        expected = set(BCIModel().to_dict()) | {"owner"}
        if set(payload) != expected:
            raise ValueError("Model JSON fields do not match the Version 1 format")
        model_payload = dict(payload)
        model_payload.pop("owner")
        return BCIModel.from_dict(model_payload)

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

        # Validate the in-memory object through the same Version 1 format used
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
            if not _is_json_number(sample.timestamp):
                raise ValueError("Session timestamp must be a JSON number")
            if type(sample.epoch_index) is not int or type(sample.raw_start) is not int or type(sample.raw_end) is not int:
                raise ValueError("Session RAW indices must be integers")
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
                raw_start=sample.raw_start,
                raw_end=sample.raw_end,
            )

        safe = self.normalize_name(name)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"{safe}.json"
        payload = {
            "schema": SESSION_SCHEMA,
            "sampling": SESSION_SAMPLING,
            "feature_schema": FEATURE_SCHEMA,
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
            raise ValueError("Session feature format does not match this application")
        if not isinstance(payload.get("feature_names"), list) or tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Session feature schema does not match this application")
        expected_top = {
            "schema", "sampling", "feature_schema", "name",
            "owner", "feature_names", "samples",
        }
        if set(payload) != expected_top:
            raise ValueError("Session JSON fields do not match the Version 1 format")
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
        sample_fields = {"label", "features", "timestamp", "trial_id", "epoch_index", "raw_start", "raw_end"}
        seen_epochs: set[tuple[str, str, int]] = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != sample_fields:
                raise ValueError("Session sample fields do not match the Version 1 format")
            label = item.get("label")
            trial_id = item.get("trial_id")
            features = item.get("features")
            timestamp = item.get("timestamp")
            epoch_index_raw = item.get("epoch_index")
            raw_start = item.get("raw_start")
            raw_end = item.get("raw_end")
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
            if not _is_json_number(timestamp):
                raise ValueError("Session timestamp must be a JSON number")
            if not isinstance(epoch_index_raw, int) or isinstance(epoch_index_raw, bool):
                raise ValueError("Session epoch index must be an integer")
            if not isinstance(raw_start, int) or isinstance(raw_start, bool) or not isinstance(raw_end, int) or isinstance(raw_end, bool):
                raise ValueError("Session RAW sample bounds must be integers")
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
                raw_start=raw_start,
                raw_end=raw_end,
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
            raise ValueError("Metadata JSON fields do not match the Version 1 format")
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
