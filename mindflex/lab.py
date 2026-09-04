from __future__ import annotations

import json
import math
import os
import random
import tempfile
import threading
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .bci import (
    BCIModel,
    FeatureExtractor,
    TrainingSample,
    TrainingSession,
    ValidationResult,
    bci_window,
    validate_model,
)
from .controller import EEGController, EEGSnapshot
from .parser import EEG_BANDS
from .settings import BCI_EPOCH_SAMPLES, BCI_STEP_SAMPLES, MINDFLEX_RAW_SAMPLE_RATE

RECORDING_SCHEMA = 1
RECORDING_FORMAT = "mindflex-raw-session-v1"
RECORDING_EXTENSION = ".mfs"
TELEMETRY_COLUMNS = (
    "timestamp",
    "raw_total_samples",
    "poor_signal",
    "attention",
    "meditation",
    "blink",
    *EEG_BANDS,
    "packets",
    "bad_checksums",
    "dropped_bytes",
    "raw_rate_hz",
    "raw_spread",
    "raw_age",
)
MAX_EVENT_TEXT = 160


@dataclass(slots=True)
class ExperimentResult:
    name: str
    folds: int
    trial_count: int
    sample_count: int
    correct: int
    decided: int
    accuracy: float
    balanced_accuracy: float
    decision_rate: float
    per_label: dict[str, float] = field(default_factory=dict)
    trials_per_label: dict[str, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def weakest_label(self) -> str:
        return min(self.per_label, key=self.per_label.get) if self.per_label else ""


@dataclass(slots=True)
class ReplaySummary:
    raw_samples: int
    duration_seconds: float
    telemetry_frames: int
    event_count: int
    trial_count: int
    mean_attention: float | None
    mean_meditation: float | None
    active_bands: int
    mean_raw_rate_hz: float
    bad_checksums: int
    dropped_bytes: int
    profile: str
    purpose: str


@dataclass(slots=True)
class OfflineBCIResult:
    profile: str
    purpose: str
    trials: int
    epochs: int
    rejected_epochs: int
    experiment: ExperimentResult


@dataclass(frozen=True, slots=True)
class RecordingEvent:
    kind: str
    timestamp: float
    raw_sample: int
    profile: str = ""
    purpose: str = ""
    label: str = ""
    trial_id: str = ""
    phase: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "timestamp": self.timestamp,
            "raw_sample": self.raw_sample,
            "profile": self.profile,
            "purpose": self.purpose,
            "label": self.label,
            "trial_id": self.trial_id,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "RecordingEvent":
        expected = {"kind", "timestamp", "raw_sample", "profile", "purpose", "label", "trial_id", "phase"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Recording event fields are invalid")
        text: dict[str, str] = {}
        for key in ("kind", "profile", "purpose", "label", "trial_id", "phase"):
            value = payload[key]
            if not isinstance(value, str) or len(value) > MAX_EVENT_TEXT or any(ord(ch) < 32 for ch in value):
                raise ValueError("Recording event text is invalid")
            text[key] = value
        timestamp = payload["timestamp"]
        raw_sample = payload["raw_sample"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
            raise ValueError("Recording event timestamp is invalid")
        if type(raw_sample) is not int or raw_sample < 0:
            raise ValueError("Recording event RAW index is invalid")
        return cls(timestamp=float(timestamp), raw_sample=raw_sample, **text)


@dataclass(slots=True)
class RawRecording:
    metadata: dict[str, object]
    raw: np.ndarray
    telemetry: np.ndarray
    events: list[RecordingEvent]

    @property
    def raw_start_sample(self) -> int | None:
        value = self.metadata.get("raw_start_sample")
        if value is None:
            if self.raw.size:
                raise ValueError("Recording RAW start index is missing")
            return None
        if type(value) is not int or value < 0:
            raise ValueError("Recording RAW start index is invalid")
        return value

    def raw_slice(self, start_sample: int, end_sample: int) -> np.ndarray:
        if type(start_sample) is not int or type(end_sample) is not int or start_sample < 0 or end_sample <= start_sample:
            raise ValueError("RAW replay bounds are invalid")
        origin = self.raw_start_sample
        if origin is None:
            return np.empty(0, dtype=np.float64)
        recording_end = origin + int(self.raw.size)
        if start_sample < origin or end_sample > recording_end:
            return np.empty(0, dtype=np.float64)
        left = start_sample - origin
        right = end_sample - origin
        return self.raw[left:right].astype(np.float64, copy=False)


def _stratified_folds(session: TrainingSession, maximum_folds: int = 5) -> list[tuple[TrainingSession, TrainingSession]]:
    """Create deterministic trial-level stratified folds without epoch leakage."""
    grouped: dict[str, dict[str, list[TrainingSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in session.samples:
        grouped[sample.label][sample.trial_id].append(sample)
    if len(grouped) < 2:
        raise ValueError("At least two classes are required")
    minimum_trials = min(len(trials) for trials in grouped.values())
    if minimum_trials < 2:
        raise ValueError("At least two independent trials per class are required for cross-validation")
    folds = min(maximum_folds, minimum_trials)

    assignment: dict[str, list[list[str]]] = {}
    for label in sorted(grouped):
        trial_ids = sorted(grouped[label])
        rng = random.Random(f"mindflex-v1:{label}:{len(trial_ids)}")
        rng.shuffle(trial_ids)
        buckets = [[] for _ in range(folds)]
        for index, trial_id in enumerate(trial_ids):
            buckets[index % folds].append(trial_id)
        assignment[label] = buckets

    result: list[tuple[TrainingSession, TrainingSession]] = []
    for fold in range(folds):
        train = TrainingSession(f"{session.name}-cv{fold + 1}-train", owner=session.owner)
        validation = TrainingSession(f"{session.name}-cv{fold + 1}-validation", owner=session.owner)
        for label in sorted(grouped):
            validation_ids = set(assignment[label][fold])
            for trial_id in sorted(grouped[label]):
                target = validation if trial_id in validation_ids else train
                for sample in grouped[label][trial_id]:
                    target.add(
                        label,
                        sample.features,
                        sample.timestamp,
                        trial_id=sample.trial_id,
                        epoch_index=sample.epoch_index,
                        raw_start=sample.raw_start,
                        raw_end=sample.raw_end,
                    )
        result.append((train, validation))
    return result


def _merge_validation_results(results: Iterable[ValidationResult], *, name: str, folds: int, session: TrainingSession) -> ExperimentResult:
    total = correct = decided = 0
    totals_by_label: dict[str, int] = defaultdict(int)
    correct_by_label: dict[str, int] = defaultdict(int)
    # ValidationResult intentionally contains aggregate counts only. Confusion
    # is reconstructed separately by the caller when fold models are available.
    for result in results:
        total += result.total
        correct += result.correct
        decided += result.decided
        for label, count in result.trials_per_label.items():
            totals_by_label[label] += count
            correct_by_label[label] += result.correct_per_label.get(label, 0)

    per_label = {
        label: (correct_by_label[label] / count if count else 0.0)
        for label, count in sorted(totals_by_label.items())
    }
    active = list(per_label.values())
    return ExperimentResult(
        name=name,
        folds=folds,
        trial_count=session.total_trials,
        sample_count=len(session.samples),
        correct=correct,
        decided=decided,
        accuracy=(correct / total) if total else 0.0,
        balanced_accuracy=(sum(active) / len(active)) if active else 0.0,
        decision_rate=(decided / total) if total else 0.0,
        per_label=per_label,
        trials_per_label=dict(totals_by_label),
        confusion={},
    )


def run_experiment(session: TrainingSession, name: str = "stratified-trial-cross-validation") -> ExperimentResult:
    """Evaluate the complete model pipeline with stratified trial cross-validation."""
    folds = _stratified_folds(session)
    fold_results: list[ValidationResult] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for train, validation_session in folds:
        model = BCIModel.train(train)
        result = validate_model(model, validation_session)
        fold_results.append(result)

        groups: dict[tuple[str, str], list[TrainingSample]] = defaultdict(list)
        for sample in validation_session.samples:
            groups[(sample.label, sample.trial_id)].append(sample)
        from .bci import TrialDecisionAccumulator
        for (expected, _trial_id), samples in sorted(groups.items()):
            evaluator = TrialDecisionAccumulator(model.labels)
            for sample in sorted(samples, key=lambda item: item.epoch_index):
                evaluator.push(model.predict(sample.features))
            prediction = evaluator.resolve()
            predicted = prediction.label if prediction is not None else "<no-decision>"
            confusion[expected][predicted] += 1

    merged = _merge_validation_results(fold_results, name=name, folds=len(folds), session=session)
    merged.confusion = {label: dict(row) for label, row in sorted(confusion.items())}
    return merged


def _validate_recording_integrity(
    *,
    raw_start_sample: int | None,
    raw_count: int,
    events: list[RecordingEvent],
    session_kind: str,
) -> str:
    """Apply the single Version 1 integrity rule to saved and loaded recordings."""
    if type(raw_count) is not int or raw_count < 0:
        raise ValueError("Recording RAW count is invalid")
    if raw_count:
        if type(raw_start_sample) is not int or raw_start_sample < 0:
            raise ValueError("Recording RAW start index is invalid")
        raw_end = raw_start_sample + raw_count
    else:
        if raw_start_sample is not None:
            raise ValueError("Empty recording must not declare a RAW start index")
        raw_end = None

    starts_lifecycle = [event for event in events if event.kind == "recording_start"]
    ends_lifecycle = [event for event in events if event.kind == "recording_end"]
    if len(starts_lifecycle) != 1 or len(ends_lifecycle) != 1:
        raise ValueError("Recording lifecycle events are incomplete or duplicated")
    final_status = ends_lifecycle[0].phase
    if not final_status:
        raise ValueError("Recording final status is missing")

    starts: dict[str, RecordingEvent] = {}
    ends: dict[str, RecordingEvent] = {}
    accepted: list[RecordingEvent] = []
    for event in events:
        if event.kind not in {"trial_start", "trial_end", "epoch_accepted"}:
            continue
        if raw_end is None or event.raw_sample < raw_start_sample or event.raw_sample > raw_end:
            raise ValueError("BCI event lies outside the captured RAW interval")
        if event.kind == "trial_start":
            if not event.trial_id or not event.label or event.trial_id in starts:
                raise ValueError("BCI trial start event is invalid or duplicated")
            starts[event.trial_id] = event
        elif event.kind == "trial_end":
            if not event.trial_id or not event.label or event.trial_id in ends:
                raise ValueError("BCI trial end event is invalid or duplicated")
            ends[event.trial_id] = event
        else:
            if not event.trial_id or not event.label:
                raise ValueError("BCI accepted-epoch event is invalid")
            accepted.append(event)

    for trial_id in set(starts) & set(ends):
        start = starts[trial_id]
        end = ends[trial_id]
        if end.raw_sample <= start.raw_sample or end.label != start.label:
            raise ValueError("BCI trial boundaries are inconsistent")
    for event in accepted:
        start = starts.get(event.trial_id)
        end = ends.get(event.trial_id)
        if start is None or event.label != start.label or event.raw_sample < start.raw_sample:
            raise ValueError("Accepted BCI epoch is not bound to its recorded trial")
        if end is not None and event.raw_sample >= end.raw_sample:
            raise ValueError("Accepted BCI epoch lies outside its recorded trial")

    if final_status == "complete" and session_kind.startswith("bci-"):
        if raw_end is None:
            raise ValueError("A completed BCI recording requires a continuous RAW stream")
        if not starts or set(starts) != set(ends):
            raise ValueError("Completed BCI recording has incomplete trial boundaries")
    return final_status


class SessionRecorder:
    """Memory-efficient RAW recorder for exact offline reproduction.

    Recording never performs disk I/O from the controller ingestion thread.
    RAW samples are appended to a compact signed-short array; the complete
    immutable session is written atomically only when recording stops.
    """

    def __init__(
        self,
        path: Path,
        *,
        session_kind: str = "monitor",
        profile: str = "",
        purpose: str = "",
        owner: str = "",
    ) -> None:
        if path.suffix.lower() != RECORDING_EXTENSION:
            raise ValueError(f"Recording files must use {RECORDING_EXTENSION}")
        self.path = path
        self.session_kind = self._clean_text(session_kind, required=True)
        self.profile = self._clean_text(profile)
        self.purpose = self._clean_text(purpose)
        self.owner = self._clean_text(owner)
        self._controller: EEGController | None = None
        self._raw = array("h")
        self._raw_start_sample: int | None = None
        self._telemetry: list[list[float]] = []
        self._events: list[RecordingEvent] = []
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._ended_at = 0.0
        self._last_telemetry_at = 0.0
        self._last_telemetry_signature: tuple[object, ...] | None = None
        self.error = ""
        self._running = False

    @staticmethod
    def _clean_text(value: object, *, required: bool = False) -> str:
        if not isinstance(value, str):
            raise ValueError("Recording text metadata must be a string")
        clean = " ".join(value.strip().split())
        if required and not clean:
            raise ValueError("Recording metadata is required")
        if len(clean) > MAX_EVENT_TEXT or any(ord(ch) < 32 for ch in clean):
            raise ValueError("Recording metadata is invalid")
        return clean

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, controller: EEGController) -> None:
        if not isinstance(controller, EEGController):
            raise ValueError("A live EEGController is required")
        with self._lock:
            if self._running:
                raise RuntimeError("Recorder is already running")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.error = ""
            self._raw = array("h")
            self._raw_start_sample = None
            self._telemetry.clear()
            self._events.clear()
            self._started_at = time.time()
            self._ended_at = 0.0
            self._last_telemetry_at = 0.0
            self._last_telemetry_signature = None
            self._controller = controller
            self._running = True
        controller.add_raw_listener(self._on_raw)
        controller.add_listener(self._on_snapshot)
        snap = controller.snapshot()
        self.mark_event("recording_start", raw_sample=snap.raw_total_samples, timestamp=self._started_at)
        self._on_snapshot(snap, force=True)

    def mark_event(
        self,
        kind: str,
        *,
        raw_sample: int | None = None,
        timestamp: float | None = None,
        profile: str | None = None,
        purpose: str | None = None,
        label: str = "",
        trial_id: str = "",
        phase: str = "",
    ) -> None:
        clean_kind = self._clean_text(kind, required=True)
        clean_profile = self._clean_text(self.profile if profile is None else profile)
        clean_purpose = self._clean_text(self.purpose if purpose is None else purpose)
        clean_label = self._clean_text(label)
        clean_trial = self._clean_text(trial_id)
        clean_phase = self._clean_text(phase)
        with self._lock:
            if not self._running:
                return
            controller = self._controller
        if raw_sample is None:
            if controller is None:
                return
            raw_sample = controller.snapshot().raw_total_samples
        if type(raw_sample) is not int or raw_sample < 0:
            raise ValueError("Recording event RAW index is invalid")
        stamp = time.time() if timestamp is None else float(timestamp)
        if not math.isfinite(stamp):
            raise ValueError("Recording event timestamp is invalid")
        event = RecordingEvent(
            kind=clean_kind,
            timestamp=stamp,
            raw_sample=raw_sample,
            profile=clean_profile,
            purpose=clean_purpose,
            label=clean_label,
            trial_id=clean_trial,
            phase=clean_phase,
        )
        with self._lock:
            if self._running:
                self._events.append(event)

    def _on_raw(self, start_sample: int, samples: tuple[int, ...], timestamp: float) -> None:
        if not samples:
            return
        with self._lock:
            if not self._running:
                return
            if type(start_sample) is not int or start_sample < 0:
                self.error = "RAW recording received an invalid absolute sample index"
                self._running = False
                return
            if self._raw_start_sample is None:
                self._raw_start_sample = start_sample
            else:
                expected = self._raw_start_sample + len(self._raw)
                if start_sample != expected:
                    self.error = f"RAW recording discontinuity: expected sample {expected}, received {start_sample}"
                    self._running = False
                    return
            try:
                self._raw.extend(samples)
            except (OverflowError, TypeError) as exc:
                self.error = str(exc)
                self._running = False

    def _on_snapshot(self, snapshot: EEGSnapshot, *, force: bool = False) -> None:
        if not isinstance(snapshot, EEGSnapshot):
            return
        timestamp = float(snapshot.timestamp or time.time())
        signature = (
            snapshot.poor_signal,
            snapshot.attention,
            snapshot.meditation,
            snapshot.blink,
            tuple(snapshot.bands.get(name) for name in EEG_BANDS),
        )
        with self._lock:
            if not self._running:
                return
            # 10 Hz is enough for TGAM summary metrics. Value changes bypass the
            # throttle, so blinks/contact changes are not hidden.
            if not force and signature == self._last_telemetry_signature and timestamp - self._last_telemetry_at < 0.1:
                return
            row = [
                timestamp,
                float(snapshot.raw_total_samples),
                self._number_or_nan(snapshot.poor_signal),
                self._number_or_nan(snapshot.attention),
                self._number_or_nan(snapshot.meditation),
                self._number_or_nan(snapshot.blink),
                *[self._number_or_nan(snapshot.bands.get(name)) for name in EEG_BANDS],
                float(snapshot.packets),
                float(snapshot.bad_checksums),
                float(snapshot.dropped_bytes),
                float(snapshot.raw_rate_hz),
                float(snapshot.raw_spread),
                float(snapshot.raw_age) if math.isfinite(snapshot.raw_age) else math.nan,
            ]
            self._telemetry.append(row)
            self._last_telemetry_at = timestamp
            self._last_telemetry_signature = signature

    @staticmethod
    def _number_or_nan(value: int | float | None) -> float:
        if value is None:
            return math.nan
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.nan

    def stop(self, *, status: str = "complete") -> Path | None:
        with self._lock:
            if not self._running and self._controller is None:
                return self.path if self.path.exists() else None
            controller = self._controller
        if controller is not None:
            controller.remove_raw_listener(self._on_raw)
            controller.remove_listener(self._on_snapshot)
        stamp = time.time()
        with self._lock:
            if self._running:
                raw_sample = controller.snapshot().raw_total_samples if controller is not None else 0
                self._events.append(
                    RecordingEvent(
                        kind="recording_end",
                        timestamp=stamp,
                        raw_sample=max(0, int(raw_sample)),
                        profile=self.profile,
                        purpose=self.purpose,
                        phase=self._clean_text(status),
                    )
                )
            self._running = False
            self._controller = None
            self._ended_at = stamp
            if self.error:
                return None
            try:
                self._write_atomic()
            except (OSError, ValueError, TypeError) as exc:
                self.error = str(exc)
                return None
        return self.path

    def _write_atomic(self) -> None:
        raw = np.frombuffer(self._raw, dtype=np.int16).copy()
        _validate_recording_integrity(
            raw_start_sample=self._raw_start_sample,
            raw_count=int(raw.size),
            events=self._events,
            session_kind=self.session_kind,
        )

        telemetry = np.asarray(self._telemetry, dtype=np.float64)
        if telemetry.size == 0:
            telemetry = np.empty((0, len(TELEMETRY_COLUMNS)), dtype=np.float64)
        elif telemetry.ndim != 2 or telemetry.shape[1] != len(TELEMETRY_COLUMNS):
            raise ValueError("Internal telemetry shape is invalid")
        metadata = {
            "schema": RECORDING_SCHEMA,
            "format": RECORDING_FORMAT,
            "sample_rate": MINDFLEX_RAW_SAMPLE_RATE,
            "session_kind": self.session_kind,
            "profile": self.profile,
            "purpose": self.purpose,
            "owner": self.owner,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "raw_start_sample": self._raw_start_sample,
            "telemetry_columns": list(TELEMETRY_COLUMNS),
            "events": [event.to_dict() for event in self._events],
        }
        encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as handle:
                temp_path = handle.name
                np.savez_compressed(
                    handle,
                    metadata=np.frombuffer(encoded, dtype=np.uint8),
                    raw=raw,
                    telemetry=telemetry,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def load_recording(path: Path) -> RawRecording:
    if path.suffix.lower() != RECORDING_EXTENSION:
        raise ValueError(f"Laboratory recording must use {RECORDING_EXTENSION}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", "raw", "telemetry"}:
            raise ValueError("Recording container fields are invalid")
        metadata_bytes = np.asarray(archive["metadata"], dtype=np.uint8)
        if metadata_bytes.ndim != 1 or metadata_bytes.size > 8 * 1024 * 1024:
            raise ValueError("Recording metadata is invalid")
        try:
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Recording metadata is not valid JSON") from exc
        raw = np.asarray(archive["raw"])
        telemetry = np.asarray(archive["telemetry"], dtype=np.float64)

    expected = {
        "schema", "format", "sample_rate", "session_kind", "profile", "purpose", "owner",
        "started_at", "ended_at", "raw_start_sample", "telemetry_columns", "events",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected:
        raise ValueError("Recording metadata fields are invalid")
    if metadata.get("schema") != RECORDING_SCHEMA or metadata.get("format") != RECORDING_FORMAT:
        raise ValueError("Recording format does not match this Version 1 application")
    if metadata.get("sample_rate") != MINDFLEX_RAW_SAMPLE_RATE:
        raise ValueError("Recording sample rate is incompatible")
    for key in ("session_kind", "profile", "purpose", "owner"):
        if not isinstance(metadata.get(key), str):
            raise ValueError("Recording text metadata is invalid")
    for key in ("started_at", "ended_at"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("Recording time metadata is invalid")
    if tuple(metadata.get("telemetry_columns", ())) != TELEMETRY_COLUMNS:
        raise ValueError("Recording telemetry schema is incompatible")
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("Recording RAW array is invalid")
    if np.any(raw < -32768) or np.any(raw > 32767):
        raise ValueError("Recording RAW values exceed TGAM range")
    raw = raw.astype(np.int16, copy=False)
    if telemetry.ndim != 2 or telemetry.shape[1] != len(TELEMETRY_COLUMNS):
        raise ValueError("Recording telemetry matrix is invalid")
    if telemetry.size and not np.isfinite(telemetry[:, :2]).all():
        raise ValueError("Recording telemetry timestamps/indices are invalid")
    raw_events = metadata.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Recording events are invalid")
    events = [RecordingEvent.from_dict(item) for item in raw_events]
    recording = RawRecording(metadata=dict(metadata), raw=raw, telemetry=telemetry, events=events)
    _validate_recording_integrity(
        raw_start_sample=recording.raw_start_sample,
        raw_count=int(raw.size),
        events=events,
        session_kind=str(metadata["session_kind"]),
    )
    return recording


def analyze_replay(path: Path) -> ReplaySummary:
    recording = load_recording(path)
    telemetry = recording.telemetry
    means: dict[str, float | None] = {"attention": None, "meditation": None}
    for name in means:
        column = TELEMETRY_COLUMNS.index(name)
        values = telemetry[:, column] if telemetry.size else np.empty(0)
        values = values[np.isfinite(values)]
        if values.size:
            means[name] = float(np.mean(values))
    active_bands = 0
    for band in EEG_BANDS:
        column = TELEMETRY_COLUMNS.index(band)
        values = telemetry[:, column] if telemetry.size else np.empty(0)
        if np.any(np.isfinite(values) & (values > 0)):
            active_bands += 1
    trial_ids = {event.trial_id for event in recording.events if event.kind == "trial_start" and event.trial_id}
    rate_values = telemetry[:, TELEMETRY_COLUMNS.index("raw_rate_hz")] if telemetry.size else np.empty(0)
    rate_values = rate_values[np.isfinite(rate_values) & (rate_values > 0)]
    def last_counter(name: str) -> int:
        if not telemetry.size:
            return 0
        values = telemetry[:, TELEMETRY_COLUMNS.index(name)]
        values = values[np.isfinite(values)]
        return int(max(0.0, values[-1])) if values.size else 0
    return ReplaySummary(
        raw_samples=int(recording.raw.size),
        duration_seconds=(float(recording.raw.size) / MINDFLEX_RAW_SAMPLE_RATE),
        telemetry_frames=int(telemetry.shape[0]),
        event_count=len(recording.events),
        trial_count=len(trial_ids),
        mean_attention=means["attention"],
        mean_meditation=means["meditation"],
        active_bands=active_bands,
        mean_raw_rate_hz=float(np.mean(rate_values)) if rate_values.size else 0.0,
        bad_checksums=last_counter("bad_checksums"),
        dropped_bytes=last_counter("dropped_bytes"),
        profile=str(recording.metadata.get("profile", "")),
        purpose=str(recording.metadata.get("purpose", "")),
    )


def training_session_from_recording(path: Path) -> tuple[TrainingSession, int]:
    """Rebuild features from original RAW and visible cue boundaries.

    Only explicit trial_start/trial_end event pairs are accepted. No heuristic
    alignment is permitted: an incomplete or discontinuous window is rejected.
    """
    recording = load_recording(path)
    starts = [event for event in recording.events if event.kind == "trial_start"]
    ends_by_trial: dict[str, RecordingEvent] = {
        event.trial_id: event for event in recording.events if event.kind == "trial_end" and event.trial_id
    }
    if not starts:
        raise ValueError("Recording contains no BCI trial events")
    labels = {event.label for event in starts if event.label}
    if len(labels) < 2:
        raise ValueError("Offline BCI analysis requires at least two recorded classes")
    profile = str(recording.metadata.get("profile", "")) or "offline"
    session = TrainingSession(f"{profile}-offline-rebuild", owner=str(recording.metadata.get("owner", "")))
    rejected = 0
    seen_trials: set[str] = set()
    for start in sorted(starts, key=lambda event: (event.raw_sample, event.trial_id)):
        if not start.trial_id or not start.label or start.trial_id in seen_trials:
            rejected += 1
            continue
        seen_trials.add(start.trial_id)
        end = ends_by_trial.get(start.trial_id)
        if end is None or end.raw_sample <= start.raw_sample:
            rejected += 1
            continue
        index = 0
        while True:
            window = bci_window(start.raw_sample, index)
            if window.end_sample > end.raw_sample:
                break
            raw = recording.raw_slice(window.start_sample, window.end_sample)
            if raw.size != BCI_EPOCH_SAMPLES:
                rejected += 1
                index += 1
                continue
            try:
                features = FeatureExtractor.from_raw(raw)
                session.add(
                    start.label,
                    features.tolist(),
                    timestamp=start.timestamp + (window.start_sample - start.raw_sample) / MINDFLEX_RAW_SAMPLE_RATE,
                    trial_id=start.trial_id,
                    epoch_index=index,
                    raw_start=window.start_sample,
                    raw_end=window.end_sample,
                )
            except (ValueError, ArithmeticError, FloatingPointError):
                rejected += 1
            index += 1
    if not session.samples:
        raise ValueError("Recording did not produce any usable BCI epochs")
    return session, rejected


def analyze_bci_recording(path: Path) -> OfflineBCIResult:
    recording = load_recording(path)
    session, rejected = training_session_from_recording(path)
    experiment = run_experiment(session, name="offline-raw-stratified-trial-cross-validation")
    return OfflineBCIResult(
        profile=str(recording.metadata.get("profile", "")) or "offline",
        purpose=str(recording.metadata.get("purpose", "")),
        trials=session.total_trials,
        epochs=len(session.samples),
        rejected_epochs=rejected,
        experiment=experiment,
    )
