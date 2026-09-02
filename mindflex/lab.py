from __future__ import annotations

import csv
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .bci import CentroidModel, TrainingSample, TrainingSession, ValidationResult, validate_model
from .controller import EEGSnapshot
from .parser import EEG_BANDS


@dataclass(slots=True)
class ExperimentResult:
    name: str
    validation: ValidationResult
    sample_count: int
    train_count: int
    validation_count: int


@dataclass(slots=True)
class ReplaySummary:
    frames: int
    duration_seconds: float
    mean_attention: float | None
    mean_meditation: float | None
    active_bands: int


def _stratified_holdout(session: TrainingSession, fraction: float = 0.25) -> tuple[TrainingSession, TrainingSession]:
    grouped: dict[str, dict[str, list[TrainingSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in session.samples:
        grouped[sample.label][sample.trial_id].append(sample)
    if len(grouped) < 2:
        raise ValueError("At least two classes are required")
    if any(len(trials) < 2 for trials in grouped.values()):
        raise ValueError("At least two independent trials per class are required for a holdout experiment")

    train = TrainingSession(f"{session.name}-train")
    validation = TrainingSession(f"{session.name}-holdout")
    for label in sorted(grouped):
        trial_ids = sorted(grouped[label])
        holdout_count = max(1, min(len(trial_ids) - 1, round(len(trial_ids) * fraction)))
        validation_ids = set(trial_ids[-holdout_count:])
        for trial_id in trial_ids:
            target = validation if trial_id in validation_ids else train
            for sample in grouped[label][trial_id]:
                target.add(
                    label, sample.features, sample.timestamp,
                    trial_id=sample.trial_id, epoch_index=sample.epoch_index, epoch_seconds=sample.epoch_seconds,
                )
    return train, validation


def run_experiment(session: TrainingSession, name: str = "centroid-trial-holdout") -> ExperimentResult:
    train, validation_session = _stratified_holdout(session)
    model = CentroidModel.train(train)
    result = validate_model(model, validation_session)
    return ExperimentResult(
        name=name,
        validation=result,
        sample_count=len(session.samples),
        train_count=train.total_trials,
        validation_count=validation_session.total_trials,
    )


class SessionRecorder:
    FIELDNAMES = [
        "timestamp",
        "attention",
        "meditation",
        "blink",
        *EEG_BANDS,
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer = None
        self._lock = threading.RLock()
        self.error = ""

    def start(self) -> None:
        with self._lock:
            if self._file is not None:
                raise RuntimeError("Recorder is already running")
            self.error = ""
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()

    def append(self, snapshot: EEGSnapshot) -> None:
        with self._lock:
            if self._writer is None:
                return
            row = {
                "timestamp": snapshot.timestamp or time.time(),
                "attention": snapshot.attention,
                "meditation": snapshot.meditation,
                "blink": snapshot.blink,
            }
            row.update({name: snapshot.bands.get(name, 0) for name in EEG_BANDS})
            try:
                self._writer.writerow(row)
            except (OSError, csv.Error, ValueError) as exc:
                self.error = str(exc)
                if self._file is not None:
                    try:
                        self._file.close()
                    except OSError:
                        pass
                self._file = None
                self._writer = None

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
            self._file = None
            self._writer = None


def replay_csv(path: Path) -> Iterator[dict[str, float | int | None]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Replay CSV contains duplicate columns")
        expected = set(SessionRecorder.FIELDNAMES)
        missing = expected - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Replay CSV is missing fields: {', '.join(sorted(missing))}")
        for row in reader:
            if None in row:
                raise ValueError("Replay CSV contains unexpected extra columns")
            clean: dict[str, float | int | None] = {}
            for key in SessionRecorder.FIELDNAMES:
                value = row.get(key)
                if value in (None, "", "None"):
                    clean[key] = None
                elif key == "timestamp":
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError("Replay CSV contains a non-finite timestamp")
                    clean[key] = parsed
                else:
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError(f"Replay CSV contains a non-finite value in {key}")
                    clean[key] = int(parsed)
            yield clean


def analyze_replay(path: Path) -> ReplaySummary:
    frames = 0
    min_timestamp: float | None = None
    max_timestamp: float | None = None
    attention_sum = 0.0
    attention_count = 0
    meditation_sum = 0.0
    meditation_count = 0
    active_band_names: set[str] = set()

    for row in replay_csv(path):
        frames += 1
        timestamp = row.get("timestamp")
        if timestamp is not None:
            ts = float(timestamp)
            min_timestamp = ts if min_timestamp is None else min(min_timestamp, ts)
            max_timestamp = ts if max_timestamp is None else max(max_timestamp, ts)
        attention = row.get("attention")
        if attention is not None:
            attention_sum += int(attention)
            attention_count += 1
        meditation = row.get("meditation")
        if meditation is not None:
            meditation_sum += int(meditation)
            meditation_count += 1
        for band in EEG_BANDS:
            if int(row.get(band) or 0) > 0:
                active_band_names.add(band)

    if frames == 0:
        raise ValueError("Replay CSV contains no data rows")
    duration = 0.0
    if min_timestamp is not None and max_timestamp is not None:
        duration = max(0.0, max_timestamp - min_timestamp)
    return ReplaySummary(
        frames=frames,
        duration_seconds=duration,
        mean_attention=(attention_sum / attention_count) if attention_count else None,
        mean_meditation=(meditation_sum / meditation_count) if meditation_count else None,
        active_bands=len(active_band_names),
    )
