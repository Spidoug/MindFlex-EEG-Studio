from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from .bci import BCI_EPOCH_SECONDS


class CalibrationPhase(str, Enum):
    IDLE = "idle"
    PREPARE = "prepare"
    TASK = "task"
    REST = "rest"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    key: str
    trials_per_label: int
    validation_trials_per_label: int
    prepare_seconds: float
    task_seconds: float
    rest_min_seconds: float
    rest_max_seconds: float
    epoch_seconds: float
    epoch_step_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not self.key
            or self.key != self.key.strip()
            or len(self.key) > 80
            or any(ord(ch) < 32 for ch in self.key)
        ):
            raise ValueError("Calibration protocol key is invalid")
        if type(self.trials_per_label) is not int or type(self.validation_trials_per_label) is not int:
            raise ValueError("Calibration trial counts must be integers")
        if self.trials_per_label < 1 or self.validation_trials_per_label < 1:
            raise ValueError("Calibration protocols require at least one trial per label")
        numeric = (
            self.prepare_seconds,
            self.task_seconds,
            self.rest_min_seconds,
            self.rest_max_seconds,
            self.epoch_seconds,
            self.epoch_step_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("Calibration timings must be finite numbers")
        if self.prepare_seconds < 0 or self.task_seconds <= 0 or self.rest_min_seconds < 0:
            raise ValueError("Calibration timings are invalid")
        if self.rest_max_seconds < self.rest_min_seconds:
            raise ValueError("Maximum rest time cannot be smaller than minimum rest time")
        if abs(self.epoch_seconds - BCI_EPOCH_SECONDS) > 1e-9 or self.epoch_seconds > self.task_seconds:
            raise ValueError(f"MindFlex model V1 requires fixed {BCI_EPOCH_SECONDS:g}s epochs that fit inside the task")
        if self.epoch_step_seconds <= 0:
            raise ValueError("Epoch step must be positive")

    def for_validation(self) -> "CalibrationProtocol":
        return CalibrationProtocol(
            key=f"{self.key}-validation",
            trials_per_label=self.validation_trials_per_label,
            validation_trials_per_label=self.validation_trials_per_label,
            prepare_seconds=self.prepare_seconds,
            task_seconds=self.task_seconds,
            rest_min_seconds=self.rest_min_seconds,
            rest_max_seconds=self.rest_max_seconds,
            epoch_seconds=self.epoch_seconds,
            epoch_step_seconds=self.epoch_step_seconds,
        )


CALIBRATION_PROTOCOLS: dict[str, CalibrationProtocol] = {
    "quick": CalibrationProtocol(
        key="quick",
        trials_per_label=4,
        validation_trials_per_label=3,
        prepare_seconds=1.5,
        task_seconds=4.0,
        rest_min_seconds=1.5,
        rest_max_seconds=2.5,
        epoch_seconds=2.0,
        epoch_step_seconds=1.0,
    ),
    "standard": CalibrationProtocol(
        key="standard",
        trials_per_label=8,
        validation_trials_per_label=5,
        prepare_seconds=2.0,
        task_seconds=5.0,
        rest_min_seconds=2.5,
        rest_max_seconds=4.0,
        epoch_seconds=2.0,
        epoch_step_seconds=1.0,
    ),
    "research": CalibrationProtocol(
        key="research",
        trials_per_label=15,
        validation_trials_per_label=8,
        prepare_seconds=3.0,
        task_seconds=6.0,
        rest_min_seconds=2.5,
        rest_max_seconds=4.5,
        epoch_seconds=2.0,
        epoch_step_seconds=1.0,
    ),
}


@dataclass(frozen=True, slots=True)
class TrialPlan:
    label: str
    trial_id: str
    trial_number: int
    total_trials: int


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    phase: CalibrationPhase
    label: str = ""
    trial_number: int = 0
    total_trials: int = 0
    seconds_remaining: float = 0.0
    epochs_collected: int = 0
    epochs_rejected: int = 0


def _canonical_labels(labels: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for label in labels:
        if not isinstance(label, str):
            raise ValueError("Calibration labels must be strings")
        clean = label.strip()
        if (
            not clean
            or clean != label
            or len(clean) > 80
            or any(ord(ch) < 32 for ch in clean)
        ):
            raise ValueError("Calibration label is invalid")
        if clean not in unique:
            unique.append(clean)
    if not unique:
        raise ValueError("At least one calibration label is required")
    return tuple(unique)


def _monotonic_stamp(now: float | None) -> float:
    if now is None:
        return time.monotonic()
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValueError("Calibration timestamp must be numeric")
    stamp = float(now)
    if not math.isfinite(stamp):
        raise ValueError("Calibration timestamp must be finite")
    return stamp


def build_balanced_plan(labels: Iterable[str], protocol: CalibrationProtocol, seed: int | None = None) -> list[TrialPlan]:
    unique = list(_canonical_labels(labels))
    rng = random.Random(seed)
    ordered: list[str] = []
    previous = ""
    for _round in range(protocol.trials_per_label):
        block = list(unique)
        rng.shuffle(block)
        if len(block) > 1 and previous and block[0] == previous:
            swap = next((i for i, value in enumerate(block[1:], 1) if value != previous), None)
            if swap is not None:
                block[0], block[swap] = block[swap], block[0]
        ordered.extend(block)
        previous = block[-1]
    total = len(ordered)
    run_id = uuid.uuid4().hex[:16]
    return [
        TrialPlan(label=label, trial_id=f"{run_id}-{index:04d}", trial_number=index, total_trials=total)
        for index, label in enumerate(ordered, 1)
    ]


class CalibrationEngine:
    """Non-blocking trial/epoch scheduler used by training and validation UIs.

    The caller owns the GUI timer and repeatedly calls tick(). feature_provider
    must return a fresh feature vector for the requested trailing epoch length,
    or None if RAW is not currently usable.
    """

    def __init__(
        self,
        labels: Iterable[str],
        protocol: CalibrationProtocol,
        feature_provider: Callable[[float], list[float] | tuple[float, ...] | None],
        epoch_sink: Callable[[str, list[float], str, int, float], None],
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(protocol, CalibrationProtocol):
            raise ValueError("A Version 1 calibration protocol is required")
        self.protocol = protocol
        self._labels = _canonical_labels(labels)
        self._seed = seed
        self.plan = build_balanced_plan(self._labels, protocol, seed=seed)
        self.feature_provider = feature_provider
        self.epoch_sink = epoch_sink
        self.phase = CalibrationPhase.IDLE
        self.index = 0
        self.phase_started = 0.0
        self.phase_deadline = 0.0
        self.next_epoch_at = 0.0
        self.epochs_collected = 0
        self.epochs_rejected = 0
        self._trial_epoch_index = 0
        self._rng = random.Random(seed)

    @property
    def current(self) -> TrialPlan | None:
        if 0 <= self.index < len(self.plan):
            return self.plan[self.index]
        return None

    @property
    def running(self) -> bool:
        return self.phase not in {CalibrationPhase.IDLE, CalibrationPhase.COMPLETE, CalibrationPhase.CANCELLED}

    def start(self, now: float | None = None) -> CalibrationProgress:
        if self.phase not in {CalibrationPhase.IDLE, CalibrationPhase.COMPLETE, CalibrationPhase.CANCELLED}:
            raise RuntimeError("Calibration is already running")
        if self.phase in {CalibrationPhase.COMPLETE, CalibrationPhase.CANCELLED}:
            self.plan = build_balanced_plan(self._labels, self.protocol, seed=self._seed)
            self._rng = random.Random(self._seed)
        self.index = 0
        self.epochs_collected = 0
        self.epochs_rejected = 0
        self._trial_epoch_index = 0
        stamp = _monotonic_stamp(now)
        self._enter(CalibrationPhase.PREPARE, stamp, self.protocol.prepare_seconds)
        return self.progress(stamp)

    def cancel(self) -> None:
        self.phase = CalibrationPhase.CANCELLED

    def tick(self, now: float | None = None) -> CalibrationProgress:
        stamp = _monotonic_stamp(now)
        if self.phase in {CalibrationPhase.IDLE, CalibrationPhase.COMPLETE, CalibrationPhase.CANCELLED}:
            return self.progress(stamp)

        # Capture at most one due epoch per tick. A delayed GUI callback cannot
        # reconstruct windows that existed in the past, so missed epochs are
        # skipped instead of duplicating the current trailing RAW window.
        if (
            self.phase == CalibrationPhase.TASK
            and stamp <= self.phase_deadline + 1e-9
            and self.next_epoch_at <= stamp + 1e-9
        ):
            trial = self.current
            if trial is not None:
                try:
                    vector = self.feature_provider(self.protocol.epoch_seconds)
                    if vector is None:
                        raise ValueError("RAW epoch is not currently usable")
                    clean = [float(value) for value in vector]
                    self.epoch_sink(
                        trial.label,
                        clean,
                        trial.trial_id,
                        self._trial_epoch_index,
                        self.protocol.epoch_seconds,
                    )
                except (ValueError, TypeError, ArithmeticError):
                    self.epochs_rejected += 1
                else:
                    self.epochs_collected += 1
                    self._trial_epoch_index += 1
            # Advance the schedule beyond the current callback time. This
            # preserves real-time spacing even when rendering temporarily lags.
            step = max(0.05, self.protocol.epoch_step_seconds)
            missed = max(1, int((stamp - self.next_epoch_at) // step) + 1)
            self.next_epoch_at += missed * step

        # Advance one or more phases if a delayed GUI callback crossed a
        # deadline. Use the scheduled boundary as the next phase start to avoid
        # drift during long calibration runs.
        while self.running and stamp >= self.phase_deadline:
            boundary = self.phase_deadline
            if self.phase == CalibrationPhase.PREPARE:
                self._enter(CalibrationPhase.TASK, boundary, self.protocol.task_seconds)
                self.next_epoch_at = boundary + self.protocol.epoch_seconds
                # If the current callback is already inside TASK, allow epoch
                # capture in the next tick; this keeps phase transitions simple.
                break
            if self.phase == CalibrationPhase.TASK:
                rest = self._rng.uniform(self.protocol.rest_min_seconds, self.protocol.rest_max_seconds)
                self._enter(CalibrationPhase.REST, boundary, rest)
                continue
            if self.phase == CalibrationPhase.REST:
                self.index += 1
                if self.index >= len(self.plan):
                    self.phase = CalibrationPhase.COMPLETE
                    self.phase_started = boundary
                    self.phase_deadline = boundary
                    break
                self._enter(CalibrationPhase.PREPARE, boundary, self.protocol.prepare_seconds)
                continue
        return self.progress(stamp)

    def progress(self, now: float | None = None) -> CalibrationProgress:
        stamp = _monotonic_stamp(now)
        trial = self.current
        remaining = max(0.0, self.phase_deadline - stamp) if self.running else 0.0
        return CalibrationProgress(
            phase=self.phase,
            label=trial.label if trial else "",
            trial_number=trial.trial_number if trial else len(self.plan),
            total_trials=len(self.plan),
            seconds_remaining=remaining,
            epochs_collected=self.epochs_collected,
            epochs_rejected=self.epochs_rejected,
        )

    def _enter(self, phase: CalibrationPhase, start: float, duration: float) -> None:
        self.phase = phase
        self.phase_started = start
        self.phase_deadline = start + max(0.0, float(duration))
        if phase == CalibrationPhase.TASK:
            self._trial_epoch_index = 0
