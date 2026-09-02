from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable


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
    prepare_seconds: float
    task_seconds: float
    rest_min_seconds: float
    rest_max_seconds: float
    epoch_seconds: float
    epoch_step_seconds: float

    def validation_copy(self) -> "CalibrationProtocol":
        # Validation must use independent trials, but it does not need to be as
        # long as training. Keep the same task/epoch timing and reduce repeats.
        return CalibrationProtocol(
            key=f"{self.key}-validation",
            trials_per_label=max(2, min(5, (self.trials_per_label + 2) // 3)),
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


def build_balanced_plan(labels: Iterable[str], protocol: CalibrationProtocol, seed: int | None = None) -> list[TrialPlan]:
    unique = [str(label).strip() for label in labels if str(label).strip()]
    unique = list(dict.fromkeys(unique))
    if not unique:
        raise ValueError("At least one calibration label is required")
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
    stamp = int(time.time() * 1000)
    return [
        TrialPlan(label=label, trial_id=f"{stamp}-{index:04d}", trial_number=index, total_trials=total)
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
        self.protocol = protocol
        self.plan = build_balanced_plan(labels, protocol, seed=seed)
        self.feature_provider = feature_provider
        self.epoch_sink = epoch_sink
        self.phase = CalibrationPhase.IDLE
        self.index = 0
        self.phase_started = 0.0
        self.phase_deadline = 0.0
        self.next_epoch_at = 0.0
        self.epochs_collected = 0
        self.epochs_rejected = 0
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
        self.index = 0
        self.epochs_collected = 0
        self.epochs_rejected = 0
        stamp = time.monotonic() if now is None else float(now)
        self._enter(CalibrationPhase.PREPARE, stamp, self.protocol.prepare_seconds)
        return self.progress(stamp)

    def cancel(self) -> None:
        self.phase = CalibrationPhase.CANCELLED

    def tick(self, now: float | None = None) -> CalibrationProgress:
        stamp = time.monotonic() if now is None else float(now)
        if self.phase in {CalibrationPhase.IDLE, CalibrationPhase.COMPLETE, CalibrationPhase.CANCELLED}:
            return self.progress(stamp)

        # Capture at most one due epoch per tick. A delayed GUI callback cannot
        # reconstruct windows that existed in the past, so missed epochs are
        # skipped instead of duplicating the current trailing RAW window.
        if self.phase == CalibrationPhase.TASK and self.next_epoch_at <= min(stamp, self.phase_deadline) + 1e-9:
            trial = self.current
            if trial is not None:
                vector = self.feature_provider(self.protocol.epoch_seconds)
                if vector is not None:
                    self.epoch_sink(
                        trial.label,
                        [float(value) for value in vector],
                        trial.trial_id,
                        self.epochs_collected,
                        self.protocol.epoch_seconds,
                    )
                    self.epochs_collected += 1
                else:
                    self.epochs_rejected += 1
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
        stamp = time.monotonic() if now is None else float(now)
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
