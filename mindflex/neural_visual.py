from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .bci import Prediction

COMMAND_IDS = ("neutral", "left", "right", "up", "down")
COMMAND_SYMBOLS = {
    "neutral": "●",
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
}


class TemporalEvidenceFilter:
    """Moving posterior average used by the visual command surface and cursor.

    The classifier itself remains stateless. This filter only stabilizes the
    presentation/control decision by averaging the most recent posterior
    distributions.
    """

    def __init__(self, labels: Iterable[str] = COMMAND_IDS, window: int = 8) -> None:
        clean = tuple(dict.fromkeys(str(label) for label in labels if str(label)))
        if len(clean) < 2:
            raise ValueError("At least two labels are required")
        self.labels = clean
        self.window = max(1, int(window))
        self._scores: deque[dict[str, float]] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._scores.clear()

    def push(self, prediction: Prediction | None) -> Prediction | None:
        if prediction is None:
            return None if not self._scores else self.current()
        row = {label: max(0.0, float(prediction.scores.get(label, 0.0))) for label in self.labels}
        total = sum(row.values())
        if total <= 0.0:
            return self.current()
        self._scores.append({label: value / total for label, value in row.items()})
        return self.current()

    def current(self) -> Prediction | None:
        if not self._scores:
            return None
        averaged = {
            label: sum(row.get(label, 0.0) for row in self._scores) / len(self._scores)
            for label in self.labels
        }
        total = sum(averaged.values()) or 1.0
        averaged = {label: value / total for label, value in averaged.items()}
        label = max(averaged, key=averaged.get)
        return Prediction(label=label, confidence=float(averaged[label]), scores=averaged)


class CommandTestPhase(str, Enum):
    IDLE = "idle"
    PREPARE = "prepare"
    CUE = "cue"
    REST = "rest"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class CommandTestResult:
    cue: str
    predicted: str | None
    confidence: float
    correct: bool
    observations: int


@dataclass(frozen=True, slots=True)
class CommandTestSnapshot:
    phase: CommandTestPhase
    cue: str = ""
    trial_number: int = 0
    total_trials: int = 0
    seconds_remaining: float = 0.0
    last_result: CommandTestResult | None = None
    completed: int = 0
    correct: int = 0


class CommandTestSession:
    """Non-blocking blind command test for a trained cursor model.

    Each trial has PREPARE -> CUE -> REST. Predictions are accumulated only
    during CUE and resolved by the average posterior score. This keeps the UI
    deterministic and avoids using one noisy instantaneous prediction.
    """

    def __init__(
        self,
        labels: Iterable[str] = COMMAND_IDS,
        *,
        prepare_seconds: float = 1.5,
        cue_seconds: float = 3.0,
        rest_seconds: float = 1.0,
        seed: int | None = None,
    ) -> None:
        clean = tuple(dict.fromkeys(str(label) for label in labels if str(label)))
        if len(clean) < 2:
            raise ValueError("At least two command labels are required")
        self.labels = clean
        self.prepare_seconds = max(0.1, float(prepare_seconds))
        self.cue_seconds = max(0.2, float(cue_seconds))
        self.rest_seconds = max(0.0, float(rest_seconds))
        self._rng = random.Random(seed)
        self.phase = CommandTestPhase.IDLE
        self.sequence: list[str] = []
        self.index = 0
        self.phase_deadline = 0.0
        self.score_sum: dict[str, float] = {}
        self.observations = 0
        self.results: list[CommandTestResult] = []
        self.last_result: CommandTestResult | None = None

    @property
    def running(self) -> bool:
        return self.phase not in {CommandTestPhase.IDLE, CommandTestPhase.COMPLETE}

    @property
    def current_cue(self) -> str:
        if 0 <= self.index < len(self.sequence):
            return self.sequence[self.index]
        return ""

    def start(self, trials: int = 20, now: float | None = None) -> CommandTestSnapshot:
        count = max(1, int(trials))
        sequence: list[str] = []
        previous = ""
        while len(sequence) < count:
            block = list(self.labels)
            self._rng.shuffle(block)
            if len(block) > 1 and previous and block[0] == previous:
                swap = next((i for i, value in enumerate(block[1:], 1) if value != previous), None)
                if swap is not None:
                    block[0], block[swap] = block[swap], block[0]
            sequence.extend(block)
            previous = block[-1]
        self.sequence = sequence[:count]
        self.index = 0
        self.results.clear()
        self.last_result = None
        self.score_sum = {label: 0.0 for label in self.labels}
        self.observations = 0
        stamp = time.monotonic() if now is None else float(now)
        self._enter(CommandTestPhase.PREPARE, stamp, self.prepare_seconds)
        return self.snapshot(stamp)

    def stop(self) -> None:
        self.phase = CommandTestPhase.IDLE
        self.sequence.clear()
        self.score_sum.clear()
        self.observations = 0

    def observe(self, prediction: Prediction | None) -> None:
        if self.phase != CommandTestPhase.CUE or prediction is None:
            return
        for label in self.labels:
            self.score_sum[label] = self.score_sum.get(label, 0.0) + max(0.0, float(prediction.scores.get(label, 0.0)))
        self.observations += 1

    def tick(self, now: float | None = None) -> CommandTestSnapshot:
        stamp = time.monotonic() if now is None else float(now)
        while self.running and stamp >= self.phase_deadline:
            boundary = self.phase_deadline
            if self.phase == CommandTestPhase.PREPARE:
                self.score_sum = {label: 0.0 for label in self.labels}
                self.observations = 0
                self._enter(CommandTestPhase.CUE, boundary, self.cue_seconds)
                break
            if self.phase == CommandTestPhase.CUE:
                self._finish_trial()
                self._enter(CommandTestPhase.REST, boundary, self.rest_seconds)
                continue
            if self.phase == CommandTestPhase.REST:
                self.index += 1
                if self.index >= len(self.sequence):
                    self.phase = CommandTestPhase.COMPLETE
                    self.phase_deadline = boundary
                    break
                self._enter(CommandTestPhase.PREPARE, boundary, self.prepare_seconds)
                continue
        return self.snapshot(stamp)

    def snapshot(self, now: float | None = None) -> CommandTestSnapshot:
        stamp = time.monotonic() if now is None else float(now)
        remaining = max(0.0, self.phase_deadline - stamp) if self.running else 0.0
        correct = sum(1 for result in self.results if result.correct)
        return CommandTestSnapshot(
            phase=self.phase,
            cue=self.current_cue,
            trial_number=min(self.index + 1, len(self.sequence)) if self.sequence else 0,
            total_trials=len(self.sequence),
            seconds_remaining=remaining,
            last_result=self.last_result,
            completed=len(self.results),
            correct=correct,
        )

    def confusion(self) -> dict[str, dict[str, int]]:
        matrix = {actual: {predicted: 0 for predicted in self.labels} for actual in self.labels}
        for result in self.results:
            if result.predicted in self.labels:
                matrix[result.cue][result.predicted] += 1
        return matrix

    def _finish_trial(self) -> None:
        cue = self.current_cue
        if self.observations <= 0:
            predicted = None
            confidence = 0.0
        else:
            averages = {label: self.score_sum.get(label, 0.0) / self.observations for label in self.labels}
            predicted = max(averages, key=averages.get)
            confidence = float(averages[predicted])
        result = CommandTestResult(
            cue=cue,
            predicted=predicted,
            confidence=confidence,
            correct=predicted == cue,
            observations=self.observations,
        )
        self.results.append(result)
        self.last_result = result

    def _enter(self, phase: CommandTestPhase, start: float, duration: float) -> None:
        self.phase = phase
        self.phase_deadline = float(start) + max(0.0, float(duration))


@dataclass(slots=True)
class CursorArena:
    """Pure cursor dynamics used by the Tk visual surface."""

    width: float = 760.0
    height: float = 420.0
    cursor_x: float = 380.0
    cursor_y: float = 210.0
    target_x: float = 620.0
    target_y: float = 210.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    hits: int = 0
    running: bool = False
    path: list[tuple[float, float]] = field(default_factory=list)
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.set_bounds(self.width, self.height)

    def start(self) -> None:
        self.running = True
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.path.clear()
        self.new_target()

    def stop(self) -> None:
        self.running = False
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def set_bounds(self, width: float, height: float) -> None:
        self.width = max(320.0, float(width))
        self.height = max(220.0, float(height))
        self.cursor_x = min(max(14.0, self.cursor_x), self.width - 14.0)
        self.cursor_y = min(max(14.0, self.cursor_y), self.height - 14.0)
        self.target_x = min(max(35.0, self.target_x), self.width - 35.0)
        self.target_y = min(max(35.0, self.target_y), self.height - 35.0)

    def new_target(self) -> None:
        margin = 50.0
        for _ in range(40):
            x = self._rng.uniform(margin, max(margin + 1.0, self.width - margin))
            y = self._rng.uniform(margin, max(margin + 1.0, self.height - margin))
            if math.hypot(x - self.cursor_x, y - self.cursor_y) >= 140.0:
                self.target_x = x
                self.target_y = y
                return
        self.target_x = self.width - margin
        self.target_y = self.height / 2.0

    def step(self, label: str | None, confidence: float, dt: float, *, threshold: float = 0.25) -> bool:
        if not self.running:
            return False
        delta = max(0.01, min(0.20, float(dt)))
        friction = 0.82 ** (delta / 0.10)
        self.velocity_x *= friction
        self.velocity_y *= friction
        conf = max(0.0, min(1.0, float(confidence)))
        if label in COMMAND_IDS and conf >= max(0.0, min(1.0, float(threshold))):
            if label == "neutral":
                self.velocity_x *= 0.30
                self.velocity_y *= 0.30
            else:
                acceleration = 18.0 + 58.0 * conf
                if label == "left":
                    self.velocity_x -= acceleration
                elif label == "right":
                    self.velocity_x += acceleration
                elif label == "up":
                    self.velocity_y -= acceleration
                elif label == "down":
                    self.velocity_y += acceleration
        speed = math.hypot(self.velocity_x, self.velocity_y)
        if speed > 92.0:
            scale = 92.0 / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
        self.cursor_x += self.velocity_x * delta
        self.cursor_y += self.velocity_y * delta
        self.cursor_x = min(max(14.0, self.cursor_x), self.width - 14.0)
        self.cursor_y = min(max(14.0, self.cursor_y), self.height - 14.0)
        self.path.append((self.cursor_x, self.cursor_y))
        if len(self.path) > 180:
            del self.path[:-180]
        if math.hypot(self.cursor_x - self.target_x, self.cursor_y - self.target_y) <= 39.0:
            self.hits += 1
            self.velocity_x *= 0.25
            self.velocity_y *= 0.25
            self.new_target()
            return True
        return False
