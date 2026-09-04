from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .bci import Prediction, TrialDecisionAccumulator

COMMAND_IDS = ("neutral", "left", "right", "up", "down")
COMMAND_SYMBOLS = {
    "neutral": "●",
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
}


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
    raw_observations: int


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

    Each trial has PREPARE -> CUE -> REST. Raw model predictions are accepted
    only during CUE and are resolved by the same stabilizer/evidence rule used
    by model validation. Tests therefore cannot define their own voting logic.
    """

    def __init__(
        self,
        labels: Iterable[str] = COMMAND_IDS,
        *,
        prepare_seconds: float = 1.5,
        cue_seconds: float = 4.0,
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
        self.phase_started = 0.0
        self.phase_deadline = 0.0
        self._evaluator = TrialDecisionAccumulator(self.labels)
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
        self._evaluator.reset()
        stamp = time.monotonic() if now is None else float(now)
        self._enter(CommandTestPhase.PREPARE, stamp, self.prepare_seconds)
        return self.snapshot(stamp)

    def stop(self) -> None:
        self.phase = CommandTestPhase.IDLE
        self.sequence.clear()
        self._evaluator.reset()

    def mark_cue_visible(self, now: float | None = None) -> None:
        """Align CUE timing to the instant the cue is actually visible."""
        if self.phase != CommandTestPhase.CUE:
            raise RuntimeError("Cue visibility can only be marked during CUE")
        stamp = time.monotonic() if now is None else float(now)
        if not math.isfinite(stamp):
            raise ValueError("Cue timestamp must be finite")
        self.phase_started = stamp
        self.phase_deadline = stamp + self.cue_seconds

    def observe(self, prediction: Prediction | None) -> Prediction | None:
        """Feed one raw model prediction through the global trial evaluator."""
        if self.phase != CommandTestPhase.CUE:
            return None
        return self._evaluator.push(prediction)

    def tick(self, now: float | None = None) -> CommandTestSnapshot:
        stamp = time.monotonic() if now is None else float(now)
        while self.running and stamp >= self.phase_deadline:
            if self.phase == CommandTestPhase.PREPARE:
                self._evaluator.reset()
                self._enter(CommandTestPhase.CUE, stamp, self.cue_seconds)
                break
            if self.phase == CommandTestPhase.CUE:
                self._finish_trial()
                self._enter(CommandTestPhase.REST, stamp, self.rest_seconds)
                continue
            if self.phase == CommandTestPhase.REST:
                self.index += 1
                if self.index >= len(self.sequence):
                    self.phase = CommandTestPhase.COMPLETE
                    self.phase_started = stamp
                    self.phase_deadline = stamp
                    break
                self._enter(CommandTestPhase.PREPARE, stamp, self.prepare_seconds)
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
        resolved = self._evaluator.resolve()
        predicted = resolved.label if resolved is not None else None
        confidence = resolved.confidence if resolved is not None else 0.0
        result = CommandTestResult(
            cue=cue,
            predicted=predicted,
            confidence=confidence,
            correct=predicted == cue,
            observations=self._evaluator.stable_observations,
            raw_observations=self._evaluator.raw_observations,
        )
        self.results.append(result)
        self.last_result = result

    def _enter(self, phase: CommandTestPhase, start: float, duration: float) -> None:
        self.phase = phase
        self.phase_started = float(start)
        self.phase_deadline = self.phase_started + max(0.0, float(duration))


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

    def step(self, label: str | None, confidence: float, dt: float) -> bool:
        if not self.running:
            return False
        delta = max(0.01, min(0.20, float(dt)))
        friction = 0.82 ** (delta / 0.10)
        self.velocity_x *= friction
        self.velocity_y *= friction
        conf = max(0.0, min(1.0, float(confidence)))
        if label in COMMAND_IDS:
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
