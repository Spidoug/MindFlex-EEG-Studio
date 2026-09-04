from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, read_json_object

APP_NAME = "MindFlex EEG Studio"
APP_AUTHOR = "Douglas Santana"
APP_HANDLE = "@spidoug"

# MindFlex/TGAM hardware and BCI rules. These values are intentionally fixed in
# Version 1 so every training, validation and live-control path uses the same
# acquisition and decision timing.
MINDFLEX_BAUDRATE = 57600
MINDFLEX_RAW_SAMPLE_RATE = 512
BCI_EPOCH_SECONDS = 1.5
BCI_STEP_SECONDS = 0.25
BCI_EPOCH_SAMPLES = int(MINDFLEX_RAW_SAMPLE_RATE * BCI_EPOCH_SECONDS)
BCI_STEP_SAMPLES = int(MINDFLEX_RAW_SAMPLE_RATE * BCI_STEP_SECONDS)
BCI_STABILIZATION_DECISIONS = 3
BCI_MIN_EVIDENCE_OVER_CHANCE = 0.20
EEG_HIGHPASS_HZ = 0.5
EEG_LOWPASS_HZ = 45.0
EEG_FILTER_TRANSITION_HZ = 1.0
BCI_WELCH_SEGMENT_SAMPLES = BCI_EPOCH_SAMPLES // 2
BCI_WELCH_STEP_SAMPLES = BCI_WELCH_SEGMENT_SAMPLES // 2

if (
    BCI_EPOCH_SAMPLES <= 0
    or BCI_STEP_SAMPLES <= 0
    or BCI_EPOCH_SAMPLES % BCI_STEP_SAMPLES
    or BCI_WELCH_SEGMENT_SAMPLES < 32
    or BCI_WELCH_STEP_SAMPLES <= 0
    or BCI_WELCH_SEGMENT_SAMPLES % BCI_WELCH_STEP_SAMPLES
    or not 0.0 < EEG_HIGHPASS_HZ < EEG_LOWPASS_HZ < MINDFLEX_RAW_SAMPLE_RATE / 2.0
):
    raise RuntimeError("Invalid fixed EEG/BCI configuration")


def program_dir() -> Path:
    """Return the physical folder that contains the portable application."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def app_data_dir() -> Path:
    """Store every runtime session beside the program, never in user AppData."""
    return program_dir() / "sessions"


@dataclass(slots=True)
class Settings:
    language: str = "en"
    connection_mode: str = "bluetooth"
    connection_endpoint: str = ""
    baudrate: int = MINDFLEX_BAUDRATE
    graph_fps: int = 20
    graph_window_seconds: float = 8.0
    raw_sample_rate: int = MINDFLEX_RAW_SAMPLE_RATE
    max_plot_points: int = 2400

    @classmethod
    def path(cls) -> Path:
        return app_data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        path = cls.path()
        if not path.exists():
            return cls()
        try:
            payload = read_json_object(path, max_bytes=256 * 1024)
        except (OSError, ValueError, TypeError):
            return cls()
        allowed = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {k: v for k, v in payload.items() if k in allowed}
        try:
            obj = cls(**clean)
        except (TypeError, ValueError):
            obj = cls()
        obj.language = str(obj.language) if isinstance(obj.language, str) else "en"
        obj.connection_mode = obj.connection_mode if obj.connection_mode in {"serial", "bluetooth"} else "bluetooth"
        obj.connection_endpoint = str(obj.connection_endpoint) if isinstance(obj.connection_endpoint, str) else ""
        obj.baudrate = MINDFLEX_BAUDRATE
        obj.graph_fps = cls._bounded_int(obj.graph_fps, default=20, minimum=10, maximum=60)
        obj.graph_window_seconds = cls._bounded_float(
            obj.graph_window_seconds, default=8.0, minimum=1.0, maximum=60.0
        )
        # The physical MindFlex/TGAM RAW stream is fixed at 512 Hz.
        obj.raw_sample_rate = MINDFLEX_RAW_SAMPLE_RATE
        obj.max_plot_points = cls._bounded_int(obj.max_plot_points, default=2400, minimum=600, maximum=10000)
        return obj

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            parsed = default
        if not math.isfinite(parsed):
            return default
        return max(minimum, min(maximum, parsed))

    def save(self) -> None:
        path = self.path()
        atomic_write_json(path, asdict(self), max_bytes=256 * 1024)
