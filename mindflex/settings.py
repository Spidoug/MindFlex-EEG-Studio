from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

APP_NAME = "MindFlex EEG Studio"
APP_AUTHOR = "Douglas Santana"
APP_HANDLE = "@spidoug"


def app_data_dir() -> Path:
    override = os.getenv("MINDFLEX_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", Path.home()))
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "mindflex-eeg-studio"


@dataclass(slots=True)
class Settings:
    language: str = "en"
    connection_mode: str = "bluetooth"
    connection_endpoint: str = ""
    baudrate: int = 57600
    graph_fps: int = 20
    graph_window_seconds: float = 8.0
    raw_sample_rate: int = 512
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
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return cls()
        allowed = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {k: v for k, v in payload.items() if k in allowed}
        try:
            obj = cls(**clean)
        except (TypeError, ValueError):
            obj = cls()
        obj.connection_mode = obj.connection_mode if obj.connection_mode in {"serial", "bluetooth"} else "bluetooth"
        obj.baudrate = 57600
        obj.graph_fps = cls._bounded_int(obj.graph_fps, default=20, minimum=10, maximum=60)
        obj.graph_window_seconds = cls._bounded_float(
            obj.graph_window_seconds, default=8.0, minimum=1.0, maximum=60.0
        )
        # The physical MindFlex/TGAM RAW stream is fixed at 512 Hz.
        obj.raw_sample_rate = 512
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
        if not minimum <= parsed <= maximum:
            return default if parsed != parsed else max(minimum, min(maximum, parsed))
        return parsed

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
