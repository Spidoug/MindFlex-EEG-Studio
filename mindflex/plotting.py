from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import font_manager, rcParams


def configure_plot_font(language: str) -> None:
    """Choose a font family that can render the active UI language."""
    candidates = (
        ("Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans")
        if language == "ja"
        else ("DejaVu Sans", "Segoe UI", "Noto Sans")
    )
    for family in candidates:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except Exception:
            continue
        rcParams["font.family"] = [family]
        break


@dataclass(slots=True)
class RenderStats:
    frames: int = 0
    dropped: int = 0
    last_ms: float = 0.0
    average_ms: float = 0.0


class FastLinePlot:
    """Single-axis Tk/Matplotlib plot optimized for streaming updates.

    The expensive Figure/Axes/Line objects are created once. New frames only
    replace Line2D data and use blitting when the backend supports it.
    """

    def __init__(
        self,
        master,
        xlabel: str = "",
        ylabel: str = "",
        ylim: tuple[float, float] | None = None,
    ) -> None:
        self.figure = Figure(figsize=(7.0, 3.0), dpi=100, layout="constrained")
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if ylim is not None:
            self.axes.set_ylim(*ylim)
        (self.line,) = self.axes.plot([], [], linewidth=1.0, animated=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.widget = self.canvas.get_tk_widget()
        self._background = None
        self._last_limits: tuple[float, float, float, float] | None = None
        self.stats = RenderStats()
        self.canvas.mpl_connect("resize_event", self._invalidate)
        self.canvas.draw()
        self._cache_background()

    def _invalidate(self, *_args) -> None:
        self._background = None

    def _cache_background(self) -> None:
        try:
            self.canvas.draw()
            self._background = self.canvas.copy_from_bbox(self.axes.bbox)
        except Exception:
            self._background = None

    def set_labels(self, xlabel: str, ylabel: str) -> None:
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self._background = None

    def update(
        self,
        x,
        y,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        started = time.perf_counter()
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size != y_arr.size:
            return
        self.line.set_data(x_arr, y_arr)
        if xlim is None and x_arr.size:
            xlim = (float(x_arr[0]), float(x_arr[-1]) if x_arr[-1] != x_arr[0] else float(x_arr[0] + 1.0))
        if ylim is None and y_arr.size:
            finite = y_arr[np.isfinite(y_arr)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                pad = max(1.0, (hi - lo) * 0.08)
                ylim = (lo - pad, hi + pad)
        changed_limits = False
        if xlim is not None:
            old = self.axes.get_xlim()
            if abs(old[0] - xlim[0]) > 1e-6 or abs(old[1] - xlim[1]) > 1e-6:
                self.axes.set_xlim(*xlim)
                changed_limits = True
        if ylim is not None:
            old = self.axes.get_ylim()
            span = max(1.0, old[1] - old[0])
            # Hysteresis prevents axis changes from forcing full redraw every frame.
            if ylim[0] < old[0] or ylim[1] > old[1] or (ylim[1] - ylim[0]) < span * 0.45:
                self.axes.set_ylim(*ylim)
                changed_limits = True
        if changed_limits or self._background is None:
            self._cache_background()
        try:
            if self._background is not None:
                self.canvas.restore_region(self._background)
                self.axes.draw_artist(self.line)
                self.canvas.blit(self.axes.bbox)
            else:
                self.canvas.draw_idle()
        except Exception:
            self.canvas.draw_idle()
        elapsed = (time.perf_counter() - started) * 1000.0
        s = self.stats
        s.frames += 1
        s.last_ms = elapsed
        s.average_ms += (elapsed - s.average_ms) / s.frames


class FrameScheduler:
    """Tk timer that keeps plotting bounded independently of acquisition rate."""

    def __init__(self, widget, callback: Callable[[], None], fps: int = 30) -> None:
        self.widget = widget
        self.callback = callback
        self.fps = max(10, min(60, int(fps)))
        self._job = None
        self._running = False
        self._next = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._next = time.perf_counter()
        self._schedule(0)

    def stop(self) -> None:
        self._running = False
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _schedule(self, delay_ms: int) -> None:
        self._job = self.widget.after(max(1, delay_ms), self._tick)

    def _tick(self) -> None:
        if not self._running:
            return
        interval = 1.0 / self.fps
        now = time.perf_counter()
        if now >= self._next:
            self.callback()
            self._next = max(self._next + interval, now + interval * 0.25)
        delay = max(1, int((self._next - time.perf_counter()) * 1000))
        self._schedule(delay)

class FastMultiLinePlot:
    """Multi-series variant used for low-rate cognitive metrics."""

    def __init__(self, master, series: tuple[str, ...], xlabel: str = "", ylabel: str = "", ylim=None) -> None:
        self.figure = Figure(figsize=(7.0, 2.6), dpi=100, layout="constrained")
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if ylim is not None:
            self.axes.set_ylim(*ylim)
        self.lines = {name: self.axes.plot([], [], linewidth=1.4, animated=True, label=name)[0] for name in series}
        if len(series) > 1:
            self.axes.legend(loc="upper right")
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.widget = self.canvas.get_tk_widget()
        self._background = None
        self.stats = RenderStats()
        self.canvas.mpl_connect("resize_event", self._invalidate)
        self.canvas.draw()
        self._cache_background()

    def _invalidate(self, *_args) -> None:
        self._background = None

    def _cache_background(self) -> None:
        try:
            self.canvas.draw()
            self._background = self.canvas.copy_from_bbox(self.axes.bbox)
        except Exception:
            self._background = None

    def set_labels(self, xlabel: str, ylabel: str, series_labels: dict[str, str] | None = None) -> None:
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if series_labels:
            for key, line in self.lines.items():
                line.set_label(series_labels.get(key, key))
            self.axes.legend(loc="upper right")
        self._background = None

    def update(self, data: dict[str, tuple[list[float], list[float]]], xlim=None, ylim=None) -> None:
        started = time.perf_counter()
        for name, line in self.lines.items():
            x, y = data.get(name, ([], []))
            line.set_data(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        changed = False
        if xlim is not None:
            old = self.axes.get_xlim()
            if abs(old[0] - xlim[0]) > 1e-6 or abs(old[1] - xlim[1]) > 1e-6:
                self.axes.set_xlim(*xlim)
                changed = True
        if ylim is not None:
            old = self.axes.get_ylim()
            if old != ylim:
                self.axes.set_ylim(*ylim)
                changed = True
        if changed or self._background is None:
            self._cache_background()
        try:
            if self._background is not None:
                self.canvas.restore_region(self._background)
                for line in self.lines.values():
                    self.axes.draw_artist(line)
                self.canvas.blit(self.axes.bbox)
            else:
                self.canvas.draw_idle()
        except Exception:
            self.canvas.draw_idle()
        elapsed = (time.perf_counter() - started) * 1000.0
        s = self.stats
        s.frames += 1
        s.last_ms = elapsed
        s.average_ms += (elapsed - s.average_ms) / s.frames
