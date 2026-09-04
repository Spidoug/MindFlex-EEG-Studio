from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import time

import numpy as np

from .parser import EEG_BANDS
from .signal_processing import band_power, preprocess_eeg, welch_psd
from .settings import MINDFLEX_RAW_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class DerivedCognitiveState:
    attention: float | None
    meditation: float | None
    bands: dict[str, int]
    updated_at: float
    feature_count: int


class CognitiveEstimator:
    """Stable RAW-derived cognitive display metrics.

    The TGAM eSense values remain the preferred source when they are fresh and
    non-zero.  This estimator exists for continuity when the headset sends zero
    or temporarily stops publishing summary rows while RAW keeps flowing.

    It deliberately produces *relative session scores*, not NeuroSky eSense.
    The UI exposes the source so a RAW estimate is never presented as a native
    TGAM value.
    """

    SAMPLE_RATE = float(MINDFLEX_RAW_SAMPLE_RATE)
    WINDOW = MINDFLEX_RAW_SAMPLE_RATE
    HOP = MINDFLEX_RAW_SAMPLE_RATE // 4
    SMOOTH = 5
    FRESH_SECONDS = 3.0

    def __init__(self) -> None:
        self._raw: deque[float] = deque(maxlen=self.WINDOW)
        self._since_feature = 0
        self._focus_history: deque[float] = deque(maxlen=240)
        self._relax_history: deque[float] = deque(maxlen=240)
        self._attention_smooth: deque[float] = deque(maxlen=self.SMOOTH)
        self._meditation_smooth: deque[float] = deque(maxlen=self.SMOOTH)
        self._state = DerivedCognitiveState(None, None, {}, 0.0, 0)

    def reset(self) -> None:
        self._raw.clear()
        self._since_feature = 0
        self._focus_history.clear()
        self._relax_history.clear()
        self._attention_smooth.clear()
        self._meditation_smooth.clear()
        self._state = DerivedCognitiveState(None, None, {}, 0.0, 0)

    @property
    def state(self) -> DerivedCognitiveState:
        return self._state

    def fresh(self, now: float | None = None) -> bool:
        stamp = time.monotonic() if now is None else float(now)
        return self._state.updated_at > 0 and (stamp - self._state.updated_at) <= self.FRESH_SECONDS

    def feed(self, samples: list[int] | tuple[int, ...], now: float | None = None) -> DerivedCognitiveState | None:
        if not samples:
            return None
        stamp = time.monotonic() if now is None else float(now)
        changed = False
        for sample in samples:
            value = float(sample)
            if not math.isfinite(value):
                continue
            was_full = len(self._raw) >= self.WINDOW
            self._raw.append(value)
            self._since_feature += 1
            first_ready = not was_full and len(self._raw) >= self.WINDOW
            hop_ready = was_full and self._since_feature >= self.HOP
            if first_ready or hop_ready:
                self._since_feature = 0
                changed = self._compute(stamp) or changed
        return self._state if changed else None

    @staticmethod
    def _adaptive_score(feature: float, history: deque[float]) -> float:
        if not math.isfinite(feature):
            return math.nan
        previous = np.asarray(list(history), dtype=np.float64)
        if previous.size < 4:
            score = 50.0 + 18.0 * math.tanh(feature / 1.5)
        else:
            center = float(np.median(previous))
            mad = float(np.median(np.abs(previous - center)))
            q25, q75 = np.percentile(previous, [25.0, 75.0])
            robust_scale = max(
                0.025,
                1.4826 * mad,
                float(q75 - q25) / 1.349 if q75 > q25 else 0.0,
            )
            z = (feature - center) / robust_scale
            score = 50.0 + 38.0 * math.tanh(z / 2.2)
        history.append(float(feature))
        return max(5.0, min(95.0, float(score)))

    def _compute(self, stamp: float) -> bool:
        if len(self._raw) < self.WINDOW:
            return False
        x = np.asarray(self._raw, dtype=np.float64)
        if x.size != self.WINDOW or not np.isfinite(x).all():
            return False
        try:
            x = preprocess_eeg(x)
        except (ValueError, ArithmeticError):
            return False

        freqs, power = welch_psd(
            x, sample_rate=self.SAMPLE_RATE, segment_samples=256, step_samples=128
        )

        powers = {
            "delta": band_power(freqs, power, 0.5, 4.0),
            "theta": band_power(freqs, power, 4.0, 8.0),
            "low_alpha": band_power(freqs, power, 8.0, 10.0),
            "high_alpha": band_power(freqs, power, 10.0, 13.0),
            "low_beta": band_power(freqs, power, 13.0, 20.0),
            "high_beta": band_power(freqs, power, 20.0, 30.0),
            "low_gamma": band_power(freqs, power, 30.0, 40.0),
            "mid_gamma": band_power(freqs, power, 40.0, 45.0),
        }

        theta = powers["theta"]
        alpha = powers["low_alpha"] + powers["high_alpha"]
        beta = powers["low_beta"] + powers["high_beta"]
        gamma = powers["low_gamma"] + powers["mid_gamma"]
        useful = theta + alpha + beta + 0.25 * gamma
        if useful <= 0.0 or not math.isfinite(useful):
            return False

        eps = max(1e-12, useful * 1e-12)
        focus_feature = math.log((beta + 0.20 * gamma + eps) / (theta + alpha + eps))
        relax_feature = math.log((alpha + 0.35 * theta + eps) / (beta + 0.20 * gamma + eps))

        attention = self._adaptive_score(focus_feature, self._focus_history)
        meditation = self._adaptive_score(relax_feature, self._relax_history)
        if not math.isfinite(attention) or not math.isfinite(meditation):
            return False

        self._attention_smooth.append(attention)
        self._meditation_smooth.append(meditation)

        # Keep the relative magnitude of the Welch powers. The BCI is calibrated
        # on the same source, so no arbitrary protocol-specific scale is needed.
        bands = {
            name: max(0, int(round(powers[name])))
            for name in EEG_BANDS
        }
        self._state = DerivedCognitiveState(
            attention=float(statistics.mean(self._attention_smooth)),
            meditation=float(statistics.mean(self._meditation_smooth)),
            bands=bands,
            updated_at=stamp,
            feature_count=self._state.feature_count + 1,
        )
        return True
