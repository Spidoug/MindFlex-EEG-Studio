from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


FILTER_SCHEMA = "mindflex-eeg-filter-v1"


@dataclass(frozen=True, slots=True)
class EEGFilterConfig:
    sample_rate: float = 512.0
    highpass_hz: float = 0.5
    lowpass_hz: float = 50.0
    mains_hz: float = 60.0
    notch_half_width_hz: float = 1.0
    transition_hz: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.sample_rate,
            self.highpass_hz,
            self.lowpass_hz,
            self.mains_hz,
            self.notch_half_width_hz,
            self.transition_hz,
        )
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError("EEG filter parameters must be finite numbers")
        nyquist = self.sample_rate / 2.0
        if self.sample_rate <= 0 or not 0 < self.highpass_hz < self.lowpass_hz < nyquist:
            raise ValueError("EEG band-pass limits are invalid")
        if not 0 < self.mains_hz < nyquist:
            raise ValueError("Mains frequency is outside the sampled spectrum")
        if self.notch_half_width_hz <= 0 or self.transition_hz <= 0:
            raise ValueError("EEG filter widths must be positive")


DEFAULT_EEG_FILTER = EEGFilterConfig()


def hampel_despike(samples: np.ndarray, *, radius: int = 3, threshold: float = 8.0) -> tuple[np.ndarray, int]:
    """Replace isolated sample glitches with a local robust median.

    A minimum absolute deviation is retained because TGAM data is integer and
    quiet valid windows can otherwise have a zero or near-zero local MAD.
    """
    if samples.ndim != 1:
        raise ValueError("Hampel filtering requires a one-dimensional signal")
    if type(radius) is not int or radius < 1 or radius > 32:
        raise ValueError("Hampel radius is invalid")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Hampel threshold is invalid")
    cleaned = samples.astype(np.float64, copy=True)
    replacements = 0
    for index in range(radius, samples.size - radius):
        window = samples[index - radius : index + radius + 1]
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        limit = max(2048.0, threshold * 1.4826 * mad)
        if abs(float(samples[index]) - median) > limit:
            cleaned[index] = median
            replacements += 1
    return cleaned, replacements


def _smooth_step(values: np.ndarray, start: float, stop: float) -> np.ndarray:
    """Raised-cosine transition from zero at start to one at stop."""
    if stop <= start:
        raise ValueError("Transition stop must be greater than start")
    phase = np.clip((values - start) / (stop - start), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


def frequency_response(sample_count: int, config: EEGFilterConfig = DEFAULT_EEG_FILTER) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic zero-phase EEG filter response for an epoch."""
    if type(sample_count) is not int or sample_count < 16:
        raise ValueError("EEG filtering requires at least 16 samples")
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / config.sample_rate)

    high = _smooth_step(
        frequencies,
        max(0.0, config.highpass_hz - config.transition_hz / 2.0),
        config.highpass_hz + config.transition_hz / 2.0,
    )
    low = 1.0 - _smooth_step(
        frequencies,
        config.lowpass_hz - config.transition_hz,
        config.lowpass_hz + config.transition_hz,
    )
    response = high * low

    notch_distance = np.abs(frequencies - config.mains_hz)
    notch = _smooth_step(
        notch_distance,
        config.notch_half_width_hz,
        config.notch_half_width_hz + config.transition_hz,
    )
    response *= notch
    response[0] = 0.0
    return frequencies, response


def preprocess_eeg(
    samples: Iterable[float],
    *,
    config: EEGFilterConfig = DEFAULT_EEG_FILTER,
    reject_artifacts: bool = True,
) -> np.ndarray:
    """Validate, detrend and zero-phase filter one TGAM RAW epoch.

    Filtering is performed in the frequency domain, so it adds no phase delay
    to BCI epochs. Raised-cosine transitions limit ringing compared with hard
    spectral bin removal. Gross clipping and impulsive contamination are
    rejected instead of being silently learned by the classifier.
    """
    x = np.asarray(list(samples), dtype=np.float64)
    if x.ndim != 1 or x.size < 16:
        raise ValueError("EEG preprocessing requires a one-dimensional epoch")
    if not np.isfinite(x).all():
        raise ValueError("RAW epoch contains non-finite values")
    if np.any(x < -32768.0) or np.any(x > 32767.0):
        raise ValueError("RAW epoch contains values outside the signed 16-bit TGAM range")
    if reject_artifacts:
        clipped = np.count_nonzero((x <= -32768.0) | (x >= 32767.0))
        if clipped:
            raise ValueError("RAW epoch contains ADC clipping")
        x, replacements = hampel_despike(x)
        maximum_replacements = max(2, int(math.ceil(x.size * 0.01)))
        if replacements > maximum_replacements:
            raise ValueError("RAW epoch contains excessive impulsive artifacts")
        differences = np.diff(x)
        if differences.size:
            center = float(np.median(differences))
            mad = float(np.median(np.abs(differences - center)))
            robust_sigma = 1.4826 * mad
            # The absolute floor avoids rejecting quiet, quantized valid EEG.
            if robust_sigma > 0 and np.max(np.abs(differences - center)) > max(2048.0, 20.0 * robust_sigma):
                raise ValueError("RAW epoch contains an impulsive artifact")

    positions = np.arange(x.size, dtype=np.float64)
    centered_positions = positions - float(np.mean(positions))
    denominator = float(np.dot(centered_positions, centered_positions))
    slope = float(np.dot(centered_positions, x - float(np.mean(x))) / denominator)
    detrended = x - (float(np.mean(x)) + slope * centered_positions)
    if float(np.std(detrended)) < 1e-6:
        raise ValueError("RAW epoch is flat")

    _frequencies, response = frequency_response(x.size, config)
    filtered = np.fft.irfft(np.fft.rfft(detrended) * response, n=x.size)
    if not np.isfinite(filtered).all() or float(np.std(filtered)) < 1e-9:
        raise ValueError("RAW epoch has no usable EEG energy after filtering")
    return filtered
