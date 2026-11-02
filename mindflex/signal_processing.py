from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np

from .settings import EEG_FILTER_TRANSITION_HZ, EEG_HIGHPASS_HZ, EEG_LOWPASS_HZ, MINDFLEX_RAW_SAMPLE_RATE

# Quantization-safe absolute floors; adaptive MAD remains the primary scale.
HAMPEL_MIN_DEVIATION = 128.0
DIFFERENCE_MIN_DEVIATION = 128.0


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
    if not np.isfinite(samples).all():
        raise ValueError("Hampel filtering requires finite samples")
    cleaned = samples.astype(np.float64, copy=True)
    # Vectorized local medians preserve the original zero-phase replacement
    # rule exactly, without hundreds of Python-level median calls per epoch.
    if samples.size <= 2 * radius:
        return cleaned, 0
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(cleaned, 2 * radius + 1)
    medians = np.median(windows, axis=1)
    deviations = np.median(np.abs(windows - medians[:, None]), axis=1)
    limits = np.maximum(HAMPEL_MIN_DEVIATION, threshold * 1.4826 * deviations)
    centers = cleaned[radius:-radius]
    replace = np.abs(centers - medians) > limits
    cleaned[radius:-radius] = np.where(replace, medians, centers)
    return cleaned, int(np.count_nonzero(replace))


def _smooth_step(values: np.ndarray, start: float, stop: float) -> np.ndarray:
    """Raised-cosine transition from zero at start to one at stop."""
    if stop <= start:
        raise ValueError("Transition stop must be greater than start")
    phase = np.clip((values - start) / (stop - start), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


@lru_cache(maxsize=16)
def _cached_frequency_response(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache read-only filter geometry used by repeated fixed-size EEG epochs."""
    if type(sample_count) is not int or sample_count < 16:
        raise ValueError("EEG filtering requires at least 16 samples")
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / MINDFLEX_RAW_SAMPLE_RATE)

    high = _smooth_step(
        frequencies,
        max(0.0, EEG_HIGHPASS_HZ - EEG_FILTER_TRANSITION_HZ / 2.0),
        EEG_HIGHPASS_HZ + EEG_FILTER_TRANSITION_HZ / 2.0,
    )
    low = 1.0 - _smooth_step(
        frequencies,
        EEG_LOWPASS_HZ - EEG_FILTER_TRANSITION_HZ,
        EEG_LOWPASS_HZ + EEG_FILTER_TRANSITION_HZ,
    )
    # The canonical BCI band ends at 45 Hz, below both 50 Hz and 60 Hz mains.
    # A separate notch would therefore be redundant and would only duplicate
    # configuration without changing the features used by the model.
    response = high * low
    response[0] = 0.0
    frequencies.flags.writeable = False
    response.flags.writeable = False
    return frequencies, response


def frequency_response(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return independent writable arrays; callers cannot mutate the cache."""
    freqs, response = _cached_frequency_response(sample_count)
    return freqs.copy(), response.copy()


def _float_signal(samples: Iterable[float]) -> np.ndarray:
    """Convert iterables to float without copying an existing NumPy array twice."""
    if isinstance(samples, np.ndarray):
        return np.asarray(samples, dtype=np.float64)
    return np.asarray(list(samples), dtype=np.float64)


@lru_cache(maxsize=16)
def _welch_window(segment_samples: int) -> np.ndarray:
    window = np.hanning(segment_samples)
    window.flags.writeable = False
    return window


def welch_psd(
    samples: Iterable[float],
    *,
    sample_rate: float,
    segment_samples: int,
    step_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an averaged one-sided Hann Welch PSD with fixed overlap."""
    x = _float_signal(samples)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Welch PSD requires a finite one-dimensional signal")
    if not math.isfinite(float(sample_rate)) or sample_rate <= 0.0:
        raise ValueError("Welch sample rate is invalid")
    if type(segment_samples) is not int or type(step_samples) is not int:
        raise ValueError("Welch segment geometry must use integers")
    if segment_samples < 32 or step_samples < 1 or step_samples > segment_samples or x.size < segment_samples:
        raise ValueError("Welch segment geometry is invalid")
    from numpy.lib.stride_tricks import sliding_window_view

    window = _welch_window(segment_samples)
    normalization = float(sample_rate * np.sum(window * window))
    if normalization <= 0.0:
        raise ValueError("Welch window normalization is invalid")
    # Process the overlapping Welch segments in one FFT batch. The public
    # function still accepts generators and arbitrary supported segment sizes.
    segments = sliding_window_view(x, segment_samples)[::step_samples]
    segments = segments - np.mean(segments, axis=1, keepdims=True)
    spectra = np.fft.rfft(segments * window, axis=1)
    power = np.mean((np.abs(spectra) ** 2) / normalization, axis=0)
    if segment_samples % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    freqs = np.fft.rfftfreq(segment_samples, d=1.0 / sample_rate)
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError("Welch power spectrum is invalid")
    return freqs, power


def band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    """Integrate uniformly spaced PSD bins over a half-open frequency band."""
    if freqs.ndim != 1 or power.ndim != 1 or freqs.shape != power.shape or freqs.size < 2:
        raise ValueError("Band-power arrays are invalid")
    if not np.isfinite(freqs).all() or not np.isfinite(power).all():
        raise ValueError("Band-power arrays contain non-finite values")
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Band-power frequency limits are invalid")
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    spacing = float(freqs[1] - freqs[0])
    return float(np.sum(power[mask]) * spacing)


def preprocess_eeg(
    samples: Iterable[float],
    *,
    reject_artifacts: bool = True,
) -> np.ndarray:
    """Validate, detrend and zero-phase filter one TGAM RAW epoch.

    Filtering is performed in the frequency domain, so it adds no phase delay
    to BCI epochs. Raised-cosine transitions limit ringing compared with hard
    spectral bin removal. Gross clipping and impulsive contamination are
    rejected instead of being silently learned by the classifier.
    """
    x = _float_signal(samples)
    if x.ndim != 1 or x.size < 16:
        raise ValueError("EEG preprocessing requires a one-dimensional epoch")
    if not np.isfinite(x).all():
        raise ValueError("RAW epoch contains non-finite values")
    if np.any(x < -32768.0) or np.any(x > 32767.0):
        raise ValueError("RAW epoch contains values outside the signed 16-bit TGAM range")
    if reject_artifacts:
        # An extended run of *identical* RAW codes is a likely sample-hold or
        # UART dropout. Detect it before Hampel or spectral filtering can hide it.
        # Use >=80 ms to avoid rejecting ordinary short integer plateaus.
        run_limit = max(41, int(MINDFLEX_RAW_SAMPLE_RATE * 0.08))
        # Consecutive run lengths can be checked without a Python loop over
        # each of the 768 samples in every live BCI decision window.
        changes = np.flatnonzero(np.diff(x) != 0.0) + 1
        if np.max(np.diff(np.concatenate(([0], changes, [x.size])))) >= run_limit:
            raise ValueError("RAW epoch contains a probable frozen-sample dropout")
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
            if np.max(np.abs(differences - center)) > max(DIFFERENCE_MIN_DEVIATION, 20.0 * robust_sigma):
                raise ValueError("RAW epoch contains an impulsive artifact")

    # Check broadband muscle contamination *before* the 45 Hz low-pass.
    # After filtering, the high-frequency muscle energy is mostly gone and
    # cannot be measured reliably. Only reject strongly broadband excess;
    # quantized clean EEG and a narrow 50/60 Hz mains tone must not trigger it.
    if reject_artifacts:
        spectrum = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(x.size))) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / MINDFLEX_RAW_SAMPLE_RATE)
        neural_band = float(np.sum(spectrum[(freqs >= 5.0) & (freqs < 40.0)]))
        high_band = float(np.sum(spectrum[(freqs >= 70.0) & (freqs < 200.0)]))
        total_band = float(np.sum(spectrum[(freqs >= 5.0) & (freqs < 200.0)]))
        if (high_band > 0.85 * max(neural_band, 1e-12)
                and high_band > 0.45 * max(total_band, 1e-12)):
            raise ValueError("RAW epoch contains probable broadband muscle artifact")

    positions = np.arange(x.size, dtype=np.float64)
    centered_positions = positions - float(np.mean(positions))
    denominator = float(np.dot(centered_positions, centered_positions))
    slope = float(np.dot(centered_positions, x - float(np.mean(x))) / denominator)
    detrended = x - (float(np.mean(x)) + slope * centered_positions)
    if float(np.std(detrended)) < 1e-6:
        raise ValueError("RAW epoch is flat")

    _frequencies, response = _cached_frequency_response(x.size)
    filtered = np.fft.irfft(np.fft.rfft(detrended) * response, n=x.size)
    if not np.isfinite(filtered).all() or float(np.std(filtered)) < 1e-9:
        raise ValueError("RAW epoch has no usable EEG energy after filtering")

    if reject_artifacts:
        # Frontal single-channel EEG is especially vulnerable to ocular and
        # facial-muscle activity. Reject only strongly dominated windows; do
        # not apply blanket denoising that could erase useful neural dynamics.
        window = np.hanning(filtered.size)
        spectrum = np.abs(np.fft.rfft(filtered * window)) ** 2
        freqs = np.fft.rfftfreq(filtered.size, d=1.0 / MINDFLEX_RAW_SAMPLE_RATE)
        usable = (freqs >= 0.5) & (freqs <= EEG_LOWPASS_HZ)
        total_power = float(np.sum(spectrum[usable]))
        if total_power <= 0.0 or not math.isfinite(total_power):
            raise ValueError("RAW epoch has invalid spectral energy")
        low_ratio = float(np.sum(spectrum[(freqs >= 0.5) & (freqs < 3.0)])) / total_power
        high_ratio = float(np.sum(spectrum[(freqs >= 30.0) & (freqs <= EEG_LOWPASS_HZ)])) / total_power
        rms = math.sqrt(float(np.mean(filtered * filtered)))
        crest = float(np.max(np.abs(filtered))) / max(rms, 1e-12)
        line_ratio = float(np.mean(np.abs(np.diff(filtered)))) / max(rms, 1e-12)
        # Heuristic gate only: these values are not calibrated blink-detector
        # thresholds. The 1.5 s epoch has ~0.667 Hz bin spacing, so the
        # 0.5 Hz high-pass edge cannot resolve slow ocular content reliably.
        if low_ratio > 0.85 and crest > 2.4:
            raise ValueError("RAW epoch is dominated by a probable ocular artifact")
        if high_ratio > 0.80 and line_ratio > 0.35:
            raise ValueError("RAW epoch is dominated by a probable muscle artifact")

    return filtered
