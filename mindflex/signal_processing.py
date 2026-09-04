from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .settings import EEG_FILTER_TRANSITION_HZ, EEG_HIGHPASS_HZ, EEG_LOWPASS_HZ, MINDFLEX_RAW_SAMPLE_RATE


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


def frequency_response(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic zero-phase EEG filter response for an epoch."""
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
    return frequencies, response


def welch_psd(
    samples: Iterable[float],
    *,
    sample_rate: float,
    segment_samples: int,
    step_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an averaged one-sided Hann Welch PSD with fixed overlap."""
    x = np.asarray(list(samples), dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Welch PSD requires a finite one-dimensional signal")
    if not math.isfinite(float(sample_rate)) or sample_rate <= 0.0:
        raise ValueError("Welch sample rate is invalid")
    if type(segment_samples) is not int or type(step_samples) is not int:
        raise ValueError("Welch segment geometry must use integers")
    if segment_samples < 32 or step_samples < 1 or step_samples > segment_samples or x.size < segment_samples:
        raise ValueError("Welch segment geometry is invalid")
    window = np.hanning(segment_samples)
    normalization = float(sample_rate * np.sum(window * window))
    if normalization <= 0.0:
        raise ValueError("Welch window normalization is invalid")
    spectra: list[np.ndarray] = []
    for start in range(0, x.size - segment_samples + 1, step_samples):
        segment = x[start : start + segment_samples]
        segment = segment - float(np.mean(segment))
        fft = np.fft.rfft(segment * window)
        psd = (np.abs(fft) ** 2) / normalization
        if psd.size > 2:
            psd[1:-1] *= 2.0
        spectra.append(psd)
    if not spectra:
        raise ValueError("Signal did not produce a Welch segment")
    power = np.mean(np.asarray(spectra, dtype=np.float64), axis=0)
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

    _frequencies, response = frequency_response(x.size)
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
        if low_ratio > 0.85 and crest > 2.4:
            raise ValueError("RAW epoch is dominated by a probable ocular artifact")
        if high_ratio > 0.80 and line_ratio > 0.35:
            raise ValueError("RAW epoch is dominated by a probable muscle artifact")

    return filtered
