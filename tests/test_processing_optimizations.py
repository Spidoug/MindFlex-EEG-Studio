"""Regression checks for optimized hot paths and their public contracts."""
from __future__ import annotations

import unittest

import numpy as np

from mindflex.benchmark import eeg
from mindflex.signal_processing import frequency_response, preprocess_eeg, welch_psd


class ProcessingOptimizationTests(unittest.TestCase):
    def test_cached_filter_cannot_be_modified_through_public_api(self):
        frequencies, response = frequency_response(768)
        original_frequencies = frequencies.copy()
        original_response = response.copy()
        frequencies[:] = -10.0
        response[:] = 100.0
        next_frequencies, next_response = frequency_response(768)
        np.testing.assert_array_equal(next_frequencies, original_frequencies)
        np.testing.assert_array_equal(next_response, original_response)
        self.assertTrue(np.isfinite(preprocess_eeg(eeg(911))).all())

    def test_batched_welch_matches_independent_segment_reference(self):
        rng = np.random.default_rng(311)
        for size, step, sample_count in ((384, 192, 768), (255, 127, 800), (64, 16, 70)):
            with self.subTest(size=size, step=step):
                x = rng.standard_normal(sample_count) * 12.0
                freq, power = welch_psd(x, sample_rate=512, segment_samples=size, step_samples=step)
                window = np.hanning(size)
                reference = []
                for start in range(0, len(x) - size + 1, step):
                    seg = x[start:start+size]
                    transformed = np.fft.rfft((seg - float(np.mean(seg))) * window)
                    one = np.abs(transformed) ** 2 / (512 * np.sum(window*window))
                    if size % 2:
                        one[1:] *= 2
                    else:
                        one[1:-1] *= 2
                    reference.append(one)
                np.testing.assert_allclose(power, np.mean(reference, axis=0), atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(freq, np.fft.rfftfreq(size, 1/512))
                # The public function still accepts iterators, not only ndarray inputs.
                _, from_generator = welch_psd(iter(x), sample_rate=512, segment_samples=size, step_samples=step)
                np.testing.assert_array_equal(power, from_generator)

    def test_frozen_raw_run_is_rejected_at_threshold(self):
        x = eeg(700)
        x[200:240] = x[199]  # 41 consecutive equal RAW samples.
        with self.assertRaisesRegex(ValueError, 'frozen-sample dropout'):
            preprocess_eeg(x)

    def test_optimized_preprocessing_does_not_mutate_input(self):
        x = eeg(701)
        original = x.copy()
        first = preprocess_eeg(x)
        second = preprocess_eeg(x)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()
