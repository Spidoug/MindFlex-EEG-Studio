"""Deterministic regression checks; hardware and clinical validation remain separate."""
import unittest

import numpy as np

from mindflex.bci import FEATURE_NAMES
from mindflex.settings import BCI_EPOCH_SAMPLES, MINDFLEX_RAW_SAMPLE_RATE
from mindflex.signal_processing import frequency_response, hampel_despike, preprocess_eeg, welch_psd


class SignalProcessingTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(BCI_EPOCH_SAMPLES) / MINDFLEX_RAW_SAMPLE_RATE
        self.clean = 30 * np.sin(2 * np.pi * 10 * t) + 12 * np.sin(2 * np.pi * 18 * t)

    def test_quiet_valid_eeg_not_overcorrected(self):
        clean, count = hampel_despike(self.clean)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(clean, self.clean)
        self.assertTrue(np.isfinite(preprocess_eeg(clean)).all())

    def test_spike_and_recover_is_repaired(self):
        noisy = self.clean.copy()
        noisy[300] += 950
        output, count = hampel_despike(noisy)
        self.assertEqual(count, 1)
        self.assertLess(abs(output[300] - self.clean[300]), 25)
        self.assertTrue(np.isfinite(preprocess_eeg(noisy)).all())

    def test_multiple_glitches_rejected(self):
        noisy = self.clean.copy()
        for i in range(30, 750, 55):
            noisy[i] += 1200
        with self.assertRaisesRegex(ValueError, 'excessive impulsive'):
            preprocess_eeg(noisy)

    def test_deterministic_features(self):
        from mindflex.bci import FeatureExtractor
        a = FeatureExtractor.from_raw(self.clean)
        b = FeatureExtractor.from_raw(self.clean)
        self.assertEqual(a.size, len(FEATURE_NAMES))
        np.testing.assert_array_equal(a, b)

    def test_frequency_response_edges(self):
        freqs, response = frequency_response(BCI_EPOCH_SAMPLES)
        self.assertEqual(response[0], 0)
        self.assertTrue(np.all((response >= 0) & (response <= 1)))
        self.assertAlmostEqual(freqs[1], 512 / BCI_EPOCH_SAMPLES)
        self.assertLess(response[freqs >= 47].max(), 1e-10)

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            hampel_despike(np.array([1., np.nan, 3.]))
        with self.assertRaises(ValueError):
            preprocess_eeg(np.full(BCI_EPOCH_SAMPLES, 32767.))
        with self.assertRaises(ValueError):
            preprocess_eeg(np.zeros(BCI_EPOCH_SAMPLES))

    def test_odd_length_welch_doubles_last_positive_bin(self):
        n = 383
        x = self.clean[:n] + np.random.default_rng(13).normal(0, 3, n)
        freqs, psd = welch_psd(x, sample_rate=512, segment_samples=n, step_samples=n)
        win = np.hanning(n)
        fft = np.fft.rfft((x - x.mean()) * win)
        expected = 2.0 * (abs(fft[-1]) ** 2) / (512.0 * np.sum(win * win))
        self.assertAlmostEqual(psd[-1], expected)

    def test_welch_finite_nonnegative(self):
        f, p = welch_psd(self.clean, sample_rate=512, segment_samples=384, step_samples=192)
        self.assertEqual(f.shape, p.shape)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all(p >= 0))

if __name__ == '__main__':
    unittest.main()
