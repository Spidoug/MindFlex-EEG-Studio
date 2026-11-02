"""EEG pipeline and integration regressions using deterministic synthetic data."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mindflex.benchmark import realistic_epoch
from mindflex.bci import BCIModel, FeatureExtractor, Prediction, PredictionStabilizer, TrainingSession
from mindflex.controller import EEGController
from mindflex.lab import SessionRecorder, analyze_replay, load_recording
from mindflex.parser import make_packet
from mindflex.rules import SignalPolicy
from mindflex.settings import BCI_EPOCH_SAMPLES, BCI_STEP_SAMPLES
from mindflex.signal_processing import hampel_despike, preprocess_eeg


class PipelineIntegrationTests(unittest.TestCase):
    def test_vectorized_hampel_matches_reference(self):
        rng = np.random.default_rng(2091)
        for radius in (1, 3, 5, 16):
            for length in (16, 768):
                x = rng.integers(-20, 20, length).astype(float)
                x[length // 2] += 900
                baseline = x.copy()
                count = 0
                for index in range(radius, len(x) - radius):
                    window = x[index - radius:index + radius + 1]
                    median = float(np.median(window))
                    mad = float(np.median(np.abs(window - median)))
                    if abs(x[index] - median) > max(128., 8. * 1.4826 * mad):
                        baseline[index] = median
                        count += 1
                result, replacements = hampel_despike(x, radius=radius)
                np.testing.assert_array_equal(result, baseline)
                self.assertEqual(replacements, count)

    def test_broadband_muscle_gate_generalizes_to_holdout_seeds(self):
        totals = {'none': 0, 'mains': 0, 'emg': 0}
        for profile in range(4):
            for trial in range(10):
                for nuisance in totals:
                    epoch = realistic_epoch(221000+profile*1000+trial*37,
                        profile=profile, task=('beta' if trial % 2 else 'alpha'), nuisance=nuisance)
                    try:
                        preprocess_eeg(epoch)
                    except ValueError:
                        totals[nuisance] += 1
        self.assertEqual(totals['none'], 0, 'Clean synthetic holdouts were rejected')
        self.assertEqual(totals['mains'], 0, 'Filter rejected 50Hz-contaminated clean epochs')
        self.assertGreaterEqual(totals['emg'], 36, 'Broadband synthetic EMG was not rejected')

    def test_flat_difference_mad_does_not_disable_spike_gate(self):
        x = np.zeros(BCI_EPOCH_SAMPLES, dtype=float)
        x[270:300] = 180
        with self.assertRaises(ValueError):
            preprocess_eeg(x)

    def test_realistic_frozen_sample_dropouts(self):
        for profile in range(4):
            for trial in range(6):
                x = realistic_epoch(301000 + profile * 100 + trial, profile=profile, nuisance='dropout')
                with self.assertRaisesRegex(ValueError, 'frozen-sample'):
                    preprocess_eeg(x)

    def test_training_model_round_trip_and_abstention(self):
        session = TrainingSession('synthetic unit calibration')
        for label_i, label in enumerate(('left','right')):
            for trial in range(4):
                for epoch in range(2):
                    task = 'alpha' if label_i == 0 else 'beta'
                    raw = realistic_epoch(320000+label_i*2000+trial*31+epoch,
                                          profile=2, task=task)
                    session.add(label, FeatureExtractor.from_raw(raw), 1500+trial,
                        trial_id=f'{label}-{trial}', epoch_index=epoch,
                        raw_start=trial*12000+epoch*BCI_STEP_SAMPLES,
                        raw_end=trial*12000+epoch*BCI_STEP_SAMPLES+BCI_EPOCH_SAMPLES)
        model=BCIModel.train(session)
        self.assertTrue(model.ready)
        self.assertEqual(model.fingerprint(),BCIModel.from_dict(model.to_dict()).fingerprint())
        self.assertNotEqual(model.training_fingerprint, '0'*64)
        invalid = model.to_dict()
        # A symmetric matrix with positive diagonal need not be a valid
        # Mahalanobis precision matrix (indefinite covariance geometry).
        invalid['precision'] = np.eye(len(model.mean)).tolist()
        invalid['precision'][0][1] = 2.0
        invalid['precision'][1][0] = 2.0
        with self.assertRaisesRegex(ValueError, 'malformed'):
            BCIModel.from_dict(invalid)

    def test_temporal_stabilizer_requires_three_new_decisions_after_reset(self):
        s = PredictionStabilizer(('left','right'))
        p = Prediction('left',1.0,{'left':1.,'right':0.})
        self.assertIsNone(s.push(p))
        self.assertIsNone(s.push(p))
        s.reset()   # Called by live UI on connection/quality failure.
        self.assertIsNone(s.push(p))
        self.assertIsNone(s.push(p))
        self.assertEqual(s.push(p).label,'left')

    def test_ingestion_record_replay_and_disconnect_retirement(self):
        x = realistic_epoch(402121).astype(np.int16)
        c = EEGController()
        try:
            c._reset_stream_state(source='BENCH',transport='simulator',endpoint_id='sim',connected=True)
            generation = c._stream_generation
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)/'sample.mfs'
                recorder = SessionRecorder(path)
                recorder.start(c)
                # Incomplete frame + deliberately bad checksum exercises recovery.
                packet = make_packet([0x80, 2, 0, 10]); corrupt = bytearray(packet); corrupt[-1] ^= 1
                c._on_bytes(bytes(corrupt))
                for start in range(0, x.size, 16):
                    payload=b''.join(bytes((0x80,2))+int(v).to_bytes(2,'big',signed=True)
                                     for v in x[start:start+16])
                    packet=make_packet(payload)
                    c._on_bytes(packet[:7])
                    c._on_bytes(packet[7:])
                snap=c.snapshot()
                self.assertEqual(snap.raw_total_samples,x.size)
                self.assertEqual(snap.bad_checksums,1)
                self.assertEqual(c.raw_slice(0,int(x.size)),x.tolist())
                self.assertEqual(recorder.stop(),path)
                self.assertFalse(recorder.error)
                loaded=load_recording(path)
                np.testing.assert_array_equal(loaded.raw,x)
                summary=analyze_replay(path)
                self.assertEqual(summary.raw_samples,x.size)
                self.assertEqual(summary.event_count,2)
            c.disconnect()
            self.assertFalse(c.snapshot().connected)
            self.assertEqual(c.snapshot().raw_total_samples,0)
            c.submit_bytes(make_packet([0x80,2,0,11]))
            # Even a batch dequeued before disconnection must be discarded.
            c._on_bytes(make_packet([0x80,2,0,11]),generation=generation)
            self.assertEqual(c.snapshot().raw_total_samples,0)
        finally:
            c.close()

    def test_offline_bci_replay_excludes_epochs_across_transport_loss(self):
        from mindflex.lab import training_session_from_recording
        c = EEGController()
        try:
            c._reset_stream_state(source='SIM', transport='simulator', endpoint_id='sim', connected=True)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'training.mfs'
                recorder = SessionRecorder(path, session_kind='bci-training', profile='cursor')
                recorder.start(c)
                for label in ('alpha', 'beta'):
                    for trial in range(2):
                        name = f'{label}-{trial}'
                        boundary = c.snapshot().raw_total_samples
                        recorder.mark_event('trial_start', raw_sample=boundary, label=label, trial_id=name)
                        raw = np.concatenate((
                            realistic_epoch(610000 + (10000 if label=='beta' else 0) + trial*41, task=label),
                            realistic_epoch(611000 + (10000 if label=='beta' else 0) + trial*41, task=label)
                        )).astype(np.int16)
                        for offset in range(0, raw.size, 16):
                            if label == 'alpha' and trial == 0 and offset == 384:
                                packet = bytearray(make_packet([0x80, 2, 0, 11]))
                                packet[-1] ^= 1
                                c._on_bytes(packet)
                            payload = b''.join(bytes((0x80, 2)) + int(value).to_bytes(2,'big',signed=True)
                                               for value in raw[offset:offset + 16])
                            c._on_bytes(make_packet(payload))
                        recorder.mark_event('trial_end', raw_sample=c.snapshot().raw_total_samples,
                                            label=label, trial_id=name)
                self.assertEqual(recorder.stop(), path, recorder.error)
                session, rejected = training_session_from_recording(path)
                self.assertEqual(session.total_trials, 4)
                self.assertGreaterEqual(rejected, 1)
                self.assertEqual(len(session.samples) + rejected, 4 * 7)
                self.assertFalse(any(s.trial_id == 'alpha-0' and s.raw_start < 384 < s.raw_end
                                     for s in session.samples))
        finally:
            c.close()

    def test_policy_rejects_loss_of_contact_or_checksum(self):
        policy=SignalPolicy()
        healthy=dict(receiving=True,poor_signal=0,raw_age=0.1,
                     raw_buffered_samples=1024,raw_rate_hz=512.0,
                     raw_spread=28.0,packets=500,bad_checksums=0)
        self.assertTrue(policy.raw_capture_usable(**healthy))
        self.assertFalse(policy.raw_capture_usable(**(healthy | {'poor_signal':200})))
        self.assertFalse(policy.raw_capture_usable(**(healthy | {'bad_checksums':400})))
        self.assertFalse(policy.raw_capture_usable(**(healthy | {'raw_age':3})))

if __name__ == '__main__': unittest.main()
