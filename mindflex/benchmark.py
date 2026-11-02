"""Offline deterministic engineering benchmark. No synthetic BCI accuracy claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from .bci import BCIModel, FeatureExtractor, PredictionStabilizer, TrainingSession
from .parser import ThinkGearParser, make_packet
from .settings import BCI_EPOCH_SAMPLES, BCI_STEP_SAMPLES, MINDFLEX_RAW_SAMPLE_RATE
from .signal_processing import hampel_despike, preprocess_eeg

RATE = MINDFLEX_RAW_SAMPLE_RATE
N = BCI_EPOCH_SAMPLES


def eeg(seed: int, *, alpha: float = 24.0, beta: float = 14.0, phase: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / RATE
    return (alpha * np.sin(2 * np.pi * 10 * t + phase) +
            beta * np.sin(2 * np.pi * 19 * t + phase / 2) +
            7 * np.sin(2 * np.pi * 5 * t + phase / 3) + rng.normal(0, 3, N))


def _run_case(name, fn):
    began = time.perf_counter()
    try:
        details = fn()
        return {"name": name, "status": "pass", "elapsed_ms": round((time.perf_counter() - began) * 1000, 2), "metrics": details}
    except Exception as exc:
        return {"name": name, "status": "fail", "elapsed_ms": round((time.perf_counter() - began) * 1000, 2), "error": f"{type(exc).__name__}: {exc}"}


def _assert(cond, message):
    if not cond:
        raise AssertionError(message)


def _expect_reject(samples, description):
    try:
        preprocess_eeg(samples)
    except ValueError:
        return {"rejected": True, "scenario": description}
    raise AssertionError(f"Contaminated {description} accepted")


def _parser_stream():
    rng = np.random.default_rng(919)
    numbers = rng.integers(-32768, 32768, 300, dtype=np.int32)
    frames = [make_packet([0x80, 2, (int(value) >> 8) & 255, int(value) & 255]) for value in numbers]
    broken = bytearray(frames[77]); broken[-1] ^= 0x12
    stream = b'\x00\x71\xAA' + b''.join(frames[:77]) + bytes(broken) + b''.join(frames[78:])
    parser = ThinkGearParser()
    found = []
    cursor = 0
    while cursor < len(stream):
        size = int(rng.integers(1, 19))
        found.extend(v for packet in parser.feed(stream[cursor: cursor + size]) for v in packet.raw)
        cursor += size
    expected = [int(v) for index, v in enumerate(numbers) if index != 77]
    _assert(found == expected, 'Parser lost framing or altered RAW values')
    _assert(parser.bad_checksums == 1, 'Checksum corruption was not counted')
    return {"raw_samples": len(found), "bad_checksums": parser.bad_checksums}


def _clean():
    x = eeg(101)
    a = preprocess_eeg(x)
    b = preprocess_eeg(x)
    _assert(np.array_equal(a, b), 'Preprocessing is non-deterministic')
    _assert(np.isfinite(a).all(), 'Invalid filtered values')
    clean, count = hampel_despike(x)
    _assert(count == 0, 'Clean signal underwent impulsive repair')
    return {"replacements": count, "filtered_rms": round(float(np.sqrt(np.mean(a*a))), 4)}


def _impulse():
    x = eeg(102)
    noisy = x.copy(); noisy[301] += 950
    repaired, count = hampel_despike(noisy)
    _assert(count == 1, f'Expected one Hampel repair, observed {count}')
    _assert(abs(repaired[301] - x[301]) < 40, 'Hampel failed to recover spike')
    filtered = preprocess_eeg(noisy)
    reference = preprocess_eeg(x)
    delta = float(np.sqrt(np.mean((filtered - reference)**2)))
    _assert(delta < 5, f'Artifact altered the filtered window by RMS {delta}')
    return {"replacements": count, "post_filter_rms_error": round(delta, 5)}


def _features():
    f = FeatureExtractor.from_raw(eeg(103))
    _assert(f.shape == (24,) and np.isfinite(f).all(), 'Feature schema invalid')
    _assert(np.array_equal(f, FeatureExtractor.from_raw(eeg(103))), 'Feature reproducibility failed')
    return {"dimensions": len(f), "sha256": hashlib.sha256(f.tobytes()).hexdigest()}


def _classification():
    session = TrainingSession('Synthetic benchmark; not physiological evidence')
    # Multiple genuinely independent synthetic trials with unique epoch positions.
    for label_index, label in enumerate(('alpha', 'beta')):
        for trial in range(8):
            for epoch in range(3):
                seed = 5000 + 1000 * label_index + trial * 17 + epoch
                x = eeg(seed, alpha=32 if label_index == 0 else 8,
                        beta=8 if label_index == 0 else 32, phase=trial / 3)
                session.add(label, FeatureExtractor.from_raw(x), 1000.0 + trial,
                            trial_id=f'{label}-{trial}', epoch_index=epoch,
                            raw_start=trial * 10000 + epoch * BCI_STEP_SAMPLES,
                            raw_end=trial * 10000 + epoch * BCI_STEP_SAMPLES + N)
    model = BCIModel.train(session)
    identical = BCIModel.train(session)
    _assert(model.fingerprint() == identical.fingerprint(), 'Model training non-deterministic')
    restored = BCIModel.from_dict(model.to_dict())
    _assert(model.fingerprint() == restored.fingerprint(), 'Model JSON roundtrip changed model')
    correct = 0; total = 0; abstain = 0
    for label_index, label in enumerate(('alpha', 'beta')):
        stabilizer = PredictionStabilizer(('alpha', 'beta'))
        for epoch in range(12):
            x = eeg(20000 + label_index * 500 + epoch,
                    alpha=32 if label_index == 0 else 8,
                    beta=8 if label_index == 0 else 32, phase=epoch / 6)
            result = stabilizer.push(model.predict(FeatureExtractor.from_raw(x)))
            if result is None:
                abstain += 1
            else:
                total += 1
                correct += int(result.label == label)
    _assert(total > 0, 'No decisions produced')
    # Informational only: synthetic patterns are deliberately separable and cannot validate neural intent.
    return {"synthetic_correct": correct, "synthetic_decisions": total,
            "abstentions": abstain, "synthetic_accuracy": round(correct / total, 3),
            "fingerprint": model.fingerprint(), "caveat": "Synthetic separability is NOT human BCI accuracy"}


def _throughput(epochs):
    x = eeg(1001)
    start = time.perf_counter()
    for _ in range(epochs):
        FeatureExtractor.from_raw(x)
    elapsed = time.perf_counter() - start
    throughput = epochs / max(elapsed, 1e-9)
    required = RATE / BCI_STEP_SAMPLES
    return {"epochs": epochs, "epochs_per_second": round(throughput, 2),
            "live_required_epochs_per_second": required,
            "headroom_factor": round(throughput / required, 2),
            "note": "Throughput informational; operating system, GUI and devices not simulated"}



# Physiologically *inspired* nuisance models, not calibrated human EEG recordings.
# All seeds, amplitudes, frequencies, and perturbations are explicit and reproducible.
def realistic_epoch(seed: int, *, profile: int = 0, task: str = "rest", nuisance: str = "none") -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / RATE
    # Approximate 1/f background using a stabilized real-valued spectral synthesis.
    freq = np.fft.rfftfreq(N, 1 / RATE)
    spectrum = (rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size)) / np.maximum(freq, 0.75) ** 0.65
    spectrum[0] = 0
    background = np.fft.irfft(spectrum, n=N)
    background = background / max(float(np.std(background)), 1e-12)
    gain = (0.7, 1.0, 1.4, 1.8)[profile % 4]
    alpha = (18, 29, 12, 23)[profile % 4] * (1.15 if task == "alpha" else 0.8)
    beta = (10, 13, 19, 15)[profile % 4] * (1.2 if task == "beta" else 0.9)
    x = gain * (alpha * np.sin(2*np.pi*(9.3+0.25*profile)*t + rng.uniform(0, 6.28))
           + beta * np.sin(2*np.pi*(18.1+0.4*profile)*t + rng.uniform(0, 6.28))
           + 8*background + rng.normal(0, 4, N))
    x += rng.uniform(-30, 30) + rng.uniform(-40, 40)*t
    if nuisance == "blink":
        center = rng.uniform(0.4, 1.05)
        x += rng.choice((-1, 1)) * rng.uniform(220, 650) * np.exp(-0.5*((t-center)/0.10)**2)
    elif nuisance == "emg":
        envelope = np.exp(-0.5*((t-0.75)/0.19)**2)
        x += envelope * rng.normal(0, 120, N)
    elif nuisance == "mains":
        x += 85*np.sin(2*np.pi*50*t)
    elif nuisance == "motion":
        x += 300*np.exp(-0.5*((t-0.8)/0.27)**2) + 200*(t > 0.65)
    elif nuisance == "impulse":
        x[int(rng.integers(25, N-25))] += 1100
    elif nuisance == "dropout":
        x[220:310] = x[219]
    elif nuisance == "disconnect":
        x[:] = 0
    elif nuisance != "none":
        raise ValueError("Unknown nuisance")
    return np.rint(x).clip(-32768, 32767).astype(np.float64)


def _population_sessions():
    """Exercise inter-subject amplitude, drift, nuisance and edge cases; report rates."""
    nuisances = ('none','blink','emg','mains','motion','impulse','dropout','disconnect')
    records = {}
    for nuisance in nuisances:
        accepted = 0; rejected = 0; errors = []; corrected = 0
        for profile in range(4):
            for trial in range(8):
                raw = realistic_epoch(120000 + profile*1000 + trial*17, profile=profile,
                                      task=('alpha' if trial % 2 else 'beta'), nuisance=nuisance)
                try:
                    filtered = preprocess_eeg(raw)
                    _assert(filtered.shape == (N,) and np.isfinite(filtered).all(), 'Invalid processed profile')
                    feats = FeatureExtractor.from_raw(raw)
                    _assert(feats.shape == (24,) and np.isfinite(feats).all(), 'Invalid profile feature vector')
                    accepted += 1
                    corrected += hampel_despike(raw)[1]
                except ValueError as exc:
                    rejected += 1
                    errors.append(str(exc))
        records[nuisance] = {'epochs': accepted+rejected, 'accepted': accepted,
                             'rejected': rejected, 'rejection_rate': round(rejected/(accepted+rejected),3),
                             'replacements_in_accepted': corrected,
                             'reasons': dict((reason, errors.count(reason)) for reason in sorted(set(errors)))}
    _assert(records['none']['accepted'] >= 29, 'Excess clean EEG rejection in synthetic population')
    _assert(records['disconnect']['rejected'] == 32, 'Disconnect/flatline is not rejected')
    return {'profiles':4,'trials_per_profile_per_condition':8,'results':records,
            'warning':'Synthetic nuisance rejection is descriptive, not a clinical sensitivity estimate'}


def _stream_long():
    """Combine fragmented transport, metadata and repeated disconnect/reconnect."""
    rng = np.random.default_rng(411)
    parser = ThinkGearParser()
    expected = []; actual = []; broken = 0
    for block in range(12):
        samples = rng.integers(-1200,1201,256)
        frames = []
        for i, value in enumerate(samples):
            raw = int(value) & 65535
            payload = [0x02, 0x00 if block % 4 != 3 else 0xc8,
                       0x04, int(i % 101), 0x05, int((i+25) % 101),
                       0x80, 2, (raw >> 8) & 255, raw & 255]
            frame = make_packet(payload)
            if i == 119 and block % 3 == 1:
                altered = bytearray(frame); altered[-1] ^= 7
                frame = bytes(altered); broken += 1
            else:
                expected.append(int(value))
            frames.append(frame)
        stream = b''.join(frames)
        cursor = 0
        while cursor < len(stream):
            step = int(rng.integers(1, 67))
            actual.extend(v for obj in parser.feed(stream[cursor:cursor+step]) for v in obj.raw)
            cursor += step
        parser.discard_partial()  # explicit offline reconnect boundary
    _assert(expected == actual, 'Repeated reconnect lost or changed valid RAW data')
    _assert(parser.bad_checksums == broken, 'Checksum count mismatch over reconnects')
    return {'raw_valid':len(actual),'frames_corrupted':broken,'reconnect_cycles':12,
            'note':'Simulates byte stream and parser reconnect, not OS Bluetooth driver'}


def _spectral_invariants():
    from .signal_processing import frequency_response
    frequencies, response = frequency_response(N)
    _assert(response.shape == frequencies.shape, 'Filter response shape mismatch')
    _assert(np.all((response >= 0) & (response <= 1)), 'Filter out of range')
    _assert(response[0] == 0, 'DC not rejected')
    _assert(response[np.argmin(abs(frequencies-10))] > 0.99, 'Alpha band attenuated')
    _assert(response[np.argmin(abs(frequencies-50))] < 0.01, '50Hz not rejected')
    return {'fft_spacing_hz':round(float(frequencies[1]),6), 'gain_10hz':round(float(response[np.argmin(abs(frequencies-10))]),5),
            'gain_50hz':round(float(response[np.argmin(abs(frequencies-50))]),5)}


def _adversarial_parser():
    rng = np.random.default_rng(773)
    parser = ThinkGearParser()
    for _ in range(600):
        payload = rng.integers(0, 256, int(rng.integers(0, 48)), dtype=np.uint8).tobytes()
        parser.feed(payload)
        _assert(len(parser._buffer) < 175, 'Parser buffer growth under malformed traffic')
    parser.discard_partial()
    expected = [-300, 0, 300]
    observed = []
    for val in expected:
        raw = val & 65535
        observed.extend(z for packet in parser.feed(make_packet([0x80,2,raw>>8,raw&255])) for z in packet.raw)
    _assert(observed == expected, 'Valid stream not recovered following adversarial noise')
    return {'random_chunks':600,'recovered_raw':observed,'max_buffer_limit':174}


def _cross_session_shift():
    """Honest out-of-session probe: returns failure modes, never asserts accuracy."""
    session = TrainingSession('Cross-session synthetic domain shift probe')
    for label in ('alpha','beta'):
        for trial in range(9):
            for epoch in range(3):
                x = realistic_epoch(60000 + trial*50 + epoch + (20000 if label=='beta' else 0),
                                    profile=0, task=label)
                session.add(label,FeatureExtractor.from_raw(x),100.0+trial,
                            trial_id=f'{label}-{trial}',epoch_index=epoch,
                            raw_start=trial*10000+epoch*BCI_STEP_SAMPLES,
                            raw_end=trial*10000+epoch*BCI_STEP_SAMPLES+N)
    model = BCIModel.train(session)
    record = {}
    for profile in range(4):
        counts = {'epochs':0,'predicted':0,'correct':0,'abstained':0}
        for label in ('alpha','beta'):
            stabilizer = PredictionStabilizer(('alpha','beta'))
            for epoch in range(20):
                x = realistic_epoch(90000+profile*1000+epoch+(300 if label=='beta' else 0),profile=profile,task=label)
                try:
                    output=stabilizer.push(model.predict(FeatureExtractor.from_raw(x)))
                except ValueError:
                    output=None
                counts['epochs']+=1
                if output is None: counts['abstained']+=1
                else:
                    counts['predicted']+=1
                    counts['correct']+=int(output.label==label)
        counts['conditional_accuracy']=round(counts['correct']/max(1,counts['predicted']),3)
        record[f'profile_{profile}']=counts
    return {'domain_shift':record,'warning':'No ground-truth human mental-state accuracy inferred'}


def _artifact_gate_probe():
    """Measure rather than assume thresholds reject every blink or EMG event."""
    checks={}
    for artifact in ('blink','emg','motion','dropout','mains'):
        ok=0; rejected=0
        for i in range(40):
            try: preprocess_eeg(realistic_epoch(30000+i,profile=i%4,nuisance=artifact))
            except ValueError: rejected+=1
            else: ok+=1
        checks[artifact]={'accepted':ok,'rejected':rejected,'rejection_rate':round(rejected/40,3)}
    return {'artifact_response':checks,'warning':'These are engineering probes; actual ocular/EMG sensitivity requires labeled recordings'}

def _heldout_artifact_guard():
    """Measure out-of-seed synthetic rejection with explicit pass criteria."""
    results = {}
    for nuisance in ('none', 'mains', 'emg', 'motion', 'blink'):
        rejected = 0
        for profile in range(4):
            for trial in range(16):
                raw = realistic_epoch(510000 + profile * 1000 + trial * 71,
                    profile=profile, task=('alpha' if trial % 2 else 'beta'), nuisance=nuisance)
                try:
                    preprocess_eeg(raw)
                except ValueError:
                    rejected += 1
        results[nuisance] = {'total': 64, 'rejected': rejected, 'accepted': 64 - rejected,
                             'rejection_fraction': round(rejected / 64, 3)}
    _assert(results['none']['accepted'] >= 60, 'Clean synthetic holdouts rejected excessively')
    _assert(results['mains']['accepted'] >= 60, 'Narrowband 50Hz contaminated holdouts rejected')
    _assert(results['emg']['rejected'] >= 58, 'Broadband muscle gate misses synthetic EMG')
    return {'holdout_conditions': results,
            'warning': 'The thresholds are heuristic, not calibrated on real EEG/EMG recordings'}


def _controller_record_replay_integrity():
    """Exercise controller -> parser -> recorder -> file -> replay, including a bad frame."""
    import tempfile
    from .controller import EEGController
    from .lab import SessionRecorder, load_recording, analyze_replay

    eeg_raw = np.concatenate((realistic_epoch(524201), realistic_epoch(524202))).astype(np.int16)
    controller = EEGController()
    try:
        controller._reset_stream_state(source='SIMULATED', transport='simulator',
                                       endpoint_id='simulation', connected=True)
        with tempfile.TemporaryDirectory(prefix='mindflex-benchmark-') as dirname:
            path = Path(dirname) / 'capture.mfs'
            recorder = SessionRecorder(path)
            recorder.start(controller)
            for offset in range(0, eeg_raw.size, 16):
                if offset == 384:
                    bad = bytearray(make_packet((0x80, 2, 0, 100)))
                    bad[-1] ^= 15
                    controller._on_bytes(bad)
                payload = b''.join(bytes((0x80, 2)) + int(value).to_bytes(2, 'big', signed=True)
                                   for value in eeg_raw[offset:offset + 16])
                frame = make_packet(payload)
                controller._on_bytes(frame[:7]); controller._on_bytes(frame[7:])
            snapshot = controller.snapshot()
            _assert(snapshot.bad_checksums == 1, 'Corrupt checksum was not tracked')
            _assert(snapshot.raw_total_samples == eeg_raw.size, 'Unexpected RAW acquisition length')
            _assert(snapshot.raw_contiguous_start_sample == 384, 'Loss boundary was not recorded')
            _assert(controller.raw_slice(0, N) == [], 'Epoch crossed a corrupt transport boundary')
            _assert(len(controller.raw_slice(384, 384 + N)) == N, 'Valid post-gap epoch was rejected')
            _assert(recorder.stop() == path, f'Recorder error: {recorder.error}')
            loaded = load_recording(path)
            _assert(np.array_equal(loaded.raw, eeg_raw), 'RAW recording differs from incoming samples')
            events = [event.kind for event in loaded.events]
            _assert(events.count('transport_gap_begin') == 1 and events.count('transport_gap_end') == 1,
                    'Recording omitted stream integrity markers')
            summary = analyze_replay(path)
            _assert(summary.raw_samples == eeg_raw.size, 'Replay RAW count mismatch')
        generation = controller._stream_generation
        controller.disconnect()
        controller._on_bytes(make_packet((0x80, 2, 0, 1)), generation=generation)
        controller.submit_bytes(make_packet((0x80, 2, 0, 2)))
        _assert(controller.snapshot().raw_total_samples == 0, 'Old source resumed after disconnect')
        return {'captured_samples': int(eeg_raw.size), 'corrupt_frames': 1,
                'integrity_boundary_sample': 384, 'replay_event_count': len(events),
                'retired_on_disconnect': True,
                'note': 'No real physical serial/Bluetooth timestamps or reconnect latency simulated'}
    finally:
        controller.close()


def _individual_calibration_probe():
    """Report within-profile held-out behavior without claiming cross-user decoding."""
    results = {}
    for profile in range(4):
        session = TrainingSession(f'profile-{profile}-synthetic')
        for label in ('alpha', 'beta'):
            for trial in range(6):
                for epoch in range(2):
                    raw = realistic_epoch(571000 + profile * 12000 +
                        (0 if label == 'alpha' else 3000) + trial * 101 + epoch,
                        profile=profile, task=label)
                    session.add(label, FeatureExtractor.from_raw(raw),
                        900.0 + trial, trial_id=f'{label}-{trial}', epoch_index=epoch,
                        raw_start=trial * 3000 + epoch * BCI_STEP_SAMPLES,
                        raw_end=trial * 3000 + epoch * BCI_STEP_SAMPLES + N)
        model = BCIModel.train(session)
        decided = correct = 0
        for label in ('alpha', 'beta'):
            stabilizer = PredictionStabilizer(model.labels)
            for epoch in range(20):
                raw = realistic_epoch(693000 + profile * 9000 +
                    (0 if label == 'alpha' else 2000) + epoch * 13,
                    profile=profile, task=label)
                out = stabilizer.push(model.predict(FeatureExtractor.from_raw(raw)))
                if out is not None:
                    decided += 1
                    correct += int(out.label == label)
        _assert(decided > 0 and 0 <= correct <= decided, 'Calibration produced invalid decisions')
        results[f'profile_{profile}'] = {
            'trained_trials': session.total_trials, 'heldout_epochs': 40,
            'decided': decided, 'correct': correct,
            'conditional_accuracy': round(correct / decided, 3),
            'abstentions': 40 - decided,
        }
    return {'individually_calibrated': results,
            'warning': 'Synthetic in-profile accuracy is not human or cross-day validation'}


def run(*, performance_epochs=24):
    if type(performance_epochs) is not int or performance_epochs < 1:
        raise ValueError('Invalid benchmark parameters')
    scenarios = [
        ('parser_fragmentation_checksum', _parser_stream),
        ('clean_signal_determinism', _clean),
        ('single_uart_spike_repair', _impulse),
        ('excessive_impulses_rejection', lambda: _expect_reject(
            np.where(np.arange(N) % 44 == 0, eeg(104) + 1100, eeg(104)), 'multiple impulses')),
        ('adc_clipping_rejection', lambda: _expect_reject(np.where(np.arange(N) == 320, 32767, eeg(105)), 'ADC clipping')),
        ('flatline_rejection', lambda: _expect_reject(np.zeros(N), 'flatline')),
        ('nonfinite_rejection', lambda: _expect_reject(np.where(np.arange(N) == 300, np.nan, eeg(106)), 'NaN')),
        ('feature_schema_determinism', _features),
        ('training_prediction_stabilization', _classification),
        ('feature_throughput', lambda: _throughput(performance_epochs)),
        ('population_realistic_sessions', _population_sessions),
        ('long_run_transport_faults', _stream_long),
        ('spectral_filter_invariants', _spectral_invariants),
        ('adversarial_serial_fuzz', _adversarial_parser),
        ('cross_session_domain_shift', _cross_session_shift),
        ('artifact_detection_probe', _artifact_gate_probe),
        ('heldout_artifact_guard', _heldout_artifact_guard),
        ('controller_record_replay_integrity', _controller_record_replay_integrity),
        ('individual_calibration_probe', _individual_calibration_probe),
    ]
    results = [_run_case(name, fn) for name, fn in scenarios]
    return {"benchmark_schema": 3, "project_version": "1.0.0", "randomness": "fixed per-scenario seeds",
            "python": platform.python_version(), "numpy": np.__version__,
            "device": "simulated only", "pass_count": sum(r['status'] == 'pass' for r in results),
            "fail_count": sum(r['status'] == 'fail' for r in results), "scenarios": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description='MindFlex EEG deterministic offline benchmark')
    parser.add_argument('--output', type=Path, help='Write structured JSON results')
    parser.add_argument('--performance-epochs', type=int, default=24)
    args = parser.parse_args(argv)
    report = run(performance_epochs=args.performance_epochs)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + '\n', encoding='utf-8')
    return 1 if report['fail_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
