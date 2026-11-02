# Testing and validation

The test suite and offline benchmark are **engineering verification**, not clinical validation or evidence of successful control by human EEG.

## Run the checks

From the repository root, with the package dependencies installed:

```bash
python -m unittest discover -s tests -v
python -m mindflex.benchmark --output benchmark-results.json
```

CI also compiles the Python sources and builds a distributable wheel on Windows and Linux. The benchmark exits nonzero if a required assertion fails and writes machine-readable JSON (scenario names, metrics, environment, pass/fail). `benchmark-results.json` is deliberately Git-ignored; regenerate it on the machine being evaluated. Throughput depends on the CPU, Python/NumPy, system load and operating system.

### GUI smoke test

A display is needed for the Tkinter smoke test. Run on Linux with Xvfb installed:

```bash
MIND_FLEX_RUN_GUI_TESTS=1 xvfb-run -a python -m unittest discover -s tests -v
```

On a graphical Windows session, set `MIND_FLEX_RUN_GUI_TESTS=1` in the environment and run the same unittest command. The smoke test verifies app creation, simulator connection, top-level navigation and clean shutdown; it does **not** exercise every control or visual layout.

## Automated coverage

| Area | What is exercised |
| --- | --- |
| Protocol | ThinkGear frame fragmentation, checksum failures, signed RAW decoding and parser recovery |
| Acquisition | Absolute sample indexing, cue window boundaries, stream corruption and reconnect isolation |
| Preprocessing | Hampel deglitching, synthetic clean signals, impulse rejection, saturation, flatlines, frozen samples, EEG filter edges and Welch scaling |
| Artifacts | Modeled blinks, broadband muscle interference, motion and 50 Hz noise, including held-out simulation seeds |
| BCI | Deterministic 24-feature schema, training, prediction, abstention, three-decision stabilization and serialization validation |
| Persistence | Recording-to-replay consistency and exclusion of RAW epochs that overlap recorded integrity losses |
| UI | Optional Tkinter lifecycle and navigation under a real or virtual display |
| Performance | Informational CPU-only feature extraction throughput; not end-to-end device latency |
| Algorithm equivalence | Scalar Welch reference versus batched FFT (even and odd segment lengths), public filter-response isolation, reproducible processing |

### Interpreting results

- A **passing test** means its assertions passed using its particular synthetic input. It does not establish universal correctness, artifact specificity, or the performance of physical TGAM modules.
- Synthetic subject profiles are reproducible variations of modeled amplitude, frequency, broadband noise and artifact injections. They are **not population data** or recordings of volunteers.
- Models trained on one profile may generalize poorly to another; individualized calibration and held-out validation remain necessary.
- Blink/motion recognition is only heuristic on a single channel. Simulated artifacts may differ from real eye movement, contact drift and EMG.
- A fast isolated feature extractor does not guarantee real-time GUI responsiveness or Bluetooth reliability.

## Checks required before a hardware or human-user performance claim

1. Collect consented, annotated RAW recordings across multiple sessions and users, with independent evaluation data.
2. Verify ADC rate, sustained RAW delivery, packet-loss handling, Bluetooth pairing and reconnect on each supported Windows/Linux target setup.
3. Measure sensitivity/false rejection of ocular, motion, electrode and muscle artifacts on recordings with labels.
4. Measure per-session balanced accuracy, decision rate, false activations per minute, latency and drift under realistic use, with uncertainty intervals.
5. Run prolonged UI, device and recording/replay tests, including crashes, incomplete files and permission errors.

Do not optimize parameters against the final held-out human test data or present synthetic accuracy as physiological decoding accuracy.

## Visual figure-test checks

The GUI test covers the symbol workspace at compact and wide resolutions, toggling
its evidence panel and opening/closing its focused presentation (Escape supported).
Use a virtual display for automated layout checks, and verify rendered fonts and
screen scaling on both Windows and Linux. These checks validate widget geometry
and controls rather than human BCI accuracy.
