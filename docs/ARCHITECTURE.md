# Architecture

MindFlex EEG Studio Version 1 has one acquisition architecture and one BCI signal path.

## Runtime flow

`Transport → ingestion buffer → ThinkGearParser → EEGController → consumers`

Transports only deliver bytes. `ThinkGearParser` interprets the protocol. `EEGController` owns the current state, RAW ring buffer, absolute RAW sample counter and derived monitoring metrics. UI, diagnostics, recorder and BCI consumers read that shared state.

## Fixed hardware rules

The MindFlex/TGAM path is fixed at:

- 57,600 baud;
- 512 RAW samples/s;
- one persistent ThinkGear parser per controller;
- one bounded RAW ring buffer with absolute sample indexing.

These values are defined once in `mindflex/settings.py` and are not configurable by BCI screens or calibration protocols.

## BCI signal path

The classifier consumes RAW only:

`RAW → preprocessing/artifact gate → overlapping Welch PSD → 24 features → class-balanced shrinkage-LDA model`

The fixed feature identity is `mindflex-v1-raw24-welch`.

Preprocessing operates on exactly 768 samples (1.5 s). It detrends and band-limits the epoch to 0.5–45 Hz, removes only isolated sample spikes, and rejects strongly ocular- or muscle-dominated windows rather than applying aggressive blanket denoising. New decision windows begin every 128 samples (0.25 s). `FeatureExtractor.from_raw()` rejects any input that is not exactly one canonical epoch.

Welch PSD uses fixed overlapping Hann segments. The 24 features are eight log band powers, relative delta/theta/alpha/beta/gamma power, theta/beta and alpha/beta ratios, spectral entropy, spectral centroid, total power, engagement, 90% spectral edge, spectral flatness, Hjorth mobility/complexity and line length.

TGAM Attention, Meditation and band summaries are monitoring/diagnostic telemetry only. There is no alternate BCI feature source.

## Absolute sample windows

`EEGController.raw_total_samples` is the stream-wide sample counter. `raw_slice(start, end)` returns only that exact interval. It never substitutes a trailing window if a requested interval is unavailable.

`bci_window(boundary, index)` defines every cue-bound window:

- `start = boundary + index × 128`
- `end = start + 768`

`latest_bci_window()` applies the same rule to continuous live control.

## Cue boundary rule

For every cue-driven activity, the UI renders the cue first and then captures the current absolute RAW sample counter. That sample is the only cue boundary used by feature extraction.

Training, validation and the blind figure/arrow test therefore cannot accept a window containing samples from PREPARE or the previous cue.

## Calibration

`CalibrationProtocol` controls only:

- training trials per class;
- validation trials per class;
- prepare duration;
- task duration;
- randomized rest range.

Epoch length and step are not protocol fields. `CalibrationEngine` requests sequential fixed-grid sample windows and can recover exact missed windows from the RAW ring buffer after a short GUI delay.

Model fitting uses the complete epoch matrix from each trial, but weighting is hierarchical: every class contributes equally, every trial inside a class contributes equally, and epochs only refine that trial’s statistics. A long trial can therefore never dominate the model merely because it contains more overlapping epochs. Validation keeps the chronological epoch sequence intact and replays it through the production decision pipeline.

## Model and decision rule

`BCIModel` implements the Version 1 class-balanced shrinkage-LDA model. Normalization uses pooled within-class variation with a bounded total-variance floor. A shared covariance matrix is then estimated with equal class/trial weighting and shrunk toward its diagonal with ridge regularization before inversion. This preserves useful feature correlations while remaining stable with relatively few independent trials. Posterior distance scaling is calibrated from trial representatives so low-separation training data remains appropriately uncertain. Validation is trial-balanced and requires the centralized thresholds from `WorkflowRules`.

Every live BCI output uses `PredictionStabilizer`, which averages exactly three consecutive posterior distributions. Evidence is normalized against the random-chance probability for the current class count, and a decision is emitted only when evidence is at least 20% above that chance baseline.

Every cue-based validation/test uses `TrialDecisionAccumulator`. It feeds raw model predictions through that same stabilizer and resolves the complete trial from the stabilized posterior evidence. Validation, figure/arrow testing and laboratory folds therefore share one decision rule. A trial with no emitted decision is scored as incorrect and contributes to the reported decision rate.

Validation approval requires global and balanced accuracy, per-class accuracy, minimum independent trials and minimum decision rate. There are no per-screen stabilization windows, cursor-specific confidence thresholds or test-specific voting formulas.

## Persistence

Version 1 calibration sessions use `mindflex-v1-raw-fixed-grid` and store, for every epoch:

- label;
- feature vector;
- timestamp;
- trial id;
- epoch index;
- absolute RAW start sample;
- absolute RAW end sample.

Session validation checks that every epoch contains exactly 768 samples and that all saved epochs in a trial lie on the same 128-sample grid. The session fingerprint covers every validated epoch, its exact RAW bounds and feature vector. Models and validation metadata are fingerprint-bound to their source sessions. Validation metadata stores global accuracy, balanced accuracy, decision rate, per-class counts and per-class decision counts, and is recomputed before restored results are trusted. All runtime artifacts live under the program-local `sessions/` directory.

### Native laboratory recording

`SessionRecorder` subscribes to the controller's exact RAW acquisition callback, not to GUI refreshes. Each `.mfs` recording is one compressed Version 1 container with:

- original signed 16-bit RAW samples;
- one absolute start sample for the continuous RAW stream;
- 512 Hz sample-rate identity;
- throttled Attention/Meditation/band/contact telemetry;
- packet/checksum/drop counters;
- cue, trial and calibration events aligned to absolute RAW indices.

Training and validation start this recorder automatically. `trial_start` is written at the same post-render RAW boundary used by online feature extraction, and `trial_end` closes the exact task interval. The Laboratory reconstructs the fixed sample grid from those events, re-extracts features from original RAW, rejects the same artifacts and runs the same cross-validation/decision path. Any RAW discontinuity invalidates the recording; replay never substitutes or shifts another time interval.

## Module responsibilities

- `settings.py`: fixed hardware and BCI constants plus UI settings.
- `transport.py` / `bluetooth_transport.py`: byte acquisition.
- `parser.py`: ThinkGear protocol parsing.
- `controller.py`: canonical EEG state, RAW buffer and exact RAW recording callbacks.
- `signal_processing.py`: RAW preprocessing.
- `bci.py`: exact windows, features, model, stabilization and persistence.
- `calibration.py`: balanced trial scheduler.
- `rules.py`: readiness, validation and workflow thresholds.
- `neuro_runtime.py`: persisted neural workflow state.
- `neuro_ui.py`: rendering, cue-boundary orchestration and automatic BCI recording events.
- `lab.py`: native RAW recording, strict replay, offline BCI reconstruction and trial-level experiments.
- `neural_visual.py`: figure test state and cursor dynamics.
- `mental_text.py`: communication vocabulary/output adapter.

## General evaluation rules

- `SignalPolicy` owns the single acquisition-quality gate for neural operations.
- Diagnostics expose connection, checksum, RAW, contact and telemetry independently, then run the exact same neural-capture gate and a real canonical feature-extraction window used by the operational workflow.
- Automatic validation, blind command tests and laboratory cross-validation all resolve trials through the same production decision pipeline.
- Laboratory experiments use deterministic stratified trial-level folds, so epochs from one trial never leak between training and validation and a single temporal split cannot dominate the result.
