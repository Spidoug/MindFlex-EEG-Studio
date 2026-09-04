# MindFlex EEG Studio

MindFlex EEG Studio is a desktop application for MindFlex/TGAM EEG acquisition, monitoring, diagnostics, BCI calibration, validation, live neural control and communication experiments.

Version 1 uses one deterministic BCI path from end to end:

`RAW 512 Hz → exact 1.5 s window → selective artifact gate → Welch PSD + 24 features → shrinkage-LDA model → 3-decision stabilizer`

TGAM Attention, Meditation and band-power summaries remain available for monitoring and diagnostics, but they do not feed the BCI classifier.

![MindFlex headset](docs/assets/mindflex_headset.jpg)

## Screenshots

| Connection and Bluetooth discovery | Live EEG monitor |
| --- | --- |
| ![Connection and Bluetooth discovery](docs/assets/screenshots/connection.png) | ![Live EEG monitor](docs/assets/screenshots/monitor.png) |

| Neural Control training | Neural Control live command workspace |
| --- | --- |
| ![Neural Control training](docs/assets/screenshots/neuro-control-training.png) | ![Neural Control live command workspace](docs/assets/screenshots/neuro-control-live.png) |

### Diagnostics

![Sequential diagnostics](docs/assets/screenshots/diagnostics.png)

## Hardware

MindFlex EEG Studio expects the TGAM1 stream at **57,600 baud** with **RAW output at 512 samples/s**. The documented startup-pad combination is **BR1=VCC, BR0=GND**. The M pad selects the hardware notch filter: **VCC=60 Hz**, **GND=50 Hz**.

Connector tables, voltage precautions, photographs and the pad diagram are in [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md).

## Requirements

- Python 3.10 or newer.
- MindFlex/TGAM RAW stream at 512 samples/s.
- Windows 10/11 for direct Bluetooth Classic discovery and RFCOMM/SPP through WinRT.
- Or a supported Serial/USB endpoint for explicit serial acquisition.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run on Windows:

```bat
RUN_STUDIO.bat
```

Or directly:

```bash
python -m mindflex
```

The Unix shell launcher is included for simulator/development use:

```bash
./RUN_STUDIO.sh
```

## Data architecture

The core flow is:

`Transport → ingestion buffer → ThinkGearParser → EEGController → Monitor / Diagnostics / Recorder / BCI`

General rules:

- Transport code only delivers bytes.
- `EEGController` is the single owner of parsed EEG state and the absolute RAW sample counter.
- BCI timing is defined by sample indices, never by USB/Bluetooth packet size or GUI callback timing.
- Every BCI mode uses the same fixed 512 Hz RAW source.
- Every BCI epoch contains exactly **768 samples (1.5 s)**.
- Consecutive decisions are separated by exactly **128 samples (0.25 s)**.
- A cue becomes active only after it is rendered; the RAW sample counter captured at that point becomes the cue boundary.
- A cue-bound epoch can never start before that boundary.
- Training, validation, figure/arrow testing, cursor control, concentration output and communication use the same feature extractor and temporal decision rule.
- Live decisions and validation use the same 3-frame posterior stabilizer. Evidence is measured above the random-chance baseline, so the same 20% minimum evidence rule is meaningful for both 2-class and multi-class models.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module-level responsibilities and persistence rules.

## Bluetooth Classic

The Connection page maintains a live Windows `DeviceWatcher` list. After a device is selected, the Bluetooth worker pairs if required, opens the `BluetoothDevice`, discovers the Serial Port Profile / RFCOMM service (`0x1101`), probes candidate channels and accepts a route only after valid ThinkGear packets are observed.

Bluetooth mode does not require a virtual COM port. `pyserial` is used only for explicit Serial/USB acquisition.

## Monitor and diagnostics

The Monitor displays RAW activity, Attention, Meditation and eight EEG bands. Native TGAM metrics and locally derived continuity metrics are identified by source.

Diagnostics checks connection, fixed 57,600-baud configuration, ThinkGear framing/checksums, RAW activity/rate, sensor contact, Attention, Meditation and EEG bands independently. It also runs the **Neural capture gate** and then executes a real 1.5 s BCI feature window through the same preprocessing/artifact gate and 24-feature extractor used by every model. A failed diagnostic does not prevent the remaining checks from running.

## BCI calibration and validation

Training uses balanced timed trials:

`Prepare → mental task → exact RAW epochs → rest`

The protocol presets only change trial count and phase duration. They cannot change BCI window size, decision step or stabilization behavior.

| Mode | Training trials/class | Validation trials/class | Prepare | Task | Approx. epochs/trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quick | 4 | 3 | 1.5 s | 4 s | 11 |
| Standard | 8 | 5 | 2 s | 5 s | 15 |
| Research | 15 | 8 | 3 s | 6 s | 19 |

Each RAW epoch is detrended, band-limited to 0.5–45 Hz, conservatively rejected when strongly dominated by probable ocular/muscle contamination, and analyzed with overlapping Welch periodograms. The fixed 24-feature vector contains eight log band powers, relative delta/theta/alpha/beta/gamma power, theta/beta and alpha/beta ratios, spectral entropy, spectral centroid, total power, engagement, 90% spectral edge, spectral flatness, Hjorth mobility/complexity and line length.

Training keeps every class and every independent trial equally weighted, then learns a shared-covariance LDA model regularized by shrinkage toward the diagonal plus a small ridge. The model therefore uses correlations between useful features without allowing the limited number of independent trials to make the covariance unstable. Posterior sharpness is calibrated from independent trial representatives so weakly separated data remains uncertain instead of becoming artificially confident. Validation does not use a separate trial-average classifier: it replays each trial epoch-by-epoch through the same three-decision stabilizer used by live control.

Every cue-based evaluation uses the same trial evaluator: raw model posteriors → global 3-decision stabilizer → evidence above chance → one resolved trial. The same evaluator is used by automatic validation and the blind figure/arrow test. A missing decision is counted as an error and is also reported separately through **decision rate**, so an abstaining model cannot obtain a misleadingly high score.

Validation is scored once per independent trial. A model is approved only when it reaches at least **70% global accuracy**, **70% balanced accuracy**, **60% accuracy in every class**, **80% decision rate**, and at least **3 independent validation trials per class**. Model/session integrity is bound by SHA-256 fingerprints.

Version 1 artifacts are strict: models, feature sessions, validation metadata and laboratory recordings must match the current Version 1 schemas exactly. The application does not convert or reinterpret artifacts from another signal pipeline.

## Cue synchronization

Cue-driven tasks use the display event itself as the EEG boundary. The symbol is rendered first; then the current absolute RAW sample index is captured. All windows for that cue are calculated from that sample index on the global 128-sample step grid.

This prevents a prediction created from the previous symbol from being scored as the new symbol. If the UI is briefly delayed, exact RAW intervals can still be recovered from the controller buffer instead of duplicating or shifting a newer window.

## Neural Control workspace

The Neural Control workspace contains Setup & Training, Validation, Live Control, Communication and Laboratory stages. The figure/arrow test, mental cursor and communication surface all use the same fixed BCI timing and stabilization rules. Laboratory model experiments use **trial-level stratified cross-validation**; every fold is evaluated by the same production validation routine.

Training, validation and blind cursor tests automatically create native `.mfs` laboratory recordings. An `.mfs` file stores the original continuous signed 16-bit RAW stream at 512 Hz with its absolute sample origin, throttled telemetry, checksum/drop counters and exact cue/trial events. The Laboratory can rebuild canonical 1.5 s/0.25 s epochs directly from the saved RAW, rerun the current 24-feature extractor and classifier offline, and report accuracy, balanced accuracy, decision rate, per-class accuracy and a confusion matrix.

## Languages

Runtime catalogs are included for English, Brazilian Portuguese, Spanish, French, German, Italian and Japanese. Source code and comments remain in English; localized UI text belongs in `mindflex/locales/*.json`.

## User data

All runtime data is portable and stays beside the program under `sessions/`. Per-person calibration data, models, validation metadata and native `.mfs` RAW recordings are stored under `sessions/users/<user>/`; monitor recordings and automatic training/validation recordings are created in that user’s `recordings/` directory. `sessions/` is ignored by Git so running the application does not pollute the repository.

Persisted artifacts are schema-checked, owner-scoped and fingerprint-validated before use.

## Safety and intended use

MindFlex EEG Studio is an experimental EEG/BCI software project. It is not a medical device and must not be used for diagnosis or medical decision-making. Disconnect power before hardware modification and verify TGAM1/adapter voltage compatibility before connection.
