# Software architecture

MindFlex EEG Studio uses one acquisition and state pipeline regardless of transport:

`Bluetooth Classic / Serial / Simulator → byte ingestion → ThinkGearParser → EEGController → Monitor / Diagnostics / Recorder / BCI`

## Design rules

1. Transport code delivers bytes and owns no persistent EEG interpretation state.
2. `EEGController` owns the single persistent `ThinkGearParser` and the canonical `EEGSnapshot`.
3. Bluetooth Classic uses a persistent WinRT worker for discovery, pairing, RFCOMM/SPP and byte reading. Serial/USB uses `pyserial` through a separate byte transport, but both feed the same controller ingestion path.
4. RAW timing is sample-sequence based at 512 Hz. Host packet-arrival timing is used for liveness/receive-rate sanity gates and diagnostics, not for BCI epoch duration.
5. Native TGAM Attention, Meditation and EEG-band rows are preferred while fresh and non-zero where applicable. A RAW-derived continuity estimate can be exposed as `RAW*` when native summary rows are not usable.
6. BCI features use native TGAM bands when available and fall back to a complete filtered RAW window. `POOR_SIGNAL` is exposed as Contact quality telemetry, not as a blocking gate or classifier feature.
7. Per-user sessions, models, validation metadata and mental vocabulary are stored under an isolated profile directory.
8. Locale catalogs are data. Source code and comments stay English; the runtime translator preserves all supported UI languages.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `parser.py` | Incremental ThinkGear packet framing, checksum validation and DataRow decoding |
| `controller.py` | Canonical EEG state, RAW buffer, metric source selection, asynchronous byte ingestion and simulator |
| `signal_processing.py` | Shared artifact validation, detrending, band-pass and mains-notch response |
| `transport.py` | Serial/USB endpoint discovery and byte transport |
| `bluetooth_transport.py` | Windows Bluetooth Classic discovery, pairing, RFCOMM service probing and continuous byte reading |
| `cognitive.py` | RAW-derived continuity metrics and spectral summaries |
| `bci.py` | EEG-band feature extraction, training sessions, centroid model, validation and persistence primitives |
| `calibration.py` | Non-blocking balanced trial/epoch scheduler |
| `neuro_runtime.py` | UI-independent neural workflow state and automatic persistence/restore |
| `neuro_ui.py` | Neural workflow presentation and interaction only |
| `neural_visual.py` | Command-test evidence and cursor dynamics |
| `diagnostics.py` | Independent connection, stream, checksum, RAW and metric checks |
| `i18n.py` | Catalog loading, parity validation and runtime translation |
| `ui_components.py` | Reusable responsive layout components |
| `ui.py` | Application shell, Connection, Monitor and Diagnostics pages |

## BCI feature processing and model

Version 1 uses the fixed feature identity `mindflex-eeg-band18-v1`. Each complete EEG-band update produces 18 finite features: eight log band powers, relative theta/alpha/beta/gamma power, theta/beta and alpha/beta ratios, band entropy, band centroid, total power and engagement index.

Native TGAM bands can be consumed directly. When RAW is the available source, `signal_processing.py` validates range, clipping and impulses, detrends the signal, and applies a raised-cosine zero-phase 0.5–50 Hz response with a 60 Hz notch before the controller calculates the same eight bands. Training uses balanced timed trials. Updates belonging to the same `(label, trial_id)` are averaged first; class-balanced z-score normalization and class centroids are then learned from those independent trial vectors.

Validation follows the same independence rule and classifies one mean vector per trial. Quick, Standard and Research collect 3, 5 and 8 validation trials per class respectively. Approval is centralized in `WorkflowRules` and requires at least 70% global accuracy, 60% in every class and 3 independent trials per class. Feature capture requires either recent finite nonzero values for all eight bands or a complete active RAW window.

## Persistence

`UserProfile` creates a deterministic per-user directory. `ModelStore` uses size-bounded, schema-checked, atomic JSON for models, timed calibration sessions and metadata. Model artifacts carry a SHA-256 fingerprint of the exact trial-balanced training inputs; validation artifacts carry both the exact model fingerprint and the exact trial-balanced validation-session fingerprint.

`NeuroRuntime` restores each artifact independently. A model whose training fingerprint does not match its saved session is discarded. A persisted validation is admitted only after its saved validation session is reclassified against the exact restored model and the recomputed result matches the stored result. Runtime readiness also checks the current in-memory training and validation fingerprints, so replacing/mutating either session fails closed. A sufficient valid training session can rebuild an absent or rejected automatic model. Frequent UI readiness polling is consolidated into one integrity pass per profile to avoid repeating the same fingerprint work within a single poll.

Schema, feature, algorithm and sampling identifiers must match the Version 1 format exactly.

## Responsive UI policy

The Neural Control workspace is visual-first. `ResponsiveSplitPane` gives the cue/arena/recognized-command canvas the primary width while actions, evidence and explanatory copy occupy a narrower secondary column. At narrow widths the panes stack instead of compressing the figure area. The workflow step bar can hide subtitles and wrap by minimum step width rather than fixed-resolution assumptions.
