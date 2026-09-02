# Software architecture

MindFlex EEG Studio uses one acquisition and state pipeline regardless of transport:

`Bluetooth Classic / Serial / Simulator → byte ingestion → ThinkGearParser → EEGController → Monitor / Diagnostics / Recorder / BCI`

## Design rules

1. Transport code delivers bytes and owns no persistent EEG interpretation state.
2. `EEGController` owns the single persistent `ThinkGearParser` and the canonical `EEGSnapshot`.
3. Bluetooth Classic uses a persistent WinRT worker for discovery, pairing, RFCOMM/SPP and byte reading. Serial/USB uses `pyserial` through a separate byte transport, but both feed the same controller ingestion path.
4. RAW timing is sample-sequence based at 512 Hz. Host packet-arrival timing is used for liveness/receive-rate sanity gates and diagnostics, not for BCI epoch duration.
5. Native TGAM Attention, Meditation and EEG-band rows are preferred while fresh and non-zero where applicable. A RAW-derived continuity estimate can be exposed as `RAW*` when native summary rows are not usable.
6. BCI features never use Attention, Meditation, `POOR_SIGNAL`, or TGAM summary bands as classifier inputs. Feature extraction operates on RAW epochs.
7. Per-user sessions, models, validation metadata and mental vocabulary are stored under an isolated profile directory.
8. Locale catalogs are data. Source code and comments stay English; the runtime translator preserves all supported UI languages.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `parser.py` | Incremental ThinkGear packet framing, checksum validation and DataRow decoding |
| `controller.py` | Canonical EEG state, RAW buffer, metric source selection, asynchronous byte ingestion and simulator |
| `transport.py` | Serial/USB endpoint discovery and byte transport |
| `bluetooth_transport.py` | Windows Bluetooth Classic discovery, pairing, RFCOMM service probing and continuous byte reading |
| `cognitive.py` | RAW-derived continuity metrics and spectral summaries |
| `bci.py` | RAW feature extraction, training sessions, centroid model, validation and persistence primitives |
| `calibration.py` | Non-blocking balanced trial/epoch scheduler |
| `neuro_runtime.py` | UI-independent neural workflow state and automatic persistence/restore |
| `neuro_ui.py` | Neural workflow presentation and interaction only |
| `neural_visual.py` | Command-test evidence and cursor dynamics |
| `diagnostics.py` | Independent connection, stream, checksum, RAW and metric checks |
| `i18n.py` | Catalog loading, parity validation and runtime translation |
| `ui_components.py` | Reusable responsive layout components |
| `ui.py` | Application shell, Connection, Monitor and Diagnostics pages |

## BCI feature and model contract

Version 1 uses a fixed 512 Hz RAW stream and exact 2.0-second/1024-sample model epochs. Each epoch produces the 18 finite features defined by `mindflex-raw18-v1`: eight log band powers, relative theta/alpha/beta/gamma power, theta/beta and alpha/beta ratios, spectral entropy, spectral centroid, RMS and line length.

Training uses balanced timed trials. Epochs belonging to the same `(label, trial_id)` are averaged first; class-balanced z-score normalization and class centroids are then learned from those independent trial vectors. This prevents overlapping epochs from giving one trial more training weight than another.

Validation follows the same independence rule and classifies one mean vector per trial. Quick, Standard and Research collect 3, 5 and 8 validation trials per class respectively. Approval is centralized in `WorkflowRules` and requires at least 70% global accuracy, 60% in every class and 3 independent trials per class. Feature capture additionally requires a complete fresh RAW buffer, an observed host rate of 400–620 samples/s and recent RAW spread ≥3.

The normative model description is [MODEL_STRATEGY.md](MODEL_STRATEGY.md).

## Persistence

`UserProfile` creates a deterministic per-user directory. `ModelStore` uses size-bounded, schema-checked, atomic JSON for models, timed calibration sessions and metadata. Model artifacts carry a SHA-256 fingerprint of the exact trial-balanced training inputs; validation artifacts carry both the exact model fingerprint and the exact trial-balanced validation-session fingerprint.

`NeuroRuntime` restores each artifact independently. A model whose training fingerprint does not match its saved session is discarded. A persisted validation is admitted only after its saved validation session is reclassified against the exact restored model and the recomputed result matches the stored result. Runtime readiness also checks the current in-memory training and validation fingerprints, so replacing/mutating either session fails closed. A sufficient valid training session can rebuild an absent or rejected automatic model. Frequent UI readiness polling is consolidated into one integrity pass per profile to avoid repeating the same fingerprint work within a single poll.

Version 1 has no alternate artifact-compatibility path: schema, feature, algorithm and sampling identifiers must match the Version 1 contract exactly.

## Responsive UI policy

The Neural Control workspace is visual-first. `ResponsiveSplitPane` gives the cue/arena/recognized-command canvas the primary width while actions, evidence and explanatory copy occupy a narrower secondary column. At narrow widths the panes stack instead of compressing the figure area. The workflow step bar can hide subtitles and wrap by minimum step width rather than fixed-resolution assumptions.
