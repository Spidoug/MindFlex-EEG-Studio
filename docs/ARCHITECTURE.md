# Software architecture

MindFlex EEG Studio uses one acquisition and state pipeline regardless of transport:

`Bluetooth Classic / Serial / Simulator → byte ingestion → ThinkGearParser → EEGController → Monitor / Diagnostics / Recorder / BCI`

## Design rules

1. Transport code delivers bytes and owns no persistent EEG interpretation state.
2. `EEGController` owns the single persistent `ThinkGearParser` and the canonical `EEGSnapshot`.
3. Bluetooth Classic uses a persistent WinRT worker for discovery, pairing, RFCOMM/SPP and byte reading. Serial/USB uses `pyserial` through a separate byte transport, but both feed the same controller ingestion path.
4. RAW timing is sample-sequence based at 512 Hz. Host packet-arrival timing is used for liveness and receive-rate diagnostics, not for BCI epoch duration.
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

## BCI feature contract

Each RAW epoch produces 18 finite features: log power for the eight ThinkGear band ranges, relative theta/alpha/beta/gamma power, theta/beta and alpha/beta ratios, spectral entropy, spectral centroid, RMS and line length. The classifier uses z-score normalization and per-class centroids.

Training uses balanced timed trials. Validation aggregates by independent trial so overlapping epochs from the same cue do not inflate validation accuracy.

## Persistence

`UserProfile` creates a deterministic per-user directory. `ModelStore` uses schema-checked JSON for models, timed calibration sessions and metadata. `NeuroRuntime` restores each artifact independently so a malformed optional artifact does not hide a valid training session; when a valid session is sufficient but the model file is absent or unusable, the runtime can rebuild the automatic model.

## Responsive UI policy

The Neural Control workspace is visual-first. `ResponsiveSplitPane` gives the cue/arena/recognized-command canvas the primary width while actions, evidence and explanatory copy occupy a narrower secondary column. At narrow widths the panes stack instead of compressing the figure area. The workflow step bar can hide subtitles and wrap by minimum step width rather than fixed-resolution assumptions.
