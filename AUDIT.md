# Complete repository audit

Audit date: **2026-09-02**

Scope: source code, language catalogs, persistence, acquisition architecture, neural-control UI layout, Bluetooth/Serial separation, hardware configuration documentation, repository structure, packaging/CI and the project-owner photographs supplied with this revision.

## Result

The repository was cleaned and converted to English for source code, comments and maintained documentation while preserving the runtime multi-language system. The audit found one critical persistence defect and one startup robustness defect, plus several maintainability/UI issues. They have been corrected in this revision.

## Findings and corrections

### Critical — neural profile persistence silently failed

`NeuroControlView._persist_profile_safely()` recursively called itself instead of calling the runtime persistence method. The recursion eventually raised `RecursionError`; because the helper caught exceptions internally, outer recursive frames could return as though persistence had succeeded. This could prevent calibration sessions/models/validation metadata from being saved even though the UI continued running.

**Correction:** the helper now calls `self.runtime.persist_profile(profile)` directly and reports only the expected persistence/data exceptions. Automatic save paths now reach `ModelStore` as intended.

### Medium — malformed settings could prevent startup

The settings loader accepted JSON values into the dataclass and only converted/clamped numeric fields afterward. A value such as `"graph_fps": "broken"` could raise during `int()`/`float()` conversion and escape the loader.

**Correction:** bounded numeric coercion now falls back to canonical defaults for malformed values while still enforcing fixed MindFlex settings (57,600 baud and 512 Hz RAW sampling).

### Medium — neural runtime state was mixed into the UI module

Persistence/restore/model readiness lived inside the already large `neuro_ui.py`, making UI code responsible for too much domain state and making defects such as the persistence recursion harder to isolate.

**Correction:** `NeuroRuntime`, profile constants, restore/persist logic and automatic model rebuilding were moved to `neuro_runtime.py`. The view imports and uses the runtime rather than defining it.

### Medium — restore recovery was too fragile

Persisted session/model/validation/metadata artifacts can fail independently (manual edits, interrupted writes from older revisions, incompatible files). Restore is now granular: each artifact is loaded independently, failures are logged, valid sibling artifacts survive, and a missing/corrupt model can be rebuilt from sufficient valid training data.

### Medium — Neural Control layout was text/panel heavy

The previous neural workflow used large step subtitles, large explanatory cards and several equally weighted panels. On 1366×768 and especially 1024×680 this displaced cue figures, command visualization and cursor arenas below the immediately visible workspace.

**Correction:** the neural workflow is now visual-first. The step bar is compact, Setup/Training and Validation use a large primary cue canvas with a narrow side control column, and all three Live Control surfaces prioritize the figure/arena while evidence/actions occupy a secondary pane. `ResponsiveSplitPane` stacks the two areas only when width is insufficient.

### Medium — source-language leakage in transport/diagnostics

Bluetooth status/error strings and some diagnostic fragments were hard-coded in Portuguese, contrary to the repository language architecture.

**Correction:** source and comments are English. User-facing Bluetooth status events use translation keys where applicable. Diagnostic technical fragments are English. Runtime locale catalogs remain available for all supported languages.

### Low — obsolete translation catalog entries

Thirty-two catalog keys belonged to removed Bluetooth dialogs, manual model/session training, capture actions, older validation UI or superseded labels and had no runtime source path.

**Correction:** those keys were removed from all seven catalogs. Catalog parity and format placeholders are validated against English at `Translator` startup.

### Low — stale repository structure

The old repository included `tests/` and `tools/` although this distribution is intended as the clean application repository, and CI depended on those paths.

**Correction:** validation was completed first; then `tests/` and `tools/` were removed as requested. CI was replaced with package installation, compile, locale integrity and import smoke checks that do not depend on removed folders.

## Architecture checks

The audit confirmed these invariants:

- UI modules do not implement FFT/signal processing or instantiate a persistent ThinkGear parser.
- `EEGController` is the sole owner of the persistent parser and canonical EEG state.
- Bluetooth Classic does not route through a COM-port compatibility layer.
- Bluetooth and Serial/USB feed the same byte ingestion/controller pipeline.
- BCI uses RAW epochs rather than Attention/Meditation/`POOR_SIGNAL` summary values.
- Model/session JSON schemas validate feature dimensions and finite numeric values.
- User data is isolated by deterministic per-user profile directory.
- The parser tolerates arbitrary packet fragmentation and rejects bad checksums without losing subsequent valid packets.
- The GUI acquisition/processing path is non-blocking: transport threads feed a separate ingest path and GUI refreshes are scheduled through Tk timers.

## Language-system checks

Supported runtime locales are preserved:

- English (`en`)
- Portuguese — Brazil (`pt_BR`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Italian (`it`)
- Japanese (`ja`)

Every catalog must have exactly the canonical English key set, non-empty string values and identical format placeholders. Language names themselves intentionally remain in their native form in `i18n.py`.

## Hardware cross-check

The expected 57,600-baud RAW configuration was cross-checked against the TGAM1 pad documentation at <https://www.myredstone.top/en/archives/5135>:

- BR1=VCC + BR0=GND → 57.6 kbaud, normal + RAW waveform output.
- M=VCC → 60 Hz notch; M=GND → 50 Hz notch.
- TGAM1 is documented for 2.97–3.63 V operation.
- Connector P1 is the electrode interface, P4 is power and P3 is UART/serial.

The repository now contains `docs/HARDWARE_SETUP.md`, an original English pad diagram and the project-owner photographs.

## Verification completed before removing the old test/tool folders

| Verification | Result |
| --- | --- |
| Existing automated regression suite | **Passed — 77 tests** |
| Python development-mode compile | **Passed** |
| Wheel package build (local build environment) | **Passed** |
| Locale parity/placeholder validation | **Passed for all 7 locales** |
| Layout smoke — 1024×680 | **Passed** |
| Layout smoke — 1366×768 | **Passed** |
| Layout smoke — 1600×900 | **Passed** |
| Malformed-settings recovery check | **Passed** |
| Direct self-recursion static scan after correction | **No recursive persistence helper found** |

The host environment reports an unrelated global `moviepy`/`Pillow` package conflict during `pip check`; neither package is a dependency of this repository. Repository dependency/build validation is therefore performed in the package-specific checks described above rather than treating that host-global conflict as a project failure.

## Validation limitations

The audit environment is not a Windows machine connected to the physical modified MindFlex headset. Consequently, WinRT Bluetooth pairing/service discovery, the exact behavior of the user's Bluetooth/UART adapter, RF conditions, solder-pad continuity and real EEG quality cannot be fully exercised here. These remain final hardware acceptance tests on Windows 10/11 with the actual headset.

Recommended physical acceptance sequence:

1. Confirm BR1/BR0/M bridges and adapter voltage levels with power off.
2. Power the headset and verify live Bluetooth Classic discovery.
3. Connect and confirm RFCOMM/SPP negotiation.
4. Run Diagnostics and verify valid ThinkGear packets, checksum stability and RAW receive rate near 512 samples/s.
5. Record a short session and inspect RAW continuity before performing BCI calibration.
6. Complete training and validation for each desired neural-control profile, then restart the application and confirm automatic session/model restore.

## Repository state after audit

- Source code/comments: English only (except intentional native language display names and UI locale data).
- Maintained documentation: English.
- Multi-language runtime: preserved.
- Legacy translation keys: removed.
- Legacy `tests/` and `tools/` distribution folders: removed after validation.
- Hardware photos: added under `docs/assets/`.
- Hardware setup/pad table: added.
- Neural Control layout: reorganized to prioritize visual components.
- Persistence recursion: fixed.
- Malformed numeric settings recovery: fixed.
