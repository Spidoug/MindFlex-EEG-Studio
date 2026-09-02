# Backend audit — Version 1

Audit date: **2026-09-02**

Scope: acquisition pipeline, ThinkGear parsing, Bluetooth/Serial transport boundaries, RAW buffering, calibration scheduling, BCI feature/model contract, model validation, persistence/restore, user isolation, laboratory CSV handling, settings, communication vocabulary, packaging and CI-facing backend checks.

## Version 1 policy

This repository is **MindFlex EEG Studio 1.0.0**. The maintained backend has one Version 1 artifact/model contract. It does not contain an alternate compatibility path, threshold fallback or migration routine for different model/session contracts.

The normative model strategy is documented in [docs/MODEL_STRATEGY.md](docs/MODEL_STRATEGY.md).

## Audit result

The backend was hardened around four invariants:

1. one canonical byte/parser/controller pipeline;
2. complete, fixed-duration RAW epochs for BCI;
3. independent-trial training/validation with strict model/session identity;
4. fail-closed persistence and restore.

The corrections below are included in the audited tree.

## Corrected findings

### Acquisition queue could grow without a hard limit

The unified byte ingestion path could accumulate data indefinitely if the consumer fell behind the producer.

**Correction:** the queue is bounded to 256 KiB. Overflow keeps the newest bytes, counts discarded bytes and forces ThinkGear framing to resynchronize before parsing the retained stream. The dropped-byte counter remains visible in the canonical snapshot for diagnostics.

### A dequeued byte batch could cross a connection reset

Clearing the shared ingest buffer was not enough to protect a batch that had already been dequeued by the worker when a transport/source reset occurred.

**Correction:** byte batches now carry a stream-generation identifier. A batch from a previous generation is discarded before it can update the parser/controller state of a new connection.

### Parser reset and parser feed could race

The persistent ThinkGear parser could previously be reset by connection lifecycle code while feed processing was running on the ingest thread.

**Correction:** parser feed and stream-state reset are serialized under the controller state lock. `discard_partial()` is used only for overflow resynchronization and preserves packet/checksum diagnostic counters.

### Failed Bluetooth channel probes could contaminate canonical EEG state

Bytes from a candidate RFCOMM channel could enter the controller before the channel had been confirmed as ThinkGear.

**Correction:** probe bytes are held locally. They are forwarded to the canonical controller only after the probe parser observes the required valid ThinkGear packets. Failed candidates are discarded completely.

### BCI capture could accept incomplete epochs

The previous feature API could produce a model vector before a full configured epoch was present.

**Correction:** Version 1 has one fixed BCI input contract: **2.0 s × 512 Hz = 1024 RAW samples**. BCI readiness uses the actual buffered-sample count, and feature extraction rejects incomplete epochs, non-finite data, flat data and values outside the signed 16-bit TGAM range.

### A full RAW buffer alone did not prove a healthy live 512 Hz stream

A stale/bursty source could theoretically accumulate 1024 samples and satisfy the old BCI buffer-count condition even when the host-observed arrival cadence was no longer compatible with the MindFlex RAW stream.

**Correction:** the canonical BCI acquisition gate now requires all of: a fresh stream, at least 1024 buffered RAW samples, a recent host-observed RAW rate between 400 and 620 samples/s, and a minimum recent RAW spread of 3 counts. These are transport-continuity/sanity gates; feature extraction remains mathematically fixed at 512 Hz.

### Epoch duration was not part of a strict feature contract

A caller could provide a different epoch duration while retaining the same 18-dimensional feature shape.

**Correction:** `mindflex-raw18-v1` requires exactly 2.0-second epochs. Calibration protocol validation, feature extraction and session persistence enforce the same duration.

### Delayed calibration callbacks could capture outside the task phase

A late GUI callback could use a current trailing RAW window for an epoch whose scheduled time had already passed, including after the mental-task deadline.

**Correction:** capture is allowed only while the scheduler is in `TASK` and the callback time is still inside the scheduled task deadline. Missed historical windows are skipped instead of being reconstructed from current data.

### Calibration trial IDs could collide or be reused

Time-derived trial IDs could theoretically collide, and restarting a completed/cancelled engine could reuse its existing plan.

**Correction:** each run receives a fresh unique run identifier and fresh trial IDs. Restarting rebuilds the plan.

### Epoch-heavy trials could dominate model training

When overlapping epochs are treated as independent training rows, a trial that produces more valid windows has more influence than a trial with fewer valid windows.

**Correction:** `class-balanced-trial-centroid-zscore-v1` first averages epochs inside each independent trial, then performs class-balanced normalization and centroid training across trial vectors. Every trial contributes equally inside its class, and every class contributes equally to feature scaling.

### Model/session persistence could become logically inconsistent after a partial write sequence

Per-profile artifacts are separate files. If a save sequence stopped after the session changed but before a new model was safely stored, a structurally valid old model could otherwise coexist with newer training data.

**Correction:** every model stores a SHA-256 fingerprint of the exact trial-balanced training inputs. Restore rejects a model whose training fingerprint differs from the current saved session, and automatic model rebuilding can regenerate it from sufficient valid training data.

### Validation metadata could not be allowed to act as an approval token

A stored accuracy object must not be sufficient by itself to unlock live use.

**Correction:** validation is bound to the exact model fingerprint. On restore, the saved validation session is reclassified against the restored model and the complete recomputed result must exactly equal the saved result before it is admitted to runtime state.

### In-memory validation state could outlive mutation of its validation session

Persistence restore already recomputed validation, but mutable runtime dataclasses meant an external caller could alter the validation session after approval while leaving the prior `ValidationResult` object in memory.

**Correction:** every validation result now carries the SHA-256 fingerprint of the exact trial-balanced validation session. Runtime approval requires that fingerprint to match the current in-memory validation session, so mutation or replacement immediately invalidates approval.

### Mutable session objects could bypass readiness checks before persistence

Strict save/load validation alone was not sufficient because dataclass fields remain mutable after collection. A malformed mutation could reach readiness code before a save cycle.

**Correction:** training and validation readiness now first validate the complete in-memory trial structure, including canonical labels/trial IDs, finite timing/features, fixed epoch duration and unique `(label, trial_id, epoch_index)` tuples. Invalid mutated state fails closed.

### Calibration inputs still allowed permissive Python coercions

`bool` values and arbitrary objects converted through `str(...)` could satisfy parts of the calibration API even though they were outside the Version 1 data contract.

**Correction:** protocol trial counts use exact integer types; timing/timestamps reject booleans and non-finite values; calibration labels must already be canonical strings rather than being silently converted.

### Validation quality gate needed per-class and sample-count constraints

Global accuracy alone can conceal a class that performs poorly.

**Correction:** approval is centralized in `WorkflowRules` and requires all of the following: global accuracy ≥70%, every class ≥60%, at least 3 independent validation trials/class, exact expected labels and internally consistent trial/correct counts. Live Control and Communication use this same rule with no alternate threshold parameter.

### Model artifacts needed an explicit Version 1 mathematical contract

Matching only vector length is not sufficient to prove two model artifacts mean the same thing.

**Correction:** persisted models/sessions declare and strictly verify schema, feature schema, sample rate, feature names, sampling strategy and model algorithm. Model values and dimensions must be finite/valid before the model becomes ready.

### Numeric overflow paths could produce invalid inference state

Finite-but-extreme values can overflow when averaged, normalized or converted to distances.

**Correction:** trial means, model normalization, normalized training arrays, centroids, prediction normalization, distances and normalized evidence are all checked for finite results. Invalid numeric state raises instead of producing a prediction.

### User ownership comparison was not canonically consistent

The deterministic user directory is case-insensitive by identity, while an artifact owner check could differ by Unicode/case spelling.

**Correction:** user identity and artifact ownership use Unicode normalization, whitespace normalization and canonical case-folded comparison. Profile names are bounded and control characters are rejected.

### Artifact names could be sanitized instead of rejected

Silently removing unsupported path/name characters can map two different caller inputs to the same persisted artifact name and hides malformed input.

**Correction:** Version 1 artifact names now reject non-string and unsupported characters instead of deleting them. Session names must also be canonical on both save and restore.

### JSON write/read constraints were asymmetric

Read paths had size protection while writes could theoretically create an artifact larger than the corresponding loader would accept.

**Correction:** the same artifact-size limits are enforced before atomic writes and during reads. JSON rejects non-finite values, is written through a temporary file, flushed/fsynced and atomically replaced. On POSIX filesystems that support directory `fsync`, the containing directory entry is synchronized after replacement as an additional crash-durability measure.

### Session replay accepted an open-ended CSV header

Replay required known fields but did not reject additional declared columns.

**Correction:** replay now requires the exact recorder column set, rejects duplicate/missing/extra columns, requires finite numeric values, preserves integer semantics and enforces eSense/blink/band ranges.

### Laboratory holdout fraction was not explicitly bounded

Invalid holdout fractions could fail indirectly or produce surprising behavior.

**Correction:** the holdout fraction must be finite and strictly between 0 and 1.

### Frequent readiness polling repeated the same integrity work

The desktop polls runtime state every 100 ms. Calling model-ready, any-validated, live-ready and communication-ready separately caused the same training/validation fingerprints to be recomputed multiple times in one poll.

**Correction:** `NeuroRuntime.readiness_flags()` evaluates each profile once and derives all four readiness flags from that pass. On the audit machine, a synthetic Research-size five-class cursor profile dropped from roughly 9.3 ms to 3.4 ms for an equivalent readiness poll while preserving all integrity checks.

### Communication had a reserved-label collision

`__silence__` is the interpreter's internal no-command state but could be registered as a user vocabulary label.

**Correction:** the label is reserved and rejected by `MentalVocabulary`. Vocabulary label/phrase sizes and interpreter confidence/window parameters are validated.

## Version 1 BCI/validation summary

| Rule | Version 1 |
| --- | --- |
| RAW sample rate | 512 Hz fixed |
| Live RAW sanity gate | 400–620 samples/s observed + spread ≥3 |
| Epoch | 2.0 s / 1024 RAW samples |
| Features | 18 (`mindflex-raw18-v1`) |
| Training algorithm | trial-balanced z-score + class centroid |
| Minimum training readiness | 4 trials/class + 8 epochs/class |
| Quick validation | 3 trials/class |
| Standard validation | 5 trials/class |
| Research validation | 8 trials/class |
| Global approval | ≥70% |
| Per-class approval | ≥60% each |
| Minimum validation count | ≥3 independent trials/class |
| Validation unit | one decision per independent trial |
| Model/session binding | SHA-256 training fingerprint |
| Validation/model binding | SHA-256 model fingerprint + recomputation |
| Validation/session binding | SHA-256 validation-session fingerprint |

## Verification performed

The audited backend was exercised with 59 deterministic integration/contract checks without leaving generated test artifacts in the repository. The checks cover:

- compilation/import of core modules;
- exact 1024-sample feature contract and rejection of incomplete RAW epochs;
- 18-feature finite extraction on a synthetic in-range EEG waveform;
- class-balanced trial-level model training;
- independent-trial validation and strict approval rules;
- live invalidation when training or validation sessions are mutated after model/approval creation;
- RAW readiness gating by buffer depth, freshness, observed rate and signal spread;
- strict calibration label/count/timestamp typing and non-finite rejection;
- consolidated runtime readiness evaluation with a Research-size synthetic performance check;
- model/session/validation persistence and restore;
- canonical owner matching;
- rejection of incompatible model contracts;
- inconsistent validation-metadata rejection through recomputation;
- unique calibration plans after restart;
- prevention of post-task delayed epoch capture;
- ThinkGear partial-frame discard and resynchronization;
- bounded ingest overflow accounting;
- stream-generation rejection for stale byte batches;
- locale-catalog parity and format-placeholder checks;
- Python package compilation and the embedded Version 1 backend contract checks;
- review of broad exception handlers: remaining broad catches are confined to external callback/event-loop/transport/UI cleanup boundaries where exceptions must not escape, and backend boundary failures are logged/reported rather than silently treated as valid state;
- Wheel build without build isolation, using the dependencies already installed in the audit environment.

Temporary caches/build outputs created by verification are removed from the final repository archive.

## Hardware/platform limitation

This audit environment is not a Windows machine connected to the modified physical MindFlex/TGAM headset. The following remain physical acceptance checks rather than software-only claims:

1. Windows WinRT DeviceWatcher behavior on the target adapter;
2. pairing and RFCOMM/SPP service negotiation with the actual device;
3. sustained physical RAW throughput near 512 samples/s under real RF conditions;
4. TGAM pad/solder continuity, adapter voltage compatibility and electrode quality;
5. real-person calibration separability and repeatability.

The recommended hardware acceptance flow is documented in [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md).
