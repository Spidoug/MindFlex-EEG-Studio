# Model strategy — Version 1

This document is the normative BCI/model contract for **MindFlex EEG Studio 1.0.0**.

Version 1 has one model path only. There is no compatibility or migration layer for other artifact contracts: a persisted artifact is accepted only when its Version 1 schema, feature contract, algorithm identifier and integrity checks all match exactly.

## 1. Signal contract

The BCI consumes only the TGAM/MindFlex RAW waveform.

- RAW sample rate: **512 Hz**.
- Model epoch duration: **2.0 seconds** exactly.
- Samples per model epoch: **1024** exactly (`512 × 2`).
- RAW values must be finite and inside the signed 16-bit TGAM range (`-32768..32767`).
- The BCI does **not** use Attention, Meditation, `POOR_SIGNAL`, blink strength or TGAM summary EEG-power rows as classifier inputs.
- A complete 1024-sample RAW buffer is required before BCI feature acquisition is considered ready.
- The live acquisition gate additionally requires a host-observed RAW arrival rate between **400 and 620 samples/s** and a minimum recent RAW spread of **3 counts**. These are continuity/sanity checks only; they do not alter the model contract, which remains fixed at 512 Hz.

Transport packet boundaries do not define an epoch. Bluetooth Classic, Serial/USB and the simulator all feed the same controller, and epoch duration is defined by RAW sequence length.

## 2. Feature vector

Every 2-second RAW epoch is DC-centered, Hann-windowed and transformed with an FFT. The Version 1 feature schema is `mindflex-raw18-v1` and contains exactly 18 finite values:

1. log delta power (0.5–4 Hz)
2. log theta power (4–8 Hz)
3. log low-alpha power (8–10 Hz)
4. log high-alpha power (10–13 Hz)
5. log low-beta power (13–20 Hz)
6. log high-beta power (20–30 Hz)
7. log low-gamma power (30–40 Hz)
8. log mid-gamma power (40–50 Hz)
9. relative theta power
10. relative alpha power
11. relative beta power
12. relative gamma power
13. log theta/beta ratio
14. log alpha/beta ratio
15. normalized spectral entropy
16. normalized spectral centroid
17. log RMS amplitude
18. log line length

Flat epochs, incomplete epochs, out-of-range RAW values and any non-finite intermediate/final value are rejected.

## 3. Calibration protocol

Training and validation use balanced timed trials:

`Prepare → mental task → overlapping 2 s RAW epochs → rest`

| Mode | Training trials/class | Validation trials/class | Prepare | Task | Epoch | Epoch step | Ideal epochs/trial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quick | 4 | 3 | 1.5 s | 4 s | 2 s | 1 s | 3 |
| Standard | 8 | 5 | 2 s | 5 s | 2 s | 1 s | 4 |
| Research | 15 | 8 | 3 s | 6 s | 2 s | 1 s | 5 |

Trial order is balanced by class and shuffled in blocks. Every calibration run receives unique trial IDs. A delayed UI callback is never allowed to fabricate missed historical epochs or capture an epoch after the mental-task phase has ended.

A rejected epoch reduces the number of epochs stored for that trial; it does not create a replacement observation at the same instant.

## 4. Training readiness

A profile may be trained only when **every expected class** has at least:

- **4 independent training trials**, and
- **8 valid epochs**.

Unexpected classes in the session make the session ineligible for automatic training. For Communication, the expected classes are exactly the current mental-vocabulary labels.

Training sessions are cumulative: a later calibration can add trials to the same profile. The model is rebuilt from the complete valid session whenever training is finalized.

## 5. Trial-balanced centroid model

The model algorithm identifier is `class-balanced-trial-centroid-zscore-v1`.

Overlapping epochs from one cue are correlated, so they must not give a long/clean trial more weight than another independent trial. Version 1 therefore trains in two levels:

1. All valid epoch feature vectors inside a `(class, trial_id)` are averaged into **one trial vector**.
2. A class-balanced global mean/variance is computed: every class contributes equal weight to the normalization statistics even when one class has accumulated more trials.
3. Each trial vector is z-score normalized with those class-balanced statistics.
4. Each class centroid is the mean of its normalized trial vectors.

This means **each independent trial contributes one observation inside its class**, and **each class contributes equal weight to feature scaling**. A command with extra calibration trials can improve its own centroid estimate without silently becoming a larger class prior.

For inference, an epoch feature vector is z-score normalized with the stored training mean/scale and compared with every class centroid using Euclidean distance. The predicted class is the nearest centroid.

The reported `confidence`/score is a softmax of negative centroid distances. It is **relative model evidence**, not a calibrated statistical probability and not a medical confidence measure.

## 6. Independent-trial validation

The validation strategy identifier is `independent-trial-v1`.

Validation follows the same independence rule as training:

1. collect valid epochs during each validation trial;
2. average those epochs into one validation trial vector;
3. classify that trial vector once;
4. calculate global and per-class accuracy from independent trials only.

Overlapping epochs are therefore **not counted as separate validation votes**.

A model is approved only when all conditions are true simultaneously:

- global accuracy **≥ 70%**;
- accuracy for **every class ≥ 60%**;
- **at least 3 independent validation trials per class**;
- validation labels match the expected profile labels exactly;
- validation counts and stored accuracies are internally consistent;
- the validation is bound to the exact model fingerprint;
- the validation result is bound to the exact independent-trial validation-session fingerprint.

Quick validation has 3 trials/class, Standard has 5, and Research has 8. Because trial counts are discrete, the effective global threshold can be stricter than 70% for small class counts. For a two-class Quick validation, for example, 5 of 6 correct trials are required to clear the global threshold.

**Standard is the recommended default.** Research provides more independent validation observations when a stronger empirical check is desired.

## 7. Persistence and integrity

Version 1 persists three independent artifact types: training/validation sessions, models and profile metadata.

The contracts are explicit:

- model schema: `1`
- session schema: `1`
- feature schema: `mindflex-raw18-v1`
- model algorithm: `class-balanced-trial-centroid-zscore-v1`
- session sampling: `timed-trials-v1`
- validation strategy: `independent-trial-v1`

Integrity is enforced at several levels:

- The model stores a SHA-256 **training-session fingerprint** over the exact trial-balanced training inputs. A model whose fingerprint does not match the saved training session is rejected and can be rebuilt from the valid session.
- The model itself has a SHA-256 **model fingerprint**. A validation result is valid only for that exact model.
- The validation result stores a SHA-256 **validation-session fingerprint** over the exact trial-balanced validation inputs. Mutating or replacing the validation session invalidates approval immediately, including before persistence.
- Restored validation metadata is not trusted by itself. The application reloads the saved validation session, reruns validation against the saved model and requires the recomputed result to match the persisted result exactly.
- Model labels must match the current profile labels exactly before live control/communication can be enabled.
- JSON writes use temporary files, flush/fsync and atomic replacement; configured size limits are enforced on both write and read paths.
- Non-finite numbers, malformed dimensions, duplicate `(label, trial_id, epoch_index)` entries and mismatched user ownership are rejected.

## 8. Live-use gate

A trained model is not sufficient for Live Control or Communication. The runtime requires a model that:

1. is structurally valid under the Version 1 contract;
2. matches its exact training session;
3. has the exact expected label set;
4. has an independently recomputed validation result bound to both its model fingerprint and exact validation-session fingerprint; and
5. passes all global/per-class/trial-count thresholds above.

Changing the model invalidates its validation. Adding a new Communication label invalidates the Communication training/model/validation because the class set has changed.

## 9. Scope and limitations

The Version 1 classifier is deliberately simple and inspectable. It is a trial-balanced nearest-centroid model, not a deep-learning model, and its softmax distance score is not probability-calibrated. EEG/BCI performance can vary substantially with electrode contact, motion, electrical noise, mental strategy and session drift.

This software is experimental and is not a medical device. Model validation in this application measures repeatability inside its calibration protocol; it does not establish clinical validity or diagnostic performance.
