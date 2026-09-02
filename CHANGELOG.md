# Changelog

## 2026-09-02 — English repository and full audit

- Added current application screenshots to the GitHub README and documentation assets.
- Converted maintained source comments, documentation and technical runtime strings to English while preserving all seven UI language catalogs.
- Fixed recursive neural-profile persistence that could silently prevent automatic session/model saves.
- Extracted `NeuroRuntime` from the neural view into a UI-independent runtime module.
- Made persisted-state restore granular and able to rebuild missing automatic models from sufficient saved training data.
- Hardened settings loading against malformed numeric values.
- Reworked Neural Control into a visual-first responsive layout with larger cue, command and cursor surfaces.
- Added structured, translatable Bluetooth status events for discovery/pairing/connection progress.
- Removed 32 unreachable legacy translation keys.
- Added TGAM1 hardware setup documentation, pad configuration table and original English pad diagram.
- Added the project-owner MindFlex/TGAM photographs under `docs/assets/`.
- Removed the old `tests/` and `tools/` distribution folders after completing regression/layout validation.
- Updated CI so the clean repository validates installation, compilation, locale integrity and core imports without removed tooling.
