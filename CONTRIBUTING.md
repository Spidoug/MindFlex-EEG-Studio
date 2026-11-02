# Contributing

Thanks for considering a contribution to MindFlex EEG Studio.

## Workflow

1. Open an issue describing the problem or proposed change. Include the platform, Python version, transport type and steps to reproduce (omit personally identifiable EEG recordings).
2. Create a focused branch and keep changes limited to one concern per pull request.
3. Run the full regression suite and offline benchmark described in [Testing and validation](docs/TESTING.md).
4. Include a regression test for functional changes, especially sample continuity, artifact rejection, calibration or model persistence.
5. Update the relevant permanent documentation instead of appending historical repair notes.

Keep code, comments and documentation in English; translate only UI catalog strings. Preserve the documented sample timing, feature schema, validation rules and persistence format unless a deliberate compatibility change is proposed. Avoid silently relaxing integrity checks.

## Data and privacy

Never commit user sessions, RAW recordings or models from identifiable participants. Use generated synthetic inputs in public tests. Explain whether an algorithmic performance claim comes from synthetic data or independently evaluated human recordings.

The software is experimental and should not be represented as a medical device or a safety-critical controller.
