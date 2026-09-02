from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .bci import TrainingSession
    from .controller import EEGSnapshot


class WorkflowStep(IntEnum):
    CONNECTION = 0
    MONITOR = 1
    NEURO_CONTROL = 2
    DIAGNOSTICS = 3


class NeuroStep(IntEnum):
    SETUP_TRAINING = 0
    VALIDATION = 1
    LIVE_CONTROL = 2
    COMMUNICATION = 3
    LABORATORY = 4


@dataclass(frozen=True, slots=True)
class SignalPolicy:
    metric_stale_after_seconds: float = 4.0
    minimum_raw_samples: int = 256
    minimum_raw_rate_hz: float = 400.0
    maximum_raw_rate_hz: float = 620.0
    minimum_raw_spread: float = 3.0
    minimum_training_trials_per_class: int = 4
    minimum_training_epochs_per_class: int = 8
    minimum_validation_accuracy: float = 0.60
    minimum_validation_trials_per_class: int = 2


@dataclass(frozen=True, slots=True)
class RuntimeState:
    connected: bool = False
    receiving: bool = False
    model_ready: bool = False
    model_validated: bool = False
    live_ready: bool = False
    communication_ready: bool = False


class WorkflowRules:
    """Single navigation and RAW-readiness rule set."""

    def __init__(self, policy: SignalPolicy | None = None) -> None:
        self.policy = policy or SignalPolicy()

    def workflow_allowed(self, step: WorkflowStep, state: RuntimeState) -> bool:
        requirements: Mapping[WorkflowStep, bool] = {
            WorkflowStep.CONNECTION: True,
            WorkflowStep.MONITOR: state.connected,
            WorkflowStep.NEURO_CONTROL: state.connected and state.receiving,
            WorkflowStep.DIAGNOSTICS: state.connected,
        }
        return requirements[step]

    def neuro_allowed(self, step: NeuroStep, state: RuntimeState) -> bool:
        # Navigation is informational and must not hide lifecycle stages. Once an
        # EEG stream is active, users may inspect Setup, Validation, Live Control
        # and Communication in any order. Individual actions enforce their own
        # concrete prerequisites (trained model, validation, etc.).
        streaming = state.connected and state.receiving
        requirements: Mapping[NeuroStep, bool] = {
            NeuroStep.SETUP_TRAINING: streaming,
            NeuroStep.VALIDATION: streaming,
            NeuroStep.LIVE_CONTROL: streaming,
            NeuroStep.COMMUNICATION: streaming,
            NeuroStep.LABORATORY: state.connected,
        }
        return requirements[step]

    def training_ready(self, session: "TrainingSession", labels) -> bool:
        expected = tuple(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
        if len(expected) < 2:
            return False
        return all(
            session.trial_count(label) >= self.policy.minimum_training_trials_per_class
            and session.count(label) >= self.policy.minimum_training_epochs_per_class
            for label in expected
        )

    def validation_ready(self, session: "TrainingSession", labels) -> bool:
        expected = tuple(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
        if len(expected) < 2:
            return False
        return all(
            session.trial_count(label) >= self.policy.minimum_validation_trials_per_class
            for label in expected
        )

    def feature_snapshot_usable(self, snapshot: "EEGSnapshot") -> bool:
        """One readiness rule for calibration, validation and live control.

        Calibration and live control depend only on a current RAW stream.
        Attention, Meditation, POOR_SIGNAL and vendor EEG-power summaries are
        monitoring outputs and can never block BCI acquisition.
        """
        return (
            snapshot.receiving
            and snapshot.raw_age <= 1.5
            and snapshot.raw_samples >= self.policy.minimum_raw_samples
        )
