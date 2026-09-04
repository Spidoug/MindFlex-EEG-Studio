from __future__ import annotations

from dataclasses import dataclass
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Mapping

from .settings import BCI_EPOCH_SAMPLES

if TYPE_CHECKING:
    from .bci import TrainingSession, ValidationResult
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
    minimum_raw_samples: int = BCI_EPOCH_SAMPLES
    minimum_raw_rate_hz: float = 400.0
    maximum_raw_rate_hz: float = 620.0
    minimum_raw_spread: float = 3.0
    maximum_poor_signal: int = 50
    minimum_training_trials_per_class: int = 4
    minimum_training_epochs_per_class: int = 8
    minimum_validation_accuracy: float = 0.70
    minimum_validation_accuracy_per_class: float = 0.60
    minimum_validation_decision_rate: float = 0.80
    minimum_validation_trials_per_class: int = 3
    warning_checksum_error_ratio: float = 0.02
    maximum_checksum_error_ratio: float = 0.10
    maximum_raw_age_seconds: float = 2.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.minimum_raw_samples,
            self.minimum_training_trials_per_class,
            self.minimum_training_epochs_per_class,
            self.minimum_validation_trials_per_class,
            self.maximum_poor_signal,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ValueError("Signal-policy count thresholds must be positive integers")
        numeric_fields = (
            self.metric_stale_after_seconds,
            self.minimum_raw_rate_hz,
            self.maximum_raw_rate_hz,
            self.minimum_raw_spread,
            self.minimum_validation_accuracy,
            self.minimum_validation_accuracy_per_class,
            self.minimum_validation_decision_rate,
            self.warning_checksum_error_ratio,
            self.maximum_checksum_error_ratio,
            self.maximum_raw_age_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_fields
        ):
            raise ValueError("Signal-policy numeric thresholds must be finite numbers")
        if self.metric_stale_after_seconds <= 0:
            raise ValueError("Metric freshness threshold must be positive")
        if self.minimum_raw_rate_hz <= 0 or self.maximum_raw_rate_hz <= self.minimum_raw_rate_hz:
            raise ValueError("RAW-rate thresholds are invalid")
        if self.minimum_raw_spread < 0:
            raise ValueError("Minimum RAW spread cannot be negative")
        if not 0.0 <= self.minimum_validation_accuracy <= 1.0:
            raise ValueError("Global validation accuracy threshold must be between 0 and 1")
        if not 0.0 <= self.minimum_validation_accuracy_per_class <= 1.0:
            raise ValueError("Per-class validation accuracy threshold must be between 0 and 1")
        if not 0.0 <= self.minimum_validation_decision_rate <= 1.0:
            raise ValueError("Validation decision-rate threshold must be between 0 and 1")
        if not 0.0 <= self.warning_checksum_error_ratio <= self.maximum_checksum_error_ratio <= 1.0:
            raise ValueError("Checksum thresholds are invalid")
        if self.maximum_raw_age_seconds <= 0:
            raise ValueError("RAW freshness threshold must be positive")

    def checksum_error_ratio(self, packets: int, bad_checksums: int) -> float:
        total = max(0, int(packets)) + max(0, int(bad_checksums))
        return (max(0, int(bad_checksums)) / total) if total else 1.0

    def raw_capture_usable(
        self,
        *,
        receiving: bool,
        poor_signal: int | None,
        raw_age: float,
        raw_buffered_samples: int,
        raw_rate_hz: float,
        raw_spread: float,
        packets: int,
        bad_checksums: int,
    ) -> bool:
        """Single acquisition-quality gate used by all neural workflows and diagnostics."""
        signal_ok = poor_signal is None or int(poor_signal) <= self.maximum_poor_signal
        checksum_ok = self.checksum_error_ratio(packets, bad_checksums) <= self.maximum_checksum_error_ratio
        return bool(
            receiving
            and signal_ok
            and checksum_ok
            and float(raw_age) <= self.maximum_raw_age_seconds
            and int(raw_buffered_samples) >= self.minimum_raw_samples
            and self.minimum_raw_rate_hz <= float(raw_rate_hz) <= self.maximum_raw_rate_hz
            and float(raw_spread) >= self.minimum_raw_spread
        )


@dataclass(frozen=True, slots=True)
class RuntimeState:
    connected: bool = False
    receiving: bool = False
    model_ready: bool = False
    model_validated: bool = False
    live_ready: bool = False
    communication_ready: bool = False


class WorkflowRules:
    """Single navigation and EEG-readiness rule set."""

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

    @staticmethod
    def _expected_labels(labels) -> tuple[str, ...]:
        normalized: list[str] = []
        for label in labels:
            if not isinstance(label, str):
                return ()
            clean = label.strip()
            if (
                not clean
                or clean != label
                or len(clean) > 80
                or any(ord(ch) < 32 for ch in clean)
            ):
                return ()
            if clean not in normalized:
                normalized.append(clean)
        return tuple(normalized)

    def training_ready(self, session: "TrainingSession", labels) -> bool:
        expected = self._expected_labels(labels)
        if len(expected) < 2:
            return False
        try:
            # Validate the mutable in-memory session before deriving readiness.
            # This prevents a caller from mutating a dataclass field after
            # collection and bypassing the Version 1 persistence format.
            session.trial_vectors()
            if set(session.labels) - set(expected):
                return False
            return all(
                session.trial_count(label) >= self.policy.minimum_training_trials_per_class
                and session.count(label) >= self.policy.minimum_training_epochs_per_class
                for label in expected
            )
        except (ValueError, TypeError, ArithmeticError):
            return False

    def validation_ready(self, session: "TrainingSession", labels) -> bool:
        expected = self._expected_labels(labels)
        if len(expected) < 2:
            return False
        try:
            session.trial_vectors()
            if set(session.labels) - set(expected):
                return False
            return all(
                session.trial_count(label) >= self.policy.minimum_validation_trials_per_class
                for label in expected
            )
        except (ValueError, TypeError, ArithmeticError):
            return False

    def validation_passed(self, result: "ValidationResult", labels) -> bool:
        expected = self._expected_labels(labels)
        if len(expected) < 2:
            return False
        expected_set = set(expected)
        if (
            type(result.total) is not int
            or type(result.correct) is not int
            or type(result.decided) is not int
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (result.accuracy, result.balanced_accuracy, result.decision_rate)
            )
        ):
            return False
        if (
            set(result.per_label) != expected_set
            or set(result.trials_per_label) != expected_set
            or set(result.correct_per_label) != expected_set
            or set(result.decided_per_label) != expected_set
        ):
            return False
        if result.total != sum(result.trials_per_label.values()):
            return False
        if result.correct != sum(result.correct_per_label.values()):
            return False
        if result.decided != sum(result.decided_per_label.values()):
            return False
        if result.total <= 0 or result.correct < 0 or result.correct > result.decided or result.decided > result.total:
            return False
        if not all(0.0 <= value <= 1.0 for value in (result.accuracy, result.balanced_accuracy, result.decision_rate)):
            return False
        if abs(result.accuracy - (result.correct / result.total)) > 1e-9:
            return False
        if abs(result.decision_rate - (result.decided / result.total)) > 1e-9:
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in result.per_label.values()
        ):
            return False
        if any(type(value) is not int or value <= 0 for value in result.trials_per_label.values()):
            return False
        if any(type(value) is not int or value < 0 for value in result.correct_per_label.values()):
            return False
        if any(type(value) is not int or value < 0 for value in result.decided_per_label.values()):
            return False
        if any(result.correct_per_label[label] > result.decided_per_label[label] for label in expected):
            return False
        if any(result.decided_per_label[label] > result.trials_per_label[label] for label in expected):
            return False
        if any(
            abs(result.per_label[label] - (result.correct_per_label[label] / result.trials_per_label[label])) > 1e-9
            for label in expected
        ):
            return False
        expected_balanced = sum(result.per_label[label] for label in expected) / len(expected)
        if abs(result.balanced_accuracy - expected_balanced) > 1e-9:
            return False
        if (
            not isinstance(result.model_fingerprint, str)
            or len(result.model_fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in result.model_fingerprint)
        ):
            return False
        if (
            not isinstance(result.validation_fingerprint, str)
            or len(result.validation_fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in result.validation_fingerprint)
        ):
            return False
        if result.accuracy < self.policy.minimum_validation_accuracy:
            return False
        if result.balanced_accuracy < self.policy.minimum_validation_accuracy:
            return False
        if result.decision_rate < self.policy.minimum_validation_decision_rate:
            return False
        return all(
            result.trials_per_label[label] >= self.policy.minimum_validation_trials_per_class
            and result.per_label[label] >= self.policy.minimum_validation_accuracy_per_class
            for label in expected
        )

    def feature_snapshot_usable(self, snapshot: "EEGSnapshot") -> bool:
        """Apply the single acquisition-quality rule to every neural operation."""
        return self.policy.raw_capture_usable(
            receiving=snapshot.receiving,
            poor_signal=snapshot.poor_signal,
            raw_age=snapshot.raw_age,
            raw_buffered_samples=snapshot.raw_buffered_samples,
            raw_rate_hz=snapshot.raw_rate_hz,
            raw_spread=snapshot.raw_spread,
            packets=snapshot.packets,
            bad_checksums=snapshot.bad_checksums,
        )
