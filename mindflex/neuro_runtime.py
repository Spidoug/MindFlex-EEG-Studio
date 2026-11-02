from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .bci import BCIModel, ModelStore, TrainingSession, ValidationResult, validate_model
from .mental_text import MentalVocabulary
from .rules import WorkflowRules

LOGGER = logging.getLogger(__name__)

CURSOR_LABELS = ("neutral", "left", "right", "up", "down")
CONCENTRATION_LABELS = ("relaxed", "focused")
PROFILES = ("concentration", "cursor", "communication")
VALIDATION_METADATA_SCHEMA = 1
VALIDATION_STRATEGY = "unified-trial-decision-v1"


def _new_sessions() -> dict[str, TrainingSession]:
    return {name: TrainingSession(name) for name in PROFILES}


@dataclass(slots=True)
class NeuroRuntime:
    """Persisted, UI-independent state for the complete neural workflow."""

    user_name: str = ""
    sessions: dict[str, TrainingSession] = field(default_factory=_new_sessions)
    models: dict[str, BCIModel] = field(default_factory=dict)
    validations: dict[str, ValidationResult] = field(default_factory=dict)
    validation_sessions: dict[str, TrainingSession] = field(default_factory=_new_sessions)
    vocabulary: MentalVocabulary = field(default_factory=MentalVocabulary)
    model_store: ModelStore = field(default_factory=ModelStore)
    active_profile: str = "concentration"
    selected_vocabulary_label: str = ""
    calibration_mode: str = "standard"

    def __post_init__(self) -> None:
        self._apply_owner()
        self.restore_automatic_state()

    @property
    def model_ready(self) -> bool:
        for profile, model in self.models.items():
            expected = self.labels_for(profile)
            if (
                model.ready
                and len(expected) >= 2
                and set(model.labels) == set(expected)
                and self._model_matches_session(profile, model)
            ):
                return True
        return False

    def readiness_flags(self, rules: WorkflowRules) -> tuple[bool, bool, bool, bool]:
        """Compute runtime readiness once per profile for one UI/backend poll.

        The desktop polls state frequently.  Keeping this as one pass avoids
        recomputing the same session/model fingerprints several times while
        preserving fail-closed integrity checks for every profile.
        Returns: (model_ready, any_validated, live_ready, communication_ready).
        """
        any_model_ready = False
        validated_by_profile: dict[str, bool] = {}
        for profile in PROFILES:
            expected = self.labels_for(profile)
            model = self.models.get(profile)
            model_ok = bool(
                model is not None
                and model.ready
                and len(expected) >= 2
                and set(model.labels) == set(expected)
                and self._model_matches_session(profile, model)
            )
            any_model_ready = any_model_ready or model_ok
            if not model_ok:
                validated_by_profile[profile] = False
                continue
            result = self.validations.get(profile)
            if result is None or not self._validation_matches_session(profile, result):
                validated_by_profile[profile] = False
                continue
            try:
                fingerprint_ok = result.model_fingerprint == model.fingerprint()
            except (ValueError, RuntimeError, TypeError, ArithmeticError):
                fingerprint_ok = False
            validated_by_profile[profile] = bool(
                fingerprint_ok and rules.validation_passed(result, expected)
            )

        any_validated = any(validated_by_profile.values())
        live_ready = validated_by_profile.get("concentration", False) or validated_by_profile.get("cursor", False)
        communication_ready = validated_by_profile.get("communication", False)
        return any_model_ready, any_validated, live_ready, communication_ready

    def validated(self, profile: str, rules: WorkflowRules) -> bool:
        model = self.models.get(profile)
        result = self.validations.get(profile)
        expected = self.labels_for(profile)
        if (
            model is None
            or result is None
            or not model.ready
            or len(expected) < 2
            or set(model.labels) != set(expected)
            or not self._model_matches_session(profile, model)
            or not self._validation_matches_session(profile, result)
        ):
            return False
        try:
            if result.model_fingerprint != model.fingerprint():
                return False
        except (ValueError, RuntimeError, TypeError, ArithmeticError):
            return False
        return rules.validation_passed(result, expected)

    def live_ready(self, rules: WorkflowRules) -> bool:
        return self.validated("concentration", rules) or self.validated("cursor", rules)

    def communication_ready(self, rules: WorkflowRules) -> bool:
        return self.validated("communication", rules)

    def any_validated(self, rules: WorkflowRules) -> bool:
        return any(self.validated(profile, rules) for profile in PROFILES)

    def labels_for(self, profile: str) -> tuple[str, ...]:
        if profile == "concentration":
            return CONCENTRATION_LABELS
        if profile == "cursor":
            return CURSOR_LABELS
        if profile == "communication":
            return tuple(self.vocabulary.phrases)
        return ()

    def automatic_name(self, profile: str) -> str:
        return self.model_store.normalize_name(f"{profile}--automatic")

    def automatic_validation_name(self, profile: str) -> str:
        return self.model_store.normalize_name(f"{profile}--validation")

    def invalidate_training(self, profile: str, *, delete_persisted: bool = False) -> None:
        self.models.pop(profile, None)
        self.validations.pop(profile, None)
        self.validation_sessions[profile].clear()
        if delete_persisted:
            name = self.automatic_name(profile)
            self.model_store.delete_model(name)
            self.model_store.delete_session(self.automatic_validation_name(profile))

    def persist_profile(self, profile: str) -> None:
        """Persist one profile using canonical automatic names."""
        name = self.automatic_name(profile)
        validation_name = self.automatic_validation_name(profile)
        self.model_store.save_session(name, self.sessions[profile])

        model = self.models.get(profile)
        if model and model.ready:
            if not self._model_matches_session(profile, model):
                raise ValueError("Model does not match the current training session")
            self.model_store.save(name, model)
        else:
            self.model_store.delete_model(name)

        validation_session = self.validation_sessions[profile]
        if validation_session.samples:
            self.model_store.save_session(validation_name, validation_session)
        else:
            self.model_store.delete_session(validation_name)

        payload: dict[str, object] = {}
        if profile == "communication":
            payload["vocabulary"] = dict(self.vocabulary.phrases)
        result = self.validations.get(profile)
        if (
            result is not None
            and model is not None
            and model.ready
            and validation_session.samples
            and self._model_matches_session(profile, model)
            and self._validation_matches_session(profile, result)
            and result.model_fingerprint == model.fingerprint()
        ):
            payload["validation"] = {
                "schema": VALIDATION_METADATA_SCHEMA,
                "strategy": VALIDATION_STRATEGY,
                "total": result.total,
                "correct": result.correct,
                "decided": result.decided,
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "decision_rate": result.decision_rate,
                "per_label": dict(result.per_label),
                "trials_per_label": dict(result.trials_per_label),
                "correct_per_label": dict(result.correct_per_label),
                "decided_per_label": dict(result.decided_per_label),
                "model_fingerprint": result.model_fingerprint,
                "validation_fingerprint": result.validation_fingerprint,
            }
        elif result is not None:
            self.validations.pop(profile, None)
        self.model_store.save_metadata(name, payload)

    def restore_automatic_state(self) -> None:
        """Restore each persisted artifact independently.

        A corrupt optional artifact must not hide a valid calibration session.
        Keeping restoration granular also allows a missing/corrupt model to be
        rebuilt automatically from a valid saved session.
        """
        models = set(self.model_store.list_models())
        sessions = set(self.model_store.list_sessions())
        for profile in PROFILES:
            name = self.automatic_name(profile)
            validation_name = self.automatic_validation_name(profile)

            if name in sessions:
                try:
                    self.sessions[profile] = self.model_store.load_session(name)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    LOGGER.warning("Could not restore %s training session: %s", profile, exc)

            if name in models:
                try:
                    restored_model = self.model_store.load(name)
                    if not self._model_matches_session(profile, restored_model):
                        raise ValueError("Model training fingerprint does not match the saved session")
                    self.models[profile] = restored_model
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    LOGGER.warning("Could not restore %s model: %s", profile, exc)
                    self.models.pop(profile, None)

            if validation_name in sessions:
                try:
                    self.validation_sessions[profile] = self.model_store.load_session(validation_name)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    LOGGER.warning("Could not restore %s validation session: %s", profile, exc)

            try:
                metadata = self.model_store.load_metadata(name)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                LOGGER.warning("Could not restore %s metadata: %s", profile, exc)
                metadata = {}
            self._restore_metadata(profile, metadata)

        self._apply_owner()

    def ensure_automatic_models(self, rules: WorkflowRules) -> None:
        """Rebuild missing models whenever saved training data is sufficient."""
        for profile in PROFILES:
            model = self.models.get(profile)
            labels = self.labels_for(profile)
            session = self.sessions[profile]
            if (
                model is not None
                and model.ready
                and set(model.labels) == set(labels)
                and self._model_matches_session(profile, model)
            ):
                continue
            self.models.pop(profile, None)
            if not rules.training_ready(session, labels):
                continue
            try:
                self.models[profile] = BCIModel.train(session)
                self.persist_profile(profile)
            except (OSError, ValueError, ArithmeticError) as exc:
                LOGGER.warning("Could not build automatic %s model: %s", profile, exc)
                self.models.pop(profile, None)

    def _restore_metadata(self, profile: str, metadata: dict) -> None:
        allowed = {"validation"}
        if profile == "communication":
            allowed.add("vocabulary")
        if set(metadata) - allowed:
            LOGGER.warning("Ignoring %s metadata with fields outside the Version 1 format", profile)
            return

        if profile == "communication":
            phrases = metadata.get("vocabulary")
            if phrases is not None:
                if not isinstance(phrases, dict):
                    LOGGER.warning("Ignoring malformed communication vocabulary metadata")
                    return
                restored = MentalVocabulary()
                try:
                    for label, phrase in phrases.items():
                        if (
                            not isinstance(label, str)
                            or not isinstance(phrase, str)
                            or label != label.strip()
                            or phrase != phrase.strip()
                        ):
                            raise ValueError("Vocabulary entries must be canonical strings")
                        restored.set_phrase(label, phrase)
                except ValueError as exc:
                    LOGGER.warning("Ignoring malformed communication vocabulary metadata: %s", exc)
                    return
                self.vocabulary.phrases.clear()
                self.vocabulary.phrases.update(restored.phrases)

        validation = metadata.get("validation")
        if validation is None:
            return
        expected_fields = {
            "schema", "strategy", "total", "correct", "decided", "accuracy",
            "balanced_accuracy", "decision_rate", "per_label", "trials_per_label",
            "correct_per_label", "decided_per_label", "model_fingerprint", "validation_fingerprint",
        }
        if not isinstance(validation, dict) or set(validation) != expected_fields:
            LOGGER.warning("Ignoring malformed %s validation metadata", profile)
            return
        if (
            type(validation.get("schema")) is not int
            or validation.get("schema") != VALIDATION_METADATA_SCHEMA
            or not isinstance(validation.get("strategy"), str)
            or validation.get("strategy") != VALIDATION_STRATEGY
        ):
            LOGGER.warning("Ignoring incompatible %s validation metadata", profile)
            return

        total = validation.get("total")
        correct = validation.get("correct")
        decided = validation.get("decided")
        accuracy = validation.get("accuracy")
        balanced_accuracy = validation.get("balanced_accuracy")
        decision_rate = validation.get("decision_rate")
        per_label = validation.get("per_label")
        trials_per_label = validation.get("trials_per_label")
        correct_per_label = validation.get("correct_per_label")
        decided_per_label = validation.get("decided_per_label")
        model_fingerprint = validation.get("model_fingerprint")
        validation_fingerprint = validation.get("validation_fingerprint")
        if (
            type(total) is not int
            or type(correct) is not int
            or type(decided) is not int
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (accuracy, balanced_accuracy, decision_rate)
            )
            or not isinstance(per_label, dict)
            or not isinstance(trials_per_label, dict)
            or not isinstance(correct_per_label, dict)
            or not isinstance(decided_per_label, dict)
            or not isinstance(model_fingerprint, str)
            or not isinstance(validation_fingerprint, str)
        ):
            LOGGER.warning("Ignoring malformed %s validation metadata", profile)
            return
        if (
            any(not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) for key, value in per_label.items())
            or any(not isinstance(key, str) or type(value) is not int for key, value in trials_per_label.items())
            or any(not isinstance(key, str) or type(value) is not int for key, value in correct_per_label.items())
            or any(not isinstance(key, str) or type(value) is not int for key, value in decided_per_label.items())
        ):
            LOGGER.warning("Ignoring malformed %s validation metadata", profile)
            return

        result = ValidationResult(
            total=total,
            correct=correct,
            decided=decided,
            accuracy=float(accuracy),
            balanced_accuracy=float(balanced_accuracy),
            decision_rate=float(decision_rate),
            per_label={key: float(value) for key, value in per_label.items()},
            trials_per_label=dict(trials_per_label),
            correct_per_label=dict(correct_per_label),
            decided_per_label=dict(decided_per_label),
            model_fingerprint=model_fingerprint,
            validation_fingerprint=validation_fingerprint,
        )
        model = self.models.get(profile)
        validation_session = self.validation_sessions[profile]
        if model is None or not model.ready or not validation_session.samples:
            LOGGER.warning("Ignoring %s validation without its exact model and validation session", profile)
            return
        try:
            recomputed = validate_model(model, validation_session)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            LOGGER.warning("Ignoring %s validation that cannot be recomputed: %s", profile, exc)
            return
        if result != recomputed:
            LOGGER.warning("Ignoring inconsistent %s validation metadata", profile)
            return
        self.validations[profile] = result

    def _model_matches_session(self, profile: str, model: BCIModel) -> bool:
        session = self.sessions.get(profile)
        if session is None or not session.samples:
            return False
        try:
            return model.training_fingerprint == session.training_fingerprint()
        except (ValueError, TypeError, ArithmeticError):
            return False

    def _validation_matches_session(self, profile: str, result: ValidationResult) -> bool:
        session = self.validation_sessions.get(profile)
        if session is None or not session.samples:
            return False
        try:
            return result.validation_fingerprint == session.training_fingerprint()
        except (ValueError, TypeError, ArithmeticError):
            return False

    def _apply_owner(self) -> None:
        owner = " ".join(self.user_name.strip().split())
        for session in (*self.sessions.values(), *self.validation_sessions.values()):
            session.owner = owner
