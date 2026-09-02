from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .bci import CentroidModel, ModelStore, TrainingSession, ValidationResult
from .mental_text import MentalVocabulary
from .rules import WorkflowRules

LOGGER = logging.getLogger(__name__)

CURSOR_LABELS = ("neutral", "left", "right", "up", "down")
CONCENTRATION_LABELS = ("relaxed", "focused")
PROFILES = ("concentration", "cursor", "communication")


def _new_sessions() -> dict[str, TrainingSession]:
    return {name: TrainingSession(name) for name in PROFILES}


@dataclass(slots=True)
class NeuroRuntime:
    """Persisted, UI-independent state for the complete neural workflow."""

    user_name: str = ""
    sessions: dict[str, TrainingSession] = field(default_factory=_new_sessions)
    models: dict[str, CentroidModel] = field(default_factory=dict)
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
        return any(model.ready for model in self.models.values())

    def validated(self, profile: str, threshold: float) -> bool:
        result = self.validations.get(profile)
        return bool(result and result.accuracy >= threshold)

    def live_ready(self, threshold: float) -> bool:
        return self.validated("concentration", threshold) or self.validated("cursor", threshold)

    def communication_ready(self, threshold: float) -> bool:
        return self.validated("communication", threshold)

    def any_validated(self, threshold: float) -> bool:
        return any(self.validated(profile, threshold) for profile in PROFILES)

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
            self.model_store.delete_metadata(name)

    def persist_profile(self, profile: str) -> None:
        """Persist one profile using canonical automatic names."""
        name = self.automatic_name(profile)
        validation_name = self.automatic_validation_name(profile)
        self.model_store.save_session(name, self.sessions[profile])

        model = self.models.get(profile)
        if model and model.ready:
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
        if result is not None:
            payload["validation"] = {
                "total": result.total,
                "correct": result.correct,
                "accuracy": result.accuracy,
                "per_label": dict(result.per_label),
            }
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
                    self.models[profile] = self.model_store.load(name)
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
            if model is not None and model.ready:
                continue
            labels = self.labels_for(profile)
            session = self.sessions[profile]
            if not rules.training_ready(session, labels):
                continue
            try:
                self.models[profile] = CentroidModel.train(session)
                self.persist_profile(profile)
            except (OSError, ValueError, ArithmeticError) as exc:
                LOGGER.warning("Could not build automatic %s model: %s", profile, exc)
                self.models.pop(profile, None)

    def _restore_metadata(self, profile: str, metadata: dict) -> None:
        if profile == "communication":
            phrases = metadata.get("vocabulary")
            if isinstance(phrases, dict):
                self.vocabulary.phrases.clear()
                self.vocabulary.phrases.update(
                    {
                        str(label): str(phrase)
                        for label, phrase in phrases.items()
                        if str(label).strip() and str(phrase).strip()
                    }
                )

        validation = metadata.get("validation")
        if not isinstance(validation, dict):
            return
        try:
            result = ValidationResult(
                int(validation.get("total", 0)),
                int(validation.get("correct", 0)),
                float(validation.get("accuracy", 0.0)),
                {str(key): float(value) for key, value in dict(validation.get("per_label", {})).items()},
            )
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring malformed %s validation metadata", profile)
            return
        if result.total < 0 or result.correct < 0 or result.correct > result.total or not 0.0 <= result.accuracy <= 1.0:
            LOGGER.warning("Ignoring inconsistent %s validation metadata", profile)
            return
        self.validations[profile] = result

    def _apply_owner(self) -> None:
        owner = " ".join(self.user_name.strip().split())
        for session in (*self.sessions.values(), *self.validation_sessions.values()):
            session.owner = owner
