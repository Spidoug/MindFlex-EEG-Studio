from __future__ import annotations

from dataclasses import dataclass, field

from .bci import Prediction, PredictionStabilizer

SILENCE_LABEL = "__silence__"


@dataclass(slots=True)
class MentalVocabulary:
    phrases: dict[str, str] = field(default_factory=dict)

    def set_phrase(self, label: str, phrase: str) -> None:
        if not isinstance(label, str) or not isinstance(phrase, str):
            raise ValueError("Label and phrase must be strings")
        label = label.strip()
        phrase = phrase.strip()
        if not label or not phrase:
            raise ValueError("Label and phrase are required")
        if label == SILENCE_LABEL:
            raise ValueError("This label is reserved by the communication interpreter")
        if len(label) > 80 or len(phrase) > 500:
            raise ValueError("Label or phrase is too long")
        if any(ord(ch) < 32 for ch in label) or "\x00" in phrase:
            raise ValueError("Label or phrase contains invalid control characters")
        self.phrases[label] = phrase

    def remove(self, label: str) -> None:
        self.phrases.pop(label, None)

    def phrase_for(self, label: str) -> str:
        return self.phrases.get(label, label)


class StableInterpreter:
    """Communication adapter over the same global BCI stabilizer used elsewhere."""

    def __init__(self, vocabulary: MentalVocabulary) -> None:
        self.vocabulary = vocabulary
        if len(self.vocabulary.phrases) >= 2:
            self._stabilizer: PredictionStabilizer | None = PredictionStabilizer(self.vocabulary.phrases)
        else:
            self._stabilizer = None
        self._last_emitted = ""

    def reset(self) -> None:
        if len(self.vocabulary.phrases) >= 2:
            self._stabilizer = PredictionStabilizer(self.vocabulary.phrases)
        else:
            self._stabilizer = None
        self._last_emitted = ""

    def push(self, prediction: Prediction) -> str | None:
        if self._stabilizer is None or set(self._stabilizer.labels) != set(self.vocabulary.phrases):
            self.reset()
        if self._stabilizer is None:
            return None
        stable = self._stabilizer.push(prediction)
        if stable is None or stable.label == self._last_emitted:
            return None
        self._last_emitted = stable.label
        return self.vocabulary.phrase_for(stable.label)
