from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from .bci import Prediction

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
        if label == SILENCE_LABEL:
            return ""
        return self.phrases.get(label, label)


class StableInterpreter:
    """General temporal voting rule shared by communication/live predictions."""

    def __init__(self, vocabulary: MentalVocabulary, window: int = 7, min_confidence: float = 0.45) -> None:
        self.vocabulary = vocabulary
        if type(window) is not int:
            raise ValueError("Voting window must be an integer")
        if window < 3 or window > 1001:
            raise ValueError("Voting window must be between 3 and 1001 samples")
        self.window = window
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValueError("Minimum confidence must be numeric")
        confidence = float(min_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Minimum confidence must be between 0 and 1")
        self.min_confidence = confidence
        self._labels: deque[str] = deque(maxlen=self.window)
        self._last_emitted = SILENCE_LABEL

    def reset(self) -> None:
        self._labels.clear()
        self._last_emitted = SILENCE_LABEL

    def push(self, prediction: Prediction) -> str | None:
        confidence = float(prediction.confidence)
        label = prediction.label
        if (
            not isinstance(label, str)
            or label not in self.vocabulary.phrases
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or confidence < self.min_confidence
        ):
            label = SILENCE_LABEL
        self._labels.append(label)
        if len(self._labels) < self.window:
            return None
        winner, votes = Counter(self._labels).most_common(1)[0]
        required = self.window // 2 + 1
        if votes < required or winner == SILENCE_LABEL:
            self._last_emitted = SILENCE_LABEL
            return None
        if winner == self._last_emitted:
            return None
        self._last_emitted = winner
        return self.vocabulary.phrase_for(winner)
