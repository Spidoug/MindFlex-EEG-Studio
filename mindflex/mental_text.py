from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .bci import Prediction

SILENCE_LABEL = "__silence__"


@dataclass(slots=True)
class MentalVocabulary:
    phrases: dict[str, str] = field(default_factory=dict)

    def set_phrase(self, label: str, phrase: str) -> None:
        label = label.strip()
        phrase = phrase.strip()
        if not label or not phrase:
            raise ValueError("Label and phrase are required")
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
        self.window = max(3, int(window))
        self.min_confidence = float(min_confidence)
        self._labels: deque[str] = deque(maxlen=self.window)
        self._last_emitted = SILENCE_LABEL

    def reset(self) -> None:
        self._labels.clear()
        self._last_emitted = SILENCE_LABEL

    def push(self, prediction: Prediction) -> str | None:
        label = prediction.label if prediction.confidence >= self.min_confidence else SILENCE_LABEL
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
