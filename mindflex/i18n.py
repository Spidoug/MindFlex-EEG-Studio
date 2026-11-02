from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Callable

LOCALE_DIR = Path(__file__).with_name("locales")
LANGUAGES = {
    "en": "English",
    "pt_BR": "Português (Brasil)",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "ja": "日本語",
}


class Translator:
    """Validated runtime translator with English as the canonical catalog."""

    def __init__(self, language: str = "en") -> None:
        self._catalogs = {code: self._load(code) for code in LANGUAGES}
        self._language = language if language in LANGUAGES else "en"
        self._listeners: list[Callable[[], None]] = []
        self._validate_catalogs()

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        if language not in LANGUAGES:
            language = "en"
        if language == self._language:
            return
        self._language = language
        for callback in tuple(self._listeners):
            callback()

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def has_key(self, key: str) -> bool:
        return key in self._catalogs["en"]

    def t(self, key: str, **values) -> str:
        catalog = self._catalogs[self._language]
        text = catalog.get(key, self._catalogs["en"].get(key, key))
        try:
            return text.format(**values)
        except (KeyError, ValueError):
            return text

    @staticmethod
    def _load(code: str) -> dict[str, str]:
        path = LOCALE_DIR / f"{code}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Locale catalog must be an object: {path.name}")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()):
            raise RuntimeError(f"Locale catalog must contain only string keys and values: {path.name}")
        return payload

    @staticmethod
    def _fields(text: str) -> set[str]:
        formatter = string.Formatter()
        return {field for _, field, _, _ in formatter.parse(text) if field}

    def _validate_catalogs(self) -> None:
        reference_catalog = self._catalogs["en"]
        reference_keys = set(reference_catalog)
        reference_fields = {key: self._fields(value) for key, value in reference_catalog.items()}
        problems: list[str] = []
        for code, catalog in self._catalogs.items():
            missing = sorted(reference_keys - set(catalog))
            extra = sorted(set(catalog) - reference_keys)
            if missing:
                problems.append(f"{code}: missing {missing}")
            if extra:
                problems.append(f"{code}: extra {extra}")
            for key in reference_keys & set(catalog):
                if not catalog[key].strip():
                    problems.append(f"{code}: empty value for {key}")
                    continue
                fields = self._fields(catalog[key])
                if fields != reference_fields[key]:
                    problems.append(
                        f"{code}: format fields for {key} are {sorted(fields)}, "
                        f"expected {sorted(reference_fields[key])}"
                    )
        if problems:
            raise RuntimeError("Invalid translation catalogs: " + "; ".join(problems))
