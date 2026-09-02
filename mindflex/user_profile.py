from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .settings import app_data_dir


def _slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return clean or "user"


@dataclass(frozen=True, slots=True)
class UserProfile:
    """One explicit user identity and one isolated data namespace."""

    full_name: str
    user_id: str
    data_dir: Path

    @classmethod
    def from_full_name(cls, full_name: str) -> "UserProfile":
        clean = " ".join(str(full_name).strip().split())
        if not clean:
            raise ValueError("Full name cannot be empty")
        digest = hashlib.sha256(clean.casefold().encode("utf-8")).hexdigest()[:10]
        user_id = f"{_slug(clean)}-{digest}"
        return cls(clean, user_id, app_data_dir() / "users" / user_id)

    def ensure_storage(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / "profile.json"
        payload = {"schema": 1, "user_id": self.user_id, "full_name": self.full_name}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

