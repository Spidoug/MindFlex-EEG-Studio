from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .settings import app_data_dir
from .storage import atomic_write_json


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
        clean = " ".join(unicodedata.normalize("NFKC", str(full_name)).strip().split())
        if not clean:
            raise ValueError("Full name cannot be empty")
        if len(clean) > 120:
            raise ValueError("Full name is too long")
        if any(ord(ch) < 32 for ch in clean):
            raise ValueError("Full name contains control characters")
        digest = hashlib.sha256(clean.casefold().encode("utf-8")).hexdigest()[:10]
        slug = _slug(clean)[:64].rstrip("-") or "user"
        user_id = f"{slug}-{digest}"
        return cls(clean, user_id, app_data_dir() / "users" / user_id)

    def ensure_storage(self) -> None:
        path = self.data_dir / "profile.json"
        payload = {"schema": 1, "user_id": self.user_id, "full_name": self.full_name}
        atomic_write_json(path, payload, max_bytes=64 * 1024)

