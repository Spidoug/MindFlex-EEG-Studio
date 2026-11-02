from __future__ import annotations

import hashlib
import math
import re
import shutil
import unicodedata
import time
from dataclasses import dataclass
from pathlib import Path

from .settings import app_data_dir
from .storage import atomic_write_json, read_json_object


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
        created_at = time.time()
        if path.exists():
            try:
                prior = read_json_object(path, max_bytes=64 * 1024)
                if prior.get("user_id") == self.user_id:
                    created_at = float(prior.get("created_at", created_at))
            except (OSError, ValueError, TypeError, OverflowError):
                pass
        payload = {
            "schema": 1,
            "user_id": self.user_id,
            "full_name": self.full_name,
            "created_at": created_at,
            "last_used_at": time.time(),
        }
        atomic_write_json(path, payload, max_bytes=64 * 1024)

    def delete_storage(self) -> None:
        """Permanently remove only this validated person's isolated namespace."""
        expected = app_data_dir() / "users" / self.user_id
        if self.data_dir != expected:
            raise ValueError("Profile storage path does not match its identity")
        users_dir = expected.parent.resolve()
        target = expected.resolve(strict=False)
        if target.parent != users_dir or target == users_dir:
            raise ValueError("Refusing to delete an unsafe profile path")
        if expected.is_symlink():
            expected.unlink()
        elif expected.exists():
            shutil.rmtree(expected)

    @classmethod
    def list_saved(cls) -> list["UserProfile"]:
        """Return only valid profiles whose directory matches their identity."""
        users_dir = app_data_dir() / "users"
        if not users_dir.is_dir():
            return []
        profiles: list[tuple[float, UserProfile]] = []
        for profile_path in users_dir.glob("*/profile.json"):
            try:
                payload = read_json_object(profile_path, max_bytes=64 * 1024)
                if payload.get("schema") != 1 or not isinstance(payload.get("full_name"), str):
                    continue
                profile = cls.from_full_name(payload["full_name"])
                if payload.get("user_id") != profile.user_id or profile_path.parent != profile.data_dir:
                    continue
                last_used = float(payload.get("last_used_at", 0.0))
                if not math.isfinite(last_used):
                    last_used = 0.0
                profiles.append((last_used, profile))
            except (OSError, ValueError, TypeError, OverflowError):
                continue
        return [profile for _stamp, profile in sorted(profiles, key=lambda item: (-item[0], item[1].full_name.casefold()))]
