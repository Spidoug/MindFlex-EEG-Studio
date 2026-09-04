from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json_object(path: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Read a UTF-8 JSON object with optional file-size protection."""
    if max_bytes is not None:
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"JSON file is too large: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return payload


def atomic_write_json(
    path: Path, payload: dict[str, Any], *, max_bytes: int | None = None
) -> None:
    """Durably replace one JSON file without exposing a partial or oversized file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ValueError(f"JSON payload is too large for {path.name}")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        # On POSIX, fsync the directory entry after replace so a completed
        # write is not lost solely because the directory metadata had not yet
        # reached stable storage. Windows does not expose O_DIRECTORY here;
        # the atomic replace still provides the intended no-partial-file rule.
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is not None:
            directory_fd: int | None = None
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY | directory_flag)
                os.fsync(directory_fd)
            except OSError:
                # Some filesystems do not support directory fsync. The file
                # itself was already flushed/fsynced before atomic replace.
                pass
            finally:
                if directory_fd is not None:
                    try:
                        os.close(directory_fd)
                    except OSError:
                        pass
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
