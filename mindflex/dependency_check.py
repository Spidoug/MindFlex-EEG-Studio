"""Startup dependency probe used by the platform launchers.

Keep this module stdlib-only until ``main`` imports the optional runtime
packages so it can explain missing dependencies through the launcher's log.
"""

from __future__ import annotations

import importlib
import sys


BASE_MODULES = ("tkinter", "numpy", "matplotlib", "serial")
WINDOWS_BLUETOOTH_MODULES = (
    "winrt.windows.foundation.collections",
    "winrt.windows.devices.bluetooth",
    "winrt.windows.devices.bluetooth.rfcomm",
    "winrt.windows.devices.enumeration",
    "winrt.windows.networking.sockets",
    "winrt.windows.storage.streams",
)


def required_modules() -> tuple[str, ...]:
    modules = list(BASE_MODULES)
    if sys.platform == "win32":
        modules.extend(WINDOWS_BLUETOOTH_MODULES)
    return tuple(modules)


def main() -> None:
    failures: list[str] = []
    for module in required_modules():
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("Missing/broken runtime dependencies:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
