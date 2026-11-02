"""Portable, dependency-aware startup for Windows and Linux.

Uses only the standard library so either platform launcher can diagnose and
prepare an isolated environment before optional dependencies are available.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TextIO

from .settings import program_dir


WINDOWS_MIN = (3, 11)
WINDOWS_MAX_EXCLUSIVE = (3, 14)
LOG_NAME = "studio-startup.log"


def linux_package_manager() -> tuple[list[str], str] | None:
    """Return a supported Linux Tk package install command and human-readable name."""
    if not sys.platform.startswith("linux"):
        return None
    candidates = (
        ("apt-get", ["apt-get", "install", "-y", "python3-tk"], "Debian/Ubuntu"),
        ("dnf", ["dnf", "install", "-y", "python3-tkinter"], "Fedora"),
        ("pacman", ["pacman", "-S", "--needed", "--noconfirm", "tk"], "Arch Linux"),
        ("zypper", ["zypper", "--non-interactive", "install", "python3-tk"], "openSUSE"),
    )
    for executable, command, distro in candidates:
        if shutil.which(executable):
            return command, distro
    return None


def tkinter_available() -> bool:
    """Check the *bootstrap interpreter* (the one supplying Tk to the venv)."""
    try:
        import tkinter  # noqa: F401 - this also checks the _tkinter native extension
    except (ImportError, OSError):
        return False
    return True


def show_linux_tk_help(log: TextIO) -> None:
    manager = linux_package_manager()
    say("[ERROR] Tkinter is not installed for this Python interpreter.", log)
    if manager:
        command, distro = manager
        # Installation is never performed without an explicit yes from the user.
        say(f"[HELP] On {distro}, install Tk support with:", log)
        say("       sudo " + " ".join(arg for arg in command if arg not in ("-y", "--noconfirm")), log)
    else:
        say("[HELP] Install the Tk/Tcl Python bindings from your Linux distribution.", log)
    say("[HELP] On Ubuntu/Debian also install python3-venv if venv creation fails.", log)
    say("[HELP] If using pyenv/Conda or a custom Python build, install Tk for that exact interpreter", log)
    say("       or select the distribution Python; a system package cannot fix every custom build.", log)
    say("[HELP] Run ./RUN_STUDIO.sh --diagnose for a detailed report.", log)


def ensure_linux_tk(*, root: Path, log: TextIO) -> bool:
    """Offer an interactive system-package repair; never silently run sudo."""
    if tkinter_available():
        return True
    show_linux_tk_help(log)
    manager = linux_package_manager()
    if manager is None or not sys.stdin.isatty():
        return False
    command, distro = manager
    if not shutil.which("sudo") and os.geteuid() != 0:
        say("[ERROR] sudo is unavailable. Install the package manually as administrator.", log)
        return False
    try:
        answer = input(f"[QUESTION] Install system Tkinter packages for {distro} now? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        say("[INFO] System-package installation cancelled.", log)
        return False
    if answer not in ("y", "yes"):
        say("[INFO] System-package installation skipped by user.", log)
        return False
    elevated = command if os.geteuid() == 0 else ["sudo", *command]
    say("[INFO] Installing system Tkinter package with user permission...", log)
    if execute(elevated, root=root, log=log) != 0:
        say("[ERROR] System package installation failed; inspect the output above.", log)
        return False
    importlib.invalidate_caches()
    if not tkinter_available():
        say("[ERROR] Tkinter remains unavailable to this interpreter after package installation.", log)
        show_linux_tk_help(log)
        return False
    say("[OK] Tkinter is now available.", log)
    return True


# This code runs in a *separate interpreter* so it diagnoses precisely the
# environment which will run the UI, not the bootstrap interpreter.
ENVIRONMENT_CHECK = r'''
import importlib
import importlib.metadata
import sys
import socket
import shutil

failed = []
modules = [
    ('Tkinter', 'tkinter'),
    ('NumPy', 'numpy'),
    ('Matplotlib Tk', 'matplotlib.backends.backend_tkagg'),
    ('Serial transport', 'serial'),
    ('Studio UI', 'mindflex.ui'),
]
if sys.platform == 'win32' and '--core-only' not in sys.argv:
    modules.extend([
        ('WinRT runtime', 'winrt.windows.foundation.collections'),
        ('Bluetooth API', 'winrt.windows.devices.bluetooth'),
        ('RFCOMM API', 'winrt.windows.devices.bluetooth.rfcomm'),
        ('Device discovery', 'winrt.windows.devices.enumeration'),
        ('Network sockets', 'winrt.windows.networking.sockets'),
        ('Stream reader', 'winrt.windows.storage.streams'),
    ])
try:
    version = importlib.metadata.version('mindflex-eeg-studio')
    if version != '1.0.0':
        raise RuntimeError('Expected version 1.0.0, got ' + version)
    print('[OK] Installed project version: ' + version, flush=True)
except Exception as exc:
    print('[FAIL] Project installation: ' + str(exc), flush=True)
    failed.append('project')
for label, module in modules:
    try:
        importlib.import_module(module)
        print('[OK] ' + label, flush=True)
    except Exception as exc:
        print('[FAIL] ' + label + ': ' + type(exc).__name__ + ': ' + str(exc), flush=True)
        failed.append(label)
if sys.platform.startswith('linux'):
    # Bluetooth is optional: report its availability without preventing the
    # Studio's serial and simulator paths from working.
    if hasattr(socket, 'AF_BLUETOOTH') and hasattr(socket, 'BTPROTO_RFCOMM'):
        print('[OK] Linux native RFCOMM socket support', flush=True)
    else:
        print('[INFO] No RFCOMM socket support in this Python; use system Python for Bluetooth', flush=True)
    if shutil.which('bluetoothctl'):
        print('[OK] BlueZ bluetoothctl available', flush=True)
    else:
        print('[INFO] BlueZ bluetoothctl not found; install bluez for native Bluetooth', flush=True)
if '--check-gui' in sys.argv:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        print('[OK] Tk display initialization', flush=True)
    except Exception as exc:
        print('[FAIL] Tk display initialization: ' + str(exc), flush=True)
        failed.append('Tk display')
sys.exit(1 if failed else 0)
'''


def supported_windows_python(version: tuple[int, int]) -> bool:
    """Match the interpreter families supported by pinned PyWinRT 3.x."""
    return WINDOWS_MIN <= version < WINDOWS_MAX_EXCLUSIVE


def environment_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def output_log(root: Path) -> tuple[Path, TextIO]:
    """Keep a readable log even if the checkout folder is read-only."""
    for base in (root / "logs", Path(tempfile.gettempdir()) / "MindFlex-EEG-Studio"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            path = base / LOG_NAME
            return path, path.open("w", encoding="utf-8", errors="replace")
        except OSError:
            continue
    raise OSError("Could not create a startup log in the repository or temporary directory")


def say(message: str, log: TextIO) -> None:
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def execute(command: list[str], *, root: Path, log: TextIO) -> int:
    """Stream all child output to both the console and a persistent log."""
    display = ["<dependency and GUI import checks>" if item == ENVIRONMENT_CHECK else item for item in command]
    say("$ " + " ".join(display), log)
    try:
        with subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
    except OSError as exc:
        say(f"[ERROR] Could not run subprocess: {exc}", log)
        return 1


def interpreter_works(python: Path, *, root: Path, log: TextIO) -> bool:
    if not python.is_file():
        return False
    return execute([
        str(python), "-c", "import sys; assert sys.prefix != sys.base_prefix; "
        "assert (3,11) <= sys.version_info[:2] < (3,14)"
        if os.name == "nt" else
        "import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info >= (3,10)",
    ], root=root, log=log) == 0


def ensure_environment(*, root: Path, log: TextIO) -> Path | None:
    """Create an isolated environment or repair one with an invalid interpreter."""
    venv = environment_python(root)
    if interpreter_works(venv, root=root, log=log):
        say("[OK] Existing .venv Python is usable.", log)
        return venv
    say("[INFO] Creating a compatible local Python environment (.venv)...", log)
    # --clear repairs incomplete environments and obsolete interpreter links.
    if execute([sys.executable, "-m", "venv", "--clear", str(root / ".venv")], root=root, log=log):
        say("[ERROR] Could not create .venv. Verify Python includes pip/venv and the folder is writable.", log)
        if sys.platform.startswith("linux"):
            say("[HELP] On Ubuntu/Debian: sudo apt-get install python3-venv python3-pip", log)
        return None
    if not interpreter_works(venv, root=root, log=log):
        say("[ERROR] The new .venv interpreter failed its self-check.", log)
        return None
    return venv


def check_dependencies(
    python: Path, *, root: Path, log: TextIO, check_gui: bool = False, core_only: bool = False
) -> bool:
    command = [str(python), "-c", ENVIRONMENT_CHECK]
    if check_gui:
        command.append("--check-gui")
    if core_only:
        command.append("--core-only")
    return execute(command, root=root, log=log) == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MindFlex EEG Studio Windows/Linux startup and diagnostics")
    parser.add_argument("--diagnose", action="store_true", help="Check the current Python environment without installing packages")
    parser.add_argument("--no-gui", action="store_true", help="Skip creating a Tk window during diagnostics")
    args = parser.parse_args(argv)
    root = program_dir()
    try:
        log_path, log = output_log(root)
    except OSError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    with log:
        say(f"MindFlex EEG Studio 1.0.0 - startup log: {log_path}", log)
        say(f"Interpreter: {sys.executable} ({sys.version.split()[0]})", log)
        if os.name == "nt" and not supported_windows_python(sys.version_info[:2]):
            say("[ERROR] For Windows, use CPython 3.11, 3.12 or 3.13 (3.12 recommended).", log)
            return 1
        if args.diagnose:
            ok = check_dependencies(Path(sys.executable), root=root, log=log, check_gui=not args.no_gui)
            if sys.platform.startswith("linux") and not tkinter_available():
                show_linux_tk_help(log)
            say("[OK] All startup checks passed." if ok else "[ERROR] Startup checks failed; see details above.", log)
            return 0 if ok else 1

        # Tkinter is supplied by the OS on most Linux distributions, not pip.
        # Detect it before creating a venv or trying to install wheel dependencies.
        if sys.platform.startswith("linux") and not ensure_linux_tk(root=root, log=log):
            return 1

        # With a manually activated venv, do not create an environment inside it.
        if sys.prefix != sys.base_prefix:
            python = Path(sys.executable)
        else:
            python = ensure_environment(root=root, log=log)
            if python is None:
                return 1
        if not check_dependencies(python, root=root, log=log):
            say("[INFO] Installing/repairing required packages (internet access may be needed)...", log)
            install_status = execute([str(python), "-m", "pip", "install", "-e", str(root)], root=root, log=log)
            if install_status or not check_dependencies(python, root=root, log=log):
                # Core studio paths don't require WinRT: allow offline serial and
                # simulation if a previously installed environment only lacks BT.
                if not check_dependencies(python, root=root, log=log, core_only=True):
                    say("[ERROR] Required packages are missing. Check internet access and the errors above.", log)
                    return 1
                say("[WARNING] Bluetooth features are unavailable or incomplete. Serial and Simulator remain usable.", log)

        say("[INFO] Starting the graphical application...", log)
        status = execute([str(python), "-m", "mindflex"], root=root, log=log)
        if status:
            say(f"[ERROR] The application stopped with exit code {status}.", log)
            launcher = "RUN_STUDIO.bat" if os.name == "nt" else "./RUN_STUDIO.sh"
            say(f"[HELP] Run {launcher} --diagnose and inspect this log for details.", log)
            if sys.platform.startswith("linux"):
                say("[HELP] Check DISPLAY / WAYLAND_DISPLAY and use a graphical desktop session.", log)
        else:
            say("[OK] Studio exited normally.", log)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
