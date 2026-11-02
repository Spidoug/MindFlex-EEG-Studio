# Installation, startup and diagnostics

Both launchers use `mindflex.launcher` to check the environment, create or reuse `.venv`, report errors and write `logs/studio-startup.log`. Install the system prerequisites for your OS, then run the platform launcher from the extracted repository directory. Never run launch scripts from inside a ZIP viewer.

# Windows

The Windows launcher is `RUN_STUDIO.bat`. Keep it in the repository root beside the `mindflex/` directory and `pyproject.toml`. **Extract the ZIP completely** before double-clicking the script; do not execute it inside a ZIP viewer.

## Supported setup

- Windows 10/11 with standard 64-bit **CPython 3.12** recommended; CPython 3.11/3.13 are also selected by the launcher.
- Install Python from [python.org](https://www.python.org/downloads/) with Tk/Tcl support. The Windows launcher deliberately does not choose a newer Python (3.14+) because the project pins the Windows Bluetooth PyWinRT 3.x dependencies. See [PyWinRT installation](https://pywinrt.readthedocs.io/en/latest/installing/).
- Internet access for the initial installation of NumPy, Matplotlib, pyserial and PyWinRT; a writable project directory for `.venv/` and `sessions/`.
- A graphical Windows session for the Tk UI. **The Simulator does not require an EEG headset.** Bluetooth hardware needs separate pairing, adapter and permissions.

## One-click start

1. Extract all files to a writable directory such as `Documents\MindFlex-EEG-Studio`. Avoid `C:\Program Files` or other protected directories.
2. Double-click `RUN_STUDIO.bat`.
3. On first run, the launcher creates `.venv` and downloads Python dependencies. Subsequent starts reuse the environment.
4. Create or select a local profile, then use **Simulator** on the Connection screen to verify that the application renders correctly.

The command prompt stays open **only if initialization or execution returns an error**. A normal exit closes the window when started by double-click.

## Diagnostics

Open Command Prompt in the extracted folder and run:

```bat
RUN_STUDIO.bat --diagnose
```

The diagnostic mode does not install packages; it checks the active Python environment, installed packages, Bluetooth imports and Tk window creation. If package setup did not finish, first run `RUN_STUDIO.bat` normally so it can repair missing packages.

For manual setup, use the supported interpreter explicitly:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m mindflex
```

If Python is missing or only an unsupported version is detected, install CPython 3.12 and run the launcher again. To rebuild an incomplete Python environment, **close the Studio**, delete only the `.venv` folder, and start `RUN_STUDIO.bat` again. The script will create it afresh. The `sessions/` folder stores profiles and recordings; do not delete it when troubleshooting.

## Error log and recurring problems

`logs/studio-startup.log` contains the complete Python exception, pip output and environment checks. If the project folder cannot be written, the launcher instead reports a log path under the user's temporary directory. The `logs/` and `.venv/` directories are Git-ignored.

| Symptom | What to check |
| --- | --- |
| `No compatible Python interpreter` | Install CPython 3.12 with Tk/Tcl; verify `py -3.12 --version` or `python --version` |
| `No module named ...` | Let the first-run installer complete; check pip errors in the startup log |
| `Could not create .venv` | Folder permissions, free disk space, a complete Python installation, antivirus restrictions |
| Tk/Tcl error or window never appears | Use `--diagnose`; ensure Windows has a graphical desktop session |
| Bluetooth unavailable | Run diagnostics; check PyWinRT packages, device pairing, and Bluetooth Classic support. Simulator and Serial may still work |
| Profile/session write error | Move the project from a protected/read-only location to a writable folder |
| Script closes immediately | Use `RUN_STUDIO.bat`; it now pauses on nonzero exit, and logs the traceback |

The launcher does not modify global Python packages. It installs inside `.venv` unless you deliberately invoke it from an activated environment. Never upload recordings, profiles or startup logs containing private information to a public GitHub issue without reviewing them.

---

# Linux

Use `RUN_STUDIO.sh` on a Linux desktop. It shares the Windows launcher's Python environment creation, dependency checks, diagnostic output and startup logs. **Extract the project completely** before starting. The Studio requires a working graphical session (X11 or Wayland/XWayland).

## Ubuntu / Debian quick start

From the extracted project directory:

```bash
chmod +x RUN_STUDIO.sh
./RUN_STUDIO.sh
```

The launcher checks Python 3.10+, tests for Tkinter, creates or reuses an isolated `.venv`, installs Python packages with pip as needed and launches the graphical application. Internet access is needed for the first pip installation. On a normal successful exit, the shell returns to the terminal. On an error, it prints the cause and saves the full output to `logs/studio-startup.log`.

**Tkinter on Linux is usually a system package, not a pip package.** On Ubuntu/Debian, if Tkinter is absent, the launcher explains the missing dependency and, in an interactive terminal, offers to install `python3-tk` with your explicit permission (`sudo` may request a password). It will not silently elevate privileges. For manual installation:

```bash
sudo apt-get update
sudo apt-get install python3-tk python3-venv python3-pip
```

If you use a third-party Python (pyenv, Conda or a custom build), the distribution's `python3-tk` package may not provide `_tkinter` for that interpreter. Use the distribution Python or install/build Tk support for that exact interpreter. A virtual environment inherits Tk from its underlying Python installation.

Other distributions: Fedora provides `python3-tkinter`; Arch uses `tk`; openSUSE commonly uses `python3-tk`. Distribution package naming may vary.

## Diagnosis and logs

For full environment, graphics and import checks without installing anything:

```bash
./RUN_STUDIO.sh --diagnose
```

To check imports without opening a Tk test window (useful over SSH or for CI):

```bash
./RUN_STUDIO.sh --diagnose --no-gui
```

Logs are written to `logs/studio-startup.log`, or a temporary directory fallback if the project directory is read-only. The launcher preserves errors rather than immediately exiting with an unhelpful traceback.

## Common problems

| Problem | Resolution |
| --- | --- |
| `ModuleNotFoundError: No module named 'tkinter'` | Install the OS Tk package matching the Python executable; on Ubuntu/Debian: `sudo apt-get install python3-tk` |
| `.venv` creation fails or `ensurepip` missing | Install `python3-venv` for the system Python; for a custom Python install its matching venv module |
| Missing NumPy, Matplotlib or pyserial | Run `./RUN_STUDIO.sh` with internet access and inspect the pip output in the startup log |
| `no display name and no $DISPLAY environment variable` | Run from a graphical desktop or connect with working X11 forwarding; an SSH-only session cannot display the interface directly |
| Serial port access denied | Check device permissions. On Ubuntu, a serial device may require membership in `dialout`; log out and in after any group changes |
| No Bluetooth discovery | Install `bluez`, turn on the adapter (`bluetoothctl show`), use the Bluetooth tab and see [Linux Bluetooth](LINUX_BLUETOOTH.md) |

The **Simulator** operates without an EEG headset, but still requires a graphical desktop. Headless benchmarks and tests do not need Tk windows unless the optional GUI smoke tests are enabled. Logs, local profiles and session recordings should be reviewed before sharing because they can include personal or hardware information.

## Native Bluetooth Classic

Linux uses BlueZ and native RFCOMM sockets instead of WinRT. See [Linux Bluetooth setup](LINUX_BLUETOOTH.md) for installation, pairing and device troubleshooting. The first run may need the `bluez` system package, but no additional Python Bluetooth library.

