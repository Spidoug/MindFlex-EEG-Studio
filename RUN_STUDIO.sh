#!/usr/bin/env sh
# Portable Linux startup: same dependency checks, virtualenv and logging as Windows.
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || {
    printf '%s\n' '[ERROR] Could not locate the MindFlex EEG Studio folder.' >&2
    exit 1
}
cd "$SCRIPT_DIR" || exit 1

printf '\n%s\n' '=========================================' \
    '      MindFlex EEG Studio 1.0.0' \
    '=========================================' \
    'Preparing environment and launching...' ''

# Prefer a user-activated environment, then the project's own environment,
# then the distribution's Python (which can load distribution-provided Tkinter).
PYTHON=''
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x './.venv/bin/python' ]; then
    PYTHON='./.venv/bin/python'
else
    for candidate in python3 python3.12 python3.13 python3.11 python3.10 python; do
        if command -v "$candidate" >/dev/null 2>&1 && \
            "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    printf '%s\n' '[ERROR] Python 3.10 or newer was not found.' \
        'Install Python 3 and its venv/Tkinter packages using your Linux package manager.' >&2
    if [ -t 0 ]; then
        printf 'Press Enter to exit... '
        read -r _unused || :
    fi
    exit 1
fi

"$PYTHON" -m mindflex.launcher "$@"
status=$?
if [ "$status" -ne 0 ]; then
    printf '\n[ERROR] MindFlex EEG Studio exited with code %s.\n' "$status" >&2
    printf '%s\n' '[HELP] See logs/studio-startup.log, or run ./RUN_STUDIO.sh --diagnose' >&2
    if [ -t 0 ]; then
        printf 'Press Enter to close this terminal... '
        read -r _unused || :
    fi
fi
exit "$status"
