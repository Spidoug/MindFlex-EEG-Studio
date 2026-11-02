@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Always start from the directory containing this file, even when launched
rem by double-clicking in Explorer or from another working directory.
pushd "%~dp0"
if errorlevel 1 (
    echo [ERROR] Could not access the MindFlex EEG Studio folder.
    pause
    exit /b 1
)

rem PyWinRT 3.x ships Windows wheels for CPython 3.11-3.13.
rem Prefer the project environment, then an explicitly supported interpreter.
set "PYTHON_COMMAND="
set "VENV_PY="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "VENV_PY=%CD%\.venv\Scripts\python.exe"
        goto launch
    )
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_COMMAND=py -3.12" & goto launch)
    py -3.13 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_COMMAND=py -3.13" & goto launch)
    py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_COMMAND=py -3.11" & goto launch)
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_COMMAND=python" & goto launch)
)

where python3 >nul 2>&1
if not errorlevel 1 (
    python3 -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_COMMAND=python3" & goto launch)
)

echo.
echo [ERROR] No compatible Python interpreter was found.
echo Install 64-bit Python 3.12 with Tkinter from https://www.python.org/downloads/
echo Python 3.11 and 3.13 are also supported for this Windows launcher.
echo Restart this script after installation. Python 3.14+ is not supported
echo by the currently pinned WinRT 3.x Bluetooth dependencies.
echo.
pause
popd
exit /b 1

:launch
echo.
echo =========================================
echo       MindFlex EEG Studio 1.0.0
echo =========================================
echo Preparing environment and launching...
echo.
if defined VENV_PY (
    "%VENV_PY%" -m mindflex.launcher %*
) else (
    %PYTHON_COMMAND% -m mindflex.launcher %*
)
set "STUDIO_EXIT_CODE=%ERRORLEVEL%"
if not "%STUDIO_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] MindFlex EEG Studio returned exit code %STUDIO_EXIT_CODE%.
    echo See the startup log path displayed above for full error details.
    echo The console will remain open so the error can be read.
    echo.
    pause
)
popd
endlocal & exit /b %STUDIO_EXIT_CODE%
