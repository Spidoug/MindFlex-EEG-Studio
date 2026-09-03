@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title MindFlex EEG Studio

set "ERROR_LOG=%CD%\STARTUP_ERROR.log"
set "TEMP_LOG=%TEMP%\MindFlex-EEG-Studio-%RANDOM%-%RANDOM%.log"
if exist "%ERROR_LOG%" del /q "%ERROR_LOG%" >nul 2>&1
if exist "%TEMP_LOG%" del /q "%TEMP_LOG%" >nul 2>&1

where py >nul 2>&1
if not errorlevel 1 goto use_py

where python >nul 2>&1
if not errorlevel 1 goto use_python

echo [ERRO] Python 3 nao foi encontrado.
echo Instale o Python 3.10 ou superior e marque "Add Python to PATH".
pause
exit /b 1

:use_py
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >"%TEMP_LOG%" 2>&1
if errorlevel 1 goto python_version_error
py -3 -c "import tkinter, numpy, matplotlib, serial" >>"%TEMP_LOG%" 2>&1
if errorlevel 1 (
    echo Instalando ou corrigindo as dependencias...
    py -3 -m pip install -r requirements.txt >>"%TEMP_LOG%" 2>&1
    if errorlevel 1 goto dependency_error
    py -3 -c "import tkinter, numpy, matplotlib, serial" >>"%TEMP_LOG%" 2>&1
    if errorlevel 1 goto dependency_error
)
py -3 -m mindflex >>"%TEMP_LOG%" 2>&1
set "APP_EXIT=%ERRORLEVEL%"
goto finished

:use_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >"%TEMP_LOG%" 2>&1
if errorlevel 1 goto python_version_error
python -c "import tkinter, numpy, matplotlib, serial" >>"%TEMP_LOG%" 2>&1
if errorlevel 1 (
    echo Instalando ou corrigindo as dependencias...
    python -m pip install -r requirements.txt >>"%TEMP_LOG%" 2>&1
    if errorlevel 1 goto dependency_error
    python -c "import tkinter, numpy, matplotlib, serial" >>"%TEMP_LOG%" 2>&1
    if errorlevel 1 goto dependency_error
)
python -m mindflex >>"%TEMP_LOG%" 2>&1
set "APP_EXIT=%ERRORLEVEL%"
goto finished

:python_version_error
echo [ERRO] E necessario Python 3.10 ou superior.>>"%TEMP_LOG%"
set "APP_EXIT=1"
goto failed

:dependency_error
echo [ERRO] Nao foi possivel instalar ou carregar as dependencias.>>"%TEMP_LOG%"
set "APP_EXIT=1"
goto failed

:finished
if "%APP_EXIT%"=="0" (
    if exist "%TEMP_LOG%" del /q "%TEMP_LOG%" >nul 2>&1
    exit /b 0
)

:failed
if exist "%TEMP_LOG%" move /y "%TEMP_LOG%" "%ERROR_LOG%" >nul 2>&1
echo.
echo [ERRO] O MindFlex EEG Studio nao iniciou corretamente.
echo Relatorio: %ERROR_LOG%
echo.
if exist "%ERROR_LOG%" type "%ERROR_LOG%"
pause
exit /b %APP_EXIT%
