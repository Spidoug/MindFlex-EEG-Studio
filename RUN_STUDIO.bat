@echo off
setlocal
cd /d "%~dp0"
python -m mindflex
if errorlevel 1 pause
