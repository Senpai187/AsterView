@echo off
setlocal
cd /d "%~dp0"
set "ASTERVIEW_PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%ASTERVIEW_PYTHON%" set "ASTERVIEW_PYTHON=%~dp0..\..\work\venv\Scripts\pythonw.exe"
if not exist "%ASTERVIEW_PYTHON%" (
  echo AsterView needs its Python environment.
  echo Open PowerShell here and run: .\Setup.ps1
  pause
  exit /b 1
)
start "" "%ASTERVIEW_PYTHON%" -m asterview %*
