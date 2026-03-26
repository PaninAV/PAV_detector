@echo off
setlocal

cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo [start] .venv not found. Run windows\install_windows.bat first.
  pause
  exit /b 1
)

set "PYTHONPATH=src"
".venv\Scripts\python.exe" -m streamlit run "src\pav_detector\ui\workbench_app.py"

