@echo off
setlocal enabledelayedexpansion

REM Full Windows installer for PAV Detector Workbench.
REM - Creates .venv (if missing)
REM - Installs requirements
REM - Ensures .env exists
REM - Initializes DB schema (best effort)

cd /d "%~dp0\.."
echo [install] Project root: %CD%

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo [install] Created .env from .env.example
) else (
  echo [install] .env already exists
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PY_CMD=python"
  ) else (
    echo [install] Python not found. Install Python 3.10+ and retry.
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo [install] Creating virtual environment...
  %PY_CMD% -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo [install] Failed to create .venv
    exit /b 1
  )
) else (
  echo [install] Using existing .venv
)

echo [install] Installing Python dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [install] Dependency install failed.
  exit /b 1
)

echo [install] Initializing PostgreSQL schema (best effort)...
set "PYTHONPATH=src"
".venv\Scripts\python.exe" -m pav_detector.db.init_db
if errorlevel 1 (
  echo [install] DB init failed. You can fix .env/DB and rerun later.
) else (
  echo [install] DB schema initialized.
)

echo [install] Done.
echo [install] Use windows\start_workbench.vbs for no-console launch.
pause
