#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[setup] Root directory: ${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] python3 not found. Please install Python 3.10+." >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "[setup] Created .env from .env.example"
else
  echo "[setup] .env already exists, keeping current values."
fi

mkdir -p models data

USE_VENV="${USE_VENV:-auto}"
PIP_CMD=""
PYTHON_CMD="python3"
PIP_USER_FLAG=""

create_venv() {
  if python3 -m venv .venv >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PIP_CMD="pip"
    PYTHON_CMD="python"
    echo "[setup] Using virtual environment: .venv"
    return 0
  fi
  return 1
}

if [[ "${USE_VENV}" == "1" || "${USE_VENV}" == "true" ]]; then
  if ! create_venv; then
    echo "[setup] Failed to create virtualenv. Install python3-venv or run with USE_VENV=0." >&2
    exit 1
  fi
elif [[ "${USE_VENV}" == "0" || "${USE_VENV}" == "false" ]]; then
  PIP_CMD="python3 -m pip"
  PYTHON_CMD="python3"
  PIP_USER_FLAG="--user"
  echo "[setup] Installing dependencies into user environment."
else
  if create_venv; then
    :
  else
    PIP_CMD="python3 -m pip"
    PYTHON_CMD="python3"
    PIP_USER_FLAG="--user"
    echo "[setup] Virtualenv unavailable, using user-level pip install."
  fi
fi

if [[ -z "${PIP_CMD}" ]]; then
  PIP_CMD="python3 -m pip"
fi

echo "[setup] Installing Python dependencies..."
${PIP_CMD} install ${PIP_USER_FLAG} --upgrade pip
${PIP_CMD} install ${PIP_USER_FLAG} -r requirements.txt

echo "[setup] Bootstrapping PostgreSQL (database + schema)..."
PYTHONPATH=src ${PYTHON_CMD} scripts/bootstrap_db.py

echo
echo "[setup] Done."
echo "[setup] Start API:      PYTHONPATH=src ${PYTHON_CMD} -m uvicorn pav_detector.api.app:app --host 0.0.0.0 --port 8000"
echo "[setup] Start UI:       PYTHONPATH=src ${PYTHON_CMD} -m streamlit run src/pav_detector/ui/streamlit_app.py"
echo "[setup] Offline mode:   PYTHONPATH=src ${PYTHON_CMD} -m pav_detector.offline.run_offline --help"
