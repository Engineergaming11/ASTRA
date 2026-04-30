#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -f "${SCRIPT_DIR}/astra_main.py" ]]; then
  printf "[ASTRA][ERROR] astra_main.py not found. Run from project root.\n" >&2
  exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  printf "[ASTRA][ERROR] .venv is missing. Run ./install_and_run.sh first.\n" >&2
  exit 1
fi

exec "${VENV_PYTHON}" "${SCRIPT_DIR}/astra_main.py"
