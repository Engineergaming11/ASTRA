#!/usr/bin/env bash
# Daily launcher for ASTRA on Linux / Raspberry Pi.
# Assumes install_and_run_linux.sh has already created .venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ "$(uname -s)" != "Linux" ]]; then
  printf "[ASTRA][ERROR] run_astra_linux.sh is for Linux. Use run_astra.sh on macOS.\n" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/astra_main.py" ]]; then
  printf "[ASTRA][ERROR] astra_main.py not found. Run from project root.\n" >&2
  exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  printf "[ASTRA][ERROR] .venv is missing. Run ./install_and_run_linux.sh first.\n" >&2
  exit 1
fi

# sky_ephemeris.py expects updatedTycho2.parquet (lowercase u).
# Keep launch robust when only UpdatedTycho2.parquet is present.
if [[ -f "${SCRIPT_DIR}/UpdatedTycho2.parquet" && ! -f "${SCRIPT_DIR}/updatedTycho2.parquet" ]]; then
  cp "${SCRIPT_DIR}/UpdatedTycho2.parquet" "${SCRIPT_DIR}/updatedTycho2.parquet"
fi

if [[ ! -f "${SCRIPT_DIR}/updatedTycho2.parquet" ]]; then
  printf "[ASTRA][ERROR] Missing updatedTycho2.parquet (or UpdatedTycho2.parquet).\n" >&2
  printf "[ASTRA][ERROR] Run ./install_and_run_linux.sh or copy the catalog into project root.\n" >&2
  exit 1
fi

exec "${VENV_PYTHON}" "${SCRIPT_DIR}/astra_main.py"
