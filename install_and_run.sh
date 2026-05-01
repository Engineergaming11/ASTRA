#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DO_LAUNCH=true
CHECK_ONLY=false

usage() {
  cat <<'EOF'
Usage: ./install_and_run.sh [--no-launch] [--check-only] [--help]

Options:
  --no-launch   Prepare environment but do not start the GUI.
  --check-only  Run dependency checks only (no installs, no launch).
  --help        Show this help message.
EOF
}

log() {
  printf "\n[ASTRA] %s\n" "$1"
}

warn() {
  printf "[ASTRA][WARN] %s\n" "$1"
}

die() {
  printf "[ASTRA][ERROR] %s\n" "$1" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-launch)
      DO_LAUNCH=false
      shift
      ;;
    --check-only)
      CHECK_ONLY=true
      DO_LAUNCH=false
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  die "This installer currently targets macOS only."
fi

if [[ ! -f "${SCRIPT_DIR}/astra_main.py" ]]; then
  die "Run this script from the ASTRA project root."
fi

log "Checking system prerequisites"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python3 is required."

if ! command -v pip3 >/dev/null 2>&1; then
  warn "pip3 not found on PATH. Will use python -m pip from virtualenv."
fi

if [[ "${CHECK_ONLY}" == false ]]; then
  log "Creating or reusing virtual environment"
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
else
  source "${VENV_DIR}/bin/activate" 2>/dev/null || warn "No .venv found; checks may fail."
fi

log "Validating required Python imports"
python - <<'PY'
import importlib
mods = ["customtkinter", "numpy", "cv2", "PIL", "pandas", "pyarrow", "astropy", "skyfield", "serial", "sep", "scipy", "matplotlib", "zwoasi"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        failed.append(m)
if failed:
    print("FAILED_IMPORTS=" + ",".join(failed))
    raise SystemExit(1)
print("Python package import check passed.")
PY

log "Checking local data files"
for required_file in "de440.bsp" "Messier_Updated.parquet"; do
  if [[ ! -f "${SCRIPT_DIR}/${required_file}" ]]; then
    warn "Missing ${required_file}. Some sky features may not work."
  fi
done

# sky_ephemeris.py expects updatedTycho2.parquet (lowercase u)
if [[ -f "${SCRIPT_DIR}/UpdatedTycho2.parquet" && ! -f "${SCRIPT_DIR}/updatedTycho2.parquet" ]]; then
  if [[ "${CHECK_ONLY}" == false ]]; then
    log "Creating updatedTycho2.parquet alias from UpdatedTycho2.parquet"
    cp "${SCRIPT_DIR}/UpdatedTycho2.parquet" "${SCRIPT_DIR}/updatedTycho2.parquet"
  else
    warn "updatedTycho2.parquet not found (UpdatedTycho2.parquet exists). Alias needed."
  fi
fi

if [[ ! -f "${SCRIPT_DIR}/updatedTycho2.parquet" ]]; then
  warn "Missing updatedTycho2.parquet (star preset catalog may fail)."
fi

log "Checking astrometry.net (solve-field)"
if command -v solve-field >/dev/null 2>&1; then
  solve-field --help >/dev/null 2>&1 || warn "solve-field found but help check failed."
  printf "[ASTRA] solve-field detected at: %s\n" "$(command -v solve-field)"
else
  if [[ "${CHECK_ONLY}" == false ]] && command -v brew >/dev/null 2>&1; then
    log "Installing astrometry.net via Homebrew"
    brew install astrometry-net || warn "Homebrew install failed. Install astrometry.net manually."
  else
    warn "solve-field not found and brew unavailable (or check-only)."
  fi
fi

if ! command -v solve-field >/dev/null 2>&1; then
  warn "Plate solving will be unavailable until astrometry.net is installed."
else
  ASTRO_DATA_DIR="${HOME}/astrometry/data"
  mkdir -p "${ASTRO_DATA_DIR}" || warn "Could not create ${ASTRO_DATA_DIR}"
  ASTRO_CFG="/opt/homebrew/etc/astrometry.cfg"
  if [[ -f "${ASTRO_CFG}" ]]; then
    CURRENT_ADD_PATH="$(awk '/^add_path /{print $2; exit}' "${ASTRO_CFG}" 2>/dev/null || true)"
    if [[ -n "${CURRENT_ADD_PATH}" && ! -d "${CURRENT_ADD_PATH}" ]]; then
      log "Updating astrometry index path to ${ASTRO_DATA_DIR}"
      TMP_CFG="$(mktemp)"
      awk -v newpath="${ASTRO_DATA_DIR}" '
        BEGIN { done=0 }
        /^add_path / && done==0 { print "add_path " newpath; done=1; next }
        { print }
        END { if (done==0) print "add_path " newpath }
      ' "${ASTRO_CFG}" > "${TMP_CFG}" && mv "${TMP_CFG}" "${ASTRO_CFG}" || warn "Could not update ${ASTRO_CFG}"
    fi
  fi
  INDEX_COUNT="$(ls "${ASTRO_DATA_DIR}"/index-*.fits 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${INDEX_COUNT}" == "0" ]]; then
    warn "No astrometry index files found in ${ASTRO_DATA_DIR}."
    warn "Plate solve can fail until indexes are downloaded."
  else
    printf "[ASTRA] Astrometry index files found: %s\n" "${INDEX_COUNT}"
  fi
fi

log "Checking ZWO ASI SDK library"
ASI_FOUND=false
for asi_lib in "libASICamera2.dylib" "libASICamera2.dylib.1.41"; do
  if [[ -f "${SCRIPT_DIR}/${asi_lib}" ]]; then
    ASI_FOUND=true
    printf "[ASTRA] Found camera SDK: %s\n" "${asi_lib}"
    break
  fi
done

if [[ "${ASI_FOUND}" == false ]]; then
  warn "ASI SDK dylib not found. GUI can still run in demo mode."
  warn "Download macOS SDK from ZWO and place libASICamera2.dylib in project root."
fi

if [[ "${CHECK_ONLY}" == true ]]; then
  log "Check-only mode complete."
  exit 0
fi

if [[ "${DO_LAUNCH}" == true ]]; then
  log "Starting ASTRA GUI"
  exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/astra_main.py"
fi

log "Setup complete (launch skipped)."
