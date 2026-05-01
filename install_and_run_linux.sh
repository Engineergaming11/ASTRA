#!/usr/bin/env bash
# Linux / Raspberry Pi installer + launcher for ASTRA.
# Mirrors install_and_run.sh (macOS) but uses apt for system packages,
# the bundled armv8 ZWO SDK, and Debian/Ubuntu-style astrometry.net paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DO_LAUNCH=true
CHECK_ONLY=false
SKIP_APT=false

usage() {
  cat <<'EOF'
Usage: ./install_and_run_linux.sh [--no-launch] [--check-only] [--skip-apt] [--help]

Options:
  --no-launch   Prepare environment but do not start the GUI.
  --check-only  Run dependency checks only (no installs, no launch).
  --skip-apt    Do not run apt-get (use this on systems without sudo or
                when system packages are already in place).
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
    --skip-apt)
      SKIP_APT=true
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

if [[ "$(uname -s)" != "Linux" ]]; then
  die "This installer targets Linux (e.g. Raspberry Pi OS). Use install_and_run.sh on macOS."
fi

if [[ ! -f "${SCRIPT_DIR}/astra_main.py" ]]; then
  die "Run this script from the ASTRA project root."
fi

# Detect 64-bit ARM (the bundled ZWO SDK is armv8 / aarch64).
ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "x86_64" ]]; then
  warn "Architecture ${ARCH} not officially tested. Bundled ZWO SDK is aarch64 only."
fi

log "Checking system prerequisites"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python3 is required."

# System packages we need on Debian/Ubuntu/Raspberry Pi OS:
#   python3-venv / python3-tk : virtualenv + Tk for customtkinter GUI
#   python3-dev / build-essential / pkg-config : compiling any pip wheels that fall back to source
#   libusb-1.0-0 : runtime dep of the ZWO camera SDK (libASICamera2.so)
#   astrometry.net : provides solve-field for plate solving
APT_PKGS=(
  python3-venv
  python3-pip
  python3-tk
  python3-dev
  build-essential
  pkg-config
  libusb-1.0-0
  libusb-1.0-0-dev
  astrometry.net
)

if [[ "${CHECK_ONLY}" == false && "${SKIP_APT}" == false ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing apt packages (sudo): ${APT_PKGS[*]}"
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update -y || warn "apt-get update failed; continuing"
      sudo apt-get install -y "${APT_PKGS[@]}" || warn "apt install reported errors; continuing"
    else
      warn "sudo not available; skipping apt install (rerun with --skip-apt to silence)."
    fi
  else
    warn "apt-get not found. Install equivalents manually: ${APT_PKGS[*]}"
  fi
fi

if [[ "${CHECK_ONLY}" == false ]]; then
  log "Creating or reusing virtual environment"
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip wheel
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

log "Checking ZWO ASI SDK library (Linux .so)"
# CamGUI loader looks for libASICamera2.so next to astra_main.py.
# The repo ships the aarch64 build under armv8/, so symlink it into place.
ASI_TARGET="${SCRIPT_DIR}/libASICamera2.so"
ASI_SOURCE_CANDIDATES=(
  "${SCRIPT_DIR}/armv8/libASICamera2.so"
  "${SCRIPT_DIR}/armv8/libASICamera2.so.1.41"
)
ASI_FOUND=false
if [[ -f "${ASI_TARGET}" ]]; then
  ASI_FOUND=true
  printf "[ASTRA] Found camera SDK: %s\n" "${ASI_TARGET}"
else
  if [[ "${CHECK_ONLY}" == false ]]; then
    for src in "${ASI_SOURCE_CANDIDATES[@]}"; do
      if [[ -f "${src}" ]]; then
        log "Linking ${src} -> ${ASI_TARGET}"
        ln -sf "${src}" "${ASI_TARGET}"
        ASI_FOUND=true
        break
      fi
    done
  fi
fi

if [[ "${ASI_FOUND}" == false ]]; then
  warn "ASI SDK .so not found. GUI can still run in demo mode."
  warn "Place libASICamera2.so in project root or under armv8/ from the ZWO Linux SDK."
fi

# udev rules let a non-root user open the ZWO camera over USB.
if [[ "${CHECK_ONLY}" == false ]]; then
  if [[ ! -f "/etc/udev/rules.d/asi.rules" ]]; then
    LOCAL_RULES_CANDIDATES=(
      "${SCRIPT_DIR}/asi.rules"
      "${SCRIPT_DIR}/armv8/asi.rules"
      "${SCRIPT_DIR}/lib/asi.rules"
    )
    INSTALLED_RULES=false
    if command -v sudo >/dev/null 2>&1; then
      for r in "${LOCAL_RULES_CANDIDATES[@]}"; do
        if [[ -f "${r}" ]]; then
          log "Installing ZWO udev rules from ${r}"
          sudo install -m 0644 "${r}" /etc/udev/rules.d/asi.rules \
            && sudo udevadm control --reload-rules \
            && sudo udevadm trigger \
            && INSTALLED_RULES=true
          break
        fi
      done
    fi
    if [[ "${INSTALLED_RULES}" == false ]]; then
      warn "ZWO udev rules not installed. Without them the camera may need root."
      warn "Drop asi.rules from the ZWO Linux SDK into the project root and re-run."
    fi
  fi
fi

log "Checking astrometry.net (solve-field)"
if command -v solve-field >/dev/null 2>&1; then
  solve-field --help >/dev/null 2>&1 || warn "solve-field found but help check failed."
  printf "[ASTRA] solve-field detected at: %s\n" "$(command -v solve-field)"
else
  warn "solve-field not found. Plate solving will be unavailable."
  warn "Install with: sudo apt install astrometry.net"
fi

if command -v solve-field >/dev/null 2>&1; then
  ASTRO_DATA_DIR="${HOME}/astrometry/data"
  mkdir -p "${ASTRO_DATA_DIR}" || warn "Could not create ${ASTRO_DATA_DIR}"
  # Debian/Pi OS install of astrometry.net puts its config here:
  ASTRO_CFG_CANDIDATES=(
    "/etc/astrometry.cfg"
    "/usr/local/etc/astrometry.cfg"
  )
  ASTRO_CFG=""
  for c in "${ASTRO_CFG_CANDIDATES[@]}"; do
    if [[ -f "${c}" ]]; then
      ASTRO_CFG="${c}"
      break
    fi
  done
  if [[ -n "${ASTRO_CFG}" ]]; then
    CURRENT_ADD_PATH="$(awk '/^add_path /{print $2; exit}' "${ASTRO_CFG}" 2>/dev/null || true)"
    if [[ -n "${CURRENT_ADD_PATH}" && ! -d "${CURRENT_ADD_PATH}" ]]; then
      log "Updating astrometry index path to ${ASTRO_DATA_DIR}"
      TMP_CFG="$(mktemp)"
      awk -v newpath="${ASTRO_DATA_DIR}" '
        BEGIN { done=0 }
        /^add_path / && done==0 { print "add_path " newpath; done=1; next }
        { print }
        END { if (done==0) print "add_path " newpath }
      ' "${ASTRO_CFG}" > "${TMP_CFG}"
      if command -v sudo >/dev/null 2>&1; then
        sudo cp "${TMP_CFG}" "${ASTRO_CFG}" || warn "Could not update ${ASTRO_CFG}"
      else
        cp "${TMP_CFG}" "${ASTRO_CFG}" 2>/dev/null || warn "Could not update ${ASTRO_CFG} (need sudo)"
      fi
      rm -f "${TMP_CFG}"
    fi
  else
    warn "No astrometry.cfg found in /etc or /usr/local/etc; index path not auto-updated."
  fi
  INDEX_COUNT="$(ls "${ASTRO_DATA_DIR}"/index-*.fits 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${INDEX_COUNT}" == "0" ]]; then
    warn "No astrometry index files found in ${ASTRO_DATA_DIR}."
    warn "Plate solve will fail until indexes are downloaded (e.g. apt install astrometry-data-tycho2)."
  else
    printf "[ASTRA] Astrometry index files found: %s\n" "${INDEX_COUNT}"
  fi
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
