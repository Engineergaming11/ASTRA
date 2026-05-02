#!/usr/bin/env bash
# Fetch 4206/4207 (12 tiles each, 2MASS 4200-series) and all 5204 LITE tiles (48 files).
# 5200-series: official listing points to NERSC LITE, not data.astrometry.net/5200.
set -euo pipefail

DEST="${HOME}/astrometry/data"
BASE420="https://data.astrometry.net/4200"
BASE520_LITE="https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE"
MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "$DEST"

min_bytes=$((2 * 1024 * 1024))

have_file() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local sz
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  (( sz > min_bytes ))
}

throttle_jobs() {
  while (( $(jobs -pr 2>/dev/null | wc -l | tr -d ' ') >= MAX_JOBS )); do
    sleep 0.25
  done
}

fetch_one() {
  local url="$1" out="$2"
  if have_file "$out"; then
    echo "skip  $out"
    return 0
  fi
  echo "get   $out"
  local tmp
  tmp="$(mktemp "${DEST}/.ast_dl.XXXXXXXX")"
  trap 'rm -f "$tmp"' RETURN
  curl -fSL --connect-timeout 30 --retry 5 --retry-delay 8 --retry-connrefused \
    -o "$tmp" "$url"
  mv -f "$tmp" "$out"
  trap - RETURN
}

for s in 4206 4207; do
  for i in $(seq 0 11); do
    ii=$(printf '%02d' "$i")
    throttle_jobs
    fetch_one "${BASE420}/index-${s}-${ii}.fits" "${DEST}/index-${s}-${ii}.fits" &
  done
done

for i in $(seq 0 47); do
  ii=$(printf '%02d' "$i")
  throttle_jobs
  fetch_one "${BASE520_LITE}/index-5204-${ii}.fits" "${DEST}/index-5204-${ii}.fits" &
done

wait
echo "Done. Indexes in: $DEST"
