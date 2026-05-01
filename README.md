# ASTRA

**Autonomous Sky Tracking and Recon Apparatus** — desktop tools for ZWO ASI cameras, iOptron HAE-series mounts, ephemeris-driven target tracking, and local astrometry plate solving.

## Quickstart (macOS, easiest path)

### 1) Clone and enter the project
```bash
git clone <YOUR_REPO_URL>
cd ASTRA-main
```

### 2) Run one setup + launch command
```bash
bash ./install_and_run.sh
```

That command will:
- create/update `.venv`,
- install Python dependencies from `requirements.txt`,
- attempt to install `astrometry.net` (`solve-field`) via Homebrew,
- check required data files,
- check for ZWO camera SDK dylib,
- start `astra_main.py`.

## Daily startup (after first setup)

```bash
bash ./run_astra.sh
```

## Optional installer modes

```bash
bash ./install_and_run.sh --no-launch
bash ./install_and_run.sh --check-only
```

- `--no-launch`: install/check only, do not open GUI.
- `--check-only`: verify dependencies only; no installs performed.

## What runs where

| Program | Purpose |
|--------|---------|
| `astra_main.py` | Main app: camera preview/capture, Mount tab, Site / Sky tools, Deep Sky stacking, Plate solve tab. |
| `SkyTracker2.py` | Optional companion target picker; writes `LiveWCS.txt` used by main GUI. |
| `Formal_PlateSolve_GUI.py` | Standalone plate solve UI. |
| `EQMountControls.py` | Standalone mount control window. |
| `deep_sky_stacker.py` | Offline stacking pipeline used by the **Deep Sky** tab (importable on its own). |

Run from the project root so paths resolve correctly.

## Hardware and optional dependencies

### ZWO ASI camera
- Keep `libASICamera2.dylib` (or versioned `libASICamera2.dylib.*`) in the project root.
- If missing, GUI still starts in demo mode (no live camera feed).

### Plate solving
- `solve-field` must be available on `PATH`.
- Installer tries to install `astrometry.net` via Homebrew when possible.
- You still need astrometry index files sized for your telescope/camera FOV.

### iOptron HAE mount
- Serial mode uses `115200` baud (`/dev/tty.*` on macOS).
- Wi-Fi default in UI is `10.10.100.254:8899` (adjust if needed).

## End-to-end usage flow

1. Start ASTRA (`bash ./run_astra.sh`).
2. Set site location in `Site / Sky` and save.
3. Connect camera and start preview/capture.
4. Plate solve a FITS image in `Plate solve`.
5. Connect mount in `Mount` tab (serial or Wi-Fi), then slew/track.
6. Optional: run `python3 SkyTracker2.py` in a second terminal for live target feed.

## Deep Sky stacking (offline)

The **Deep Sky** tab in the main GUI lets you combine multiple captured FITS frames into a single calibrated, stretched image — fully offline, no internet or cloud calls.

Workflow:

1. Capture lights with `Frame Type = Light` so they land in `~/ASICAP/CapGUI/<project>/Light/`. Optionally also capture `Dark`, `Flat`, `Bias` frames in the same project.
2. Open the **Deep Sky** tab.
3. For each frame kind, click **Add files…** or **Add folder…**. Defaults point to the matching subfolder under your active project.
4. Tune settings if needed:
   - **Combine**: `mean`, `median`, or `sigma_clip` (default — outlier-rejected mean).
   - **σ low / σ high**: rejection thresholds when using sigma-clip.
   - **Reference**: `auto (most stars)` picks the best frame; `first frame` forces frame 1.
   - **Bayer**: `auto` reads `BAYERPAT` from the FITS header. Select `mono` to skip debayering, or pick an explicit pattern.
   - **Stretch**: `asinh` (default — emphasizes faint nebulosity), `zscale`, `percentile`, or `linear` for the preview PNG.
5. Click **Stack**. Progress and per-frame status appear in the log; a stretched preview is shown when finished.
6. Use **Use stacked FITS in plate solve** to immediately load the result into the Plate solve tab, or **Open output folder** / **Save preview as…**.

Outputs land in `<save_path>/<project>/Stacks/<timestamp>/`:

- `stack_<timestamp>.fits` — 32-bit float stacked FITS (color is stored channel-first as `(3, H, W)`).
- `stack_<timestamp>_preview.png` — 8-bit stretched preview suitable for sharing.

What each calibration frame does:

- **Bias** — sensor read noise floor. Subtracted from lights and from flats.
- **Dark** — thermal + bias signal at the same exposure/temperature. Subtracted from lights.
- **Flat** — vignetting / dust pattern at uniform illumination. Lights are divided by the (bias-subtracted, normalized) master flat.

Offline guarantee: the stacker (`deep_sky_stacker.py`) makes no HTTP, FTP, or socket calls. It uses `numpy`, `opencv-python`, `astropy`, `sep`, `scipy`, and `astroalign` — all pure-Python or wheel-distributed and operate on local arrays only. Stacked FITS files can be fed directly into the offline `solve-field`-based plate solve tab.

## Validation checks

If you want to verify setup without launching UI:

```bash
bash ./install_and_run.sh --check-only
```

Checks include:
- Python import sanity for core modules (`zwoasi`, `skyfield`, `astropy`, etc.),
- `solve-field` availability,
- required local data files (`de440.bsp`, Tycho/Messier catalogs),
- ASI SDK dylib detection.

## Advanced/manual setup

If you prefer manual control:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python astra_main.py
```

## Troubleshooting

- Camera missing: check USB + ASI SDK dylib in project root.
- `solve-field` missing: install `astrometry.net` and ensure it is on `PATH`.
- Plate solves slow/failing: install matching astrometry index files for your FOV.
- Plate solve path errors on macOS: ASTRA now auto-creates `~/astrometry/data` and repoints Homebrew `astrometry.cfg` away from stale versioned Cellar paths.
- Hard solve cases: ASTRA plate solve now runs multi-pass (hinted -> relaxed -> blind) and falls back to nearby debayered PNG files when FITS does not solve cleanly.
- Sky tracker blank: run `SkyTracker2.py` from this same project directory.
- Deep Sky tab logs `astroalign unavailable`: run `pip install astroalign` (or re-run `install_and_run.sh`). Stacking still works in this state via a translation-only centroid fallback, but is less robust to field rotation.
- Deep Sky stack is mostly black: try a different stretch (`asinh` -> `zscale`), or make sure your master flat is not nearly zero in some pixels (the pipeline clamps tiny flat values but a badly exposed flat will still over-darken).
