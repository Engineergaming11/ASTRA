#!/usr/bin/env python3
"""
test_solve_local.py - Professional Edition

Modified GUI that:
 - Professional, formal appearance (greys, blacks, greens)
 - Night mode with red-scaling for night viewing
 - Clean, minimal design
 - Only accepts RA, Dec, and FOV (degrees) in the settings pane
 - Calls local astrometry.net `solve-field` via subprocess
 - Provides annotated PNG and solved image viewing
 - Uses threads for responsive UI

Dependencies (same as before + astropy, sep):
 - tkinter (builtin)
 - numpy
 - PIL (Pillow)
 - astropy
 - sep
 - scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog

from astra_dialogs import askyesno, showerror, showwarning
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import subprocess
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path
import time
import traceback

from ui_display_profile import ui_compact

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import sep

SOLVE_FIELD_CMD = "solve-field"


def _preferred_astrometry_data_dir() -> Path:
    return Path.home() / "astrometry" / "data"


def _candidate_astrometry_cfg_paths() -> list[Path]:
    """Ordered search for astrometry-engine config. Override with ASTRA_ASTROMETRY_CFG or ASTROMETRY_CFG."""
    paths: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        try:
            cand = p.expanduser()
            if not cand.is_file():
                return
            key = str(cand.resolve())
        except OSError:
            return
        if key not in seen:
            seen.add(key)
            paths.append(cand)

    for env_key in ("ASTRA_ASTROMETRY_CFG", "ASTROMETRY_CFG"):
        raw = (os.environ.get(env_key) or "").strip()
        if raw:
            add(Path(raw))
    # Linux / Raspberry Pi OS (apt install astrometry.net)
    for p in (Path("/etc/astrometry.cfg"), Path("/usr/local/etc/astrometry.cfg")):
        add(p)
    # User-writable copies when /etc is missing or not preferred (common on embedded setups)
    home = Path.home()
    for p in (home / "astrometry" / "astrometry.cfg", home / ".config" / "astrometry" / "astrometry.cfg"):
        add(p)
    # macOS Homebrew
    hb = Path("/opt/homebrew/etc/astrometry.cfg")
    add(hb)
    cellar = Path("/opt/homebrew/Cellar/astrometry-net")
    if cellar.exists():
        for ver in sorted(cellar.iterdir(), reverse=True):
            add(ver / "etc" / "astrometry.cfg")
            break
    return paths


def _primary_astrometry_cfg_path() -> Path | None:
    cands = _candidate_astrometry_cfg_paths()
    return cands[0] if cands else None


def _normalize_astrometry_cfg_path_line(cfg_path: Path, preferred_data_dir: Path) -> tuple[bool, str]:
    """Ensure add_path points to an existing stable user-writable directory."""
    try:
        content = cfg_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Could not read {cfg_path}: {e}"
    lines = content.splitlines()
    add_idx = None
    current = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("add_path "):
            add_idx = i
            current = s.split(" ", 1)[1].strip()
            break
    preferred_line = f"add_path {preferred_data_dir}"
    if add_idx is None:
        lines.insert(0, preferred_line)
    else:
        old_path = Path(current).expanduser() if current else None
        old_exists = bool(old_path and old_path.exists())
        if old_exists and str(old_path) == str(preferred_data_dir):
            return True, f"Astrometry config already uses {preferred_data_dir}"
        if old_exists and str(old_path) != str(preferred_data_dir):
            # Keep valid user path chosen by user.
            return True, f"Astrometry config keeps existing valid path {old_path}"
        lines[add_idx] = preferred_line
    new_content = "\n".join(lines) + ("\n" if content.endswith("\n") or not lines else "")
    if new_content == content:
        return True, f"Astrometry config unchanged at {cfg_path}"
    try:
        cfg_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return False, f"Could not update {cfg_path}: {e}"
    return True, f"Updated {cfg_path} add_path -> {preferred_data_dir}"


def ensure_astrometry_runtime_setup() -> dict:
    """
    Best-effort local setup:
    - create ~/astrometry/data
    - repoint astrometry cfg add_path away from broken Cellar data paths
    """
    preferred = _preferred_astrometry_data_dir()
    status_lines = []
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        status_lines.append(f"Using astrometry data dir: {preferred}")
    except Exception as e:
        status_lines.append(f"Could not create astrometry data dir {preferred}: {e}")
    cfg_updated = False
    for cfg in _candidate_astrometry_cfg_paths():
        if not cfg.exists():
            continue
        ok, msg = _normalize_astrometry_cfg_path_line(cfg, preferred)
        status_lines.append(msg)
        cfg_updated = cfg_updated or ok
        # first valid config is enough for Homebrew install.
        if ok:
            break
    index_count = len(list(preferred.glob("index-*.fits"))) if preferred.exists() else 0
    status_lines.append(f"Found {index_count} index file(s) in {preferred}")
    return {"data_dir": str(preferred), "index_count": index_count, "messages": status_lines, "cfg_checked": cfg_updated}

_BRIGHT_STAR_CATALOG = [
    {"name": "Sirius", "ra_deg": 101.287155, "dec_deg": -16.716116, "distance_ly": 8.6},
    {"name": "Canopus", "ra_deg": 95.987958, "dec_deg": -52.695661, "distance_ly": 310.0},
    {"name": "Arcturus", "ra_deg": 213.915300, "dec_deg": 19.182410, "distance_ly": 36.7},
    {"name": "Vega", "ra_deg": 279.234734, "dec_deg": 38.783688, "distance_ly": 25.0},
    {"name": "Capella", "ra_deg": 79.172327, "dec_deg": 45.997991, "distance_ly": 42.9},
    {"name": "Rigel", "ra_deg": 78.634467, "dec_deg": -8.201639, "distance_ly": 860.0},
    {"name": "Procyon", "ra_deg": 114.825493, "dec_deg": 5.224993, "distance_ly": 11.5},
    {"name": "Betelgeuse", "ra_deg": 88.792939, "dec_deg": 7.407064, "distance_ly": 548.0},
    {"name": "Achernar", "ra_deg": 24.428611, "dec_deg": -57.236753, "distance_ly": 139.0},
    {"name": "Hadar", "ra_deg": 210.955833, "dec_deg": -60.373056, "distance_ly": 390.0},
    {"name": "Altair", "ra_deg": 297.695828, "dec_deg": 8.868322, "distance_ly": 16.7},
    {"name": "Acrux", "ra_deg": 186.649563, "dec_deg": -63.099092, "distance_ly": 321.0},
    {"name": "Aldebaran", "ra_deg": 68.980163, "dec_deg": 16.509302, "distance_ly": 65.0},
    {"name": "Antares", "ra_deg": 247.351915, "dec_deg": -26.432002, "distance_ly": 550.0},
    {"name": "Spica", "ra_deg": 201.298247, "dec_deg": -11.161322, "distance_ly": 250.0},
    {"name": "Pollux", "ra_deg": 116.328958, "dec_deg": 28.026183, "distance_ly": 33.7},
    {"name": "Fomalhaut", "ra_deg": 344.412750, "dec_deg": -29.622236, "distance_ly": 25.1},
    {"name": "Deneb", "ra_deg": 310.357979, "dec_deg": 45.280339, "distance_ly": 2615.0},
    {"name": "Regulus", "ra_deg": 152.092962, "dec_deg": 11.967208, "distance_ly": 79.0},
    {"name": "Adhara", "ra_deg": 104.656453, "dec_deg": -28.972086, "distance_ly": 430.0},
    {"name": "Shaula", "ra_deg": 263.402167, "dec_deg": -37.103824, "distance_ly": 570.0},
    {"name": "Castor", "ra_deg": 113.649345, "dec_deg": 31.888282, "distance_ly": 51.6},
    {"name": "Gacrux", "ra_deg": 187.791498, "dec_deg": -57.113214, "distance_ly": 88.6},
    {"name": "Bellatrix", "ra_deg": 81.282764, "dec_deg": 6.349703, "distance_ly": 250.0},
    {"name": "Elnath", "ra_deg": 81.572971, "dec_deg": 28.607451, "distance_ly": 131.0},
    {"name": "Miaplacidus", "ra_deg": 138.300833, "dec_deg": -69.717222, "distance_ly": 111.0},
    {"name": "Alnilam", "ra_deg": 84.053389, "dec_deg": -1.201917, "distance_ly": 1340.0},
    {"name": "Alnair", "ra_deg": 332.058271, "dec_deg": -46.960975, "distance_ly": 101.0},
    {"name": "Alioth", "ra_deg": 193.507292, "dec_deg": 55.959822, "distance_ly": 82.6},
    {"name": "Alnitak", "ra_deg": 85.189694, "dec_deg": -1.942572, "distance_ly": 736.0},
]

# Color schemes
THEMES = {
    "day": {
        "bg_primary": "#f5f5f5",
        "bg_secondary": "#ebebeb",
        "bg_tertiary": "#e0e0e0",
        "fg_primary": "#1a1a1a",
        "fg_secondary": "#4a4a4a",
        "fg_tertiary": "#7a7a7a",
        "accent": "#2d7a3e",  # Forest green
        "accent_light": "#4a9f54",
        "accent_dark": "#1f5428",
        "canvas_bg": "#ffffff",
        "border": "#cccccc",
    },
    "night": {
        "bg_primary": "#1a0a05",
        "bg_secondary": "#2d1410",
        "bg_tertiary": "#3d1f1a",
        "fg_primary": "#ff6666",
        "fg_secondary": "#cc5555",
        "fg_tertiary": "#994444",
        "accent": "#ff4444",
        "accent_light": "#ff6666",
        "accent_dark": "#cc0000",
        "canvas_bg": "#0d0503",
        "border": "#4d2020",
    }
}

# ---------- Utility functions ----------

def _index_scale_ranges_arcmin() -> list[tuple[int, float, float]]:
    """Skymark diameter ranges (arcmin) per 4XXY scale digit, per astrometry.net docs."""
    return [
        (0, 2.0, 2.8),
        (1, 2.8, 4.0),
        (2, 4.0, 5.6),
        (3, 5.6, 8.0),
        (4, 8.0, 11.0),
        (5, 11.0, 16.0),
        (6, 16.0, 22.0),
        (7, 22.0, 30.0),
        (8, 30.0, 42.0),
        (9, 42.0, 60.0),
        (10, 60.0, 85.0),
        (11, 85.0, 120.0),
        (12, 120.0, 170.0),
        (13, 170.0, 240.0),
        (14, 240.0, 340.0),
        (15, 340.0, 480.0),
        (16, 480.0, 680.0),
        (17, 680.0, 1000.0),
        (18, 1000.0, 1400.0),
        (19, 1400.0, 2000.0),
    ]


def _scales_recommended_for_fov(fov_deg: float, low_frac: float = 0.25, high_frac: float = 1.0) -> list[int]:
    """Return 4XXY scale digits whose skymark range overlaps the chosen FOV fractions.

    Default 25%-100% of FOV is a practical sweet spot: small enough to download
    (avoiding the 48-tile families at the smallest scales when not strictly needed)
    while still giving solve-field plenty of usable quads.
    """
    if fov_deg is None or fov_deg <= 0:
        return [5, 6, 7, 8]
    fov_arcmin = float(fov_deg) * 60.0
    lo, hi = fov_arcmin * low_frac, fov_arcmin * high_frac
    chosen = [s for s, a, b in _index_scale_ranges_arcmin() if b >= lo and a <= hi]
    return chosen or [5, 6, 7, 8]


def _installed_index_scales(data_dir: Path) -> set[int]:
    """Collect the 4XXY scale digits present in `data_dir` (4200-series naming)."""
    scales: set[int] = set()
    if not data_dir.exists():
        return scales
    for p in data_dir.glob("index-42*.fits"):
        stem = p.stem  # e.g. index-4205-03 or index-4208
        try:
            parts = stem.split("-")
            family_token = parts[1]  # 4205, 4208, 4210, ...
            digits = family_token[2:]  # "05", "08", "10", ...
            scale = int(digits)
            scales.add(scale)
        except (IndexError, ValueError):
            continue
    return scales


def run_solve_field_local(
    image_path,
    ra,
    dec,
    fov_deg,
    out_basename,
    solve_field_cmd=SOLVE_FIELD_CMD,
    timeout=300,
    output_dir=None,
):
    """Run local astrometry.net solve-field with robust fallbacks.

    If ``output_dir`` is set, astrometry logs and FITS products are written there
    (the input image may live elsewhere). Otherwise outputs stay next to the image.
    """
    image_path = Path(image_path).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else image_path.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    out_basename_path = out_dir / out_basename
    setup_info = ensure_astrometry_runtime_setup()
    setup_log = "\n".join(setup_info.get("messages", []))

    # Force solve-field to write outputs under our chosen basename so we can find them.
    out_args_for = lambda base: [
        "--out", base,
        "--new-fits", str(out_dir / (base + ".new.fits")),
    ]

    def _solve_with_args(img: Path, extra_args: list[str], pass_name: str, base: str):
        # Always include ~/astrometry/data so indexes from "Download indexes for FOV"
        # are visible on Linux/Pi even when /etc/astrometry.cfg only lists /usr/share/...
        # (astrometry-engine treats --index-dir as additional to the config file).
        data_dir = Path(setup_info.get("data_dir") or _preferred_astrometry_data_dir())
        cfg_path = _primary_astrometry_cfg_path()
        cfg_args = ["--config", str(cfg_path)] if cfg_path else []
        cmd = (
            [solve_field_cmd]
            + cfg_args
            + ["--index-dir", str(data_dir), str(img), "--overwrite"]
            + out_args_for(base)
            + extra_args
        )
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(out_dir),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as te:
            return te.stdout or "", f"{pass_name}: Timeout running solve-field", 124
        hdr = f"\n=== {pass_name} ===\nCMD: {' '.join(cmd)}\n"
        out = hdr + (proc.stdout or "")
        err = hdr + (proc.stderr or "")
        return out, err, proc.returncode

    def _find_solved_fits(base: str) -> str | None:
        """solve-field success indicator is `<base>.solved`. The plate is `<base>.new.fits`
        (we forced it) or `<base>.new` (default). Prefer the .new.fits file."""
        base_path = out_dir / base
        solved_marker = base_path.with_suffix(".solved")
        if not solved_marker.exists():
            return None
        for cand in (
            out_dir / (base + ".new.fits"),
            out_dir / (base + ".new"),
        ):
            if cand.exists():
                return str(cand)
        # Fallback: any FITS file with our basename.
        for f in sorted(glob.glob(str(base_path) + "*.fits")):
            if f.endswith((".new.fits", ".axy.fits", ".corr.fits", ".rdls.fits", ".match.fits")):
                if f.endswith(".new.fits"):
                    return f
                continue
            return f
        return None

    ra_f = float(ra) if ra is not None else None
    dec_f = float(dec) if dec is not None else None
    fov_f = max(0.01, float(fov_deg))
    radius_deg = max(0.5, min(15.0, fov_f * 1.2))
    # Allow very small pixel scales for high-resolution sensors with narrow FOV.
    scale_low = max(0.05, (fov_f * 3600.0 / 5000.0) * 0.45)
    scale_high = min(5.0, (fov_f * 3600.0 / 500.0) * 1.65)
    # Blind solve still knows FOV from the UI; constraining plate scale avoids a
    # huge astrometry search space and reduces spurious timeouts under --cpulimit.
    scale_blind_args = [
        "--scale-units", "arcsecperpix",
        "--scale-low", str(max(0.05, scale_low * 0.5)),
        "--scale-high", str(min(10.0, scale_high * 2.0)),
    ]

    passes = []
    if ra_f is not None and dec_f is not None:
        passes.extend([
            ("Hinted solve (tight)", [
                "--no-plots", "--downsample", "2", "--sigma", "3", "--objs", "300",
                "--ra", str(ra_f), "--dec", str(dec_f), "--radius", str(radius_deg),
                "--scale-units", "arcsecperpix", "--scale-low", str(scale_low), "--scale-high", str(scale_high),
            ]),
            ("Hinted solve (relaxed)", [
                "--no-plots", "--downsample", "2", "--sigma", "2", "--objs", "1000", "--cpulimit", "180",
                "--ra", str(ra_f), "--dec", str(dec_f), "--radius", str(max(radius_deg, 5.0)),
                "--scale-units", "arcsecperpix",
                "--scale-low", str(max(0.05, scale_low * 0.5)),
                "--scale-high", str(min(10.0, scale_high * 2.0)),
            ]),
        ])
    passes.append(
        (
            "Blind solve fallback",
            [
                "--no-plots",
                "--downsample",
                "2",
                "--sigma",
                "2",
                "--objs",
                "1000",
                "--cpulimit",
                "400",
            ]
            + scale_blind_args,
        )
    )
    all_stdout = [f"Astrometry setup:\n{setup_log}\n"]
    all_stderr = []
    final_rc = 1
    solved_fits: str | None = None
    for pass_name, args in passes:
        out, err, rc = _solve_with_args(image_path, args, pass_name, out_basename)
        all_stdout.append(out)
        all_stderr.append(err)
        final_rc = rc
        solved_fits = _find_solved_fits(out_basename)
        if solved_fits:
            break

    annotated_candidates = []
    annotated_candidates += glob.glob(str(out_basename_path) + "*.png")
    annotated_candidates += glob.glob(str(out_basename_path) + "*.jpg")
    annotated_candidates += glob.glob(str(out_dir / (out_basename + "-*.*")))
    annotated_candidates += glob.glob(str(out_dir / (image_path.stem + "*annot*.png")))
    annotated_candidates = [p for p in annotated_candidates if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    annotated_candidates = list(dict.fromkeys(annotated_candidates))

    annotated_png = annotated_candidates[0] if annotated_candidates else None

    # If FITS path does not solve, try debayered PNG next to the FITS.
    if not solved_fits and image_path.suffix.lower() in {".fits", ".fit", ".fts"}:
        png_candidate = image_path.with_name(image_path.stem.replace("_RAW8", "") + "_debayered.png")
        if not png_candidate.exists():
            alt = image_path.with_suffix("").with_name(image_path.stem + "_debayered.png")
            png_candidate = alt if alt.exists() else png_candidate
        if png_candidate.exists():
            png_base = out_basename + "-png"
            out, err, rc = _solve_with_args(
                png_candidate,
                [
                    "--no-plots",
                    "--downsample",
                    "2",
                    "--sigma",
                    "2",
                    "--objs",
                    "1000",
                    "--cpulimit",
                    "400",
                ]
                + scale_blind_args,
                "PNG fallback solve",
                png_base,
            )
            all_stdout.append(out)
            all_stderr.append(err)
            final_rc = rc
            solved_fits = _find_solved_fits(png_base)

    # If still not solved, attach an index-coverage diagnostic.
    if not solved_fits:
        try:
            data_dir = Path(setup_info.get("data_dir", "")) if setup_info else None
            if data_dir and data_dir.exists():
                installed = _installed_index_scales(data_dir)
                wanted = set(_scales_recommended_for_fov(fov_f))
                missing = sorted(wanted - installed)
                if missing:
                    ranges = dict((s, (a, b)) for s, a, b in _index_scale_ranges_arcmin())
                    miss_lines = ", ".join(
                        f"42{s:02d} ({ranges[s][0]:g}-{ranges[s][1]:g}')" for s in missing
                    )
                    inst_lines = ", ".join(f"42{s:02d}" for s in sorted(installed)) or "(none)"
                    diag = (
                        "\n=== Index coverage check ===\n"
                        f"FOV {fov_f:.3f}° (~{fov_f*60:.1f}' wide). Want skymarks ~10%-100% of FOV.\n"
                        f"Recommended index families for this FOV: {miss_lines}\n"
                        f"Installed families in {data_dir}: {inst_lines}\n"
                        "Solve-field exited but no `.solved` file was produced. "
                        "This usually means too few quad matches - install the recommended "
                        "index files and retry.\n"
                    )
                    all_stderr.append(diag)
        except Exception:
            pass

    return {
        'success': bool(solved_fits),
        'stdout': "\n".join(all_stdout),
        'stderr': "\n".join(all_stderr),
        'annotated_png': annotated_png,
        'solved_fits': solved_fits,
        'out_basename': str(out_basename_path),
        'setup': setup_info,
        'return_code': final_rc,
    }


def read_solved_fits_center_radec_deg(solved_fits_path):
    """RA/Dec in degrees at the WCS reference (plate center) from a solved FITS."""
    with fits.open(solved_fits_path) as hd:
        w = WCS(hd[0].header)
        ra, dec = w.wcs.crval[0], w.wcs.crval[1]
    return float(ra), float(dec)


def create_annotated_from_fits(solved_fits_path, out_png_path, max_stars=200):
    """Create an annotated PNG from a solved FITS."""
    with fits.open(solved_fits_path) as hd:
        data = hd[0].data
        w = WCS(hd[0].header)
    if data is None:
        raise RuntimeError("Solved FITS has no image data")
    if data.ndim > 2:
        data = data.squeeze()
        if data.ndim > 2:
            data = data[0]

    arr = np.array(data, dtype=np.float32)

    try:
        bkg = sep.Background(arr)
        data_sub = arr - bkg.back()
        thresh = 3.0 * np.std(data_sub)
        sources = sep.extract(data_sub, thresh)
    except Exception:
        sources = np.empty((0,))

    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax <= vmin:
        vmax = arr.max()
        vmin = arr.min()
    scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    img8 = (scaled * 255).astype(np.uint8)
    pil = Image.fromarray(img8)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    if isinstance(sources, np.ndarray) and sources.size:
        if 'flux' in sources.dtype.names:
            idx_sorted = np.argsort(sources['flux'])[::-1]
        else:
            idx_sorted = np.arange(len(sources))
        labeled_names = set()
        star_coords = [
            (
                star,
                SkyCoord(ra=float(star["ra_deg"]) * u.deg, dec=float(star["dec_deg"]) * u.deg, frame="icrs"),
            )
            for star in _BRIGHT_STAR_CATALOG
        ]
        for i_idx, sidx in enumerate(idx_sorted[:max_stars]):
            sx = float(sources['x'][sidx])
            sy = float(sources['y'][sidx])
            r = 4 if i_idx < 15 else 2
            color = (0, 255, 0) if i_idx < 15 else (0, 200, 0)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=color, width=2)

            # Label only brightest detections to avoid unreadable clutter.
            if i_idx >= 40:
                continue
            try:
                best = None
                best_sep_arcmin = None
                source_coord = w.pixel_to_world(sx, sy)
                for star, star_coord in star_coords:
                    sep_arcmin = source_coord.separation(star_coord).arcminute
                    if best_sep_arcmin is None or sep_arcmin < best_sep_arcmin:
                        best = star
                        best_sep_arcmin = sep_arcmin
            except Exception:
                continue

            # Keep matching conservative to reduce false labels.
            if not best or best_sep_arcmin is None or best_sep_arcmin > 8.0:
                continue
            star_name = best.get("name")
            if not star_name or star_name in labeled_names:
                continue
            labeled_names.add(star_name)
            distance_ly = best.get("distance_ly")
            if distance_ly:
                label = f"{star_name} ({distance_ly:g} ly)"
            else:
                label = star_name
            tx = sx + 6
            ty = sy - 10
            draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label, fill=(255, 255, 120), font=font)

    pil.save(out_png_path)
    return out_png_path


from scipy import ndimage

class StarDetector:
    """Light wrapper for local detection."""
    def __init__(self, threshold_sigma=3.0, min_area=5):
        self.threshold_sigma = threshold_sigma
        self.min_area = min_area

    def detect_stars(self, image_data):
        arr = np.array(image_data, dtype=float)
        background = ndimage.median_filter(arr, size=20)
        img_sub = arr - background
        sigma = np.std(img_sub)
        thresh = self.threshold_sigma * sigma
        binary = img_sub > thresh
        labeled, num = ndimage.label(binary)
        stars = []
        for i in range(1, num+1):
            mask = labeled == i
            area = np.sum(mask)
            if area < self.min_area: continue
            coords = np.argwhere(mask)
            y, x = coords.mean(axis=0)
            brightness = np.sum(img_sub[mask])
            stars.append((x, y, brightness))
        stars.sort(key=lambda s: s[2], reverse=True)
        return stars[:100]


def default_sessions_root() -> Path:
    return Path(__file__).resolve().parent / "sessions"


def create_session_folder(sessions_root: Path | None = None) -> Path:
    """Create and return a new session directory named with local date and time."""
    root = sessions_root or default_sessions_root()
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    n = 0
    while True:
        name = base if n == 0 else f"{base}_{n}"
        path = root / name
        try:
            path.mkdir(parents=False, exist_ok=False)
            return path
        except FileExistsError:
            n += 1


def unique_dest_in_dir(dest_dir: Path, filename: str) -> Path:
    """Return dest_dir / filename, or stem_2.ext, stem_3.ext, ... if needed."""
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        cand = dest_dir / f"{stem}_{n}{suffix}"
        if not cand.exists():
            return cand
        n += 1


class StarFinderGUI:
    def __init__(self, root, session_dir: Path | None = None):
        self.root = root
        self.session_dir = Path(session_dir).resolve() if session_dir else None
        title = "Eosnyx Sky Tracker"
        if self.session_dir:
            title += f" — {self.session_dir.name}"
        self.root.title(title)
        self._ui_compact = ui_compact(root)
        if self._ui_compact:
            self.root.geometry("1008x600")
            self.root.minsize(720, 480)
        else:
            self.root.geometry("1200x800")

        self.star_detector = StarDetector()
        self.current_image_path = None
        self.current_image = None
        self.detected_stars = []
        self.solving = False
        self.solved_fits_path = None
        self.annotated_png_path = None
        
        self.night_mode = False
        self.theme = THEMES["day"]
        
        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        # Top control bar
        control_bar = tk.Frame(self.root)
        _pad = (6, 4) if self._ui_compact else (12, 8)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=_pad[0], pady=_pad[1])
        
        title_label = tk.Label(control_bar, text="SKY FINDER", font=("Segoe UI", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=6)

        session_txt = (
            f"Session: {self.session_dir.name}"
            if self.session_dir
            else "No session (saves next to image)"
        )
        self.session_info_label = tk.Label(
            control_bar, text=session_txt, font=("Segoe UI", 9), fg=self.theme["fg_secondary"]
        )
        self.session_info_label.pack(side=tk.LEFT, padx=12)
        
        spacer = tk.Frame(control_bar)
        spacer.pack(side=tk.LEFT, expand=True)
        
        self.night_mode_btn = tk.Button(control_bar, text="☀ Day Mode", command=self.toggle_night_mode, 
                                        font=("Segoe UI", 9), padx=12, pady=4)
        self.night_mode_btn.pack(side=tk.RIGHT, padx=6)

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=_pad[0], pady=_pad[1])

        # Left: image preview
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        preview_label = tk.Label(left_frame, text="IMAGE PREVIEW", font=("Segoe UI", 11, "bold"))
        preview_label.pack(pady=(0, 8))

        _cv = 380 if self._ui_compact else 550
        self.image_canvas = tk.Canvas(left_frame, width=_cv, height=_cv, highlightthickness=1)
        self.image_canvas.pack(padx=0, pady=0)
        
        self.image_label = tk.Label(self.image_canvas, text="No image loaded\nClick 'Load Image'",
                                    font=("Segoe UI", 11), justify=tk.CENTER)
        self.image_canvas.create_window(_cv // 2, _cv // 2, window=self.image_label)

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=10, fill=tk.X)

        self.load_btn = tk.Button(btn_frame, text="Load Image", command=self.choose_image,
                                  font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        self.load_btn.pack(side=tk.LEFT, padx=3)

        self.detect_local_btn = tk.Button(btn_frame, text="Detect Stars", command=self.detect_stars_only,
                                          font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        self.detect_local_btn.pack(side=tk.LEFT, padx=3)

        # Right: controls & results
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Settings section
        settings_label = tk.Label(right_frame, text="PLATE SOLVE HINTS", font=("Segoe UI", 11, "bold"))
        settings_label.pack(pady=(0, 8))

        settings_inner = tk.Frame(right_frame)
        settings_inner.pack(padx=0, pady=0, fill=tk.X)

        # RA
        ra_frame = tk.Frame(settings_inner)
        ra_frame.pack(fill=tk.X, pady=5)
        tk.Label(ra_frame, text="RA (°):", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.ra_entry = tk.Entry(ra_frame, width=16, font=("Segoe UI", 10))
        self.ra_entry.pack(side=tk.RIGHT, padx=0)
        self.ra_entry.insert(0, "0.0")

        # Dec
        dec_frame = tk.Frame(settings_inner)
        dec_frame.pack(fill=tk.X, pady=5)
        tk.Label(dec_frame, text="Dec (°):", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.dec_entry = tk.Entry(dec_frame, width=16, font=("Segoe UI", 10))
        self.dec_entry.pack(side=tk.RIGHT, padx=0)
        self.dec_entry.insert(0, "0.0")

        # FOV
        fov_frame = tk.Frame(settings_inner)
        fov_frame.pack(fill=tk.X, pady=5)
        tk.Label(fov_frame, text="FOV (°):", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.fov_entry = tk.Entry(fov_frame, width=16, font=("Segoe UI", 10))
        self.fov_entry.pack(side=tk.RIGHT, padx=0)
        self.fov_entry.insert(0, "2.0")

        info_label = tk.Label(right_frame, text="Enter RA, Dec, FOV as hints for the solver.",
                              font=("Segoe UI", 8, "italic"), justify=tk.LEFT)
        info_label.pack(pady=(6, 12), anchor="w")

        # Solve button
        self.solve_btn = tk.Button(right_frame, text="SOLVE", command=self.start_solving,
                                   font=("Segoe UI", 12, "bold"), padx=16, pady=10)
        self.solve_btn.pack(pady=12, fill=tk.X)

        # Progress
        self.progress_label = tk.Label(right_frame, text="", font=("Segoe UI", 9))
        self.progress_label.pack()
        self.progress = ttk.Progressbar(right_frame, mode="indeterminate", length=350)

        # Results
        results_label = tk.Label(right_frame, text="OUTPUT", font=("Segoe UI", 11, "bold"))
        results_label.pack(pady=(12, 6))

        results_inner = tk.Frame(right_frame)
        results_inner.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_inner, height=14, width=45, font=("Courier New", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.results_text.insert("1.0", "Ready. Load an image and enter hints.")
        self.results_text.config(state=tk.DISABLED)

        # Bottom buttons
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(pady=8, fill=tk.X)
        
        self.view_annot_btn = tk.Button(bottom_frame, text="View Annotated", command=self.view_annotated, 
                                        state=tk.DISABLED, font=("Segoe UI", 9), padx=10, pady=4)
        self.view_annot_btn.pack(side=tk.LEFT, padx=2)
        
        self.view_solved_btn = tk.Button(bottom_frame, text="View Solved", command=self.view_solved, 
                                         state=tk.DISABLED, font=("Segoe UI", 9), padx=10, pady=4)
        self.view_solved_btn.pack(side=tk.LEFT, padx=2)

    def apply_theme(self):
        """Apply color theme to all widgets."""
        self.root.configure(bg=self.theme["bg_primary"])
        
        for widget in self.root.winfo_children():
            self._apply_theme_recursive(widget)

    def _apply_theme_recursive(self, widget):
        """Recursively apply theme to widget and children."""
        try:
            if isinstance(widget, tk.Label):
                widget.configure(bg=self.theme["bg_primary"], fg=self.theme["fg_primary"])
            elif isinstance(widget, tk.Button):
                widget.configure(bg=self.theme["accent"], fg="white" if not self.night_mode else self.theme["fg_primary"],
                               activebackground=self.theme["accent_light"], relief=tk.FLAT, bd=0)
            elif isinstance(widget, tk.Entry):
                widget.configure(bg=self.theme["bg_secondary"], fg=self.theme["fg_primary"],
                               insertbackground=self.theme["fg_primary"], relief=tk.FLAT, bd=1, 
                               highlightcolor=self.theme["accent"], highlightbackground=self.theme["border"])
            elif isinstance(widget, tk.Frame):
                widget.configure(bg=self.theme["bg_primary"])
            elif isinstance(widget, tk.Canvas):
                widget.configure(bg=self.theme["canvas_bg"], highlightbackground=self.theme["border"])
            elif isinstance(widget, tk.Text):
                widget.configure(bg=self.theme["bg_secondary"], fg=self.theme["fg_primary"],
                               insertbackground=self.theme["fg_primary"], relief=tk.FLAT, bd=1)
        except:
            pass
        
        for child in widget.winfo_children():
            self._apply_theme_recursive(child)

    def toggle_night_mode(self):
        """Toggle between day and night mode."""
        self.night_mode = not self.night_mode
        self.theme = THEMES["night"] if self.night_mode else THEMES["day"]
        self.night_mode_btn.configure(text="🌙 Night Mode" if self.night_mode else "☀ Day Mode")
        self.apply_theme()

    def choose_image(self):
        filetypes = (("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.fits *.fit"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title="Load image", filetypes=filetypes)
        if filename:
            src = Path(filename).resolve()
            load_path = str(src)
            if self.session_dir:
                in_session = src.parent.resolve() == self.session_dir.resolve()
                if not in_session:
                    try:
                        dest = unique_dest_in_dir(self.session_dir, src.name)
                        shutil.copy2(src, dest)
                        load_path = str(dest)
                    except OSError as e:
                        showerror("Session", f"Could not copy image into session folder:\n{e}")
                        return
            self.current_image_path = load_path
            self.detected_stars = []
            self.solved_fits_path = None
            self.annotated_png_path = None
            self.display_image(load_path)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete("1.0", tk.END)
            note = ""
            if self.session_dir and Path(load_path).resolve() != src:
                note = f"\n(copied into session from {src.name})"
            self.results_text.insert("1.0", f"✓ Loaded: {Path(load_path).name}{note}\n\nDetect stars or enter hints and solve.")
            self.results_text.config(state=tk.DISABLED)
            self.view_annot_btn.config(state=tk.DISABLED)
            self.view_solved_btn.config(state=tk.DISABLED)

    def display_image(self, image_path, stars=None):
        try:
            img = Image.open(image_path)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            self.current_image = img.copy()
            self._display_pil_image(img, stars=stars)
        except Exception as e:
            self.image_label.config(text="Error loading image")

    def _display_pil_image(self, img, stars=None):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        display_img = img.copy()

        if stars:
            draw = ImageDraw.Draw(display_img)
            for i, (x, y, b) in enumerate(stars[:200]):
                r = 4 if i < 12 else 2
                color = (0, 255, 0) if i < 12 else (0, 200, 0)
                draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)

        imgw, imgh = display_img.size
        canvas_size = 550
        scale = min(canvas_size / imgw, canvas_size / imgh)
        neww, newh = max(1, int(imgw * scale)), max(1, int(imgh * scale))
        display_img = display_img.resize((neww, newh), Image.Resampling.LANCZOS)
        
        canvas_img = Image.new("RGB", (canvas_size, canvas_size), 
                              tuple(int(self.theme["canvas_bg"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        offx = (canvas_size - neww) // 2
        offy = (canvas_size - newh) // 2
        canvas_img.paste(display_img, (offx, offy))

        photo = ImageTk.PhotoImage(canvas_img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def detect_stars_only(self):
        if not self.current_image_path:
            showwarning("No image", "Load an image first.")
            return
        self.progress_label.config(text="Detecting stars...")
        self.detect_local_btn.config(state=tk.DISABLED)

        def do_detect():
            try:
                img = Image.open(self.current_image_path)
                if img.mode != 'L':
                    img = img.convert('L')
                arr = np.array(img)
                try:
                    bkg = sep.Background(arr.astype(np.float32))
                    data_sub = arr.astype(np.float32) - bkg.back()
                    stars = sep.extract(data_sub, 3.0)
                    star_list = [(float(s['x']), float(s['y']), float(s.get('flux', 0.0))) for s in stars]
                except Exception:
                    sd = StarDetector()
                    star_list = sd.detect_stars(arr)
                self.detected_stars = star_list
                self.root.after(0, self._show_detected_stars, star_list)
            except Exception as e:
                self.root.after(0, lambda err=str(e): showerror("Error", err, parent=self.root))
                self.detect_local_btn.config(state=tk.NORMAL)
                self.progress_label.config(text="")

        t = threading.Thread(target=do_detect, daemon=True)
        t.start()

    def _show_detected_stars(self, stars):
        self.detect_local_btn.config(state=tk.NORMAL)
        self.progress_label.config(text=f"Found {len(stars)} stars")
        if self.current_image:
            self._display_pil_image(self.current_image, stars=stars)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        s = f"DETECTION RESULTS\nFound {len(stars)} stars\n\nTop positions:\n"
        for i, (x, y, b) in enumerate(stars[:10], 1):
            s += f"{i}. x={x:.0f} y={y:.0f}\n"
        self.results_text.insert("1.0", s)
        self.results_text.config(state=tk.DISABLED)

    def start_solving(self):
        if not self.current_image_path:
            showwarning("No image", "Load an image first.")
            return
        if self.solving:
            return
        try:
            ra = float(self.ra_entry.get().strip())
            dec = float(self.dec_entry.get().strip())
            fov = float(self.fov_entry.get().strip())
        except Exception:
            showerror("Invalid input", "RA, Dec, FOV must be numeric.")
            return

        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Solving...")
        self.progress.pack(pady=6)
        self.progress.start(10)
        thread = threading.Thread(target=self._solve_thread, args=(ra, dec, fov), daemon=True)
        thread.start()

    def _solve_thread(self, ra, dec, fov):
        try:
            image_path = Path(self.current_image_path)
            base = image_path.stem
            out_basename = base + "-solved"
            self._append_result(f"Starting solve-field...\n")
            res = run_solve_field_local(str(image_path), ra, dec, fov, out_basename)
            self._append_result("--- Output ---\n" + (res['stdout'][-500:] if res['stdout'] else "(no output)") + "\n")
            
            if not res['success']:
                self.root.after(0, self._solve_failed, res)
                return

            solved_fits = res.get('solved_fits')
            annotated = res.get('annotated_png')

            if (not annotated) and solved_fits:
                try:
                    generated_png = str(Path(solved_fits).with_suffix(".annotated.png"))
                    self._append_result("Generating annotated image...\n")
                    create_annotated_from_fits(solved_fits, generated_png)
                    annotated = generated_png
                except Exception as e:
                    self._append_result(f"Could not generate: {str(e)}\n")

            self.solved_fits_path = solved_fits
            self.annotated_png_path = annotated
            self.root.after(0, self._solve_succeeded, res)

        except Exception as e:
            self._append_result(f"Exception: {str(e)}")
            self.root.after(0, self._solve_failed, {'error': str(e)})
        finally:
            self.solving = False
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.progress.pack_forget())
            self.root.after(0, lambda: self.solve_btn.config(state=tk.NORMAL))

    def _append_result(self, txt):
        self.root.after(0, lambda: self._append_result_mainthread(txt))

    def _append_result_mainthread(self, txt):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, txt + "\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    def _solve_failed(self, res):
        self.progress_label.config(text="✗ Failed")
        self._append_result("Solve failed.")
        showerror("Error", "Solving failed. Check output above.")

    def _solve_succeeded(self, res):
        self.progress_label.config(text="✓ Success")
        self._append_result("Solve complete.")
        if self.annotated_png_path and Path(self.annotated_png_path).exists():
            self.view_annot_btn.config(state=tk.NORMAL)
        if self.solved_fits_path and Path(self.solved_fits_path).exists():
            self.view_solved_btn.config(state=tk.NORMAL)

    def view_annotated(self):
        if not self.annotated_png_path or not Path(self.annotated_png_path).exists():
            showwarning("Not available", "No annotated image.")
            return
        self.display_image(self.annotated_png_path)

    def view_solved(self):
        if not self.solved_fits_path or not Path(self.solved_fits_path).exists():
            showwarning("Not available", "No solved FITS.")
            return
        out_png = str(Path(self.solved_fits_path).with_suffix(".solved.png"))
        try:
            hd = fits.open(self.solved_fits_path)
            arr = hd[0].data
            if arr is None:
                raise RuntimeError("No data in FITS")
            if arr.ndim > 2:
                arr = arr.squeeze()
                if arr.ndim > 2:
                    arr = arr[0]
            vmin, vmax = np.percentile(arr, [2, 99.5])
            if vmax <= vmin:
                vmax = arr.max()
                vmin = arr.min()
            scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
            img8 = (scaled * 255).astype(np.uint8)
            pil = Image.fromarray(img8)
            pil.save(out_png)
            self.display_image(out_png)
        except Exception as e:
            showerror("Error", f"Could not render: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    session_path = None
    try:
        if askyesno(
            "Session",
            "Start a new observing session?\n\n"
            "Yes: create a folder (date and time) under the app's sessions/ directory.\n"
            "Loaded images are copied there; plate-solve outputs stay in the same folder.\n\n"
            "No: work without a session (solver writes next to the image you load).",
            parent=root,
        ):
            session_path = create_session_folder()
    except OSError as e:
        showerror("Session", f"Could not create session folder:\n{e}", parent=root)
    root.deiconify()
    app = StarFinderGUI(root, session_dir=session_path)
    root.mainloop()
