#!/usr/bin/env python3
"""Offline deep-sky image stacking pipeline for ASTRA.

This module is intentionally GUI-free: ``CamGUI`` constructs a :class:`StackConfig`
and calls :func:`run_stack_job` from a worker thread. All work is done locally
with numpy / OpenCV / astropy / sep / astroalign — no network calls of any kind.

Pipeline per light frame:

1. Read FITS as float32.
2. Subtract master bias, master dark; divide by normalized master flat.
3. Debayer (cv2) when the frame is single-channel and a Bayer pattern is known.
4. Align to a reference frame using ``astroalign.register`` (with a sep-centroid
   fallback for translation-only alignment when astroalign cannot triangulate).
5. Combine all aligned frames with mean, median, or sigma-clipped mean.
6. Stretch the result for an 8-bit preview PNG; also write a 32-bit FITS.

The pipeline is deliberately tolerant: a failure on any single frame is logged
and the frame is skipped, rather than aborting the whole job.
"""
from __future__ import annotations

import os
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - opencv is in requirements
    cv2 = None  # type: ignore

from astropy.io import fits

# astroalign is loaded lazily so that ``import deep_sky_stacker`` never breaks
# CamGUI startup when the package is missing. The GUI surfaces a clear error
# when the user tries to stack without it installed.
_astroalign_mod = None
_astroalign_err: Optional[str] = None


def _load_astroalign():
    global _astroalign_mod, _astroalign_err
    if _astroalign_mod is not None:
        return _astroalign_mod
    if _astroalign_err is not None:
        return None
    try:
        import astroalign as aa  # type: ignore
        _astroalign_mod = aa
        return aa
    except Exception as e:
        _astroalign_err = f"astroalign unavailable: {e}"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────────────


COMBINE_METHODS = ("mean", "median", "sigma_clip")
STRETCH_MODES = ("linear", "asinh", "zscale", "percentile")
BAYER_PATTERNS = ("auto", "RGGB", "GRBG", "BGGR", "GBRG", "mono")


@dataclass
class StackConfig:
    light_paths: list[str]
    dark_paths: list[str] = field(default_factory=list)
    flat_paths: list[str] = field(default_factory=list)
    bias_paths: list[str] = field(default_factory=list)

    output_dir: str = ""
    output_basename: str = "stack"

    combine: str = "sigma_clip"
    sigma_low: float = 3.0
    sigma_high: float = 3.0

    reference_index: Optional[int] = None  # None -> auto (most stars)
    stretch: str = "asinh"
    bayer_pattern: str = "auto"

    # Optional cancellation hook. The GUI sets this and the worker checks it
    # between frames so a long stack can be aborted cleanly.
    cancel_event: Optional[threading.Event] = None


@dataclass
class StackResult:
    success: bool
    message: str = ""
    fits_path: str = ""
    preview_path: str = ""
    n_used: int = 0
    n_skipped: int = 0
    total_exposure_s: float = 0.0
    width: int = 0
    height: int = 0
    channels: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# IO + calibration helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_fits_as_float(path: str) -> tuple[np.ndarray, fits.Header]:
    """Read a FITS file as float32. Squeezes singleton dimensions."""
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    if data.ndim > 2:
        data = np.squeeze(data)
        if data.ndim > 2:
            data = data[0]
    return data.astype(np.float32, copy=False), header


def make_master(paths: list[str], method: str = "median") -> Optional[np.ndarray]:
    """Combine calibration frames into a single master frame (float32)."""
    if not paths:
        return None
    frames = []
    ref_shape = None
    for p in paths:
        try:
            arr, _ = load_fits_as_float(p)
        except Exception:
            continue
        if ref_shape is None:
            ref_shape = arr.shape
        if arr.shape != ref_shape:
            continue
        frames.append(arr)
    if not frames:
        return None
    stack = np.stack(frames, axis=0)
    if method == "mean":
        return np.mean(stack, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.float32)


def calibrate(
    light: np.ndarray,
    master_bias: Optional[np.ndarray],
    master_dark: Optional[np.ndarray],
    master_flat: Optional[np.ndarray],
) -> np.ndarray:
    """Apply standard bias/dark/flat correction to a light frame.

    Calibration is done in raw (pre-debayer) space when a Bayer mosaic is
    present, since the calibration frames carry the same Bayer pattern.
    """
    out = light.astype(np.float32, copy=True)
    if master_bias is not None and master_bias.shape == out.shape:
        out -= master_bias
    if master_dark is not None and master_dark.shape == out.shape:
        out -= master_dark
    if master_flat is not None and master_flat.shape == out.shape:
        flat = master_flat.astype(np.float32, copy=True)
        if master_bias is not None and master_bias.shape == flat.shape:
            flat = flat - master_bias
        med = float(np.nanmedian(flat))
        if med > 1e-6:
            flat = flat / med
        flat = np.where(flat < 1e-3, 1.0, flat)
        out = out / flat
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Debayer
# ─────────────────────────────────────────────────────────────────────────────


_BAYER_TO_CV = {
    # OpenCV's bayer code names come from the second-row pixel pair.
    "RGGB": "COLOR_BayerRG2RGB",
    "GRBG": "COLOR_BayerGR2RGB",
    "BGGR": "COLOR_BayerBG2RGB",
    "GBRG": "COLOR_BayerGB2RGB",
}


def debayer_if_needed(
    frame: np.ndarray,
    header: fits.Header,
    pattern_pref: str = "auto",
) -> tuple[np.ndarray, str]:
    """Return (RGB or mono float32 frame, label).

    - If the frame is already 3-channel, returns it untouched.
    - If ``pattern_pref`` is "mono", skips debayering.
    - Otherwise picks the Bayer pattern from the header (``BAYERPAT``) or from
      ``pattern_pref`` and uses OpenCV to interpolate to RGB.
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.astype(np.float32, copy=False), "rgb"
    if pattern_pref == "mono":
        return frame.astype(np.float32, copy=False), "mono"
    if cv2 is None:
        return frame.astype(np.float32, copy=False), "mono"

    pat = pattern_pref
    if pat == "auto":
        pat = str(header.get("BAYERPAT", "")).strip().upper()
        if pat not in _BAYER_TO_CV:
            pat = "RGGB" if "BAYERPAT" not in header else pat
            if pat not in _BAYER_TO_CV:
                return frame.astype(np.float32, copy=False), "mono"

    # OpenCV bayer demosaic needs uint8 or uint16 input.
    f = frame
    if f.dtype not in (np.uint8, np.uint16):
        fmin = float(np.nanmin(f))
        fmax = float(np.nanmax(f))
        if fmax <= fmin:
            return f.astype(np.float32, copy=False), "mono"
        scaled = (f - fmin) / (fmax - fmin)
        f16 = (np.clip(scaled, 0.0, 1.0) * 65535.0).astype(np.uint16)
        rgb = cv2.cvtColor(f16, getattr(cv2, _BAYER_TO_CV[pat]))
        rgb_f = rgb.astype(np.float32) / 65535.0 * (fmax - fmin) + fmin
        return rgb_f.astype(np.float32, copy=False), pat
    rgb = cv2.cvtColor(f, getattr(cv2, _BAYER_TO_CV[pat]))
    return rgb.astype(np.float32, copy=False), pat


def _to_luminance(frame: np.ndarray) -> np.ndarray:
    """Single-channel float32 used for star detection / alignment."""
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Rec. 601 luma — matches what astroalign sees in typical RGB inputs.
        r = frame[:, :, 0]
        g = frame[:, :, 1]
        b = frame[:, :, 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    return np.squeeze(frame).astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Reference picking + alignment
# ─────────────────────────────────────────────────────────────────────────────


def _star_count(frame_lum: np.ndarray) -> int:
    """Estimate the number of detectable stars via SEP (offline)."""
    try:
        import sep  # type: ignore
    except Exception:
        return 0
    try:
        data = np.ascontiguousarray(frame_lum, dtype=np.float32)
        bkg = sep.Background(data)
        thresh = max(2.5, 3.0 * float(bkg.globalrms))
        sources = sep.extract(data - bkg, thresh)
        return int(0 if sources is None else len(sources))
    except Exception:
        return 0


def pick_reference(luminances: list[np.ndarray]) -> int:
    """Return the index of the frame with the most detected stars."""
    best_idx = 0
    best_n = -1
    for i, lum in enumerate(luminances):
        n = _star_count(lum)
        if n > best_n:
            best_n = n
            best_idx = i
    return best_idx


def _sep_centroid_align(
    frame: np.ndarray, frame_lum: np.ndarray, ref_lum: np.ndarray
) -> Optional[np.ndarray]:
    """Translation-only fallback: shift frame to align brightest centroid.

    Used when astroalign cannot find a triangle match. Not as robust as
    astroalign for rotation, but better than dropping the frame entirely.
    """
    try:
        import sep  # type: ignore
    except Exception:
        return None
    if cv2 is None:
        return None
    try:
        a = np.ascontiguousarray(frame_lum, dtype=np.float32)
        b = np.ascontiguousarray(ref_lum, dtype=np.float32)
        ba = sep.Background(a); bb = sep.Background(b)
        sa = sep.extract(a - ba, max(2.5, 3.0 * float(ba.globalrms)))
        sb = sep.extract(b - bb, max(2.5, 3.0 * float(bb.globalrms)))
        if sa is None or sb is None or len(sa) == 0 or len(sb) == 0:
            return None
        ax = float(sorted(sa, key=lambda s: -float(s["flux"]))[0]["x"])
        ay = float(sorted(sa, key=lambda s: -float(s["flux"]))[0]["y"])
        bx = float(sorted(sb, key=lambda s: -float(s["flux"]))[0]["x"])
        by = float(sorted(sb, key=lambda s: -float(s["flux"]))[0]["y"])
        dx = bx - ax
        dy = by - ay
        M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        if frame.ndim == 2:
            warped = cv2.warpAffine(
                frame, M, (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        else:
            channels = []
            for c in range(frame.shape[2]):
                channels.append(
                    cv2.warpAffine(
                        frame[:, :, c], M, (frame.shape[1], frame.shape[0]),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    )
                )
            warped = np.stack(channels, axis=-1)
        return warped.astype(np.float32, copy=False)
    except Exception:
        return None


def align_to_reference(
    frame: np.ndarray, ref: np.ndarray
) -> tuple[Optional[np.ndarray], str]:
    """Align ``frame`` to ``ref``. Returns (aligned, info string)."""
    aa = _load_astroalign()
    frame_lum = _to_luminance(frame)
    ref_lum = _to_luminance(ref)

    if aa is not None:
        try:
            if frame.ndim == 2:
                aligned, _ = aa.register(frame, ref_lum)
                return aligned.astype(np.float32, copy=False), "astroalign"
            transform, _ = aa.find_transform(frame_lum, ref_lum)
            channels = []
            h, w = ref_lum.shape
            for c in range(frame.shape[2]):
                ch_aligned, _ = aa.apply_transform(
                    transform, frame[:, :, c], ref_lum
                )
                if ch_aligned.shape != (h, w):
                    pad = np.zeros((h, w), dtype=np.float32)
                    hh = min(h, ch_aligned.shape[0])
                    ww = min(w, ch_aligned.shape[1])
                    pad[:hh, :ww] = ch_aligned[:hh, :ww]
                    ch_aligned = pad
                channels.append(ch_aligned)
            aligned = np.stack(channels, axis=-1)
            return aligned.astype(np.float32, copy=False), "astroalign"
        except Exception as e:
            err = f"astroalign failed ({type(e).__name__}); trying centroid fallback"
            fallback = _sep_centroid_align(frame, frame_lum, ref_lum)
            if fallback is not None:
                return fallback, err + " — translation only"
            return None, err

    fallback = _sep_centroid_align(frame, frame_lum, ref_lum)
    if fallback is not None:
        return fallback, "centroid fallback (astroalign not installed)"
    return None, _astroalign_err or "no alignment available"


# ─────────────────────────────────────────────────────────────────────────────
# Combine + stretch
# ─────────────────────────────────────────────────────────────────────────────


def combine(
    stack_4d: np.ndarray,
    method: str = "sigma_clip",
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> np.ndarray:
    """Combine an (N, H, W) or (N, H, W, C) stack along the first axis."""
    if stack_4d.shape[0] == 1:
        return stack_4d[0].astype(np.float32, copy=False)
    if method == "mean":
        return np.mean(stack_4d, axis=0).astype(np.float32)
    if method == "median":
        return np.median(stack_4d, axis=0).astype(np.float32)
    # sigma clip — mask outliers along axis 0 then take the mean.
    arr = stack_4d.astype(np.float32, copy=False)
    med = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=0, keepdims=True) * 1.4826
    mad = np.where(mad < 1e-6, 1e-6, mad)
    z = (arr - med) / mad
    mask = (z >= -float(sigma_low)) & (z <= float(sigma_high))
    masked = np.where(mask, arr, np.nan)
    out = np.nanmean(masked, axis=0)
    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        med_full = np.median(arr, axis=0)
        out = np.where(nan_mask, med_full, out)
    return out.astype(np.float32, copy=False)


def stretch(img: np.ndarray, mode: str = "asinh") -> np.ndarray:
    """Map a float frame to uint8 for a preview PNG, with a chosen stretch."""
    a = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape if a.ndim <= 3 else a.shape[:3], dtype=np.uint8)

    if a.ndim == 3 and a.shape[2] == 3:
        out = np.zeros_like(a, dtype=np.uint8)
        for c in range(3):
            out[:, :, c] = stretch(a[:, :, c], mode=mode)
        return out

    vals = a[finite]
    if mode == "linear":
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    elif mode == "percentile":
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
    elif mode == "zscale":
        try:
            from astropy.visualization import ZScaleInterval  # type: ignore
            lo, hi = ZScaleInterval().get_limits(vals)
            lo = float(lo); hi = float(hi)
        except Exception:
            lo = float(np.percentile(vals, 1.0))
            hi = float(np.percentile(vals, 99.5))
    else:  # "asinh"
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.7))
        if hi > lo:
            scaled = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
            # Asinh with a soft knee — emphasizes faint nebulosity.
            asinh = np.arcsinh(scaled * 10.0) / np.arcsinh(10.0)
            return (asinh * 255.0).astype(np.uint8)
        lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals))

    if hi <= lo:
        lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals))
        if hi <= lo:
            return np.zeros(a.shape, dtype=np.uint8)
    scaled = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────


def write_outputs(
    cfg: StackConfig,
    stacked: np.ndarray,
    preview_u8: np.ndarray,
    n_used: int,
    n_skipped: int,
    total_exposure_s: float,
    combine_method: str,
    stretch_mode: str,
    reference_path: str,
) -> tuple[str, str]:
    """Write stacked FITS + preview PNG, return (fits_path, png_path)."""
    out_dir = Path(cfg.output_dir or os.getcwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    base = cfg.output_basename or "stack"

    fits_path = str(out_dir / f"{base}.fits")
    png_path = str(out_dir / f"{base}_preview.png")

    header = fits.Header()
    header["STACKER"] = ("ASTRA", "Stacked by ASTRA deep_sky_stacker")
    header["NSTACK"] = (int(n_used), "Number of frames stacked")
    header["NSKIP"] = (int(n_skipped), "Number of frames skipped")
    header["EXPTOT"] = (float(total_exposure_s), "Total integration time (s)")
    header["COMBMETH"] = (str(combine_method), "Combine method")
    header["STRETCH"] = (str(stretch_mode), "Preview stretch mode")
    header["REFFRAME"] = (os.path.basename(reference_path), "Alignment reference frame")
    header["DATE"] = (datetime.utcnow().isoformat(), "Stacking timestamp (UTC)")
    if stacked.ndim == 3 and stacked.shape[2] == 3:
        # FITS convention is channel-first; transpose so HDU has (3, H, W).
        data = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
        header["NAXIS3"] = (3, "Color planes (R, G, B)")
    else:
        data = stacked.astype(np.float32)
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path, overwrite=True)

    if cv2 is not None:
        if preview_u8.ndim == 3 and preview_u8.shape[2] == 3:
            cv2.imwrite(png_path, cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(png_path, preview_u8)
    else:
        # Fallback via PIL when OpenCV is missing — pillow is in requirements.
        from PIL import Image  # type: ignore
        if preview_u8.ndim == 3 and preview_u8.shape[2] == 3:
            Image.fromarray(preview_u8, mode="RGB").save(png_path)
        else:
            Image.fromarray(preview_u8, mode="L").save(png_path)

    return fits_path, png_path


# ─────────────────────────────────────────────────────────────────────────────
# Top-level job
# ─────────────────────────────────────────────────────────────────────────────


_LogCb = Optional[Callable[[str], None]]
_ProgCb = Optional[Callable[[int, int, str], None]]


def _log(log_cb: _LogCb, msg: str) -> None:
    if log_cb is not None:
        try:
            log_cb(msg)
        except Exception:
            pass
    else:
        print(msg)


def _progress(prog_cb: _ProgCb, done: int, total: int, label: str) -> None:
    if prog_cb is not None:
        try:
            prog_cb(done, total, label)
        except Exception:
            pass


def _cancelled(cfg: StackConfig) -> bool:
    return cfg.cancel_event is not None and cfg.cancel_event.is_set()


def run_stack_job(
    cfg: StackConfig,
    progress_cb: _ProgCb = None,
    log_cb: _LogCb = None,
) -> StackResult:
    """Execute the full stacking pipeline. Synchronous; safe to run in a thread."""
    if not cfg.light_paths:
        return StackResult(success=False, message="No light frames selected.")

    if cfg.combine not in COMBINE_METHODS:
        return StackResult(
            success=False, message=f"Unknown combine method: {cfg.combine}"
        )
    if cfg.stretch not in STRETCH_MODES:
        return StackResult(
            success=False, message=f"Unknown stretch mode: {cfg.stretch}"
        )

    aa = _load_astroalign()
    if aa is None:
        _log(
            log_cb,
            f"WARNING: {_astroalign_err or 'astroalign missing'} — "
            "alignment will fall back to centroid-only translation.",
        )

    total_steps = (
        len(cfg.light_paths)
        + (1 if cfg.bias_paths else 0)
        + (1 if cfg.dark_paths else 0)
        + (1 if cfg.flat_paths else 0)
        + 2  # combine + write
    )
    step = 0

    # Master calibration frames -------------------------------------------------
    master_bias = master_dark = master_flat = None
    if cfg.bias_paths:
        step += 1
        _progress(progress_cb, step, total_steps, "Building master bias")
        master_bias = make_master(cfg.bias_paths, method="median")
        _log(log_cb, f"Master bias: {len(cfg.bias_paths)} frames -> "
             f"{'OK' if master_bias is not None else 'FAILED'}")
        if _cancelled(cfg):
            return StackResult(success=False, message="Cancelled.")
    if cfg.dark_paths:
        step += 1
        _progress(progress_cb, step, total_steps, "Building master dark")
        master_dark = make_master(cfg.dark_paths, method="median")
        _log(log_cb, f"Master dark: {len(cfg.dark_paths)} frames -> "
             f"{'OK' if master_dark is not None else 'FAILED'}")
        if _cancelled(cfg):
            return StackResult(success=False, message="Cancelled.")
    if cfg.flat_paths:
        step += 1
        _progress(progress_cb, step, total_steps, "Building master flat")
        master_flat = make_master(cfg.flat_paths, method="median")
        _log(log_cb, f"Master flat: {len(cfg.flat_paths)} frames -> "
             f"{'OK' if master_flat is not None else 'FAILED'}")
        if _cancelled(cfg):
            return StackResult(success=False, message="Cancelled.")

    # First pass: load + calibrate + debayer -----------------------------------
    processed: list[np.ndarray] = []
    luminances: list[np.ndarray] = []
    paths_used: list[str] = []
    total_exposure = 0.0
    skipped = 0

    ref_shape: Optional[tuple[int, ...]] = None
    for i, p in enumerate(cfg.light_paths):
        if _cancelled(cfg):
            return StackResult(success=False, message="Cancelled.")
        step += 1
        _progress(progress_cb, step, total_steps, f"Calibrating {i + 1}/{len(cfg.light_paths)}")
        try:
            raw, hdr = load_fits_as_float(p)
        except Exception as e:
            _log(log_cb, f"  [skip] {os.path.basename(p)}: read failed ({e})")
            skipped += 1
            continue

        try:
            cal = calibrate(raw, master_bias, master_dark, master_flat)
        except Exception as e:
            _log(log_cb, f"  [skip] {os.path.basename(p)}: calibration failed ({e})")
            skipped += 1
            continue

        try:
            color, label = debayer_if_needed(cal, hdr, cfg.bayer_pattern)
        except Exception as e:
            _log(log_cb, f"  [skip] {os.path.basename(p)}: debayer failed ({e})")
            skipped += 1
            continue

        if ref_shape is None:
            ref_shape = color.shape
        elif color.shape != ref_shape:
            _log(
                log_cb,
                f"  [skip] {os.path.basename(p)}: shape {color.shape} != {ref_shape}",
            )
            skipped += 1
            continue

        try:
            exp = float(hdr.get("EXPOSURE", hdr.get("EXPTIME", 0.0)) or 0.0)
            total_exposure += exp
        except Exception:
            pass

        processed.append(color)
        luminances.append(_to_luminance(color))
        paths_used.append(p)
        _log(log_cb, f"  [ok]   {os.path.basename(p)} ({label})")

    if not processed:
        return StackResult(
            success=False, message="No usable light frames after calibration.", n_skipped=skipped
        )

    # Reference selection ------------------------------------------------------
    if cfg.reference_index is not None and 0 <= cfg.reference_index < len(processed):
        ref_idx = cfg.reference_index
        _log(log_cb, f"Reference: index {ref_idx} (user)")
    else:
        ref_idx = pick_reference(luminances)
        _log(log_cb, f"Reference: index {ref_idx} (auto, most stars)")
    ref_frame = processed[ref_idx]
    ref_path = paths_used[ref_idx]

    # Second pass: align + accumulate -----------------------------------------
    aligned: list[np.ndarray] = [ref_frame]
    for i, frame in enumerate(processed):
        if i == ref_idx:
            continue
        if _cancelled(cfg):
            return StackResult(success=False, message="Cancelled.")
        step_in_align = step + i + 1
        _progress(
            progress_cb,
            min(step_in_align, total_steps - 2),
            total_steps,
            f"Aligning {i + 1}/{len(processed)}",
        )
        try:
            warped, info = align_to_reference(frame, ref_frame)
        except Exception as e:
            warped, info = None, f"alignment crashed: {e}"
        if warped is None:
            _log(log_cb, f"  [skip] {os.path.basename(paths_used[i])}: {info}")
            skipped += 1
            continue
        aligned.append(warped)
        _log(log_cb, f"  [aligned] {os.path.basename(paths_used[i])} via {info}")

    if not aligned:
        return StackResult(
            success=False, message="No frames aligned successfully.", n_skipped=skipped
        )

    # Combine ------------------------------------------------------------------
    step = total_steps - 1
    _progress(progress_cb, step, total_steps, f"Combining ({cfg.combine})")
    try:
        stack_arr = np.stack(aligned, axis=0)
        stacked = combine(stack_arr, cfg.combine, cfg.sigma_low, cfg.sigma_high)
    except Exception as e:
        return StackResult(success=False, message=f"Combine failed: {e}")

    # Stretch + write ----------------------------------------------------------
    step = total_steps
    _progress(progress_cb, step, total_steps, "Writing outputs")
    try:
        preview = stretch(stacked, cfg.stretch)
        fits_path, png_path = write_outputs(
            cfg,
            stacked=stacked,
            preview_u8=preview,
            n_used=len(aligned),
            n_skipped=skipped,
            total_exposure_s=total_exposure,
            combine_method=cfg.combine,
            stretch_mode=cfg.stretch,
            reference_path=ref_path,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return StackResult(success=False, message=f"Write failed: {e}\n{tb}")

    h, w = stacked.shape[:2]
    ch = stacked.shape[2] if stacked.ndim == 3 else 1
    return StackResult(
        success=True,
        message=f"Stacked {len(aligned)} of {len(cfg.light_paths)} frames "
                f"({skipped} skipped). Total integration {total_exposure:.1f} s.",
        fits_path=fits_path,
        preview_path=png_path,
        n_used=len(aligned),
        n_skipped=skipped,
        total_exposure_s=total_exposure,
        width=w,
        height=h,
        channels=ch,
    )
