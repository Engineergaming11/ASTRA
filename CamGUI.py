import zwoasi as asi
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageDraw, ImageTk
import threading
import os
import sys
import json
import time
import math
from datetime import datetime, time as dt_time
from pathlib import Path
from urllib.request import urlopen
try:
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    import matplotlib.patches as mpatches  # pyright: ignore[reportMissingImports]
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # pyright: ignore[reportMissingImports]
    from matplotlib.backends.backend_pdf import PdfPages  # pyright: ignore[reportMissingImports]
    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    mpatches = None
    FigureCanvasTkAgg = None
    PdfPages = None
    _HAS_MATPLOTLIB = False

from astropy.io import fits
from astropy.wcs import WCS
import sep

from Formal_PlateSolve_GUI import (
    run_solve_field_local,
    read_solved_fits_center_radec_deg,
    _scales_recommended_for_fov,
    _installed_index_scales,
    _index_scale_ranges_arcmin,
)
from mount_controls_frame import MountControlsFrame
from sky_ephemeris import (
    DEFAULT_ELEV_M,
    DEFAULT_LAT,
    DEFAULT_LON,
    OFFLINE_STAR_FACTS,
    SOLAR_SYSTEM_BODIES,
    TYCHO_PRESET_STARS,
    major_bodies_separation_table,
    polar_align_mvp_text,
    radec_to_altaz_deg,
    sun_apparent_radec_deg,
    tycho_nearest_identification,
)
from ioptron_mount import (
    TRACK_LUNAR,
    TRACK_SIDEREAL,
    TRACK_SOLAR,
    _deg_to_dms,
    _deg_to_hms,
    _parse_dec_input,
    _parse_ra_input,
)
from location_gps import try_read_lat_lon_nmea

import deep_sky_stacker
from xy_scroll_frame import attach_mousewheel_dispatch, create_xy_scroll_area, theme_xy_area


def _default_capture_save_path() -> str:
    """Cross-platform default capture directory (under the current user's home)."""
    return str(Path.home() / "ASICAP" / "CapGUI")


def _is_raspberry_pi() -> bool:
    """True when running on Raspberry Pi hardware (Linux device tree)."""
    try:
        model_path = Path("/proc/device-tree/model")
        if not model_path.is_file():
            return False
        return "raspberry pi" in model_path.read_text(
            encoding="utf-8", errors="ignore"
        ).lower()
    except OSError:
        return False


def _resolved_asi_library_path(app_dir: Path) -> str:
    """Prefer an ASI SDK binary next to this script; extension depends on OS."""
    if sys.platform == "darwin":
        candidates = ("libASICamera2.dylib", "libASICamera2.dylib.1.41")
    elif sys.platform.startswith("linux"):
        candidates = ("libASICamera2.so",)
    elif sys.platform == "win32":
        candidates = ("ASICamera2.dll",)
    else:
        candidates = ("libASICamera2.so",)
    for name in candidates:
        p = app_dir / name
        if p.is_file():
            return str(p)
    return str(app_dir / candidates[0])


def _enumerate_asi_cameras():
    """Return [(index, model name), ...] without constructing ``asi.Camera``.

    A temporary ``Camera`` opens the device and ``__del__`` closes it when the
    object is collected, which invalidates any other open handle to the same id
    (``ZWO_IOError: Camera closed`` on the next SDK call).
    """
    num = asi.get_num_cameras()
    if num <= 0:
        return []
    names = asi.list_cameras()
    return [(i, names[i]) for i in range(num)]


# THEME CONFIGURATION
THEMES = {
    "day": {
        "bg_primary": "#f5f5f5",
        "bg_secondary": "#ebebeb",
        "bg_tertiary": "#e0e0e0",
        "fg_primary": "#1a1a1a",
        "fg_secondary": "#4a4a4a",
        "fg_tertiary": "#7a7a7a",
        "accent": "#0b5c8f",
        "accent_light": "#1a7ab5",
        "canvas_bg": "#ffffff",
        "border": "#cccccc",
    },
    "night": {
        "bg_primary": "#1a0a05",
        "bg_secondary": "#2d1410",
        "bg_tertiary": "#3d1f1a",
        "fg_primary": "#ff6666",
        "fg_secondary": "#ff6666",
        "fg_tertiary": "#ff6666",
        "accent": "#ff4444",
        "accent_light": "#ff6666",
        "canvas_bg": "#0d0503",
        "border": "#4d2020",
    }
}

# Plate-solve tab uses a permanent red palette regardless of day/night mode,
# so observers in the field always get a low-impact, dark-adapted view.
PLATE_RED_PAL = {
    "bg":           "#1a0a05",   # near-black with faint warm tint
    "bg_alt":       "#2d1410",
    "bg_panel":     "#3d1f1a",
    "fg":           "#ff6666",
    "fg_dim":       "#cc5555",
    "fg_muted":     "#7a3a3a",
    "accent":       "#d11515",
    "accent_hover": "#ff4444",
    "accent_dim":   "#7a0c0c",
    "border":       "#4d2020",
    "btn_text":     "#ffe6e6",
}

# Slider track + thumb: use red palette everywhere so CTkSlider never falls back to a light/white track.
_RED_SLIDER = {
    "button_color": PLATE_RED_PAL["accent"],
    "button_hover_color": PLATE_RED_PAL["accent_hover"],
    "progress_color": PLATE_RED_PAL["accent"],
    "fg_color": PLATE_RED_PAL["bg_alt"],
}


def _theme_native_tk_under_camera_column(parent: tk.Misc, bg: str, fg: str) -> None:
    """Recursively set native ``tk.Frame`` / ``tk.Label`` colours under a subtree.

    Pierces CTk containers (e.g. tab views) but skips leaf CustomTkinter controls so
    slider rows and path frames pick up the dark/red panel instead of default grey.
    """
    _leaf_ctk = (
        ctk.CTkComboBox,
        ctk.CTkButton,
        ctk.CTkSlider,
        ctk.CTkEntry,
        ctk.CTkOptionMenu,
        ctk.CTkSwitch,
        ctk.CTkCheckBox,
    )
    try:
        children = parent.winfo_children()
    except tk.TclError:
        return
    for child in children:
        if isinstance(child, tk.Label):
            try:
                child.configure(bg=bg, fg=fg)
            except tk.TclError:
                pass
            continue
        if isinstance(child, tk.Frame):
            if type(child) is tk.Frame:
                try:
                    child.configure(bg=bg)
                except tk.TclError:
                    pass
            _theme_native_tk_under_camera_column(child, bg, fg)
            continue
        if isinstance(child, _leaf_ctk):
            continue
        _theme_native_tk_under_camera_column(child, bg, fg)


# Default equipment preset. The user always images from Tucson, AZ with a
# 2032 mm focal length Schmidt–Cassegrain (Celestron 8" SCT). Camera pixel
# size and sensor dimensions are filled from the connected ZWO camera the
# first time the preset is saved, but sensible fallbacks are provided so the
# preset is usable even if the camera has not been connected yet.
DEFAULT_ASTRO_PRESET = {
    "name": "Tucson SCT 2032mm",
    "site": {
        "label": "Tucson, AZ",
        "lat": 32.2329,
        "lon": -110.9479,
        "elev_m": 728.0,
    },
    "telescope": {
        "label": "8\" Schmidt-Cassegrain",
        "focal_length_mm": 2032.0,
    },
    "camera": {
        # Will be auto-filled from the connected ZWO camera the first time
        # the user clicks "Auto-fill from camera". These defaults match
        # ZWO ASI585MC sensor specs.
        "label": "ZWO ASI585MC",
        "pixel_size_um": 2.9,
        "sensor_w_px": 3840,
        "sensor_h_px": 2160,
    },
}

# Plate-solve equipment tab: quick telescope choices (focal length drives FOV hints).
PLATE_SOLVE_TELESCOPE_MENU_LABELS = (
    '8" SCT @ 2032 mm',
    '4" refractor @ 660 mm',
    "Different telescope",
)
PLATE_SOLVE_REFRACTOR_4IN_LABEL = '4" Refractor'
PLATE_SOLVE_REFRACTOR_4IN_FOCAL_MM = 660.0


def _compute_plate_scale_arcsec_per_px(pixel_size_um: float, focal_length_mm: float) -> float:
    """Plate scale in arcsec/pixel from pixel size (μm) and focal length (mm)."""
    if focal_length_mm <= 0.0 or pixel_size_um <= 0.0:
        return 0.0
    return 206.264806 * pixel_size_um / focal_length_mm


def _compute_fov_deg(
    pixel_size_um: float,
    focal_length_mm: float,
    sensor_w_px: float,
    sensor_h_px: float,
) -> tuple[float, float]:
    """Return (long_axis_fov_deg, short_axis_fov_deg) for the configured rig."""
    arcsec_per_px = _compute_plate_scale_arcsec_per_px(pixel_size_um, focal_length_mm)
    if arcsec_per_px <= 0.0:
        return 0.0, 0.0
    fov_w = arcsec_per_px * float(sensor_w_px) / 3600.0
    fov_h = arcsec_per_px * float(sensor_h_px) / 3600.0
    return (max(fov_w, fov_h), min(fov_w, fov_h))


def _resolve_target_name(name: str) -> tuple[float, float] | None:
    """Best-effort RA/Dec lookup for a named target.

    Tries the Messier catalog (e.g. ``M42``), planets/Sun/Moon (current
    apparent position), and a small built-in bright-star list. Returns
    ``(ra_deg, dec_deg)`` or ``None`` if not resolvable locally.
    """
    raw = (name or "").strip()
    if not raw:
        return None
    key = raw.upper().replace(" ", "")

    # Messier: M1..M110, accept "M42", "M  42", "Messier 42".
    if key.startswith("MESSIER"):
        key = "M" + key[len("MESSIER"):]
    if key.startswith("M") and key[1:].isdigit():
        from sky_ephemeris import get_messier_coordinates  # local import; module loads catalogs at import time

        ra_deg, dec_deg = get_messier_coordinates(key)
        if ra_deg is not None and dec_deg is not None:
            return float(ra_deg), float(dec_deg)

    # Solar-system bodies (apparent right now, geocentric is fine for hints).
    body_key = raw.title()
    if body_key in SOLAR_SYSTEM_BODIES:
        try:
            from sky_ephemeris import body_apparent_radec_deg

            ra, dec = body_apparent_radec_deg(
                SOLAR_SYSTEM_BODIES[body_key], DEFAULT_LAT, DEFAULT_LON, DEFAULT_ELEV_M
            )
            return float(ra), float(dec)
        except Exception:
            if body_key == "Sun":
                ra, dec = sun_apparent_radec_deg(DEFAULT_LAT, DEFAULT_LON, DEFAULT_ELEV_M)
                return float(ra), float(dec)

    # Bright/named stars (Tycho preset list — a small set bundled with the app).
    star_key = raw.title()
    if star_key in TYCHO_PRESET_STARS:
        try:
            from sky_ephemeris import catalog as _tycho_catalog

            tycho_id = TYCHO_PRESET_STARS[star_key]
            row = _tycho_catalog[_tycho_catalog["TYC_ID"] == tycho_id]
            if not row.empty:
                ra = float(row.iloc[0]["RA_deg"])
                dec = float(row.iloc[0]["Dec_deg"])
                return ra, dec
        except Exception:
            pass

    # Fallback: tiny static bright-star table (covers common alignment stars).
    _BRIGHT = {
        "SIRIUS": (101.287155, -16.716116),
        "CANOPUS": (95.987958, -52.695661),
        "ARCTURUS": (213.915300, 19.182410),
        "VEGA": (279.234734, 38.783688),
        "CAPELLA": (79.172327, 45.997991),
        "RIGEL": (78.634467, -8.201639),
        "PROCYON": (114.825493, 5.224993),
        "BETELGEUSE": (88.792939, 7.407064),
        "ALTAIR": (297.695828, 8.868322),
        "ALDEBARAN": (68.980163, 16.509302),
        "ANTARES": (247.351915, -26.432002),
        "SPICA": (201.298247, -11.161322),
        "POLLUX": (116.328958, 28.026183),
        "FOMALHAUT": (344.412750, -29.622236),
        "DENEB": (310.357979, 45.280339),
        "REGULUS": (152.092962, 11.967208),
        "POLARIS": (37.954561, 89.264109),
    }
    if key in _BRIGHT:
        return _BRIGHT[key]
    return None


# When False, skip the Session and daytime Sun dialogs on launch; main GUI opens directly.
SHOW_SESSION_STARTUP_DIALOG = False
SETUP_TARGET_BRIGHT_STARS = "Main bright stars in Tucson"
SETUP_TARGET_MOON = "Moon"
SETUP_TARGET_CONSTELLATIONS = "Constellations"

# Offline bright-star facts: see ``offline_star_facts.OFFLINE_STAR_FACTS`` (re-exported from sky_ephemeris).

# Offline lunar region fact catalog for plate-solved Moon frames.
OFFLINE_MOON_REGION_FACTS = {
    "near_side_center": "The near-side maria are ancient lava plains formed by huge impacts and later basalt floods.",
    "north_limb": "The Moon's north-limb highlands are heavily cratered, preserving very old crust.",
    "south_limb": "The south-limb region transitions toward the South Pole-Aitken basin, one of the Solar System's largest impact basins.",
    "east_limb": "The eastern limb often reveals foreshortened crater walls and shifting detail with libration.",
    "west_limb": "The western limb frequently highlights rugged terrain and crater chains near the lunar edge.",
}


class ZWOCameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRA")
        self.root.geometry("1680x860")
        self.root.minsize(1280, 700)

        # Raspberry Pi: start fullscreen (kiosk-style). Set ASTRA_FULLSCREEN=0 to disable.
        # Press Escape to leave fullscreen.
        if _is_raspberry_pi():
            _fs_env = os.environ.get("ASTRA_FULLSCREEN", "1").strip().lower()
            if _fs_env not in ("0", "false", "no", "off"):
                try:
                    self.root.update_idletasks()
                    self.root.attributes("-fullscreen", True)
                except tk.TclError:
                    pass

                def _exit_fullscreen(_ev=None):
                    try:
                        self.root.attributes("-fullscreen", False)
                    except tk.TclError:
                        pass

                self.root.bind("<Escape>", _exit_fullscreen)

        # Theme settings
        self.night_mode = False
        self.theme = THEMES["day"]

        # Initialize camera
        self.camera = None
        self.is_capturing = False
        self.is_recording = False
        self.video_writer = None
        self.available_cameras = []
        self.camera_initialized = False

        # Store full sensor dimensions
        self.full_sensor_width = None
        self.full_sensor_height = None

        self._app_dir = Path(__file__).resolve().parent
        self.livewcs_path = str(self._app_dir / "LiveWCS.txt")
        self.site_settings_path = self._app_dir / "site_settings.json"
        self.astro_preset_path = self._app_dir / "astro_preset.json"
        self.save_path = _default_capture_save_path()
        self.site_lat = DEFAULT_LAT
        self.site_lon = DEFAULT_LON
        self.site_elev_m = DEFAULT_ELEV_M
        self.wedge_acknowledged = False
        self.mount_panel = None
        self._polar_align_mode = False
        self._polar_align_panel = None
        self._polar_align_text = None
        self._polar_align_fig = None
        self._polar_align_ax = None
        self._polar_align_canvas = None
        self._polar_align_fov_deg = 5.0
        self._polar_align_history: list[tuple[float, float]] = []
        self._last_solve_ra_deg = None
        self._last_solve_dec_deg = None
        self._last_plate_solve_input_path = None
        # In-app session captures for Session photos tab (fits_path, captured_at iso, ra/dec if solved)
        self._session_photos = []
        # Focus assistant state (camera-side Focus tab). Samples are dicts with
        # keys: index, position, score (higher better), kind ("hfr"/"laplacian"),
        # raw (HFR px or Laplacian variance), n_stars, captured_at.
        self._focus_samples: list[dict] = []
        self._focus_run_busy = False
        self._latest_preview_full_rgb: np.ndarray | None = None
        self._latest_preview_full_ts: float = 0.0
        self._focus_fig = None
        self._focus_ax = None
        self._focus_canvas = None
        # Deep Sky stacking tab state. Lights/darks/flats/bias are absolute paths.
        self._dsk_lights: list[str] = []
        self._dsk_darks: list[str] = []
        self._dsk_flats: list[str] = []
        self._dsk_bias: list[str] = []
        self._dsk_cancel = threading.Event()
        self._dsk_thread: threading.Thread | None = None
        self._dsk_last_result = None
        self._dsk_preview_imgtk: ImageTk.PhotoImage | None = None
        # Bidirectional scroll regions (see ``xy_scroll_frame``); mouse wheel dispatch uses this list.
        self._xy_scroll_areas: list = []
        self._xy_plate = None
        self._xy_polar = None
        self._polar_align_shell = None
        self._dsk_count_lights_var = None
        self._dsk_count_darks_var = None
        self._dsk_count_flats_var = None
        self._dsk_count_bias_var = None
        # Equipment preset (Tucson, AZ + 2032 mm SCT). Loaded from disk if present;
        # otherwise the bundled default is used and persisted on next save.
        self.astro_preset = json.loads(json.dumps(DEFAULT_ASTRO_PRESET))  # deep-ish copy
        self._load_site_settings()
        self._load_astro_preset()

        try:
            os.makedirs(self.save_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create save directory {self.save_path}: {e}")
            # Persist a Linux/macOS/Windows-safe fallback when a stale path from
            # another machine (for example /Users/... on Raspberry Pi) is loaded.
            self.save_path = _default_capture_save_path()
            try:
                os.makedirs(self.save_path, exist_ok=True)
            except Exception:
                self.save_path = str(Path.home())
            self._save_site_settings()

        self._livewcs_last_mtime = None
        self._livewcs_poll_running = False
        self._livewcs_active = False

        # Busy indicator state (set up by setup_ui). _busy_count supports nested
        # async operations so the spinner only hides when everything is done.
        self._busy_count = 0
        self._busy_messages: list[str] = []
        self._busy_anim_index = 0
        self._busy_anim_after_id: str | None = None
        # Braille spinner – smooth and very lightweight to render in Tk.
        self._busy_anim_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Initialize ZWO library
        try:
            asi.init(_resolved_asi_library_path(self._app_dir))
            num_cameras = asi.get_num_cameras()
            if num_cameras > 0:
                self.available_cameras = _enumerate_asi_cameras()
                print(f"Cameras found: {num_cameras}")
            else:
                print("No cameras found - GUI will run in demo mode")
        except Exception as e:
            print(f"Failed to initialize camera library: {e}")
            err = str(e).lower()
            if sys.platform == "darwin" and "libusb" in err:
                print(
                    "Tip: The ZWO SDK needs libusb. On macOS with Homebrew run: brew install libusb"
                )
            if sys.platform.startswith("linux"):
                if any(
                    s in err
                    for s in (
                        "removed",
                        "permission",
                        "denied",
                        "access",
                        "libusb",
                        "could not claim",
                        "busy",
                    )
                ):
                    print(
                        "Tip (Linux / Raspberry Pi): ZWO cameras need udev rules + usbfs memory.\n"
                        "  1) Plug in the camera, then run: lsusb | grep 03c3\n"
                        "  2) Install rules: sudo install -m 0644 asi.rules /etc/udev/rules.d/asi.rules\n"
                        "     (asi.rules ships in the ASTRA project root from the ZWO SDK layout.)\n"
                        "  3) sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                        "  4) Verify: cat /sys/module/usbcore/parameters/usbfs_memory_mb  (expect 200)\n"
                        "  5) Ensure libASICamera2.so matches aarch64: file libASICamera2.so\n"
                        "  6) Quit other apps using the camera (ASICAP, INDI). Unplug/replug USB.\n"
                        "  7) If still failing, test once with: sudo -E ./run_astra_linux.sh"
                    )
            print("GUI will run in demo mode")

        self.setup_ui()
        self.apply_theme()

        # Start polling LiveWCS.txt for sky tracker data
        self._livewcs_poll_running = True
        self._poll_livewcs()

        if SHOW_SESSION_STARTUP_DIALOG:
            self.viewing_mode = None
            self.root.after(100, self._show_session_startup_dialog)
        else:
            self.viewing_mode = "night"
            self.wedge_acknowledged = True
            self._apply_mode_settings(night_mode=True, image_format="RAW8")

        # Auto-select first camera if available
        if self.available_cameras:
            self.camera_select_var.set(f"{self.available_cameras[0][0]}: {self.available_cameras[0][1]}")
            self.connect_camera()

    def _show_session_startup_dialog(self):
        """Single startup window: mode, safety confirmations, and workflow tips."""
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Session")
        dlg.geometry("460x500")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self.root)

        self.root.update_idletasks()
        w, h = 460, 500
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - w // 2
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - h // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        ctk.CTkLabel(dlg, text="Session setup", font=("Segoe UI", 18, "bold")).pack(pady=(18, 6))
        ctk.CTkLabel(dlg, text="What are you observing?", font=("Segoe UI", 12)).pack(pady=(0, 8))

        seg = ctk.CTkSegmentedButton(
            dlg,
            values=["Day", "Night", "Deep sky"],
            font=("Segoe UI", 11, "bold"),
            height=36,
            width=400,
            dynamic_resizing=False,
        )
        seg.pack(pady=(0, 12))

        label_to_mode = {"Day": "day", "Night": "night", "Deep sky": "deep_sky"}

        chk_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        chk_frame.pack(fill=tk.X, padx=24, pady=(4, 8))

        chk_wedge = ctk.CTkCheckBox(
            chk_frame,
            text="Manual 2-axis wedge is set per my mount manual",
            font=("Segoe UI", 11),
        )

        chk_solar = ctk.CTkCheckBox(
            chk_frame,
            text="A proper solar filter is installed (never point at the Sun without one)",
            font=("Segoe UI", 11),
        )

        tips = ctk.CTkLabel(
            dlg,
            text="",
            font=("Segoe UI", 11),
            text_color=("gray50", "gray65"),
            wraplength=400,
            justify="left",
        )
        tips.pack(padx=24, pady=(8, 6), anchor="w")

        err = ctk.CTkLabel(dlg, text="", font=("Segoe UI", 11), text_color="#e53935")
        err.pack(pady=(0, 6))

        def tip_for(mode: str) -> str:
            if mode == "day":
                return (
                    "After Continue, open the mount panel \u2192 Auto tracking tab \u2192 Solar tracking. "
                    "The 3-step Tucson workflow lives there: slew, manually center, then 'Aligned \u2014 Sync & Track'."
                )
            if mode == "night":
                return (
                    "After polar alignment, choose a setup target (bright stars or Moon) and the "
                    "mount will slew and start the matching tracking rate."
                )
            return (
                "After polar alignment, use setup target selection to slew to an anchor object, "
                "then continue to plate solve and framing."
            )

        def refresh_mode():
            mode = label_to_mode[seg.get()]
            tips.configure(text=tip_for(mode))
            err.configure(text="")
            chk_wedge.pack_forget()
            chk_solar.pack_forget()
            chk_wedge.pack(anchor="w", pady=4)
            if mode == "day":
                chk_solar.pack(anchor="w", pady=4)
            else:
                chk_solar.deselect()

        seg.configure(command=lambda _: refresh_mode())
        seg.set("Day")
        refresh_mode()

        def on_continue():
            mode = label_to_mode[seg.get()]
            err.configure(text="")
            if not chk_wedge.get():
                err.configure(text="Confirm the wedge is set before continuing.")
                return
            if mode == "day" and not chk_solar.get():
                err.configure(text="Confirm your solar filter before daytime solar work.")
                return

            self.viewing_mode = mode
            self.wedge_acknowledged = True
            print(f"Session mode: {mode}")
            dlg.destroy()

            if mode == "day":
                self._apply_mode_settings(night_mode=False, image_format="RAW8")
            elif mode == "night":
                self._apply_mode_settings(night_mode=True, image_format="RAW8")
            else:
                self._apply_mode_settings(night_mode=True, image_format="RAW16")
            self.root.after(80, lambda m=mode: self._show_mount_setup_workflow_dialog(m))

        bf = ctk.CTkFrame(dlg, fg_color="transparent")
        bf.pack(side=tk.BOTTOM, pady=(0, 18))
        ctk.CTkButton(
            bf,
            text="Continue",
            command=on_continue,
            width=200,
            height=40,
            font=("Segoe UI", 12, "bold"),
            fg_color="#0b5c8f",
            hover_color="#1a7ab5",
        ).pack()

    def _load_site_settings(self):
        try:
            if self.site_settings_path.is_file():
                with open(self.site_settings_path, "r") as f:
                    d = json.load(f)
                self.site_lat = float(d.get("lat", self.site_lat))
                self.site_lon = float(d.get("lon", self.site_lon))
                self.site_elev_m = float(d.get("elev_m", self.site_elev_m))
                cap = d.get("capture_save_path")
                if isinstance(cap, str) and cap.strip():
                    self.save_path = os.path.normpath(os.path.expanduser(cap.strip()))
        except Exception:
            pass

    def _save_site_settings(self):
        try:
            with open(self.site_settings_path, "w") as f:
                json.dump(
                    {
                        "lat": self.site_lat,
                        "lon": self.site_lon,
                        "elev_m": self.site_elev_m,
                        "capture_save_path": self.save_path,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Site settings save error: {e}")

    # ── Equipment preset (telescope + camera) ──────────────────────────────
    def _load_astro_preset(self):
        """Load the saved equipment preset, merging it on top of the default."""
        try:
            if self.astro_preset_path.is_file():
                with open(self.astro_preset_path, "r") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    for section in ("site", "telescope", "camera"):
                        if isinstance(saved.get(section), dict):
                            self.astro_preset.setdefault(section, {}).update(saved[section])
                    if isinstance(saved.get("name"), str):
                        self.astro_preset["name"] = saved["name"]
                    site = self.astro_preset.get("site", {})
                    if "lat" in site:
                        self.site_lat = float(site["lat"])
                    if "lon" in site:
                        self.site_lon = float(site["lon"])
                    if "elev_m" in site:
                        self.site_elev_m = float(site["elev_m"])
        except Exception as e:
            print(f"Astro preset load error: {e}")

    def _save_astro_preset(self):
        try:
            with open(self.astro_preset_path, "w") as f:
                json.dump(self.astro_preset, f, indent=2)
        except Exception as e:
            print(f"Astro preset save error: {e}")

    def _apply_mode_settings(self, night_mode, image_format):
        """Apply theme and image format based on the selected viewing mode."""
        # Apply theme if needed
        if night_mode and not self.night_mode:
            self.night_mode = True
            self.theme = THEMES["night"]
            self.theme_btn.configure(text="🌙 Night Mode")
            self.apply_theme()
        elif not night_mode and self.night_mode:
            self.night_mode = False
            self.theme = THEMES["day"]
            self.theme_btn.configure(text="☀ Day Mode")
            self.apply_theme()

        # Apply image format
        self.format_var.set(image_format)
        self.update_image_format()

    def _show_mount_setup_workflow_dialog(self, mode: str):
        """Guided setup flow after polar alignment: select target, slew, and track."""
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Mount setup workflow")
        dlg.geometry("560x360")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self.root)

        self.root.update_idletasks()
        w, h = 560, 360
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - w // 2
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - h // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        title = "Polar alignment complete — choose setup target"
        ctk.CTkLabel(dlg, text=title, font=("Segoe UI", 16, "bold")).pack(padx=16, pady=(16, 8), anchor="w")

        info_text = (
            "Night workflow: select bright stars or Moon and run GoTo + tracking.\n"
            "Day workflow: open the mount panel \u2192 Auto tracking tab \u2192 Solar tracking, "
            "then run the 3-step Sun workflow there."
        )
        ctk.CTkLabel(
            dlg,
            text=info_text,
            justify="left",
            wraplength=520,
            text_color=("gray45", "gray65"),
        ).pack(padx=16, pady=(0, 10), anchor="w")

        options = [SETUP_TARGET_BRIGHT_STARS, SETUP_TARGET_MOON, SETUP_TARGET_CONSTELLATIONS]
        target_var = tk.StringVar(value=options[0])

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(fill=tk.X, padx=16, pady=(4, 8))
        ctk.CTkLabel(row, text="Target").pack(side=tk.LEFT, padx=(0, 8))
        target_combo = ctk.CTkComboBox(
            row,
            values=options,
            variable=target_var,
            width=320,
            state="readonly",
        )
        target_combo.pack(side=tk.LEFT, padx=(0, 8))

        status_lbl = ctk.CTkLabel(
            dlg,
            text="Select target, then click Go to target + start tracking.",
            justify="left",
            wraplength=520,
            text_color=("gray45", "gray65"),
        )
        status_lbl.pack(padx=16, pady=(2, 12), anchor="w")

        def _set_status(msg: str):
            status_lbl.configure(text=msg)
            self.update_status(msg)

        def _run_target():
            selected = target_var.get()
            if selected == SETUP_TARGET_CONSTELLATIONS:
                txt = (
                    "Constellation mode currently shows guidance only. "
                    "Use Bright stars or Moon for automatic slew and tracking."
                )
                _set_status(txt)
                messagebox.showinfo("Constellations", txt, parent=dlg)
                return

            result = self._resolve_setup_target_radec(selected)
            if result is None:
                msg = f"Could not resolve coordinates for '{selected}'."
                _set_status(msg)
                messagebox.showerror("Setup target", msg, parent=dlg)
                return
            target_name, ra, dec = result
            self._goto_target_and_start_tracking(target_name, ra, dec)

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 16))
        ctk.CTkButton(btn_row, text="Close", width=120, command=dlg.destroy).pack(side=tk.RIGHT, padx=(8, 0))
        ctk.CTkButton(
            btn_row,
            text="Go to target + start tracking",
            width=260,
            command=_run_target,
            fg_color="#0b5c8f",
            hover_color="#1a7ab5",
        ).pack(side=tk.RIGHT)

    def _resolve_setup_target_radec(self, selected_target: str) -> tuple[str, float, float] | None:
        """Resolve setup dropdown target into a concrete (name, ra_deg, dec_deg)."""
        if selected_target == SETUP_TARGET_MOON:
            res = _resolve_target_name("Moon")
            if res is None:
                return None
            return "Moon", float(res[0]), float(res[1])
        if selected_target == SETUP_TARGET_BRIGHT_STARS:
            best_name = self._pick_visible_bright_star_name()
            res = _resolve_target_name(best_name)
            if res is None:
                return None
            return best_name, float(res[0]), float(res[1])
        return None

    def _pick_visible_bright_star_name(self) -> str:
        """Pick a high-altitude bright star from the built-in star list."""
        candidates = [
            "Sirius",
            "Canopus",
            "Arcturus",
            "Vega",
            "Capella",
            "Rigel",
            "Procyon",
            "Betelgeuse",
            "Altair",
            "Aldebaran",
            "Antares",
            "Spica",
            "Pollux",
            "Fomalhaut",
            "Deneb",
            "Regulus",
            "Polaris",
        ]
        best_name = "Polaris"
        best_alt = -90.0
        for name in candidates:
            try:
                resolved = _resolve_target_name(name)
                if resolved is None:
                    continue
                alt, _az = radec_to_altaz_deg(
                    float(resolved[0]),
                    float(resolved[1]),
                    self.site_lat,
                    self.site_lon,
                    self.site_elev_m,
                )
                if alt > best_alt:
                    best_alt = alt
                    best_name = name
            except Exception:
                continue
        return best_name

    def _goto_target_and_start_tracking(self, target_name: str, ra: float, dec: float):
        """Slew to target and enable target-specific tracking rate."""
        if not self.mount_panel:
            messagebox.showwarning("Mount", "Mount panel is not available.")
            return
        mount = self.mount_panel.get_mount()
        if not mount:
            messagebox.showwarning("Mount", "Connect the mount on the right first.")
            return

        if target_name.strip().lower() == "moon":
            track_code = TRACK_LUNAR
            track_label = "Lunar"
        elif target_name.strip().lower() == "sun":
            track_code = TRACK_SOLAR
            track_label = "Solar"
        else:
            track_code = TRACK_SIDEREAL
            track_label = "Sidereal"

        self.update_status(f"Setup GoTo: slewing to {target_name}…")

        def _worker():
            ok, msg = mount.goto_radec(ra, dec)
            if not ok:
                raise RuntimeError(msg or f"Mount failed GoTo to {target_name}.")
            mount.set_tracking_rate(track_code)
            mount.tracking_on()
            return msg, track_label

        def _done(result):
            goto_msg, label = result
            self.update_status(f"{target_name}: GoTo done, {label} tracking ON.")
            messagebox.showinfo(
                "Setup complete",
                f"{goto_msg}\n\nTracking enabled: {label}",
                parent=self.root,
            )

        def _fail(exc):
            messagebox.showerror("Setup target error", str(exc), parent=self.root)
            self.update_status(f"Setup target error: {exc}")

        self._run_async(
            _worker,
            busy_message=f"Slewing to {target_name} and enabling tracking…",
            on_done=_done,
            on_error=_fail,
        )

    def _get_exposure_seconds(self):
        """Return current exposure as seconds (float), used for timeout and FPS calculations."""
        range_text = self.exposure_range_var.get()
        exp_val = self.exposure_var.get()
        if range_text == "32-1000us":
            return exp_val / 1_000_000
        elif range_text in ("1-10s", "10-60s"):
            return exp_val
        else:
            return exp_val / 1000.0

    def _exposure_us_display_seconds(self, range_text: str, exposure_val: float) -> tuple[int, str, float]:
        if range_text == "32-1000us":
            exposure_us = int(exposure_val)
            exposure_display = f"{exposure_val:.0f} µs"
            exposure_s = exposure_val / 1_000_000
        elif range_text in ("1-10s", "10-60s"):
            exposure_us = int(exposure_val * 1_000_000)
            exposure_display = f"{exposure_val:.1f} s"
            exposure_s = exposure_val
        else:
            exposure_us = int(exposure_val * 1000)
            exposure_display = f"{exposure_val:.1f} ms"
            exposure_s = exposure_val / 1000.0
        return exposure_us, exposure_display, exposure_s

    def _stop_preview_for_still(self) -> bool:
        was_previewing = self.is_capturing
        if was_previewing:
            self.is_capturing = False
            time.sleep(0.2)
            try:
                if self.camera and self.camera_initialized:
                    self.camera.stop_video_capture()
            except Exception:
                pass
            self.update_status("Preview stopped for capture")
        return was_previewing

    def _configure_still_capture_hardware(
        self,
        *,
        bin_value: int,
        image_format: str,
        range_text: str,
        exposure_val: float,
        gain: int,
    ) -> tuple[str, float]:
        if not self.camera or not self.camera_initialized:
            raise RuntimeError("No camera connected")
        if self.full_sensor_width is None or self.full_sensor_height is None:
            raise RuntimeError("Camera sensor size is unknown")

        binned_width = self.full_sensor_width // bin_value
        binned_height = self.full_sensor_height // bin_value
        self.camera.set_roi(
            start_x=0,
            start_y=0,
            width=binned_width,
            height=binned_height,
            bins=bin_value,
        )

        format_map = {"RAW8": asi.ASI_IMG_RAW8, "RAW16": asi.ASI_IMG_RAW16}
        self.camera.set_image_type(format_map.get(image_format, asi.ASI_IMG_RAW16))

        exposure_us, exposure_display, exposure_s = self._exposure_us_display_seconds(range_text, exposure_val)
        self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
        self.camera.set_control_value(asi.ASI_GAIN, int(gain))
        return exposure_display, exposure_s

    def _grab_still_frame(self):
        frame = self.camera.capture()
        if frame is None:
            raise RuntimeError("Camera capture returned no frame")
        return frame

    def _mk_xy(self, parent, bg: str):
        """Create a horizontal+vertical scroll area and register it for theme + mouse wheel."""
        a = create_xy_scroll_area(parent, bg=bg)
        self._xy_scroll_areas.append(a)
        return a

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        # TOP BAR
        self.top_bar = tk.Frame(self.root)
        self.top_bar.pack(fill=tk.X, padx=12, pady=8)

        self.title_label = tk.Label(
            self.top_bar, text="CAMERA CONTROL", font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(side=tk.LEFT)

        spacer = tk.Frame(self.top_bar)
        spacer.pack(side=tk.LEFT, expand=True)

        # Busy indicator (animated spinner + status text). Visible only while
        # a long-running click handler is in flight, so the user always has
        # immediate feedback that the GUI accepted their click.
        self.busy_frame = tk.Frame(self.top_bar)
        self.busy_frame.pack(side=tk.RIGHT, padx=(0, 16))
        self.busy_spinner_label = tk.Label(
            self.busy_frame,
            text="",
            font=("Segoe UI", 14, "bold"),
            width=2,
        )
        self.busy_spinner_label.pack(side=tk.LEFT)
        self.busy_text_label = tk.Label(
            self.busy_frame,
            text="",
            font=("Segoe UI", 10),
            padx=4,
        )
        self.busy_text_label.pack(side=tk.LEFT)

        # Temperature display
        self.temp_label = tk.Label(
            self.top_bar,
            text="Temp: --°C",
            font=("Segoe UI", 10),
            padx=12
        )
        self.temp_label.pack(side=tk.RIGHT, padx=(0, 12))

        self.theme_btn = ctk.CTkButton(
            self.top_bar,
            text="☀ Day Mode",
            command=self.toggle_theme,
            font=("Segoe UI", 9),
            width=120,
            height=32,
            fg_color="#e0e0e0",
            text_color="#1a1a1a",
            hover_color="#0b5c8f",
            corner_radius=6
        )
        self.theme_btn.pack(side=tk.RIGHT)
        self.polar_align_mode_btn = ctk.CTkButton(
            self.top_bar,
            text="Polar Align Mode",
            command=self._toggle_polar_align_mode,
            font=("Segoe UI", 9, "bold"),
            width=150,
            height=32,
            fg_color="#7a0c0c",
            text_color="#ffe6e6",
            hover_color="#d11515",
            corner_radius=6,
        )
        self.polar_align_mode_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # MAIN CONTAINER — horizontal splitters so camera / center / mount can be resized (~1/3–2/3)
        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        t0 = self.theme
        self._pan_outer = tk.PanedWindow(
            self.main,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief=tk.RAISED,
            bg=t0["bg_primary"],
            bd=0,
        )
        self._pan_outer.pack(fill=tk.BOTH, expand=True)

        # Tk PanedWindow only accepts native Tk widgets; bidirectional scroll inside tk.Frame.
        self._left_pane_wrap = tk.Frame(self._pan_outer, bg=t0["bg_primary"])
        _left_xy = self._mk_xy(self._left_pane_wrap, t0["bg_primary"])
        _left_xy.outer.pack(fill=tk.BOTH, expand=True)
        self.left = _left_xy.inner

        tk.Label(
            self.left, text="CAMERA CONTROLS", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Camera selection
        tk.Label(
            self.left, text="Camera:", font=("Segoe UI", 9)
        ).pack(anchor="w")

        camera_frame = tk.Frame(self.left)
        camera_frame.pack(fill=tk.X, pady=4)

        self.camera_select_var = tk.StringVar()
        camera_options = [f"{idx}: {name}" for idx, name in self.available_cameras]
        if not camera_options:
            camera_options = ["No cameras detected"]

        self.camera_dropdown = ctk.CTkComboBox(
            camera_frame,
            variable=self.camera_select_var,
            values=camera_options,
            state="readonly",
            width=180,
            fg_color="white",
            button_color="#0b5c8f",
            button_hover_color="#1a7ab5",
            border_color="#cccccc",
            dropdown_fg_color="white",
            dropdown_hover_color="#f0f0f0",
            corner_radius=6
        )
        self.camera_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if camera_options[0] != "No cameras detected":
            self.camera_dropdown.set(camera_options[0])

        self.connect_btn = ctk.CTkButton(
            camera_frame,
            text="Connect",
            command=self.connect_camera,
            font=("Segoe UI", 8),
            width=70,
            height=28,
            fg_color="#ebebeb",
            text_color="#1a1a1a",
            hover_color="#e0e0e0",
            corner_radius=6
        )
        self.connect_btn.pack(side=tk.LEFT, padx=(4, 0))

        self.refresh_btn = ctk.CTkButton(
            camera_frame,
            text="⟳",
            command=self.refresh_cameras,
            font=("Segoe UI", 10),
            width=35,
            height=28,
            fg_color="#ebebeb",
            text_color="#1a1a1a",
            hover_color="#e0e0e0",
            corner_radius=6
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=(2, 0))

        self._camera_tabs = ctk.CTkTabview(
            self.left,
            fg_color=t0["bg_primary"],
            segmented_button_fg_color=t0["bg_tertiary"],
            segmented_button_selected_color=t0["accent"],
            segmented_button_selected_hover_color=t0["accent_light"],
            segmented_button_unselected_color=t0["bg_secondary"],
            segmented_button_unselected_hover_color=t0["bg_tertiary"],
            text_color=t0["fg_primary"],
            height=620,
        )
        self._camera_tabs.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._camera_tabs.add("Camera")
        _cam_xy = self._mk_xy(self._camera_tabs.tab("Camera"), t0["bg_primary"])
        _cam_xy.outer.pack(fill=tk.BOTH, expand=True)
        camera_parent = _cam_xy.inner
        self._camera_tab_inner = camera_parent

        # Helper function to create dropdown controls
        def create_dropdown_control(label_text, var, options, command=None):
            tk.Label(camera_parent, text=label_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
            dropdown = ctk.CTkComboBox(
                camera_parent,
                variable=var if var else None,
                values=options,
                state="readonly",
                width=240,
                fg_color="white",
                button_color="#0b5c8f",
                button_hover_color="#1a7ab5",
                border_color="#cccccc",
                dropdown_fg_color="white",
                dropdown_hover_color="#f0f0f0",
                corner_radius=6,
                command=command
            )
            dropdown.pack(fill=tk.X, pady=4)
            return dropdown

        # BINNING
        self.binning_dropdown = create_dropdown_control(
            "Binning:",
            None,
            ["1x1", "2x2", "3x3", "4x4"],
            self.update_binning
        )
        self.binning_dropdown.set("1x1")

        # IMAGE FORMAT
        self.format_var = tk.StringVar(value="RAW16")
        self.format_dropdown = create_dropdown_control(
            "Image Format:",
            self.format_var,
            ["RAW8", "RAW16"],
            self.update_image_format
        )

        # CAPTURE COUNT
        self.capture_count_dropdown = create_dropdown_control(
            "Capture Count:",
            None,
            ["1", "5", "10", "25", "50"]
        )
        self.capture_count_dropdown.set("1")

        # EXPOSURE RANGE — includes new 32-1000µs option
        self.exposure_range_var = tk.StringVar(value="1-100ms")
        self.exposure_range_dropdown = create_dropdown_control(
            "Exposure Range:",
            self.exposure_range_var,
            ["32-1000us", "1-100ms", "100ms-1000ms", "1-10s", "10-60s"],
            self.update_exposure_range
        )

        # Helper function to create slider controls
        def create_slider_control(label_text, var, from_, to, command, display_width=10):
            tk.Label(camera_parent, text=label_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
            frame = tk.Frame(camera_parent)
            frame.pack(fill=tk.X, pady=4)

            slider = ctk.CTkSlider(
                frame,
                from_=from_,
                to=to,
                variable=var,
                command=command,
                width=160,
                height=16,
                **_RED_SLIDER,
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

            label = tk.Label(frame, text="", width=display_width, font=("Segoe UI", 9))
            label.pack(side=tk.LEFT, padx=(4, 0))

            return slider, label

        # EXPOSURE
        self.exposure_var = tk.DoubleVar(value=10)
        self.exposure_slider, self.exposure_label = create_slider_control(
            "Exposure:",
            self.exposure_var,
            1,
            100,
            self.update_exposure
        )
        self.exposure_label.configure(text="10.0 ms")

        # GAIN
        self.gain_var = tk.IntVar(value=0)
        self.gain_slider, self.gain_label = create_slider_control(
            "Gain:",
            self.gain_var,
            0,
            500,
            self.update_gain
        )
        self.gain_label.configure(text="0")

        # FILE PATH
        tk.Label(
            camera_parent, text="Save Path:", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        path_frame = tk.Frame(camera_parent)
        path_frame.pack(fill=tk.X, pady=4)

        self.path_entry = ctk.CTkEntry(
            path_frame,
            font=("Segoe UI", 8),
            height=28,
            fg_color="white",
            text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6
        )
        self.path_entry.insert(0, self.save_path)
        self.path_entry.configure(state="readonly")
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_btn = ctk.CTkButton(
            path_frame,
            text="Browse...",
            command=self.select_save_path,
            font=("Segoe UI", 8),
            width=80,
            height=28,
            fg_color="#ebebeb",
            text_color="#1a1a1a",
            hover_color="#e0e0e0",
            corner_radius=6
        )
        self.browse_btn.pack(side=tk.LEFT, padx=(4, 0))

        # PROJECT NAME
        tk.Label(
            camera_parent, text="Project Name:", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        self.project_name_entry = ctk.CTkEntry(
            camera_parent,
            font=("Segoe UI", 8),
            height=28,
            fg_color="white",
            text_color="#1a1a1a",
            placeholder_text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6,
            placeholder_text="e.g. Jupiter_2026"
        )
        self.project_name_entry.pack(fill=tk.X, pady=4)

        # FRAME TYPE
        tk.Label(
            camera_parent, text="Frame Type:", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        self.frame_type_var = tk.StringVar(value="Light")
        self.frame_type_dropdown = ctk.CTkComboBox(
            camera_parent,
            variable=self.frame_type_var,
            values=["Light", "Dark", "Flat", "Bias"],
            font=("Segoe UI", 9),
            height=28,
            fg_color="white",
            text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6,
            button_color="#0b5c8f",
            button_hover_color="#1a7ab5",
            state="readonly",
            dropdown_fg_color="white",
            dropdown_hover_color="#f0f0f0",
            dropdown_text_color="#1a1a1a",
        )
        self.frame_type_dropdown.pack(fill=tk.X, pady=4)

        # CAPTURE BUTTONS
        ttk.Separator(camera_parent).pack(fill=tk.X, pady=10)

        tk.Label(
            camera_parent, text="CAPTURE", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Helper function to create action buttons
        def create_action_button(text, command):
            return ctk.CTkButton(
                camera_parent,
                text=text,
                command=command,
                font=("Segoe UI", 9, "bold"),
                height=36,
                fg_color="#0b5c8f",
                text_color="white",
                hover_color="#1a7ab5",
                corner_radius=6
            )

        self.capture_img_btn = create_action_button("Capture Image", self.capture_image)
        self.capture_img_btn.pack(fill=tk.X, pady=4)

        self.record_video_btn = create_action_button("Start Recording", self.toggle_recording)
        self.record_video_btn.pack(fill=tk.X, pady=4)

        # FOCUS ASSISTANT TAB — sibling of "Camera"
        self._camera_tabs.add("Focus")
        _focus_xy = self._mk_xy(self._camera_tabs.tab("Focus"), t0["bg_primary"])
        _focus_xy.outer.pack(fill=tk.BOTH, expand=True)
        self._focus_tab_inner = _focus_xy.inner
        self._xy_camera_tab_scroll = _cam_xy
        self._xy_focus_tab_scroll = _focus_xy
        self._build_focus_tab(_focus_xy.inner)

        # SKY TRACKER SECTION
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left, text="SKY TRACKER", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Sky tracker status info
        skytrack_info_frame = tk.Frame(self.left)
        skytrack_info_frame.pack(fill=tk.X, pady=(6, 2))

        self.skytrack_signal_dot = tk.Label(
            skytrack_info_frame,
            text="●",
            font=("Segoe UI", 10),
            fg="#999999"
        )
        self.skytrack_signal_dot.pack(side=tk.LEFT, padx=(2, 4))

        self.skytrack_signal_label = tk.Label(
            skytrack_info_frame,
            text="No sky target",
            font=("Segoe UI", 9, "italic"),
            fg="#999999"
        )
        self.skytrack_signal_label.pack(side=tk.LEFT)

        self.skytrack_target_label = tk.Label(
            self.left,
            text="Target:",
            font=("Segoe UI", 9),
            anchor="w"
        )
        self.skytrack_target_label.pack(fill=tk.X, padx=4)

        self.skytrack_ra_label = tk.Label(
            self.left,
            text="RA :",
            font=("Segoe UI", 9, "bold"),
            anchor="w"
        )
        self.skytrack_ra_label.pack(fill=tk.X, padx=4)

        self.skytrack_dec_label = tk.Label(
            self.left,
            text="DEC :",
            font=("Segoe UI", 9, "bold"),
            anchor="w"
        )
        self.skytrack_dec_label.pack(fill=tk.X, padx=4)

        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left, text="STATUS", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 4))

        self.status_label = tk.Label(
            self.left,
            text="Ready - No camera connected",
            font=("Segoe UI", 9, "italic"),
            wraplength=240,
            justify=tk.LEFT,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)

        # Inner splitter: center tabs | mount (each side keeps a minimum width ≈ 1/3 of inner area)
        self._pan_inner = tk.PanedWindow(
            self._pan_outer,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief=tk.RAISED,
            bg=t0["bg_primary"],
            bd=0,
        )
        self.right = tk.Frame(self._pan_inner, bg=t0["bg_primary"])
        self.side_tools = tk.Frame(self._pan_inner, bg=t0["bg_primary"])

        # Inner needs room for center + mount mins; outer mins must fit (camera + inner ≥ window min).
        self._pan_outer.add(self._left_pane_wrap, minsize=260, stretch="never")
        self._pan_outer.add(self._pan_inner, minsize=980)
        self._pan_inner.add(self.right, minsize=380)
        self._pan_inner.add(self.side_tools, minsize=520)

        self.root.after_idle(self._set_initial_pane_sashes)

        self._build_center_tabs()
        self._build_mount_column()
        attach_mousewheel_dispatch(
            self.root,
            self._xy_scroll_areas,
            extra_area_lists=[
                getattr(self.mount_panel, "_mount_xy_areas", []),
            ],
        )

    def _build_center_tabs(self):
        t = self.theme
        self._center_tabs = ctk.CTkTabview(
            self.right,
            fg_color=t["bg_primary"],
            segmented_button_fg_color=t["bg_tertiary"],
            segmented_button_selected_color=t["accent"],
            segmented_button_selected_hover_color=t["accent_light"],
            segmented_button_unselected_color=t["bg_secondary"],
            segmented_button_unselected_hover_color=t["bg_tertiary"],
            text_color=t["fg_primary"],
        )
        self._center_tabs.pack(fill=tk.BOTH, expand=True)

        self._center_tabs.add("Live preview")
        live_tab = self._center_tabs.tab("Live preview")
        _live_xy = self._mk_xy(live_tab, t["bg_primary"])
        _live_xy.outer.pack(fill=tk.BOTH, expand=True)
        self.preview_label = tk.Label(
            _live_xy.inner,
            text="No preview available\nConnect a camera for live preview",
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._center_tabs.add("Session photos")
        session_tab = self._center_tabs.tab("Session photos")
        _sess_xy = self._mk_xy(session_tab, t["bg_primary"])
        _sess_xy.outer.pack(fill=tk.BOTH, expand=True)
        session_inner = _sess_xy.inner
        sbtn = ctk.CTkFrame(session_inner, fg_color="transparent")
        sbtn.pack(fill=tk.X, padx=8, pady=6)
        ctk.CTkButton(
            sbtn,
            text="Use selected for plate solve",
            command=self._session_use_selected_for_plate,
            width=200,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkButton(
            sbtn,
            text="Scan for solved FITS",
            command=self._backfill_session_radec_from_disk,
            width=150,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)
        ctk.CTkButton(
            sbtn,
            text="Generate PDF report",
            command=self._generate_session_pdf_report,
            width=170,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(8, 0))
        list_wrap = tk.Frame(session_inner)
        list_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._session_listbox = tk.Listbox(
            list_wrap,
            height=12,
            font=("Courier New", 10),
            selectmode=tk.BROWSE,
            activestyle="dotbox",
        )
        sb = tk.Scrollbar(list_wrap, orient=tk.VERTICAL, command=self._session_listbox.yview)
        self._session_listbox.configure(yscrollcommand=sb.set)
        self._session_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._center_tabs.add("Site / Sky")
        site_tab = self._center_tabs.tab("Site / Sky")
        _site_xy = self._mk_xy(site_tab, t["bg_primary"])
        _site_xy.outer.pack(fill=tk.BOTH, expand=True)
        site = _site_xy.inner
        ctk.CTkLabel(site, text="Observer site (degrees, WGS84)", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=8, pady=(6, 4)
        )
        ctk.CTkLabel(site, text="Latitude").pack(anchor="w", padx=8)
        self._site_lat_entry = ctk.CTkEntry(site, width=220)
        self._site_lat_entry.insert(0, str(self.site_lat))
        self._site_lat_entry.pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkLabel(site, text="Longitude (E +)").pack(anchor="w", padx=8)
        self._site_lon_entry = ctk.CTkEntry(site, width=220)
        self._site_lon_entry.insert(0, str(self.site_lon))
        self._site_lon_entry.pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkLabel(site, text="Elevation (m)").pack(anchor="w", padx=8)
        self._site_elev_entry = ctk.CTkEntry(site, width=220)
        self._site_elev_entry.insert(0, str(self.site_elev_m))
        self._site_elev_entry.pack(anchor="w", padx=8, pady=(0, 6))
        ctk.CTkButton(site, text="Save site to disk", command=self._apply_site_from_ui, width=200).pack(
            anchor="w", padx=8, pady=4
        )

        gpsf = ctk.CTkFrame(site, fg_color="transparent")
        gpsf.pack(fill=tk.X, padx=8, pady=8)
        ctk.CTkLabel(gpsf, text="GPS serial").pack(side=tk.LEFT)
        try:
            import serial.tools.list_ports as st

            gps_ports = [p.device for p in st.comports()] or ["(none)"]
        except Exception:
            gps_ports = ["(none)"]
        self._gps_port_menu = ctk.CTkComboBox(gpsf, values=gps_ports, width=160)
        self._gps_port_menu.set(gps_ports[0])
        self._gps_port_menu.pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(gpsf, text="Read GPS", command=self._read_gps_fill_site, width=100).pack(side=tk.LEFT)

        ctk.CTkButton(site, text="Update body table (needs last plate solve)", command=self._refresh_body_table).pack(
            anchor="w", padx=8, pady=6
        )
        ctk.CTkButton(site, text="Polar hints (MVP, from last solve)", command=self._show_polar_hints_text).pack(
            anchor="w", padx=8, pady=4
        )
        self._sky_log = tk.Text(site, height=14, width=48, font=("Courier New", 9), wrap=tk.WORD)
        self._sky_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self._center_tabs.add("Deep Sky")
        deep = self._center_tabs.tab("Deep Sky")
        self._build_deep_sky_tab(deep)

        self._center_tabs.add("Plate solve")
        plate = self._center_tabs.tab("Plate solve")
        self._build_plate_solve_tab(plate)

    # ──────────────────────────────────────────────────────────────────────
    #  Deep Sky tab — offline stacking workflow (lights + calibration frames)
    # ──────────────────────────────────────────────────────────────────────
    def _build_deep_sky_tab(self, parent):
        t = self.theme

        try:
            parent.configure(fg_color=t["bg_primary"])
        except Exception:
            pass

        _dsk_xy = self._mk_xy(parent, t["bg_primary"])
        _dsk_xy.outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        scroll = _dsk_xy.inner

        ctk.CTkLabel(
            scroll,
            text="Deep Sky stacking (offline)",
            font=("Segoe UI", 13, "bold"),
            text_color=t["fg_primary"],
        ).pack(anchor="w", padx=8, pady=(8, 2))
        ctk.CTkLabel(
            scroll,
            text="Pick FITS lights and optional dark/flat/bias frames, then "
                 "stack — runs locally, no internet required.",
            font=("Segoe UI", 9),
            text_color=t["fg_secondary"],
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        self._dsk_count_lights_var = tk.StringVar(value="0 frames")
        self._dsk_count_darks_var = tk.StringVar(value="0 frames")
        self._dsk_count_flats_var = tk.StringVar(value="0 frames")
        self._dsk_count_bias_var = tk.StringVar(value="0 frames")

        self._dsk_build_picker_row(
            scroll, "Lights (required)", self._dsk_count_lights_var, "lights"
        )
        self._dsk_build_picker_row(
            scroll, "Darks (optional)", self._dsk_count_darks_var, "darks"
        )
        self._dsk_build_picker_row(
            scroll, "Flats (optional)", self._dsk_count_flats_var, "flats"
        )
        self._dsk_build_picker_row(
            scroll, "Bias (optional)", self._dsk_count_bias_var, "bias"
        )

        # Settings ------------------------------------------------------------
        settings = ctk.CTkFrame(scroll, fg_color=t["bg_secondary"])
        settings.pack(fill=tk.X, padx=8, pady=(4, 4))

        ctk.CTkLabel(
            settings, text="Settings", font=("Segoe UI", 10, "bold"),
            text_color=t["fg_primary"],
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 4))

        ctk.CTkLabel(settings, text="Combine", text_color=t["fg_primary"]).grid(
            row=1, column=0, sticky="w", padx=8, pady=2
        )
        self._dsk_combine_menu = ctk.CTkOptionMenu(
            settings, values=list(deep_sky_stacker.COMBINE_METHODS), width=120,
        )
        self._dsk_combine_menu.set("sigma_clip")
        self._dsk_combine_menu.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ctk.CTkLabel(settings, text="σ low", text_color=t["fg_primary"]).grid(
            row=1, column=2, sticky="e", padx=(8, 2), pady=2
        )
        self._dsk_sigma_low_entry = ctk.CTkEntry(settings, width=60)
        self._dsk_sigma_low_entry.insert(0, "3.0")
        self._dsk_sigma_low_entry.grid(row=1, column=3, sticky="w", padx=2, pady=2)

        ctk.CTkLabel(settings, text="σ high", text_color=t["fg_primary"]).grid(
            row=1, column=4, sticky="e", padx=(8, 2), pady=2
        )
        self._dsk_sigma_high_entry = ctk.CTkEntry(settings, width=60)
        self._dsk_sigma_high_entry.insert(0, "3.0")
        self._dsk_sigma_high_entry.grid(row=1, column=5, sticky="w", padx=2, pady=2)

        ctk.CTkLabel(settings, text="Reference", text_color=t["fg_primary"]).grid(
            row=2, column=0, sticky="w", padx=8, pady=2
        )
        self._dsk_ref_menu = ctk.CTkOptionMenu(
            settings, values=["auto (most stars)", "first frame"], width=170,
        )
        self._dsk_ref_menu.set("auto (most stars)")
        self._dsk_ref_menu.grid(row=2, column=1, columnspan=2, sticky="w", padx=4, pady=2)

        ctk.CTkLabel(settings, text="Bayer", text_color=t["fg_primary"]).grid(
            row=2, column=3, sticky="e", padx=(8, 2), pady=2
        )
        self._dsk_bayer_menu = ctk.CTkOptionMenu(
            settings, values=list(deep_sky_stacker.BAYER_PATTERNS), width=110,
        )
        self._dsk_bayer_menu.set("auto")
        self._dsk_bayer_menu.grid(row=2, column=4, columnspan=2, sticky="w", padx=2, pady=2)

        ctk.CTkLabel(settings, text="Stretch", text_color=t["fg_primary"]).grid(
            row=3, column=0, sticky="w", padx=8, pady=2
        )
        self._dsk_stretch_menu = ctk.CTkOptionMenu(
            settings, values=list(deep_sky_stacker.STRETCH_MODES), width=120,
        )
        self._dsk_stretch_menu.set("asinh")
        self._dsk_stretch_menu.grid(row=3, column=1, sticky="w", padx=4, pady=(2, 6))

        # Output row ----------------------------------------------------------
        out_row = ctk.CTkFrame(scroll, fg_color="transparent")
        out_row.pack(fill=tk.X, padx=8, pady=(2, 4))
        ctk.CTkLabel(out_row, text="Output folder", text_color=t["fg_primary"]).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._dsk_output_entry = ctk.CTkEntry(out_row, width=320)
        self._dsk_output_entry.insert(0, self._dsk_default_output_dir())
        self._dsk_output_entry.pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(
            out_row, text="Browse…", width=80, command=self._dsk_browse_output,
        ).pack(side=tk.LEFT, padx=4)

        # Action row ----------------------------------------------------------
        action = ctk.CTkFrame(scroll, fg_color="transparent")
        action.pack(fill=tk.X, padx=8, pady=(2, 4))
        self._dsk_stack_btn = ctk.CTkButton(
            action, text="Stack", width=120, command=self._dsk_start_stack,
            fg_color=t["accent"], hover_color=t["accent_light"],
        )
        self._dsk_stack_btn.pack(side=tk.LEFT)
        self._dsk_cancel_btn = ctk.CTkButton(
            action, text="Cancel", width=80, command=self._dsk_cancel_stack,
            state="disabled",
        )
        self._dsk_cancel_btn.pack(side=tk.LEFT, padx=4)
        self._dsk_progress = ttk.Progressbar(
            action, orient="horizontal", mode="determinate", length=240,
        )
        self._dsk_progress.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        self._dsk_status_var = tk.StringVar(value="Idle")
        ctk.CTkLabel(
            action, textvariable=self._dsk_status_var, text_color=t["fg_secondary"],
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=4)

        # Log -----------------------------------------------------------------
        log_wrap = tk.Frame(scroll, bg=t["bg_primary"])
        log_wrap.pack(fill=tk.BOTH, expand=False, padx=8, pady=(4, 4))
        self._dsk_log = tk.Text(
            log_wrap, height=10, font=("Courier New", 9), wrap=tk.WORD,
            bg=t["bg_secondary"], fg=t["fg_primary"], insertbackground=t["fg_primary"],
        )
        log_sb = tk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self._dsk_log.yview)
        self._dsk_log.configure(yscrollcommand=log_sb.set)
        self._dsk_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._dsk_log.insert("end", "Ready. Pick frames and click Stack.\n")
        self._dsk_log.configure(state="disabled")

        # Result viewer + actions --------------------------------------------
        result_wrap = ctk.CTkFrame(scroll, fg_color=t["bg_secondary"])
        result_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        ctk.CTkLabel(
            result_wrap, text="Stacked result", font=("Segoe UI", 10, "bold"),
            text_color=t["fg_primary"],
        ).pack(anchor="w", padx=8, pady=(6, 2))

        self._dsk_preview_label = tk.Label(
            result_wrap, text="No stack yet", bg=t["bg_secondary"], fg=t["fg_secondary"],
            width=64, height=20, justify=tk.CENTER,
        )
        self._dsk_preview_label.pack(padx=8, pady=4)

        result_btns = ctk.CTkFrame(result_wrap, fg_color="transparent")
        result_btns.pack(fill=tk.X, padx=8, pady=(2, 8))
        self._dsk_open_folder_btn = ctk.CTkButton(
            result_btns, text="Open output folder", width=160,
            command=self._dsk_open_output_folder, state="disabled",
        )
        self._dsk_open_folder_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._dsk_use_plate_btn = ctk.CTkButton(
            result_btns, text="Use stacked FITS in plate solve", width=220,
            command=self._dsk_use_in_plate, state="disabled",
        )
        self._dsk_use_plate_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._dsk_save_preview_btn = ctk.CTkButton(
            result_btns, text="Save preview as…", width=160,
            command=self._dsk_save_preview_as, state="disabled",
        )
        self._dsk_save_preview_btn.pack(side=tk.LEFT)

    # ----- Deep Sky helpers --------------------------------------------------

    def _dsk_build_picker_row(self, parent, label: str, count_var: tk.StringVar, kind: str):
        """One picker row (Add files / Add folder / Clear) for a frame kind."""
        t = self.theme
        row = ctk.CTkFrame(parent, fg_color=t["bg_secondary"])
        row.pack(fill=tk.X, padx=8, pady=2)
        ctk.CTkLabel(
            row, text=label, font=("Segoe UI", 10, "bold"),
            text_color=t["fg_primary"], width=170, anchor="w",
        ).pack(side=tk.LEFT, padx=(8, 4), pady=4)
        ctk.CTkLabel(
            row, textvariable=count_var, text_color=t["fg_secondary"], width=80,
        ).pack(side=tk.LEFT, padx=2, pady=4)
        ctk.CTkButton(
            row, text="Add files…", width=90,
            command=lambda k=kind: self._dsk_add_files(k),
        ).pack(side=tk.LEFT, padx=2, pady=4)
        ctk.CTkButton(
            row, text="Add folder…", width=100,
            command=lambda k=kind: self._dsk_add_folder(k),
        ).pack(side=tk.LEFT, padx=2, pady=4)
        ctk.CTkButton(
            row, text="Clear", width=70,
            command=lambda k=kind: self._dsk_clear(k),
        ).pack(side=tk.LEFT, padx=2, pady=4)

    def _dsk_kind_to_attrs(self, kind: str) -> tuple[list[str], tk.StringVar]:
        if kind == "lights":
            return self._dsk_lights, self._dsk_count_lights_var
        if kind == "darks":
            return self._dsk_darks, self._dsk_count_darks_var
        if kind == "flats":
            return self._dsk_flats, self._dsk_count_flats_var
        if kind == "bias":
            return self._dsk_bias, self._dsk_count_bias_var
        raise ValueError(f"unknown frame kind: {kind}")

    def _dsk_default_output_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        project = ""
        try:
            project = self.project_name_entry.get().strip()
        except Exception:
            pass
        if project:
            return os.path.join(self.save_path, project, "Stacks", ts)
        return os.path.join(self.save_path, "Stacks", ts)

    def _dsk_default_dir_for_kind(self, kind: str) -> str:
        kind_to_folder = {
            "lights": "Light", "darks": "Dark", "flats": "Flat", "bias": "Bias",
        }
        sub = kind_to_folder.get(kind, "")
        try:
            project = self.project_name_entry.get().strip()
        except Exception:
            project = ""
        candidates = []
        if project:
            candidates.append(os.path.join(self.save_path, project, sub))
            candidates.append(os.path.join(self.save_path, project))
        candidates.append(os.path.join(self.save_path, sub))
        candidates.append(self.save_path)
        for c in candidates:
            if os.path.isdir(c):
                return c
        return self.save_path

    def _dsk_add_files(self, kind: str):
        paths_list, count_var = self._dsk_kind_to_attrs(kind)
        initdir = self._dsk_default_dir_for_kind(kind)
        files = filedialog.askopenfilenames(
            title=f"Select {kind} FITS files",
            initialdir=initdir,
            filetypes=[("FITS", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not files:
            return
        added = 0
        for f in files:
            ap = os.path.abspath(f)
            if ap not in paths_list:
                paths_list.append(ap)
                added += 1
        count_var.set(f"{len(paths_list)} frames")
        self._dsk_log_line(f"Added {added} {kind} file(s); total {len(paths_list)}.")

    def _dsk_add_folder(self, kind: str):
        paths_list, count_var = self._dsk_kind_to_attrs(kind)
        initdir = self._dsk_default_dir_for_kind(kind)
        folder = filedialog.askdirectory(
            title=f"Select {kind} folder", initialdir=initdir
        )
        if not folder:
            return
        added = 0
        for name in sorted(os.listdir(folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in (".fits", ".fit", ".fts"):
                continue
            ap = os.path.abspath(os.path.join(folder, name))
            if ap not in paths_list:
                paths_list.append(ap)
                added += 1
        count_var.set(f"{len(paths_list)} frames")
        self._dsk_log_line(
            f"Added {added} {kind} file(s) from {folder}; total {len(paths_list)}."
        )

    def _dsk_clear(self, kind: str):
        paths_list, count_var = self._dsk_kind_to_attrs(kind)
        paths_list.clear()
        count_var.set("0 frames")
        self._dsk_log_line(f"Cleared {kind}.")

    def _dsk_browse_output(self):
        current = self._dsk_output_entry.get().strip() or self.save_path
        parent = current if os.path.isdir(current) else (
            os.path.dirname(current) if current else self.save_path
        )
        folder = filedialog.askdirectory(title="Output folder", initialdir=parent)
        if folder:
            self._dsk_output_entry.delete(0, "end")
            self._dsk_output_entry.insert(0, folder)

    def _dsk_log_line(self, msg: str):
        try:
            self._dsk_log.configure(state="normal")
            self._dsk_log.insert("end", msg + "\n")
            self._dsk_log.see("end")
            self._dsk_log.configure(state="disabled")
        except Exception:
            pass

    def _dsk_set_busy(self, busy: bool):
        try:
            self._dsk_stack_btn.configure(state="disabled" if busy else "normal")
            self._dsk_cancel_btn.configure(state="normal" if busy else "disabled")
        except Exception:
            pass

    def _dsk_start_stack(self):
        if self._dsk_thread is not None and self._dsk_thread.is_alive():
            return
        if not self._dsk_lights:
            messagebox.showinfo("Deep Sky", "Add at least one light frame first.")
            return

        try:
            sig_lo = float(self._dsk_sigma_low_entry.get())
            sig_hi = float(self._dsk_sigma_high_entry.get())
        except Exception:
            messagebox.showwarning("Deep Sky", "Sigma low/high must be numbers.")
            return

        out_dir = self._dsk_output_entry.get().strip()
        if not out_dir:
            out_dir = self._dsk_default_output_dir()
            self._dsk_output_entry.insert(0, out_dir)

        ref_choice = self._dsk_ref_menu.get()
        ref_index = 0 if ref_choice == "first frame" else None

        cfg = deep_sky_stacker.StackConfig(
            light_paths=list(self._dsk_lights),
            dark_paths=list(self._dsk_darks),
            flat_paths=list(self._dsk_flats),
            bias_paths=list(self._dsk_bias),
            output_dir=out_dir,
            output_basename="stack_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            combine=self._dsk_combine_menu.get(),
            sigma_low=sig_lo,
            sigma_high=sig_hi,
            reference_index=ref_index,
            stretch=self._dsk_stretch_menu.get(),
            bayer_pattern=self._dsk_bayer_menu.get(),
            cancel_event=self._dsk_cancel,
        )

        self._dsk_cancel.clear()
        self._dsk_set_busy(True)
        self._dsk_progress.configure(value=0, maximum=100)
        self._dsk_status_var.set("Starting…")
        self._dsk_log_line("=" * 50)
        self._dsk_log_line(
            f"Stacking {len(cfg.light_paths)} lights "
            f"(D={len(cfg.dark_paths)}, F={len(cfg.flat_paths)}, "
            f"B={len(cfg.bias_paths)}); combine={cfg.combine}, stretch={cfg.stretch}."
        )

        def _worker():
            try:
                result = deep_sky_stacker.run_stack_job(
                    cfg,
                    progress_cb=self._dsk_progress_cb,
                    log_cb=self._dsk_log_cb,
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                result = deep_sky_stacker.StackResult(
                    success=False, message=f"Worker crashed: {e}\n{tb}"
                )
            self.root.after(0, lambda r=result: self._dsk_on_done(r))

        self._dsk_thread = threading.Thread(target=_worker, daemon=True)
        self._dsk_thread.start()

    def _dsk_progress_cb(self, done: int, total: int, label: str):
        def _apply():
            try:
                pct = 0 if total <= 0 else int(round(100.0 * done / max(1, total)))
                self._dsk_progress.configure(value=max(0, min(100, pct)))
                self._dsk_status_var.set(f"{label} ({done}/{total})")
            except Exception:
                pass
        self.root.after(0, _apply)

    def _dsk_log_cb(self, msg: str):
        self.root.after(0, lambda m=msg: self._dsk_log_line(m))

    def _dsk_cancel_stack(self):
        if self._dsk_thread is not None and self._dsk_thread.is_alive():
            self._dsk_cancel.set()
            self._dsk_status_var.set("Cancelling…")
            self._dsk_log_line("Cancellation requested.")

    def _dsk_on_done(self, result):
        self._dsk_set_busy(False)
        self._dsk_last_result = result
        if not result.success:
            self._dsk_status_var.set("Failed")
            self._dsk_log_line(f"FAILED: {result.message}")
            messagebox.showerror("Deep Sky", result.message or "Stacking failed.")
            return
        self._dsk_progress.configure(value=100)
        self._dsk_status_var.set(
            f"Done — {result.n_used} stacked, {result.n_skipped} skipped"
        )
        self._dsk_log_line("SUCCESS: " + result.message)
        self._dsk_log_line(f"  FITS:    {result.fits_path}")
        self._dsk_log_line(f"  Preview: {result.preview_path}")
        self._dsk_show_preview(result.preview_path)
        try:
            self._dsk_open_folder_btn.configure(state="normal")
            self._dsk_use_plate_btn.configure(state="normal")
            self._dsk_save_preview_btn.configure(state="normal")
        except Exception:
            pass
        self.update_status(
            f"Stacked {result.n_used} frames -> {os.path.basename(result.fits_path)}"
        )

    def _dsk_show_preview(self, png_path: str):
        try:
            if not png_path or not os.path.isfile(png_path):
                return
            pil = Image.open(png_path)
            max_w, max_h = 720, 480
            w, h = pil.size
            scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
            if scale < 1.0:
                pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self._dsk_preview_imgtk = ImageTk.PhotoImage(pil)
            self._dsk_preview_label.configure(
                image=self._dsk_preview_imgtk, text="", width=pil.size[0], height=pil.size[1],
            )
        except Exception as e:
            self._dsk_log_line(f"Preview display failed: {e}")

    def _dsk_open_output_folder(self):
        if self._dsk_last_result is None or not self._dsk_last_result.fits_path:
            return
        folder = os.path.dirname(self._dsk_last_result.fits_path) or self.save_path
        try:
            if sys.platform == "darwin":
                os.system(f'open "{folder}"')
            elif sys.platform.startswith("linux"):
                os.system(f'xdg-open "{folder}"')
            elif sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showwarning("Deep Sky", f"Could not open folder: {e}")

    def _dsk_use_in_plate(self):
        if self._dsk_last_result is None or not self._dsk_last_result.fits_path:
            return
        if not hasattr(self, "_plate_path_entry"):
            messagebox.showinfo("Deep Sky", "Plate solve tab is not available.")
            return
        try:
            self._plate_path_entry.delete(0, "end")
            self._plate_path_entry.insert(0, self._dsk_last_result.fits_path)
            self._focus_center_tab("Plate solve")
            self.update_status("Stacked FITS copied — open Plate solve")
        except Exception as e:
            messagebox.showwarning("Deep Sky", f"Could not load into plate solve: {e}")

    def _dsk_save_preview_as(self):
        if self._dsk_last_result is None or not self._dsk_last_result.preview_path:
            return
        src = self._dsk_last_result.preview_path
        if not os.path.isfile(src):
            return
        target = filedialog.asksaveasfilename(
            title="Save preview PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=os.path.basename(src),
        )
        if not target:
            return
        try:
            with open(src, "rb") as fr, open(target, "wb") as fw:
                fw.write(fr.read())
            self._dsk_log_line(f"Saved preview copy -> {target}")
        except Exception as e:
            messagebox.showwarning("Deep Sky", f"Save failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    #  Plate-solve tab — always-red palette, equipment preset, target tools
    # ──────────────────────────────────────────────────────────────────────
    def _build_plate_solve_tab(self, plate):
        """Build the redesigned plate-solve tab with red theme and presets."""
        pal = PLATE_RED_PAL

        # The CTk tab background itself (so red shows beneath everything).
        try:
            plate.configure(fg_color=pal["bg"])
        except Exception:
            pass

        self._xy_plate = self._mk_xy(plate, pal["bg"])
        self._xy_plate.outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        outer = ctk.CTkFrame(self._xy_plate.inner, fg_color=pal["bg"], border_color=pal["border"], border_width=0)
        outer.pack(fill=tk.BOTH, expand=True)
        # Track all plate-tab widgets so apply_theme can reapply red colours.
        self._plate_widgets: list = [plate, outer]

        def _track(w):
            self._plate_widgets.append(w)
            return w

        def _label(parent, text, *, bold=False, dim=False, size=10):
            font = ("Segoe UI", size, "bold") if bold else ("Segoe UI", size)
            fg = pal["fg_dim"] if dim else pal["fg"]
            return _track(ctk.CTkLabel(parent, text=text, font=font, text_color=fg))

        def _section_frame(parent):
            f = ctk.CTkFrame(
                parent,
                fg_color=pal["bg_alt"],
                border_color=pal["border"],
                border_width=1,
                corner_radius=6,
            )
            return _track(f)

        def _entry(parent, *, width=110, placeholder=""):
            e = ctk.CTkEntry(
                parent,
                width=width,
                fg_color=pal["bg"],
                text_color=pal["fg"],
                border_color=pal["border"],
                placeholder_text=placeholder,
                placeholder_text_color=pal["fg_muted"],
            )
            return _track(e)

        def _btn(parent, text, command, *, width=160, primary=False):
            b = ctk.CTkButton(
                parent,
                text=text,
                command=command,
                width=width,
                font=("Segoe UI", 9, "bold" if primary else "normal"),
                fg_color=pal["accent"] if primary else pal["accent_dim"],
                hover_color=pal["accent_hover"],
                text_color=pal["btn_text"],
                border_color=pal["border"],
                border_width=1,
            )
            return _track(b)

        title_row = ctk.CTkFrame(outer, fg_color="transparent")
        _track(title_row)
        title_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        _label(title_row, "PLATE SOLVE — local astrometry.net", bold=True, size=12).pack(
            side=tk.LEFT
        )
        self._plate_preset_label = _label(
            title_row, self._format_preset_summary(), dim=True, size=9
        )
        self._plate_preset_label.pack(side=tk.RIGHT)

        # ── Equipment preset section ─────────────────────────────────────
        preset_box = _section_frame(outer)
        preset_box.pack(fill=tk.X, padx=8, pady=4)
        _label(preset_box, "EQUIPMENT PRESET", bold=True).pack(anchor="w", padx=8, pady=(6, 2))

        preset_grid = ctk.CTkFrame(preset_box, fg_color="transparent")
        _track(preset_grid)
        preset_grid.pack(fill=tk.X, padx=8, pady=(0, 4))

        # Telescope row.
        scope_row = ctk.CTkFrame(preset_grid, fg_color="transparent")
        _track(scope_row)
        scope_row.pack(fill=tk.X, pady=2)
        _label(scope_row, "Telescope", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._preset_scope_label_entry = _entry(scope_row, width=170)
        self._preset_scope_label_entry.pack(side=tk.LEFT, padx=2)
        _label(scope_row, "Focal length (mm)", size=9).pack(side=tk.LEFT, padx=(8, 4))
        self._preset_focal_entry = _entry(scope_row, width=70)
        self._preset_focal_entry.pack(side=tk.LEFT, padx=2)

        # Camera row.
        cam_row = ctk.CTkFrame(preset_grid, fg_color="transparent")
        _track(cam_row)
        cam_row.pack(fill=tk.X, pady=2)
        _label(cam_row, "Camera", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._preset_cam_label_entry = _entry(cam_row, width=170)
        self._preset_cam_label_entry.pack(side=tk.LEFT, padx=2)
        _label(cam_row, "Pixel (μm)", size=9).pack(side=tk.LEFT, padx=(8, 4))
        self._preset_pixel_entry = _entry(cam_row, width=60)
        self._preset_pixel_entry.pack(side=tk.LEFT, padx=2)

        # Sensor row.
        sensor_row = ctk.CTkFrame(preset_grid, fg_color="transparent")
        _track(sensor_row)
        sensor_row.pack(fill=tk.X, pady=2)
        _label(sensor_row, "Sensor (px)", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._preset_sensor_w_entry = _entry(sensor_row, width=70)
        self._preset_sensor_w_entry.pack(side=tk.LEFT, padx=2)
        _label(sensor_row, "×", size=9).pack(side=tk.LEFT, padx=2)
        self._preset_sensor_h_entry = _entry(sensor_row, width=70)
        self._preset_sensor_h_entry.pack(side=tk.LEFT, padx=2)
        _btn(sensor_row, "Auto-fill from camera", self._preset_autofill_from_camera, width=170).pack(
            side=tk.LEFT, padx=8
        )

        # Site label (read-only summary; full site editor remains under Site / Sky).
        site_row = ctk.CTkFrame(preset_grid, fg_color="transparent")
        _track(site_row)
        site_row.pack(fill=tk.X, pady=2)
        _label(site_row, "Site", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._preset_site_label = _label(
            site_row, self._format_site_line(), dim=True, size=9
        )
        self._preset_site_label.pack(side=tk.LEFT, padx=4)

        # Computed plate scale & FOV summary.
        calc_row = ctk.CTkFrame(preset_grid, fg_color="transparent")
        _track(calc_row)
        calc_row.pack(fill=tk.X, pady=(2, 6))
        self._preset_scale_label = _label(
            calc_row, "Plate scale: — arcsec/px   |   FOV: — × — °", dim=True, size=9
        )
        self._preset_scale_label.pack(side=tk.LEFT)

        # Preset buttons row — telescope quick-select (three options) + save / recompute.
        preset_btn_row = ctk.CTkFrame(preset_box, fg_color="transparent")
        _track(preset_btn_row)
        preset_btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        _label(preset_btn_row, "Telescope preset", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._plate_telescope_combo = _track(
            ctk.CTkComboBox(
                preset_btn_row,
                values=list(PLATE_SOLVE_TELESCOPE_MENU_LABELS),
                width=220,
                state="readonly",
                fg_color=pal["bg"],
                border_color=pal["border"],
                button_color=pal["accent_dim"],
                button_hover_color=pal["accent_hover"],
                text_color=pal["fg"],
                dropdown_fg_color=pal["bg_alt"],
                dropdown_hover_color=pal["accent_dim"],
                dropdown_text_color=pal["fg"],
            )
        )
        self._plate_telescope_combo.set(PLATE_SOLVE_TELESCOPE_MENU_LABELS[0])
        self._plate_telescope_combo.pack(side=tk.LEFT, padx=(0, 6))
        _btn(
            preset_btn_row,
            "Apply telescope",
            self._apply_selected_plate_telescope_preset,
            primary=True,
            width=130,
        ).pack(side=tk.LEFT, padx=(0, 6))
        _btn(preset_btn_row, "Save preset", self._save_preset_from_ui, width=110).pack(
            side=tk.LEFT, padx=2
        )
        _btn(preset_btn_row, "Recompute FOV", self._recompute_preset_metrics, width=130).pack(
            side=tk.LEFT, padx=2
        )

        # Re-run preview metrics whenever any input changes.
        for w in (
            self._preset_focal_entry,
            self._preset_pixel_entry,
            self._preset_sensor_w_entry,
            self._preset_sensor_h_entry,
        ):
            w.bind("<KeyRelease>", lambda _e: self._recompute_preset_metrics())
            w.bind("<FocusOut>", lambda _e: self._recompute_preset_metrics())

        # ── Image input ──────────────────────────────────────────────────
        image_box = _section_frame(outer)
        image_box.pack(fill=tk.X, padx=8, pady=4)
        _label(image_box, "IMAGE", bold=True).pack(anchor="w", padx=8, pady=(6, 2))
        path_row = ctk.CTkFrame(image_box, fg_color="transparent")
        _track(path_row)
        path_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._plate_path_entry = _entry(
            path_row, width=380, placeholder="Path to FITS / PNG / JPG image"
        )
        self._plate_path_entry.pack(side=tk.LEFT, padx=(0, 6))
        _btn(path_row, "Browse…", self._browse_plate_fits, width=90).pack(side=tk.LEFT, padx=2)
        _btn(path_row, "Use last capture", self._session_use_last_for_plate, width=130).pack(
            side=tk.LEFT, padx=2
        )

        # ── Target / hints ───────────────────────────────────────────────
        target_box = _section_frame(outer)
        target_box.pack(fill=tk.X, padx=8, pady=4)
        _label(target_box, "TARGET HINTS", bold=True).pack(anchor="w", padx=8, pady=(6, 2))

        # Target name + resolve.
        name_row = ctk.CTkFrame(target_box, fg_color="transparent")
        _track(name_row)
        name_row.pack(fill=tk.X, padx=8, pady=2)
        _label(name_row, "Target", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._plate_target_entry = _entry(
            name_row, width=200, placeholder="e.g. M42, Polaris, Jupiter"
        )
        self._plate_target_entry.pack(side=tk.LEFT, padx=2)
        self._plate_target_entry.bind("<Return>", lambda _e: self._resolve_target_into_radec())
        _btn(name_row, "Resolve → RA/Dec", self._resolve_target_into_radec, width=160).pack(
            side=tk.LEFT, padx=6
        )

        # RA / Dec row.
        coord_row = ctk.CTkFrame(target_box, fg_color="transparent")
        _track(coord_row)
        coord_row.pack(fill=tk.X, padx=8, pady=2)
        _label(coord_row, "RA", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._plate_ra_hint = _entry(
            coord_row, width=130, placeholder="deg or HH:MM:SS"
        )
        self._plate_ra_hint.pack(side=tk.LEFT, padx=2)
        _label(coord_row, "Dec", size=9).pack(side=tk.LEFT, padx=(8, 4))
        self._plate_dec_hint = _entry(
            coord_row, width=130, placeholder="deg or ±DD:MM:SS"
        )
        self._plate_dec_hint.pack(side=tk.LEFT, padx=2)

        # FOV row + auto-fill.
        fov_row = ctk.CTkFrame(target_box, fg_color="transparent")
        _track(fov_row)
        fov_row.pack(fill=tk.X, padx=8, pady=2)
        _label(fov_row, "FOV°", size=9).pack(side=tk.LEFT, padx=(0, 4))
        self._plate_fov = _entry(fov_row, width=80, placeholder="auto")
        self._plate_fov.pack(side=tk.LEFT, padx=2)
        _btn(fov_row, "Auto FOV from preset", self._fill_fov_from_preset, width=170).pack(
            side=tk.LEFT, padx=6
        )

        # Helper buttons.
        helper_row = ctk.CTkFrame(target_box, fg_color="transparent")
        _track(helper_row)
        helper_row.pack(fill=tk.X, padx=8, pady=(2, 8))
        _btn(helper_row, "Use mount RA/Dec", self._use_mount_radec_for_plate, width=150).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        _btn(helper_row, "Use last solve", self._use_last_solve_for_plate, width=130).pack(
            side=tk.LEFT, padx=4
        )
        _btn(helper_row, "Clear", self._clear_plate_target, width=80).pack(side=tk.LEFT, padx=4)

        # ── Solve / output ───────────────────────────────────────────────
        action_row = ctk.CTkFrame(outer, fg_color="transparent")
        _track(action_row)
        action_row.pack(fill=tk.X, padx=8, pady=(2, 4))
        self._plate_solve_btn = _btn(
            action_row,
            "SOLVE",
            self._run_embedded_plate_solve,
            primary=True,
            width=200,
        )
        self._plate_solve_btn.pack(side=tk.LEFT, padx=(0, 6))
        _btn(
            action_row,
            "Download indexes for FOV",
            self._download_recommended_astrometry_indexes,
            width=240,
        ).pack(side=tk.LEFT, padx=4)
        self._plate_solve_status = tk.StringVar(value="Idle")
        self._plate_solve_status_lbl = _label(action_row, "Status: Idle", dim=True, size=9)
        self._plate_solve_status_lbl.configure(textvariable=self._plate_solve_status)
        self._plate_solve_status_lbl.pack(side=tk.LEFT, padx=(10, 0))

        out_box = _section_frame(outer)
        out_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))
        _label(out_box, "OUTPUT", bold=True).pack(anchor="w", padx=8, pady=(6, 2))
        self._plate_out = tk.Text(
            out_box,
            height=8,
            width=48,
            font=("Courier New", 9),
            wrap=tk.WORD,
            bg=pal["bg"],
            fg=pal["fg"],
            insertbackground=pal["fg"],
            relief=tk.FLAT,
            bd=1,
            highlightthickness=1,
            highlightbackground=pal["border"],
            highlightcolor=pal["accent"],
        )
        self._plate_widgets.append(self._plate_out)
        self._plate_out.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Push current preset into the UI and recompute live readouts.
        self._apply_astro_preset_to_ui()
        self._recompute_preset_metrics()

    # ── Preset helpers ────────────────────────────────────────────────────
    def _format_site_line(self) -> str:
        site = self.astro_preset.get("site", {}) or {}
        label = site.get("label") or "Site"
        lat = site.get("lat", self.site_lat)
        lon = site.get("lon", self.site_lon)
        elev = site.get("elev_m", self.site_elev_m)
        try:
            return f"{label}  •  {float(lat):.4f}°, {float(lon):.4f}°, {float(elev):.0f} m"
        except Exception:
            return f"{label}"

    def _format_preset_summary(self) -> str:
        scope = self.astro_preset.get("telescope", {}) or {}
        cam = self.astro_preset.get("camera", {}) or {}
        try:
            f_mm = float(scope.get("focal_length_mm", 0.0))
        except Exception:
            f_mm = 0.0
        scope_lbl = scope.get("label", "scope")
        cam_lbl = cam.get("label", "camera")
        return f"{scope_lbl} @ {f_mm:.0f} mm  •  {cam_lbl}"

    def _apply_astro_preset_to_ui(self):
        """Mirror self.astro_preset into the plate-tab entry widgets."""
        if not getattr(self, "_preset_focal_entry", None):
            return
        scope = self.astro_preset.get("telescope", {}) or {}
        cam = self.astro_preset.get("camera", {}) or {}
        for entry, val in (
            (self._preset_scope_label_entry, scope.get("label", "")),
            (self._preset_focal_entry, scope.get("focal_length_mm", "")),
            (self._preset_cam_label_entry, cam.get("label", "")),
            (self._preset_pixel_entry, cam.get("pixel_size_um", "")),
            (self._preset_sensor_w_entry, cam.get("sensor_w_px", "")),
            (self._preset_sensor_h_entry, cam.get("sensor_h_px", "")),
        ):
            try:
                entry.delete(0, "end")
                entry.insert(0, str(val))
            except Exception:
                pass
        if getattr(self, "_preset_site_label", None):
            try:
                self._preset_site_label.configure(text=self._format_site_line())
            except Exception:
                pass
        if getattr(self, "_plate_preset_label", None):
            try:
                self._plate_preset_label.configure(text=self._format_preset_summary())
            except Exception:
                pass
        self._sync_plate_telescope_combo_from_preset()

    def _sync_plate_telescope_combo_from_preset(self):
        """Align the plate-solve telescope combobox with the saved focal length."""
        combo = getattr(self, "_plate_telescope_combo", None)
        if combo is None:
            return
        scope = self.astro_preset.get("telescope", {}) or {}
        try:
            f_mm = float(scope.get("focal_length_mm", 0) or 0)
        except Exception:
            f_mm = 0.0
        sct_f = float(DEFAULT_ASTRO_PRESET["telescope"]["focal_length_mm"])
        labels = PLATE_SOLVE_TELESCOPE_MENU_LABELS
        if abs(f_mm - sct_f) < 0.5:
            combo.set(labels[0])
        elif abs(f_mm - PLATE_SOLVE_REFRACTOR_4IN_FOCAL_MM) < 0.5:
            combo.set(labels[1])
        else:
            combo.set(labels[2])

    def _apply_selected_plate_telescope_preset(self):
        """Apply the combobox choice: bundled SCT, 4\" / 660 mm refractor, or custom hint."""
        sel = self._plate_telescope_combo.get()
        labels = PLATE_SOLVE_TELESCOPE_MENU_LABELS
        if sel == labels[0]:
            self._apply_default_tucson_preset()
        elif sel == labels[1]:
            self.astro_preset.setdefault("telescope", {}).update(
                {
                    "label": PLATE_SOLVE_REFRACTOR_4IN_LABEL,
                    "focal_length_mm": float(PLATE_SOLVE_REFRACTOR_4IN_FOCAL_MM),
                }
            )
            self._save_astro_preset()
            self._apply_astro_preset_to_ui()
            self._recompute_preset_metrics()
            self._fill_fov_from_preset()
            self.update_status(
                'Applied 4" refractor preset (660 mm focal length, 4" aperture).'
            )
        else:
            messagebox.showinfo(
                "Different telescope",
                'Enter the telescope name and focal length (mm) in the fields above, then '
                'use "Save preset" and "Auto FOV from preset" as needed.',
            )

    def _read_preset_inputs(self) -> dict:
        """Pull current preset inputs back from the UI into a dict."""
        def _f(entry, default=0.0):
            try:
                return float(entry.get().strip())
            except Exception:
                return float(default)

        return {
            "scope_label": self._preset_scope_label_entry.get().strip()
            or DEFAULT_ASTRO_PRESET["telescope"]["label"],
            "focal_length_mm": _f(self._preset_focal_entry, DEFAULT_ASTRO_PRESET["telescope"]["focal_length_mm"]),
            "cam_label": self._preset_cam_label_entry.get().strip() or "camera",
            "pixel_size_um": _f(self._preset_pixel_entry, DEFAULT_ASTRO_PRESET["camera"]["pixel_size_um"]),
            "sensor_w_px": _f(self._preset_sensor_w_entry, DEFAULT_ASTRO_PRESET["camera"]["sensor_w_px"]),
            "sensor_h_px": _f(self._preset_sensor_h_entry, DEFAULT_ASTRO_PRESET["camera"]["sensor_h_px"]),
        }

    def _recompute_preset_metrics(self):
        """Update the plate-scale / FOV readout from the current entry values."""
        if not getattr(self, "_preset_scale_label", None):
            return
        try:
            data = self._read_preset_inputs()
            arcsec = _compute_plate_scale_arcsec_per_px(
                data["pixel_size_um"], data["focal_length_mm"]
            )
            fov_long, fov_short = _compute_fov_deg(
                data["pixel_size_um"],
                data["focal_length_mm"],
                data["sensor_w_px"],
                data["sensor_h_px"],
            )
            if arcsec > 0:
                txt = (
                    f"Plate scale: {arcsec:.2f} arcsec/px   |   "
                    f"FOV: {fov_long:.2f}° × {fov_short:.2f}°"
                )
            else:
                txt = "Plate scale: —   |   FOV: —"
            self._preset_scale_label.configure(text=txt)
            if getattr(self, "_plate_preset_label", None):
                self._plate_preset_label.configure(text=self._format_preset_summary())
        except Exception:
            pass

    def _apply_default_tucson_preset(self):
        """Reset the equipment preset to the bundled Tucson SCT @ 2032 mm default."""
        self.astro_preset = json.loads(json.dumps(DEFAULT_ASTRO_PRESET))
        # Site mirrors the preset for consistency with site_settings.json.
        site = self.astro_preset["site"]
        self.site_lat = float(site["lat"])
        self.site_lon = float(site["lon"])
        self.site_elev_m = float(site["elev_m"])
        self._save_site_settings()
        self._save_astro_preset()
        # Refresh the Site / Sky tab entries if they exist.
        for entry, val in (
            (getattr(self, "_site_lat_entry", None), self.site_lat),
            (getattr(self, "_site_lon_entry", None), self.site_lon),
            (getattr(self, "_site_elev_entry", None), self.site_elev_m),
        ):
            if entry is not None:
                try:
                    entry.delete(0, "end")
                    entry.insert(0, str(val))
                except Exception:
                    pass
        self._apply_astro_preset_to_ui()
        self._recompute_preset_metrics()
        self._fill_fov_from_preset()
        self.update_status("Loaded Tucson SCT preset (2032 mm).")

    def _save_preset_from_ui(self):
        """Persist the equipment preset using whatever the user has typed."""
        try:
            data = self._read_preset_inputs()
        except Exception as e:
            messagebox.showwarning("Preset", f"Could not read preset values: {e}")
            return
        self.astro_preset["telescope"] = {
            "label": data["scope_label"],
            "focal_length_mm": float(data["focal_length_mm"]),
        }
        self.astro_preset["camera"] = {
            "label": data["cam_label"],
            "pixel_size_um": float(data["pixel_size_um"]),
            "sensor_w_px": int(round(data["sensor_w_px"])),
            "sensor_h_px": int(round(data["sensor_h_px"])),
        }
        self.astro_preset.setdefault("site", {}).update(
            {
                "label": self.astro_preset.get("site", {}).get("label", "Tucson, AZ"),
                "lat": self.site_lat,
                "lon": self.site_lon,
                "elev_m": self.site_elev_m,
            }
        )
        self._save_astro_preset()
        self._recompute_preset_metrics()
        self._sync_plate_telescope_combo_from_preset()
        self.update_status(f"Saved preset to {self.astro_preset_path.name}")

    def _preset_autofill_from_camera(self):
        """Read pixel size + sensor size from the connected ZWO camera."""
        info = getattr(self, "camera_info", None)
        if not info:
            messagebox.showinfo(
                "Auto-fill from camera",
                "Connect to a camera first; the preset will then read pixel size and sensor "
                "dimensions directly from the ZWO SDK.",
            )
            return
        try:
            pixel = float(info.get("PixelSize", 0.0))
            mw = int(info.get("MaxWidth", 0))
            mh = int(info.get("MaxHeight", 0))
            name = str(info.get("Name", "ZWO Camera"))
        except Exception as e:
            messagebox.showerror("Auto-fill from camera", f"Could not read camera info:\n{e}")
            return
        if pixel <= 0 or mw <= 0 or mh <= 0:
            messagebox.showwarning(
                "Auto-fill from camera",
                "Camera reported invalid pixel size or sensor dimensions.",
            )
            return
        self._preset_cam_label_entry.delete(0, "end")
        self._preset_cam_label_entry.insert(0, name)
        self._preset_pixel_entry.delete(0, "end")
        self._preset_pixel_entry.insert(0, f"{pixel:g}")
        self._preset_sensor_w_entry.delete(0, "end")
        self._preset_sensor_w_entry.insert(0, str(mw))
        self._preset_sensor_h_entry.delete(0, "end")
        self._preset_sensor_h_entry.insert(0, str(mh))
        self._recompute_preset_metrics()
        self.update_status(f"Camera preset filled from {name}.")

    def _fill_fov_from_preset(self):
        """Populate the FOV° entry from the current preset (long axis)."""
        try:
            data = self._read_preset_inputs()
        except Exception:
            return
        fov_long, _ = _compute_fov_deg(
            data["pixel_size_um"],
            data["focal_length_mm"],
            data["sensor_w_px"],
            data["sensor_h_px"],
        )
        if fov_long > 0:
            self._plate_fov.delete(0, "end")
            self._plate_fov.insert(0, f"{fov_long:.3f}")

    # ── Target / coordinate helpers ───────────────────────────────────────
    def _resolve_target_into_radec(self):
        """Resolve the target name into RA/Dec and fill the hint entries."""
        name = (self._plate_target_entry.get() or "").strip()
        if not name:
            messagebox.showinfo(
                "Resolve target",
                "Enter an object name (e.g. M42, Polaris, Jupiter) and try again.",
            )
            return
        coords = _resolve_target_name(name)
        if coords is None:
            messagebox.showwarning(
                "Resolve target",
                f"Could not resolve {name!r} from the local catalogs.\n"
                "Try a Messier ID (M1–M110), a planet/Sun/Moon, or one of the bundled "
                "alignment stars (Vega, Sirius, Polaris, …).",
            )
            return
        ra, dec = coords
        self._plate_ra_hint.delete(0, "end")
        self._plate_ra_hint.insert(0, _deg_to_hms(ra))
        self._plate_dec_hint.delete(0, "end")
        self._plate_dec_hint.insert(0, _deg_to_dms(dec))
        if not self._plate_fov.get().strip():
            self._fill_fov_from_preset()
        self._append_plate_log(
            f"Resolved {name}: RA {_deg_to_hms(ra)}  ({ra:.5f}°)   "
            f"Dec {_deg_to_dms(dec)}  ({dec:.5f}°)\n"
        )

    def _use_mount_radec_for_plate(self):
        """Pull current RA/Dec from the connected mount, if any."""
        mp = getattr(self, "mount_panel", None)
        mount = getattr(mp, "mount", None) if mp else None
        if not mount or not getattr(mount, "connected", False):
            messagebox.showinfo(
                "Mount RA/Dec",
                "No mount connected. Connect on the right-hand panel and start polling first.",
            )
            return
        try:
            ra_deg, dec_deg, _pier = mount.get_current_radec()
        except Exception as e:
            messagebox.showerror("Mount RA/Dec", f"Could not read mount position:\n{e}")
            return
        self._plate_ra_hint.delete(0, "end")
        self._plate_ra_hint.insert(0, _deg_to_hms(float(ra_deg)))
        self._plate_dec_hint.delete(0, "end")
        self._plate_dec_hint.insert(0, _deg_to_dms(float(dec_deg)))
        if not self._plate_fov.get().strip():
            self._fill_fov_from_preset()
        self._append_plate_log(
            f"Mount: RA {_deg_to_hms(float(ra_deg))}  Dec {_deg_to_dms(float(dec_deg))}\n"
        )

    def _use_last_solve_for_plate(self):
        if self._last_solve_ra_deg is None or self._last_solve_dec_deg is None:
            messagebox.showinfo(
                "Last solve", "No previous plate solve in this session."
            )
            return
        self._plate_ra_hint.delete(0, "end")
        self._plate_ra_hint.insert(0, _deg_to_hms(self._last_solve_ra_deg))
        self._plate_dec_hint.delete(0, "end")
        self._plate_dec_hint.insert(0, _deg_to_dms(self._last_solve_dec_deg))
        self._append_plate_log("Reused last solved center.\n")

    def _clear_plate_target(self):
        for entry in (
            self._plate_target_entry,
            self._plate_ra_hint,
            self._plate_dec_hint,
            self._plate_fov,
        ):
            try:
                entry.delete(0, "end")
            except Exception:
                pass

    def _append_plate_log(self, text: str):
        out = getattr(self, "_plate_out", None)
        if out is None:
            return
        try:
            out.insert(tk.END, text)
            out.see(tk.END)
        except Exception:
            pass

    def _build_mount_column(self):
        t = self.theme
        self.side_tools.configure(bg=t["bg_primary"])
        self.mount_panel = MountControlsFrame(
            self.side_tools,
            standalone=False,
            tracking_prerequisite=self._mount_tracking_prerequisite,
            camgui_theme=self.theme,
        )
        self.mount_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)
        self._build_polar_align_panel()

    def _build_polar_align_panel(self):
        """Red-mode right panel: displays plate-solve based polar alignment guidance."""
        pal = PLATE_RED_PAL
        self._xy_polar = self._mk_xy(self.side_tools, pal["bg"])
        self._polar_align_shell = self._xy_polar.outer
        panel = ctk.CTkFrame(
            self._xy_polar.inner,
            fg_color=pal["bg"],
            border_color=pal["border"],
            border_width=1,
            corner_radius=8,
        )
        title = ctk.CTkLabel(
            panel,
            text="POLAR ALIGN MODE",
            font=("Segoe UI", 14, "bold"),
            text_color=pal["fg"],
        )
        title.pack(anchor="w", padx=12, pady=(10, 2))
        subtitle = ctk.CTkLabel(
            panel,
            text="Rough align, plate solve, then adjust mount from this guidance.",
            font=("Segoe UI", 10),
            text_color=pal["fg_dim"],
            justify="left",
            anchor="w",
        )
        subtitle.pack(fill=tk.X, padx=12, pady=(0, 8))
        fov_row = ctk.CTkFrame(panel, fg_color="transparent")
        fov_row.pack(fill=tk.X, padx=12, pady=(0, 8))
        ctk.CTkLabel(
            fov_row,
            text="FOV",
            font=("Segoe UI", 9, "bold"),
            text_color=pal["fg"],
        ).pack(side=tk.LEFT, padx=(0, 6))
        self._polar_align_fov_label = ctk.CTkLabel(
            fov_row,
            text=f"{self._polar_align_fov_deg:.1f}°",
            font=("Segoe UI", 9),
            text_color=pal["fg_dim"],
        )
        self._polar_align_fov_label.pack(side=tk.RIGHT)
        self._polar_align_fov_slider = ctk.CTkSlider(
            fov_row,
            from_=0.1,
            to=10.0,
            number_of_steps=99,
            button_color=pal["accent"],
            button_hover_color=pal["accent_hover"],
            progress_color=pal["accent"],
            fg_color=pal["bg_alt"],
            command=self._on_polar_align_fov_change,
        )
        self._polar_align_fov_slider.set(self._polar_align_fov_deg)
        self._polar_align_fov_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        chart_frame = ctk.CTkFrame(panel, fg_color=pal["bg_alt"], border_color=pal["border"], border_width=1)
        chart_frame.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 8))
        if _HAS_MATPLOTLIB:
            self._polar_align_fig, self._polar_align_ax = plt.subplots(figsize=(4.4, 4.4), dpi=100)
            self._polar_align_canvas = FigureCanvasTkAgg(self._polar_align_fig, master=chart_frame)
            self._polar_align_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self._polar_align_fig = None
            self._polar_align_ax = None
            self._polar_align_canvas = None
            ctk.CTkLabel(
                chart_frame,
                text="Polar diagram unavailable (install matplotlib).",
                font=("Segoe UI", 10),
                text_color=pal["fg_dim"],
            ).pack(fill=tk.X, padx=10, pady=12)
        self._polar_align_text = tk.Text(
            panel,
            height=14,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg=pal["bg_alt"],
            fg=pal["fg"],
            insertbackground=pal["fg"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=pal["border"],
            highlightcolor=pal["accent"],
            padx=8,
            pady=8,
        )
        self._polar_align_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._polar_align_panel = panel
        self._refresh_polar_align_panel()

    def _toggle_polar_align_mode(self):
        self._set_polar_align_mode(not self._polar_align_mode)

    def _set_polar_align_mode(self, enabled: bool):
        self._polar_align_mode = enabled
        if enabled:
            if self.mount_panel is not None:
                self.mount_panel.pack_forget()
            if self._polar_align_shell is not None:
                self._polar_align_shell.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)
            self._refresh_polar_align_panel()
            self.update_status("Polar Align Mode enabled.")
        else:
            if self._polar_align_shell is not None:
                self._polar_align_shell.pack_forget()
            if self.mount_panel is not None:
                self.mount_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)
            self.update_status("Mount controls restored.")
        self.apply_theme()

    def _refresh_polar_align_panel(self):
        self._redraw_polar_align_plot()
        out = self._polar_align_text
        if out is None:
            return
        out.delete("1.0", tk.END)
        out.insert(
            tk.END,
            "Workflow:\n"
            "1) Roughly polar align the mount.\n"
            "2) Run a plate solve in the center tab.\n"
            "3) Use the solved offset guidance below to adjust altitude/azimuth.\n\n",
        )
        if self._last_solve_ra_deg is None or self._last_solve_dec_deg is None:
            out.insert(tk.END, "No solved field yet. Plate solve a frame first.\n")
            return
        out.insert(
            tk.END,
            f"Last solved center: RA {self._last_solve_ra_deg:.5f}°  "
            f"Dec {self._last_solve_dec_deg:.5f}°\n\n",
        )
        try:
            txt = polar_align_mvp_text(
                self.site_lat,
                self.site_lon,
                self._last_solve_ra_deg,
                self._last_solve_dec_deg,
                elev_m=self.site_elev_m,
            )
            out.insert(tk.END, txt + "\n")
        except Exception as exc:
            out.insert(tk.END, f"Could not compute polar guidance: {exc}\n")

    def _on_polar_align_fov_change(self, val: float) -> None:
        self._polar_align_fov_deg = float(val)
        lbl = getattr(self, "_polar_align_fov_label", None)
        if lbl is not None:
            lbl.configure(text=f"{self._polar_align_fov_deg:.1f}°")
        self._redraw_polar_align_plot()

    def _local_sidereal_time_hours(self, longitude_deg: float) -> float:
        dt = datetime.utcnow()
        year, month, day = dt.year, dt.month, dt.day
        hour = dt.hour + dt.minute / 60 + dt.second / 3600
        if month <= 2:
            year -= 1
            month += 12
        a = math.floor(year / 100)
        b = 2 - a + math.floor(a / 4)
        jd = (
            math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + day
            + b
            - 1524.5
            + hour / 24.0
        )
        t = (jd - 2451545.0) / 36525.0
        gmst = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t**2
            - t**3 / 38710000.0
        )
        return ((gmst + longitude_deg) % 360.0) / 15.0

    @staticmethod
    def _polar_project(ra_hours: float, dec_deg: float, lst_hours: float, pole_sign: int = 1) -> tuple[float, float]:
        co_lat = 90.0 - pole_sign * dec_deg
        co_lat_rad = math.radians(co_lat)
        r = 2.0 * math.tan(co_lat_rad / 2.0)
        ha_deg = (lst_hours - ra_hours) * 15.0
        angle = math.pi / 2.0 + pole_sign * math.radians(ha_deg)
        return (r * math.cos(angle), r * math.sin(angle))

    @staticmethod
    def _polar_edge_radius(fov_deg: float) -> float:
        return 2.0 * math.tan(math.radians(fov_deg) / 2.0)

    def _redraw_polar_align_plot(self) -> None:
        if (
            not _HAS_MATPLOTLIB
            or self._polar_align_fig is None
            or self._polar_align_ax is None
            or self._polar_align_canvas is None
            or mpatches is None
        ):
            return
        pal = PLATE_RED_PAL
        ax = self._polar_align_ax
        fig = self._polar_align_fig
        fig.patch.set_facecolor(pal["bg"])
        ax.clear()
        ax.set_facecolor("#0d0503")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(pal["border"])
        fov_deg = max(0.1, min(10.0, float(getattr(self, "_polar_align_fov_deg", 5.0))))
        r_edge = self._polar_edge_radius(fov_deg)
        ax.set_xlim(-r_edge * 1.05, r_edge * 1.05)
        ax.set_ylim(-r_edge * 1.05, r_edge * 1.05)

        # Match PolarAlignRed look: dense declination rings with periodic labels.
        dec_step = 1.0
        for i in range(1, int(fov_deg / dec_step) + 1):
            co_lat = i * dec_step
            if co_lat > fov_deg:
                break
            r = self._polar_edge_radius(co_lat)
            major = (i % 2) == 0
            circle = mpatches.Circle(
                (0, 0),
                r,
                fill=False,
                edgecolor="#7a3030",
                linewidth=0.8 if major else 0.45,
                alpha=0.62 if major else 0.35,
            )
            ax.add_patch(circle)

        lst = self._local_sidereal_time_hours(self.site_lon)
        # RA hour spokes + edge labels (angle markers from original chart style).
        for h in range(24):
            hx, hy = self._polar_project(h, 90.0 - fov_deg, lst, pole_sign=1)
            major = (h % 6) == 0
            ax.plot(
                [0, hx],
                [0, hy],
                color="#7a3030",
                linewidth=0.75 if major else 0.4,
                alpha=0.58 if major else 0.32,
            )
            tx, ty = self._polar_project(h, 90.0 - fov_deg * 0.93, lst, pole_sign=1)
            ax.text(
                tx,
                ty,
                f"{h}h",
                color=pal["fg"],
                fontsize=6.3,
                alpha=0.92,
                ha="center",
                va="center",
                family="monospace",
            )

        # Ring labels (deg from pole), similar to PolarAlignRed.
        for deg in (2, 4, 6, 8):
            if deg > fov_deg:
                continue
            lx, ly = self._polar_project(lst, 90.0 - deg, lst, pole_sign=1)
            ax.text(
                lx + r_edge * 0.02,
                ly - r_edge * 0.02,
                f"{deg:.0f}°",
                color=pal["fg_dim"],
                fontsize=6.0,
                alpha=0.8,
                family="monospace",
            )

        polaris_ra_h = 37.95456067 / 15.0
        polaris_dec = 89.26410897
        px, py = self._polar_project(polaris_ra_h, polaris_dec, lst, pole_sign=1)
        ax.plot(px, py, "o", color="#ff7777", markersize=11, alpha=0.25)
        ax.plot(px, py, "o", color="#ffcccc", markersize=5.5)
        ax.text(px + r_edge * 0.03, py + r_edge * 0.02, "Polaris", color="#ffcccc", fontsize=8)
        ax.plot(0, 0, "o", color=pal["fg"], markersize=4)
        mx, my = self._polar_project(lst, 90.0 - fov_deg, lst, pole_sign=1)
        ax.plot([0, mx], [0, my], color=pal["accent"], linewidth=1.4, alpha=0.78)
        ax.text(mx, my + r_edge * 0.02, "Meridian", color=pal["accent"], fontsize=8, ha="center", fontweight="bold")
        history = getattr(self, "_polar_align_history", [])
        if history:
            total = len(history)
            for i, (hra, hdec) in enumerate(history):
                hx, hy = self._polar_project(hra / 15.0, hdec, lst, pole_sign=1)
                if not math.isfinite(hx) or not math.isfinite(hy):
                    continue
                alpha = 0.25 + 0.55 * ((i + 1) / max(1, total))
                size = 2.5 + 1.5 * ((i + 1) / max(1, total))
                ax.plot(hx, hy, "o", color="#ff7777", markersize=size, alpha=alpha)
            if len(history) >= 2:
                xs = []
                ys = []
                for hra, hdec in history:
                    hx, hy = self._polar_project(hra / 15.0, hdec, lst, pole_sign=1)
                    if math.isfinite(hx) and math.isfinite(hy):
                        xs.append(hx)
                        ys.append(hy)
                if len(xs) >= 2:
                    ax.plot(xs, ys, color="#ff6666", linewidth=0.9, alpha=0.45)
        if self._last_solve_ra_deg is not None and self._last_solve_dec_deg is not None:
            sx, sy = self._polar_project(self._last_solve_ra_deg / 15.0, self._last_solve_dec_deg, lst, pole_sign=1)
            ax.plot(sx, sy, "o", color="#ff4444", markersize=8, alpha=0.35)
            ax.plot(sx, sy, "o", color="#ff6666", markersize=4)
            ax.text(sx + r_edge * 0.03, sy - r_edge * 0.03, "Solved center", color="#ff6666", fontsize=8)
            ax.plot([sx, 0], [sy, 0], color="#ff6666", linewidth=0.8, alpha=0.7, linestyle="--")
        try:
            self._polar_align_canvas.draw_idle()
        except Exception:
            pass

    # ── Busy indicator + async helpers ─────────────────────────────────────
    def _begin_busy(self, message: str = "Working…"):
        """Show the spinner with ``message``. Safe to call from the Tk thread.

        Reference-counted so nested async calls keep the spinner up until the
        last one finishes.
        """
        self._busy_count += 1
        self._busy_messages.append(message)
        try:
            self.busy_text_label.configure(text=message)
        except Exception:
            pass
        if self._busy_anim_after_id is None:
            self._busy_animate_tick()

    def _end_busy(self):
        """Decrement busy ref count; hide the spinner when it reaches zero."""
        if self._busy_count <= 0:
            self._busy_count = 0
            self._busy_messages.clear()
            self._busy_hide()
            return
        self._busy_count -= 1
        if self._busy_messages:
            self._busy_messages.pop()
        if self._busy_count == 0:
            self._busy_hide()
        else:
            try:
                self.busy_text_label.configure(text=self._busy_messages[-1])
            except Exception:
                pass

    def _busy_hide(self):
        if self._busy_anim_after_id is not None:
            try:
                self.root.after_cancel(self._busy_anim_after_id)
            except Exception:
                pass
            self._busy_anim_after_id = None
        try:
            self.busy_spinner_label.configure(text="")
            self.busy_text_label.configure(text="")
        except Exception:
            pass

    def _busy_animate_tick(self):
        if self._busy_count <= 0:
            self._busy_anim_after_id = None
            return
        try:
            ch = self._busy_anim_chars[self._busy_anim_index % len(self._busy_anim_chars)]
            self.busy_spinner_label.configure(text=ch)
        except Exception:
            pass
        self._busy_anim_index += 1
        # 80ms cadence is smooth without flooding the event loop.
        self._busy_anim_after_id = self.root.after(80, self._busy_animate_tick)

    def _run_async(self, work, *, busy_message: str = "Working…", on_done=None, on_error=None):
        """Run ``work()`` on a daemon thread; show the spinner; marshal the
        result back to the Tk main thread.

        ``work`` must be safe to run off the Tk thread. ``on_done(result)`` and
        ``on_error(exc)`` are optional callbacks; both run on the Tk thread
        after the spinner is dismissed.
        """
        self._begin_busy(busy_message)

        def _runner():
            try:
                result = work()
            except Exception as exc:
                def _fail(e=exc):
                    self._end_busy()
                    if on_error is not None:
                        try:
                            on_error(e)
                        except Exception:
                            pass
                    else:
                        messagebox.showerror("Error", str(e))
                self.root.after(0, _fail)
                return

            def _done(r=result):
                self._end_busy()
                if on_done is not None:
                    try:
                        on_done(r)
                    except Exception as cb_exc:
                        messagebox.showerror("Error", str(cb_exc))
            self.root.after(0, _done)

        threading.Thread(target=_runner, daemon=True).start()

    def _set_initial_pane_sashes(self):
        """Place sashes near 1/4–1/3 camera vs rest, and ~60/40 center vs mount."""
        # tk.PanedWindow uses sash_place / sash_coord. ttk.PanedWindow uses
        # sashpos. We are on tk.PanedWindow here, but hasattr() can lie on
        # some Tk builds (Tk auto-generates command stubs), so try the modern
        # API first and fall back without ever raising into the user's UI.
        def _place(pw, x):
            try:
                pw.sashpos(0, x)
                return
            except (AttributeError, tk.TclError):
                pass
            try:
                _, y = pw.sash_coord(0)
            except (AttributeError, tk.TclError):
                y = 1
            try:
                pw.sash_place(0, x, y)
            except (AttributeError, tk.TclError):
                pass

        try:
            self.root.update_idletasks()
            ow = max(self._pan_outer.winfo_width(), 800)
            _place(self._pan_outer, int(ow * 0.26))
            iw = max(self._pan_inner.winfo_width(), 600)
            _place(self._pan_inner, int(iw * 0.58))
        except tk.TclError:
            pass

    def _mount_tracking_prerequisite(self):
        """Mount panel calls this before enabling tracking; requires a successful plate solve."""
        if self._last_solve_ra_deg is None or self._last_solve_dec_deg is None:
            return (
                False,
                "Turn on mount tracking only after a successful plate solve.\n\n"
                "Open the Plate solve tab, choose a FITS file, enter RA°, Dec°, and FOV° hints, "
                "then use Submit photo for plate solve. When the solve finishes, you can enable "
                "tracking on the mount.",
            )
        return (True, "")

    @staticmethod
    def _local_time_daytime_solar_window():
        """True when local clock is between 05:00 and 20:00 inclusive (solar filter reminder window)."""
        t = datetime.now().time()
        return dt_time(5, 0) <= t <= dt_time(20, 0)

    def _maybe_warn_solar_filter_daytime(self, activity: str):
        if not self._local_time_daytime_solar_window():
            return
        messagebox.showwarning(
            "Solar filter",
            f"Local time is between 5:00 and 20:00. Before {activity}, make sure a proper "
            "solar filter is installed if you will observe or image the Sun.\n\n"
            "Never point an unfiltered telescope or camera at the Sun.",
            parent=self.root,
        )

    def _focus_center_tab(self, name: str):
        try:
            self._center_tabs.set(name)
        except Exception:
            pass

    def _session_line_for_record(self, rec: dict) -> str:
        ts = rec.get("captured_at", "")
        try:
            dt = datetime.fromisoformat(ts)
            t_disp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            t_disp = ts[:19] if ts else "?"
        ra, dec = rec.get("ra_deg"), rec.get("dec_deg")
        if ra is not None and dec is not None:
            rad = f"RA {ra:.4f}°"
            dd = f"Dec {dec:.4f}°"
        else:
            rad, dd = "RA —", "Dec —"
        base = os.path.basename(rec.get("fits_path", ""))
        return f"{t_disp}  {rad}  {dd}  {base}"

    def _refresh_session_photos_listbox(self):
        if not getattr(self, "_session_listbox", None):
            return
        self._session_listbox.delete(0, tk.END)
        for rec in self._session_photos:
            self._session_listbox.insert(tk.END, self._session_line_for_record(rec))

    def _backfill_session_radec_from_disk(self):
        """Fill RA/Dec from astrometry outputs (*-astra*.fits) next to session captures."""
        updated = 0
        for rec in self._session_photos:
            if rec.get("ra_deg") is not None:
                continue
            fp = rec.get("fits_path")
            if not fp or not os.path.isfile(fp):
                continue
            p = Path(fp)
            for cand in sorted(p.parent.glob(p.stem + "-astra*.fits")):
                try:
                    cra, cdec = read_solved_fits_center_radec_deg(str(cand))
                    rec["ra_deg"] = cra
                    rec["dec_deg"] = cdec
                    updated += 1
                    break
                except Exception:
                    continue
        self._refresh_session_photos_listbox()
        if updated:
            self.update_status(f"Updated RA/Dec for {updated} session photo(s) from disk.")
        else:
            self.update_status("No new solved FITS found for session photos.")

    def _session_use_selected_for_plate(self):
        sel = self._session_listbox.curselection()
        if not sel:
            messagebox.showinfo("Session photos", "Select a row first.")
            return
        i = int(sel[0])
        if i < 0 or i >= len(self._session_photos):
            return
        path = self._session_photos[i].get("fits_path", "")
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Session photos", "Selected file is missing on disk.")
            return
        self._plate_path_entry.delete(0, "end")
        self._plate_path_entry.insert(0, path)
        self._focus_center_tab("Plate solve")
        self.update_status("FITS path copied — open Plate solve")

    def _session_use_last_for_plate(self):
        if not self._session_photos:
            messagebox.showinfo("Session photos", "No captures in this session yet.")
            return
        path = self._session_photos[-1].get("fits_path", "")
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Session photos", "Last capture file is missing on disk.")
            return
        self._plate_path_entry.delete(0, "end")
        self._plate_path_entry.insert(0, path)
        self._focus_center_tab("Plate solve")
        self.update_status("Last capture copied — open Plate solve")

    def _update_session_photo_solved(self, fits_input_path: str, ra_deg: float, dec_deg: float):
        """Attach solved coordinates to a session list entry when paths match."""
        try:
            inp = str(Path(fits_input_path).resolve())
        except Exception:
            inp = os.path.abspath(fits_input_path)
        for rec in self._session_photos:
            try:
                rp = str(Path(rec["fits_path"]).resolve())
            except Exception:
                rp = os.path.abspath(rec.get("fits_path", ""))
            if rp == inp:
                rec["ra_deg"] = ra_deg
                rec["dec_deg"] = dec_deg
                break
        self._refresh_session_photos_listbox()

    def _find_solved_fits_for_capture(self, fits_input_path: str) -> str | None:
        """Find a solved astrometry FITS file generated from a capture FITS."""
        if not fits_input_path:
            return None
        p = Path(fits_input_path)
        if not p.parent.exists():
            return None
        candidates = sorted(p.parent.glob(p.stem + "-astra*.fits"))
        for cand in candidates:
            if cand.is_file():
                return str(cand)
        return None

    def _generate_session_pdf_report(self):
        """Build a printable PDF report for the current session captures."""
        if not self._session_photos:
            messagebox.showinfo("Session report", "No captures in this session yet.")
            return
        if not _HAS_MATPLOTLIB or PdfPages is None:
            messagebox.showwarning("Session report", "Matplotlib PDF backend is unavailable.")
            return

        default_name = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        save_path = filedialog.asksaveasfilename(
            title="Save session PDF report",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF", "*.pdf")],
        )
        if not save_path:
            return

        photos = list(self._session_photos)
        self.update_status("Building session PDF report...")

        try:
            with PdfPages(save_path) as pdf:
                # Cover page
                fig = plt.figure(figsize=(8.5, 11))
                fig.patch.set_facecolor("white")
                fig.text(0.08, 0.93, "ASTRA Session Report", fontsize=22, fontweight="bold")
                fig.text(0.08, 0.885, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
                fig.text(0.08, 0.85, f"Site: lat {self.site_lat:.4f}, lon {self.site_lon:.4f}, elev {self.site_elev_m:.0f} m", fontsize=10)
                fig.text(0.08, 0.82, f"Total captures: {len(photos)}", fontsize=10)
                fig.text(
                    0.08,
                    0.77,
                    "Includes all captured images.\nPlate-solved captures include ASTRA annotation overlays when available.",
                    fontsize=10,
                )
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                page_items = []
                for idx, rec in enumerate(photos, start=1):
                    fits_path = rec.get("fits_path", "")
                    if not fits_path or not os.path.isfile(fits_path):
                        continue

                    solved_fits = self._find_solved_fits_for_capture(fits_path)
                    annotated_path = str(Path(fits_path).with_name(Path(fits_path).stem + "-astra-facts.png"))
                    is_solved = solved_fits is not None

                    if is_solved and not os.path.isfile(annotated_path):
                        try:
                            self._annotate_plate_solve_with_offline_facts(
                                solved_fits_path=solved_fits,
                                input_image_path=fits_path,
                                field_center_ra_deg=rec.get("ra_deg"),
                                field_center_dec_deg=rec.get("dec_deg"),
                                log_to_plate=False,
                            )
                        except Exception:
                            pass

                    show_path = annotated_path if (is_solved and os.path.isfile(annotated_path)) else fits_path
                    _gray, img_pil = self._read_image_for_plate_annotations(show_path)
                    if img_pil is None:
                        continue
                    img_arr = np.asarray(img_pil)

                    cap_ts = rec.get("captured_at", "") or "unknown"
                    header = f"Capture {idx}/{len(photos)}"
                    details = [f"File: {os.path.basename(fits_path)}", f"Captured: {cap_ts}"]
                    if rec.get("ra_deg") is not None and rec.get("dec_deg") is not None:
                        details.append(f"Solved center: RA {rec['ra_deg']:.5f}°  Dec {rec['dec_deg']:.5f}°")
                    else:
                        details.append("Solved center: not available")
                    details.append(
                        "Annotations: included"
                        if (is_solved and os.path.isfile(annotated_path))
                        else ("Annotations: pending" if is_solved else "Annotations: not applicable")
                    )
                    page_items.append((img_arr, header, "\n".join(details)))

                for start in range(0, len(page_items), 2):
                    chunk = page_items[start:start + 2]
                    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
                    fig.patch.set_facecolor("white")
                    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.32)
                    if not isinstance(axes, np.ndarray):
                        axes = np.array([axes])
                    for i, ax in enumerate(axes):
                        if i >= len(chunk):
                            ax.axis("off")
                            continue
                        img_arr, header, detail_text = chunk[i]
                        ax.imshow(img_arr, cmap="gray")
                        ax.set_axis_off()
                        y_text = 1.02
                        ax.text(
                            0.0,
                            y_text + 0.06,
                            header,
                            transform=ax.transAxes,
                            fontsize=11,
                            fontweight="bold",
                            va="bottom",
                            ha="left",
                        )
                        ax.text(
                            0.0,
                            y_text,
                            detail_text,
                            transform=ax.transAxes,
                            fontsize=8.5,
                            va="bottom",
                            ha="left",
                        )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            self.update_status(f"Session report saved: {save_path}")
            messagebox.showinfo("Session report", f"PDF report created:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Session report", f"Could not generate report:\n{e}")

    def _apply_site_from_ui(self):
        try:
            self.site_lat = float(self._site_lat_entry.get().strip())
            self.site_lon = float(self._site_lon_entry.get().strip())
            self.site_elev_m = float(self._site_elev_entry.get().strip())
            self._save_site_settings()
            messagebox.showinfo("Site", "Site saved.")
        except ValueError:
            messagebox.showwarning("Site", "Enter numeric latitude, longitude, and elevation (m).")

    def _read_gps_fill_site(self):
        port = self._gps_port_menu.get()
        if not port or port == "(none)":
            messagebox.showwarning("GPS", "Select a serial port.")
            return

        def worker():
            r = try_read_lat_lon_nmea(port)
            if not r:
                self.root.after(0, lambda: messagebox.showwarning("GPS", "No fix read from port."))
                return
            lat, lon = r
            self.site_lat, self.site_lon = lat, lon
            self.root.after(0, self._fill_site_entries_from_session)

        threading.Thread(target=worker, daemon=True).start()

    def _fill_site_entries_from_session(self):
        self._site_lat_entry.delete(0, "end")
        self._site_lat_entry.insert(0, str(self.site_lat))
        self._site_lon_entry.delete(0, "end")
        self._site_lon_entry.insert(0, str(self.site_lon))
        self._save_site_settings()
        messagebox.showinfo("GPS", "Latitude / longitude updated from GPS.")

    def _refresh_body_table(self):
        if self._last_solve_ra_deg is None:
            self._sky_log.delete("1.0", tk.END)
            self._sky_log.insert(tk.END, "Run a successful plate solve first (Plate solve center tab).\n")
            return
        rows = major_bodies_separation_table(
            self._last_solve_ra_deg,
            self._last_solve_dec_deg,
            self.site_lat,
            self.site_lon,
            self.site_elev_m,
        )
        self._sky_log.delete("1.0", tk.END)
        self._sky_log.insert(
            tk.END,
            f"Field center RA {self._last_solve_ra_deg:.5f}°  Dec {self._last_solve_dec_deg:.5f}°\n\n",
        )
        for name, sep, up, alt, az in rows:
            vis = "visible" if up else "below horizon"
            self._sky_log.insert(
                tk.END,
                f"{name:12}  {sep:6.2f}° from center  {vis:14}  alt {alt:+6.1f}°  az {az:6.1f}°\n",
            )

    def _show_polar_hints_text(self):
        if self._last_solve_ra_deg is None:
            messagebox.showinfo("Polar hints", "Plate solve a field first.")
            return
        txt = polar_align_mvp_text(
            self.site_lat,
            self.site_lon,
            self._last_solve_ra_deg,
            self._last_solve_dec_deg,
            elev_m=self.site_elev_m,
        )
        self._sky_log.delete("1.0", tk.END)
        self._sky_log.insert(tk.END, txt + "\n")
        self._refresh_polar_align_panel()

    def _browse_plate_fits(self):
        p = filedialog.askopenfilename(
            title="Image for plate solving",
            filetypes=[
                ("Solver image", "*.fits *.fit *.fts *.png *.jpg *.jpeg *.tif *.tiff"),
                ("FITS", "*.fits *.fit *.fts"),
                ("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All", "*.*"),
            ],
        )
        if p:
            self._plate_path_entry.delete(0, "end")
            self._plate_path_entry.insert(0, p)

    def _run_embedded_plate_solve(self):
        path = self._plate_path_entry.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Plate solve", "Choose a valid FITS or image file.")
            return
        self._last_plate_solve_input_path = os.path.abspath(path)

        ra_raw = self._plate_ra_hint.get().strip()
        dec_raw = self._plate_dec_hint.get().strip()
        fov_raw = self._plate_fov.get().strip()
        try:
            ra = _parse_ra_input(ra_raw) if ra_raw else None
            dec = _parse_dec_input(dec_raw) if dec_raw else None
        except Exception:
            messagebox.showwarning(
                "Plate solve",
                "RA/Dec must be either decimal degrees or HMS/DMS "
                "(e.g. 05:35:17 or +22:00:52).",
            )
            return
        if (ra is None) != (dec is None):
            messagebox.showwarning(
                "Plate solve",
                "Enter both RA and Dec hints, or leave both blank for blind solve.",
            )
            return

        # FOV: explicit value wins; otherwise compute from the equipment preset.
        fov = None
        if fov_raw:
            try:
                fov = float(fov_raw)
            except ValueError:
                messagebox.showwarning("Plate solve", "FOV must be a number in degrees.")
                return
        if fov is None:
            try:
                data = self._read_preset_inputs()
                fov_long, _ = _compute_fov_deg(
                    data["pixel_size_um"],
                    data["focal_length_mm"],
                    data["sensor_w_px"],
                    data["sensor_h_px"],
                )
                if fov_long > 0:
                    fov = fov_long
                    self._plate_fov.delete(0, "end")
                    self._plate_fov.insert(0, f"{fov:.3f}")
            except Exception:
                pass
        if fov is None or fov <= 0:
            messagebox.showwarning(
                "Plate solve",
                "Could not determine FOV. Enter a value in degrees, or fill the equipment preset.",
            )
            return

        if ra is not None and dec is not None:
            self._plate_out.insert(
                tk.END,
                f"Solving with hints: RA {_deg_to_hms(ra)} ({ra:.4f}°)  "
                f"Dec {_deg_to_dms(dec)} ({dec:.4f}°)  FOV {fov:.3f}°\n"
                "(multi-pass with auto astrometry setup and PNG fallback; this may take a while)\n",
            )
        else:
            self._plate_out.insert(
                tk.END,
                f"Blind solving (no RA/Dec hints)  FOV {fov:.3f}°\n"
                "(this can take longer than hinted solve)\n"
                "(multi-pass with auto astrometry setup and PNG fallback; this may take a while)\n",
            )
        self._plate_out.see(tk.END)
        self._set_plate_solve_busy(True, "Solving...")

        def worker():
            base = Path(path).stem + "-astra"
            try:
                res = run_solve_field_local(path, ra, dec, fov, out_basename=base, timeout=400)
            except Exception as e:
                self.root.after(0, lambda err=str(e): self._plate_solve_done(None, err))
                return
            self.root.after(0, lambda: self._plate_solve_done(res, None))

        threading.Thread(target=worker, daemon=True).start()

    def _set_plate_solve_busy(self, busy: bool, msg: str = ""):
        btn = getattr(self, "_plate_solve_btn", None)
        status_var = getattr(self, "_plate_solve_status", None)
        if btn is not None:
            try:
                btn.configure(state="disabled" if busy else "normal")
            except Exception:
                pass
        if status_var is not None:
            status_var.set(f"Status: {msg}" if msg else "Status: Idle")

    @staticmethod
    def _recommended_index_filenames(fov_deg: float | None = None):
        """Return 4200-series index filenames sized for the given FOV (degrees).

        Per astrometry.net docs, a good rule is to install indexes whose skymark
        diameters fall in 10%-100% of the image FOV. Scales 0-4 are split into 48
        healpix tiles, 5-7 into 12 tiles, and 8+ are single files.
        """
        scales = _scales_recommended_for_fov(fov_deg if fov_deg and fov_deg > 0 else 1.0)
        files = []
        for s in scales:
            family = f"42{s:02d}"
            if s <= 4:
                tiles = 48
            elif s <= 7:
                tiles = 12
            else:
                tiles = 0  # single file
            if tiles == 0:
                files.append(f"index-{family}.fits")
            else:
                for i in range(tiles):
                    files.append(f"index-{family}-{i:02d}.fits")
        return files

    def _download_recommended_astrometry_indexes(self):
        target_dir = Path.home() / "astrometry" / "data"
        base_url = "https://data.astrometry.net/4200"
        # Prefer the FOV currently entered in the UI (or computed from preset)
        # so we fetch the right scales for the user's optics.
        fov_for_picker = None
        try:
            raw = self._plate_fov.get().strip()
            if raw:
                fov_for_picker = float(raw)
        except Exception:
            fov_for_picker = None
        if fov_for_picker is None:
            try:
                data = self._read_preset_inputs()
                fov_long, _ = _compute_fov_deg(
                    data["pixel_size_um"],
                    data["focal_length_mm"],
                    data["sensor_w_px"],
                    data["sensor_h_px"],
                )
                if fov_long > 0:
                    fov_for_picker = fov_long
            except Exception:
                pass
        wanted = self._recommended_index_filenames(fov_for_picker)
        # Skip files that already exist locally so the count reflects what we'll actually fetch.
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        to_fetch = [
            n for n in wanted
            if not (target_dir / n).exists() or (target_dir / n).stat().st_size <= 1024 * 1024
        ]
        if fov_for_picker:
            self._plate_out.insert(
                tk.END,
                f"FOV {fov_for_picker:.3f}° -> selecting "
                f"{len(wanted)} index file(s); {len(to_fetch)} need download.\n",
            )
        else:
            self._plate_out.insert(
                tk.END,
                "No FOV available, falling back to medium-field index set.\n",
            )
        if not to_fetch:
            self._plate_out.insert(tk.END, "Nothing to download; all recommended indexes are present.\n")
            self._plate_out.see(tk.END)
            return
        # Indexes 4200-4204 are split into 48 healpix tiles and individual files
        # can be 25-450 MB each, so a full set may run into many GB.
        if len(to_fetch) > 50:
            ok = messagebox.askyesno(
                "Plate solve",
                f"This will download {len(to_fetch)} index files from "
                "data.astrometry.net. The smaller-scale families (4200-4204) "
                "are split into 48 healpix tiles and the total can exceed "
                "several GB. Continue?",
            )
            if not ok:
                self._plate_out.insert(tk.END, "Download cancelled by user.\n")
                self._plate_out.see(tk.END)
                return
        wanted = to_fetch
        self._plate_out.insert(tk.END, f"Preparing index download into {target_dir}\n")
        self._plate_out.see(tk.END)

        def worker():
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.root.after(0, lambda: self._plate_out.insert(tk.END, f"Could not create {target_dir}: {e}\n"))
                return

            downloaded = 0
            skipped = 0
            failed = 0
            for name in wanted:
                dest = target_dir / name
                if dest.exists() and dest.stat().st_size > 1024 * 1024:
                    skipped += 1
                    self.root.after(0, lambda n=name: self._plate_out.insert(tk.END, f"Have {n} (skip)\n"))
                    continue
                url = f"{base_url}/{name}"
                self.root.after(0, lambda n=name: self._plate_out.insert(tk.END, f"Downloading {n}...\n"))
                try:
                    bytes_written = 0
                    with urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            bytes_written += len(chunk)
                    if bytes_written < 1024 * 1024:
                        raise RuntimeError(f"download too small ({bytes_written} bytes)")
                    downloaded += 1
                    self.root.after(0, lambda n=name: self._plate_out.insert(tk.END, f"OK {n}\n"))
                except Exception as e:
                    try:
                        if dest.exists():
                            dest.unlink()
                    except Exception:
                        pass
                    failed += 1
                    self.root.after(0, lambda n=name, err=e: self._plate_out.insert(tk.END, f"Failed {n}: {err}\n"))

            self.root.after(
                0,
                lambda: self._plate_out.insert(
                    tk.END,
                    f"Index download complete. Downloaded={downloaded}, skipped={skipped}, failed={failed}\n",
                ),
            )
            self.root.after(0, self._plate_out.see, tk.END)

        threading.Thread(target=worker, daemon=True).start()

    def _plate_solve_done(self, res, err):
        self._set_plate_solve_busy(False, "Idle")
        if err:
            self._plate_out.insert(tk.END, f"Error: {err}\n")
            return
        setup = res.get("setup") or {}
        if setup.get("messages"):
            self._plate_out.insert(tk.END, "\n".join(setup["messages"]) + "\n")
        sf = res.get("solved_fits")
        if not res.get("success") or not sf:
            stderr_text = res.get("stderr", "") or ""
            tail = stderr_text[-2000:] if stderr_text else ""
            self._plate_out.insert(tk.END, "Plate solve did not produce a WCS.\n")
            if tail:
                self._plate_out.insert(tk.END, tail.rstrip() + "\n")
            try:
                fov_raw = self._plate_fov.get().strip()
                fov_val = float(fov_raw) if fov_raw else 0.0
            except Exception:
                fov_val = 0.0
            try:
                data_dir = Path(setup.get("data_dir") or (Path.home() / "astrometry" / "data"))
                installed = _installed_index_scales(data_dir)
                wanted = set(_scales_recommended_for_fov(fov_val))
                missing = sorted(wanted - installed)
                if missing:
                    ranges = dict((s, (a, b)) for s, a, b in _index_scale_ranges_arcmin())
                    miss_str = ", ".join(
                        f"42{s:02d} ({ranges[s][0]:g}-{ranges[s][1]:g}')" for s in missing
                    )
                    self._plate_out.insert(
                        tk.END,
                        "Hint: missing recommended index families for this FOV: "
                        f"{miss_str}.\n"
                        "Click \"Download indexes for FOV\" to fetch them.\n",
                    )
            except Exception:
                pass
            self._plate_out.see(tk.END)
            return
        self._plate_out.insert(tk.END, f"Solved FITS: {sf}\n")
        if sf and os.path.isfile(sf):
            try:
                cra, cdec = read_solved_fits_center_radec_deg(sf)
                self._last_solve_ra_deg = cra
                self._last_solve_dec_deg = cdec
                hist = getattr(self, "_polar_align_history", None)
                if hist is not None:
                    hist.append((cra, cdec))
                    # Keep the most recent solves so the trail stays readable.
                    if len(hist) > 25:
                        del hist[:-25]
                self._plate_out.insert(tk.END, f"Center RA {cra:.5f}°  Dec {cdec:.5f}°\n")
                self._refresh_polar_align_panel()
                inp = getattr(self, "_last_plate_solve_input_path", None)
                if inp:
                    self._update_session_photo_solved(inp, cra, cdec)
            except Exception as e:
                self._plate_out.insert(tk.END, f"Could not read center from WCS: {e}\n")
            try:
                self._annotate_plate_solve_with_offline_facts(
                    solved_fits_path=sf,
                    input_image_path=getattr(self, "_last_plate_solve_input_path", None),
                    field_center_ra_deg=getattr(self, "_last_solve_ra_deg", None),
                    field_center_dec_deg=getattr(self, "_last_solve_dec_deg", None),
                )
            except Exception as e:
                self._plate_out.insert(tk.END, f"Offline annotation skipped: {e}\n")
        self._plate_out.see(tk.END)

    @staticmethod
    def _ang_sep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
        r1 = math.radians(float(ra1_deg))
        d1 = math.radians(float(dec1_deg))
        r2 = math.radians(float(ra2_deg))
        d2 = math.radians(float(dec2_deg))
        cos_sep = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
        cos_sep = max(-1.0, min(1.0, cos_sep))
        return math.degrees(math.acos(cos_sep))

    @staticmethod
    def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        finite = np.isfinite(a)
        if not finite.any():
            return np.zeros_like(a, dtype=np.uint8)
        vals = a[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.5))
        if hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
            if hi <= lo:
                return np.zeros_like(a, dtype=np.uint8)
        scaled = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def _read_image_for_plate_annotations(self, image_path: str) -> tuple[np.ndarray | None, Image.Image | None]:
        if not image_path:
            return None, None
        p = Path(image_path)
        if not p.is_file():
            return None, None
        ext = p.suffix.lower()
        if ext in (".fits", ".fit", ".fts"):
            with fits.open(str(p)) as hdul:
                data = np.asarray(hdul[0].data)
            if data is None:
                return None, None
            if data.ndim > 2:
                data = np.squeeze(data)
                if data.ndim > 2:
                    data = data[0]
            gray = np.asarray(data, dtype=np.float32)
            disp8 = self._normalize_for_display(gray)
            pil = Image.fromarray(disp8, mode="L").convert("RGB")
            return gray, pil
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, None
        pil = Image.fromarray(gray).convert("RGB")
        return np.asarray(gray, dtype=np.float32), pil

    @staticmethod
    def _identify_star_from_offline_catalog(ra_deg: float, dec_deg: float) -> tuple[str, str, str]:
        nearest = None
        best_sep = float("inf")
        for row in OFFLINE_STAR_FACTS:
            sep = ZWOCameraGUI._ang_sep_deg(ra_deg, dec_deg, row["ra_deg"], row["dec_deg"])
            if sep < best_sep:
                best_sep = sep
                nearest = row
        # Curated facts: generous radius for coarse plate solutions vs. catalog positions.
        if nearest is not None and best_sep <= 0.55:
            return str(nearest["name"]), str(nearest["fact"]), str(nearest["source"])
        tycho = tycho_nearest_identification(ra_deg, dec_deg, max_sep_deg=0.11)
        if tycho is not None:
            return tycho[0], tycho[1], tycho[2]
        if nearest is not None and best_sep <= 1.0:
            return str(nearest["name"]), str(nearest["fact"]), str(nearest["source"])
        return "Unidentified field star", "No match in offline bright-star list or Tycho-2 within search radius.", "Unknown"

    def _extract_largest_star_sources(
        self, img_gray: np.ndarray, solved_wcs: WCS
    ) -> list[dict]:
        if img_gray is None or solved_wcs is None:
            return []
        data = np.asarray(img_gray, dtype=np.float32)
        if data.ndim != 2 or data.size == 0:
            return []
        try:
            bkg = sep.Background(data)
            data_sub = data - bkg
            thresh = max(2.5, 3.0 * float(bkg.globalrms))
            sources = sep.extract(data_sub, thresh)
        except Exception:
            return []
        if sources is None or len(sources) == 0:
            return []
        rows = sorted(
            [s for s in sources if np.isfinite(s["x"]) and np.isfinite(s["y"])],
            key=lambda s: (float(s["npix"]), float(s["flux"])),
            reverse=True,
        )
        picked = []
        for s in rows:
            if len(picked) >= 2:
                break
            x = float(s["x"])
            y = float(s["y"])
            try:
                ra_deg, dec_deg = solved_wcs.all_pix2world(x, y, 0)
            except Exception:
                continue
            star_name, fact, source = self._identify_star_from_offline_catalog(float(ra_deg), float(dec_deg))
            picked.append(
                {
                    "x": x,
                    "y": y,
                    "area_px": float(s["npix"]),
                    "flux": float(s["flux"]),
                    "ra_deg": float(ra_deg),
                    "dec_deg": float(dec_deg),
                    "name": star_name,
                    "fact": fact,
                    "catalog_source": source,
                }
            )
        return picked

    def _moon_region_fact_if_present(
        self,
        field_center_ra_deg: float | None,
        field_center_dec_deg: float | None,
        image_shape: tuple[int, int] | None,
    ) -> tuple[str, str] | None:
        if field_center_ra_deg is None or field_center_dec_deg is None:
            return None
        if not image_shape or image_shape[0] <= 0 or image_shape[1] <= 0:
            return None
        try:
            from sky_ephemeris import body_apparent_radec_deg
        except Exception:
            return None
        try:
            moon_ra, moon_dec = body_apparent_radec_deg(
                SOLAR_SYSTEM_BODIES["Moon"], self.site_lat, self.site_lon, self.site_elev_m
            )
        except Exception:
            return None
        sep_deg = self._ang_sep_deg(field_center_ra_deg, field_center_dec_deg, moon_ra, moon_dec)
        try:
            fov_long = float(self._plate_fov.get().strip())
        except Exception:
            fov_long = 0.0
        if fov_long <= 0:
            return None
        h_px, w_px = float(image_shape[0]), float(image_shape[1])
        if w_px <= 0 or h_px <= 0:
            return None
        if w_px >= h_px:
            fov_w, fov_h = fov_long, fov_long * (h_px / w_px)
        else:
            fov_h, fov_w = fov_long, fov_long * (w_px / h_px)
        frame_diag_half_deg = 0.5 * math.hypot(fov_w, fov_h)
        moon_radius_deg = 0.26
        if sep_deg > frame_diag_half_deg + moon_radius_deg:
            return None

        dra = (field_center_ra_deg - moon_ra) * math.cos(math.radians(moon_dec))
        ddec = field_center_dec_deg - moon_dec
        if sep_deg < moon_radius_deg * 0.35:
            key = "near_side_center"
            region = "near-side central maria"
        elif abs(dra) >= abs(ddec):
            if dra >= 0:
                key = "east_limb"
                region = "eastern lunar limb"
            else:
                key = "west_limb"
                region = "western lunar limb"
        else:
            if ddec >= 0:
                key = "north_limb"
                region = "northern lunar region"
            else:
                key = "south_limb"
                region = "southern lunar region"
        return region, OFFLINE_MOON_REGION_FACTS.get(key, "")

    def _annotate_plate_solve_with_offline_facts(
        self,
        solved_fits_path: str,
        input_image_path: str | None,
        field_center_ra_deg: float | None,
        field_center_dec_deg: float | None,
        log_to_plate: bool = True,
    ):
        if not solved_fits_path or not os.path.isfile(solved_fits_path):
            return
        with fits.open(solved_fits_path) as hdul:
            solved_wcs = WCS(hdul[0].header)

        img_gray, img_pil = self._read_image_for_plate_annotations(input_image_path or "")
        if img_gray is None or img_pil is None:
            if log_to_plate and getattr(self, "_plate_out", None):
                self._plate_out.insert(
                    tk.END,
                    "Offline annotations: could not load the original image for overlay rendering.\n",
                )
            return

        stars = self._extract_largest_star_sources(img_gray, solved_wcs)
        draw = ImageDraw.Draw(img_pil)
        for idx, star in enumerate(stars, start=1):
            x, y = float(star["x"]), float(star["y"])
            r = 14 if idx == 1 else 11
            color = (255, 92, 92) if idx == 1 else (255, 180, 120)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
            lbl = f"{idx}) {star['name']}"
            draw.text((x + r + 3, y - 10), lbl, fill=color)
            if log_to_plate and getattr(self, "_plate_out", None):
                self._plate_out.insert(
                    tk.END,
                    f"Star {idx}: {star['name']}  RA {star['ra_deg']:.5f}°  Dec {star['dec_deg']:.5f}°\n"
                    f"  Fact: {star['fact']}\n"
                    f"  Catalog: {star['catalog_source']}\n",
                )
        if not stars:
            if log_to_plate and getattr(self, "_plate_out", None):
                self._plate_out.insert(
                    tk.END,
                    "Offline annotations: no reliable stellar sources found for top-2 star facts.\n",
                )

        moon_info = self._moon_region_fact_if_present(
            field_center_ra_deg=field_center_ra_deg,
            field_center_dec_deg=field_center_dec_deg,
            image_shape=img_gray.shape[:2],
        )
        if moon_info:
            moon_region, moon_fact = moon_info
            draw.text((12, 12), f"Moon: {moon_region}", fill=(150, 220, 255))
            if log_to_plate and getattr(self, "_plate_out", None):
                self._plate_out.insert(
                    tk.END,
                    f"Moon region in frame: {moon_region}\n"
                    f"  Fact: {moon_fact}\n"
                    "  Catalog: regional lunar geology fact set (offline)\n",
                )

        base_path = Path(input_image_path) if input_image_path else Path(solved_fits_path)
        out_path = base_path.with_name(base_path.stem + "-astra-facts.png")
        try:
            img_pil.save(str(out_path))
            if log_to_plate and getattr(self, "_plate_out", None):
                self._plate_out.insert(tk.END, f"Annotated image saved: {out_path}\n")
                self._plate_out.insert(
                    tk.END,
                    "Offline data sources used: expanded bright-star fact list, Tycho-2 nearest match fallback, and local lunar region facts.\n",
                )
        except Exception as e:
            if log_to_plate and getattr(self, "_plate_out", None):
                self._plate_out.insert(tk.END, f"Could not save annotation image: {e}\n")

    def apply_theme(self):
        """Apply the current theme to all UI elements"""
        t = self.theme
        _rp = PLATE_RED_PAL
        try:
            st = ttk.Style()
            st.theme_use("clam")
            st.configure("TSeparator", background=t["border"])
            st.configure("Horizontal.TSeparator", background=t["border"])
        except Exception:
            pass

        def _descendant_ctk_buttons(root_widget):
            found = []
            stack = [root_widget]
            while stack:
                w = stack.pop()
                try:
                    kids = w.winfo_children()
                except Exception:
                    kids = []
                for child in kids:
                    stack.append(child)
                    if isinstance(child, ctk.CTkButton):
                        found.append(child)
            return found

        # Root and main containers
        self.root.configure(bg=t["bg_primary"])

        for frame in [
            self.top_bar,
            self.main,
            self.right,
            getattr(self, "side_tools", None),
            getattr(self, "_left_pane_wrap", None),
            getattr(self, "_pan_outer", None),
            getattr(self, "_pan_inner", None),
        ]:
            if frame is None:
                continue
            try:
                frame.configure(bg=t["bg_primary"])
            except Exception:
                pass

        # Bidirectional scroll regions (tk Canvas + scrollbars)
        _cam_scr = getattr(self, "_xy_camera_tab_scroll", None)
        _foc_scr = getattr(self, "_xy_focus_tab_scroll", None)
        _trough_main = t["bg_tertiary"]
        for a in getattr(self, "_xy_scroll_areas", []):
            if a is getattr(self, "_xy_plate", None) or a is getattr(self, "_xy_polar", None):
                continue
            if a is _cam_scr or a is _foc_scr:
                _sbg = t["bg_primary"] if self.night_mode else _rp["bg"]
                _str = t["bg_tertiary"] if self.night_mode else _rp["bg_alt"]
                theme_xy_area(a, _sbg, trough=_str)
            else:
                theme_xy_area(a, t["bg_primary"], trough=_trough_main)

        # Labels
        label_configs = [
            (self.title_label, t["bg_primary"], t["fg_primary"]),
            (self.temp_label, t["bg_primary"], t["fg_primary"]),
            (self.status_label, t["bg_primary"], t["fg_primary"]),
            (self.exposure_label, t["bg_primary"], t["fg_primary"]),
            (self.gain_label, t["bg_primary"], t["fg_primary"]),
            (self.preview_label, t["canvas_bg"], t["fg_tertiary"])
        ]

        # Busy indicator follows the theme. The spinner glyph picks up the
        # accent color so it pops against either the day or night palette.
        try:
            self.busy_frame.configure(bg=t["bg_primary"])
            self.busy_spinner_label.configure(bg=t["bg_primary"], fg=t["accent"])
            self.busy_text_label.configure(bg=t["bg_primary"], fg=t["fg_primary"])
        except Exception:
            pass

        for label, bg, fg in label_configs:
            label.configure(bg=bg, fg=fg)

        # Update all other labels in containers
        configured_labels = [lbl for lbl, _, _ in label_configs]
        for container in [self.left, self.right, self.top_bar]:
            try:
                children = container.winfo_children()
            except:
                children = []
            for widget in children:
                if isinstance(widget, tk.Label) and widget not in configured_labels:
                    try:
                        widget.configure(bg=t["bg_primary"], fg=t["fg_primary"])
                    except:
                        pass
                try:
                    for sub in widget.winfo_children():
                        if isinstance(sub, tk.Label) and sub not in configured_labels:
                            try:
                                sub.configure(bg=t["bg_primary"], fg=t["fg_primary"])
                            except:
                                pass
                except:
                    pass

        # Update all tk.Frame widgets inside left panel
        try:
            for widget in self.left.winfo_children():
                if isinstance(widget, tk.Frame):
                    try:
                        widget.configure(bg=t["bg_primary"])
                    except:
                        pass
        except:
            pass

        # Update CTkButtons
        action_buttons = [self.capture_img_btn, self.record_video_btn]
        small_buttons = [self.connect_btn, self.refresh_btn, self.browse_btn]

        btn_text_color = "#fccfcf" if self.night_mode else "white"
        for btn in action_buttons:
            btn.configure(fg_color=t["accent"], hover_color=t["accent_light"], text_color=btn_text_color)

        for btn in small_buttons:
            btn.configure(
                fg_color=t["bg_secondary"],
                text_color=t["fg_primary"],
                hover_color=t["bg_tertiary"],
                border_color=t["border"],
                border_width=1
            )

        # Keep Session photos and Site/Sky action buttons red in night mode.
        try:
            tab_buttons = []
            for tab_name in ("Session photos", "Site / Sky"):
                tab_buttons.extend(_descendant_ctk_buttons(self._center_tabs.tab(tab_name)))
            for btn in tab_buttons:
                if self.night_mode:
                    btn.configure(
                        fg_color=t["accent"],
                        hover_color=t["accent_light"],
                        text_color="#fccfcf",
                        border_color=t["border"],
                        border_width=1,
                    )
                else:
                    btn.configure(
                        fg_color=t["bg_secondary"],
                        hover_color=t["bg_tertiary"],
                        text_color=t["fg_primary"],
                        border_color=t["border"],
                        border_width=1,
                    )
        except Exception:
            pass

        # Path / project entries — match red/dark camera controls (no white fields in day mode).
        self.path_entry.configure(
            fg_color=t["bg_secondary"] if self.night_mode else _rp["bg"],
            text_color=t["fg_primary"] if self.night_mode else _rp["fg"],
            border_color=t["border"] if self.night_mode else _rp["border"],
        )

        self.project_name_entry.configure(
            fg_color=t["bg_secondary"] if self.night_mode else _rp["bg"],
            text_color=t["fg_primary"] if self.night_mode else _rp["fg"],
            placeholder_text_color=t["fg_tertiary"] if self.night_mode else _rp["fg_muted"],
            border_color=t["border"] if self.night_mode else _rp["border"],
        )

        _tab_sel = t["accent"] if self.night_mode else _rp["accent"]
        _tab_sel_h = t["accent_light"] if self.night_mode else _rp["accent_hover"]
        try:
            self._center_tabs.configure(
                fg_color=t["bg_primary"],
                segmented_button_fg_color=t["bg_tertiary"],
                segmented_button_selected_color=_tab_sel,
                segmented_button_selected_hover_color=_tab_sel_h,
                segmented_button_unselected_color=t["bg_secondary"],
                segmented_button_unselected_hover_color=t["bg_tertiary"],
                text_color=t["fg_primary"],
            )
        except Exception:
            pass

        try:
            if self.night_mode:
                self._camera_tabs.configure(
                    fg_color=t["bg_primary"],
                    segmented_button_fg_color=t["bg_tertiary"],
                    segmented_button_selected_color=_tab_sel,
                    segmented_button_selected_hover_color=_tab_sel_h,
                    segmented_button_unselected_color=t["bg_secondary"],
                    segmented_button_unselected_hover_color=t["bg_tertiary"],
                    text_color=t["fg_primary"],
                )
            else:
                self._camera_tabs.configure(
                    fg_color=_rp["bg"],
                    segmented_button_fg_color=_rp["bg_panel"],
                    segmented_button_selected_color=_rp["accent"],
                    segmented_button_selected_hover_color=_rp["accent_hover"],
                    segmented_button_unselected_color=_rp["bg_alt"],
                    segmented_button_unselected_hover_color=_rp["bg_panel"],
                    text_color=_rp["fg"],
                )
        except Exception:
            pass

        try:
            lb_bg = t["canvas_bg"] if not self.night_mode else t["bg_tertiary"]
            lb_fg = t["fg_primary"]
            self._session_listbox.configure(bg=lb_bg, fg=lb_fg, selectbackground=t["accent"])
        except Exception:
            pass

        # Focus assistant tab styling: keep the embedded plot, labels, and
        # listbox legible in both day and night modes. The capture/reset
        # buttons stay on the permanent red palette regardless of theme so the
        # observer keeps dark adaptation while focusing at the eyepiece.
        try:
            red_pal = PLATE_RED_PAL
            cap_btn = getattr(self, "focus_capture_btn", None)
            if cap_btn is not None:
                cap_btn.configure(
                    fg_color=red_pal["accent"],
                    hover_color=red_pal["accent_hover"],
                    text_color=red_pal["btn_text"],
                )
            reset_btn = getattr(self, "focus_reset_btn", None)
            if reset_btn is not None:
                reset_btn.configure(
                    fg_color=red_pal["accent_dim"],
                    hover_color=red_pal["accent"],
                    text_color=red_pal["btn_text"],
                )
            focus_lb = getattr(self, "focus_listbox", None)
            if focus_lb is not None:
                lb_bg = _rp["bg"] if not self.night_mode else t["bg_tertiary"]
                lb_fg = _rp["fg"] if not self.night_mode else t["fg_primary"]
                focus_lb.configure(
                    bg=lb_bg,
                    fg=lb_fg,
                    selectbackground=t["accent"],
                    selectforeground=("white" if not self.night_mode else t["bg_primary"]),
                    highlightbackground=t["border"] if self.night_mode else _rp["border"],
                )
            if getattr(self, "_focus_canvas", None) is not None:
                self._focus_redraw_plot()
        except Exception:
            pass

        is_live = getattr(self, "_livewcs_active", False)
        signal_color = t["accent"] if is_live else t["fg_tertiary"]
        signal_text = "Sky target" if is_live else "No sky target"
        self.skytrack_signal_dot.configure(bg=t["bg_primary"], fg=signal_color)
        self.skytrack_signal_label.configure(bg=t["bg_primary"], fg=signal_color, text=signal_text)
        for lbl in [self.skytrack_target_label, self.skytrack_ra_label, self.skytrack_dec_label]:
            lbl.configure(bg=t["bg_primary"], fg=t["fg_secondary"])

        if self.night_mode:
            self.theme_btn.configure(fg_color=t["accent"], text_color="white", hover_color=t["accent_light"])
        else:
            self.theme_btn.configure(fg_color=t["bg_tertiary"], text_color=t["fg_primary"], hover_color=t["accent"])
        try:
            pal = PLATE_RED_PAL
            if self._polar_align_mode:
                self.polar_align_mode_btn.configure(
                    fg_color=pal["accent"],
                    hover_color=pal["accent_hover"],
                    text_color=pal["btn_text"],
                    text="Motor Control",
                )
            else:
                self.polar_align_mode_btn.configure(
                    fg_color=pal["accent_dim"],
                    hover_color=pal["accent"],
                    text_color=pal["btn_text"],
                    text="Polar Align Mode",
                )
            if self._polar_align_panel is not None:
                self._polar_align_panel.configure(
                    fg_color=pal["bg"],
                    border_color=pal["border"],
                )
            theme_xy_area(getattr(self, "_xy_polar", None), pal["bg"], trough=pal["bg_alt"])
            if self._polar_align_text is not None:
                self._polar_align_text.configure(
                    bg=pal["bg_alt"],
                    fg=pal["fg"],
                    insertbackground=pal["fg"],
                    highlightbackground=pal["border"],
                    highlightcolor=pal["accent"],
                )
            if getattr(self, "_polar_align_fov_slider", None) is not None:
                self._polar_align_fov_slider.configure(**_RED_SLIDER)
        except Exception:
            pass

        combo_settings = {
            "fg_color": t["bg_secondary"] if self.night_mode else PLATE_RED_PAL["bg_alt"],
            "text_color": t["fg_primary"] if self.night_mode else PLATE_RED_PAL["fg"],
            "border_color": t["border"] if self.night_mode else PLATE_RED_PAL["border"],
            "button_color": PLATE_RED_PAL["accent"],
            "button_hover_color": PLATE_RED_PAL["accent_hover"],
            "dropdown_fg_color": t["bg_tertiary"] if self.night_mode else PLATE_RED_PAL["bg_panel"],
            "dropdown_hover_color": t["bg_secondary"] if self.night_mode else PLATE_RED_PAL["bg_alt"],
        }

        combo_list = [
            self.camera_dropdown,
            self.binning_dropdown,
            self.format_dropdown,
            self.capture_count_dropdown,
            self.exposure_range_dropdown,
            self.frame_type_dropdown,
        ]
        for combo in combo_list:
            combo.configure(**combo_settings)

        self.exposure_slider.configure(**_RED_SLIDER)
        self.gain_slider.configure(**_RED_SLIDER)

        mp = getattr(self, "mount_panel", None)
        if mp is not None:
            try:
                mp.apply_camgui_theme(t)
            except Exception:
                pass

        # Native tk frames/labels under the camera column (including Camera/Focus tab bodies):
        # default Tk grey backgrounds do not follow CTk theme.
        try:
            _theme_native_tk_under_camera_column(self.left, t["bg_primary"], t["fg_primary"])
            _cbg = t["bg_primary"] if self.night_mode else _rp["bg"]
            _cfg = t["fg_primary"] if self.night_mode else _rp["fg"]
            if getattr(self, "_camera_tab_inner", None):
                _theme_native_tk_under_camera_column(self._camera_tab_inner, _cbg, _cfg)
            if getattr(self, "_focus_tab_inner", None):
                _theme_native_tk_under_camera_column(self._focus_tab_inner, _cbg, _cfg)
        except Exception:
            pass

        # Plate-solve tab uses a permanent red palette so the field-night view
        # stays dark-adapted regardless of day/night mode.
        self._reapply_plate_red_palette()

    def _reapply_plate_red_palette(self):
        """Force the plate-solve tab back to its red palette after a theme change."""
        pal = PLATE_RED_PAL
        theme_xy_area(getattr(self, "_xy_plate", None), pal["bg"], trough=pal["bg_alt"])
        widgets = getattr(self, "_plate_widgets", None)
        if not widgets:
            return
        for w in widgets:
            try:
                if isinstance(w, ctk.CTkFrame):
                    # Preserve transparent inner frames.
                    cur = None
                    try:
                        cur = w.cget("fg_color")
                    except Exception:
                        cur = None
                    if cur != "transparent":
                        w.configure(fg_color=pal["bg_alt"], border_color=pal["border"])
                elif isinstance(w, ctk.CTkLabel):
                    # Bold/dim styling stays per-label; only refresh foreground.
                    cur_fg = None
                    try:
                        cur_fg = w.cget("text_color")
                    except Exception:
                        cur_fg = None
                    new_fg = pal["fg_dim"] if cur_fg == pal["fg_dim"] else pal["fg"]
                    w.configure(text_color=new_fg)
                elif isinstance(w, ctk.CTkEntry):
                    w.configure(
                        fg_color=pal["bg"],
                        text_color=pal["fg"],
                        border_color=pal["border"],
                        placeholder_text_color=pal["fg_muted"],
                    )
                elif isinstance(w, ctk.CTkButton):
                    cur = None
                    try:
                        cur = w.cget("fg_color")
                    except Exception:
                        cur = None
                    fg = pal["accent"] if cur == pal["accent"] else pal["accent_dim"]
                    w.configure(
                        fg_color=fg,
                        hover_color=pal["accent_hover"],
                        text_color=pal["btn_text"],
                        border_color=pal["border"],
                    )
                elif isinstance(w, tk.Text):
                    w.configure(
                        bg=pal["bg"],
                        fg=pal["fg"],
                        insertbackground=pal["fg"],
                        highlightbackground=pal["border"],
                        highlightcolor=pal["accent"],
                    )
            except Exception:
                pass

    def toggle_theme(self):
        self.night_mode = not self.night_mode
        self.theme = THEMES["night"] if self.night_mode else THEMES["day"]
        self.theme_btn.configure(text="🌙 Night Mode" if self.night_mode else "☀ Day Mode")
        self.apply_theme()

    def setup_camera(self):
        if not self.camera or not self.camera_initialized:
            self.update_status("No camera connected")
            return

        try:
            controls = self.camera.get_controls()
            camera_property = self.camera.get_camera_property()
            self.full_sensor_width = camera_property['MaxWidth']
            self.full_sensor_height = camera_property['MaxHeight']

            print(f"Full sensor dimensions: {self.full_sensor_width}x{self.full_sensor_height}")

            self.camera.set_control_value(asi.ASI_GAIN, self.gain_var.get())
            self.camera.set_control_value(asi.ASI_EXPOSURE, int(self.exposure_var.get() * 1000))
            self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, controls['BandWidth']['DefaultValue'])

            format_map = {"RAW8": asi.ASI_IMG_RAW8, "RAW16": asi.ASI_IMG_RAW16}
            selected_format = format_map.get(self.format_var.get(), asi.ASI_IMG_RAW16)
            self.camera.set_image_type(selected_format)

            self.update_status("Camera initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup camera: {e}")

    def refresh_cameras(self):
        if self.is_capturing:
            self.stop_preview()

        if self.camera:
            try:
                self.camera.close()
            except:
                pass
            self.camera = None
            self.camera_initialized = False
        self.available_cameras = []

        def work():
            num_cameras = asi.get_num_cameras()
            cams = _enumerate_asi_cameras() if num_cameras > 0 else []
            return num_cameras, cams

        def done(result):
            num_cameras, cams = result
            self.available_cameras = cams
            if num_cameras > 0 and cams:
                camera_options = [f"{idx}: {name}" for idx, name in cams]
                self.camera_dropdown.configure(values=camera_options)
                self.camera_dropdown.set(camera_options[0])
                self.update_status(f"Found {num_cameras} camera(s)")
            else:
                self.camera_dropdown.configure(values=["No cameras detected"])
                self.camera_dropdown.set("No cameras detected")
                self.update_status("No cameras found")

        def fail(exc):
            messagebox.showerror("Error", f"Failed to refresh cameras: {exc}")
            self.update_status("Error refreshing cameras")

        self._run_async(work, busy_message="Refreshing cameras…", on_done=done, on_error=fail)

    def connect_camera(self):
        if not self.available_cameras:
            messagebox.showwarning("Warning", "No cameras available to connect")
            return
        if self.is_capturing:
            self.stop_preview()
        if self.camera:
            try:
                self.camera.close()
            except:
                pass

        selection = self.camera_select_var.get()
        if selection == "No cameras detected":
            return
        try:
            camera_idx = int(selection.split(":")[0])
        except Exception:
            messagebox.showwarning("Warning", "Pick a camera from the list first.")
            return

        # Disable the connect button while we work so users don't click again.
        try:
            self.connect_btn.configure(state="disabled")
        except Exception:
            pass

        def work():
            cam = asi.Camera(camera_idx)
            info = cam.get_camera_property()
            return cam, info

        def done(result):
            cam, info = result
            self.camera = cam
            self.camera_info = info
            self.camera_initialized = True
            self.setup_camera()
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass
            self.update_status(f"Connected to {info['Name']}")
            messagebox.showinfo("Success", f"Connected to {info['Name']}")
            self.start_preview()

        def fail(exc):
            self.camera = None
            self.camera_initialized = False
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass
            messagebox.showerror("Error", f"Failed to connect to camera: {exc}")
            self.update_status("Failed to connect to camera")

        self._run_async(work, busy_message="Connecting to camera…", on_done=done, on_error=fail)

    def update_binning(self, choice=None):
        if not self.camera or not self.camera_initialized:
            return

        try:
            bin_text = self.binning_dropdown.get()
            bin_value = int(bin_text.split('x')[0])
        except Exception as e:
            self.update_status(f"Error setting binning: {e}")
            return

        was_capturing = self.is_capturing
        binned_width = self.full_sensor_width // bin_value
        binned_height = self.full_sensor_height // bin_value

        def work():
            if was_capturing:
                self.camera.stop_video_capture()
            self.camera.set_roi(
                start_x=0,
                start_y=0,
                width=binned_width,
                height=binned_height,
                bins=bin_value,
            )
            print(f"Set ROI: {binned_width}x{binned_height} with {bin_value}x{bin_value} binning")
            if was_capturing:
                self.camera.start_video_capture()
            return bin_value

        def done(_):
            self.update_status(f"Binning set to {bin_value}x{bin_value}")

        def fail(exc):
            self.update_status(f"Error setting binning: {exc}")
            import traceback
            traceback.print_exc()

        self._run_async(work, busy_message="Setting binning…", on_done=done, on_error=fail)

    def update_exposure_range(self, choice=None):
        """Update exposure slider range based on selected range."""
        range_text = self.exposure_range_var.get()

        range_map = {
            "32-1000us":     (32,   1000,  "us"),
            "1-100ms":       (1,    100,   "ms"),
            "100ms-1000ms":  (100,  1000,  "ms"),
            "1-10s":         (1,    10,    "s"),
            "10-60s":        (10,   60,    "s"),
        }

        if range_text not in range_map:
            return

        min_val, max_val, unit = range_map[range_text]

        self.exposure_slider.configure(from_=min_val, to=max_val)

        current = self.exposure_var.get()
        if current < min_val or current > max_val:
            self.exposure_var.set(min_val)
            self.exposure_slider.set(min_val)
            self.update_exposure(min_val)

    def update_exposure(self, value):
        """Update camera exposure."""
        val = float(value)
        range_text = self.exposure_range_var.get()

        if range_text == "32-1000us":
            self.exposure_label.configure(text=f"{val:.0f} µs")
            exposure_us = int(val)
        elif range_text in ("1-10s", "10-60s"):
            self.exposure_label.configure(text=f"{val:.1f} s")
            exposure_us = int(val * 1_000_000)
        else:
            self.exposure_label.configure(text=f"{val:.1f} ms")
            exposure_us = int(val * 1000)

        if not self.camera or not self.camera_initialized:
            return

        try:
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
        except Exception as e:
            pass

    def update_gain(self, value):
        gain = int(float(value))
        self.gain_label.configure(text=str(gain))

        if not self.camera or not self.camera_initialized:
            return

        try:
            self.camera.set_control_value(asi.ASI_GAIN, gain)
        except Exception as e:
            self.update_status(f"Error setting gain: {e}")

    def update_image_format(self, choice=None):
        if not self.camera or not self.camera_initialized:
            return

        was_capturing = self.is_capturing
        if was_capturing:
            self.stop_preview()

        format_map = {"RAW8": asi.ASI_IMG_RAW8, "RAW16": asi.ASI_IMG_RAW16}
        fmt_text = self.format_var.get()
        selected_format = format_map.get(fmt_text, asi.ASI_IMG_RAW16)

        def work():
            self.camera.set_image_type(selected_format)
            return fmt_text

        def done(_):
            self.update_status(f"Image format set to {fmt_text}")
            if was_capturing:
                self.start_preview()

        def fail(exc):
            self.update_status(f"Error setting format: {exc}")

        self._run_async(work, busy_message=f"Switching to {fmt_text}…", on_done=done, on_error=fail)

    def select_save_path(self):
        path = filedialog.askdirectory(initialdir=self.save_path)
        if path:
            self.save_path = path
            self.path_entry.configure(state="normal")
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)
            self.path_entry.configure(state="readonly")
            self._save_site_settings()

    # ---------------------------------------------------------------- Focus tab
    def _build_focus_tab(self, parent):
        """Focus assistant: step a manual focuser through positions and find the best.

        Workflow expected by the user: aim at an isolated bright star, turn the
        focus knob a small amount, click "Capture sample", and hold still while
        the GUI waits a settle delay (default 3 s) for the telescope to stop
        shaking. The latest preview frame after the settle is then scored using
        the Half-Flux Radius of detected stars (lower is sharper). The plot and
        sample list highlight the best-focus sample so the observer can return
        the focuser to that position.
        """
        intro = tk.Label(
            parent,
            text=(
                "Aim at an isolated bright star with the live preview running."
            ),
            font=("Segoe UI", 9),
            wraplength=240,
            justify=tk.LEFT,
        )
        intro.pack(anchor="w", pady=(0, 4))
        steps = tk.Label(
            parent,
            text=(
                "1) Turn the focus knob a small amount.\n"
                "2) Click Capture sample, then keep hands off.\n"
                "3) Repeat across the focus range — the plot and list mark the\n"
                "   sharpest sample as BEST so you can return to it."
            ),
            font=("Segoe UI", 9, "italic"),
            wraplength=240,
            justify=tk.LEFT,
        )
        steps.pack(anchor="w", pady=(0, 8))

        # Settle seconds entry
        settle_row = tk.Frame(parent)
        settle_row.pack(fill=tk.X, pady=(2, 4))
        tk.Label(settle_row, text="Settle (s):", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.focus_settle_var = tk.DoubleVar(value=3.0)
        self.focus_settle_entry = ctk.CTkEntry(
            settle_row,
            textvariable=self.focus_settle_var,
            width=70,
            height=26,
            fg_color="white",
            text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6,
        )
        self.focus_settle_entry.pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(
            settle_row,
            text="(wait for vibration to die)",
            font=("Segoe UI", 8, "italic"),
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Position label entry (free-form: turn count, focuser microns, etc.)
        pos_row = tk.Frame(parent)
        pos_row.pack(fill=tk.X, pady=(2, 4))
        tk.Label(pos_row, text="Position:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.focus_pos_var = tk.StringVar(value="")
        self.focus_pos_entry = ctk.CTkEntry(
            pos_row,
            textvariable=self.focus_pos_var,
            height=26,
            fg_color="white",
            text_color="#1a1a1a",
            placeholder_text_color="#999999",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6,
            placeholder_text="optional knob/turn label",
        )
        self.focus_pos_entry.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        # Capture sample (primary) + reset run buttons. Red palette so the
        # focus controls stay dark-adapted at the eyepiece.
        red_pal = PLATE_RED_PAL
        self.focus_capture_btn = ctk.CTkButton(
            parent,
            text="Capture sample",
            command=self._focus_capture_sample,
            font=("Segoe UI", 10, "bold"),
            height=36,
            fg_color=red_pal["accent"],
            text_color=red_pal["btn_text"],
            hover_color=red_pal["accent_hover"],
            corner_radius=6,
        )
        self.focus_capture_btn.pack(fill=tk.X, pady=(8, 4))

        self.focus_reset_btn = ctk.CTkButton(
            parent,
            text="Reset run",
            command=self._focus_reset_run,
            font=("Segoe UI", 9),
            height=28,
            fg_color=red_pal["accent_dim"],
            text_color=red_pal["btn_text"],
            hover_color=red_pal["accent"],
            corner_radius=6,
        )
        self.focus_reset_btn.pack(fill=tk.X, pady=(0, 8))

        self.focus_status_label = tk.Label(
            parent,
            text="No samples yet. Connect the camera and start the live preview.",
            font=("Segoe UI", 9, "italic"),
            wraplength=240,
            justify=tk.LEFT,
            anchor=tk.W,
        )
        self.focus_status_label.pack(fill=tk.X, pady=(0, 4))

        self.focus_best_label = tk.Label(
            parent,
            text="Best focus: \u2014",
            font=("Segoe UI", 10, "bold"),
            wraplength=240,
            justify=tk.LEFT,
            anchor=tk.W,
        )
        self.focus_best_label.pack(fill=tk.X, pady=(0, 6))

        # Embedded matplotlib plot of focus score per sample
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill=tk.X, pady=(0, 6))
        if _HAS_MATPLOTLIB:
            self._focus_fig, self._focus_ax = plt.subplots(figsize=(2.8, 2.0), dpi=100)
            self._focus_fig.subplots_adjust(left=0.22, right=0.96, top=0.90, bottom=0.24)
            self._focus_canvas = FigureCanvasTkAgg(self._focus_fig, master=plot_frame)
            self._focus_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self._focus_fig = None
            self._focus_ax = None
            self._focus_canvas = None
            tk.Label(
                plot_frame,
                text="(install matplotlib to see the focus plot)",
                font=("Segoe UI", 9, "italic"),
            ).pack(fill=tk.X, padx=4, pady=4)

        tk.Label(parent, text="Samples:", font=("Segoe UI", 9, "bold")).pack(
            anchor="w", pady=(4, 2)
        )
        list_frame = tk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.focus_listbox = tk.Listbox(
            list_frame,
            height=5,
            font=("Courier New", 9),
            yscrollcommand=sb.set,
            activestyle="none",
            relief=tk.FLAT,
            highlightthickness=1,
            exportselection=False,
        )
        self.focus_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.focus_listbox.yview)

        self._focus_redraw_plot()
        self._focus_refresh_table()

    def _focus_capture_sample(self):
        """Begin a single focus sample: settle wait, then score the next preview frame."""
        if self._focus_run_busy:
            return
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Focus", "Please connect to a camera first.")
            return
        # Need preview running so frames stream into the cache.
        if not self.is_capturing:
            try:
                self.start_preview()
            except Exception as exc:
                messagebox.showerror("Focus", f"Could not start preview: {exc}")
                return
            if not self.is_capturing:
                return
        try:
            settle_s = float(self.focus_settle_var.get())
        except Exception:
            settle_s = 3.0
        # Clamp to a sane range so a stray entry can't hang the GUI for minutes.
        settle_s = max(0.0, min(30.0, settle_s))

        self._focus_run_busy = True
        try:
            self.focus_capture_btn.configure(state="disabled")
        except Exception:
            pass

        # Pull the user-typed position label here on the main thread; default to
        # an auto-incrementing "#N" if the field is empty so the worker never
        # needs to touch Tk variables.
        user_pos = self.focus_pos_var.get().strip()
        sample_idx = len(self._focus_samples) + 1
        position_label = user_pos if user_pos else f"#{sample_idx}"

        threading.Thread(
            target=self._focus_settle_worker,
            kwargs={
                "settle_s": settle_s,
                "sample_idx": sample_idx,
                "position_label": position_label,
            },
            daemon=True,
        ).start()

    def _focus_settle_worker(self, *, settle_s: float, sample_idx: int, position_label: str):
        try:
            deadline = time.time() + settle_s
            # Countdown banner
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                secs_left = int(math.ceil(remaining))
                self.root.after(
                    0,
                    lambda s=secs_left: self.focus_status_label.configure(
                        text=f"Settling… {s}s remaining (keep hands off)."
                    ),
                )
                time.sleep(min(0.25, max(0.05, remaining)))
            self.root.after(
                0,
                lambda: self.focus_status_label.configure(text="Measuring sample…"),
            )

            # Wait for a frame whose timestamp is at or past the settle deadline
            # so we score the post-shake state instead of a stale frame.
            try:
                exposure_s = float(self._get_exposure_seconds())
            except Exception:
                exposure_s = 0.0
            frame_wait_s = max(6.0, 2.0 * exposure_s + 2.0)
            timeout_at = time.time() + frame_wait_s
            frame_rgb = None
            while time.time() < timeout_at:
                ts = float(self._latest_preview_full_ts or 0.0)
                arr = self._latest_preview_full_rgb
                if arr is not None and ts >= deadline:
                    frame_rgb = arr
                    break
                time.sleep(0.05)
            if frame_rgb is None:
                # Fall back to whatever is most recent rather than dropping the click.
                frame_rgb = self._latest_preview_full_rgb
            if frame_rgb is None:
                self.root.after(
                    0,
                    lambda: self.focus_status_label.configure(
                        text="No preview frame available. Start the live preview and aim at a star."
                    ),
                )
                return

            score, kind, n_stars, raw_value = self._focus_compute_score(frame_rgb)
            sample = {
                "index": sample_idx,
                "position": position_label,
                "score": float(score),
                "kind": kind,
                "n_stars": int(n_stars),
                "raw": float(raw_value),
                "captured_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._focus_samples.append(sample)
            self.root.after(0, self._focus_finalize_sample, sample)
        except Exception as exc:
            err_msg = f"Focus error: {exc}"
            self.root.after(
                0, lambda m=err_msg: self.focus_status_label.configure(text=m)
            )
        finally:
            self.root.after(0, self._focus_finish_busy)

    def _focus_finalize_sample(self, sample: dict):
        self._focus_redraw_plot()
        self._focus_refresh_table()
        self._update_best_focus_label()
        kind = sample.get("kind", "laplacian")
        if kind == "hfr":
            msg = (
                f"Sample {sample['position']} \u2192 HFR {sample['raw']:.2f} px "
                f"({sample['n_stars']} star{'s' if sample['n_stars'] != 1 else ''}). "
                f"Lower is sharper."
            )
        else:
            msg = (
                f"Sample {sample['position']} \u2192 sharpness {sample['raw']:.0f} "
                f"(no stars detected; using Laplacian fallback)."
            )
        self.focus_status_label.configure(text=msg)

    def _focus_finish_busy(self):
        self._focus_run_busy = False
        try:
            self.focus_capture_btn.configure(state="normal")
        except Exception:
            pass

    def _focus_reset_run(self):
        if self._focus_run_busy:
            return
        self._focus_samples = []
        self._focus_redraw_plot()
        self._focus_refresh_table()
        self._update_best_focus_label()
        try:
            self.focus_status_label.configure(
                text="Run reset. Turn the focuser, then capture a sample."
            )
        except Exception:
            pass

    def _focus_compute_score(self, frame_rgb: np.ndarray):
        """Score a frame's focus quality.

        Returns ``(score, kind, n_stars, raw)`` where ``score`` is always
        higher-is-sharper (used for ranking), ``kind`` is ``"hfr"`` if a
        star-based metric was available or ``"laplacian"`` for the fallback,
        ``n_stars`` is the number of stars used for HFR, and ``raw`` is the
        primary value to display: HFR in pixels (lower better) when ``kind``
        is ``"hfr"``, or the Laplacian variance otherwise.
        """
        arr = np.asarray(frame_rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = (
                0.299 * arr[..., 0].astype(np.float32)
                + 0.587 * arr[..., 1].astype(np.float32)
                + 0.114 * arr[..., 2].astype(np.float32)
            )
        else:
            gray = arr.astype(np.float32)

        # Primary metric: median half-flux radius of the brightest detected stars.
        try:
            data = np.ascontiguousarray(gray, dtype=np.float32)
            bkg = sep.Background(data)
            data_sub = data - bkg
            thresh_val = max(2.5, 4.0 * float(bkg.globalrms))
            sources = sep.extract(data_sub, thresh_val, minarea=5)
            if sources is not None and len(sources) > 0:
                sorted_sources = sorted(
                    sources, key=lambda s: float(s["flux"]), reverse=True
                )
                top = sorted_sources[: min(10, len(sorted_sources))]
                xs = np.array([float(s["x"]) for s in top])
                ys = np.array([float(s["y"]) for s in top])
                a_vals = np.array([float(s["a"]) for s in top])
                # Cap aperture to the minimum of image half-extent so flux_radius
                # never asks for samples outside the array.
                h, w = data_sub.shape
                max_aperture = max(8.0, 0.45 * min(h, w))
                r_apertures = np.clip(6.0 * a_vals, 12.0, max_aperture)
                try:
                    radii, _flags = sep.flux_radius(
                        data_sub, xs, ys, r_apertures, 0.5
                    )
                except Exception:
                    radii = None
                if radii is not None:
                    radii_arr = np.asarray(radii, dtype=np.float64)
                    ok = np.isfinite(radii_arr) & (radii_arr > 0)
                    if np.any(ok):
                        hfr = float(np.median(radii_arr[ok]))
                        score = 1.0 / max(hfr, 1e-3)
                        return score, "hfr", int(np.sum(ok)), hfr
        except Exception:
            pass

        # Fallback metric: Laplacian variance — works on any structured frame,
        # but values are not directly comparable to HFR-based runs.
        try:
            lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            return lap_var, "laplacian", 0, lap_var
        except Exception:
            return 0.0, "laplacian", 0, 0.0

    def _best_sample_index(self) -> int | None:
        if not self._focus_samples:
            return None
        hfr_indices = [
            i for i, s in enumerate(self._focus_samples) if s.get("kind") == "hfr"
        ]
        if hfr_indices:
            return min(hfr_indices, key=lambda i: self._focus_samples[i]["raw"])
        return max(
            range(len(self._focus_samples)),
            key=lambda i: self._focus_samples[i].get("score", 0.0),
        )

    def _focus_redraw_plot(self) -> None:
        ax = getattr(self, "_focus_ax", None)
        canvas = getattr(self, "_focus_canvas", None)
        fig = getattr(self, "_focus_fig", None)
        if ax is None or canvas is None or fig is None:
            return
        t = self.theme
        bg = t["bg_secondary"]
        fg = t["fg_primary"]
        grid = t["fg_tertiary"]
        accent = t["accent"]
        try:
            fig.patch.set_facecolor(bg)
            ax.clear()
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(grid)
            ax.set_xlabel("Sample #", color=fg, fontsize=9)

            samples = self._focus_samples
            if not samples:
                ax.text(
                    0.5,
                    0.5,
                    "no samples yet",
                    ha="center",
                    va="center",
                    color=grid,
                    fontsize=9,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                canvas.draw_idle()
                return

            has_hfr = any(s.get("kind") == "hfr" for s in samples)
            if has_hfr:
                ax.set_ylabel("HFR (px) — lower better", color=fg, fontsize=9)
                plot_samples = [s for s in samples if s.get("kind") == "hfr"]
            else:
                ax.set_ylabel("Sharpness — higher better", color=fg, fontsize=9)
                plot_samples = list(samples)

            xs = [s["index"] for s in plot_samples]
            ys = [s["raw"] for s in plot_samples]
            ax.plot(
                xs,
                ys,
                color=accent,
                marker="o",
                markersize=4,
                linewidth=1.4,
            )
            best_idx = self._best_sample_index()
            if best_idx is not None and 0 <= best_idx < len(samples):
                best = samples[best_idx]
                if (has_hfr and best.get("kind") == "hfr") or not has_hfr:
                    ax.plot(
                        [best["index"]],
                        [best["raw"]],
                        marker="*",
                        markersize=14,
                        color=accent,
                        markeredgecolor=fg,
                        markeredgewidth=1.0,
                        linestyle="None",
                    )
            ax.grid(color=grid, alpha=0.3, linestyle="--", linewidth=0.6)
            canvas.draw_idle()
        except Exception:
            pass

    def _focus_refresh_table(self) -> None:
        lb = getattr(self, "focus_listbox", None)
        if lb is None:
            return
        try:
            lb.delete(0, tk.END)
        except tk.TclError:
            return
        best_idx = self._best_sample_index()
        for i, s in enumerate(self._focus_samples):
            position = str(s.get("position", f"#{s.get('index', i + 1)}"))
            kind = s.get("kind", "laplacian")
            if kind == "hfr":
                value_txt = f"HFR {s['raw']:5.2f} px ({s['n_stars']} stars)"
            else:
                value_txt = f"sharp {s['raw']:7.0f}"
            marker = "  *BEST*" if i == best_idx else ""
            line = f"{s['index']:>3}  {position:<10.10}  {value_txt}{marker}"
            lb.insert(tk.END, line)
        if best_idx is not None:
            try:
                lb.selection_clear(0, tk.END)
                lb.selection_set(best_idx)
                lb.see(best_idx)
            except tk.TclError:
                pass

    def _update_best_focus_label(self) -> None:
        lbl = getattr(self, "focus_best_label", None)
        if lbl is None:
            return
        best_idx = self._best_sample_index()
        if best_idx is None:
            lbl.configure(text="Best focus: \u2014")
            return
        s = self._focus_samples[best_idx]
        if s.get("kind") == "hfr":
            txt = (
                f"Best focus: sample {s['position']} — HFR {s['raw']:.2f} px"
            )
        else:
            txt = (
                f"Best focus: sample {s['position']} — sharpness {s['raw']:.0f}"
            )
        lbl.configure(text=txt)

    def start_preview(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Warning", "Please connect to a camera first")
            return

        self._maybe_warn_solar_filter_daytime("starting the live preview")

        try:
            self.camera.start_video_capture()
            self.is_capturing = True
            self.update_status("Preview started")
            threading.Thread(target=self.preview_loop, daemon=True).start()
            threading.Thread(target=self.temperature_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start preview: {e}")

    def stop_preview(self):
        try:
            self.is_capturing = False
            if self.is_recording:
                self.toggle_recording()
            if self.camera and self.camera_initialized:
                self.camera.stop_video_capture()
            self.update_status("Preview stopped")
        except Exception as e:
            self.update_status(f"Error stopping preview: {e}")

    def _set_temperature_label(self, text: str):
        """Update temperature readout; must run on the Tk main thread."""
        try:
            self.temp_label.configure(text=text)
        except tk.TclError:
            pass

    def temperature_loop(self):
        while self.is_capturing and self.camera and self.camera_initialized:
            try:
                controls = self.camera.get_controls()
                if "Temperature" in controls:
                    temp = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
                    label_text = f"Temp: {temp:.1f}°C"
                else:
                    label_text = "Temp: --°C"
            except Exception:
                label_text = "Temp: --°C"
            self.root.after(0, lambda t=label_text: self._set_temperature_label(t))
            time.sleep(2)

    def _update_preview_image(self, img_rgb: np.ndarray):
        """Show a resized RGB frame on the preview label (Tk main thread only)."""
        if not self.is_capturing:
            return
        try:
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.preview_label.configure(image=img_tk, text="")
            self.preview_label.image = img_tk
        except tk.TclError:
            pass

    def preview_loop(self):
        while self.is_capturing:
            try:
                # FIX: Compute a timeout that accounts for the actual exposure duration.
                # The ZWO API default timeout is too short for exposures longer than ~1s,
                # causing capture_video_frame() to raise an error before the frame arrives.
                # Formula: exposure_ms * 2 + 500ms gives a safe margin in all ranges.
                exposure_s = self._get_exposure_seconds()
                exposure_ms = exposure_s * 1000
                timeout_ms = int(exposure_ms * 2 + 500)

                frame = self.camera.capture_video_frame(timeout=timeout_ms)

                if frame is not None:
                    current_format = self.format_var.get()

                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                        frame_2d = frame if len(frame.shape) == 2 else frame.reshape(frame.shape[0], frame.shape[1])

                        if current_format == "RAW16":
                            frame_8bit = (frame_2d / 256).astype(np.uint8)
                            img = cv2.cvtColor(frame_8bit, cv2.COLOR_BayerRG2RGB)
                        elif current_format == "RAW8":
                            img = cv2.cvtColor(frame_2d, cv2.COLOR_BayerRG2RGB)
                        else:
                            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    else:
                        img = frame

                    if self.is_recording and self.video_writer:
                        self.video_writer.write(img)

                    # Cache the latest full-resolution RGB frame for the Focus
                    # assistant. Copy() so subsequent buffer reuse by the SDK
                    # doesn't mutate what we hand to the focus scorer.
                    try:
                        self._latest_preview_full_rgb = img.copy()
                        self._latest_preview_full_ts = time.time()
                    except Exception:
                        pass

                    height, width = img.shape[:2]
                    scale = min(640 / width, 480 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)

                    img_resized = cv2.resize(img, (new_width, new_height))
                    # Tk is not thread-safe; marshal UI updates to the main thread.
                    frame_rgb = np.ascontiguousarray(img_resized)
                    self.root.after(0, lambda a=frame_rgb: self._update_preview_image(a))

            except Exception as e:
                # ZWO preview can occasionally time out when no frame is ready yet.
                # Treat timeout as a transient condition and continue preview loop.
                if isinstance(e, asi.ZWO_IOError) and "timeout" in str(e).lower():
                    continue
                print(f"Preview error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def capture_image(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Warning", "Please connect to a camera first")
            return
        if getattr(self, "_capture_in_progress", False):
            self.update_status("Capture already in progress…")
            return

        self._maybe_warn_solar_filter_daytime("capturing images")

        current_format = self.format_var.get()
        bin_value = int(self.binning_dropdown.get().split('x')[0])
        capture_count = int(self.capture_count_dropdown.get())

        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = self.project_name_entry.get().strip()
        frame_type = self.frame_type_var.get()
        if project_name:
            base_path = os.path.join(self.save_path, project_name, frame_type)
        else:
            base_path = os.path.join(self.save_path, frame_type)
        capture_path = base_path

        try:
            os.makedirs(capture_path, exist_ok=True)
            print(f"Capture directory: {capture_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create capture folder: {e}")
            return

        range_text = self.exposure_range_var.get()
        exposure_val = float(self.exposure_var.get())
        gain_val = int(self.gain_var.get())

        # Snapshot UI state up front so the worker thread never touches Tk vars.
        self._capture_in_progress = True
        try:
            self.capture_img_btn.configure(state="disabled")
        except Exception:
            pass
        self._begin_busy(f"Capturing 1/{capture_count}…")
        threading.Thread(
            target=self._capture_image_worker,
            kwargs=dict(
                current_format=current_format,
                bin_value=bin_value,
                capture_count=capture_count,
                session_timestamp=session_timestamp,
                capture_path=capture_path,
                range_text=range_text,
                exposure_val=exposure_val,
                gain_val=gain_val,
            ),
            daemon=True,
        ).start()

    def _capture_image_worker(
        self,
        *,
        current_format: str,
        bin_value: int,
        capture_count: int,
        session_timestamp: str,
        capture_path: str,
        range_text: float,
        exposure_val: float,
        gain_val: int,
    ):
        was_previewing = False
        try:
            was_previewing = self._stop_preview_for_still()

            _, exposure_display, exposure_s = self._exposure_us_display_seconds(range_text, exposure_val)

            self._configure_still_capture_hardware(
                bin_value=bin_value,
                image_format=current_format,
                range_text=range_text,
                exposure_val=exposure_val,
                gain=gain_val,
            )

            all_saved_files = []

            for img_num in range(1, capture_count + 1):
                print(f"\n{'=' * 60}")
                print(f"CAPTURING IMAGE {img_num}/{capture_count}")
                print(f"{'=' * 60}")

                msg_cap = f"Capturing image {img_num}/{capture_count}..."
                self.root.after(0, lambda m=msg_cap: (self.update_status(m), self.busy_text_label.configure(text=m)))

                frame = self._grab_still_frame()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                try:
                    temp = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
                except:
                    temp = None

                metadata = {
                    'image_number': img_num,
                    'total_images': capture_count,
                    'timestamp': timestamp,
                    'datetime': datetime.now().isoformat(),
                    'camera': self.camera_info['Name'],
                    'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                    'bins': bin_value,
                    'format': current_format,
                    'bit_depth': 16 if current_format == "RAW16" else 8,
                    'dtype': str(frame.dtype),
                    'buffer_size': frame.nbytes,
                    'shape': frame.shape,
                    'exposure_display': exposure_display,
                    'exposure_s': exposure_s,
                    'gain': gain_val,
                    'temperature_c': temp
                }

                print(f"Capture metadata for image {img_num}:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")

                fits_filename = f"image_{img_num:03d}_{timestamp}_{current_format}.fits"
                fits_filepath = os.path.join(capture_path, fits_filename)

                msg_save = f"Saving FITS {img_num}/{capture_count}..."
                self.root.after(0, lambda m=msg_save: (self.update_status(m), self.busy_text_label.configure(text=m)))

                header = fits.Header()
                header['INSTRUME'] = self.camera_info['Name']
                header['EXPOSURE'] = (exposure_s, 'Exposure time in seconds')
                header['GAIN'] = (gain_val, 'Gain value')
                header['BINNING'] = (bin_value, 'Binning factor')
                header['XBINNING'] = (bin_value, 'X-axis binning')
                header['YBINNING'] = (bin_value, 'Y-axis binning')
                header['XPIXSZ'] = (2.9, 'Pixel size in microns (unbinned)')
                header['YPIXSZ'] = (2.9, 'Pixel size in microns (unbinned)')
                header['IMAGETYP'] = ('Light Frame', 'Type of image')
                header['BAYERPAT'] = ('RGGB', 'Bayer pattern')
                header['COLORTYP'] = ('RGGB', 'Color type')
                header['DATE-OBS'] = (datetime.now().isoformat(), 'Date of observation')
                header['IMGNUM'] = (img_num, f'Image number in sequence of {capture_count}')
                header['BITDEPTH'] = (metadata['bit_depth'], 'Bit depth')
                if temp is not None:
                    header['CCD-TEMP'] = (temp, 'CCD temperature in Celsius')

                hdu = fits.PrimaryHDU(data=frame, header=header)
                hdu.writeto(fits_filepath, overwrite=True)

                print(f"Saved FITS file: {fits_filepath}")
                self._session_photos.append(
                    {
                        "fits_path": os.path.abspath(fits_filepath),
                        "captured_at": metadata["datetime"],
                        "ra_deg": None,
                        "dec_deg": None,
                    }
                )
                self.root.after(0, self._refresh_session_photos_listbox)
                all_saved_files.append(fits_filename)

                if current_format == "RAW8":
                    msg_png = f"Creating PNG {img_num}/{capture_count}..."
                    self.root.after(0, lambda m=msg_png: (self.update_status(m), self.busy_text_label.configure(text=m)))

                    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                        debayered = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
                    else:
                        debayered = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    png_filename = f"image_{img_num:03d}_{timestamp}_debayered.png"
                    png_filepath = os.path.join(capture_path, png_filename)
                    cv2.imwrite(png_filepath, cv2.cvtColor(debayered, cv2.COLOR_RGB2BGR))

                    print(f"Saved debayered PNG: {png_filepath}")
                    all_saved_files.append(png_filename)

                metadata_filename = f"image_{img_num:03d}_{timestamp}_metadata.txt"
                metadata_filepath = os.path.join(capture_path, metadata_filename)

                with open(metadata_filepath, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"CAPTURE METADATA - IMAGE {img_num}/{capture_count}\n")
                    f.write("=" * 60 + "\n\n")

                    for key, value in [
                        ('Image Number', f"{metadata['image_number']} of {metadata['total_images']}"),
                        ('Camera', metadata['camera']),
                        ('Format', metadata['format']),
                        ('Timestamp', metadata['datetime']),
                        ('Resolution', metadata['resolution']),
                        ('Binning', f"{metadata['bins']}x{metadata['bins']}"),
                        ('Bit Depth', metadata['bit_depth']),
                        ('Data Type', metadata['dtype']),
                        ('Buffer Size', f"{metadata['buffer_size']} bytes"),
                        ('Array Shape', metadata['shape']),
                        ('Exposure', metadata['exposure_display']),
                        ('Gain', metadata['gain'])
                    ]:
                        f.write(f"{key}: {value}\n")

                    if metadata['temperature_c'] is not None:
                        f.write(f"CCD Temperature: {metadata['temperature_c']:.1f}°C\n")

                    f.write("\n" + "=" * 60 + "\n")
                    f.write("FILES FOR THIS IMAGE\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"  - {fits_filename}\n")
                    if current_format == "RAW8":
                        f.write(f"  - {png_filename}\n")
                    f.write(f"  - {metadata_filename}\n")

                print(f"Saved metadata: {metadata_filepath}")
                all_saved_files.append(metadata_filename)
                print(f"Completed image {img_num}/{capture_count}")

            summary_filename = f"_session_summary_{session_timestamp}.txt"
            summary_filepath = os.path.join(capture_path, summary_filename)

            with open(summary_filepath, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("CAPTURE SESSION SUMMARY\n")
                f.write("=" * 60 + "\n\n")

                for key, value in [
                    ('Session ID', session_timestamp),
                    ('Total Images', capture_count),
                    ('Format', current_format),
                    ('Camera', self.camera_info['Name']),
                    ('Session Started', session_timestamp),
                    ('Exposure', exposure_display),
                    ('Gain', gain_val),
                    ('Binning', f"{bin_value}x{bin_value}")
                ]:
                    f.write(f"{key}: {value}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write(f"ALL FILES ({len(all_saved_files)} total)\n")
                f.write("=" * 60 + "\n\n")
                for fname in sorted(all_saved_files):
                    f.write(f"  - {fname}\n")
                f.write(f"  - {summary_filename}\n")

            all_saved_files.append(summary_filename)
            print(f"\nSaved session summary: {summary_filepath}")

            done_msg = f"Completed: {capture_count} images saved to {capture_path}"
            files_description = (
                "- FITS file\n- PNG file (debayered)\n- Metadata file"
                if current_format == "RAW8"
                else "- FITS file\n- Metadata file"
            )
            success_msg = (
                f"Capture session completed!\n\n"
                f"Format: {current_format}\n"
                f"Images captured: {capture_count}\n"
                f"Folder:\n{capture_path}\n\n"
                f"Files per image:\n{files_description}\n\n"
                f"Total files: {len(all_saved_files)}"
            )

            def _notify_success():
                self.update_status(done_msg)
                messagebox.showinfo("Success", success_msg)

            self.root.after(0, _notify_success)

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_text = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to capture images: {err_text}"))

        finally:
            def _finalize():
                self._end_busy()
                self._capture_in_progress = False
                try:
                    self.capture_img_btn.configure(state="normal")
                except Exception:
                    pass
                if was_previewing:
                    self.update_status("Restarting preview...")
                    self.start_preview()

            self.root.after(0, _finalize)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Warning", "Please connect to a camera first")
            return

        try:
            if not self.is_capturing:
                self.start_preview()
            else:
                self._maybe_warn_solar_filter_daytime("starting video recording")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.avi"
            project_name = self.project_name_entry.get().strip()
            frame_type = self.frame_type_var.get()
            if project_name:
                video_dir = os.path.join(self.save_path, project_name, frame_type)
            else:
                video_dir = os.path.join(self.save_path, frame_type)
            os.makedirs(video_dir, exist_ok=True)
            filepath = os.path.join(video_dir, filename)

            info = self.camera.get_roi_format()
            width = info[0]
            height = info[1]

            # FIX: Derive FPS from actual exposure time instead of hardcoding 30.
            # A hardcoded 30 FPS causes the AVI to play back wildly accelerated for
            # long exposures (e.g. 5s exposure at 30 FPS = 150x real speed), and the
            # write buffer backs up or drops frames. We cap at 30 FPS for short
            # exposures and floor at 1 FPS so the VideoWriter always stays valid.
            exposure_s = self._get_exposure_seconds()
            fps = min(30.0, 1.0 / exposure_s) if exposure_s > 0 else 30.0
            fps = max(fps, 1.0)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            self.is_recording = True
            self.record_video_btn.configure(text="Stop Recording")
            self.update_status(f"Recording to: {filename} @ {fps:.1f} FPS")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def stop_recording(self):
        try:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_video_btn.configure(text="Start Recording")
            self.update_status("Recording stopped")
            messagebox.showinfo("Success", "Video saved successfully")
        except Exception as e:
            self.update_status(f"Error stopping recording: {e}")

    def update_status(self, message):
        self.status_label.configure(text=message)
        print(message)

    def open_plate_solver(self):
        self._focus_center_tab("Plate solve")
        self.update_status("Plate solve")

    def open_skytrack(self):
        self._focus_center_tab("Site / Sky")
        self.update_status("Site / Sky")

    def open_mount_controls(self):
        self.update_status("Mount controls are on the right")

    def _poll_livewcs(self):
        if not self._livewcs_poll_running:
            return
        try:
            if os.path.exists(self.livewcs_path):
                mtime = os.path.getmtime(self.livewcs_path)
                if mtime != self._livewcs_last_mtime:
                    self._livewcs_last_mtime = mtime
                    with open(self.livewcs_path, "r") as f:
                        data = json.load(f)
                    self._apply_livewcs(data)
            else:
                self._clear_livewcs()
        except Exception as e:
            print(f"LiveWCS read error: {e}")
            self._clear_livewcs()

        self.root.after(1000, self._poll_livewcs)

    def _apply_livewcs(self, data):
        target = data.get("object", "--")
        ra = data.get("ra_WCS", "--")
        dec = data.get("dec_WCS", "--")

        t = self.theme
        self._livewcs_active = bool(target and str(target).strip() and target != "--")
        status_text = "Sky target" if self._livewcs_active else "No sky target"
        status_color = t["accent"] if self._livewcs_active else t["fg_tertiary"]

        self.skytrack_signal_dot.configure(text="●", fg=status_color)
        self.skytrack_signal_label.configure(text=status_text, fg=status_color)
        self.skytrack_target_label.configure(text=f"Target:  {target}")
        self.skytrack_ra_label.configure(text=f"RA :    {ra}")
        self.skytrack_dec_label.configure(text=f"DEC :  {dec}")

    def _clear_livewcs(self):
        t = self.theme
        self._livewcs_active = False
        not_tracking_color = t["fg_tertiary"]
        self.skytrack_signal_dot.configure(text="●", fg=not_tracking_color)
        self.skytrack_signal_label.configure(text="No sky target", fg=not_tracking_color)
        self.skytrack_target_label.configure(text="Target:")
        self.skytrack_ra_label.configure(text="RA :")
        self.skytrack_dec_label.configure(text="DEC :")

    def cleanup(self):
        self._livewcs_poll_running = False
        try:
            if getattr(self, "mount_panel", None):
                self.mount_panel.cleanup_embedded()
        except Exception:
            pass
        try:
            if self.is_recording:
                self.stop_recording()
            if self.is_capturing:
                self.stop_preview()
            if self.camera:
                self.camera.close()
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = ZWOCameraGUI(root)
    closing = {"done": False}

    def _close_app():
        if closing["done"]:
            return
        closing["done"] = True
        try:
            app.cleanup()
        except Exception:
            pass
        # Let pending Tk/CTk callbacks unwind before destruction.
        try:
            root.withdraw()
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass
        try:
            root.after(80, root.destroy)
        except Exception:
            try:
                root.destroy()
            except Exception:
                pass

    root.protocol("WM_DELETE_WINDOW", _close_app)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        _close_app()


if __name__ == "__main__":
    main()
