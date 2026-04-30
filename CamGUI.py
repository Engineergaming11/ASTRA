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
from datetime import datetime, time as dt_time
from pathlib import Path
from urllib.request import urlopen

from astropy.io import fits
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
    SOLAR_SYSTEM_BODIES,
    TYCHO_PRESET_STARS,
    major_bodies_separation_table,
    polar_align_mvp_text,
    radec_to_altaz_deg,
    sun_apparent_radec_deg,
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


def _default_capture_save_path() -> str:
    """Cross-platform default capture directory (under the current user's home)."""
    return str(Path.home() / "ASICAP" / "CapGUI")


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
        # the user clicks "Auto-fill from camera". These fallbacks roughly
        # match a typical 1.25" small-pixel astro camera.
        "label": "ZWO (auto)",
        "pixel_size_um": 3.76,
        "sensor_w_px": 1936,
        "sensor_h_px": 1096,
    },
}


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


class ZWOCameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRA")
        self.root.geometry("1680x860")
        self.root.minsize(1280, 700)

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
        self._last_solve_ra_deg = None
        self._last_solve_dec_deg = None
        self._last_plate_solve_input_path = None
        self._sun_crosshair_enabled = True
        self._sun_crosshair_size_px = 18
        self._sun_crosshair_thickness = 2
        # In-app session captures for Session photos tab (fits_path, captured_at iso, ra/dec if solved)
        self._session_photos = []
        # Equipment preset (Tucson, AZ + 2032 mm SCT). Loaded from disk if present;
        # otherwise the bundled default is used and persisted on next save.
        self.astro_preset = json.loads(json.dumps(DEFAULT_ASTRO_PRESET))  # deep-ish copy
        self._load_site_settings()
        self._load_astro_preset()

        try:
            os.makedirs(self.save_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create save directory {self.save_path}: {e}")
            self.save_path = str(Path.home())

        self._livewcs_last_mtime = None
        self._livewcs_poll_running = False
        self._livewcs_active = False

        # Focus-assist wizard (quarter-turn sweep)
        self._focus_cancel = threading.Event()
        self._focus_generation = 0
        self._focus_saved_camera = None
        self._focus_scores = []
        self._focus_sweep_metric_mode = None

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

    def _clamp_gain_for_camera(self, gain: int) -> int:
        if not self.camera or not self.camera_initialized:
            return gain
        try:
            ctrl = self.camera.get_controls().get("Gain") or self.camera.get_controls().get("ASI_GAIN")
            if ctrl:
                lo = int(ctrl.get("MinValue", 0))
                hi = int(ctrl.get("MaxValue", 600))
                return max(lo, min(hi, int(gain)))
        except Exception:
            pass
        return max(0, min(600, int(gain)))

    def _frame_to_luminance_u8(self, frame: np.ndarray, image_format: str) -> np.ndarray:
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        frame_2d = frame if len(frame.shape) == 2 else frame.reshape(frame.shape[0], frame.shape[1])
        if image_format == "RAW16":
            g8 = (frame_2d / 256.0).clip(0, 255).astype(np.uint8)
        else:
            g8 = np.clip(frame_2d, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(g8, cv2.COLOR_BayerRG2RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _prep_focus_gray(self, frame: np.ndarray, image_format: str) -> np.ndarray:
        gray = self._frame_to_luminance_u8(frame, image_format)
        h, w = gray.shape[:2]
        if max(h, w) > 2048:
            scale = 2048 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return gray

    def _focus_sep_quality_from_gray(self, gray: np.ndarray):
        arr = gray.astype(np.float32)
        try:
            bkg = sep.Background(arr)
            sub = arr - bkg.back()
            thresh = 2.5 * float(np.std(sub)) + 1e-6
            sources = sep.extract(sub, thresh)
        except Exception:
            return None

        min_stars = 5
        if sources is None or len(sources) < min_stars:
            return None
        if "flux" in sources.dtype.names:
            order = np.argsort(sources["flux"])[::-1]
        else:
            order = np.arange(len(sources))
        take = min(40, len(order))
        sizes = []
        for i in range(take):
            sidx = int(order[i])
            a = float(sources["a"][sidx])
            b = float(sources["b"][sidx])
            if a <= 0 or b <= 0:
                continue
            fwhm = 2.355 * 0.5 * (a + b)
            sizes.append(fwhm)
        if len(sizes) < min_stars:
            return None
        med = float(np.median(np.array(sizes)))
        return (-med, f"SEP median FWHM~px {med:.2f} (lower is sharper)")

    def _focus_laplacian_quality_from_gray(self, gray: np.ndarray) -> tuple[float, str]:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        v = float(lap.var())
        return (v, f"Laplacian variance {v:.1f}")

    def _focus_quality_from_frame(self, frame: np.ndarray, image_format: str) -> tuple[float, str]:
        """Return (quality_higher_is_better, description)."""
        gray = self._prep_focus_gray(frame, image_format)
        sep_q = self._focus_sep_quality_from_gray(gray)
        if sep_q is not None:
            return sep_q
        return self._focus_laplacian_quality_from_gray(gray)

    def _focus_quality_for_sweep_step(self, frame: np.ndarray, image_format: str) -> tuple[float, str]:
        """One metric family for the whole sweep (plan: lock score type per run)."""
        gray = self._prep_focus_gray(frame, image_format)
        if self._focus_sweep_metric_mode is None:
            sep_q = self._focus_sep_quality_from_gray(gray)
            if sep_q is not None:
                self._focus_sweep_metric_mode = "sep"
                return sep_q
            self._focus_sweep_metric_mode = "lap"
            return self._focus_laplacian_quality_from_gray(gray)
        if self._focus_sweep_metric_mode == "sep":
            sep_q = self._focus_sep_quality_from_gray(gray)
            if sep_q is not None:
                return sep_q
            lq, ld = self._focus_laplacian_quality_from_gray(gray)
            return (lq, f"{ld} (SEP unavailable this step)")
        return self._focus_laplacian_quality_from_gray(gray)

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

        # Tk PanedWindow only accepts native Tk widgets; wrap CTkScrollableFrame in tk.Frame.
        self._left_pane_wrap = tk.Frame(self._pan_outer, bg=t0["bg_primary"])
        self.left_scroll = ctk.CTkScrollableFrame(
            self._left_pane_wrap,
            width=300,
            fg_color="transparent",
            scrollbar_button_color="#999999",
            scrollbar_button_hover_color="#777777",
        )
        self.left_scroll.pack(fill=tk.BOTH, expand=True)
        self.left = self.left_scroll  # alias so all existing widget code works unchanged

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
        self._camera_tabs.add("Focusing")
        camera_parent = self._camera_tabs.tab("Camera")

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
                button_color="#0b5c8f",
                button_hover_color="#1a7ab5",
                progress_color="#0b5c8f",
                fg_color="#e0e0e0",
                width=160,
                height=16
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

        self.start_preview_btn = create_action_button("Start Preview", self.toggle_preview)
        self.start_preview_btn.pack(fill=tk.X, pady=4)

        self.capture_img_btn = create_action_button("Capture Image", self.capture_image)
        self.capture_img_btn.pack(fill=tk.X, pady=4)

        self.record_video_btn = create_action_button("Start Recording", self.toggle_recording)
        self.record_video_btn.pack(fill=tk.X, pady=4)

        self._build_focusing_tab(self._camera_tabs.tab("Focusing"))

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

        self._focus_update_start_enabled()

    _FOCUS_FN_PRESETS = {
        "f/2.8": ("32-1000us", 500.0, 90),
        "f/4": ("1-100ms", 12.0, 120),
        "f/5.6": ("1-100ms", 28.0, 150),
        "f/8": ("1-100ms", 55.0, 180),
        "f/10": ("100ms-1000ms", 200.0, 220),
        "f/14": ("100ms-1000ms", 550.0, 280),
    }

    def _build_focusing_tab(self, focus_tab: ctk.CTkFrame):
        t = self.theme
        intro = (
            "Slew to a modest star field. You will capture at the starting focus position, then after each "
            "shot you rotate the focuser exactly one quarter-turn in the chosen direction, take your hands off, "
            "wait for the settle countdown, and the app captures again through the full travel."
        )
        ctk.CTkLabel(
            focus_tab,
            text=intro,
            font=("Segoe UI", 9),
            wraplength=260,
            justify="left",
            anchor="w",
        ).pack(anchor="w", padx=4, pady=(6, 8))

        row = ctk.CTkFrame(focus_tab, fg_color="transparent")
        row.pack(fill=tk.X, padx=4, pady=2)
        ctk.CTkLabel(row, text="Telescope f/# (preset)", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._focus_fn_var = tk.StringVar(value="f/8")
        self._focus_fn_menu = ctk.CTkComboBox(
            row,
            values=["Custom", "f/2.8", "f/4", "f/5.6", "f/8", "f/10", "f/14"],
            variable=self._focus_fn_var,
            width=120,
            state="readonly",
        )
        self._focus_fn_menu.set("f/8")
        self._focus_fn_menu.pack(side=tk.RIGHT, padx=(4, 0))

        row2 = ctk.CTkFrame(focus_tab, fg_color="transparent")
        row2.pack(fill=tk.X, padx=4, pady=2)
        ctk.CTkLabel(row2, text="Settle time (s)", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._focus_settle_var = tk.StringVar(value="3")
        self._focus_settle_entry = ctk.CTkEntry(row2, width=50, textvariable=self._focus_settle_var)
        self._focus_settle_entry.pack(side=tk.RIGHT)

        row3 = ctk.CTkFrame(focus_tab, fg_color="transparent")
        row3.pack(fill=tk.X, padx=4, pady=2)
        ctk.CTkLabel(row3, text="Quarter-turns to sweep", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._focus_turns_var = tk.StringVar(value="20")
        self._focus_turns_entry = ctk.CTkEntry(row3, width=50, textvariable=self._focus_turns_var)
        self._focus_turns_entry.pack(side=tk.RIGHT)

        row4 = ctk.CTkFrame(focus_tab, fg_color="transparent")
        row4.pack(fill=tk.X, padx=4, pady=2)
        ctk.CTkLabel(row4, text="Each step, turn toward", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._focus_dir_var = tk.StringVar(value="infinity")
        ctk.CTkRadioButton(row4, text="Infinity", variable=self._focus_dir_var, value="infinity").pack(side=tk.LEFT, padx=4)
        ctk.CTkRadioButton(row4, text="Near", variable=self._focus_dir_var, value="near").pack(side=tk.LEFT)

        self._focus_phase_label = ctk.CTkLabel(
            focus_tab,
            text="Idle.",
            font=("Segoe UI", 10, "bold"),
            wraplength=270,
            justify="left",
            anchor="w",
        )
        self._focus_phase_label.pack(anchor="w", padx=4, pady=(10, 4))

        self._focus_countdown_label = ctk.CTkLabel(
            focus_tab,
            text="",
            font=("Segoe UI", 11),
            wraplength=270,
            justify="left",
            anchor="w",
        )
        self._focus_countdown_label.pack(anchor="w", padx=4, pady=2)

        self._focus_progress_label = ctk.CTkLabel(focus_tab, text="", font=("Segoe UI", 9))
        self._focus_progress_label.pack(anchor="w", padx=4, pady=2)

        self._focus_results = ctk.CTkTextbox(focus_tab, height=120, width=280, font=("Courier New", 9))
        self._focus_results.pack(fill=tk.BOTH, expand=True, padx=4, pady=6)

        bf = ctk.CTkFrame(focus_tab, fg_color="transparent")
        bf.pack(fill=tk.X, padx=4, pady=6)
        self._focus_arm_btn = ctk.CTkButton(
            bf,
            text="Apply f/# preset to Camera tab",
            command=self._focus_apply_preset_only,
            width=200,
            font=("Segoe UI", 9),
        )
        self._focus_arm_btn.pack(fill=tk.X, pady=2)
        self._focus_start_btn = ctk.CTkButton(
            bf,
            text="Start focus sweep",
            command=self._focus_start_sweep,
            width=200,
            font=("Segoe UI", 9, "bold"),
            fg_color=t["accent"],
            hover_color=t["accent_light"],
        )
        self._focus_start_btn.pack(fill=tk.X, pady=2)
        self._focus_cancel_btn = ctk.CTkButton(
            bf,
            text="Cancel sweep",
            command=self._focus_cancel_sweep,
            width=200,
            font=("Segoe UI", 9),
            state="disabled",
        )
        self._focus_cancel_btn.pack(fill=tk.X, pady=2)
        self._focus_restore_btn = ctk.CTkButton(
            bf,
            text="Restore previous camera settings",
            command=self._focus_restore_saved,
            width=200,
            font=("Segoe UI", 9),
        )
        self._focus_restore_btn.pack(fill=tk.X, pady=2)

        self._focus_update_start_enabled()

    def _focus_update_start_enabled(self):
        ok = bool(self.camera and self.camera_initialized)
        try:
            self._focus_start_btn.configure(state="normal" if ok else "disabled")
        except Exception:
            pass

    def _focus_save_camera_snapshot(self):
        self._focus_saved_camera = {
            "exposure_range": self.exposure_range_var.get(),
            "exposure": float(self.exposure_var.get()),
            "gain": int(self.gain_var.get()),
            "binning": self.binning_dropdown.get(),
            "format": self.format_var.get(),
        }

    def _focus_restore_saved(self):
        if not self._focus_saved_camera:
            messagebox.showinfo("Focus assist", "No saved settings to restore. Run a sweep first or use Apply preset.")
            return
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Focus assist", "Connect a camera first.")
            return
        self._focus_apply_snapshot_to_ui(self._focus_saved_camera)
        self._focus_saved_camera = None
        self.update_status("Restored camera settings from before focus assist.")

    def _focus_apply_preset_only(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Focus assist", "Connect a camera first.")
            return
        fn = self._focus_fn_var.get()
        if fn == "Custom":
            messagebox.showinfo("Focus assist", "Choose an f/# other than Custom to apply a preset.")
            return
        if self._focus_saved_camera is None:
            self._focus_save_camera_snapshot()
        preset = self._FOCUS_FN_PRESETS.get(fn)
        if not preset:
            return
        rng, exp, gain = preset
        gain = self._clamp_gain_for_camera(gain)
        self.exposure_range_var.set(rng)
        self.update_exposure_range()
        self.exposure_var.set(float(exp))
        self.exposure_slider.set(float(exp))
        self.update_exposure(float(exp))
        self.gain_var.set(int(gain))
        self.gain_slider.set(int(gain))
        self.update_gain(int(gain))
        self.binning_dropdown.set("1x1")
        self.update_binning()
        self.format_var.set("RAW16")
        self.update_image_format()
        self.update_status(f"Focus preset applied: {fn}, RAW16, 1×1, {rng}.")

    def _focus_apply_snapshot_to_ui(self, s: dict):
        self.exposure_range_var.set(s["exposure_range"])
        self.update_exposure_range()
        self.exposure_var.set(s["exposure"])
        self.exposure_slider.set(s["exposure"])
        self.update_exposure(s["exposure"])
        self.gain_var.set(s["gain"])
        self.gain_slider.set(s["gain"])
        self.update_gain(s["gain"])
        self.binning_dropdown.set(s["binning"])
        self.update_binning()
        self.format_var.set(s["format"])
        self.update_image_format()

    def _focus_cancel_sweep(self):
        self._focus_generation += 1
        self._focus_cancel.set()
        try:
            self._focus_countdown_label.configure(text="Cancelled.")
            self._focus_phase_label.configure(text="Idle (cancelled).")
            self._focus_start_btn.configure(state="normal" if self.camera and self.camera_initialized else "disabled")
            self._focus_cancel_btn.configure(state="disabled")
        except Exception:
            pass
        if self._focus_saved_camera:
            self._focus_apply_snapshot_to_ui(self._focus_saved_camera)
        self._focus_cancel.clear()

    def _focus_run_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.save_path, "FocusAssist", ts)

    def _focus_start_sweep(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Focus assist", "Connect a camera first.")
            return
        try:
            settle_s = max(0, int(self._focus_settle_var.get().strip()))
            n_turns = max(1, int(self._focus_turns_var.get().strip()))
        except ValueError:
            messagebox.showwarning("Focus assist", "Enter integers for settle time and quarter-turns.")
            return

        self._maybe_warn_solar_filter_daytime("running the focus sweep")

        if self._focus_saved_camera is None:
            self._focus_save_camera_snapshot()

        fn = self._focus_fn_var.get()
        if fn != "Custom":
            preset = self._FOCUS_FN_PRESETS.get(fn)
            if preset:
                rng, exp, gain = preset
                gain = self._clamp_gain_for_camera(gain)
                self.exposure_range_var.set(rng)
                self.update_exposure_range()
                self.exposure_var.set(float(exp))
                self.exposure_slider.set(float(exp))
                self.update_exposure(float(exp))
                self.gain_var.set(int(gain))
                self.gain_slider.set(int(gain))
                self.update_gain(int(gain))
                self.binning_dropdown.set("1x1")
                self.update_binning()
                self.format_var.set("RAW16")
                self.update_image_format()

        n_captures = n_turns + 1
        self._focus_scores = []
        self._focus_sweep_metric_mode = None
        self._focus_cancel.clear()
        self._focus_generation += 1
        gen = self._focus_generation

        self._focus_results.delete("1.0", tk.END)
        run_dir = self._focus_run_dir()
        try:
            os.makedirs(run_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Focus assist", f"Cannot create folder:\n{e}")
            return

        self._focus_was_previewing = self._stop_preview_for_still()
        range_text = self.exposure_range_var.get()
        exp_val = float(self.exposure_var.get())
        gain = int(self.gain_var.get())
        bin_value = int(self.binning_dropdown.get().split("x")[0])
        img_fmt = self.format_var.get()

        try:
            self._configure_still_capture_hardware(
                bin_value=bin_value,
                image_format=img_fmt,
                range_text=range_text,
                exposure_val=exp_val,
                gain=gain,
            )
        except Exception as e:
            messagebox.showerror("Focus assist", str(e))
            if getattr(self, "_focus_was_previewing", False):
                self.start_preview()
            return

        self._focus_start_btn.configure(state="disabled")
        self._focus_cancel_btn.configure(state="normal")
        self._focus_phase_label.configure(text="Prepare: point at stars and move focus to your starting end of travel.")
        self._focus_countdown_label.configure(text="")

        toward = self._focus_dir_var.get()
        toward_txt = "infinity" if toward == "infinity" else "near focus"

        def after_sweep_done():
            if gen != self._focus_generation:
                return
            self._focus_start_btn.configure(state="normal" if self.camera and self.camera_initialized else "disabled")
            self._focus_cancel_btn.configure(state="disabled")
            if getattr(self, "_focus_was_previewing", False):
                self.start_preview()
            if not self._focus_scores:
                self._focus_phase_label.configure(text="Sweep finished (no scores).")
                return
            qualities = [q for _, q, _ in self._focus_scores]
            best_i = int(np.argmax(qualities))
            best_q = qualities[best_i]
            last_i = len(self._focus_scores) - 1
            delta = last_i - best_i
            opp = "near focus" if toward == "infinity" else "infinity"
            mode_note = f"Metric mode for this run: {self._focus_sweep_metric_mode or 'n/a'}"
            lines = [
                f"Folder: {run_dir}",
                mode_note,
                f"Best sample: step {best_i} (0 = first capture at start position)",
                f"Metric (higher is sharper): {best_q:.4g}",
                "",
            ]
            for i, q, desc in self._focus_scores:
                lines.append(f"  step {i:02d}: {q:.4g}  ({desc})")
            lines.append("")
            if delta == 0:
                lines.append("You are already at the best focus position (last step).")
            else:
                lines.append(
                    f"From your current focus position (step {last_i}), turn the focuser {delta} quarter-turn(s) "
                    f"toward {opp} (opposite to the sweep direction), hands off between turns, then verify with preview."
                )
            self._focus_results.insert(tk.END, "\n".join(lines))
            self._focus_phase_label.configure(text=f"Best focus at step {best_i}. See results below.")
            self.update_status("Focus sweep complete.")

        def capture_step(step_idx: int):
            if gen != self._focus_generation or self._focus_cancel.is_set():
                after_sweep_done()
                return

            def do_cap():
                if gen != self._focus_generation or self._focus_cancel.is_set():
                    self.root.after(0, after_sweep_done)
                    return
                try:
                    frame = self._grab_still_frame()
                    q, desc = self._focus_quality_for_sweep_step(frame, img_fmt)
                    fn_out = os.path.join(run_dir, f"step_{step_idx:03d}.fits")
                    _, exp_s = self._exposure_us_display_seconds(range_text, exp_val)
                    header = fits.Header()
                    header["INSTRUME"] = self.camera_info["Name"]
                    header["EXPOSURE"] = (exp_s, "Exposure time in seconds")
                    header["GAIN"] = (gain, "Gain")
                    header["BINNING"] = (bin_value, "Binning")
                    header["FOCUSSTEP"] = (step_idx, "Focus sweep step index")
                    header["FOCUSMETR"] = (q, "Focus quality (higher sharper)")
                    fits.PrimaryHDU(data=frame, header=header).writeto(fn_out, overwrite=True)
                    self._focus_scores.append((step_idx, q, desc))
                    self.root.after(
                        0,
                        lambda s=step_idx, ds=desc, tot=n_captures: self._focus_progress_label.configure(
                            text=f"Captured step {s + 1}/{tot} — {ds}"
                        ),
                    )
                except Exception as e:
                    self.root.after(
                        0,
                        lambda err=e, si=step_idx: messagebox.showerror(
                            "Focus assist", f"Capture failed at step {si}:\n{err}"
                        ),
                    )
                    self.root.after(0, after_sweep_done)
                    return

                if step_idx >= n_captures - 1:
                    self.root.after(0, after_sweep_done)
                    return

                def next_turn():
                    if gen != self._focus_generation or self._focus_cancel.is_set():
                        self.root.after(0, after_sweep_done)
                        return
                    self._focus_phase_label.configure(
                        text=(
                            f"Step {step_idx + 2}/{n_captures}: rotate the focuser exactly ONE quarter-turn "
                            f"toward {toward_txt}. Remove your hands from the telescope."
                        )
                    )
                    self._schedule_focus_settle(settle_s, lambda: capture_step(step_idx + 1))

                self.root.after(0, next_turn)

            threading.Thread(target=do_cap, daemon=True).start()

        def settle_then_first():
            if gen != self._focus_generation or self._focus_cancel.is_set():
                after_sweep_done()
                return
            self._focus_phase_label.configure(
                text="Starting position: hands off the scope. Wait for the countdown, then we capture."
            )
            self._schedule_focus_settle(settle_s, lambda: capture_step(0))

        settle_then_first()

    def _schedule_focus_settle(self, settle_s: int, on_done):
        if self._focus_cancel.is_set():
            return

        def tick(remaining: int):
            if self._focus_cancel.is_set():
                return
            if remaining <= 0:
                self._focus_countdown_label.configure(text="Capturing now.")
                try:
                    self.root.bell()
                except Exception:
                    pass
                on_done()
                return
            self._focus_countdown_label.configure(
                text=f"Stabilizing… {remaining}s — hands off the tripod / scope."
            )
            self.root.after(1000, lambda: tick(remaining - 1))

        tick(settle_s)

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
        live = self._center_tabs.tab("Live preview")
        self.preview_label = tk.Label(
            live,
            text="No preview available\nConnect camera and start preview",
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._center_tabs.add("Session photos")
        session_tab = self._center_tabs.tab("Session photos")
        sbtn = ctk.CTkFrame(session_tab, fg_color="transparent")
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
            text="Use last capture",
            command=self._session_use_last_for_plate,
            width=140,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkButton(
            sbtn,
            text="Scan for solved FITS",
            command=self._backfill_session_radec_from_disk,
            width=150,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)
        list_wrap = tk.Frame(session_tab)
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
        site = self._center_tabs.tab("Site / Sky")
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

        self._center_tabs.add("Plate solve")
        plate = self._center_tabs.tab("Plate solve")
        self._build_plate_solve_tab(plate)

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

        outer = ctk.CTkFrame(plate, fg_color=pal["bg"], border_color=pal["border"], border_width=0)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
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

        # Preset buttons row.
        preset_btn_row = ctk.CTkFrame(preset_box, fg_color="transparent")
        _track(preset_btn_row)
        preset_btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        _btn(
            preset_btn_row,
            "Apply Tucson SCT @ 2032 mm",
            self._apply_default_tucson_preset,
            primary=True,
            width=200,
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
        _btn(
            action_row,
            "SOLVE",
            self._run_embedded_plate_solve,
            primary=True,
            width=200,
        ).pack(side=tk.LEFT, padx=(0, 6))
        _btn(
            action_row,
            "Download indexes for FOV",
            self._download_recommended_astrometry_indexes,
            width=240,
        ).pack(side=tk.LEFT, padx=4)

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
        if ra is None or dec is None:
            messagebox.showwarning(
                "Plate solve",
                "Enter RA and Dec hints (decimal degrees or sexagesimal). "
                "You can use the Resolve, Use mount, or Use last solve buttons.",
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

        self._plate_out.insert(
            tk.END,
            f"Solving with hints: RA {_deg_to_hms(ra)} ({ra:.4f}°)  "
            f"Dec {_deg_to_dms(dec)} ({dec:.4f}°)  FOV {fov:.3f}°\n"
            "(multi-pass with auto astrometry setup and PNG fallback; this may take a while)\n",
        )
        self._plate_out.see(tk.END)

        def worker():
            base = Path(path).stem + "-astra"
            try:
                res = run_solve_field_local(path, ra, dec, fov, out_basename=base, timeout=400)
            except Exception as e:
                self.root.after(0, lambda: self._plate_solve_done(None, str(e)))
                return
            self.root.after(0, lambda: self._plate_solve_done(res, None))

        threading.Thread(target=worker, daemon=True).start()

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
                self._plate_out.insert(tk.END, f"Center RA {cra:.5f}°  Dec {cdec:.5f}°\n")
                inp = getattr(self, "_last_plate_solve_input_path", None)
                if inp:
                    self._update_session_photo_solved(inp, cra, cdec)
            except Exception as e:
                self._plate_out.insert(tk.END, f"Could not read center from WCS: {e}\n")
        self._plate_out.see(tk.END)

    def apply_theme(self):
        """Apply the current theme to all UI elements"""
        t = self.theme

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

        # Style the scrollable panel
        try:
            self.left_scroll.configure(
                fg_color=t["bg_primary"],
                scrollbar_button_color=t["bg_tertiary"],
                scrollbar_button_hover_color=t["bg_secondary"],
            )
        except:
            pass

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
        action_buttons = [self.start_preview_btn, self.capture_img_btn, self.record_video_btn]
        fs = getattr(self, "_focus_start_btn", None)
        if fs is not None:
            action_buttons.append(fs)
        small_buttons = [self.connect_btn, self.refresh_btn, self.browse_btn]
        for fb in (getattr(self, "_focus_arm_btn", None), getattr(self, "_focus_cancel_btn", None), getattr(self, "_focus_restore_btn", None)):
            if fb is not None:
                small_buttons.append(fb)

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

        # Path entry
        self.path_entry.configure(
            fg_color=t["bg_secondary"] if self.night_mode else "white",
            text_color=t["fg_primary"],
            border_color=t["border"]
        )

        # Project name entry
        self.project_name_entry.configure(
            fg_color=t["bg_secondary"] if self.night_mode else "white",
            text_color=t["fg_primary"],
            placeholder_text_color=t["fg_primary"],
            border_color=t["border"]
        )

        try:
            self._center_tabs.configure(
                fg_color=t["bg_primary"],
                segmented_button_fg_color=t["bg_tertiary"],
                segmented_button_selected_color=t["accent"],
                segmented_button_selected_hover_color=t["accent_light"],
                segmented_button_unselected_color=t["bg_secondary"],
                segmented_button_unselected_hover_color=t["bg_tertiary"],
                text_color=t["fg_primary"],
            )
        except Exception:
            pass

        try:
            self._camera_tabs.configure(
                fg_color=t["bg_primary"],
                segmented_button_fg_color=t["bg_tertiary"],
                segmented_button_selected_color=t["accent"],
                segmented_button_selected_hover_color=t["accent_light"],
                segmented_button_unselected_color=t["bg_secondary"],
                segmented_button_unselected_hover_color=t["bg_tertiary"],
                text_color=t["fg_primary"],
            )
        except Exception:
            pass

        try:
            lb_bg = t["canvas_bg"] if not self.night_mode else t["bg_tertiary"]
            lb_fg = t["fg_primary"]
            self._session_listbox.configure(bg=lb_bg, fg=lb_fg, selectbackground=t["accent"])
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

        combo_settings = {
            "fg_color": t["bg_secondary"] if self.night_mode else "white",
            "text_color": t["fg_primary"] if self.night_mode else "#1a1a1a",
            "border_color": t["border"] if self.night_mode else "#cccccc",
            "button_color": t["accent"],
            "button_hover_color": t["accent_light"],
            "dropdown_fg_color": t["bg_tertiary"] if self.night_mode else "white",
            "dropdown_hover_color": t["bg_secondary"] if self.night_mode else "#f0f0f0"
        }

        focus_combo = getattr(self, "_focus_fn_menu", None)
        combo_list = [
            self.camera_dropdown,
            self.binning_dropdown,
            self.format_dropdown,
            self.capture_count_dropdown,
            self.exposure_range_dropdown,
            self.frame_type_dropdown,
        ]
        if focus_combo is not None:
            combo_list.append(focus_combo)
        for combo in combo_list:
            combo.configure(**combo_settings)

        slider_settings = {
            "button_color": t["accent"],
            "button_hover_color": t["accent_light"],
            "progress_color": t["accent"],
            "fg_color": t["bg_tertiary"]
        }

        self.exposure_slider.configure(**slider_settings)
        self.gain_slider.configure(**slider_settings)

        entry_settings = {
            "fg_color": t["bg_secondary"] if self.night_mode else "white",
            "text_color": t["fg_primary"],
            "border_color": t["border"],
        }
        for wname in ("_focus_settle_entry", "_focus_turns_entry"):
            w = getattr(self, wname, None)
            if w is not None:
                try:
                    w.configure(**entry_settings)
                except Exception:
                    pass

        tb_bg = t["canvas_bg"] if not self.night_mode else t["bg_tertiary"]
        fr = getattr(self, "_focus_results", None)
        if fr is not None:
            try:
                fr.configure(fg_color=tb_bg, text_color=t["fg_primary"], border_color=t["border"])
            except Exception:
                pass

        mp = getattr(self, "mount_panel", None)
        if mp is not None:
            try:
                mp.apply_camgui_theme(t)
            except Exception:
                pass

        # Plate-solve tab uses a permanent red palette so the field-night view
        # stays dark-adapted regardless of day/night mode.
        self._reapply_plate_red_palette()

    def _reapply_plate_red_palette(self):
        """Force the plate-solve tab back to its red palette after a theme change."""
        pal = PLATE_RED_PAL
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
            self._focus_update_start_enabled()

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
            self._focus_update_start_enabled()

        def fail(exc):
            self.camera = None
            self.camera_initialized = False
            try:
                self.connect_btn.configure(state="normal")
            except Exception:
                pass
            messagebox.showerror("Error", f"Failed to connect to camera: {exc}")
            self.update_status("Failed to connect to camera")
            self._focus_update_start_enabled()

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

    def toggle_preview(self):
        if not self.is_capturing:
            self.start_preview()
            self.start_preview_btn.configure(text="Stop Preview")
        else:
            self.stop_preview()
            self.start_preview_btn.configure(text="Start Preview")

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
            if self._sun_crosshair_enabled:
                w, h = img_pil.size
                cx, cy = w // 2, h // 2
                s = int(self._sun_crosshair_size_px)
                t = int(self._sun_crosshair_thickness)
                dr = ImageDraw.Draw(img_pil)
                color = (255, 80, 80) if self.night_mode else (255, 0, 0)
                dr.line((cx - s, cy, cx + s, cy), fill=color, width=t)
                dr.line((cx, cy - s, cx, cy + s), fill=color, width=t)
                dr.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color, outline=color)
                hint = "Center crosshair on Sun, then click 'Aligned \u2014 Sync & Track' in the mount panel"
                text_y = max(8, cy - s - 24)
                # Soft background stripe keeps text readable on bright solar frames.
                dr.rectangle((6, text_y - 2, w - 6, text_y + 16), fill=(0, 0, 0))
                dr.text((10, text_y), hint, fill=(255, 140, 140) if self.night_mode else (255, 230, 230))
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
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    root.mainloop()


if __name__ == "__main__":
    main()
