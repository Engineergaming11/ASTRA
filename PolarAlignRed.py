"""
Sky Chart — interactive polar view of the celestial sphere.
CustomTkinter UI with embedded matplotlib chart.

Plot stars by entering Right Ascension (HH:MM:SS) and Declination (±DD:MM:SS).
Uses a stereographic projection centered on the celestial pole. The chart
rotates so that the current Local Sidereal Time (LST) sits at the top.

Run:
    python PolarAlign_ctk.py
"""

from datetime import datetime, timezone
import math
import re

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

from ui_display_profile import ui_compact
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# skyfield is used for apparent position of Polaris
try:
    from skyfield.api import load, Star
    _SKYFIELD_OK = True
except ImportError:
    _SKYFIELD_OK = False


# ─────────────────────────── Polaris apparent position ───────────────────────

def polaris_apparent(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    if not _SKYFIELD_OK:
        return 2.5303, 89.2641
    ts = load.timescale()
    t = ts.from_datetime(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))
    polaris = Star(
        ra_hours=2.530301028,
        dec_degrees=89.264109444,
        ra_mas_per_year=44.22,
        dec_mas_per_year=-11.74,
        parallax_mas=7.54,
        radial_km_per_s=-17.4,
    )
    from skyfield.api import load as sf_load
    eph = sf_load("de421.bsp")
    ra, dec, _ = eph["earth"].at(t).observe(polaris).apparent().radec("date")
    return ra.hours, dec.degrees


# ─────────────────────────── Coordinate helpers ──────────────────────────────

def parse_ra(s):
    s = s.strip()
    parts = [p for p in re.split(r"[\s:hms]+", s) if p]
    if not parts:
        raise ValueError("empty RA")
    if len(parts) == 1:
        return float(parts[0])
    h = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0
    sec = float(parts[2]) if len(parts) > 2 else 0
    return h + m / 60 + sec / 3600


def parse_dec(s):
    s = s.strip()
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    parts = [p for p in re.split(r"[\s:dm'\"°]+", s) if p]
    if not parts:
        raise ValueError("empty Dec")
    if len(parts) == 1:
        return sign * float(parts[0])
    d = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0
    sec = float(parts[2]) if len(parts) > 2 else 0
    return sign * (d + m / 60 + sec / 3600)


def local_sidereal_time(longitude, dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60 + (dt.second + dt.microsecond / 1e6) / 3600
    if month <= 2:
        year -= 1
        month += 12
    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    JD = (math.floor(365.25 * (year + 4716))
          + math.floor(30.6001 * (month + 1))
          + day + B - 1524.5 + hour / 24)
    T = (JD - 2451545.0) / 36525.0
    GMST = (280.46061837
            + 360.98564736629 * (JD - 2451545.0)
            + 0.000387933 * T ** 2
            - T ** 3 / 38710000.0)
    LST_deg = (GMST + longitude) % 360
    if LST_deg < 0:
        LST_deg += 360
    return LST_deg / 15.0


def format_hms(hours):
    h = int(hours) % 24
    mf = (hours - int(hours)) * 60
    m = int(mf)
    s = (mf - m) * 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def format_dms(degrees):
    sign = "-" if degrees < 0 else "+"
    a = abs(degrees)
    d = int(a)
    mf = (a - d) * 60
    m = int(mf)
    s = (mf - m) * 60
    return f"{sign}{d:02d}:{m:02d}:{s:04.1f}"


# ─────────────────────────── Projection ──────────────────────────────────────

def project(ra_hours, dec_deg, pole_sign=1, lst_hours=0.0):
    ra = np.asarray(ra_hours, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    co_lat = 90 - pole_sign * dec
    co_lat_rad = np.deg2rad(co_lat)
    r = 2 * np.tan(co_lat_rad / 2)
    ha = (lst_hours - ra) * 15
    ha_rad = np.deg2rad(ha)
    angle = np.pi / 2 + pole_sign * ha_rad
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    if np.ndim(x) == 0:
        return (np.nan, np.nan) if co_lat >= 180 else (float(x), float(y))
    x = np.where(co_lat >= 180, np.nan, x)
    y = np.where(co_lat >= 180, np.nan, y)
    return x, y


def edge_radius(fov_deg):
    return 2 * np.tan(np.deg2rad(fov_deg) / 2)


# ─────────────────────────── Night Theme (matches CamGUI) ────────────────────
# Mirrors THEMES["night"] from CamGUI_fix7.py so this window blends with the
# rest of the suite when the observer is dark-adapted.

BG_PRIMARY    = "#1a0a05"   # window background
BG_SECONDARY  = "#2d1410"   # panel / entry background
BG_TERTIARY   = "#3d1f1a"   # slider track / hover
FG_PRIMARY    = "#ff6666"   # primary red text
FG_SECONDARY  = "#ff6666"
FG_TERTIARY   = "#ff6666"
ACCENT        = "#ff4444"   # bright red — buttons / highlights
ACCENT_LIGHT  = "#ff6666"   # hover state
CANVAS_BG     = "#0d0503"   # matplotlib chart background
BORDER        = "#4d2020"   # dark red border
BTN_TEXT      = "#fccfcf"   # button label color (matches CamGUI night btn_text_color)

# Star plot colors — kept distinct from UI red so they remain readable on
# the dark chart, but desaturated to fit the night palette.
STAR_COL      = "#ffb3b3"
POLARIS_COL   = "#ffcccc"


# ─────────────────────────── App ─────────────────────────────────────────────

class SkyChartApp(ctk.CTk):
    DEFAULT_LON = -110.9265

    def __init__(self):
        super().__init__()

        # ── state ──
        self.stars      = []
        self.pole_sign  = 1
        self.fov        = 2.0
        self.longitude  = self.DEFAULT_LON
        self.lst        = local_sidereal_time(self.longitude)
        self._chart_artists = []

        # ── window ──
        ctk.set_appearance_mode("dark")
        self.title("Polar Align — Sky Chart")
        if ui_compact(self):
            self.geometry("1008x580")
        else:
            self.geometry("1100x750")
        self.configure(fg_color=BG_PRIMARY)
        self.resizable(True, True)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._build_layout()
        self._seed_examples()
        self._redraw()
        self._tick()   # start 10-second LST refresh loop

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        # TOP BAR (matches CamGUI top bar styling)
        self.top_bar = tk.Frame(self, bg=BG_PRIMARY)
        self.top_bar.pack(fill=tk.X, padx=12, pady=8)

        self.title_label = tk.Label(
            self.top_bar,
            text="POLAR ALIGN",
            font=("Segoe UI", 16, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        )
        self.title_label.pack(side=tk.LEFT)

        spacer = tk.Frame(self.top_bar, bg=BG_PRIMARY)
        spacer.pack(side=tk.LEFT, expand=True)

        # LST clock display (right side of top bar)
        self.lbl_lst = tk.Label(
            self.top_bar,
            text="LST --:--:--",
            font=("Segoe UI", 10, "bold"),
            bg=BG_PRIMARY,
            fg=ACCENT,
            padx=12,
        )
        self.lbl_lst.pack(side=tk.RIGHT, padx=(0, 12))

        # MAIN CONTAINER
        self.main = tk.Frame(self, bg=BG_PRIMARY)
        self.main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # LEFT PANEL — control panel (scrollable, mirrors CamGUI left panel)
        self.left_scroll = ctk.CTkScrollableFrame(
            self.main,
            width=290,
            fg_color=BG_PRIMARY,
            scrollbar_button_color=BG_TERTIARY,
            scrollbar_button_hover_color=BG_SECONDARY,
        )
        self.left_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.left = self.left_scroll

        tk.Label(
            self.left,
            text="STAR INPUT",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(0, 8))

        # RA
        tk.Label(
            self.left,
            text="Right Ascension (HH:MM:SS):",
            font=("Segoe UI", 9),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(4, 0))

        self.entry_ra = ctk.CTkEntry(
            self.left,
            placeholder_text="HH:MM:SS",
            font=("Segoe UI", 10),
            height=28,
            fg_color=BG_SECONDARY,
            text_color=FG_PRIMARY,
            placeholder_text_color=FG_TERTIARY,
            border_color=BORDER,
            border_width=1,
            corner_radius=6,
        )
        self.entry_ra.insert(0, "14:15:40")
        self.entry_ra.pack(fill=tk.X, pady=4)
        self.entry_ra.bind("<Return>", lambda e: self._on_plot())

        # Dec
        tk.Label(
            self.left,
            text="Declination (±DD:MM:SS):",
            font=("Segoe UI", 9),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(8, 0))

        self.entry_dec = ctk.CTkEntry(
            self.left,
            placeholder_text="±DD:MM:SS",
            font=("Segoe UI", 10),
            height=28,
            fg_color=BG_SECONDARY,
            text_color=FG_PRIMARY,
            placeholder_text_color=FG_TERTIARY,
            border_color=BORDER,
            border_width=1,
            corner_radius=6,
        )
        self.entry_dec.insert(0, "+19:10:49")
        self.entry_dec.pack(fill=tk.X, pady=4)
        self.entry_dec.bind("<Return>", lambda e: self._on_plot())

        # Plot / Clear buttons (CamGUI action button style)
        def make_action_btn(text, command):
            return ctk.CTkButton(
                self.left,
                text=text,
                command=command,
                font=("Segoe UI", 9, "bold"),
                height=36,
                fg_color=ACCENT,
                text_color=BTN_TEXT,
                hover_color=ACCENT_LIGHT,
                corner_radius=6,
            )

        self.plot_btn = make_action_btn("Plot Star", self._on_plot)
        self.plot_btn.pack(fill=tk.X, pady=(10, 4))

        self.clear_btn = ctk.CTkButton(
            self.left,
            text="Clear",
            command=self._on_clear,
            font=("Segoe UI", 9, "bold"),
            height=32,
            fg_color=BG_SECONDARY,
            text_color=FG_PRIMARY,
            hover_color=BG_TERTIARY,
            border_color=BORDER,
            border_width=1,
            corner_radius=6,
        )
        self.clear_btn.pack(fill=tk.X, pady=4)

        # OBSERVER section
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left,
            text="OBSERVER",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(
            self.left,
            text="Longitude (°, East +):",
            font=("Segoe UI", 9),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(4, 0))

        self.entry_lon = ctk.CTkEntry(
            self.left,
            font=("Segoe UI", 10),
            height=28,
            fg_color=BG_SECONDARY,
            text_color=FG_PRIMARY,
            border_color=BORDER,
            border_width=1,
            corner_radius=6,
        )
        self.entry_lon.insert(0, f"{self.longitude:.4f}")
        self.entry_lon.pack(fill=tk.X, pady=4)
        self.entry_lon.bind("<Return>", lambda e: self._on_longitude())

        # FOV slider
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left,
            text="FIELD OF VIEW",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(0, 8))

        fov_frame = tk.Frame(self.left, bg=BG_PRIMARY)
        fov_frame.pack(fill=tk.X, pady=4)

        self.slider_fov = ctk.CTkSlider(
            fov_frame,
            from_=0.1,
            to=5.0,
            number_of_steps=49,
            button_color=ACCENT,
            button_hover_color=ACCENT_LIGHT,
            progress_color=ACCENT,
            fg_color=BG_TERTIARY,
            width=160,
            height=16,
            command=self._on_fov,
        )
        self.slider_fov.set(self.fov)
        self.slider_fov.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.lbl_fov_val = tk.Label(
            fov_frame,
            text=f"{self.fov:.1f}°",
            font=("Segoe UI", 9),
            width=8,
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        )
        self.lbl_fov_val.pack(side=tk.LEFT, padx=(4, 0))

        # Pole selector
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left,
            text="CENTER",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(0, 8))

        self.pole_var = ctk.StringVar(value="North pole")

        for label in ("North pole", "South pole"):
            ctk.CTkRadioButton(
                self.left,
                text=label,
                variable=self.pole_var,
                value=label,
                fg_color=ACCENT,
                hover_color=ACCENT_LIGHT,
                border_color=BORDER,
                text_color=FG_PRIMARY,
                font=("Segoe UI", 10),
                command=self._on_pole,
            ).pack(anchor="w", padx=4, pady=2)

        # STATUS section
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left,
            text="STATUS",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(anchor="w", pady=(0, 4))

        self.lbl_status = tk.Label(
            self.left,
            text="Ready",
            font=("Segoe UI", 9, "italic"),
            wraplength=240,
            justify=tk.LEFT,
            anchor=tk.W,
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        )
        self.lbl_status.pack(fill=tk.X)

        # RIGHT PANEL — chart
        self.right = tk.Frame(self.main, bg=BG_PRIMARY)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # chart header (title + subtitle + utc/longitude info)
        chart_header = tk.Frame(self.right, bg=BG_PRIMARY)
        chart_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            chart_header,
            text="CELESTIAL SPHERE",
            font=("Segoe UI", 11, "bold"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        ).pack(side=tk.LEFT)

        self.lbl_utc = tk.Label(
            chart_header,
            text="",
            font=("Segoe UI", 9),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        )
        self.lbl_utc.pack(side=tk.RIGHT)

        # subtitle row
        sub_row = tk.Frame(self.right, bg=BG_PRIMARY)
        sub_row.pack(fill=tk.X, pady=(0, 4))

        self.lbl_subtitle = tk.Label(
            sub_row,
            text="Equatorial grid · meridian at top · north celestial pole",
            font=("Segoe UI", 9, "italic"),
            bg=BG_PRIMARY,
            fg=FG_PRIMARY,
        )
        self.lbl_subtitle.pack(side=tk.LEFT)

        self.lbl_polaris = tk.Label(
            sub_row,
            text="",
            font=("Segoe UI", 9),
            bg=BG_PRIMARY,
            fg=POLARIS_COL,
        )
        self.lbl_polaris.pack(side=tk.RIGHT)

        # matplotlib figure — themed dark to match night palette
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.patch.set_facecolor(BG_PRIMARY)
        self.ax.set_facecolor(CANVAS_BG)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg=BG_PRIMARY, highlightthickness=0)
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    # ── seed ──────────────────────────────────────────────────────────────────

    def _seed_examples(self):
        pol_ra, pol_dec = polaris_apparent()
        self.stars.append({"ra": pol_ra, "dec": pol_dec, "label": "Polaris"})
        self.stars.append({"ra": 13.792, "dec": 49.313, "label": "Alkaid"})
        self.stars.append({"ra": 14.261, "dec": 19.182, "label": "Arcturus"})

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_plot(self):
        try:
            ra = parse_ra(self.entry_ra.get())
            dec = parse_dec(self.entry_dec.get())
        except (ValueError, IndexError):
            self.lbl_status.configure(text="⚠  Invalid coordinates")
            return
        ra = ra % 24
        if not (-90 <= dec <= 90):
            self.lbl_status.configure(text="⚠  Declination must be in [-90°, +90°]")
            return
        self.stars.append({
            "ra": ra, "dec": dec,
            "label": f"{format_hms(ra)}  {format_dms(dec)}",
        })
        self._redraw()

    def _on_clear(self):
        self.stars = [s for s in self.stars if s["label"] == "Polaris"]
        self._redraw()

    def _on_fov(self, val):
        self.fov = float(val)
        self.lbl_fov_val.configure(text=f"{self.fov:.1f}°")
        self._redraw()

    def _on_pole(self):
        label = self.pole_var.get()
        self.pole_sign = 1 if label == "North pole" else -1
        pole_name = "north" if self.pole_sign == 1 else "south"
        self.lbl_subtitle.configure(
            text=f"Equatorial grid · meridian at top · {pole_name} celestial pole")
        self._redraw()

    def _on_longitude(self):
        try:
            self.longitude = float(self.entry_lon.get().strip())
        except ValueError:
            self.lbl_status.configure(text="⚠  Invalid longitude")
            return
        self.lst = local_sidereal_time(self.longitude)
        self._redraw()

    def _tick(self):
        self.lst = local_sidereal_time(self.longitude)
        self._redraw()
        self._tick_id = self.after(10_000, self._tick)

    def _on_close(self):
        # cancel our own periodic tick
        if hasattr(self, '_tick_id'):
            try:
                self.after_cancel(self._tick_id)
            except Exception:
                pass

        # close the matplotlib figure so it releases its Tk references
        try:
            plt.close(self.fig)
        except Exception:
            pass

        # quit the mainloop first, THEN destroy. This lets pending
        # after() callbacks (e.g. CustomTkinter's internal _check_dpi_scaling
        # and update callbacks) unwind cleanly instead of firing against
        # already-destroyed widgets.
        self.quit()
        self.destroy()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        ax = self.ax

        for artist in self._chart_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._chart_artists = []

        def _add(a):
            self._chart_artists.append(a)
            return a

        r_edge = edge_radius(self.fov)
        ax.set_xlim(-r_edge * 1.05, r_edge * 1.05)
        ax.set_ylim(-r_edge * 1.05, r_edge * 1.05)

        # Grid color — desaturated red so it's visible on the near-black chart
        # background but still in palette with the night theme.
        grid_col = "#7a3030"

        # declination circles
        dec_step = 0.5 if self.fov <= 5 else (1 if self.fov <= 15 else 5)
        for i in range(1, int(self.fov / dec_step) + 2):
            co_lat = i * dec_step
            if co_lat > self.fov:
                break
            r = edge_radius(co_lat)
            major = (i * dec_step) % (dec_step * 5) == 0
            c = patches.Circle((0, 0), r, fill=False, edgecolor=grid_col,
                                alpha=0.6 if major else 0.35,
                                linewidth=0.8 if major else 0.4)
            ax.add_patch(c); _add(c)
            if major and co_lat < self.fov:
                dec_val = self.pole_sign * (90 - co_lat)
                lx, ly = project(self.lst, dec_val, self.pole_sign, self.lst)
                _add(ax.text(lx + r_edge * 0.01, ly - r_edge * 0.015,
                             f"{dec_val:+.0f}°", color=FG_PRIMARY,
                             alpha=0.75, fontsize=7, family="monospace"))

        # RA hour lines
        for h in range(24):
            xe, ye = project(h, self.pole_sign * (90 - self.fov), self.pole_sign, self.lst)
            major = h % 6 == 0
            _add(ax.plot([0, xe], [0, ye], color=grid_col,
                         alpha=0.6 if major else 0.35,
                         linewidth=0.8 if major else 0.4)[0])
            lx, ly = project(h, self.pole_sign * (90 - self.fov * 0.95), self.pole_sign, self.lst)
            _add(ax.text(lx, ly, f"{h}h", color=FG_PRIMARY, alpha=0.9,
                         fontsize=9, family="monospace", ha="center", va="center"))

        # meridian
        mx, my = project(self.lst, self.pole_sign * (90 - self.fov), self.pole_sign, self.lst)
        _add(ax.plot([0, mx], [0, my], color=ACCENT, alpha=0.5, linewidth=1.5, zorder=1)[0])
        _add(ax.text(mx, my + r_edge * 0.03, "MERIDIAN", color=ACCENT, alpha=0.95,
                     fontsize=8, family="monospace", ha="center", va="bottom", fontweight="bold"))

        # pole dot
        _add(ax.plot(0, 0, "o", color=FG_PRIMARY, markersize=3, alpha=0.9)[0])

        # stars
        visible = 0
        for s in self.stars:
            x, y = project(s["ra"], s["dec"], self.pole_sign, self.lst)
            if np.isnan(x) or np.hypot(x, y) > r_edge * 1.02:
                continue
            visible += 1
            _add(ax.plot(x, y, "o", color=ACCENT, markersize=14, alpha=0.25)[0])
            _add(ax.plot(x, y, "o", color=ACCENT, markersize=6,
                         markeredgecolor=FG_PRIMARY, markeredgewidth=0.5)[0])
            _add(ax.text(x + r_edge * 0.015, y + r_edge * 0.015, s["label"],
                         color=STAR_COL, fontsize=9, family="monospace"))

        self.canvas.draw_idle()

        # update labels
        pol_ra, pol_dec = polaris_apparent()
        self.lbl_polaris.configure(
            text=f"Polaris (α UMi)  RA {format_hms(pol_ra)}  Dec {format_dms(pol_dec)}")

        utc_now = datetime.now(timezone.utc)
        self.lbl_lst.configure(text=f"LST  {format_hms(self.lst)}")
        self.lbl_utc.configure(
            text=f"λ = {self.longitude:+.3f}°  ·  UTC {utc_now.strftime('%H:%M:%S')}")

        pole_name = "NCP" if self.pole_sign == 1 else "SCP"
        self.lbl_status.configure(
            text=f"Center: {pole_name}  ·  FOV: {self.fov:.1f}°  ·  "
                 f"Plotted: {len(self.stars)} (visible: {visible})  ·  "
                 f"Meridian RA: {format_hms(self.lst)}")


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == "__main__":
    app = SkyChartApp()
    app.mainloop()