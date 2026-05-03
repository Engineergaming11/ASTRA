"""Embeddable iOptron HAE16C mount control UI (CustomTkinter)."""
import customtkinter as ctk
import tkinter as tk
from astra_dialogs import askyesno, showerror, showwarning
import queue
import threading
import time
import serial.tools.list_ports
from typing import Callable, Dict, Optional, Tuple

from ioptron_mount import (
    IOptronHAE,
    IOptronHAEWifi,
    TRACK_SIDEREAL,
    TRACK_LUNAR,
    TRACK_SOLAR,
    WIFI_DEFAULT_IP,
    WIFI_DEFAULT_PORT,
    _deg_to_dms,
    _deg_to_hms,
    _encode_ra,
    _encode_dec,
    _parse_angle_dms,
    _parse_dec_input,
    _parse_ra_input,
)
from sky_ephemeris import radec_to_altaz_deg, sun_apparent_radec_deg
from ui_display_profile import ui_compact
from xy_scroll_frame import create_xy_scroll_area, theme_xy_area

import motor_rgb_led

# Tucson fixed-site assumption for daytime sun tracking workflow.
TUCSON_LAT = 32.2226
TUCSON_LON = -110.9747
TUCSON_ELEV_M = 728.0

# ═══════════════════════════════════════════════════════════════════════════════
#  Motor-column reds (darker than CamGUI night accent #ff4444 / hover #ff6666)
# ═══════════════════════════════════════════════════════════════════════════════

MOTOR_BTN = "#c62828"
MOTOR_BTN_HOVER = "#dc5050"
MOTOR_BTN_MUTED = "#6a1a1a"
MOTOR_BTN_MUTED_HOVER = "#802222"


# ═══════════════════════════════════════════════════════════════════════════════
#  Colour palette (standalone window — dark theme)
# ═══════════════════════════════════════════════════════════════════════════════

STANDALONE_PAL: Dict[str, str] = {
    "bg":             "#0d0f14",
    "panel":          "#13161e",
    "border":         "#1e2330",
    "accent":         MOTOR_BTN_HOVER,
    "accent2":        MOTOR_BTN,
    "accent2_hover":  MOTOR_BTN_HOVER,
    "accent3":        MOTOR_BTN,
    "accent3_hover":  MOTOR_BTN_HOVER,
    "danger":         MOTOR_BTN,
    "danger_hover":   MOTOR_BTN_HOVER,
    "success":        "#22c55e",
    "text":           "#e2e8f0",
    "muted":          "#64748b",
    "entry_bg":       "#1a1f2e",
    "btn_hover":      MOTOR_BTN_HOVER,
    "motor_btn":         MOTOR_BTN,
    "motor_btn_hover":   MOTOR_BTN_HOVER,
    "motor_btn_muted":   MOTOR_BTN_MUTED,
    "motor_btn_muted_hover": MOTOR_BTN_MUTED_HOVER,
    "track_on":       MOTOR_BTN,
    "track_off":      MOTOR_BTN_MUTED,
    "track_off_hover": MOTOR_BTN_MUTED_HOVER,
}

# Backward compatibility (e.g. EQMountControls standalone launcher)
PAL = STANDALONE_PAL


def _mount_palette_for_camgui_theme(t: Dict[str, str]) -> Dict[str, str]:
    """Semantic colours aligned with ``CamGUI.THEMES`` day/night tokens."""
    is_day = t.get("bg_primary") == "#f5f5f5"
    # Match CamGUI red field palette in day mode (avoid bright white entry wells).
    entry_bg = "#2d1410" if is_day else t["bg_tertiary"]
    return {
        "bg":              t["bg_primary"],
        "panel":           t["bg_secondary"],
        "border":          t["border"],
        "accent":          t["accent"],
        "accent2":         MOTOR_BTN,
        "accent2_hover":   MOTOR_BTN_HOVER,
        "accent3":         MOTOR_BTN,
        "accent3_hover":   MOTOR_BTN_HOVER,
        "danger":          MOTOR_BTN,
        "danger_hover":    MOTOR_BTN_HOVER,
        "success":         "#16a34a",
        "text":            t["fg_primary"],
        "muted":           t["fg_tertiary"],
        "entry_bg":        entry_bg,
        "btn_hover":       MOTOR_BTN_HOVER,
        "motor_btn":         MOTOR_BTN,
        "motor_btn_hover":   MOTOR_BTN_HOVER,
        "motor_btn_muted":   MOTOR_BTN_MUTED,
        "motor_btn_muted_hover": MOTOR_BTN_MUTED_HOVER,
        "track_on":        MOTOR_BTN,
        "track_off":       MOTOR_BTN_MUTED,
        "track_off_hover": MOTOR_BTN_MUTED_HOVER,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════════

class MountControlsFrame(ctk.CTkFrame):

    def __init__(
        self,
        parent,
        *,
        standalone: bool = False,
        tracking_prerequisite: Optional[Callable[[], Tuple[bool, str]]] = None,
        camgui_theme: Optional[Dict[str, str]] = None,
    ):
        self._match_camgui = camgui_theme is not None
        self._pal: Dict[str, str] = (
            _mount_palette_for_camgui_theme(camgui_theme)
            if self._match_camgui
            else dict(STANDALONE_PAL)
        )
        self._panel_outer_frames: list[ctk.CTkFrame] = []
        self._section_title_labels: list[ctk.CTkLabel] = []
        self._section_separators: list[ctk.CTkFrame] = []
        super().__init__(parent, fg_color=self._pal["bg"])
        self._standalone = standalone
        self._tracking_prerequisite = tracking_prerequisite
        self.mount: Optional[IOptronHAE] = None
        self._polling       = False
        self._poll_thread   = None
        self._tracking_on   = False
        self._tracking_rate = TRACK_SIDEREAL
        self._conn_mode     = tk.StringVar(value="wifi")
        self._nudge_active: set = set()
        self._nudge_cmd_queue = queue.Queue()
        self._nudge_worker_thread: Optional[threading.Thread] = None
        self._mount_xy_areas: list = []
        # Compact = small Pi display OR embedded into a narrow side column.
        # Stacked rows + tighter padding make every control reachable without
        # horizontal scrolling at any width down to ~240 px.
        self._compact = ui_compact(self) or not standalone

        self._build_ui()
        self._refresh_ports()
        self._set_mount_controls(False)
        motor_rgb_led.ensure_idle_display()

    def _build_ui(self):
        self._build_header()

        # Connection always visible above mount tabs
        _outer_padx = 6 if self._compact else 8
        _outer_pady_btm = 4 if self._compact else 6
        self._conn_outer = ctk.CTkFrame(self, fg_color=self._pal["bg"])
        self._conn_outer.pack(fill="x", padx=_outer_padx, pady=(0, _outer_pady_btm))
        self._build_connection_panel(self._conn_outer)

        self._tabs = ctk.CTkTabview(
            self,
            fg_color=self._pal["bg"],
            segmented_button_fg_color=self._pal["panel"],
            segmented_button_selected_color=self._pal["motor_btn"],
            segmented_button_selected_hover_color=self._pal["motor_btn_hover"],
            segmented_button_unselected_color=self._pal["panel"],
            segmented_button_unselected_hover_color=self._pal["border"],
            text_color=self._pal["text"],
        )
        self._tabs.pack(
            fill="both", expand=True,
            padx=4 if self._compact else 6,
            pady=(0, 6 if self._compact else 8),
        )
        self._tabs.add("Manual")
        self._tabs.add("Auto tracking")
        self._tabs.add("Setup")

        self._build_manual_tab(self._tabs.tab("Manual"))
        self._build_auto_tracking_tab(self._tabs.tab("Auto tracking"))
        self._build_set_tab(self._tabs.tab("Setup"))

    def _build_manual_tab(self, parent):
        area = create_xy_scroll_area(parent, bg=self._pal["bg"])
        self._mount_xy_areas.append(area)
        area.outer.pack(fill="both", expand=True)
        content = ctk.CTkFrame(area.inner, fg_color="transparent")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        # Single column avoids a wide RA/Dec + nudge side‑by‑side minimum on narrow panes.
        content.rowconfigure(0, weight=0)
        content.rowconfigure(1, weight=0)
        content.rowconfigure(2, weight=1)

        left = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        log_row = ctk.CTkFrame(content, fg_color="transparent")
        log_row.grid(row=2, column=0, sticky="nsew")

        self._build_position_panel(left)
        self._build_nudge_panel(right)
        self._build_log_panel(log_row)

    def _build_auto_tracking_tab(self, parent):
        area = create_xy_scroll_area(parent, bg=self._pal["bg"])
        self._mount_xy_areas.append(area)
        area.outer.pack(fill="both", expand=True)
        self._build_tracking_panel(area.inner)

    def _build_set_tab(self, parent):
        area = create_xy_scroll_area(parent, bg=self._pal["bg"])
        self._mount_xy_areas.append(area)
        area.outer.pack(fill="both", expand=True)
        content = ctk.CTkFrame(area.inner, fg_color="transparent")
        content.pack(fill="both", expand=True)

        panel = self._panel(content, "LOCATION")

        self._lat_entry = self._stacked_form_row(
            panel,
            label="Latitude",
            hint="±DD.dddd  or  ±DD:MM:SS   (N = +)",
            placeholder="e.g. 32:13:00",
            pady=(0, 8),
        )
        self._lon_entry = self._stacked_form_row(
            panel,
            label="Longitude",
            hint="±DDD.dddd  or  ±DDD:MM:SS  (E = +)",
            placeholder="e.g. -110:56:00",
            pady=(0, 12),
        )

        self._set_location_btn = ctk.CTkButton(
            panel, text="Set Lat / Long",
            height=38 if self._compact else 40, corner_radius=8,
            fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(size=13 if self._compact else 14, weight="bold"),
            command=self._set_location,
        )
        self._set_location_btn.pack(fill="x")

        # GoTo / stop / sync live here (removed from Manual tab — arrows only there)
        self._build_goto_panel(content)

        # ── Plate Solve Calibration panel ──────────────────────────────────────
        ps_panel = self._panel(content, "PLATE SOLVE CALIBRATION")

        self._ps_ra_entry = self._stacked_form_row(
            ps_panel,
            label="Target RA",
            hint="HH:MM:SS  or  decimal degrees",
            placeholder="e.g. 05:34:32",
            pady=(0, 8),
        )
        self._ps_dec_entry = self._stacked_form_row(
            ps_panel,
            label="Target Dec",
            hint="±DD:MM:SS  or  decimal degrees",
            placeholder="e.g. +22:00:52",
            pady=(0, 10),
        )

        # Pier side toggle: heading on top, two equal-width buttons below.
        ctk.CTkLabel(
            ps_panel, text="Pier Side  —  which side of the mount is the scope on?",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(size=11), anchor="w", justify="left",
        ).pack(anchor="w", pady=(0, 4))

        self._ps_pier_var = tk.StringVar(value="1")   # default West (most common for Polaris)
        pier_btn_frame = ctk.CTkFrame(ps_panel, fg_color="transparent")
        pier_btn_frame.pack(fill="x", pady=(0, 10))
        pier_btn_frame.columnconfigure((0, 1), weight=1, uniform="pier")

        _pier_h = 32 if self._compact else 34
        self._pier_east_btn = ctk.CTkButton(
            pier_btn_frame, text="◐  EAST", height=_pier_h, corner_radius=6,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn_muted_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=lambda: self._set_pier_side("0"),
        )
        self._pier_east_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._pier_west_btn = ctk.CTkButton(
            pier_btn_frame, text="◑  WEST", height=_pier_h, corner_radius=6,
            fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=lambda: self._set_pier_side("1"),
        )
        self._pier_west_btn.grid(row=0, column=1, sticky="ew")

        # CM calibrate button — short text so it never overflows on narrow widths.
        self._ps_calibrate_btn = ctk.CTkButton(
            ps_panel,
            text="⊕  CALIBRATE (Set RA / Dec / Pier)",
            height=42 if self._compact else 44, corner_radius=8,
            fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(size=12 if self._compact else 13, weight="bold"),
            command=self._calibrate_plate_solve,
        )
        self._ps_calibrate_btn.pack(fill="x")

    # ── Header ─────────────────────────────────────────────────────────────────

    def _build_header(self):
        # Lower header in compact mode reclaims ~10 px of vertical space and
        # keeps the title and status pill from colliding on narrow widths.
        _h = 36 if self._compact else 48
        self._hdr = ctk.CTkFrame(
            self, fg_color=self._pal["panel"], corner_radius=0, height=_h
        )
        self._hdr.pack(fill="x")
        self._hdr.pack_propagate(False)

        self._hdr_title = ctk.CTkLabel(
            self._hdr,
            text="◈  iOptron HAE16C" if self._compact else "◈  iOptron HAE16C  ·  Equatorial",
            font=ctk.CTkFont(
                family="Courier New",
                size=12 if self._compact else 13,
                weight="bold",
            ),
            text_color=self._pal["motor_btn"],
        )
        self._hdr_title.pack(side="left", padx=10 if self._compact else 12, pady=8)

        self._conn_indicator = ctk.CTkLabel(
            self._hdr,
            text="● OFFLINE" if self._compact else "● DISCONNECTED",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=self._pal["danger"],
        )
        self._conn_indicator.pack(side="right", padx=10 if self._compact else 12)

    # ── Connection panel ───────────────────────────────────────────────────────

    def _build_connection_panel(self, parent):
        panel = self._panel(parent, "CONNECTION")

        toggle_row = ctk.CTkFrame(panel, fg_color="transparent")
        toggle_row.pack(fill="x", pady=(0, 10))

        self._btn_mode_serial = ctk.CTkButton(
            toggle_row, text="⬡  SERIAL", width=90, height=30, corner_radius=6,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn_muted_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=lambda: self._switch_conn_mode("serial"),
        )
        self._btn_mode_serial.pack(side="left", padx=(0, 6))

        self._btn_mode_wifi = ctk.CTkButton(
            toggle_row, text="⬡  WI-FI", width=90, height=30, corner_radius=6,
            fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=lambda: self._switch_conn_mode("wifi"),
        )
        self._btn_mode_wifi.pack(side="left")

        # Serial sub-panel
        self._serial_panel = ctk.CTkFrame(panel, fg_color="transparent")
        row = ctk.CTkFrame(self._serial_panel, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(row, text="Port", text_color=self._pal["muted"],
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))
        self._port_var  = tk.StringVar(value="COM3")
        self._port_menu = ctk.CTkOptionMenu(
            row, variable=self._port_var, width=130,
            fg_color=self._pal["entry_bg"],
            button_color=self._pal["motor_btn"],
            button_hover_color=self._pal["motor_btn_hover"],
            text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=12),
        )
        self._port_menu.pack(side="left")
        self._refresh_ports_btn = ctk.CTkButton(
            row, text="⟳", width=36,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn"],
            text_color="#ffffff",
            command=self._refresh_ports,
        )
        self._refresh_ports_btn.pack(side="left", padx=6)

        # WiFi sub-panel (shown by default)
        self._wifi_panel = ctk.CTkFrame(panel, fg_color="transparent")
        wifi_fields = ctk.CTkFrame(self._wifi_panel, fg_color="transparent")
        wifi_fields.pack(fill="x", pady=(0, 8))
        wifi_fields.columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(wifi_fields, text="IP ADDRESS",
                     text_color=self._pal["muted"], font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(wifi_fields, text="PORT",
                     text_color=self._pal["muted"], font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self._wifi_ip_var = tk.StringVar(value=WIFI_DEFAULT_IP)
        self._wifi_ip_entry = ctk.CTkEntry(
            wifi_fields, textvariable=self._wifi_ip_var,
            fg_color=self._pal["entry_bg"], border_color=self._pal["border"],
            text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=13), height=38,
        )
        self._wifi_ip_entry.grid(row=1, column=0, sticky="ew")

        # StringVar (not IntVar): CTkEntry + IntVar raises TclError when the field is empty.
        self._wifi_port_var = tk.StringVar(value=str(WIFI_DEFAULT_PORT))
        self._wifi_port_entry = ctk.CTkEntry(
            wifi_fields, textvariable=self._wifi_port_var,
            fg_color=self._pal["entry_bg"], border_color=self._pal["border"],
            text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=13), height=38,
        )
        self._wifi_port_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        self._wifi_panel.pack(fill="x", pady=(0, 8))

        # Buttons — 2x2 grid so labels stay readable even on narrow side panels
        # (a single row of 4 buttons clipped them at ~240 px).
        btn_grid = ctk.CTkFrame(panel, fg_color="transparent")
        btn_grid.pack(fill="x")
        btn_grid.columnconfigure((0, 1), weight=1, uniform="conn_btns")

        _btn_h = 34 if self._compact else 36
        _btn_font = ctk.CTkFont(family="Courier New", size=11, weight="bold")

        self._connect_btn = ctk.CTkButton(
            btn_grid, text="CONNECT", height=_btn_h,
            fg_color=self._pal["accent2"], hover_color=self._pal["accent2_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            command=self._toggle_connect,
        )
        self._connect_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self._home_btn = ctk.CTkButton(
            btn_grid, text="⌂  GOTO ZERO", height=_btn_h,
            fg_color=self._pal["danger"], hover_color=self._pal["danger_hover"],
            font=_btn_font,
            text_color="#ffffff", command=self._goto_zero,
        )
        self._home_btn.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        self._btn_set_zero = ctk.CTkButton(
            btn_grid, text="SET ZERO HERE", height=_btn_h,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn"],
            text_color="#ffffff",
            font=_btn_font,
            command=self._set_zero_at_current,
        )
        self._btn_set_zero.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self._btn_mech_zero = ctk.CTkButton(
            btn_grid, text="FIND MECH 0", height=_btn_h,
            fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=_btn_font,
            command=self._search_mechanical_zero,
        )
        self._btn_mech_zero.grid(row=1, column=1, sticky="ew")

    # ── Live position panel ────────────────────────────────────────────────────

    def _build_position_panel(self, parent):
        panel = self._panel(parent, "CURRENT POSITION  —  live EQ coordinates")

        grid = ctk.CTkFrame(panel, fg_color="transparent")
        grid.pack(fill="x", expand=True)
        grid.columnconfigure(0, weight=0)
        grid.columnconfigure(1, weight=1, minsize=120)

        def big_row(row_idx, tag, color):
            ctk.CTkLabel(grid, text=tag, text_color=self._pal["muted"],
                         font=ctk.CTkFont(size=11), anchor="w",
                         ).grid(row=row_idx, column=0, sticky="w", pady=2, padx=(0, 8))
            var = tk.StringVar(value="---")
            val_lbl = ctk.CTkLabel(
                grid,
                textvariable=var,
                text_color=color,
                font=ctk.CTkFont(family="Courier New", size=15, weight="bold"),
                anchor="e",
            )
            val_lbl.grid(row=row_idx, column=1, sticky="ew", padx=(0, 4))
            return var, val_lbl

        self._disp_ra, self._disp_ra_lbl = big_row(0, "RA", self._pal["motor_btn"])
        self._disp_dec, self._disp_dec_lbl = big_row(1, "DEC", self._pal["motor_btn_hover"])
        self._disp_fw, self._disp_fw_lbl = big_row(2, "FW", self._pal["muted"])
        self._disp_pier, self._disp_pier_lbl = big_row(3, "PIER", self._pal["muted"])

        self._mk_sep(panel)

        rate_row = ctk.CTkFrame(panel, fg_color="transparent")
        rate_row.pack(fill="x")
        ctk.CTkLabel(rate_row, text="Slew rate", text_color=self._pal["muted"],
                     font=ctk.CTkFont(size=12)).pack(side="left")
        self._rate_var = tk.IntVar(value=9)
        self._rate_slider = ctk.CTkSlider(
            rate_row, from_=1, to=9, number_of_steps=8,
            variable=self._rate_var,
            button_color=self._pal["motor_btn"],
            button_hover_color=self._pal["motor_btn_hover"],
            progress_color=self._pal["motor_btn"],
            fg_color=self._pal["border"],
            command=self._on_rate_change,
        )
        self._rate_slider.pack(side="left", fill="x", expand=True, padx=10)
        self._rate_lbl = ctk.CTkLabel(
            rate_row, text="9", text_color=self._pal["motor_btn"],
            font=ctk.CTkFont(family="Courier New", size=14, weight="bold"), width=20,
        )
        self._rate_lbl.pack(side="left")

        self._mk_sep(panel)

        self._poll_btn = ctk.CTkButton(
            panel, text="▶  START LIVE POLLING", height=34,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11),
            command=self._toggle_polling,
        )
        self._poll_btn.pack(fill="x")

    # ── Tracking panel ─────────────────────────────────────────────────────────

    def _build_tracking_panel(self, parent):
        panel = self._panel(parent, "TRACKING  —  rate + master toggle")

        rate_sel = ctk.CTkFrame(panel, fg_color="transparent")
        rate_sel.pack(fill="x", pady=(0, 8))
        for c in range(3):
            rate_sel.columnconfigure(c, weight=1, uniform="track_rate")

        self._track_rate_var = tk.StringVar(value="Sidereal")
        self._track_rate_radios = []
        for col, (label, code) in enumerate(
            [("Sidereal", TRACK_SIDEREAL), ("Solar", TRACK_SOLAR), ("Lunar", TRACK_LUNAR)]
        ):
            rb = ctk.CTkRadioButton(
                rate_sel, text=label, value=label,
                variable=self._track_rate_var,
                font=ctk.CTkFont(family="Courier New", size=12),
                text_color=self._pal["text"],
                fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
                command=lambda c=code: self._on_track_rate_select(c),
            )
            rb.grid(row=0, column=col, sticky="ew", padx=(0, 6 if col < 2 else 0))
            self._track_rate_radios.append(rb)

        self._track_toggle_btn = ctk.CTkButton(
            panel, text="⏺  TRACKING OFF", height=38,
            fg_color=self._pal["track_off"], hover_color=self._pal["track_off_hover"],
            text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            command=self._toggle_tracking,
        )
        self._track_toggle_btn.pack(fill="x")

        self._track_status_lbl = ctk.CTkLabel(
            panel, text="Tracking: OFF",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(family="Courier New", size=11),
        )
        self._track_status_lbl.pack(anchor="w", pady=(4, 0))

        self._build_solar_workflow_panel(parent)

    # ── Solar tracking workflow (Tucson) ───────────────────────────────────────

    def _build_solar_workflow_panel(self, parent):
        panel = self._panel(parent, "SOLAR TRACKING  —  Tucson  ·  3-step workflow")

        intro = ctk.CTkLabel(
            panel,
            text=(
                "Assumes the mount is roughly polar aligned.\n"
                " 1.  Slew to the Sun's expected Tucson position.\n"
                " 2.  Manually nudge until the Sun is centered.\n"
                " 3.  Click 'Aligned — Sync & Track' to lock on and start tracking."
            ),
            text_color=self._pal["muted"],
            font=ctk.CTkFont(family="Courier New", size=10),
            justify="left",
            anchor="w",
        )
        intro.pack(fill="x", pady=(0, 6))

        self._sun_info_lbl = ctk.CTkLabel(
            panel,
            text="Sun/Tucson: press refresh",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(family="Courier New", size=10),
            justify="left",
            anchor="w",
        )
        self._sun_info_lbl.pack(fill="x")

        self._sun_step_lbl = ctk.CTkLabel(
            panel,
            text="Step 1 of 3 — refresh the Sun position, then slew.",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            justify="left",
            anchor="w",
        )
        self._sun_step_lbl.pack(fill="x", pady=(4, 8))

        # Refresh on its own full-width row, Slew + Sync side-by-side below.
        # 3 buttons in a single row clipped "Sync & Track" on narrow side-panels.
        sun_btns = ctk.CTkFrame(panel, fg_color="transparent")
        sun_btns.pack(fill="x")
        sun_btns.columnconfigure((0, 1), weight=1, uniform="sun_btns")
        _sun_h = 32 if self._compact else 34

        self._sun_refresh_btn = ctk.CTkButton(
            sun_btns,
            text="↻  Refresh Sun position",
            height=_sun_h,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11),
            command=self._refresh_sun_tucson_info,
        )
        self._sun_refresh_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        self._sun_slew_btn = ctk.CTkButton(
            sun_btns,
            text="⊕  Slew to Sun",
            height=_sun_h,
            fg_color=self._pal["motor_btn"],
            hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=self._slew_to_sun_tucson,
        )
        self._sun_slew_btn.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self._sun_track_btn = ctk.CTkButton(
            sun_btns,
            text="✓  Sync & Track",
            height=_sun_h,
            fg_color=self._pal["motor_btn"],
            hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=self._point_and_track_sun_tucson,
        )
        self._sun_track_btn.grid(row=1, column=1, sticky="ew")

    # ── EQ GoTo panel ──────────────────────────────────────────────────────────

    def _build_goto_panel(self, parent):
        panel = self._panel(parent, "GOTO TARGET  —  RA  /  DEC")

        eq_fields = ctk.CTkFrame(panel, fg_color="transparent")
        eq_fields.pack(fill="x", pady=(0, 8))
        eq_fields.columnconfigure(0, weight=1)

        # Stacked layout: each axis gets its own label + full-width entry block,
        # so neither field can be pushed off-screen on narrow viewports.
        ctk.CTkLabel(
            eq_fields, text="RIGHT ASCENSION   (HH:MM:SS  ·  hours  ·  °)",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(size=11), anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self._ra_entry = ctk.CTkEntry(
            eq_fields, placeholder_text="05:34:32",
            fg_color=self._pal["entry_bg"], border_color=self._pal["border"],
            text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=14 if self._compact else 15),
            height=40 if self._compact else 44,
        )
        self._ra_entry.grid(row=1, column=0, sticky="ew", pady=(2, 6))

        ctk.CTkLabel(
            eq_fields, text="DECLINATION   (−90 … +90°)",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(size=11), anchor="w",
        ).grid(row=2, column=0, sticky="w")
        self._dec_entry = ctk.CTkEntry(
            eq_fields, placeholder_text="+22.01",
            fg_color=self._pal["entry_bg"], border_color=self._pal["border"],
            text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=14 if self._compact else 15),
            height=40 if self._compact else 44,
        )
        self._dec_entry.grid(row=3, column=0, sticky="ew", pady=(2, 0))

        eq_btns = ctk.CTkFrame(panel, fg_color="transparent")
        eq_btns.pack(fill="x", pady=(0, 8))
        # Grid keeps STOP visible alongside SLEW even at very narrow widths;
        # SLEW gets the lion's share, STOP a fixed-but-readable pill.
        eq_btns.columnconfigure(0, weight=1)
        eq_btns.columnconfigure(1, weight=0)

        _btn_h = 40 if self._compact else 42
        self._eq_goto_btn = ctk.CTkButton(
            eq_btns, text="⊕  SLEW TO TARGET", height=_btn_h,
            fg_color=self._pal["motor_btn"], hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=12 if self._compact else 13, weight="bold"),
            command=self._goto_eq,
        )
        self._eq_goto_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            eq_btns, text="⬛  STOP", width=80 if self._compact else 88, height=_btn_h,
            fg_color=self._pal["danger"], hover_color=self._pal["danger_hover"],
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            text_color="#ffffff", command=self._stop,
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew")

        # Sync row — label wraps above the button so the row does not need horizontal scroll.
        self._mk_sep(panel)
        sync_block = ctk.CTkFrame(panel, fg_color="transparent")
        sync_block.pack(fill="x")
        ctk.CTkLabel(
            sync_block,
            text="Sync mount to the RA/Dec fields above, then:",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        self._sync_btn = ctk.CTkButton(
            sync_block, text="⊙  SYNC HERE", height=34,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=self._sync_eq,
        )
        self._sync_btn.pack(fill="x")

    # ── Nudge panel ────────────────────────────────────────────────────────────

    def _build_nudge_panel(self, parent):
        panel = self._panel(parent, "MANUAL NUDGE")
        ctk.CTkLabel(
            panel,
            text="Hold buttons · ▲▼ = Dec · ◀▶ = RA",
            text_color=self._pal["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(anchor="w", pady=(0, 6))

        dpad = ctk.CTkFrame(panel, fg_color="transparent")
        dpad.pack()

        # Compact dpad still ≥56 px (well above touch-target minimum) but fits
        # in a narrow embedded sidebar without horizontal scrolling.
        _dpad_sz = 58 if self._compact else 64
        _dpad_font = 20 if self._compact else 22
        btn_cfg = dict(
            width=_dpad_sz, height=_dpad_sz, corner_radius=8,
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff", font=ctk.CTkFont(size=_dpad_font),
        )

        self._nudge_btns = {}

        def _make(text, direction, row, col):
            btn = ctk.CTkButton(dpad, text=text, **btn_cfg)
            btn.grid(row=row, column=col, padx=3, pady=3)
            btn.bind("<ButtonPress-1>",   lambda e, d=direction: self._nudge_press(d))
            btn.bind("<ButtonRelease-1>", lambda e, d=direction: self._nudge_release(d))
            self._nudge_btns[direction] = btn
            return btn

        _make("▲\nDec+", "s", 0, 1)   # :ms# = Dec+
        _make("◀\nRA−",  "e", 1, 0)   # :me# = RA−
        ctk.CTkLabel(
            dpad, text="✛", text_color=self._pal["muted"],
            font=ctk.CTkFont(size=18), width=_dpad_sz, height=_dpad_sz,
        ).grid(row=1, column=1)
        _make("▶\nRA+",  "w", 1, 2)   # :mw# = RA+
        _make("▼\nDec−", "n", 2, 1)   # :mn# = Dec−

    # ── Log panel ──────────────────────────────────────────────────────────────

    def _build_log_panel(self, parent):
        panel = self._panel(parent, "ACTIVITY LOG", expand=True)
        self._log_box = ctk.CTkTextbox(
            panel,
            fg_color=self._pal["entry_bg"], text_color=self._pal["text"],
            font=ctk.CTkFont(family="Courier New", size=11),
            border_color=self._pal["border"], border_width=1,
            state="disabled", wrap="word",
        )
        self._log_box.pack(fill="both", expand=True)

    # ── Widget helpers ─────────────────────────────────────────────────────────

    def _mk_sep(self, parent):
        ctk.CTkFrame(parent, fg_color=self._pal["border"], height=1).pack(
            fill="x", pady=6 if self._compact else 8
        )

    def _panel(self, parent, title: str, expand: bool = False) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(parent, fg_color=self._pal["panel"],
                             corner_radius=10, border_color=self._pal["border"], border_width=1)
        self._panel_outer_frames.append(outer)
        outer.pack(fill="both", expand=expand, pady=(0, 8 if self._compact else 10))
        title_lbl = ctk.CTkLabel(
            outer, text=title,
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=self._pal["motor_btn"],
        )
        self._section_title_labels.append(title_lbl)
        _padx = 10 if self._compact else 14
        _pady_top = (8, 0) if self._compact else (10, 0)
        title_lbl.pack(anchor="w", padx=_padx, pady=_pady_top)
        sep = ctk.CTkFrame(outer, fg_color=self._pal["border"], height=1)
        self._section_separators.append(sep)
        sep.pack(fill="x", padx=_padx, pady=(4, 0))
        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=_padx, pady=8 if self._compact else 10)
        return inner

    def _stacked_form_row(
        self,
        parent,
        *,
        label: str,
        hint: str,
        placeholder: str,
        pady: tuple = (0, 8),
    ) -> ctk.CTkEntry:
        """Header row (label + faint hint) on top, full-width entry below.

        Stacking avoids the old "label  hint  [180px entry]" pattern which forced
        the entry off-screen on narrow side-panels and small Pi displays.
        """
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=pady)

        head = ctk.CTkFrame(row, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(
            head, text=label, text_color=self._pal["muted"],
            font=ctk.CTkFont(size=12 if self._compact else 13, weight="bold"),
            anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            head, text=hint, text_color=self._pal["muted"],
            font=ctk.CTkFont(size=10 if self._compact else 11),
            anchor="w",
        ).pack(side="left", padx=(6, 0))

        entry = ctk.CTkEntry(
            row,
            fg_color=self._pal["entry_bg"], border_color=self._pal["border"],
            text_color=self._pal["text"], placeholder_text=placeholder,
            height=34 if self._compact else 36,
        )
        entry.pack(fill="x", pady=(2, 0))
        return entry

    def _set_mount_controls(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (
            self._eq_goto_btn, self._stop_btn, self._sync_btn,
            self._home_btn, self._btn_set_zero, self._btn_mech_zero,
            self._poll_btn, self._rate_slider, self._track_toggle_btn,
            self._set_location_btn, self._ps_calibrate_btn,
            self._sun_refresh_btn, self._sun_slew_btn, self._sun_track_btn,
        ):
            w.configure(state=state)
        for btn in self._nudge_btns.values():
            btn.configure(state=state)

    # ── Logging ────────────────────────────────────────────────────────────────

    def log(self, msg: str, color: Optional[str] = None):
        if color is None:
            color = self._pal["text"]
        ts = time.strftime("%H:%M:%S")
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"[{ts}] {msg}\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    # ── Connection mode toggle ─────────────────────────────────────────────────

    def _switch_conn_mode(self, mode: str):
        self._conn_mode.set(mode)
        if mode == "serial":
            self._wifi_panel.pack_forget()
            self._serial_panel.pack(fill="x", pady=(0, 8))
            self._btn_mode_serial.configure(
                fg_color=self._pal["motor_btn"], text_color="#ffffff")
            self._btn_mode_wifi.configure(
                fg_color=self._pal["motor_btn_muted"], text_color="#ffffff")
        else:
            self._serial_panel.pack_forget()
            self._wifi_panel.pack(fill="x", pady=(0, 8))
            self._btn_mode_wifi.configure(
                fg_color=self._pal["motor_btn"], text_color="#ffffff")
            self._btn_mode_serial.configure(
                fg_color=self._pal["motor_btn_muted"], text_color="#ffffff")

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["(no ports found)"]
        self._port_menu.configure(values=ports)
        self._port_var.set(ports[0])

    # ── Connect / Disconnect ───────────────────────────────────────────────────

    def _toggle_connect(self):
        if self.mount and self.mount.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        if self._conn_mode.get() == "wifi":
            self._connect_wifi()
        else:
            self._connect_serial()

    def _connect_serial(self):
        port = self._port_var.get()
        self.log(f"Connecting to {port}…")
        self.mount = IOptronHAE(port)
        try:
            info = self.mount.connect()
            fw   = self.mount.get_firmware()
            self._on_connected(f"Serial {port}", info, fw)
        except Exception as exc:
            self.mount = None
            self.log(f"Connection failed: {exc}", self._pal["danger"])
            showerror("Connection Error", str(exc))

    def _connect_wifi(self):
        ip = self._wifi_ip_var.get().strip() or WIFI_DEFAULT_IP
        raw_port = str(self._wifi_port_var.get()).strip()
        try:
            port = int(raw_port) if raw_port else WIFI_DEFAULT_PORT
        except ValueError:
            port = WIFI_DEFAULT_PORT
        self.log(f"Connecting via WiFi  {ip}:{port}…")
        self._connect_btn.configure(
            state="disabled",
            text="CONNECTING…",
            fg_color=self._pal["motor_btn_muted"],
            text_color="#ffffff",
        )

        def _worker():
            try:
                mount = IOptronHAEWifi(ip, port)
                info  = mount.connect()
                fw    = mount.get_firmware()
                self.mount = mount
                self.after(0, lambda i=info, f=fw: self._on_connected(f"WiFi {ip}:{port}", i, f))
            except Exception as exc:
                err = str(exc)
                self.after(0, lambda e=err: self._on_wifi_fail(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_connected(self, label: str, info: str, fw: str):
        self._disp_fw.set(fw or "N/A")
        self._conn_indicator.configure(
            text="● ONLINE" if self._compact else "● CONNECTED",
            text_color=self._pal["success"],
        )
        self._connect_btn.configure(
            text="DISCONNECT", state="normal",
            fg_color=self._pal["danger"], hover_color=self._pal["danger_hover"],
            text_color="#ffffff",
        )
        self._set_mount_controls(True)
        self.log(f"Connected ({label}) — model:{info}  FW:{fw}", self._pal["success"])
        # Sync mount UTC time from system clock on every connect
        try:
            ok, utc_str = self.mount.set_utc_time()
            if ok:
                self.log(f"UTC time synced to mount  ({utc_str}).", self._pal["success"])
            else:
                self.log(f"UTC time sync failed (mount rejected :SUT#).", self._pal["danger"])
        except Exception as exc:
            self.log(f"UTC time sync error: {exc}", self._pal["danger"])
        self.log("Tip: start polling to see live RA/Dec position.", self._pal["muted"])
        try:
            self.mount.unpark()
            self.log("Unparked automatically (:MP0#).", self._pal["success"])
        except Exception as exc:
            self.log(f"Unpark on connect: {exc}", self._pal["muted"])
        self._refresh_sun_tucson_info()

    def _on_wifi_fail(self, err: str):
        self.mount = None
        self._connect_btn.configure(
            state="normal",
            text="CONNECT",
            fg_color=self._pal["accent2"],
            hover_color=self._pal["accent2_hover"],
        )
        self.log(f"WiFi connection failed: {err}", self._pal["danger"])
        try:
            showerror("WiFi Error", err, parent=self.winfo_toplevel())
        except tk.TclError:
            pass

    def _disconnect(self):
        self._stop_polling()
        self._clear_nudge_queue()
        self._stop_nudge_motion_off_thread()
        if self.mount:
            self.mount.disconnect()
            self.mount = None
        self._tracking_on = False
        self._update_track_ui()
        self._conn_indicator.configure(
            text="● OFFLINE" if self._compact else "● DISCONNECTED",
            text_color=self._pal["danger"],
        )
        self._connect_btn.configure(
            text="CONNECT",
            state="normal",
            fg_color=self._pal["accent2"],
            hover_color=self._pal["accent2_hover"],
        )
        self._set_mount_controls(False)
        for v in (self._disp_ra, self._disp_dec, self._disp_fw, self._disp_pier):
            v.set("---")
        self._sun_info_lbl.configure(text="Sun/Tucson: press refresh")
        self._sun_step_lbl.configure(
            text="Step 1 of 3 — refresh the Sun position, then slew.",
            text_color=self._pal["muted"],
        )
        self.log("Disconnected.")

    # ── EQ GoTo ────────────────────────────────────────────────────────────────

    def _parse_inputs(self) -> Tuple[float, float]:
        """Parse RA and Dec fields. Returns (ra_deg, dec_deg). Raises ValueError."""
        ra_deg  = _parse_ra_input(self._ra_entry.get())
        dec_deg = _parse_dec_input(self._dec_entry.get())
        if not (-90.0 <= dec_deg <= 90.0):
            raise ValueError("Declination out of range (−90 … +90).")
        return ra_deg, dec_deg

    def _goto_eq(self):
        if not self._check_connected(): return
        try:
            ra_deg, dec_deg = self._parse_inputs()
        except ValueError as e:
            showwarning("Invalid Input", str(e))
            return
        ra_hms = _deg_to_hms(ra_deg)
        dec_dms = _deg_to_dms(dec_deg)
        self.log(f"[EQ GoTo]  RA={ra_hms}  Dec={dec_dms}", self._pal["motor_btn"])
        threading.Thread(target=self._run_eq_goto, args=(ra_deg, dec_deg), daemon=True).start()

    def _run_eq_goto(self, ra_deg, dec_deg):
        ok = False
        motor_rgb_led.begin_movement()
        try:
            ok, msg = self.mount.goto_radec(ra_deg, dec_deg)
            color = self._pal["success"] if ok else self._pal["danger"]
            self.after(0, lambda: self.log(f"  → {msg}", color))
        except Exception as exc:
            self.after(0, lambda: self.log(f"Error: {exc}", self._pal["danger"]))
        finally:
            if ok:
                motor_rgb_led.schedule_end_movement(25.0)
            else:
                motor_rgb_led.end_movement()

    # ── Sync ───────────────────────────────────────────────────────────────────

    def _sync_eq(self):
        if not self._check_connected(): return
        try:
            ra_deg, dec_deg = self._parse_inputs()
        except ValueError as e:
            showwarning("Invalid Input", str(e))
            return
        ra_hms = _deg_to_hms(ra_deg)
        dec_dms = _deg_to_dms(dec_deg)
        self.log(f"[Sync]  RA={ra_hms}  Dec={dec_dms}", self._pal["muted"])
        try:
            ok = self.mount.sync_radec(ra_deg, dec_deg)
            msg = "Sync accepted." if ok else "Sync failed."
            self.after(0, lambda: self.log(f"  → {msg}", self._pal["success"] if ok else self._pal["danger"]))
        except Exception as exc:
            self.log(f"Sync error: {exc}", self._pal["danger"])

    # ── Stop / zero position ───────────────────────────────────────────────────

    def _stop(self):
        if not self._check_connected(): return
        try:
            self.mount.stop_slew()
            motor_rgb_led.cancel_scheduled_movement_ends()
            self.log("STOP — slew halted.", self._pal["danger"])
        except Exception as exc:
            self.log(f"Error: {exc}", self._pal["danger"])

    def _set_zero_at_current(self):
        if not self._check_connected():
            return
        try:
            ok, resp = self.mount.set_zero_at_current()
            if ok:
                self.log("Zero position set at current location (:SZP#).", self._pal["success"])
            else:
                self.log(f"Set zero rejected: {resp}", self._pal["danger"])
        except Exception as exc:
            self.log(f"Error: {exc}", self._pal["danger"])

    def _goto_zero(self):
        if not self._check_connected():
            return
        motor_rgb_led.begin_movement()
        try:
            self.mount.goto_zero()
            self.log("Slewing to saved zero position (:MH#)…", self._pal["muted"])
        except Exception as exc:
            motor_rgb_led.end_movement()
            self.log(f"Error: {exc}", self._pal["danger"])
        else:
            motor_rgb_led.schedule_end_movement(30.0)

    def _search_mechanical_zero(self):
        if not self._check_connected():
            return
        motor_rgb_led.begin_movement()
        try:
            ok, resp = self.mount.search_mechanical_zero()
            if ok:
                self.log("Mechanical zero search started (:MSH#)…", self._pal["muted"])
            else:
                self.log(f"Mechanical zero not accepted: {resp}", self._pal["danger"])
        except Exception as exc:
            motor_rgb_led.end_movement()
            self.log(f"Error: {exc}", self._pal["danger"])
        else:
            if ok:
                motor_rgb_led.schedule_end_movement(180.0)
            else:
                motor_rgb_led.end_movement()

    # ── Slew rate ──────────────────────────────────────────────────────────────

    def _on_rate_change(self, val):
        rate = int(round(float(val)))
        self._rate_lbl.configure(text=str(rate))
        if self.mount and self.mount.connected:
            try:
                self.mount.set_slew_rate(rate)
            except Exception:
                pass

    # ── Tracking ───────────────────────────────────────────────────────────────

    def _on_track_rate_select(self, code: str):
        self._tracking_rate = code
        label = self._track_rate_var.get()
        self.log(f"Tracking rate selected: {label}")
        if self._tracking_on and self.mount and self.mount.connected:
            try:
                self.mount.set_tracking_rate(code)
            except Exception as exc:
                self.log(f"Rate change error: {exc}", self._pal["danger"])

    def _toggle_tracking(self):
        if not self._check_connected():
            return
        will_enable = not self._tracking_on
        if will_enable and self._tracking_prerequisite:
            try:
                ok, msg = self._tracking_prerequisite()
            except Exception as exc:
                ok, msg = False, f"Could not verify plate solve state: {exc}"
            if not ok:
                title = "Plate solve required"
                text = msg or (
                    "Run a successful plate solve before enabling mount tracking "
                    "(center tab: Plate solve)."
                )
                try:
                    showwarning(title, text, parent=self.winfo_toplevel())
                except tk.TclError:
                    showwarning(title, text)
                return
        self._tracking_on = will_enable
        try:
            if self._tracking_on:
                self.mount.set_tracking_rate(self._tracking_rate)
                self.mount.tracking_on()
                self.log(f"Tracking ON  [{self._track_rate_var.get()}]", self._pal["success"])
            else:
                self.mount.tracking_off()
                self.log("Tracking OFF.", self._pal["muted"])
        except Exception as exc:
            self._tracking_on = not self._tracking_on   # revert
            self.log(f"Tracking error: {exc}", self._pal["danger"])
        self._update_track_ui()

    def _update_track_ui(self):
        if self._tracking_on:
            label = self._track_rate_var.get()
            self._track_toggle_btn.configure(
                text=f"⏹  TRACKING ON  [{label}]",
                fg_color=self._pal["track_on"], hover_color=self._pal["motor_btn_hover"],
            )
            self._track_status_lbl.configure(
                text=f"Tracking: ON  ({label})", text_color=self._pal["success"])
        else:
            self._track_toggle_btn.configure(
                text="⏺  TRACKING OFF",
                fg_color=self._pal["track_off"], hover_color=self._pal["track_off_hover"],
            )
            self._track_status_lbl.configure(
                text="Tracking: OFF", text_color=self._pal["muted"])

    def _refresh_sun_tucson_info(self):
        try:
            ra, dec = sun_apparent_radec_deg(TUCSON_LAT, TUCSON_LON, TUCSON_ELEV_M)
            alt, az = radec_to_altaz_deg(ra, dec, TUCSON_LAT, TUCSON_LON, TUCSON_ELEV_M)
            self._sun_info_lbl.configure(
                text=f"Sun/Tucson RA {ra:.3f}  Dec {dec:.3f}  Alt {alt:.1f}  Az {az:.1f}"
            )
        except Exception as exc:
            self._sun_info_lbl.configure(text=f"Sun/Tucson compute error: {exc}")

    def _set_sun_step(self, msg: str, color: Optional[str] = None) -> None:
        """Update the solar workflow step indicator on the Tk thread."""
        self._sun_step_lbl.configure(
            text=msg,
            text_color=color or self._pal["muted"],
        )

    def _confirm_solar_safety(self) -> bool:
        """Single safety gate shared by every step that drives the mount toward the Sun."""
        return askyesno(
            "Solar safety confirmation",
            "Solar observing is hazardous.\n\n"
            "Confirm a proper solar filter is installed on ALL optics (scope, finder, and camera) "
            "before continuing.",
            parent=self.winfo_toplevel(),
        )

    def _slew_to_sun_tucson(self):
        """Step 1: assume the mount is roughly polar aligned and slew to the Sun's ephemeris."""
        if not self._check_connected():
            return
        if not self._confirm_solar_safety():
            self.log("Solar slew cancelled (solar safety confirmation not accepted).", self._pal["muted"])
            return

        self._set_sun_step("Step 1 of 3 — slewing to Sun's expected Tucson position…")
        self.log("Slewing to Sun (Tucson ephemeris)…", self._pal["motor_btn"])

        def _worker():
            ok_goto = False
            motor_rgb_led.begin_movement()
            try:
                ra, dec = sun_apparent_radec_deg(TUCSON_LAT, TUCSON_LON, TUCSON_ELEV_M)
                alt, az = radec_to_altaz_deg(ra, dec, TUCSON_LAT, TUCSON_LON, TUCSON_ELEV_M)
                ok_goto, msg = self.mount.goto_radec(ra, dec)
                if not ok_goto:
                    self.after(
                        0,
                        lambda m=msg: (
                            self._set_sun_step(
                                "Step 1 failed — fix mount/alignment and retry slew.",
                                self._pal["danger"],
                            ),
                            self.log(f"Slew to Sun failed: {m}", self._pal["danger"]),
                        ),
                    )
                    return
                self.after(
                    0,
                    lambda r=ra, d=dec, a=alt, z=az: (
                        self._sun_info_lbl.configure(
                            text=f"Sun/Tucson RA {r:.3f}  Dec {d:.3f}  Alt {a:.1f}  Az {z:.1f}"
                        ),
                        self._set_sun_step(
                            "Step 2 of 3 — manually nudge until the Sun is centered, then click 'Aligned'.",
                            self._pal["motor_btn"],
                        ),
                        self.log(
                            "Slew to Sun complete. Manually center, then click 'Aligned — Sync & Track'.",
                            self._pal["success"],
                        ),
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda e=exc: (
                        self._set_sun_step(
                            "Step 1 errored — see log, then retry slew.",
                            self._pal["danger"],
                        ),
                        self.log(f"Slew to Sun error: {e}", self._pal["danger"]),
                    ),
                )
            finally:
                if ok_goto:
                    motor_rgb_led.schedule_end_movement(25.0)
                else:
                    motor_rgb_led.end_movement()

        threading.Thread(target=_worker, daemon=True).start()

    def _point_and_track_sun_tucson(self):
        """Step 3: user has manually centered the Sun — sync to current ephemeris and track."""
        if not self._check_connected():
            return
        if not self._confirm_solar_safety():
            self.log("Sun tracking cancelled (solar safety confirmation not accepted).", self._pal["muted"])
            return

        self._set_sun_step("Step 3 of 3 — syncing to Sun and starting solar tracking…")
        self.log("Syncing to Sun (Tucson ephemeris) and enabling solar tracking…", self._pal["motor_btn"])

        def _worker():
            try:
                ra, dec = sun_apparent_radec_deg(TUCSON_LAT, TUCSON_LON, TUCSON_ELEV_M)
                alt, az = radec_to_altaz_deg(ra, dec, TUCSON_LAT, TUCSON_LON, TUCSON_ELEV_M)
                ok = self.mount.sync_radec(ra, dec)
                if not ok:
                    self.after(
                        0,
                        lambda: (
                            self._set_sun_step(
                                "Sync rejected — re-center the Sun and click 'Aligned' again.",
                                self._pal["danger"],
                            ),
                            self.log(
                                "Sun sync rejected by mount. Re-center Sun and retry.",
                                self._pal["danger"],
                            ),
                        ),
                    )
                    return
                self.mount.set_tracking_rate(TRACK_SOLAR)
                self.mount.tracking_on()
                self._tracking_on = True
                self._tracking_rate = TRACK_SOLAR
                self.after(
                    0,
                    lambda r=ra, d=dec, a=alt, z=az: (
                        self._track_rate_var.set("Solar"),
                        self._update_track_ui(),
                        self._sun_info_lbl.configure(
                            text=f"Sun/Tucson RA {r:.3f}  Dec {d:.3f}  Alt {a:.1f}  Az {z:.1f}"
                        ),
                        self._set_sun_step(
                            "Tracking the Sun. Click TRACKING ON/OFF above to stop.",
                            self._pal["success"],
                        ),
                        self.log(
                            "Solar tracking ON from current Sun lock (Tucson assumptions).",
                            self._pal["success"],
                        ),
                    ),
                )
            except Exception as exc:
                self.after(
                    0,
                    lambda e=exc: (
                        self._set_sun_step(
                            "Sync/track errored — see log, then retry alignment.",
                            self._pal["danger"],
                        ),
                        self.log(f"Sun tracking error: {e}", self._pal["danger"]),
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    # ── Nudge (I/O runs off the Tk thread so the UI does not beach-ball) ───────

    def _ensure_nudge_worker(self):
        if self._nudge_worker_thread is not None and self._nudge_worker_thread.is_alive():
            return
        self._nudge_worker_thread = threading.Thread(
            target=self._nudge_worker_loop, daemon=True, name="MountNudge"
        )
        self._nudge_worker_thread.start()

    def _nudge_worker_loop(self):
        while True:
            try:
                item = self._nudge_cmd_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            direction, start = item
            try:
                if self.mount and self.mount.connected:
                    self.mount.move_direction(direction, start)
            except Exception as exc:
                self.after(0, lambda e=str(exc), d=direction: self._nudge_async_error(e, d))

    def _nudge_async_error(self, err: str, direction: str):
        self._nudge_active.discard(direction)
        if direction in self._nudge_btns:
            self._nudge_btns[direction].configure(fg_color=self._pal["motor_btn_muted"])
        self.log(f"Nudge error: {err}", self._pal["danger"])

    def _clear_nudge_queue(self):
        try:
            while True:
                self._nudge_cmd_queue.get_nowait()
        except queue.Empty:
            pass

    def _stop_nudge_motion_off_thread(self, timeout: float = 2.5):
        """Halt any axis still marked nudging; runs blocking I/O off the UI thread."""
        dirs = list(self._nudge_active)
        self._nudge_active.clear()
        if not dirs or not self.mount:
            return
        m = self.mount

        def _run():
            if not m.connected:
                return
            for d in dirs:
                try:
                    m.move_direction(d, False)
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)

    def _nudge_press(self, direction: str):
        if not self._check_connected():
            return
        if direction in self._nudge_active:
            return
        self._nudge_active.add(direction)
        self._ensure_nudge_worker()
        axis = "Dec" if direction in ("n", "s") else "RA"
        sign = "+" if direction in ("s", "w") else "−"
        self.log(f"Nudge {axis}{sign} — moving…")
        if direction in self._nudge_btns:
            self._nudge_btns[direction].configure(fg_color=self._pal["motor_btn"])
        self._nudge_cmd_queue.put((direction, True))

    def _nudge_release(self, direction: str):
        self._nudge_active.discard(direction)
        axis = "Dec" if direction in ("n", "s") else "RA"
        self.log(f"Nudge {axis} — stopped.")
        if direction in self._nudge_btns:
            self._nudge_btns[direction].configure(fg_color=self._pal["motor_btn_muted"])
        self._ensure_nudge_worker()
        self._nudge_cmd_queue.put((direction, False))

    # ── Live polling ───────────────────────────────────────────────────────────

    def _toggle_polling(self):
        if self._polling:
            self._stop_polling()
        else:
            self._start_polling()

    def _start_polling(self):
        if self._polling:
            return   # already running, don't launch a second thread
        self._polling = True
        self._poll_btn.configure(
            text="⏸  STOP POLLING",
            fg_color=self._pal["motor_btn"],
            hover_color=self._pal["motor_btn_hover"],
            text_color="#ffffff",
        )
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.log("Live position polling started (1 s interval).")

    def _stop_polling(self):
        self._polling = False
        self._poll_btn.configure(
            text="▶  START LIVE POLLING",
            fg_color=self._pal["motor_btn_muted"],
            hover_color=self._pal["motor_btn"],
            text_color="#ffffff",
        )

    def _poll_loop(self):
        """Background thread: query :GEP# every second and update displays."""
        consecutive_errors = 0
        while self._polling and self.mount and self.mount.connected:
            try:
                ra_deg, dec_deg, pier = self.mount.get_current_radec()
                ra_str   = _deg_to_hms(ra_deg)
                dec_str  = _deg_to_dms(dec_deg)
                pier_str = {0: "East (normal)", 1: "West (flipped)", 2: "Unknown"}.get(pier, f"? ({pier})")
                consecutive_errors = 0
                self.after(0, lambda r=ra_str, d=dec_str, p=pier_str: (
                    self._disp_ra.set(r),
                    self._disp_dec.set(d),
                    self._disp_pier.set(p),
                ))
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    self.after(0, lambda e=exc: self.log(f"Poll error: {e}", self._pal["muted"]))
                elif consecutive_errors == 4:
                    self.after(0, lambda: self.log("Poll errors suppressed (repeated)…", self._pal["muted"]))
            time.sleep(1)

    # ── Set Location ───────────────────────────────────────────────────────────

    def _set_location(self):
        if not self._check_connected():
            return
        try:
            lat = _parse_angle_dms(self._lat_entry.get())
            lon = _parse_angle_dms(self._lon_entry.get())
        except ValueError:
            showwarning("Invalid Input",
                "Enter latitude and longitude as decimal degrees "
                "or DD:MM:SS (e.g. 32:13:00 / -110:56:00).")
            return
        if not (-90.0 <= lat <= 90.0):
            showwarning("Invalid Input", "Latitude must be between −90 and +90.")
            return
        if not (-180.0 <= lon <= 180.0):
            showwarning("Invalid Input", "Longitude must be between −180 and +180.")
            return

        def _worker():
            try:
                lat_ok = self.mount.set_latitude(lat)
                lon_ok = self.mount.set_longitude(lon)
                if lat_ok and lon_ok:
                    msg = (f"Location set — Lat {_deg_to_dms(lat)}  Lon {_deg_to_dms(lon)}",
                           self._pal["success"])
                else:
                    msg = (f"Location partially failed (lat={'OK' if lat_ok else 'FAIL'}, "
                           f"lon={'OK' if lon_ok else 'FAIL'})", self._pal["danger"])
            except Exception as exc:
                msg = (f"Location error: {exc}", self._pal["danger"])
            self.after(0, lambda m=msg: self.log(*m))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Plate Solve Calibration ────────────────────────────────────────────────

    def _set_pier_side(self, side: str):
        """Toggle pier side button highlight. side='0'=East, '1'=West."""
        self._ps_pier_var.set(side)
        if side == "0":
            self._pier_east_btn.configure(
                fg_color=self._pal["motor_btn"], text_color="#ffffff")
            self._pier_west_btn.configure(
                fg_color=self._pal["motor_btn_muted"], text_color="#ffffff")
        else:
            self._pier_west_btn.configure(
                fg_color=self._pal["motor_btn"], text_color="#ffffff")
            self._pier_east_btn.configure(
                fg_color=self._pal["motor_btn_muted"], text_color="#ffffff")

    def _calibrate_plate_solve(self):
        if not self._check_connected():
            return
        try:
            ra_deg  = _parse_ra_input(self._ps_ra_entry.get())
            dec_deg = _parse_dec_input(self._ps_dec_entry.get())
        except ValueError:
            showwarning("Invalid Input",
                "Enter RA as HH:MM:SS (or decimal degrees) "
                "and Dec as \u00b1DD:MM:SS (or decimal degrees).")
            return
        if not (-90.0 <= dec_deg <= 90.0):
            showwarning("Invalid Input", "Declination must be between \u221290 and +90.")
            return

        ra_hms  = _deg_to_hms(ra_deg)
        dec_dms = _deg_to_dms(dec_deg)
        pier    = self._ps_pier_var.get()
        pier_label = "West" if pier == "1" else "East"
        self.log(f"[Plate Solve Cal]  :SRA \u2192 {ra_hms}  :Sd \u2192 {dec_dms}  Pier: {pier_label}", self._pal["motor_btn"])

        def _worker():
            try:
                r1 = self.mount.send_command(f":SRA{_encode_ra(ra_deg)}#")
                r2 = self.mount.send_command(f":Sd{_encode_dec(dec_deg)}#")
                if r1[:1] != "1" or r2[:1] != "1":
                    msg = f"Mount rejected RA/Dec (SRA\u2192{r1}, Sd\u2192{r2})"
                    self.after(0, lambda m=msg: self.log(m, self._pal["danger"]))
                    return
                self.mount.send_command(f":SPS{pier}#")
                resp = self.mount.send_command(":CM#")
                ok   = resp[:1] == "1"
                msg  = ("Calibration accepted (:CM# \u2192 1)." if ok
                        else f"Calibration failed (:CM# \u2192 {resp}).")
                col  = self._pal["success"] if ok else self._pal["danger"]
                self.after(0, lambda m=msg, c=col: self.log(m, c))
            except Exception as exc:
                self.after(0, lambda e=exc: self.log(f"Calibration error: {e}", self._pal["danger"]))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _check_connected(self) -> bool:
        if not (self.mount and self.mount.connected):
            showwarning("Not Connected", "Connect to the mount first.")
            return False
        return True

    def apply_camgui_theme(self, theme: Dict[str, str]) -> None:
        """Sync colours with :class:`CamGUI.ZWOCameraGUI` day/night ``THEMES`` entry."""
        if not self._match_camgui:
            return
        t = theme
        self._pal = _mount_palette_for_camgui_theme(t)
        p = self._pal

        for a in getattr(self, "_mount_xy_areas", []):
            theme_xy_area(a, p["bg"])

        self.configure(fg_color=p["bg"])
        self._conn_outer.configure(fg_color=p["bg"])
        self._tabs.configure(
            fg_color=t["bg_primary"],
            segmented_button_fg_color=t["bg_tertiary"],
            segmented_button_selected_color=p["motor_btn"],
            segmented_button_selected_hover_color=p["motor_btn_hover"],
            segmented_button_unselected_color=t["bg_secondary"],
            segmented_button_unselected_hover_color=t["bg_tertiary"],
            text_color=t["fg_primary"],
        )

        self._hdr.configure(fg_color=p["panel"])
        self._hdr_title.configure(text_color=p["motor_btn"])
        if self.mount and self.mount.connected:
            self._conn_indicator.configure(text_color=p["success"])
        else:
            self._conn_indicator.configure(text_color=p["danger"])

        for outer in self._panel_outer_frames:
            outer.configure(fg_color=p["panel"], border_color=p["border"])
        for lbl in self._section_title_labels:
            lbl.configure(text_color=p["motor_btn"])
        for sep in self._section_separators:
            sep.configure(fg_color=p["border"])

        is_day_ui = t.get("bg_primary") == "#f5f5f5"
        entry_text = "#ffadad" if is_day_ui else p["text"]
        entry_kw = dict(
            fg_color=p["entry_bg"],
            border_color=p["border"],
            text_color=entry_text,
        )
        for w in (
            self._lat_entry,
            self._lon_entry,
            self._ra_entry,
            self._dec_entry,
            self._ps_ra_entry,
            self._ps_dec_entry,
            self._wifi_ip_entry,
            self._wifi_port_entry,
        ):
            try:
                w.configure(**entry_kw)
            except tk.TclError:
                pass

        self._log_box.configure(
            fg_color=p["entry_bg"],
            text_color=entry_text,
            border_color=p["border"],
        )

        self._port_menu.configure(
            fg_color=p["entry_bg"],
            button_color=p["motor_btn"],
            button_hover_color=p["motor_btn_hover"],
            text_color=entry_text,
        )
        try:
            self._refresh_ports_btn.configure(
                fg_color=p["motor_btn_muted"],
                hover_color=p["motor_btn"],
                text_color="#ffffff",
            )
        except (tk.TclError, AttributeError):
            pass

        self._switch_conn_mode(self._conn_mode.get())

        if self.mount and self.mount.connected:
            self._connect_btn.configure(
                text="DISCONNECT",
                fg_color=p["motor_btn"],
                hover_color=p["motor_btn_hover"],
                text_color="#ffffff",
            )
        else:
            self._connect_btn.configure(
                text="CONNECT",
                fg_color=p["motor_btn"],
                hover_color=p["motor_btn_hover"],
                text_color="#ffffff",
            )

        self._btn_set_zero.configure(
            fg_color=p["motor_btn_muted"],
            hover_color=p["motor_btn"],
            text_color="#ffffff",
        )
        self._home_btn.configure(
            fg_color=p["danger"], hover_color=p["danger_hover"], text_color="#ffffff"
        )
        self._btn_mech_zero.configure(
            fg_color=p["motor_btn"], hover_color=p["motor_btn_hover"], text_color="#ffffff"
        )
        self._set_location_btn.configure(
            fg_color=p["motor_btn"],
            hover_color=p["motor_btn_hover"],
            text_color="#ffffff",
        )
        self._ps_calibrate_btn.configure(
            fg_color=p["motor_btn"],
            hover_color=p["motor_btn_hover"],
            text_color="#ffffff",
        )

        self._set_pier_side(self._ps_pier_var.get())

        self._eq_goto_btn.configure(
            fg_color=p["motor_btn"],
            hover_color=p["motor_btn_hover"],
            text_color="#ffffff",
        )
        self._stop_btn.configure(
            fg_color=p["motor_btn"], hover_color=p["motor_btn_hover"], text_color="#ffffff"
        )
        self._sync_btn.configure(
            fg_color=p["motor_btn_muted"],
            hover_color=p["motor_btn"],
            text_color="#ffffff",
        )

        self._disp_ra_lbl.configure(text_color=p["motor_btn"])
        self._disp_dec_lbl.configure(text_color=p["motor_btn_hover"])
        self._disp_fw_lbl.configure(text_color=p["muted"])
        self._disp_pier_lbl.configure(text_color=p["muted"])

        self._rate_slider.configure(
            button_color=p["motor_btn"],
            button_hover_color=p["motor_btn_hover"],
            progress_color=p["motor_btn"],
            fg_color=p["border"],
        )
        self._rate_lbl.configure(text_color=p["motor_btn"])

        if self._polling:
            self._poll_btn.configure(
                text="⏸  STOP POLLING",
                fg_color=p["motor_btn"],
                hover_color=p["motor_btn_hover"],
                text_color="#ffffff",
            )
        else:
            self._poll_btn.configure(
                text="▶  START LIVE POLLING",
                fg_color=p["motor_btn_muted"],
                hover_color=p["motor_btn"],
                text_color="#ffffff",
            )

        for rb in getattr(self, "_track_rate_radios", ()):
            rb.configure(
                text_color=p["text"],
                fg_color=p["motor_btn"],
                hover_color=p["motor_btn_hover"],
            )
        self._sun_refresh_btn.configure(
            fg_color=p["motor_btn_muted"],
            hover_color=p["motor_btn"],
            text_color="#ffffff",
        )
        self._sun_slew_btn.configure(
            fg_color=p["motor_btn"],
            hover_color=p["motor_btn_hover"],
            text_color="#ffffff",
        )
        self._sun_track_btn.configure(
            fg_color=p["motor_btn"],
            hover_color=p["motor_btn_hover"],
            text_color="#ffffff",
        )
        # Step indicator inherits whichever colour was last set (status text), so leave alone.
        self._update_track_ui()

        for direction, btn in self._nudge_btns.items():
            active = direction in self._nudge_active
            btn.configure(
                fg_color=p["motor_btn"] if active else p["motor_btn_muted"],
                hover_color=p["motor_btn_hover"],
                text_color="#ffffff",
            )

    def get_mount(self):
        """Return the active mount instance if connected, else None."""
        if self.mount and self.mount.connected:
            return self.mount
        return None

    def on_window_close(self):
        self._stop_polling()
        self._clear_nudge_queue()
        self._stop_nudge_motion_off_thread(timeout=3.0)
        if self.mount:
            self.mount.disconnect()
            self.mount = None
        motor_rgb_led.cleanup()
        if self._standalone:
            self.winfo_toplevel().destroy()

    def cleanup_embedded(self):
        """Stop polling and disconnect when embedded (does not destroy this widget)."""
        self._stop_polling()
        self._clear_nudge_queue()
        self._stop_nudge_motion_off_thread(timeout=3.0)
        if self.mount:
            try:
                self.mount.disconnect()
            except Exception:
                pass
            self.mount = None
        motor_rgb_led.cleanup()

