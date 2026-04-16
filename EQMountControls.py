"""
iOptron HAE16C Mount Controller — EQ Mode Only
===============================================
Dark-themed observatory controller with:
  • Equatorial (RA/Dec) GoTo mode
  • Tracking rate selector: Sidereal / Solar / Lunar
  • Tracking ON / OFF toggle
  • Manual nudge d-pad
  • Live RA/Dec position polling (real-time)
  • Activity log

Commands verified against iOptron RS-232 Command Language v3.10 (Jan 4, 2021).

Dependencies:
    pip install customtkinter pyserial
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import datetime
import serial
import serial.tools.list_ports
import socket
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

BAUD_RATE = 115200
TIMEOUT   = 4.0

WIFI_DEFAULT_IP   = "10.10.100.254"
WIFI_DEFAULT_PORT = 8899

# Tracking rate codes — :RT<n># (v3.10)
TRACK_SIDEREAL = "0"
TRACK_LUNAR    = "1"
TRACK_SOLAR    = "2"

# iOptron uses 0.01 arc-second units (360000 units per degree)
_UNITS_PER_DEG = 360000


# ═══════════════════════════════════════════════════════════════════════════════
#  Coordinate helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _deg_to_iopval(deg: float) -> int:
    """Decimal degrees → iOptron 0.01 arcsec integer units."""
    return round(deg * _UNITS_PER_DEG)


def _iopval_to_deg(val: int) -> float:
    """iOptron 0.01 arcsec integer units → decimal degrees."""
    return val / _UNITS_PER_DEG


def _encode_ra(ra_deg: float) -> str:
    """
    Encode RA for :SRA<TTTTTTTTT>#
    9 digits, unsigned, 0.01 arcsec units.
    Range [0, 129,600,000] (= 0..360°).
    """
    v = _deg_to_iopval(ra_deg % 360.0)
    return f"{v:09d}"


def _encode_dec(dec_deg: float) -> str:
    """
    Encode Dec for :Sd<s><TTTTTTTT>#
    sign + 8 digits, 0.01 arcsec units.
    Range [-32,400,000, +32,400,000] (= -90..+90°).
    """
    v = _deg_to_iopval(dec_deg)
    sign = "+" if v >= 0 else "-"
    return f"{sign}{abs(v):08d}"


def _parse_gep(raw: str) -> Tuple[float, float, int]:
    """
    Parse :GEP# response — sTTTTTTTTTTTTTTTTTnn
    Per v3.10 spec:
      sign + digits[0:8]  = declination (0.01 arcsec, signed)
      digits[8:17]        = right ascension (0.01 arcsec, unsigned)
      digits[17]          = pier side  (0=East/default, 1=West, 2=??)
      digits[18]          = pointing state
    Returns (ra_deg, dec_deg, pier_side).

    On a GEM, the same sky position can be reached from East or West of the
    meridian, giving RA values that differ by 12 h (180°).  We normalise to
    the range [0, 360°) so the display is always consistent.
    """
    raw = raw.strip()
    sign = -1 if raw.startswith("-") else 1
    digits = raw.lstrip("+-")
    dec_val  = sign * int(digits[:8])
    ra_val   = int(digits[8:17])
    pier     = int(digits[17]) if len(digits) > 17 else 0
    ra_deg   = _iopval_to_deg(ra_val) % 360.0
    dec_deg  = _iopval_to_deg(dec_val)
    return ra_deg, dec_deg, pier


def _deg_to_hms(deg: float) -> str:
    """Convert RA in decimal degrees to HH:MM:SS string."""
    total_h = (deg % 360.0) / 15.0
    h = int(total_h)
    rem_m = (total_h - h) * 60.0
    m = int(rem_m)
    s = round((rem_m - m) * 60.0)
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        h = (h + 1) % 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def _deg_to_dms(deg: float) -> str:
    """Convert Dec in decimal degrees to ±DD°MM'SS\" string."""
    sign = "+" if deg >= 0 else "-"
    deg = abs(deg)
    d = int(deg)
    rem_m = (deg - d) * 60.0
    m = int(rem_m)
    s = round((rem_m - m) * 60.0)
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d += 1
    return f"{sign}{d:02d}°{m:02d}′{s:02d}″"


def _parse_ra_input(raw: str) -> float:
    """
    Parse RA user input → decimal degrees.
    Accepts: HH:MM:SS  or  HH MM SS  or  decimal degrees/hours.
    """
    raw = raw.strip()
    # HH:MM:SS or HH MM SS
    sep = ":" if ":" in raw else (" " if raw.count(" ") >= 2 else None)
    if sep:
        parts = raw.split(sep)
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return (h + m / 60.0 + s / 3600.0) * 15.0
    # Plain decimal — if < 24 assume hours, else assume degrees
    val = float(raw)
    return val * 15.0 if val < 24.0 else val


def _parse_angle_dms(raw: str) -> float:
    """
    Parse a signed angle (lat or lon) in ±DD:MM:SS, ±DD MM SS, or decimal degrees.
    The sign on the degrees field (or a leading '-') applies to the full value.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty angle input.")

    sep = ":" if ":" in raw else (" " if raw.count(" ") >= 2 else None)
    if sep:
        parts = raw.split(sep)
        if len(parts) != 3:
            raise ValueError(
                f"Expected DD{sep}MM{sep}SS format, got {len(parts)} part(s): {raw!r}"
            )
        d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        if not (0.0 <= m < 60.0):
            raise ValueError(f"Minutes out of range [0, 60): {m}")
        if not (0.0 <= s < 60.0):
            raise ValueError(f"Seconds out of range [0, 60): {s}")
        sign = -1.0 if d < 0 or raw.startswith("-") else 1.0
        return sign * (abs(d) + m / 60.0 + s / 3600.0)

    return float(raw)


def _parse_dec_input(raw: str) -> float:
    """
    Parse Dec user input → decimal degrees.
    Accepts: ±DD:MM:SS  or  ±DD MM SS  or  decimal degrees.
    The sign on the degrees field is applied to the whole value.
    """
    raw = raw.strip()
    sep = ":" if ":" in raw else (" " if raw.count(" ") >= 2 else None)
    if sep:
        parts = raw.split(sep)
        if len(parts) == 3:
            d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            sign = -1.0 if d < 0 or raw.startswith("-") else 1.0
            return sign * (abs(d) + m / 60.0 + s / 3600.0)
    return float(raw)


# ═══════════════════════════════════════════════════════════════════════════════
#  iOptron serial backend  (v3.10 verified commands)
# ═══════════════════════════════════════════════════════════════════════════════

class IOptronHAE:
    """RS-232/USB interface for iOptron HAE-series mounts (protocol v3.10)."""

    def __init__(self, port: str):
        self.port = port
        self._ser: Optional[serial.Serial] = None

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> str:
        self._ser = serial.Serial(
            port=self.port, baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=TIMEOUT,
        )
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        info = self.send_command(":MountInfo#")
        if not info:
            raise ConnectionError("No response from mount.")
        return info

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    @property
    def connected(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    # ── I/O ────────────────────────────────────────────────────────────────────

    def send_command(self, cmd: str) -> str:
        if not self.connected:
            raise RuntimeError("Not connected.")
        self._ser.reset_input_buffer()
        self._ser.write(cmd.encode("ascii"))
        self._ser.flush()
        raw = self._ser.read_until(b"#", size=256)
        return raw.decode("ascii", errors="replace").rstrip("#").strip()

    def send_no_reply(self, cmd: str):
        if not self.connected:
            raise RuntimeError("Not connected.")
        self._ser.reset_input_buffer()
        self._ser.write(cmd.encode("ascii"))
        self._ser.flush()
        time.sleep(0.1)

    # ── Info ───────────────────────────────────────────────────────────────────

    def get_firmware(self) -> str:
        return self.send_command(":FW1#")

    def set_utc_time(self) -> bool:
        """
        Set mount UTC time via :SUT<XXXXXXXXXXXXX>#
        Format: 13-digit unsigned ms since J2000, matching iOptron driver.
        """
        J2000     = 2451545.0
        now       = datetime.datetime.now(datetime.timezone.utc)
        # Julian Day of now
        epoch     = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        JD        = J2000 + (now - epoch).total_seconds() / 86400.0
        ms_jd     = int((JD - J2000) * 8.64e7)
        cmd       = f":SUT{ms_jd:013d}#"
        resp      = self.send_command(cmd)
        return resp[:1] == "1"

    def set_utc_offset(self) -> bool:
        """
        Set mount UTC offset via :SG<s><MMM>#
        Format: sign + 3-digit minutes (e.g. MST = -420 → :SG-420#)
        """
        offset_min = int(-time.altzone / 60) if time.daylight else int(-time.timezone / 60)
        sign       = "+" if offset_min >= 0 else "-"
        cmd        = f":SG{sign}{abs(offset_min):03d}#"
        resp       = self.send_command(cmd)
        return resp[:1] == "1"

    # ── Location  :SLO#  :SLA# ────────────────────────────────────────────────

    def set_longitude(self, lon_deg: float) -> bool:
        """
        :SLOsTTTTTTTT# — set longitude. East positive.
        Range [-64,800,000, +64,800,000] in 0.01 arcsec units.
        """
        v    = round(lon_deg * 360000)
        sign = "+" if v >= 0 else "-"
        cmd  = f":SLO{sign}{abs(v):08d}#"
        return self.send_command(cmd)[:1] == "1"

    def set_latitude(self, lat_deg: float) -> bool:
        """
        :SLAsTTTTTTTT# — set latitude. North positive.
        Range [-32,400,000, +32,400,000] in 0.01 arcsec units.
        """
        v    = round(lat_deg * 360000)
        sign = "+" if v >= 0 else "-"
        cmd  = f":SLA{sign}{abs(v):08d}#"
        return self.send_command(cmd)[:1] == "1"

    # ── Tracking  :RT<n>#  :ST0#  :ST1# ───────────────────────────────────────

    def set_tracking_rate(self, rate_code: str):
        """:RT0#=sidereal  :RT1#=lunar  :RT2#=solar"""
        self.send_command(f":RT{rate_code}#")

    def tracking_on(self):
        """:ST1#"""
        self.send_command(":ST1#")

    def tracking_off(self):
        """:ST0#"""
        self.send_command(":ST0#")

    # ── Slew rate  :SRn# ───────────────────────────────────────────────────────

    def set_slew_rate(self, rate: int):
        """:SR<1-9>#  — 1=1×sidereal … 9=max speed"""
        self.send_command(f":SR{max(1, min(9, rate))}#")

    # ── EQ GoTo  :SRA<9d>#  :Sd<s><8d>#  :MS1# ───────────────────────────────

    def goto_radec(self, ra_deg: float, dec_deg: float) -> Tuple[bool, str]:
        """
        Slew to RA/Dec (v3.10).
        :SRA<TTTTTTTTT>#  — set target RA (9 digits unsigned, 0.01 arcsec)
        :Sd<s><TTTTTTTT># — set target Dec (sign + 8 digits, 0.01 arcsec)
        :MS1#             — slew to normal (non-CW-up) position
        Response of :MS1# : "1"=accepted, "0"=below limit / mechanical limit.
        After slewing, tracking is enabled automatically by the mount.
        """
        r1 = self.send_command(f":SRA{_encode_ra(ra_deg)}#")
        r2 = self.send_command(f":Sd{_encode_dec(dec_deg)}#")
        if r1[:1] != "1" or r2[:1] != "1":
            return False, f"Mount rejected coordinates (SRA→{r1}, Sd→{r2})"
        resp = self.send_command(":MS1#")
        # Some firmware versions return "1" followed by a position dump,
        # e.g. "1+3240000010008706521". Check the first character only.
        first = resp[:1] if resp else ""
        msgs = {"1": "Slewing…", "0": "Below altitude limit or mechanical limit"}
        return first == "1", msgs.get(first, f"Unknown response: {resp}")

    def sync_radec(self, ra_deg: float, dec_deg: float) -> bool:
        """
        Sync / calibrate mount to current RA/Dec.
        :SRA<TTTTTTTTT>#  — define RA
        :Sd<s><TTTTTTTT># — define Dec
        :CM#              — synchronize (returns "1")
        """
        self.send_command(f":SRA{_encode_ra(ra_deg)}#")
        self.send_command(f":Sd{_encode_dec(dec_deg)}#")
        resp = self.send_command(":CM#")
        return resp[:1] == "1"

    # ── Position readback  :GEP# ───────────────────────────────────────────────

    def get_current_radec(self) -> Tuple[float, float, int]:
        """
        Returns (ra_deg, dec_deg, pier_side).
        pier_side: 0=East (normal), 1=West (flipped), 2=unknown
        """
        raw = self.send_command(":GEP#")
        return _parse_gep(raw)

    # ── Mount status  :GLS# ────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        :GLS# — get longitude, latitude, and all status flags.
        Returns dict with keys: system_status, tracking_rate, hemisphere, etc.
        system_status codes: 0=stopped(non-zero), 1=tracking, 2=slewing,
          3=auto-guiding, 4=meridian-flip, 5=tracking+PEC, 6=parked, 7=at-zero.
        """
        raw = self.send_command(":GLS#")
        if len(raw) < 22:
            return {}
        return {
            "system_status":  int(raw[17]),
            "tracking_rate":  int(raw[18]),
            "slew_speed":     int(raw[19]),
            "time_source":    int(raw[20]),
            "hemisphere":     int(raw[21]),
        }

    # ── Park / Home ────────────────────────────────────────────────────────────

    def go_home(self):
        """:MH# — slew to zero/home position."""
        self.send_command(":MH#")

    def unpark(self):
        """:MP0# — unpark the mount."""
        self.send_command(":MP0#")

    # ── Stop ───────────────────────────────────────────────────────────────────

    def stop_slew(self):
        """:Q# — stop all slewing; tracking unaffected."""
        self.send_command(":Q#")

    # ── Manual nudge  :mn# :ms# :me# :mw#  :qD# :qR# ─────────────────────────
    #   Per v3.10:
    #     :mn# = Dec−   :ms# = Dec+   :me# = RA−   :mw# = RA+
    #   Stop:
    #     :qD# stops N/S (Dec)    :qR# stops E/W (RA)

    def move_direction(self, direction: str, start: bool):
        """Start/stop manual motion in N/S/E/W (Dec/RA axes)."""
        if start:
            cmd = {
                "n": ":mn#",   # Dec−
                "s": ":ms#",   # Dec+
                "e": ":me#",   # RA−
                "w": ":mw#",   # RA+
            }.get(direction.lower())
            if cmd:
                self.send_no_reply(cmd)
        else:
            if direction.lower() in ("n", "s"):
                self.send_command(":qD#")   # stop Dec axis
            else:
                self.send_command(":qR#")   # stop RA axis


# ═══════════════════════════════════════════════════════════════════════════════
#  iOptron WiFi backend  (TCP — same protocol, different transport)
# ═══════════════════════════════════════════════════════════════════════════════

class IOptronHAEWifi(IOptronHAE):
    """WiFi/TCP transport for iOptron HAE-series mounts."""

    def __init__(self, ip: str, port: int = WIFI_DEFAULT_PORT):
        super().__init__(port="")
        self._ip   = ip
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()   # serialises all socket I/O

    def connect(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self._ip, self._port))
        self._sock = s
        info = self.send_command(":MountInfo#")
        if not info:
            raise ConnectionError("No response from mount over WiFi.")
        return info

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def send_command(self, cmd: str) -> str:
        if not self.connected:
            raise RuntimeError("Not connected.")
        with self._lock:
            self._sock.sendall(cmd.encode("ascii"))
            buf = b""
            deadline = time.time() + TIMEOUT
            while time.time() < deadline:
                try:
                    chunk = self._sock.recv(256)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buf += chunk
                if b"#" in buf:
                    break
            raw = buf.decode("ascii", errors="replace")
            return raw.split("#")[0].strip()

    def send_no_reply(self, cmd: str):
        if not self.connected:
            raise RuntimeError("Not connected.")
        with self._lock:
            self._sock.sendall(cmd.encode("ascii"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Colour palette
# ═══════════════════════════════════════════════════════════════════════════════

PAL = {
    "bg":        "#0d0f14",
    "panel":     "#13161e",
    "border":    "#1e2330",
    "accent":    "#a855f7",    # purple — EQ
    "accent2":   "#3b82f6",    # blue   — connectivity
    "accent3":   "#22c55e",    # green  — success / tracking on
    "danger":    "#ef4444",
    "success":   "#22c55e",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
    "entry_bg":  "#1a1f2e",
    "btn_hover": "#9333ea",
    "track_on":  "#22c55e",
    "track_off": "#374151",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ═══════════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════════

class MountControlApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.mount: Optional[IOptronHAE] = None
        self._polling       = False
        self._poll_thread   = None
        self._tracking_on   = False
        self._tracking_rate = TRACK_SIDEREAL
        self._conn_mode     = tk.StringVar(value="wifi")
        self._nudge_active: set = set()

        self._setup_window()
        self._build_ui()
        self._refresh_ports()
        self._set_mount_controls(False)

    # ── Window ─────────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.title("iOptron HAE16C  ·  EQ Controller")
        self.geometry("980x820")
        self.minsize(860, 700)
        self.configure(fg_color=PAL["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Top-level layout ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        self._tabs = ctk.CTkTabview(
            self,
            fg_color=PAL["bg"],
            segmented_button_fg_color=PAL["panel"],
            segmented_button_selected_color=PAL["accent"],
            segmented_button_selected_hover_color=PAL["btn_hover"],
            segmented_button_unselected_color=PAL["panel"],
            segmented_button_unselected_hover_color=PAL["border"],
            text_color=PAL["text"],
        )
        self._tabs.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self._tabs.add("EQ Controller")
        self._tabs.add("Set")

        self._build_eq_tab(self._tabs.tab("EQ Controller"))
        self._build_set_tab(self._tabs.tab("Set"))

    def _build_eq_tab(self, parent):
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left  = ctk.CTkFrame(content, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_connection_panel(left)
        self._build_position_panel(left)
        self._build_tracking_panel(left)

        self._build_goto_panel(right)
        self._build_nudge_panel(right)
        self._build_log_panel(right)

    def _build_set_tab(self, parent):
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.pack(fill="both", expand=True)

        panel = self._panel(content, "LOCATION")

        # Latitude row
        lat_row = ctk.CTkFrame(panel, fg_color="transparent")
        lat_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(lat_row, text="Latitude",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=13), width=80, anchor="w",
                     ).pack(side="left")
        ctk.CTkLabel(lat_row, text="(±DD.dddd  or  ±DD:MM:SS   N=+)",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=11), anchor="w",
                     ).pack(side="left", padx=(4, 0))
        self._lat_entry = ctk.CTkEntry(
            lat_row, width=180,
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"], placeholder_text="e.g. 32:13:00",
        )
        self._lat_entry.pack(side="right")

        # Longitude row
        lon_row = ctk.CTkFrame(panel, fg_color="transparent")
        lon_row.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(lon_row, text="Longitude",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=13), width=80, anchor="w",
                     ).pack(side="left")
        ctk.CTkLabel(lon_row, text="(±DDD.dddd  or  ±DDD:MM:SS  E=+)",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=11), anchor="w",
                     ).pack(side="left", padx=(4, 0))
        self._lon_entry = ctk.CTkEntry(
            lon_row, width=180,
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"], placeholder_text="e.g. -110:56:00",
        )
        self._lon_entry.pack(side="right")

        # Set button
        self._set_location_btn = ctk.CTkButton(
            panel, text="Set Lat/Long",
            height=40, corner_radius=8,
            fg_color=PAL["accent2"], hover_color="#2563eb",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._set_location,
        )
        self._set_location_btn.pack(fill="x")

        # ── Plate Solve Calibration panel ──────────────────────────────────────
        ps_panel = self._panel(content, "PLATE SOLVE CALIBRATION  —  :SRA / :Sd / :CM")

        # RA row
        ps_ra_row = ctk.CTkFrame(ps_panel, fg_color="transparent")
        ps_ra_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(ps_ra_row, text="Target RA",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=13), width=90, anchor="w",
                     ).pack(side="left")
        ctk.CTkLabel(ps_ra_row, text="(HH:MM:SS  or  decimal °)",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=11), anchor="w",
                     ).pack(side="left", padx=(4, 0))
        self._ps_ra_entry = ctk.CTkEntry(
            ps_ra_row, width=180,
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"], placeholder_text="e.g. 05:34:32",
        )
        self._ps_ra_entry.pack(side="right")

        # Dec row
        ps_dec_row = ctk.CTkFrame(ps_panel, fg_color="transparent")
        ps_dec_row.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(ps_dec_row, text="Target Dec",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=13), width=90, anchor="w",
                     ).pack(side="left")
        ctk.CTkLabel(ps_dec_row, text="(±DD:MM:SS  or  decimal °)",
                     text_color=PAL["muted"],
                     font=ctk.CTkFont(size=11), anchor="w",
                     ).pack(side="left", padx=(4, 0))
        self._ps_dec_entry = ctk.CTkEntry(
            ps_dec_row, width=180,
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"], placeholder_text="e.g. +22:00:52",
        )
        self._ps_dec_entry.pack(side="right")

        # CM calibrate button
        self._ps_calibrate_btn = ctk.CTkButton(
            ps_panel,
            text="⊕  Set RA/Dec  (only use for plate solve calibration)",
            height=44, corner_radius=8,
            fg_color="#7c3aed", hover_color="#6d28d9",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._calibrate_plate_solve,
        )
        self._ps_calibrate_btn.pack(fill="x")

    # ── Header ─────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=PAL["panel"], corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="◈  iOptron HAE16C  ·  Equatorial Controller",
            font=ctk.CTkFont(family="Courier New", size=16, weight="bold"),
            text_color=PAL["accent"],
        ).pack(side="left", padx=20, pady=14)

        self._conn_indicator = ctk.CTkLabel(
            hdr, text="● DISCONNECTED",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            text_color=PAL["danger"],
        )
        self._conn_indicator.pack(side="right", padx=20)

    # ── Connection panel ───────────────────────────────────────────────────────

    def _build_connection_panel(self, parent):
        panel = self._panel(parent, "CONNECTION")

        toggle_row = ctk.CTkFrame(panel, fg_color="transparent")
        toggle_row.pack(fill="x", pady=(0, 10))

        self._btn_mode_serial = ctk.CTkButton(
            toggle_row, text="⬡  SERIAL", width=90, height=30, corner_radius=6,
            fg_color=PAL["entry_bg"], hover_color="#2563eb",
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=lambda: self._switch_conn_mode("serial"),
        )
        self._btn_mode_serial.pack(side="left", padx=(0, 6))

        self._btn_mode_wifi = ctk.CTkButton(
            toggle_row, text="⬡  WI-FI", width=90, height=30, corner_radius=6,
            fg_color=PAL["accent2"], hover_color="#2563eb",
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=lambda: self._switch_conn_mode("wifi"),
        )
        self._btn_mode_wifi.pack(side="left")

        # Serial sub-panel
        self._serial_panel = ctk.CTkFrame(panel, fg_color="transparent")
        row = ctk.CTkFrame(self._serial_panel, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(row, text="Port", text_color=PAL["muted"],
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 8))
        self._port_var  = tk.StringVar(value="COM3")
        self._port_menu = ctk.CTkOptionMenu(
            row, variable=self._port_var, width=130,
            fg_color=PAL["entry_bg"], button_color=PAL["accent"],
            button_hover_color=PAL["btn_hover"], text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=12),
        )
        self._port_menu.pack(side="left")
        ctk.CTkButton(
            row, text="⟳", width=36, fg_color=PAL["border"],
            hover_color=PAL["accent"], text_color=PAL["text"],
            command=self._refresh_ports,
        ).pack(side="left", padx=6)

        # WiFi sub-panel (shown by default)
        self._wifi_panel = ctk.CTkFrame(panel, fg_color="transparent")
        wifi_fields = ctk.CTkFrame(self._wifi_panel, fg_color="transparent")
        wifi_fields.pack(fill="x", pady=(0, 8))
        wifi_fields.columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(wifi_fields, text="IP ADDRESS",
                     text_color=PAL["muted"], font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(wifi_fields, text="PORT",
                     text_color=PAL["muted"], font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self._wifi_ip_var = tk.StringVar(value=WIFI_DEFAULT_IP)
        ctk.CTkEntry(
            wifi_fields, textvariable=self._wifi_ip_var,
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=13), height=38,
        ).grid(row=1, column=0, sticky="ew")

        self._wifi_port_var = tk.IntVar(value=WIFI_DEFAULT_PORT)
        ctk.CTkEntry(
            wifi_fields, textvariable=self._wifi_port_var,
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=13), height=38,
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0))

        self._wifi_panel.pack(fill="x", pady=(0, 8))

        # Buttons
        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(fill="x")

        self._connect_btn = ctk.CTkButton(
            btn_row, text="CONNECT", width=110,
            fg_color=PAL["accent2"], hover_color="#2563eb",
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            command=self._toggle_connect,
        )
        self._connect_btn.pack(side="left", padx=(0, 8))

        self._home_btn = ctk.CTkButton(
            btn_row, text="⌂ ZERO", width=90, height=36,
            fg_color=PAL["danger"], hover_color="#b91c1c",
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            text_color="#ffffff", command=self._go_home,
        )
        self._home_btn.pack(side="left", padx=(0, 6))

        self._unpark_btn = ctk.CTkButton(
            btn_row, text="UNPARK", width=90,
            fg_color=PAL["accent3"], hover_color="#16a34a",
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            command=self._unpark,
        )
        self._unpark_btn.pack(side="left")

    # ── Live position panel ────────────────────────────────────────────────────

    def _build_position_panel(self, parent):
        panel = self._panel(parent, "CURRENT POSITION  —  live EQ coordinates")

        grid = ctk.CTkFrame(panel, fg_color="transparent")
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        def big_row(row_idx, tag, color):
            ctk.CTkLabel(grid, text=tag, text_color=PAL["muted"],
                         font=ctk.CTkFont(size=11), anchor="w",
                         ).grid(row=row_idx, column=0, sticky="w", pady=2, padx=(0, 12))
            var = tk.StringVar(value="---")
            ctk.CTkLabel(grid, textvariable=var, text_color=color,
                         font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
                         anchor="e").grid(row=row_idx, column=1, sticky="e")
            return var

        self._disp_ra   = big_row(0, "RA",   PAL["accent2"])
        self._disp_dec  = big_row(1, "DEC",  PAL["accent"])
        self._disp_fw   = big_row(2, "FW",   PAL["muted"])
        self._disp_pier = big_row(3, "PIER", PAL["muted"])

        self._mk_sep(panel)

        # Slew rate
        rate_row = ctk.CTkFrame(panel, fg_color="transparent")
        rate_row.pack(fill="x")
        ctk.CTkLabel(rate_row, text="Slew rate", text_color=PAL["muted"],
                     font=ctk.CTkFont(size=12)).pack(side="left")
        self._rate_var = tk.IntVar(value=9)
        self._rate_slider = ctk.CTkSlider(
            rate_row, from_=1, to=9, number_of_steps=8,
            variable=self._rate_var,
            button_color=PAL["accent"], button_hover_color=PAL["btn_hover"],
            progress_color=PAL["accent"],
            command=self._on_rate_change,
        )
        self._rate_slider.pack(side="left", fill="x", expand=True, padx=10)
        self._rate_lbl = ctk.CTkLabel(
            rate_row, text="9", text_color=PAL["accent"],
            font=ctk.CTkFont(family="Courier New", size=14, weight="bold"), width=20,
        )
        self._rate_lbl.pack(side="left")

        self._mk_sep(panel)

        # Poll button
        self._poll_btn = ctk.CTkButton(
            panel, text="▶  START LIVE POLLING", height=34,
            fg_color=PAL["border"], hover_color=PAL["accent2"],
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=11),
            command=self._toggle_polling,
        )
        self._poll_btn.pack(fill="x")

    # ── Tracking panel ─────────────────────────────────────────────────────────

    def _build_tracking_panel(self, parent):
        panel = self._panel(parent, "TRACKING")

        rate_sel = ctk.CTkFrame(panel, fg_color="transparent")
        rate_sel.pack(fill="x", pady=(0, 8))

        self._track_rate_var = tk.StringVar(value="Sidereal")
        for label, code in [("Sidereal", TRACK_SIDEREAL),
                             ("Solar",    TRACK_SOLAR),
                             ("Lunar",    TRACK_LUNAR)]:
            ctk.CTkRadioButton(
                rate_sel, text=label, value=label,
                variable=self._track_rate_var,
                font=ctk.CTkFont(family="Courier New", size=12),
                text_color=PAL["text"],
                fg_color=PAL["accent"], hover_color=PAL["btn_hover"],
                command=lambda c=code: self._on_track_rate_select(c),
            ).pack(side="left", padx=(0, 14))

        self._track_toggle_btn = ctk.CTkButton(
            panel, text="⏺  TRACKING OFF", height=38,
            fg_color=PAL["track_off"], hover_color="#4b5563",
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            command=self._toggle_tracking,
        )
        self._track_toggle_btn.pack(fill="x")

        self._track_status_lbl = ctk.CTkLabel(
            panel, text="Tracking: OFF",
            text_color=PAL["muted"],
            font=ctk.CTkFont(family="Courier New", size=11),
        )
        self._track_status_lbl.pack(anchor="w", pady=(4, 0))

    # ── EQ GoTo panel ──────────────────────────────────────────────────────────

    def _build_goto_panel(self, parent):
        panel = self._panel(parent, "GOTO TARGET  —  RIGHT ASCENSION & DECLINATION")

        eq_fields = ctk.CTkFrame(panel, fg_color="transparent")
        eq_fields.pack(fill="x", pady=(0, 8))
        eq_fields.columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(eq_fields, text="RIGHT ASCENSION  (HH:MM:SS  or  hours  or  °)",
                     text_color=PAL["muted"], font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(eq_fields, text="DECLINATION  (−90 … +90 °)",
                     text_color=PAL["muted"], font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self._ra_entry = ctk.CTkEntry(
            eq_fields, placeholder_text="05:34:32",
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=15), height=44,
        )
        self._ra_entry.grid(row=1, column=0, sticky="ew")

        self._dec_entry = ctk.CTkEntry(
            eq_fields, placeholder_text="+22.01",
            fg_color=PAL["entry_bg"], border_color=PAL["border"],
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=15), height=44,
        )
        self._dec_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        eq_btns = ctk.CTkFrame(panel, fg_color="transparent")
        eq_btns.pack(fill="x", pady=(0, 8))

        self._eq_goto_btn = ctk.CTkButton(
            eq_btns, text="⊕  SLEW TO TARGET", height=42,
            fg_color=PAL["accent"], hover_color=PAL["btn_hover"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Courier New", size=13, weight="bold"),
            command=self._goto_eq,
        )
        self._eq_goto_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            eq_btns, text="⬛  STOP", width=100, height=42,
            fg_color=PAL["danger"], hover_color="#b91c1c",
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            text_color="#ffffff", command=self._stop,
        )
        self._stop_btn.pack(side="left")

        # Sync row
        self._mk_sep(panel)
        sync_row = ctk.CTkFrame(panel, fg_color="transparent")
        sync_row.pack(fill="x")
        ctk.CTkLabel(sync_row, text="Sync mount to entered coords  →",
                     text_color=PAL["muted"], font=ctk.CTkFont(size=11)
                     ).pack(side="left", padx=(0, 8))
        self._sync_btn = ctk.CTkButton(
            sync_row, text="⊙  SYNC HERE", height=34, width=130,
            fg_color=PAL["border"], hover_color=PAL["accent"],
            text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            command=self._sync_eq,
        )
        self._sync_btn.pack(side="left")

    # ── Nudge panel ────────────────────────────────────────────────────────────

    def _build_nudge_panel(self, parent):
        panel = self._panel(parent, "MANUAL NUDGE  —  hold to move  (N/S=Dec  E/W=RA)")

        dpad = ctk.CTkFrame(panel, fg_color="transparent")
        dpad.pack()

        btn_cfg = dict(
            width=56, height=56, corner_radius=8,
            fg_color=PAL["entry_bg"], hover_color=PAL["border"],
            text_color=PAL["text"], font=ctk.CTkFont(size=20),
        )

        self._nudge_btns = {}

        def _make(text, direction, row, col):
            btn = ctk.CTkButton(dpad, text=text, **btn_cfg)
            btn.grid(row=row, column=col, padx=4, pady=4)
            btn.bind("<ButtonPress-1>",   lambda e, d=direction: self._nudge_press(d))
            btn.bind("<ButtonRelease-1>", lambda e, d=direction: self._nudge_release(d))
            self._nudge_btns[direction] = btn
            return btn

        _make("▲\nDec+", "s", 0, 1)   # :ms# = Dec+
        _make("◀\nRA−",  "e", 1, 0)   # :me# = RA−
        ctk.CTkLabel(dpad, text="✛", text_color=PAL["muted"],
                     font=ctk.CTkFont(size=20), width=56, height=56).grid(row=1, column=1)
        _make("▶\nRA+",  "w", 1, 2)   # :mw# = RA+
        _make("▼\nDec−", "n", 2, 1)   # :mn# = Dec−

    # ── Log panel ──────────────────────────────────────────────────────────────

    def _build_log_panel(self, parent):
        panel = self._panel(parent, "ACTIVITY LOG", expand=True)
        self._log_box = ctk.CTkTextbox(
            panel, height=120,
            fg_color=PAL["entry_bg"], text_color=PAL["text"],
            font=ctk.CTkFont(family="Courier New", size=11),
            border_color=PAL["border"], border_width=1,
            state="disabled", wrap="word",
        )
        self._log_box.pack(fill="both", expand=True)

    # ── Widget helpers ─────────────────────────────────────────────────────────

    def _mk_sep(self, parent):
        ctk.CTkFrame(parent, fg_color=PAL["border"], height=1).pack(fill="x", pady=8)

    def _panel(self, parent, title: str, expand: bool = False) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(parent, fg_color=PAL["panel"],
                             corner_radius=10, border_color=PAL["border"], border_width=1)
        outer.pack(fill="both", expand=expand, pady=(0, 10))
        ctk.CTkLabel(
            outer, text=title,
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=PAL["accent"],
        ).pack(anchor="w", padx=14, pady=(10, 0))
        ctk.CTkFrame(outer, fg_color=PAL["border"], height=1).pack(fill="x", padx=14, pady=(4, 0))
        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=14, pady=10)
        return inner

    def _set_mount_controls(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (
            self._eq_goto_btn, self._stop_btn, self._sync_btn,
            self._home_btn, self._unpark_btn,
            self._poll_btn, self._rate_slider, self._track_toggle_btn,
            self._set_location_btn, self._ps_calibrate_btn,
        ):
            w.configure(state=state)
        for btn in self._nudge_btns.values():
            btn.configure(state=state)

    # ── Logging ────────────────────────────────────────────────────────────────

    def log(self, msg: str, color: str = PAL["text"]):
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
            self._btn_mode_serial.configure(fg_color=PAL["accent2"])
            self._btn_mode_wifi.configure(fg_color=PAL["entry_bg"])
        else:
            self._serial_panel.pack_forget()
            self._wifi_panel.pack(fill="x", pady=(0, 8))
            self._btn_mode_wifi.configure(fg_color=PAL["accent2"])
            self._btn_mode_serial.configure(fg_color=PAL["entry_bg"])

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
        self._connect_btn.configure(state="disabled", text="CONNECTING…")

        def _worker():
            try:
                mount = IOptronHAE(port)
                info  = mount.connect()
                fw    = mount.get_firmware()
                self.mount = mount
                self.after(0, lambda i=info, f=fw: self._on_connected(f"Serial {port}", i, f))
            except Exception as exc:
                err = str(exc)
                self.after(0, lambda e=err: self._on_serial_fail(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_serial_fail(self, err: str):
        self.mount = None
        self._connect_btn.configure(state="normal", text="CONNECT")
        self.log(f"Connection failed: {err}", PAL["danger"])
        messagebox.showerror("Connection Error", err)

    def _connect_wifi(self):
        ip = self._wifi_ip_var.get().strip() or WIFI_DEFAULT_IP
        try:
            port = int(str(self._wifi_port_var.get()).strip())
        except (ValueError, tk.TclError):
            port = WIFI_DEFAULT_PORT
        self.log(f"Connecting via WiFi  {ip}:{port}…")
        self._connect_btn.configure(state="disabled", text="CONNECTING…")

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
        self._conn_indicator.configure(text="● CONNECTED", text_color=PAL["success"])
        self._connect_btn.configure(
            text="DISCONNECT", state="normal",
            fg_color=PAL["danger"], hover_color="#b91c1c",
        )
        self._set_mount_controls(True)
        self.log(f"Connected ({label}) — model:{info}  FW:{fw}", PAL["success"])
        # Sync UTC time and offset to mount on every connect
        try:
            if self.mount.set_utc_time():
                self.log("UTC time synced to mount.", PAL["success"])
            else:
                self.log("UTC time sync failed (mount rejected :SUT#).", PAL["danger"])
        except Exception as exc:
            self.log(f"UTC time sync error: {exc}", PAL["danger"])
        try:
            if self.mount.set_utc_offset():
                self.log("UTC offset synced to mount.", PAL["success"])
            else:
                self.log("UTC offset sync failed (mount rejected :SG#).", PAL["danger"])
        except Exception as exc:
            self.log(f"UTC offset sync error: {exc}", PAL["danger"])
        self.log("Tip: start polling to see live RA/Dec position.", PAL["muted"])

    def _on_wifi_fail(self, err: str):
        self.mount = None
        self._connect_btn.configure(state="normal", text="CONNECT")
        self.log(f"WiFi connection failed: {err}", PAL["danger"])
        messagebox.showerror("WiFi Error", err)

    def _disconnect(self):
        self._stop_polling()
        if self.mount:
            self.mount.disconnect()
            self.mount = None
        self._tracking_on = False
        self._update_track_ui()
        self._conn_indicator.configure(text="● DISCONNECTED", text_color=PAL["danger"])
        self._connect_btn.configure(text="CONNECT", state="normal",
                                    fg_color=PAL["accent2"], hover_color="#2563eb")
        self._set_mount_controls(False)
        for v in (self._disp_ra, self._disp_dec, self._disp_fw, self._disp_pier):
            v.set("---")
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
            messagebox.showwarning("Invalid Input", str(e))
            return
        ra_hms = _deg_to_hms(ra_deg)
        dec_dms = _deg_to_dms(dec_deg)
        self.log(f"[EQ GoTo]  RA={ra_hms}  Dec={dec_dms}", PAL["accent"])
        threading.Thread(target=self._run_eq_goto, args=(ra_deg, dec_deg), daemon=True).start()

    def _run_eq_goto(self, ra_deg, dec_deg):
        try:
            ok, msg = self.mount.goto_radec(ra_deg, dec_deg)
            color = PAL["success"] if ok else PAL["danger"]
            self.after(0, lambda: self.log(f"  → {msg}", color))
        except Exception as exc:
            self.after(0, lambda: self.log(f"Error: {exc}", PAL["danger"]))

    # ── Sync ───────────────────────────────────────────────────────────────────

    def _sync_eq(self):
        if not self._check_connected(): return
        try:
            ra_deg, dec_deg = self._parse_inputs()
        except ValueError as e:
            messagebox.showwarning("Invalid Input", str(e))
            return
        ra_hms = _deg_to_hms(ra_deg)
        dec_dms = _deg_to_dms(dec_deg)
        self.log(f"[Sync]  RA={ra_hms}  Dec={dec_dms}", PAL["muted"])

        def _worker():
            try:
                ok = self.mount.sync_radec(ra_deg, dec_deg)
                msg = "Sync accepted." if ok else "Sync failed."
                col = PAL["success"] if ok else PAL["danger"]
                self.after(0, lambda m=msg, c=col: self.log(f"  → {m}", c))
            except Exception as exc:
                self.after(0, lambda e=exc: self.log(f"Sync error: {e}", PAL["danger"]))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Stop / Home / Unpark ───────────────────────────────────────────────────

    def _stop(self):
        if not self._check_connected(): return
        def _worker():
            try:
                self.mount.stop_slew()
                self.after(0, lambda: self.log("STOP — slew halted.", PAL["danger"]))
            except Exception as exc:
                self.after(0, lambda e=exc: self.log(f"Error: {e}", PAL["danger"]))
        threading.Thread(target=_worker, daemon=True).start()

    def _go_home(self):
        if not self._check_connected(): return
        def _worker():
            try:
                self.mount.go_home()
                self.after(0, lambda: self.log("Slewing to zero/home position (:MH#)…", PAL["muted"]))
            except Exception as exc:
                self.after(0, lambda e=exc: self.log(f"Error: {e}", PAL["danger"]))
        threading.Thread(target=_worker, daemon=True).start()

    def _unpark(self):
        if not self._check_connected(): return
        def _worker():
            try:
                self.mount.unpark()
                self.after(0, lambda: self.log("Unparked (:MP0#).", PAL["success"]))
            except Exception as exc:
                self.after(0, lambda e=exc: self.log(f"Error: {e}", PAL["danger"]))
        threading.Thread(target=_worker, daemon=True).start()

    # ── Slew rate ──────────────────────────────────────────────────────────────

    def _on_rate_change(self, val):
        rate = int(round(float(val)))
        self._rate_lbl.configure(text=str(rate))
        if self.mount and self.mount.connected:
            # Cancel any pending rate-change and schedule a new one (debounce)
            if hasattr(self, "_rate_after_id") and self._rate_after_id:
                self.after_cancel(self._rate_after_id)
            def _send(r=rate):
                self._rate_after_id = None
                def _worker():
                    try:
                        self.mount.set_slew_rate(r)
                    except Exception:
                        pass
                threading.Thread(target=_worker, daemon=True).start()
            self._rate_after_id = self.after(200, _send)

    # ── Tracking ───────────────────────────────────────────────────────────────

    def _on_track_rate_select(self, code: str):
        self._tracking_rate = code
        label = self._track_rate_var.get()
        self.log(f"Tracking rate selected: {label}")
        if self._tracking_on and self.mount and self.mount.connected:
            def _worker():
                try:
                    self.mount.set_tracking_rate(code)
                except Exception as exc:
                    self.after(0, lambda e=exc: self.log(f"Rate change error: {e}", PAL["danger"]))
            threading.Thread(target=_worker, daemon=True).start()

    def _toggle_tracking(self):
        if not self._check_connected(): return
        self._tracking_on = not self._tracking_on
        self._update_track_ui()   # update UI immediately; revert on error
        tracking_on = self._tracking_on
        rate_code   = self._tracking_rate

        def _worker():
            try:
                if tracking_on:
                    self.mount.set_tracking_rate(rate_code)
                    self.mount.tracking_on()
                    label = self._track_rate_var.get()
                    self.after(0, lambda l=label: self.log(f"Tracking ON  [{l}]", PAL["success"]))
                else:
                    self.mount.tracking_off()
                    self.after(0, lambda: self.log("Tracking OFF.", PAL["muted"]))
            except Exception as exc:
                # Revert the toggle on error
                self.after(0, lambda e=exc: self._tracking_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _tracking_error(self, exc):
        self._tracking_on = not self._tracking_on   # revert
        self._update_track_ui()
        self.log(f"Tracking error: {exc}", PAL["danger"])

    def _update_track_ui(self):
        if self._tracking_on:
            label = self._track_rate_var.get()
            self._track_toggle_btn.configure(
                text=f"⏹  TRACKING ON  [{label}]",
                fg_color=PAL["track_on"], hover_color="#16a34a",
            )
            self._track_status_lbl.configure(
                text=f"Tracking: ON  ({label})", text_color=PAL["success"])
        else:
            self._track_toggle_btn.configure(
                text="⏺  TRACKING OFF",
                fg_color=PAL["track_off"], hover_color="#4b5563",
            )
            self._track_status_lbl.configure(
                text="Tracking: OFF", text_color=PAL["muted"])

    # ── Nudge ──────────────────────────────────────────────────────────────────

    def _nudge_press(self, direction: str):
        if not self._check_connected(): return
        if direction in self._nudge_active: return
        self._nudge_active.add(direction)
        axis = "Dec" if direction in ("n", "s") else "RA"
        sign = "+" if direction in ("s", "w") else "−"
        self.log(f"Nudge {axis}{sign} — moving…")
        if direction in self._nudge_btns:
            self._nudge_btns[direction].configure(fg_color=PAL["accent"])

        def _worker():
            try:
                self.mount.move_direction(direction, True)
            except Exception as exc:
                self._nudge_active.discard(direction)
                self.after(0, lambda e=exc: self.log(f"Error: {e}", PAL["danger"]))
                if direction in self._nudge_btns:
                    self.after(0, lambda d=direction: self._nudge_btns[d].configure(fg_color=PAL["entry_bg"]))

        threading.Thread(target=_worker, daemon=True).start()

    def _nudge_release(self, direction: str):
        self._nudge_active.discard(direction)
        if direction in self._nudge_btns:
            self._nudge_btns[direction].configure(fg_color=PAL["entry_bg"])

        def _worker():
            try:
                self.mount.move_direction(direction, False)
                axis = "Dec" if direction in ("n", "s") else "RA"
                self.after(0, lambda a=axis: self.log(f"Nudge {a} — stopped."))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

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
        self._poll_btn.configure(text="⏸  STOP POLLING", fg_color=PAL["accent2"])
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.log("Live position polling started (1 s interval).")

    def _stop_polling(self):
        self._polling = False
        self._poll_btn.configure(text="▶  START LIVE POLLING", fg_color=PAL["border"])

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
                    self.after(0, lambda e=exc: self.log(f"Poll error: {e}", PAL["muted"]))
                elif consecutive_errors == 4:
                    self.after(0, lambda: self.log("Poll errors suppressed (repeated)…", PAL["muted"]))
            time.sleep(1)

    # ── Set Location ───────────────────────────────────────────────────────────

    def _set_location(self):
        if not self._check_connected():
            return
        try:
            lat = _parse_angle_dms(self._lat_entry.get())
            lon = _parse_angle_dms(self._lon_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid Input",
                "Enter latitude and longitude as decimal degrees "
                "or DD:MM:SS (e.g. 32:13:00 / -110:56:00).")
            return
        if not (-90.0 <= lat <= 90.0):
            messagebox.showwarning("Invalid Input", "Latitude must be between −90 and +90.")
            return
        if not (-180.0 <= lon <= 180.0):
            messagebox.showwarning("Invalid Input", "Longitude must be between −180 and +180.")
            return

        def _worker():
            try:
                lat_ok = self.mount.set_latitude(lat)
                lon_ok = self.mount.set_longitude(lon)
                if lat_ok and lon_ok:
                    msg = (f"Location set — Lat {_deg_to_dms(lat)}  Lon {_deg_to_dms(lon)}",
                           PAL["success"])
                else:
                    msg = (f"Location partially failed (lat={'OK' if lat_ok else 'FAIL'}, "
                           f"lon={'OK' if lon_ok else 'FAIL'})", PAL["danger"])
            except Exception as exc:
                msg = (f"Location error: {exc}", PAL["danger"])
            self.after(0, lambda m=msg: self.log(*m))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Plate Solve Calibration ────────────────────────────────────────────────

    def _calibrate_plate_solve(self):
        if not self._check_connected():
            return
        try:
            ra_deg  = _parse_ra_input(self._ps_ra_entry.get())
            dec_deg = _parse_dec_input(self._ps_dec_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid Input",
                "Enter RA as HH:MM:SS (or decimal degrees) "
                "and Dec as \u00b1DD:MM:SS (or decimal degrees).")
            return
        if not (-90.0 <= dec_deg <= 90.0):
            messagebox.showwarning("Invalid Input", "Declination must be between \u221290 and +90.")
            return

        ra_hms  = _deg_to_hms(ra_deg)
        dec_dms = _deg_to_dms(dec_deg)
        self.log(f"[Plate Solve Cal]  :SRA → {ra_hms}  :Sd → {dec_dms}", PAL["accent"])

        def _worker():
            try:
                r1 = self.mount.send_command(f":SRA{_encode_ra(ra_deg)}#")
                r2 = self.mount.send_command(f":Sd{_encode_dec(dec_deg)}#")
                if r1[:1] != "1" or r2[:1] != "1":
                    msg = f"Mount rejected RA/Dec (SRA→{r1}, Sd→{r2})"
                    self.after(0, lambda m=msg: self.log(m, PAL["danger"]))
                    return
                resp = self.mount.send_command(":CM#")
                ok   = resp[:1] == "1"
                msg  = ("Calibration accepted (:CM# → 1)." if ok
                        else f"Calibration failed (:CM# → {resp}).")
                col  = PAL["success"] if ok else PAL["danger"]
                self.after(0, lambda m=msg, c=col: self.log(m, c))
            except Exception as exc:
                self.after(0, lambda e=exc: self.log(f"Calibration error: {e}", PAL["danger"]))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _check_connected(self) -> bool:
        if not (self.mount and self.mount.connected):
            messagebox.showwarning("Not Connected", "Connect to the mount first.")
            return False
        return True

    def _on_close(self):
        self._stop_polling()
        for direction in list(self._nudge_active):
            try:
                self.mount.move_direction(direction, False)
            except Exception:
                pass
        if self.mount:
            self.mount.disconnect()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = MountControlApp()
    app.mainloop()
