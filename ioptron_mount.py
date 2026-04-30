"""
iOptron HAE-series mount driver and coordinate helpers (RS-232 / WiFi).
Protocol v3.10. UI-free — used by mount_controls_frame and CamGUI workflows.
"""

import datetime
import errno
import serial
import socket
import threading
import time
from typing import Optional, Tuple, Union

BAUD_RATE = 115200
TIMEOUT = 4.0
WIFI_CONNECT_TIMEOUT = 12.0

WIFI_DEFAULT_IP = "10.10.100.254"
WIFI_DEFAULT_PORT = 8899


def _wifi_oserror_hint(ip: str, port: int, err: OSError) -> str:
    """Human-readable hint for common WiFi TCP failures (esp. multi-homed macOS)."""
    no = err.errno
    base = f"Could not open TCP to {ip}:{port} ({err.strerror or err})."
    if no in (errno.ENETUNREACH, errno.EHOSTUNREACH):
        return (
            base
            + " No route to the mount — join the mount's Wi‑Fi (e.g. HBX…), confirm IP in the hand "
            "controller, and on Mac turn off Ethernet/VPN or set Wi‑Fi above Ethernet for local traffic."
        )
    if no in (errno.ETIMEDOUT, errno.ECONNABORTED):
        return (
            base
            + " Timed out — move closer to the mount, confirm Wi‑Fi/Ethernet mode on the mount, "
            "and that port 8899 matches the controller."
        )
    if no == errno.ECONNREFUSED:
        return (
            base
            + " Refused — mount may not be in Wi‑Fi command mode yet; power-cycle Wi‑Fi or use USB/serial."
        )
    return base

TRACK_SIDEREAL = "0"
TRACK_LUNAR = "1"
TRACK_SOLAR = "2"

_UNITS_PER_DEG = 360000


def _deg_to_iopval(deg: float) -> int:
    return round(deg * _UNITS_PER_DEG)


def _iopval_to_deg(val: int) -> float:
    return val / _UNITS_PER_DEG


def _encode_ra(ra_deg: float) -> str:
    v = _deg_to_iopval(ra_deg % 360.0)
    return f"{v:09d}"


def _encode_dec(dec_deg: float) -> str:
    v = _deg_to_iopval(dec_deg)
    sign = "+" if v >= 0 else "-"
    return f"{sign}{abs(v):08d}"


def _parse_gep(raw: str) -> Tuple[float, float, int]:
    raw = raw.strip()
    sign = -1 if raw.startswith("-") else 1
    digits = raw.lstrip("+-")
    dec_val = sign * int(digits[:8])
    ra_val = int(digits[8:17])
    pier = int(digits[17]) if len(digits) > 17 else 0
    ra_deg = _iopval_to_deg(ra_val) % 360.0
    dec_deg = _iopval_to_deg(dec_val)
    return ra_deg, dec_deg, pier


def _deg_to_hms(deg: float) -> str:
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
    raw = raw.strip()
    sep = ":" if ":" in raw else (" " if raw.count(" ") >= 2 else None)
    if sep:
        parts = raw.split(sep)
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return (h + m / 60.0 + s / 3600.0) * 15.0
    val = float(raw)
    return val * 15.0 if val < 24.0 else val


def _parse_angle_dms(raw: str) -> float:
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
    raw = raw.strip()
    sep = ":" if ":" in raw else (" " if raw.count(" ") >= 2 else None)
    if sep:
        parts = raw.split(sep)
        if len(parts) == 3:
            d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            sign = -1.0 if d < 0 or raw.startswith("-") else 1.0
            return sign * (abs(d) + m / 60.0 + s / 3600.0)
    return float(raw)


class IOptronHAE:
    """RS-232/USB interface for iOptron HAE-series mounts (protocol v3.10)."""

    def __init__(self, port: str):
        self.port = port
        self._ser: Optional[serial.Serial] = None

    def connect(self) -> str:
        self._ser = serial.Serial(
            port=self.port,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT,
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

    def get_firmware(self) -> str:
        return self.send_command(":FW1#")

    def set_utc_time(self) -> tuple:
        j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        offset_ms = round((now - j2000).total_seconds() * 1000)
        # :SUT expects UTC milliseconds from J2000 encoded as a 13-digit value.
        if offset_ms < 0:
            raise ValueError("J2000 UTC offset is negative; cannot encode :SUT payload.")
        cmd = f":SUT{offset_ms:013d}#"
        resp = self.send_command(cmd)
        utc_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        return resp[:1] == "1", utc_str

    def set_longitude(self, lon_deg: float) -> bool:
        v = round(lon_deg * 360000)
        sign = "+" if v >= 0 else "-"
        cmd = f":SLO{sign}{abs(v):08d}#"
        return self.send_command(cmd)[:1] == "1"

    def set_latitude(self, lat_deg: float) -> bool:
        v = round(lat_deg * 360000)
        sign = "+" if v >= 0 else "-"
        cmd = f":SLA{sign}{abs(v):08d}#"
        return self.send_command(cmd)[:1] == "1"

    def set_tracking_rate(self, rate_code: str):
        self.send_command(f":RT{rate_code}#")

    def tracking_on(self):
        self.send_command(":ST1#")

    def tracking_off(self):
        self.send_command(":ST0#")

    def set_slew_rate(self, rate: int):
        self.send_command(f":SR{max(1, min(9, rate))}#")

    def goto_radec(self, ra_deg: float, dec_deg: float) -> Tuple[bool, str]:
        r1 = self.send_command(f":SRA{_encode_ra(ra_deg)}#")
        r2 = self.send_command(f":Sd{_encode_dec(dec_deg)}#")
        if r1[:1] != "1" or r2[:1] != "1":
            return False, f"Mount rejected coordinates (SRA→{r1}, Sd→{r2})"
        resp = self.send_command(":MS1#")
        first = resp[:1] if resp else ""
        msgs = {"1": "Slewing…", "0": "Below altitude limit or mechanical limit"}
        return first == "1", msgs.get(first, f"Unknown response: {resp}")

    def sync_radec(self, ra_deg: float, dec_deg: float) -> bool:
        self.send_command(f":SRA{_encode_ra(ra_deg)}#")
        self.send_command(f":Sd{_encode_dec(dec_deg)}#")
        resp = self.send_command(":CM#")
        return resp[:1] == "1"

    def calibrate_at_radec_with_pier(
        self, ra_deg: float, dec_deg: float, pier_side: str
    ) -> Tuple[bool, str]:
        """Plate-solve style sync: SRA, Sd, SPS pier, CM."""
        r1 = self.send_command(f":SRA{_encode_ra(ra_deg)}#")
        r2 = self.send_command(f":Sd{_encode_dec(dec_deg)}#")
        if r1[:1] != "1" or r2[:1] != "1":
            return False, f"Mount rejected (SRA→{r1}, Sd→{r2})"
        self.send_command(f":SPS{pier_side}#")
        resp = self.send_command(":CM#")
        return resp[:1] == "1", resp

    def get_current_radec(self) -> Tuple[float, float, int]:
        raw = self.send_command(":GEP#")
        return _parse_gep(raw)

    def get_status(self) -> dict:
        raw = self.send_command(":GLS#")
        if len(raw) < 22:
            return {}
        return {
            "system_status": int(raw[17]),
            "tracking_rate": int(raw[18]),
            "slew_speed": int(raw[19]),
            "time_source": int(raw[20]),
            "hemisphere": int(raw[21]),
        }

    def go_home(self):
        """Slew to the user-defined zero / home position (:MH#)."""
        self.send_command(":MH#")

    def goto_zero(self):
        """Alias for :MH# — slew to the zero position last set with :SZP#."""
        self.go_home()

    def set_zero_at_current(self) -> Tuple[bool, str]:
        """Set the current mount position as the zero / home position (:SZP#)."""
        resp = self.send_command(":SZP#")
        ok = resp[:1] == "1"
        return ok, resp or "(no response)"

    def search_mechanical_zero(self) -> Tuple[bool, str]:
        """Search mechanical zero via homing sensors; may overwrite saved zero (:MSH#)."""
        resp = self.send_command(":MSH#")
        ok = resp[:1] == "1"
        return ok, resp or "(no response)"

    def unpark(self):
        self.send_command(":MP0#")

    def stop_slew(self):
        self.send_command(":Q#")

    def move_direction(self, direction: str, start: bool):
        if start:
            cmd = {
                "n": ":mn#",
                "s": ":ms#",
                "e": ":me#",
                "w": ":mw#",
            }.get(direction.lower())
            if cmd:
                self.send_no_reply(cmd)
        else:
            if direction.lower() in ("n", "s"):
                self.send_command(":qD#")
            else:
                self.send_command(":qR#")


class IOptronHAEWifi(IOptronHAE):
    """WiFi/TCP transport for iOptron HAE-series mounts."""

    def __init__(self, ip: str, port: int = WIFI_DEFAULT_PORT):
        super().__init__(port="")
        self._ip = ip
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self) -> str:
        self.disconnect()
        s: Optional[socket.socket] = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(WIFI_CONNECT_TIMEOUT)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Do not bind to a specific interface (e.g. en0): on multi-homed Macs that
            # breaks connecting to the mount WiFi (10.10.x.x) when Ethernet is en0.
            s.connect((self._ip, self._port))
            self._sock = s
            s = None
            time.sleep(0.5)
            self._sock.settimeout(1.0)
            try:
                self._sock.recv(64)
            except socket.timeout:
                pass
            self._sock.settimeout(0.5)
            info = self.send_command(":MountInfo#")
            if not info:
                raise ConnectionError(
                    "No response from mount over WiFi (:MountInfo#). "
                    "Check IP/port on the hand controller and that the mount finished booting."
                )
            return info
        except OSError as e:
            self.disconnect()
            raise ConnectionError(_wifi_oserror_hint(self._ip, self._port, e)) from e
        except Exception:
            self.disconnect()
            raise
        finally:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

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

    def _drain(self):
        self._sock.settimeout(0.0)
        try:
            while self._sock.recv(512):
                pass
        except (socket.timeout, BlockingIOError):
            pass
        finally:
            self._sock.settimeout(0.5)

    def send_command(self, cmd: str) -> str:
        if not self.connected:
            raise RuntimeError("Not connected.")
        with self._lock:
            self._drain()
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
                    self._sock.settimeout(0.15)
                    try:
                        while True:
                            extra = self._sock.recv(256)
                            if not extra:
                                break
                            buf += extra
                    except socket.timeout:
                        pass
                    finally:
                        self._sock.settimeout(0.5)
                    break
            raw = buf.decode("ascii", errors="replace")
            first_token = raw.split("#")[0]
            return first_token.strip()

    def send_no_reply(self, cmd: str):
        if not self.connected:
            raise RuntimeError("Not connected.")
        with self._lock:
            self._drain()
            self._sock.sendall(cmd.encode("ascii"))
            time.sleep(0.05)


IOptronMount = Union[IOptronHAE, IOptronHAEWifi]

__all__ = [
    "BAUD_RATE",
    "TIMEOUT",
    "WIFI_DEFAULT_IP",
    "WIFI_DEFAULT_PORT",
    "TRACK_SIDEREAL",
    "TRACK_LUNAR",
    "TRACK_SOLAR",
    "IOptronHAE",
    "IOptronHAEWifi",
    "IOptronMount",
    "_deg_to_hms",
    "_deg_to_dms",
    "_parse_ra_input",
    "_parse_dec_input",
    "_parse_angle_dms",
    "_encode_ra",
    "_encode_dec",
]
