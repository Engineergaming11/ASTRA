"""
Optional USB GPS (NMEA) read for site latitude/longitude.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple


def _parse_dm_to_deg(dm: str, hemi: str, is_lat: bool) -> float:
    dm = dm.strip()
    if is_lat:
        deg = int(dm[:2])
        minutes = float(dm[2:])
    else:
        deg = int(dm[:3])
        minutes = float(dm[3:])
    v = deg + minutes / 60.0
    if hemi in ("S", "W"):
        v = -v
    return v


def try_read_lat_lon_nmea(port: str, baudrate: int = 9600, timeout_s: float = 5.0) -> Optional[Tuple[float, float]]:
    """
    Read first valid $GPRMC with status A, or $GPGGA with fix quality > 0.
    Returns (latitude_deg, longitude_deg) east-positive, north-positive.
    """
    try:
        import serial
    except ImportError:
        return None

    deadline = time.monotonic() + timeout_s
    try:
        ser = serial.Serial(port, baudrate, timeout=0.3)
    except Exception:
        return None
    try:
        while time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            tag = parts[0]
            if tag.endswith("RMC") and len(parts) >= 7 and parts[2] == "A":
                lat = _parse_dm_to_deg(parts[3], parts[4], True)
                lon = _parse_dm_to_deg(parts[5], parts[6], False)
                return lat, lon
            if tag.endswith("GGA") and len(parts) > 6 and parts[2] and parts[4]:
                try:
                    if int(parts[6]) == 0:
                        continue
                except (ValueError, IndexError):
                    continue
                lat = _parse_dm_to_deg(parts[2], parts[3], True)
                lon = _parse_dm_to_deg(parts[4], parts[5], False)
                return lat, lon
    except Exception:
        return None
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return None
