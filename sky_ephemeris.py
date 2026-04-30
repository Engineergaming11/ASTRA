"""
Skyfield ephemeris, catalogs, and LiveWCS helpers for ASTRA.
All resource paths are resolved relative to this file's directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from skyfield.api import Loader, Star, load, load_file, wgs84

_BASE = Path(__file__).resolve().parent

# Default site (University of Arizona Steward) — used when no GPS/session site.
DEFAULT_LAT = 32.2329
DEFAULT_LON = -110.9479
DEFAULT_ELEV_M = 728.0

catalog = pd.read_parquet(_BASE / "updatedTycho2.parquet")
messier_catalog = pd.read_parquet(_BASE / "Messier_Updated.parquet")
ts = load.timescale()
_eph_path = _BASE / "de440.bsp"
# `load(full_path)` is not supported — Skyfield treats it as a basename and builds a bad URL.
if _eph_path.is_file():
    eph = load_file(str(_eph_path))
else:
    eph = Loader(str(_BASE))("de440.bsp")

TYCHO_PRESET_STARS = {
    "Avior": "8579 02692 1",
    "Miaplacidus": "9200 02603 1",
    "Regulus": "0833 01381 1",
    "Rigel": "5331 01752 1",
    "Alioth": "3845 01190 1",
    "Alkaid": "3467 01257 1",
}

SOLAR_SYSTEM_BODIES = {
    "Sun": "sun",
    "Moon": "moon",
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars barycenter",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter",
    "Neptune": "neptune barycenter",
}

NEBULAE: Dict[str, str] = {}

DESCRIPTIONS: Dict[str, str] = {
    "Avior": "Avior (Epsilon Carinae) is a binary star system in the constellation Carina.",
    "Miaplacidus": "Miaplacidus (Beta Carinae) is the second-brightest star in Carina.",
    "Regulus": "Regulus (Alpha Leonis) is the brightest star in Leo.",
    "Rigel": "Rigel (Beta Orionis) is a blue supergiant star in Orion.",
    "Alioth": "Alioth (Epsilon Ursae Majoris) is the brightest star in Ursa Major.",
    "Alkaid": "Alkaid (Eta Ursae Majoris) forms the end of the Big Dipper's handle.",
    "Sun": "The Sun is the star at the center of our Solar System.",
    "Moon": "Earth's Moon is the brightest object in our night sky after the Sun.",
    "Mercury": "Mercury is the smallest planet and closest to the Sun.",
    "Venus": "Venus is Earth's closest planetary neighbor.",
    "Mars": "Mars, the Red Planet, is the fourth planet from the Sun.",
    "Jupiter": "Jupiter is the largest planet in our Solar System.",
    "Saturn": "Saturn is famous for its spectacular ring system.",
    "Uranus": "Uranus is an ice giant that rotates on its side.",
    "Neptune": "Neptune is the farthest planet from the Sun.",
    "Orion Nebula": "M42, the Orion Nebula, is one of the brightest nebulae visible.",
    "Eagle Nebula": "M16, the Eagle Nebula, is famous for the 'Pillars of Creation'.",
    "Crab Nebula": "M1, the Crab Nebula, is a supernova remnant in Taurus.",
    "Ring Nebula": "M57, the Ring Nebula, is a planetary nebula in Lyra.",
    "Dumbbell Nebula": "M27, the Dumbbell Nebula, is a planetary nebula in Vulpecula.",
    "Helix Nebula": "NGC 7293, the Helix Nebula, is one of the closest planetary nebulae.",
}


def load_messier_objects() -> None:
    global NEBULAE
    try:
        for _, obj in messier_catalog.iterrows():
            messier_id = obj["MESSIER_ID"]
            obj_type = obj.get("Type", "") if "Type" in messier_catalog.columns else ""
            display_name = f"{messier_id} - {obj_type}" if obj_type else messier_id
            NEBULAE[display_name] = messier_id
            desc = (
                f"{messier_id} is a {obj_type} in the Messier catalog."
                if obj_type
                else f"{messier_id} is in the Messier catalog."
            )
            if "Description" in messier_catalog.columns and pd.notna(obj.get("Description")):
                desc = obj["Description"]
            elif "Distance_ly" in messier_catalog.columns and pd.notna(obj.get("Distance_ly")):
                distance = obj["Distance_ly"]
                desc = (
                    f"{messier_id} is a {obj_type} located approximately {distance:,.0f} light-years away."
                    if obj_type
                    else f"{messier_id} is located approximately {distance:,.0f} light-years away."
                )
            DESCRIPTIONS[display_name] = desc
    except Exception as e:
        print(f"Warning: Could not load Messier catalog: {e}")
        NEBULAE.update(
            {
                "Orion Nebula": "M42",
                "Eagle Nebula": "M16",
                "Crab Nebula": "M1",
                "Ring Nebula": "M57",
                "Dumbbell Nebula": "M27",
            }
        )


def get_messier_coordinates(messier_id: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        match = messier_catalog[messier_catalog["MESSIER_ID"] == messier_id]
        if match.empty:
            return None, None
        obj = match.iloc[0]
        return float(obj["RA_deg"]), float(obj["Dec_deg"])
    except Exception as e:
        print(f"Error getting coordinates for {messier_id}: {e}")
        return None, None


load_messier_objects()


def observer_from_wgs84(lat_deg: float, lon_deg: float, elevation_m: float = 0.0):
    """Build a Skyfield WGS84 geographic position (add to eph['earth'] for topocentric)."""
    return wgs84.latlon(lat_deg, lon_deg, elevation_m=elevation_m)


def default_observer():
    return observer_from_wgs84(DEFAULT_LAT, DEFAULT_LON, DEFAULT_ELEV_M)


def sun_apparent_radec_deg(
    lat_deg: float, lon_deg: float, elev_m: float = 0.0, t=None
) -> Tuple[float, float]:
    """Apparent geocentric-of-date RA/Dec of the Sun in degrees at time t (default now)."""
    t = t or ts.now()
    obs = eph["earth"] + observer_from_wgs84(lat_deg, lon_deg, elev_m)
    astrometric = obs.at(t).observe(eph["sun"]).apparent()
    ra, dec, _ = astrometric.radec("date")
    return ra.hours * 15.0, dec.degrees


def sun_bottom_limb_radec_deg(
    lat_deg: float,
    lon_deg: float,
    elev_m: float = 0.0,
    t=None,
    solar_radius_deg: float = 0.2666,
) -> Tuple[float, float]:
    """
    Apparent RA/Dec (degrees) of the Sun's lower limb (bottom edge) at time t.

    The point is derived in local Alt/Az space (altitude minus solar radius) and
    converted back to apparent RA/Dec for mount sync workflows.
    """
    t = t or ts.now()
    obs = eph["earth"] + observer_from_wgs84(lat_deg, lon_deg, elev_m)
    sun_ast = obs.at(t).observe(eph["sun"]).apparent()
    alt, az, _ = sun_ast.altaz()
    limb_alt = alt.degrees - float(solar_radius_deg)
    limb_ast = obs.at(t).from_altaz(alt_degrees=limb_alt, az_degrees=az.degrees)
    ra, dec, _ = limb_ast.radec("date")
    return ra.hours * 15.0, dec.degrees


def body_apparent_radec_deg(
    body_key: str,
    lat_deg: float,
    lon_deg: float,
    elev_m: float = 0.0,
    t=None,
) -> Tuple[float, float]:
    t = t or ts.now()
    obs = eph["earth"] + observer_from_wgs84(lat_deg, lon_deg, elev_m)
    astrometric = obs.at(t).observe(eph[body_key]).apparent()
    ra, dec, _ = astrometric.radec("date")
    return ra.hours * 15.0, dec.degrees


def radec_to_altaz_deg(
    ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float, elev_m: float = 0.0, t=None
) -> Tuple[float, float]:
    """Topocentric altitude and azimuth (degrees) for J2000-like input via apparent chain."""
    t = t or ts.now()
    obs = eph["earth"] + observer_from_wgs84(lat_deg, lon_deg, elev_m)
    # Build a Star at this RA/Dec for this epoch (approximation for display)
    star = Star(ra_hours=ra_deg / 15.0, dec_degrees=dec_deg)
    astrometric = obs.at(t).observe(star).apparent()
    alt, az, _ = astrometric.altaz()
    return alt.degrees, az.degrees


def is_above_horizon(
    lat_deg: float,
    lon_deg: float,
    elev_m: float = 0.0,
    *,
    target_star=None,
    body_key: Optional[str] = None,
    t=None,
) -> Tuple[bool, float, float]:
    t = t or ts.now()
    observer = eph["earth"] + observer_from_wgs84(lat_deg, lon_deg, elev_m)
    if target_star is not None:
        astrometric = observer.at(t).observe(target_star).apparent()
    elif body_key is not None:
        astrometric = observer.at(t).observe(eph[body_key]).apparent()
    else:
        raise ValueError("Provide either target_star or body_key")
    alt, az, _ = astrometric.altaz()
    return alt.degrees > 0, alt.degrees, az.degrees


def angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """Great-circle separation in degrees."""
    r1, d1 = np.radians(ra1_deg), np.radians(dec1_deg)
    r2, d2 = np.radians(ra2_deg), np.radians(dec2_deg)
    cos_sep = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sep)))


def major_bodies_separation_table(
    field_ra_deg: float,
    field_dec_deg: float,
    lat_deg: float,
    lon_deg: float,
    elev_m: float = 0.0,
    t=None,
) -> List[Tuple[str, float, bool, float, float]]:
    """
    For each major solar-system body: name, separation from field center (deg),
    above_horizon, alt_deg, az_deg.
    """
    rows = []
    t = t or ts.now()
    for name, key in SOLAR_SYSTEM_BODIES.items():
        try:
            ra_b, dec_b = body_apparent_radec_deg(key, lat_deg, lon_deg, elev_m, t)
            sep = angular_separation_deg(field_ra_deg, field_dec_deg, ra_b, dec_b)
            up, alt, az = is_above_horizon(lat_deg, lon_deg, elev_m, body_key=key, t=t)
            rows.append((name, sep, up, alt, az))
        except Exception:
            continue
    rows.sort(key=lambda x: x[1])
    return rows


def polar_align_mvp_text(
    lat_deg: float,
    lon_deg: float,
    solved_ra_deg: float,
    solved_dec_deg: float,
    elev_m: float = 0.0,
    t=None,
) -> str:
    """
    MVP polar-align hints: true pole altitude, Polaris az/alt, and offset from
    solved boresight to north celestial pole direction (approximate).
    """
    t = t or ts.now()
    obs = eph["earth"] + observer_from_wgs84(lat_deg, lon_deg, elev_m)
    pole = Star(ra_hours=0.0, dec_degrees=90.0)
    ast_pole = obs.at(t).observe(pole).apparent()
    alt_p, az_p, _ = ast_pole.altaz()

    polaris = eph["Polaris"]
    ast_pol = obs.at(t).observe(polaris).apparent()
    alt_pol, az_pol, _ = ast_pol.altaz()

    sep_pole = angular_separation_deg(solved_ra_deg, solved_dec_deg, 0.0, 90.0)

    hem = "north" if lat_deg >= 0 else "south"
    lines = [
        "Polar alignment (MVP hints)",
        f"Site latitude: {lat_deg:.4f}° ({hem}ern hemisphere). Approx celestial pole altitude: {abs(lat_deg):.1f}° above your {hem}ern horizon.",
        f"North celestial pole (apparent): ALT {alt_p.degrees:.2f}°, AZ {az_p.degrees:.2f}°.",
        f"Polaris (apparent): ALT {alt_pol.degrees:.2f}°, AZ {az_pol.degrees:.2f}°.",
        f"Solved field center: RA {solved_ra_deg:.4f}°, Dec {solved_dec_deg:.4f}°.",
        f"Angular distance from field center to celestial pole (~90° Dec): {sep_pole:.2f}°.",
        "Adjust mount altitude/azimuth knobs to bring the pole into alignment; repeat plate solve after each adjustment.",
    ]
    return "\n".join(lines)


def write_livewcs(path: str | Path, ra, dec, object_name: str = "Unknown") -> None:
    """Write LiveWCS JSON atomically (ra, dec are Skyfield Angle objects with .hstr() / .dstr())."""
    ra_deg = ra.hours * 15.0
    dec_deg = dec.degrees
    dec_wcs = dec.dstr().rstrip('"')
    data: Dict[str, Any] = {
        "object": object_name,
        "ra_WCS": ra.hstr(),
        "dec_WCS": dec_wcs,
        "ra_deg": round(ra_deg, 6),
        "dec_deg": round(dec_deg, 6),
    }
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def livewcs_path(default_name: str = "LiveWCS.txt") -> Path:
    return _BASE / default_name
