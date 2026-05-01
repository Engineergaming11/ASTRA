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

from offline_star_facts import OFFLINE_STAR_FACTS

_BASE = Path(__file__).resolve().parent

# Default site (University of Arizona Steward) — used when no GPS/session site.
DEFAULT_LAT = 32.2329
DEFAULT_LON = -110.9479
DEFAULT_ELEV_M = 728.0

catalog = pd.read_parquet(_BASE / "updatedTycho2.parquet")
messier_catalog = pd.read_parquet(_BASE / "Messier_Updated.parquet")

# Lazy-built unit vectors for fast nearest Tycho match (plate-solve annotations).
_TYC_ANN_U32: Optional[np.ndarray] = None
_TYC_ANN_TYC: Optional[np.ndarray] = None
_TYC_ANN_VT: Optional[np.ndarray] = None
_TYC_ANN_BT: Optional[np.ndarray] = None

ts = load.timescale()
_eph_path = _BASE / "de440.bsp"
# `load(full_path)` is not supported — Skyfield treats it as a basename and builds a bad URL.
if _eph_path.is_file():
    eph = load_file(str(_eph_path))
else:
    eph = Loader(str(_BASE))("de440.bsp")

TYCHO_PRESET_STARS = {
    "Achernar": "8478 01395 1",
    "Acrux": "8979 03464 1",
    "Adhara": "6535 03619 1",
    "Aldebaran": "1266 01416 1",
    "Alnair": "8438 01959 1",
    "Alnilam": "4766 02450 1",
    "Alnitak": "4771 01188 1",
    "Alioth": "3845 01190 1",
    "Altair": "1058 03399 1",
    "Alkaid": "3467 01257 1",
    "Alphard": "5460 01592 1",
    "Antares": "6803 02158 1",
    "Arcturus": "1472 01436 2",
    "Avior": "8579 02692 1",
    "Bellatrix": "0113 01856 1",
    "Betelgeuse": "0129 01873 1",
    "Canopus": "8534 02277 1",
    "Capella": "3358 03141 1",
    "Castor": "2457 02407 1",
    "Deneb": "3574 03347 1",
    "Elnath": "1859 01470 1",
    "Fomalhaut": "6977 01267 1",
    "Gacrux": "8654 03422 1",
    "Hadar": "9005 03919 1",
    "Miaplacidus": "9200 02603 1",
    "Pollux": "1920 02194 1",
    "Procyon": "0187 02184 1",
    "Regulus": "0833 01381 1",
    "Rigel": "5331 01752 1",
    "Shaula": "7388 01093 1",
    "Sirius": "5949 02777 1",
    "Spica": "5547 01518 1",
    "Vega": "3105 02070 1",
}


def _norm_tyc_id(tid: object) -> str:
    return " ".join(str(tid).strip().split())


TYCHO_NAME_BY_ID: Dict[str, str] = {
    _norm_tyc_id(tid): name for name, tid in TYCHO_PRESET_STARS.items()
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
    "Achernar": "Achernar (Alpha Eridani) is a hot, rapidly rotating star at the mouth of Eridanus.",
    "Acrux": "Acrux (Alpha Crucis) is the brightest star in the Southern Cross.",
    "Adhara": "Adhara (Epsilon Canis Majoris) is a brilliant blue supergiant in Canis Major.",
    "Aldebaran": "Aldebaran (Alpha Tauri) is an orange giant marking the eye of Taurus.",
    "Alkaid": "Alkaid (Eta Ursae Majoris) forms the end of the Big Dipper's handle.",
    "Alnair": "Alnair (Alpha Gruis) is the brightest star in Grus.",
    "Alnilam": "Alnilam (Epsilon Orionis) is the central star in Orion's Belt.",
    "Alnitak": "Alnitak (Zeta Orionis) is the eastern belt star in Orion.",
    "Alioth": "Alioth (Epsilon Ursae Majoris) is the brightest star in Ursa Major.",
    "Altair": "Altair (Alpha Aquilae) is a rapid rotator and one of the Summer Triangle vertices.",
    "Alphard": "Alphard (Alpha Hydrae) is an orange giant in the long constellation Hydra.",
    "Antares": "Antares (Alpha Scorpii) is a red supergiant paired with a hot companion.",
    "Arcturus": "Arcturus (Alpha Bootis) is a nearby orange giant and a navigation landmark.",
    "Avior": "Avior (Epsilon Carinae) is a binary star system in the constellation Carina.",
    "Bellatrix": "Bellatrix (Gamma Orionis) is a hot B giant in Orion's shoulder.",
    "Betelgeuse": "Betelgeuse (Alpha Orionis) is a red supergiant nearing the end of its life.",
    "Canopus": "Canopus (Alpha Carinae) is the second-brightest night-sky star after Sirius.",
    "Capella": "Capella (Alpha Aurigae) is a bright quadruple star in Auriga.",
    "Castor": "Castor (Alpha Geminorum) is a famous multiple-star system in Gemini.",
    "Deneb": "Deneb (Alpha Cygni) is a luminous supergiant at the tail of Cygnus.",
    "Elnath": "Elnath (Beta Tauri) is a B giant on the border of Taurus and Auriga.",
    "Fomalhaut": "Fomalhaut (Alpha Piscis Austrini) has a well-known debris disk.",
    "Gacrux": "Gacrux (Gamma Crucis) is a red giant at the top of the Southern Cross.",
    "Hadar": "Hadar (Beta Centauri) is a binary system and one of the brightest stars in the sky.",
    "Miaplacidus": "Miaplacidus (Beta Carinae) is the second-brightest star in Carina.",
    "Pollux": "Pollux (Beta Geminorum) is an orange giant with a known exoplanet.",
    "Procyon": "Procyon (Alpha Canis Minoris) is a nearby binary with a white dwarf.",
    "Regulus": "Regulus (Alpha Leonis) is the brightest star in Leo.",
    "Rigel": "Rigel (Beta Orionis) is a blue supergiant star in Orion.",
    "Shaula": "Shaula (Lambda Scorpii) is a hot binary in the tail of Scorpius.",
    "Sirius": "Sirius (Alpha Canis Majoris) is the brightest star in the night sky.",
    "Spica": "Spica (Alpha Virginis) is a massive close binary in Virgo.",
    "Vega": "Vega (Alpha Lyrae) is a bright A star and a calibration anchor for photometry.",
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


def _ensure_tycho_ann_index() -> None:
    """Build float32 unit-sphere vectors for Tycho rows with valid coordinates (lazy, once)."""
    global _TYC_ANN_U32, _TYC_ANN_TYC, _TYC_ANN_VT, _TYC_ANN_BT
    if _TYC_ANN_U32 is not None:
        return
    ra_deg = catalog["RA_deg"].to_numpy(dtype=np.float64, copy=False)
    dec_deg = catalog["Dec_deg"].to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    ra_deg = ra_deg[ok]
    dec_deg = dec_deg[ok]
    _TYC_ANN_TYC = catalog["TYC_ID"].to_numpy(dtype=object)[ok]
    _TYC_ANN_VT = catalog["VT_mag"].to_numpy(dtype=np.float64, copy=False)[ok]
    _TYC_ANN_BT = catalog["BT_mag"].to_numpy(dtype=np.float64, copy=False)[ok]
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd = np.cos(dec)
    u = np.stack((cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)), axis=1).astype(np.float32, copy=False)
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    _TYC_ANN_U32 = (u / norms).astype(np.float32, copy=False)


def tycho_nearest_identification(
    ra_deg: float,
    dec_deg: float,
    *,
    max_sep_deg: float = 0.11,
) -> Optional[Tuple[str, str, str]]:
    """
    Nearest Tycho-2 catalog entry to (RA, Dec) in degrees (J2000-like catalog frame).

    Returns (display_name, fact, catalog_source) or None if no star within ``max_sep_deg``.
    Tight default separation avoids spurious IDs where the bundled Tycho subset is sparse
    (e.g. near the celestial poles).
    """
    _ensure_tycho_ann_index()
    assert _TYC_ANN_U32 is not None and _TYC_ANN_TYC is not None
    r1 = np.radians(float(ra_deg))
    d1 = np.radians(float(dec_deg))
    cd1 = np.cos(d1)
    q = np.array([cd1 * np.cos(r1), cd1 * np.sin(r1), np.sin(d1)], dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn <= 0:
        return None
    q /= qn
    cosv = (_TYC_ANN_U32 @ q).astype(np.float64, copy=False)
    cosv = np.clip(cosv, -1.0, 1.0)
    sep_rad = np.arccos(cosv)
    sep_deg = np.degrees(sep_rad)
    j = int(np.argmin(sep_deg))
    best_sep = float(sep_deg[j])
    if best_sep > float(max_sep_deg) or not np.isfinite(best_sep):
        return None
    tid_key = _norm_tyc_id(_TYC_ANN_TYC[j])
    preset_name = TYCHO_NAME_BY_ID.get(tid_key)
    vt = float(_TYC_ANN_VT[j])
    bt = float(_TYC_ANN_BT[j])
    mag_bits: List[str] = []
    if np.isfinite(vt) and 0.0 < vt < 25.0:
        mag_bits.append(f"V_T ≈ {vt:.2f}")
    if np.isfinite(bt) and 0.0 < bt < 25.0:
        mag_bits.append(f"B_T ≈ {bt:.2f}")
    mag_txt = ", ".join(mag_bits) if mag_bits else "Tycho magnitude fields unavailable for this row"

    if preset_name:
        display = preset_name
        fact = DESCRIPTIONS.get(
            preset_name,
            f"{preset_name} is a Tycho-2 reference star in ASTRA ({mag_txt}).",
        )
        src = "Tycho-2 (named preset)"
    else:
        display = f"TYC {tid_key.replace(' ', '-')}"
        fact = (
            f"Nearest Tycho-2 catalog star to the detection ({mag_txt}). "
            f"Great-circle separation ≈ {best_sep * 3600.0:.0f} arcsec."
        )
        src = "Tycho-2"
    return display, fact, src


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

    # Polaris is a fixed star, not a SPICE solar-system body key.
    polaris = Star(ra_hours=37.95456067 / 15.0, dec_degrees=89.26410897)
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
