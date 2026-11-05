"""Measurement maths on the WGS84 ellipsoid (pure numpy / math: no imgui, no wgpu - unit-testable).

World metres of the viewer (bgv.geo) are Web-Mercator metres scaled by ONE cos(latitude) for the whole grid, so a county-wide
distance read from them can be off by about 0.3 %.  Measurements therefore go through longitude / latitude:

    path_length_m(lng, lat)            length of a polyline: every segment uses the ellipsoid's radii of curvature at its mid
                                       latitude (local tangent plane); error about (L / 6400 km)^2, i.e. < 0.01 % up to 60 km.
    cumulative_m(lng, lat)             running distance along the polyline (first value 0)
    polygon_area_m2(lng, lat)          EXACT ellipsoidal area of the polygon whose edges are straight in (lng, lat) (Lambert
                                       cylindrical equal-area on the authalic sphere - the same maths as EPSG:6933, which the
                                       tests use as the independent reference).
    world_to_lnglat(frame, X, Y)       vectorised bgv.geo conversions
    lnglat_to_world(frame, lng, lat)
    fmt_length(m, imperial) / fmt_area(m2, imperial)   (text, secondary text)

Units setting: "metric" shows m / km and m2 / ha / km2, "imperial" ft / mi and ft2 / acres / mi2.  Every readout also gives the
other system in a second, dimmer line.
"""
from __future__ import annotations

import math

import numpy as np

A_WGS84 = 6378137.0
E2 = 0.00669437999014133
E = math.sqrt(E2)
R_MERC = 6378137.0                    # Web Mercator sphere radius (bgv.geo)
M_PER_FT = 0.3048
M_PER_MI = 1609.344
M2_PER_ACRE = 4046.8564224
M2_PER_HA = 10000.0
M2_PER_FT2 = M_PER_FT ** 2
M2_PER_MI2 = M_PER_MI ** 2


# ---------------------------------------------------------------------------------------------------- conversions
def world_to_lnglat(frame, X, Y):
    """Vectorised `GridFrame.world_to_lnglat`: returns (lng, lat) float64 arrays in degrees."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    x = X / frame.k + frame.xc
    y = Y / frame.k + frame.yc
    lng = np.degrees(x / R_MERC)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / R_MERC)) - math.pi / 2.0)
    return lng, lat


def lnglat_to_merc(lng, lat):
    lng = np.asarray(lng, np.float64)
    lat = np.asarray(lat, np.float64)
    x = R_MERC * np.radians(lng)
    y = R_MERC * np.log(np.tan(math.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def merc_to_lnglat(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    return np.degrees(x / R_MERC), np.degrees(2.0 * np.arctan(np.exp(y / R_MERC)) - math.pi / 2.0)


def lnglat_to_world(frame, lng, lat):
    x, y = lnglat_to_merc(lng, lat)
    return (x - frame.xc) * frame.k, (y - frame.yc) * frame.k


# ---------------------------------------------------------------------------------------------------- distance
def _radii(lat_rad):
    s2 = np.sin(lat_rad) ** 2
    w = np.sqrt(1.0 - E2 * s2)
    n = A_WGS84 / w                                    # prime vertical radius
    m = A_WGS84 * (1.0 - E2) / (w ** 3)                # meridian radius
    return m, n


def segment_lengths_m(lng, lat) -> np.ndarray:
    """Ground length (m) of every segment of the polyline through (lng, lat) degrees."""
    lng = np.asarray(lng, np.float64)
    lat = np.asarray(lat, np.float64)
    if lng.size < 2:
        return np.zeros(0)
    dlng = np.radians(np.diff(lng))
    dlat = np.radians(np.diff(lat))
    latm = np.radians(0.5 * (lat[1:] + lat[:-1]))
    m, n = _radii(latm)
    return np.hypot(n * np.cos(latm) * dlng, m * dlat)


def cumulative_m(lng, lat) -> np.ndarray:
    seg = segment_lengths_m(lng, lat)
    return np.concatenate([[0.0], np.cumsum(seg)])


def path_length_m(lng, lat) -> float:
    return float(segment_lengths_m(lng, lat).sum())


# ---------------------------------------------------------------------------------------------------- area
def _q(lat_rad):
    """Authalic function q(phi) (Snyder 3-12)."""
    s = np.sin(lat_rad)
    es = E * s
    return (1.0 - E2) * (s / (1.0 - es * es) - (1.0 / (2.0 * E)) * np.log((1.0 - es) / (1.0 + es)))


_QP = float(_q(np.array(math.pi / 2.0)))
_RQ2 = A_WGS84 ** 2 * _QP / 2.0                       # squared radius of the authalic sphere


def polygon_area_m2(lng, lat) -> float:
    """Area (m2) of the polygon (open ring: the closing edge is implied) whose edges are straight in (lng, lat)."""
    lng = np.asarray(lng, np.float64)
    lat = np.asarray(lat, np.float64)
    if lng.size < 3:
        return 0.0
    lam = np.radians(lng - lng[0])                    # relative to the first vertex: no precision loss, no antimeridian issue here
    sb = _q(np.radians(lat)) / _QP                    # sin of the authalic latitude
    a = float(np.sum(lam * np.roll(sb, -1) - np.roll(lam, -1) * sb)) * 0.5
    return abs(a) * _RQ2


def polygon_perimeter_m(lng, lat) -> float:
    if np.size(lng) < 2:
        return 0.0
    lng2 = np.append(lng, lng[0])
    lat2 = np.append(lat, lat[0])
    return path_length_m(lng2, lat2)


# ---------------------------------------------------------------------------------------------------- text
def fmt_length(m: float, imperial: bool = False) -> tuple[str, str]:
    """(primary, secondary) text of a length in the chosen system; the secondary is the other system."""
    def metric(v):
        return f"{v / 1000.0:.3f} km" if v >= 1000.0 else (f"{v:.1f} m" if v >= 10.0 else f"{v:.2f} m")

    def imp(v):
        ft = v / M_PER_FT
        return f"{v / M_PER_MI:.3f} mi" if ft >= 5280.0 else (f"{ft:.0f} ft" if ft >= 100.0 else f"{ft:.1f} ft")
    a, b = (imp(m), metric(m)) if imperial else (metric(m), imp(m))
    return a, b


def fmt_area(m2: float, imperial: bool = False) -> tuple[str, str]:
    def metric(v):
        if v >= 1.0e6:
            return f"{v / 1.0e6:.3f} km²"
        if v >= 1.0e4:
            return f"{v / M2_PER_HA:.2f} ha"
        return f"{v:.0f} m²"

    def imp(v):
        if v >= M2_PER_MI2 * 0.1:
            return f"{v / M2_PER_MI2:.3f} mi²"
        if v >= M2_PER_ACRE * 0.1:
            return f"{v / M2_PER_ACRE:.2f} acres"
        return f"{v / M2_PER_FT2:.0f} ft²"
    a, b = (imp(m2), metric(m2)) if imperial else (metric(m2), imp(m2))
    return a, b
