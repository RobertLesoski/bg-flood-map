"""Pure formatting / unit / compass helpers for the UI (no imgui, no wgpu: unit-testable).

    m_to_ft(m) ft_to_m(ft) ms_to_mph(v) m2_to_ft2 km_to_mi
    bearing_deg(u, v)       compass bearing (deg clockwise from north) the water MOVES toward, u = east, v = north
    cardinal(deg, n=16)     "NNE" ...
    hour_label(t)           12.5 -> "12:30"           hour_of_24(t) -> "Hour 12 of 24"
    depth_text(d_m, ...)    "0.45 m (1.5 ft)"; tiers-data aware: severe -> ">= 1.2 m (>= 3.9 ft)"
    speed_text(v_ms)        "0.82 m/s (1.8 mph)"
    nice_length(target_m, imperial)  -> (length_m, label) for the scale bar
    metres_per_pixel(dist, fov, height_px)
    north_angle(yaw_deg)    screen rotation (deg clockwise from screen-up) of the north arrow
"""
from __future__ import annotations

import math

M_TO_FT = 3.280839895
MS_TO_MPH = 2.236936292
KM_TO_MI = 0.621371192
M2_TO_FT2 = 10.763910417
KM2_TO_MI2 = 0.386102159
M3_TO_ACFT = 1.0 / 1233.48184
M3_TO_GAL = 264.172052

COMPASS16 = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
COMPASS8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

TIER_NAMES = ("Shallow", "Moderate", "Deep", "Severe")
TIER_RANGES_M = ((0.1, 0.3), (0.3, 0.6), (0.6, 1.2), (1.2, None))
SEVERE_EDGE_M = 1.2


def m_to_ft(m: float) -> float:
    return m * M_TO_FT


def ft_to_m(ft: float) -> float:
    return ft / M_TO_FT


def ms_to_mph(v: float) -> float:
    return v * MS_TO_MPH


def mph_to_ms(v: float) -> float:
    return v / MS_TO_MPH


# ---------------------------------------------------------------------------------------------------- compass
def bearing_deg(u: float, v: float) -> float:
    """Compass bearing (0 = north, 90 = east, clockwise) of a velocity with u toward east and v toward north.
    Returns 0.0 for the zero vector."""
    if u == 0.0 and v == 0.0:
        return 0.0
    return math.degrees(math.atan2(u, v)) % 360.0


def cardinal(deg: float, n: int = 16) -> str:
    """Nearest compass point name: n = 8 or 16."""
    names = COMPASS16 if n == 16 else COMPASS8
    step = 360.0 / len(names)
    return names[int(((deg % 360.0) + step / 2.0) // step) % len(names)]


def screen_arrow(u: float, v: float, yaw_deg: float) -> tuple[float, float]:
    """Unit direction (dx, dy) in SCREEN space (y down) in which a world velocity (u east, v north) points when the
    camera looks toward compass heading `yaw_deg` (north is up on screen at yaw 0).  (0, 0) for the zero vector."""
    n = math.hypot(u, v)
    if n == 0.0:
        return (0.0, 0.0)
    a = math.radians(bearing_deg(u, v) - yaw_deg)     # angle clockwise from screen-up
    return (math.sin(a), -math.cos(a))


def north_angle(yaw_deg: float) -> float:
    """Where the north arrow points on screen, degrees clockwise from screen-up (camera heading `yaw_deg`)."""
    return (-yaw_deg) % 360.0


# ---------------------------------------------------------------------------------------------------- time
def hour_label(t: float) -> str:
    """Continuous hour -> 'HH:MM' (24 -> '24:00')."""
    t = max(0.0, float(t))
    mins = int(round(t * 60.0))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def snapshot_hour(t: float) -> int:
    """The hour of the data snapshot the statistics use (round(t), clamped 0..24)."""
    return int(min(24, max(0, math.floor(float(t) + 0.5))))


def hour_of_24(t: float) -> str:
    return f"Hour {snapshot_hour(t)} of 24"


# ---------------------------------------------------------------------------------------------------- numbers
def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"


def fmt_money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if v != v or v <= 0:
        return "-"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def fmt_area_km2(a: float, imperial: bool = False) -> str:
    if imperial:
        return f"{a * KM2_TO_MI2:.2f} mi²"
    return f"{a:.2f} km²"


def fmt_volume(m3: float) -> str:
    if m3 >= 1e6:
        return f"{m3 / 1e6:.2f} million m³ ({m3 * M3_TO_ACFT:,.0f} ac-ft)"
    return f"{m3:,.0f} m³ ({m3 * M3_TO_ACFT:,.1f} ac-ft)"


def fmt_length_km(m: float, imperial: bool = False) -> str:
    if imperial:
        return f"{m / 1000.0 * KM_TO_MI:.1f} mi"
    return f"{m / 1000.0:.1f} km"


# ---------------------------------------------------------------------------------------------------- depth / speed
def depth_text(d_m: float, *, tier: int | None = None, approx: bool = False, both: bool = True, ge: str = ">=") -> str:
    """'0.45 m (1.5 ft)'.  `approx` (depth-tier source) prefixes '~' and a severe building (tier 4) reads '>= 1.2 m'
    because that tier is open ended: the data cannot say how deep it really is."""
    if approx and tier == 4:
        s = f"{ge} {SEVERE_EDGE_M:.1f} m"
        return s + (f" ({ge} {SEVERE_EDGE_M * M_TO_FT:.1f} ft)" if both else "")
    if d_m < 0.005:
        return "dry"
    pre = "~" if approx else ""
    s = f"{pre}{d_m:.2f} m"
    return s + (f" ({pre}{d_m * M_TO_FT:.1f} ft)" if both else "")


def depth_short(d_m: float, *, tier: int | None = None, approx: bool = False, ge: str = ">=") -> str:
    if approx and tier == 4:
        return f"{ge}1.2 m"
    if d_m < 0.005:
        return "dry"
    return f"{'~' if approx else ''}{d_m:.2f} m"


def speed_text(v_ms: float, both: bool = True) -> str:
    s = f"{v_ms:.2f} m/s"
    return s + (f" ({v_ms * MS_TO_MPH:.1f} mph)" if both else "")


def elevation_text(z_m: float) -> str:
    return f"{z_m:.1f} m ({z_m * M_TO_FT:.0f} ft)"


def tier_range_text(tier: int) -> str:
    if tier < 1 or tier > 4:
        return "< 0.1 m"
    lo, hi = TIER_RANGES_M[tier - 1]
    return f">= {lo:g} m" if hi is None else f"{lo:g}-{hi:g} m"


# ---------------------------------------------------------------------------------------------------- scale bar
def metres_per_pixel(dist_m: float, fov_rad: float, height_px: float) -> float:
    """Ground metres per screen pixel at the orbit target (distance `dist_m` from the eye)."""
    return 2.0 * dist_m * math.tan(fov_rad * 0.5) / max(float(height_px), 1.0)


def _nice(x: float) -> float:
    """Largest 1-2-5 x 10^n value <= x."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x))
    f = x / 10 ** e
    for m in (5.0, 2.0, 1.0):
        if f >= m:
            return m * 10 ** e
    return 1.0 * 10 ** e


def nice_length(target_m: float, imperial: bool = False) -> tuple[float, str]:
    """A round length (<= target_m) for a scale bar and its label.  Returns (length in metres, label)."""
    if imperial:
        ft = target_m * M_TO_FT
        if ft >= 5280.0 * 0.5:
            mi = _nice(ft / 5280.0)
            return mi * 5280.0 / M_TO_FT, f"{mi:g} mi"
        f = _nice(ft)
        return f / M_TO_FT, f"{f:g} ft"
    if target_m >= 1000.0:
        km = _nice(target_m / 1000.0)
        return km * 1000.0, f"{km:g} km"
    m = _nice(target_m)
    return m, f"{m:g} m"


# ---------------------------------------------------------------------------------------------------- land use
LAND_USE_NAMES = {"R": "Residential", "C": "Commercial", "I": "Industrial", "A": "Agricultural", "V": "Vacant"}


def land_use_label(code: str) -> str:
    """PVA land-use code -> text; codes whose meaning is not documented in the data stay as 'code X'."""
    code = (code or "").strip()
    if not code:
        return "-"
    name = LAND_USE_NAMES.get(code.upper())
    if name:
        return f"{name} ({code})"
    if code[:1].islower():                         # OpenStreetMap building=* value (county buildings): 'semidetached_house'
        return code.replace("_", " ").capitalize() + " (OSM)"
    return f"code {code}"


def hour_note(app) -> str:
    """'' or a short label when the displayed hour has no solver snapshot (the grid of some storms starts at hour 2 or 3:
    earlier hours are dry, the first wet frame is the nearest available hour)."""
    try:
        hrs = list(app.storm.hours)
        h = snapshot_hour(app.t)
        if hrs and h >= 1 and h not in hrs and h < hrs[0]:
            return f"  (dry: first solver hour is {hrs[0]})"
    except Exception:
        pass
    return ""
