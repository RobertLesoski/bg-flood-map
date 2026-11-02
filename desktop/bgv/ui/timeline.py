"""Series for the bottom timeline chart (pure numpy; no imgui).

    s = chart_series(app_timeline, storm_meta)   ->  ChartSeries(hours, flooded, wet_area, max_depth, done, complete)

`app_timeline` is the stats hub's dict (may be None / partial, NaN = not computed yet); the storm's meta.json `hourly`
list gives wet area / max depth instantly (so the chart is never empty), the building counts arrive from the hub.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_HOURS = 25


@dataclass
class ChartSeries:
    hours: np.ndarray            # 0..24
    flooded: np.ndarray          # float, NaN = unknown
    wet_area: np.ndarray         # km2
    max_depth: np.ndarray
    done: int                    # hours with building counts
    complete: bool

    @property
    def has_flooded(self) -> bool:
        return bool(np.isfinite(self.flooded).any())


def _arr(v, n=N_HOURS) -> np.ndarray:
    a = np.full(n, np.nan)
    if v is None:
        return a
    v = np.asarray(v, np.float64).reshape(-1)
    a[:min(n, v.size)] = v[:n]
    return a


def meta_series(meta: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """(wet_area_km2[25], max_depth_m[25]) from meta['hourly'] (NaN where missing; hours without a file are 0)."""
    wet = np.full(N_HOURS, np.nan)
    mx = np.full(N_HOURS, np.nan)
    if not meta:
        return wet, mx
    for h in meta.get("hourly") or []:
        try:
            k = int(h["hour"])
            if 0 <= k < N_HOURS:
                wet[k] = float(h.get("wet_area_km2", np.nan))
                mx[k] = float(h.get("max_depth_m", np.nan))
        except (KeyError, TypeError, ValueError):
            continue
    have = set(int(h) for h in (meta.get("hours") or []))
    for k in range(N_HOURS):                      # hours that have no file are dry
        if k not in have and not np.isfinite(wet[k]):
            wet[k] = 0.0
            mx[k] = 0.0
    return wet, mx


def chart_series(timeline: dict | None, meta: dict | None) -> ChartSeries:
    wet_m, mx_m = meta_series(meta)
    flooded = _arr(timeline.get("flooded")) if timeline else np.full(N_HOURS, np.nan)
    wet = _arr(timeline.get("wet_area_km2")) if timeline else np.full(N_HOURS, np.nan)
    mx = _arr(timeline.get("max_depth_m")) if timeline else np.full(N_HOURS, np.nan)
    wet = np.where(np.isfinite(wet), wet, wet_m)
    mx = np.where(np.isfinite(mx), mx, mx_m)
    done = int(np.isfinite(flooded).sum())
    return ChartSeries(np.arange(N_HOURS, dtype=np.float64), flooded, wet, mx, done, done >= N_HOURS)


def value_at(arr: np.ndarray, t: float) -> float:
    """Linear interpolation of an hourly series at continuous hour t (NaN if a neighbour is unknown)."""
    t = min(max(float(t), 0.0), len(arr) - 1.0)
    i = int(t)
    j = min(i + 1, len(arr) - 1)
    f = t - i
    a, b = arr[i], arr[j]
    if not (np.isfinite(a) and (f == 0 or np.isfinite(b))):
        return float("nan")
    return float(a + (b - a) * f) if f else float(a)
