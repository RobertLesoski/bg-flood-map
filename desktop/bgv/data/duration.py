"""Depth-duration: how many hours a point / building stays at or above 10, 30 and 60 cm, per storm, from the hourly solver tiles;
and the whole-domain 'hours flooded' raster (hours >= 10 cm) of one storm, resampled onto the flow-grid frame.

Hours are counted from the hourly snapshots (hour files 1..24, each standing for one hour), so a value is good to +-1 h: water
that comes and goes between two snapshots is not seen.  Source label: DURATION_NOTE.
"""
from __future__ import annotations

import threading
import zlib
from pathlib import Path

import numpy as np

THRESHOLDS_CM = (10, 30, 60)
DURATION_NOTE = "counted from hourly solver-tile snapshots (+-1 h); model not validated"
HEAT_VERSION = "hoursflooded1"


def _hours(ts):
    return [h for h in sorted(ts.hours) if h >= 1]          # hour 0 is the start state (15 s), not an hour of flooding


def series_point(svc, storm: str, lng: float, lat: float) -> np.ndarray | None:
    """float64[25] depth (m) at hours 0..24 (NaN = no snapshot / no data), or None without tiles."""
    ts = svc.tileset(storm)
    if ts is None:
        return None
    out = np.full(25, np.nan)
    for h in ts.hours:
        if 0 <= h <= 24:
            r = svc.depth_lnglat(storm, lng, lat, h)
            if r.m is not None:
                out[h] = r.m
    return out


def series_building(svc, storm: str, bid: int) -> np.ndarray | None:
    """Footprint-maximum depth (m) at hours 0..24 for one building (needs the tile footprint index), else None."""
    from . import cog as cogmod
    ts = svc.tileset(storm)
    idx = svc.index_for(storm) if ts is not None else None
    if idx is None or not (0 <= bid < idx.n):
        return None
    pix = idx.pixels_of(int(bid))
    out = np.full(25, np.nan)
    if pix.size == 0:
        return out
    for h in ts.hours:
        c = ts.cog(h)
        if c is None or not 0 <= h <= 24:
            continue
        raw, st = c.sample_pixels(pix % idx.width, pix // idx.width)
        good = st == cogmod.ST_VALUE
        if good.any():
            out[h] = float(raw[good].max()) / 100.0
    return out


def hours_above(series_m: np.ndarray, thresholds_cm=THRESHOLDS_CM) -> dict:
    s = np.asarray(series_m, float)[1:]
    return {t: int(np.sum(np.nan_to_num(s, nan=0.0) >= t / 100.0)) for t in thresholds_cm}


def table(svc, storms, *, lnglat=None, bid=None) -> list[tuple[str, dict | None]]:
    """[(storm, {10: h, 30: h, 60: h} | None)] for a point (lnglat) or a building (bid)."""
    rows = []
    for s in storms:
        ser = series_building(svc, s, bid) if bid is not None else series_point(svc, s, *lnglat)
        rows.append((s, None if ser is None else hours_above(ser)))
    return rows


# ------------------------------------------------------------------------------------------------ whole-domain heat layer
def hours_flooded_grid(svc, storm: str, frame, threshold_cm: int = 10, progress=None) -> np.ndarray | None:
    """uint8[H, W] on the flow-grid frame: number of hourly snapshots (1..24) with depth >= threshold at each cell centre;
    255 = no data (outside the modelled area)."""
    from . import cog as cogmod
    ts = svc.tileset(storm)
    if ts is None:
        return None
    H, W = int(frame.height), int(frame.width)
    jj, ii = np.mgrid[0:H, 0:W]
    mx = frame.xmin + (ii.ravel() + 0.5) * (frame.xmax - frame.xmin) / W        # cell centres, web mercator (row 0 = north)
    my = frame.ymax - (jj.ravel() + 0.5) * (frame.ymax - frame.ymin) / H
    cnt = np.zeros(H * W, np.uint8)
    seen = np.zeros(H * W, bool)
    hs = _hours(ts)
    for k, h in enumerate(hs):
        c = ts.cog(h)
        if c is None:
            continue
        raw, st = c.sample_many(mx, my)
        ok = st == cogmod.ST_VALUE
        seen |= ok
        cnt += (ok & (raw >= threshold_cm)).astype(np.uint8)
        if progress is not None:
            progress((k + 1) / len(hs))
    cnt[~seen] = 255
    return cnt.reshape(H, W)


def cached_hours_flooded(svc, storm: str, frame, cache_dir, threshold_cm: int = 10, progress=None):
    """As hours_flooded_grid, cached on disk (npz keyed by storm, tile signature and frame)."""
    ts = svc.tileset(storm)
    if ts is None:
        return None
    sig = ts.grid_signature()
    key = f"{HEAT_VERSION}_{ts.scope}_{storm}_{threshold_cm}_{zlib.crc32(repr((sig, frame.width, frame.height, round(frame.xmin, 2), round(frame.ymax, 2))).encode())}"
    p = Path(cache_dir) / f"{key}.npz"
    if p.is_file():
        try:
            return np.load(p)["h"]
        except Exception:
            pass
    g = hours_flooded_grid(svc, storm, frame, threshold_cm, progress)
    if g is not None:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, h=g)
        except OSError:
            pass
    return g


class Background:
    """One background job at a time with a result slot (used by the UI)."""

    def __init__(self):
        self.key = None
        self.result = None
        self.error = ""
        self.progress = 0.0
        self.busy = False

    def start(self, key, fn):
        if self.busy or self.key == key:
            return
        self.key, self.result, self.error, self.progress, self.busy = key, None, "", 0.0, True

        def run():
            try:
                r = fn(lambda f: setattr(self, "progress", float(f)))
                if self.key == key:
                    self.result = r
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.busy = False
        threading.Thread(target=run, name="bgv-features", daemon=True).start()
