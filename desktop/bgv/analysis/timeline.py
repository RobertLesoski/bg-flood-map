"""Domain-wide flood timeline: flooded buildings, wet road length and flooded area against the hour of the selected storm.

    tl = grid_timeline(hub, storm_name)                       # from the stats hub's snapshots (flow grid, exactly the sidebar numbers)
    tl = compute_tile_timeline(tile_storm, buildings, roads, frame, progress=cb, cancel=ev)   # from the COG depth tiles, full resolution
    w = TimelineWorker(app_like)                              # background, cached in memory and (tiles) on disk
    tl = w.get(storm)                                         # FloodTimeline, `done` says which hours are in

Rules (the same thresholds as the statistics in data/stats.py):
    flooded building   depth >= 0.1 m;  flow grid: max over the all-touched pixels;  tiles: max over the footprint corner vertices
                       and the centroid, each read at its 6.25 m tile pixel (see series.py);
    wet road length    sum of the road samples (about 10 m of road each, data/roads.py) whose depth >= 0.1 m;
    flooded area       cells with depth >= 0.02 m times the cell's ground area (tiles: (6.25 m x k(row))^2, with the same k as the flow
                       grid's cell_ground_m / cell_m, corrected by cos(latitude) per row; flow grid: cell_ground_m^2);
    volume             sum of depth x cell area.
Hours are the hourly snapshots 0..24; tile hours without a file are dry.  `label` says which data the numbers come from.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..data import cachelib
from . import geodesy
from .series import N_HOURS

VERSION = "tl1"
WET_M = 0.02
FLOOD_M = 0.1
TIER_EDGES = (0.1, 0.3, 0.6, 1.2)


@dataclass
class FloodTimeline:
    storm: str
    source: str                       # 'grid' | 'tiles'
    label: str
    flooded: np.ndarray = field(default_factory=lambda: np.full(N_HOURS, np.nan))
    tiers: np.ndarray = field(default_factory=lambda: np.full((N_HOURS, 4), np.nan))
    road_km: np.ndarray = field(default_factory=lambda: np.full(N_HOURS, np.nan))
    area_km2: np.ndarray = field(default_factory=lambda: np.full(N_HOURS, np.nan))
    max_depth_m: np.ndarray = field(default_factory=lambda: np.full(N_HOURS, np.nan))
    volume_m3: np.ndarray = field(default_factory=lambda: np.full(N_HOURS, np.nan))
    n_buildings: int = 0
    road_total_km: float = 0.0

    @property
    def done(self) -> int:
        return int(np.isfinite(self.flooded).sum())

    @property
    def complete(self) -> bool:
        return self.done >= N_HOURS

    def copy(self) -> "FloodTimeline":
        return FloodTimeline(self.storm, self.source, self.label, self.flooded.copy(), self.tiers.copy(), self.road_km.copy(),
                             self.area_km2.copy(), self.max_depth_m.copy(), self.volume_m3.copy(), self.n_buildings,
                             self.road_total_km)

    def peak_hour(self, which: str = "flooded") -> int:
        a = getattr(self, which)
        return int(np.nanargmax(a)) if np.isfinite(a).any() else -1


# ---------------------------------------------------------------------------------------------------- flow grid (hub)
def grid_timeline(hub, storm: str, label: str = "") -> FloodTimeline:
    tl = FloodTimeline(storm, "grid", label or "flow grid")
    if hub is None:
        return tl
    for h in range(N_HOURS):
        s = hub.get_snapshot(storm, h)
        if s is None:
            continue
        tl.flooded[h] = float(s.buildings.flooded_count)
        tl.tiers[h] = np.asarray(s.buildings.tier_counts, np.float64)
        tl.road_km[h] = float(s.roads.flooded_length_m) / 1000.0
        tl.area_km2[h] = float(s.grid.wet_area_km2)
        tl.max_depth_m[h] = float(s.grid.max_depth_m)
        tl.volume_m3[h] = float(s.grid.volume_m3)
        tl.road_total_km = float(s.roads.total_length_m) / 1000.0
        tl.n_buildings = int(len(s.buildings.depth_m))
    return tl


# ---------------------------------------------------------------------------------------------------- tiles
def building_points(buildings):
    """(X, Y, building id) world samples of every footprint: de-duplicated mesh corner vertices + the centroid."""
    m = buildings.mesh
    bid = np.asarray(m.building_id, np.int64)
    xy = np.round(np.asarray(m.positions[:, :2], np.float64) * 100.0).astype(np.int64)
    key = np.column_stack([bid, xy])
    key = np.unique(key, axis=0)
    X = key[:, 1] / 100.0
    Y = key[:, 2] / 100.0
    ids = key[:, 0]
    fin = np.isfinite(buildings.cx) & np.isfinite(buildings.cy)
    cid = np.flatnonzero(fin)
    return (np.concatenate([X, np.asarray(buildings.cx, np.float64)[cid]]),
            np.concatenate([Y, np.asarray(buildings.cy, np.float64)[cid]]),
            np.concatenate([ids, cid]))


def _tile_pixels(raster, frame, X, Y):
    """World metres -> (row, col) integer pixels of a EPSG:3857 tile raster (-1 outside)."""
    i = raster.info()
    x = X / frame.k + frame.xc
    y = Y / frame.k + frame.yc
    c = np.floor((x - i["left"]) / i["resx"]).astype(np.int64)
    r = np.floor((i["top"] - y) / i["resy"]).astype(np.int64)
    ok = (c >= 0) & (c < i["width"]) & (r >= 0) & (r < i["height"])
    return np.where(ok, r, -1), np.where(ok, c, -1)


def row_cell_area_m2(raster, frame, r0: int, r1: int) -> np.ndarray:
    """Ground area (m2) of one cell for every row of [r0, r1): (res_merc * k(row))^2, k(row) = frame.k * cos(lat_row) / cos(lat_c)."""
    i = raster.info()
    y = i["top"] - (np.arange(r0, r1) + 0.5) * i["resy"]
    lat = 2.0 * np.arctan(np.exp(y / geodesy.R_MERC)) - math.pi / 2.0
    latc = 2.0 * np.arctan(np.exp(frame.yc / geodesy.R_MERC)) - math.pi / 2.0
    k = frame.k * np.cos(lat) / math.cos(latc)
    return (i["resx"] * k) * (i["resy"] * k)


class _Cancelled(Exception):
    pass


def tile_hour_stats(raster, frame, bpts, rpts, band_rows: int = 512):
    """One hourly tile -> dict(flooded, tiers[4], road_km, area_km2, max_depth_m, volume_m3).  bpts / rpts: precomputed sample
    pixels (see compute_tile_timeline)."""
    i = raster.info()
    H = i["height"]
    brow, bcol, bid, nb = bpts
    rrow, rcol, rw = rpts
    bmax = np.zeros(nb, np.float32)
    road_m = 0.0
    area = 0.0
    vol = 0.0
    mx = 0.0
    for r0 in range(0, H, band_rows):
        r1 = min(r0 + band_rows, H)
        band = raster.read_rows(r0, r1)
        wet = band >= WET_M
        cnt = wet.sum(axis=1)
        if cnt.any():
            ca = row_cell_area_m2(raster, frame, r0, r1)
            area += float((cnt * ca).sum())
            vol += float((np.where(wet, band, 0.0).sum(axis=1, dtype=np.float64) * ca).sum())
            mx = max(mx, float(np.nanmax(band)))
        a, b = np.searchsorted(brow, [r0, r1])
        if b > a:
            v = np.nan_to_num(band[brow[a:b] - r0, bcol[a:b]], nan=0.0)
            np.maximum.at(bmax, bid[a:b], v)
        a, b = np.searchsorted(rrow, [r0, r1])
        if b > a:
            v = np.nan_to_num(band[rrow[a:b] - r0, rcol[a:b]], nan=0.0)
            road_m += float(rw[a:b][v >= FLOOD_M].sum())
    fl = bmax >= FLOOD_M
    tiers = np.bincount(np.searchsorted(TIER_EDGES, bmax[fl], side="right") - 1, minlength=4)[:4].astype(np.float64)
    return {"flooded": float(fl.sum()), "tiers": tiers, "road_km": road_m / 1000.0, "area_km2": area / 1e6,
            "max_depth_m": mx, "volume_m3": vol}


def prepare_points(raster, frame, buildings, roads):
    """Pixel tables (sorted by row) of the building footprint samples and the road samples for one tile raster grid."""
    X, Y, ids = building_points(buildings)
    r, c = _tile_pixels(raster, frame, X, Y)
    ok = r >= 0
    r, c, ids = r[ok], c[ok], ids[ok]
    o = np.argsort(r, kind="stable")
    bpts = (r[o], c[o], ids[o].astype(np.int64), int(buildings.n))
    rx = np.asarray(roads.sample_x, np.float64)
    ry = np.asarray(roads.sample_y, np.float64)
    rr, rc = _tile_pixels(raster, frame, rx, ry)
    w = np.asarray(roads.sample_weight_m, np.float64)
    ok = rr >= 0
    rr, rc, w = rr[ok], rc[ok], w[ok]
    o = np.argsort(rr, kind="stable")
    return bpts, (rr[o], rc[o], w[o])


def compute_tile_timeline(tile_storm, buildings, roads, frame, label: str = "", progress=None, cancel: threading.Event | None = None,
                          into: FloodTimeline | None = None, hours=range(N_HOURS)) -> FloodTimeline:
    """All 25 hourly tiles of a storm at full resolution (about 1 - 3 s per hour, in the calling thread).  `into`: a timeline to
    fill in place (the UI reads it while it grows).  Raises nothing for a missing hour file: that hour is dry."""
    tl = into or FloodTimeline(tile_storm.name, "tiles", label or "tiles")
    tl.n_buildings = int(buildings.n)
    tl.road_total_km = float(roads.total_length_m) / 1000.0
    r_any = None
    for h in range(N_HOURS):
        r_any = tile_storm.raster(h)
        if r_any is not None:
            break
    if r_any is None:
        raise FileNotFoundError(f"no tiles for {tile_storm.name}")
    crs = r_any.crs
    if crs is None or crs.to_epsg() != 3857:
        raise ValueError("bulk tile statistics need EPSG:3857 tiles")
    bpts, rpts = prepare_points(r_any, frame, buildings, roads)
    for h in hours:
        if cancel is not None and cancel.is_set():
            raise _Cancelled()
        if np.isfinite(tl.flooded[h]):
            continue
        r = tile_storm.raster(h)
        if r is None:
            st = {"flooded": 0.0, "tiers": np.zeros(4), "road_km": 0.0, "area_km2": 0.0, "max_depth_m": 0.0, "volume_m3": 0.0}
        else:
            st = tile_hour_stats(r, frame, bpts, rpts)
        tl.tiers[h] = st["tiers"]
        tl.road_km[h] = st["road_km"]
        tl.area_km2[h] = st["area_km2"]
        tl.max_depth_m[h] = st["max_depth_m"]
        tl.volume_m3[h] = st["volume_m3"]
        tl.flooded[h] = st["flooded"]                    # last: `done` counts hours whose every field is in
        if progress is not None:
            progress(h + 1, N_HOURS)
    return tl


# ---------------------------------------------------------------------------------------------------- disk cache
def _cache_file(tile_storm, frame, buildings, roads):
    files = [tile_storm.files[h] for h in sorted(tile_storm.files)]
    extra = f"{tile_storm.name}|b{buildings.n}|r{roads.n_samples}|{tile_storm.scale}"
    return cachelib.cache_path("tltiles", frame, files, VERSION, extra)


def save_cache(tl: FloodTimeline, path):
    cachelib.save_npz(path, {"flooded": tl.flooded, "tiers": tl.tiers, "road_km": tl.road_km, "area_km2": tl.area_km2,
                             "max_depth_m": tl.max_depth_m, "volume_m3": tl.volume_m3},
                      {"storm": tl.storm, "n_buildings": tl.n_buildings, "road_total_km": tl.road_total_km, "version": VERSION})


def load_cache(tile_storm, frame, buildings, roads, label: str = "") -> FloodTimeline | None:
    try:
        got = cachelib.load_npz(_cache_file(tile_storm, frame, buildings, roads))
    except OSError:
        return None
    if got is None:
        return None
    a, m = got
    tl = FloodTimeline(tile_storm.name, "tiles", label or "tiles", a["flooded"], a["tiers"], a["road_km"], a["area_km2"],
                       a["max_depth_m"], a["volume_m3"], int(m.get("n_buildings", 0)), float(m.get("road_total_km", 0.0)))
    return tl if tl.complete else None


class TimelineWorker:
    """Tile timelines in ONE low-priority background thread; the newest requested storm is computed first, finished timelines
    are cached in memory and on disk (desktop/cache, keyed by the tile files' size + mtime)."""

    def __init__(self, scene_getter, source_getter):
        self._scene = scene_getter           # -> (buildings, roads, frame) or None while the scene loads
        self._src = source_getter
        self._cv = threading.Condition()
        self._want: list[str] = []
        self._timelines: dict[tuple, FloodTimeline] = {}
        self._stop = False
        self._cancel = threading.Event()
        self.busy = ""
        self.error = ""
        self.t_elapsed = 0.0
        self._th = threading.Thread(target=self._run, name="bgv-timeline", daemon=True)
        self._th.start()

    def get(self, storm: str) -> FloodTimeline | None:
        """The timeline (maybe partial) of a tile storm, queueing its computation; None when tiles are not used for it."""
        src = self._src()
        if not src.uses_tiles(storm):
            return None
        key = (src.tiles.dir.as_posix(), storm)
        with self._cv:
            tl = self._timelines.get(key)
            if tl is not None and (tl.complete or self.busy == storm):
                return tl
            if storm in self._want:
                self._want.remove(storm)
            self._want.insert(0, storm)
            if self.busy and self.busy != storm:
                self._cancel.set()                    # the newest storm first; the aborted one keeps its finished hours
            self._cv.notify()
            return tl

    def _run(self):
        try:
            from ..gpu.bgthread import lower_priority
            lower_priority()
        except Exception:
            pass
        while True:
            with self._cv:
                while not self._want and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                storm = self._want.pop(0)
                self._cancel.clear()
            try:
                self._do(storm)
            except _Cancelled:
                pass
            except Exception as e:
                self.error = f"{storm}: {type(e).__name__}: {e}"[:200]
            finally:
                self.busy = ""

    def _do(self, storm: str):
        sc = self._scene()
        src = self._src()
        if sc is None or not src.uses_tiles(storm):
            time.sleep(0.2)
            return
        buildings, roads, frame = sc
        ts = src.tiles.storm(storm)
        key = (src.tiles.dir.as_posix(), storm)
        label = src.tiles.label(storm)
        with self._cv:
            tl = self._timelines.get(key)
        if tl is None:
            tl = load_cache(ts, frame, buildings, roads, label)
            if tl is not None:
                with self._cv:
                    self._timelines[key] = tl
                return
            tl = FloodTimeline(storm, "tiles", label)
            with self._cv:
                self._timelines[key] = tl
        if tl.complete:
            return
        self.busy = storm
        t0 = time.perf_counter()
        compute_tile_timeline(ts, buildings, roads, frame, label, cancel=self._cancel, into=tl)
        self.t_elapsed = time.perf_counter() - t0
        try:
            save_cache(tl, _cache_file(ts, frame, buildings, roads))
        except Exception:
            pass

    def close(self):
        with self._cv:
            self._stop = True
            self._cancel.set()
            self._cv.notify()
