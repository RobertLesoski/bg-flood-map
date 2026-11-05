"""Depth-vs-hour series of a picked point or building, for the selected storm and for EVERY storm of the domain.

    t = point_target(lng, lat)                         a map point
    t = building_target(app_buildings, i, frame)       a building (max over its footprint, the statistics rule)
    w = SeriesWorker(depth_source)
    w.request(t, storms, first="100yr")                background, current storm first; results appear progressively
    r = w.latest(t)  ->  SeriesResult | None           depth[n_storm, 25] (NaN = not computed yet), peak per storm, hour of the peak

Rules (stated in the UI):
* depth at integer hours 0..24 from whichever back end `depth_source` uses for that storm (tiles or the flow grid; `source[i]`);
* a point is the bilinear depth at that point; a building is the MAX depth over its footprint: on the flow grid exactly the
  statistics rule (all-touched pixels, `evaluate_buildings`), on the tiles the max over the footprint's corner vertices and
  its centroid (the tile pixels are 6.25 m, a house is 2 to 4 of them);
* 'peak' is the peak of the HOURLY snapshots; the solver may peak between two hours.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import geodesy

N_HOURS = 25


@dataclass(frozen=True)
class Target:
    kind: str                      # 'point' | 'building'
    key: tuple
    label: str
    lng_t: tuple                   # sample longitudes / latitudes used on the TILES (a point: one sample)
    lat_t: tuple
    lng_g: tuple                   # ... and on the FLOW GRID (a building: its all-touched pixel centres)
    lat_g: tuple

    def arrays(self, backend: str):
        if backend == "tiles":
            return np.asarray(self.lng_t, np.float64), np.asarray(self.lat_t, np.float64)
        return np.asarray(self.lng_g, np.float64), np.asarray(self.lat_g, np.float64)


def point_target(lng: float, lat: float, label: str = "") -> Target:
    p = (float(lng),)
    q = (float(lat),)
    return Target("point", ("p", round(float(lng), 7), round(float(lat), 7)), label or f"{lat:.5f}, {lng:.5f}", p, q, p, q)


def building_footprint_samples(buildings, i: int):
    """World (x, y) of building i's footprint corner vertices (from the render mesh, de-duplicated) plus its centroid."""
    m = buildings.mesh
    ids = getattr(buildings, "_bid_sorted", None)
    if ids is None:                                    # one-off sort of the mesh vertices by building id
        order = np.argsort(m.building_id, kind="stable")
        sid = m.building_id[order]
        starts = np.searchsorted(sid, np.arange(buildings.n + 1))
        try:
            buildings._bid_sorted = (order, starts)
        except Exception:
            pass
        ids = (order, starts)
    order, starts = ids
    sel = order[starts[i]:starts[i + 1]]
    xy = np.unique(np.round(m.positions[sel][:, :2].astype(np.float64), 2), axis=0) if sel.size else np.zeros((0, 2))
    c = np.array([[float(buildings.cx[i]), float(buildings.cy[i])]])
    return np.vstack([xy, c]) if xy.size else c


def building_target(buildings, i: int, frame, label: str = "") -> Target:
    xy = building_footprint_samples(buildings, i)
    lng_t, lat_t = geodesy.world_to_lnglat(frame, xy[:, 0], xy[:, 1])
    pix = np.asarray(buildings.index.pixels_of(i), np.int64)
    if pix.size:
        cols, rows = pix % frame.width, pix // frame.width
        xm = frame.xmin + (cols + 0.5) * frame.cell_m
        ym = frame.ymax - (rows + 0.5) * frame.cell_m
        lng_g, lat_g = geodesy.merc_to_lnglat(xm, ym)
    else:
        lng_g, lat_g = lng_t, lat_t
    addr = str(buildings.address[i]) if hasattr(buildings, "address") else ""
    return Target("building", ("b", int(i)), label or (addr or f"Building {i}"),
                  tuple(lng_t.tolist()), tuple(lat_t.tolist()), tuple(lng_g.tolist()), tuple(lat_g.tolist()))


@dataclass
class SeriesResult:
    target: Target
    storms: list
    depth: np.ndarray                  # [n_storm, 25], NaN = not computed yet
    source: list                       # 'tiles' | 'grid' per storm
    labels: list                       # honest source text per storm
    done: int = 0
    total: int = 0
    error: str = ""
    t_start: float = field(default_factory=time.perf_counter)

    @property
    def complete(self) -> bool:
        return self.done >= self.total > 0

    def peak(self, i: int) -> float:
        row = self.depth[i]
        return float(np.nanmax(row)) if np.isfinite(row).any() else float("nan")

    def peak_hour(self, i: int) -> int:
        row = self.depth[i]
        return int(np.nanargmax(row)) if np.isfinite(row).any() else -1

    def index(self, storm: str) -> int:
        return self.storms.index(storm) if storm in self.storms else -1


def compute_row(source, target: Target, storm: str, pool: ThreadPoolExecutor | None = None) -> tuple[np.ndarray, str]:
    """Depth of the target at hours 0..24 for one storm.  Returns (float64[25], backend key)."""
    backend = source.source_key(storm)
    lng, lat = target.arrays(backend)
    red = np.max if target.kind == "building" else (lambda a: a[0])

    def one(h):
        d = source.depth_many(lng, lat, storm, float(h), method="nearest" if target.kind == "building" else None)
        return float(red(d)) if d.size else 0.0
    if pool is not None and backend == "tiles":
        row = list(pool.map(one, range(N_HOURS)))
    else:
        row = [one(h) for h in range(N_HOURS)]
    return np.asarray(row, np.float64), backend


class SeriesWorker:
    """One background thread; the newest request wins (an older one stops at the next storm / hour)."""

    def __init__(self, source_getter, cache_size: int = 24):
        self._src = source_getter                        # callable -> HybridDepthSource (it can change with settings)
        self._cv = threading.Condition()
        self._req = None
        self._gen = 0
        self._stop = False
        self._cache: "OrderedDict[tuple, SeriesResult]" = OrderedDict()
        self._cache_size = cache_size
        self.busy = False
        self.error = ""
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="bgv-series")
        self._th = threading.Thread(target=self._run, name="bgv-series-main", daemon=True)
        self._th.start()

    # ------------------------------------------------------------------------------------------
    def _key(self, target: Target, storms, mode_key: str):
        return (target.key, tuple(storms), mode_key)

    def request(self, target: Target, storms, first: str | None = None) -> SeriesResult:
        """Start (or reuse) the computation; returns the live result object (filled progressively)."""
        src = self._src()
        mode_key = f"{src.mode}|{src.tiles is not None}"
        storms = list(storms)
        if first in storms:
            storms.remove(first)
            storms.insert(0, first)
        key = self._key(target, sorted(storms), mode_key)
        with self._cv:
            r = self._cache.get(key)
            if r is not None and not r.error:
                self._cache.move_to_end(key)
                if first in r.storms and r.storms[0] != first and r.complete is False:
                    pass
                self._latest = r
                if r.complete:
                    return r
            else:
                r = SeriesResult(target, storms, np.full((len(storms), N_HOURS), np.nan), ["?"] * len(storms),
                                 [""] * len(storms), 0, len(storms))
                self._cache[key] = r
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._latest = r
            self._gen += 1
            self._req = (self._gen, r, src)
            self._cv.notify()
        return r

    def latest(self, target: Target | None = None) -> SeriesResult | None:
        r = getattr(self, "_latest", None)
        if r is None or (target is not None and r.target.key != target.key):
            return None
        return r

    # ------------------------------------------------------------------------------------------
    def _run(self):
        try:
            from ..gpu.bgthread import lower_priority
            lower_priority()
        except Exception:
            pass
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                (gen, res, src), self._req = self._req, None
                self.busy = True
            try:
                self._fill(gen, res, src)
            except Exception as e:                        # never kill the worker
                res.error = f"{type(e).__name__}: {e}"
                self.error = res.error
            finally:
                self.busy = False

    def _fill(self, gen, res: SeriesResult, src):
        for i, storm in enumerate(res.storms):
            if gen != self._gen:
                return
            if np.isfinite(res.depth[i]).all():
                continue
            try:
                row, backend = compute_row(src, res.target, storm, self._pool)
            except Exception as e:                        # a storm whose files are unreadable: leave its row empty
                res.error = f"{storm}: {type(e).__name__}: {e}"[:160]
                res.depth[i] = 0.0
                res.source[i] = "error"
                res.done += 1
                continue
            res.depth[i] = row
            res.source[i] = backend
            try:
                res.labels[i] = src.label(storm)
            except Exception:
                res.labels[i] = backend
            res.done += 1

    def close(self):
        with self._cv:
            self._stop = True
            self._gen += 1
            self._cv.notify()
        self._pool.shutdown(wait=False, cancel_futures=True)
