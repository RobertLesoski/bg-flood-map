"""Depth read-outs for the cinematic tools: solver depth COG tiles (native COG reader, shared tile cache) with the flow grid as fallback.

Layout (the folder name is configurable so finer native tiles can be dropped in later):

    <data root>/layers/<tile_dir>/<domain>/<storm>/depth_<seconds>s.tif   hourly depth, uint16 centimetres, nodata 65535
                                                   max_depth.tif          peak depth over the storm
                                                   cog_meta.json

`tile_dir` is "cog" unless `settings["cinematic"]["tile_dir"]` or the environment variable BGV_TILE_DIR says otherwise.  Any
resolution / CRS-3857 tiling works (the pixel size is read from the file, the 6.25 m of the first delivery is not assumed).

    open_depth_source(app) -> DepthSource   tiles when the domain / storm has them (never for synthetic demo data), else the
                                            flow-grid adapter, else a source that says "no depth data"
    DepthSource.series(lng, lat, radius_m)   float64[25] depth (m) at hours 0..24, NaN where an hour is missing
    DepthSource.depth_at_t(t, lng, lat, r)   depth (m) at continuous hour t (two hours mixed, as the renderer does)
    DepthSource.peak(lng, lat, r)            peak depth (m) over the storm
    DepthSource.label                        the honest source text shown next to every number read here

All reads are windowed and cached; a DepthSource is thread-safe (one lock per open dataset).  Nothing here writes to the data
folder.
"""
from __future__ import annotations

import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

TILE_DIR_DEFAULT = "cog"
WEB_COG_PROGRESS = Path(__file__).resolve().parents[3] / "WEB_COG_PROGRESS.md"
N_HOURS = 25
RADIUS_DEFAULT_M = 25.0                    # read-outs take the maximum within this distance of the point
_DEPTH_RE = re.compile(r"^depth_(\d+)s\.tif$", re.I)
R_MERC = 6378137.0


def tile_dir_name(settings: dict | None = None) -> str:
    """Folder (below `layers/`) that holds `<domain>/<storm>/depth_*.tif`: settings, then BGV_TILE_DIR, then "cog"."""
    st = settings or {}
    s = (st.get("cog_folder") or (st.get("cinematic") or {}).get("tile_dir") or st.get("tile_dir")
         or os.environ.get("BGV_COG_FOLDER") or os.environ.get("BGV_TILE_DIR"))
    return str(s or TILE_DIR_DEFAULT)


def tiles_verified(progress_file: Path | str | None = None) -> bool:
    """True once WEB_COG_PROGRESS.md (main repo, read-only here) carries a line `Status: verified` or `Verification: verified`
    (case-insensitive).  Anything else, an unreadable file included, means the tiles are still unverified."""
    p = Path(progress_file) if progress_file else WEB_COG_PROGRESS
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return re.search(r"(?im)^\s*[-*]?\s*(?:status|verification)\s*:\s*verified\b", txt) is not None


def tile_label(res_m: float, verified: bool) -> str:
    return f"solver depth (tiles, {res_m:g} m; {'verified' if verified else 'verification pending'})"


def lnglat_to_merc(lng: float, lat: float) -> tuple[float, float]:
    x = R_MERC * math.radians(lng)
    y = R_MERC * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def hour_of_seconds(sec: int) -> int:
    """File names are not exact multiples of 3600 (depth_003604s): hour = round(seconds / 3600)."""
    return int(round(sec / 3600.0))


def tile_folder(root, domain_key: str, storm: str, settings: dict | None = None) -> Path:
    return Path(root) / "layers" / tile_dir_name(settings) / domain_key / storm


def tiles_exist(root, domain_key: str, storm: str, settings: dict | None = None) -> bool:
    try:
        return any(_DEPTH_RE.match(f.name) for f in tile_folder(root, domain_key, storm, settings).iterdir())
    except OSError:
        return False


class DepthSource:
    """Base class: `kind` 'tiles' | 'flowgrid' | 'none', `label` (honest source text), `resolution_m`."""
    kind = "none"
    label = "no depth data"
    resolution_m = 0.0

    def series(self, lng: float, lat: float, radius_m: float = RADIUS_DEFAULT_M) -> np.ndarray:
        return np.full(N_HOURS, np.nan)

    def peak(self, lng: float, lat: float, radius_m: float = RADIUS_DEFAULT_M) -> float:
        s = self.series(lng, lat, radius_m)
        return float(np.nanmax(s)) if np.isfinite(s).any() else float("nan")

    def depth_at_t(self, t: float, lng: float, lat: float, radius_m: float = RADIUS_DEFAULT_M) -> float:
        s = self.series(lng, lat, radius_m)
        return mix_hours(s, t)

    def series_many(self, pts, radius_m: float = RADIUS_DEFAULT_M) -> list:
        """[(lng, lat)] -> [float64[25]]: sources that decode whole hours override this to read every hour once."""
        return [self.series(lng, lat, radius_m) for lng, lat in pts]

    def peak_many(self, pts, radius_m: float = RADIUS_DEFAULT_M) -> list:
        return [self.peak(lng, lat, radius_m) for lng, lat in pts]

    def close(self):
        pass


def mix_hours(series: np.ndarray, t: float) -> float:
    """Linear mix of the two hourly samples around continuous hour t (NaN when either is missing)."""
    t = min(max(float(t), 0.0), float(len(series) - 1))
    h0 = int(math.floor(t))
    h1 = min(h0 + 1, len(series) - 1)
    m = t - h0
    a, b = float(series[h0]), float(series[h1])
    if m <= 0.0 or h1 == h0:
        return a
    return a * (1.0 - m) + b * m


class TileDepth(DepthSource):
    kind = "tiles"

    def __init__(self, folder, verified: bool | None = None):
        from . import cog as cogmod                       # native reader + the ONE shared tile LRU (cog.TILE_CACHE)
        self.folder = Path(folder)
        self.files: dict[int, Path] = {}
        for f in sorted(self.folder.iterdir()):
            m = _DEPTH_RE.match(f.name)
            if m:
                self.files[hour_of_seconds(int(m.group(1)))] = f
        if not self.files:
            raise FileNotFoundError(f"no depth_*s.tif in {self.folder}")
        self.max_file = self.folder / "max_depth.tif"
        self._lock = threading.Lock()
        self._ds: dict[Path, object] = {}                  # path -> cog.Cog (tiles are cached in cog.TILE_CACHE, shared)
        self._cache: OrderedDict = OrderedDict()
        first = self._open(next(iter(self.files.values())))
        self.res = float(first.res)
        self.nodata = first.nodata
        self.bounds = tuple(first.bounds)
        self.verified = tiles_verified() if verified is None else bool(verified)
        self.resolution_m = round(self.res, 3)             # the pixel size the delivery names (Mercator metres: 6.25)
        self.label = tile_label(self.resolution_m, self.verified)

    def _open(self, path: Path):
        from . import cog as cogmod
        ds = self._ds.get(path)
        if ds is None:
            ds = cogmod.open_cog(path)
            self._ds[path] = ds
        return ds

    def _read(self, path: Path, lng: float, lat: float, radius_m: float) -> float:
        """Max depth (m) within `radius_m` of the point in one file (0 for nodata = outside the model, NaN outside the tile)."""
        key = (path.name, round(lng, 6), round(lat, 6), float(radius_m))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            x, y = lnglat_to_merc(lng, lat)
            l, b, r, t = self.bounds
            if not (l <= x <= r and b <= y <= t):
                v = float("nan")
            else:
                ds = self._open(path)
                rp = radius_m / max(math.cos(math.radians(lat)), 0.2) / self.res          # radius in pixels
                c0, r0 = (x - l) / self.res, (t - y) / self.res
                i0, i1 = int(math.floor(c0 - rp)), int(math.floor(c0 + rp)) + 1
                j0, j1 = int(math.floor(r0 - rp)), int(math.floor(r0 + rp)) + 1
                i0, j0 = max(i0, 0), max(j0, 0)
                i1, j1 = min(max(i1, i0 + 1), ds.width), min(max(j1, j0 + 1), ds.height)
                a = ds.read_window(0, i0, j0, i1, j1).astype(np.float64)
                jj, ii = np.mgrid[j0:j1, i0:i1]
                inside = (ii + 0.5 - c0) ** 2 + (jj + 0.5 - r0) ** 2 <= max(rp, 0.71) ** 2
                a = np.where(inside & (a != (self.nodata if self.nodata is not None else -1)), a, np.nan)
                v = float(np.nanmax(a)) / 100.0 if np.isfinite(a).any() else 0.0
            self._cache[key] = v
            while len(self._cache) > 8192:
                self._cache.popitem(last=False)
            return v

    def depth_hour(self, hour: int, lng: float, lat: float, radius_m: float = RADIUS_DEFAULT_M) -> float:
        p = self.files.get(int(hour))
        return float("nan") if p is None else self._read(p, lng, lat, radius_m)

    def series(self, lng, lat, radius_m=RADIUS_DEFAULT_M):
        return np.array([self.depth_hour(h, lng, lat, radius_m) for h in range(N_HOURS)], np.float64)

    def depth_at_t(self, t, lng, lat, radius_m=RADIUS_DEFAULT_M):
        t = min(max(float(t), 0.0), N_HOURS - 1.0)
        h0 = int(math.floor(t))
        h1 = min(h0 + 1, N_HOURS - 1)
        a = self.depth_hour(h0, lng, lat, radius_m)
        m = t - h0
        if m <= 0.0 or h1 == h0:
            return a
        return a * (1.0 - m) + self.depth_hour(h1, lng, lat, radius_m) * m

    def peak(self, lng, lat, radius_m=RADIUS_DEFAULT_M):
        if self.max_file.is_file():
            return self._read(self.max_file, lng, lat, radius_m)
        return super().peak(lng, lat, radius_m)

    def close(self):
        with self._lock:
            for ds in self._ds.values():
                try:
                    ds.close()
                except Exception:
                    pass
            self._ds.clear()


class FlowGridDepth(DepthSource):
    """Fallback: depth from the flow grid the renderer shows (bilinear, the same numbers as the probe tooltip)."""
    kind = "flowgrid"

    def __init__(self, storm, demo: bool = False):
        self.storm = storm
        self.demo = demo
        self.resolution_m = float(getattr(storm, "cell_ground_m", 0.0) or storm.meta.get("cell_ground_m", 0.0))
        src = getattr(storm, "source", "")
        what = "DEMO DATA (synthetic)" if demo else ("preview from depth tiers" if src == "tiers" else "flow-grid depth")
        self.label = f"{what}, {self.resolution_m:.0f} m flow grid" if self.resolution_m else what
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _depth_of(self, grid, X: float, Y: float, radius_m: float) -> float:
        offs = [(0.0, 0.0)] if radius_m <= 0 else [(0.0, 0.0)] + [(radius_m * math.cos(a), radius_m * math.sin(a))
                                                                   for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        best = 0.0
        for dx, dy in offs:
            s = self.storm.sample_world(grid, None, 0.0, X + dx, Y + dy)
            if s["valid"]:
                best = max(best, float(s["depth"]))
        return best

    def series_many(self, pts, radius_m=RADIUS_DEFAULT_M):
        """Every hour is decoded once for all the points (a hour is tens of MB: never once per point)."""
        keys = [(round(lng, 6), round(lat, 6), float(radius_m)) for lng, lat in pts]
        with self._lock:
            todo = sorted({i for i, k in enumerate(keys) if k not in self._cache}, key=lambda i: i)
        if todo:
            seen, uniq = set(), []
            for i in todo:
                if keys[i] not in seen:
                    seen.add(keys[i])
                    uniq.append(i)
            xy = [self.storm.frame.lnglat_to_world(*pts[i]) for i in uniq]
            arr = np.zeros((len(uniq), N_HOURS))
            for h in range(N_HOURS):
                g = self.storm.load_hour(h)
                for j, (X, Y) in enumerate(xy):
                    arr[j, h] = self._depth_of(g, X, Y, radius_m)
            with self._lock:
                for j, i in enumerate(uniq):
                    self._cache[keys[i]] = arr[j]
        with self._lock:
            return [self._cache[k] for k in keys]

    def series(self, lng, lat, radius_m=RADIUS_DEFAULT_M):
        return self.series_many([(lng, lat)], radius_m)[0]

    def peak_many(self, pts, radius_m=RADIUS_DEFAULT_M):
        return [float(np.nanmax(s)) if np.isfinite(s).any() else float("nan") for s in self.series_many(pts, radius_m)]


def open_depth_source(app, storm_name: str | None = None) -> DepthSource:
    """The best depth source for `app`'s current domain and storm (see the module docstring).  `storm_name` of another storm:
    tiles only (the flow grid of a storm that is not loaded is not read), else a source that says "no depth data"."""
    dom, storm = app.domain, app.storm
    if storm is None:
        return DepthSource()
    name = storm_name or storm.name
    if not getattr(dom, "demo", False) and tiles_exist(app.root, dom.key, name, app.settings):
        try:
            return TileDepth(tile_folder(app.root, dom.key, name, app.settings))
        except Exception as e:                                   # rasterio missing / damaged file: fall back, say so once
            try:
                app.report_message(f"depth tiles not usable ({type(e).__name__}: {e}); using the flow grid", toast=False)
            except Exception:
                pass
    if name != storm.name:
        return DepthSource()
    try:
        return FlowGridDepth(storm, demo=bool(getattr(dom, "demo", False) or getattr(storm, "synthetic", False)))
    except Exception:
        return DepthSource()


class SeriesCache:
    """Background depth series.  One worker thread; the requests that arrive while it works are read as ONE batch per source
    (`series_many`), so a panel asking for twenty places costs one pass over the hours, not twenty.  `get` returns the cached
    array, or None while it is being read."""

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bgv-depth")
        self._done: dict = {}
        self._pending: dict = {}                 # key -> (source, lng, lat, radius)
        self._inflight: set = set()              # keys of the batch being read now
        self._queued = False
        self._lock = threading.Lock()

    def get(self, source: DepthSource, lng: float, lat: float, radius_m: float = RADIUS_DEFAULT_M):
        key = (id(source), round(lng, 6), round(lat, 6), float(radius_m))
        with self._lock:
            if key in self._done:
                return self._done[key]
            if key not in self._inflight:
                self._pending.setdefault(key, (source, lng, lat, radius_m))
            if not self._queued:
                self._queued = True
                self._pool.submit(self._drain)
        return None

    def _drain(self):
        while True:
            with self._lock:
                if not self._pending:
                    self._queued = False
                    return
                batch = dict(self._pending)
                self._inflight.update(batch)
                self._pending.clear()
            groups: dict = {}
            for key, (src, lng, lat, r) in batch.items():
                groups.setdefault((id(src), r), (src, r, []))[2].append((key, lng, lat))
            for src, r, items in groups.values():
                try:
                    res = src.series_many([(lng, lat) for _, lng, lat in items], r)
                except Exception:
                    res = [np.full(N_HOURS, np.nan)] * len(items)
                with self._lock:
                    for (key, _, _), v in zip(items, res):
                        self._done[key] = np.asarray(v, np.float64)
                        self._inflight.discard(key)
                    while len(self._done) > 4096:
                        self._done.pop(next(iter(self._done)))

    def get_blocking(self, source: DepthSource, lng: float, lat: float, radius_m: float = RADIUS_DEFAULT_M, timeout: float = 120.0):
        import time
        t0 = time.perf_counter()
        while True:
            v = self.get(source, lng, lat, radius_m)
            if v is not None:
                return v
            if time.perf_counter() - t0 > timeout:
                return np.full(N_HOURS, np.nan)
            time.sleep(0.01)

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
