"""Depth adapter for the analysis tools:  depth_at(lng, lat, storm, hour)  ->  metres.

Every analysis feature (profile, point / building series, compare) asks a `DepthSource`; none of them touches a file format.

    src = make_depth_source(app)                 # tiles where they exist (county), the flow grid otherwise / as fallback
    src.depth_at(lng, lat, "100yr", 12.0)        # float, metres; 0 = dry, below the wet threshold or outside the model domain
    src.depth_many(lng_array, lat_array, "100yr", 12.5)    # float64 array, same rules (vectorised, 1e5 points in ~0.1 s)
    src.label("100yr")                           # honest source text for the UI ('solver depth (tiles, 6.25 m; verification pending)')
    src.uses_tiles("100yr")                      # which backend answered for this storm

Back ends
---------
FlowGridDepthSource   the flow grid (`bgv.data.flow.Storm`): bilinear in code space, each hour decoded, hours blended - exactly
                      `Storm.sample_t` (the tests check 100+ points against it).
TileDepthSource       the COG depth tiles (`<root>/layers/cog/<domain>/<storm>/depth_*s.tif`, `cog_meta.json`): uint16 centimetres,
                      EPSG:3857, 6.25 m, read-only windowed reads through `rastersrc.BlockRaster` (LRU block cache).  The folder is
                      configurable (settings['tile_dir'], env BGV_TILE_DIR, `make_depth_source(tile_dir=...)`): finer native tiles in
                      the same layout (any resolution / CRS / unit given by cog_meta.json) drop in without code changes.
HybridDepthSource     picks per storm: tiles when that storm has them (and the user did not force the grid), else the grid.

Hours: the hour index is the viewer's (0..24).  A continuous hour blends the two neighbouring hourly rasters after decoding,
like the shaders.  Depth below `WET_THRESHOLD_M` (0.02 m) reads 0, as in the hover probe.

Honesty: tiles are labelled 'verification pending' until the tile pipeline's progress log says it was verified against the
solver rasters (`tiles_verified`; only the explicit markers listed there count, or settings['tiles_verified'] = True).
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path

import numpy as np

from ..data import flow as flowdata
from . import geodesy
from .rastersrc import BlockRaster

WET_THRESHOLD_M = flowdata.WET_THRESHOLD_M
VERIFIED_MARKERS = ("status: verified",)          # kept for importers; the policy lives in bgv.data.depthtiles.tiles_verified
PROGRESS_LOG_NAMES = ("WEB_COG_PROGRESS.md",)
DEFAULT_PROGRESS_LOG = str(Path(__file__).resolve().parents[3] / "WEB_COG_PROGRESS.md")


# ---------------------------------------------------------------------------------------------------- verification gate
def progress_log_path(root=None, settings: dict | None = None) -> Path | None:
    cands = []
    if settings and settings.get("tiles_progress_log"):
        cands.append(Path(str(settings["tiles_progress_log"])))
    if os.environ.get("BGV_COG_PROGRESS"):
        cands.append(Path(os.environ["BGV_COG_PROGRESS"]))
    if root is not None:
        cands.append(Path(root) / "WEB_COG_PROGRESS.md")
    cands.append(Path(DEFAULT_PROGRESS_LOG))
    for p in cands:
        try:
            if p.is_file():
                return p
        except OSError:
            pass
    return None


def tiles_verified(root=None, settings: dict | None = None) -> bool:
    """ONE verification policy shared with the rest of the viewer (`bgv.data.depthtiles.tiles_verified`): True only for
    settings['tiles_verified'] = True or a `Status: verified` line in WEB_COG_PROGRESS.md (read-only).  An explicit log path
    (settings['tiles_progress_log'], env BGV_COG_PROGRESS) is honoured for tests and other checkouts."""
    if settings and settings.get("tiles_verified") is True:
        return True
    from ..data import depthtiles as dt
    p = progress_log_path(root, settings)
    return dt.tiles_verified(p) if p is not None else False


# ---------------------------------------------------------------------------------------------------- flow grid
def sample_depth_codes(rgba_a, rgba_b, mix: float, x_merc, y_merc, meta: dict) -> np.ndarray:
    """Vectorised depth of `flow.sample_grids`: bilinear in code space at Mercator points, each hour decoded, blended
    (1 - mix) * A + mix * B.  A grid may be None (dry).  Returns float64 metres, 0 where invalid (outside the model domain)."""
    x = np.asarray(x_merc, np.float64).reshape(-1)
    y = np.asarray(y_merc, np.float64).reshape(-1)
    W, H, cell = int(meta["width"]), int(meta["height"]), float(meta["cell_m"])
    b = meta["bounds_merc"]
    gx = (x - b[0]) / cell
    gy = (b[3] - y) / cell
    inside = (gx >= 0.0) & (gx <= W) & (gy >= 0.0) & (gy <= H)
    out = np.zeros(x.shape, np.float64)
    if not inside.any():
        return out
    idx = np.flatnonzero(inside)
    fx, fy = gx[idx] - 0.5, gy[idx] - 0.5
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    tx, ty = fx - x0, fy - y0
    xa, xb = np.maximum(x0, 0), np.minimum(x0 + 1, W - 1)
    ya, yb = np.maximum(y0, 0), np.minimum(y0 + 1, H - 1)
    w00, w10, w01, w11 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty
    dmax = float((meta.get("depth_encoding") or {}).get("max_m", flowdata.DEPTH_MAX_M))
    m = 0.0 if not mix > 0 else (1.0 if mix >= 1 else float(mix))

    def one(arr):
        if arr is None:
            return np.zeros(idx.size), np.zeros(idx.size)
        r = (arr[ya, xa, 0] * w00 + arr[ya, xb, 0] * w10 + arr[yb, xa, 0] * w01 + arr[yb, xb, 0] * w11)
        a = (arr[ya, xa, 3] * w00 + arr[ya, xb, 3] * w10 + arr[yb, xa, 3] * w01 + arr[yb, xb, 3] * w11)
        t = r / 255.0
        return t * t * dmax, a

    dA, aA = one(rgba_a)
    dB, aB = one(rgba_b)
    valid = np.maximum(aA, aB) >= 127.5
    out[idx] = np.where(valid, dA * (1 - m) + dB * m, 0.0)
    return out


class FlowGridDepthSource:
    kind = "grid"

    def __init__(self, open_storm, current=None, storms=None):
        """open_storm(name) -> data.flow.Storm; current() -> the viewer's live Storm (shares its hour cache) or None."""
        self._open = open_storm
        self._current = current or (lambda: None)
        self._storms: dict = {}
        self._names = list(storms or [])
        self._lock = threading.Lock()

    def storms(self) -> list[str]:
        return list(self._names)

    def has(self, storm: str) -> bool:
        return storm in self._names or self._current_is(storm)

    def _current_is(self, storm):
        c = self._current()
        return c is not None and getattr(c, "name", None) == storm

    def storm_obj(self, name: str):
        c = self._current()
        if c is not None and getattr(c, "name", None) == name:
            return c
        with self._lock:
            s = self._storms.get(name)
            if s is None:
                s = self._storms[name] = self._open(name)
            return s

    def depth_many(self, lng, lat, storm: str, hour: float) -> np.ndarray:
        st = self.storm_obj(storm)
        shape = np.shape(lng)
        x, y = geodesy.lnglat_to_merc(np.asarray(lng).reshape(-1), np.asarray(lat).reshape(-1))
        h0, h1, mix = flowdata.hour_pair(hour)
        a = None if st.is_zero(h0) else st.load_hour(h0)
        b = None if (mix <= 0 or st.is_zero(h1)) else st.load_hour(h1)
        d = sample_depth_codes(a, b, mix, x, y, st.meta)
        d[d < st.wet_threshold] = 0.0
        return d.reshape(shape)

    def depth_at(self, lng, lat, storm, hour) -> float:
        return float(self.depth_many(np.array([lng]), np.array([lat]), storm, hour)[0])

    def label(self, storm: str) -> str:
        st = self.storm_obj(storm)
        cell = float(st.meta.get("cell_ground_m", 0.0))
        if st.synthetic:
            return f"DEMO DATA - synthetic flow grid ({cell:.1f} m)"
        if st.source == "tiers":
            return f"preview from depth tiers (flow grid, {cell:.1f} m)"
        return f"solver depth (flow grid, {cell:.1f} m)"

    def resolution_m(self, storm: str) -> float:
        return float(self.storm_obj(storm).meta.get("cell_ground_m", 0.0))

    def close(self):
        with self._lock:
            for s in self._storms.values():
                try:
                    s.close()
                except Exception:
                    pass
            self._storms.clear()


# ---------------------------------------------------------------------------------------------------- tiles
class TileStorm:
    """One storm's tile folder: `cog_meta.json` + `depth_<seconds>s.tif` (hour = round(seconds / 3600)) + `max_depth.tif`."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        meta = {}
        mp = self.folder / "cog_meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        self.meta = meta
        units = str(meta.get("units", "cm")).lower()
        self.scale = {"cm": 0.01, "m": 1.0, "mm": 0.001, "ft": 0.3048}.get(units, 0.01)
        self.nodata = meta.get("nodata", 65535)
        self.res_m = float(meta.get("res_m", 6.25))
        self.files: dict[int, Path] = {}
        hm = meta.get("hours")
        if isinstance(hm, dict):
            for k, fn in hm.items():
                try:
                    p = self.folder / str(fn)
                    if p.is_file():
                        self.files[int(k)] = p
                except (TypeError, ValueError):
                    pass
        if not self.files:                                     # no / unreadable meta: name -> hour by rounding
            best: dict = {}
            for p in self.folder.glob("depth_*s.tif"):
                m = re.match(r"depth_(\d+)s\.tif$", p.name)
                if m:
                    sec = int(m.group(1))
                    h = int(round(sec / 3600.0))
                    if 0 <= h <= 24 and (h not in best or abs(sec - h * 3600) < abs(best[h][0] - h * 3600)):
                        best[h] = (sec, p)
            self.files = {h: p for h, (_, p) in best.items()}
        self.max_path = self.folder / "max_depth.tif"
        self._rasters: dict[int, BlockRaster] = {}
        self._lock = threading.Lock()
        self.bounds_merc = tuple(meta["bounds_merc"]) if meta.get("bounds_merc") else None
        self.width = int(meta.get("width", 0))
        self.height = int(meta.get("height", 0))

    @property
    def usable(self) -> bool:
        return bool(self.files)

    def raster(self, hour: int) -> BlockRaster | None:
        p = self.files.get(int(hour))
        if p is None:
            return None
        with self._lock:
            r = self._rasters.get(int(hour))
            if r is None:
                r = self._rasters[int(hour)] = BlockRaster(p, scale=self.scale, nodata=self.nodata)
            return r

    def max_raster(self) -> BlockRaster | None:
        if not self.max_path.is_file():
            return None
        with self._lock:
            r = self._rasters.get(-1)
            if r is None:
                r = self._rasters[-1] = BlockRaster(self.max_path, scale=self.scale, nodata=self.nodata)
            return r

    def signature(self) -> str:
        """Size + mtime of every file (cache key for derived data)."""
        parts = []
        for h in sorted(self.files):
            st = os.stat(self.files[h])
            parts.append(f"{h}:{st.st_size}:{st.st_mtime_ns}")
        return "|".join(parts)


class TileDepthSource:
    kind = "tiles"

    def __init__(self, tiles_dir, settings: dict | None = None, root=None):
        self.dir = Path(tiles_dir)
        self.settings = settings or {}
        self.root = root
        self._storms: dict[str, TileStorm] = {}
        self._lock = threading.Lock()
        self._crs_cache: dict = {}

    @staticmethod
    def discover(root, domain_key: str, settings: dict | None = None):
        """TileDepthSource for this domain, or None when there is no tile folder (the city today)."""
        settings = settings or {}
        cands = []
        if settings.get("tile_dir") and os.path.isabs(str(settings["tile_dir"])):      # legacy: a full path
            cands.append(Path(str(settings["tile_dir"])))
        if os.environ.get("BGV_TILE_DIR") and os.path.isabs(os.environ["BGV_TILE_DIR"]):
            cands.append(Path(os.environ["BGV_TILE_DIR"]))
        from ..data.depthtiles import tile_dir_name                                  # cog_folder / BGV_COG_FOLDER (+ aliases): a name under layers/
        cands.append(Path(root) / "layers" / tile_dir_name(settings) / domain_key)
        for c in cands:
            try:
                if c.is_dir() and any(p.is_dir() for p in c.iterdir()):
                    return TileDepthSource(c, settings, root)
            except OSError:
                continue
        return None

    def storm(self, name: str) -> TileStorm | None:
        with self._lock:
            if name in self._storms:
                return self._storms[name]
        p = self.dir / name
        ts = TileStorm(p) if p.is_dir() else None
        if ts is not None and not ts.usable:
            ts = None
        with self._lock:
            self._storms[name] = ts
        return ts

    def has(self, storm: str) -> bool:
        return self.storm(storm) is not None

    def storms(self) -> list[str]:
        try:
            return sorted(p.name for p in self.dir.iterdir() if p.is_dir() and self.has(p.name))
        except OSError:
            return []

    def resolution_m(self, storm: str) -> float:
        ts = self.storm(storm)
        return ts.res_m if ts else 0.0

    def verified(self) -> bool:
        return tiles_verified(self.root, self.settings)

    def label(self, storm: str) -> str:
        ts = self.storm(storm)
        res = ts.res_m if ts else 0.0
        return f"solver depth (tiles, {res:g} m; {'verified' if self.verified() else 'verification pending'})"

    def _to_raster_crs(self, r: BlockRaster, lng, lat):
        crs = r.crs
        if crs is not None and crs.to_epsg() == 3857:
            return geodesy.lnglat_to_merc(lng, lat)
        from rasterio.warp import transform
        xs, ys = transform("EPSG:4326", crs, np.asarray(lng, np.float64).tolist(), np.asarray(lat, np.float64).tolist())
        return np.asarray(xs), np.asarray(ys)

    def depth_hour(self, lng, lat, storm: str, hour: int, method: str = "bilinear"):
        """Depth (m) at points for ONE hourly raster; NaN where outside the model domain / no tile.  (np.ndarray)"""
        ts = self.storm(storm)
        lng = np.asarray(lng, np.float64).reshape(-1)
        lat = np.asarray(lat, np.float64).reshape(-1)
        r = ts.raster(hour) if ts else None
        if r is None:
            return np.zeros(lng.shape)                       # an hour without a file is dry by definition
        x, y = self._to_raster_crs(r, lng, lat)
        return r.sample(x, y, method)

    def depth_many(self, lng, lat, storm: str, hour: float, method: str = "bilinear") -> np.ndarray:
        shape = np.shape(lng)
        h0, h1, mix = flowdata.hour_pair(hour)
        a = self.depth_hour(lng, lat, storm, h0, method)
        if mix > 0:
            b = self.depth_hour(lng, lat, storm, h1, method)
            # outside the domain in one hour only: use the other one, both NaN -> NaN
            d = np.where(np.isnan(a), b, np.where(np.isnan(b), a, a * (1 - mix) + b * mix))
        else:
            d = a
        d = np.nan_to_num(d, nan=0.0)
        d[d < WET_THRESHOLD_M - 1e-6] = 0.0              # (float32 holds 0.02 as 0.0199999996: a 2 cm cell is wet)
        return d.reshape(shape)

    def depth_at(self, lng, lat, storm, hour) -> float:
        return float(self.depth_many(np.array([lng]), np.array([lat]), storm, hour)[0])

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------- hybrid
class HybridDepthSource:
    """Tiles for the storms that have them, the flow grid for everything else.  `mode`: 'auto' | 'grid' (force the grid)."""

    def __init__(self, grid: FlowGridDepthSource, tiles: TileDepthSource | None = None, mode: str = "auto"):
        self.grid = grid
        self.tiles = tiles
        self.mode = mode

    def uses_tiles(self, storm: str) -> bool:
        return bool(self.tiles is not None and self.mode != "grid" and self.tiles.has(storm))

    def backend(self, storm: str):
        return self.tiles if self.uses_tiles(storm) else self.grid

    def depth_many(self, lng, lat, storm: str, hour: float, method: str | None = None) -> np.ndarray:
        """`method='nearest'` reads the tile cell containing each point (footprint / road statistics); default bilinear.  The flow
        grid back end ignores it (its building / road samples are pixel centres, where both are the same number)."""
        b = self.backend(storm)
        if method is not None and b is self.tiles:
            return b.depth_many(lng, lat, storm, hour, method)
        return b.depth_many(lng, lat, storm, hour)

    def depth_at(self, lng, lat, storm: str, hour: float) -> float:
        return self.backend(storm).depth_at(lng, lat, storm, hour)

    def label(self, storm: str) -> str:
        return self.backend(storm).label(storm)

    def resolution_m(self, storm: str) -> float:
        return self.backend(storm).resolution_m(storm)

    def source_key(self, storm: str) -> str:
        return "tiles" if self.uses_tiles(storm) else "grid"

    def storms(self) -> list[str]:
        return self.grid.storms()

    def close(self):
        self.grid.close()


def make_depth_source(app=None, *, root=None, domain=None, settings=None, tile_dir=None, current=None, mode: str | None = None):
    """The adapter for a running App (`app`) or for tools / tests (`root` + `domain`).  `tile_dir` overrides the tile folder."""
    if app is not None:
        root = app.root
        domain = app.domain
        settings = app.settings if settings is None else settings
        current = current or (lambda: app.storm)
    settings = settings or {}
    grid = FlowGridDepthSource(domain.open_storm, current, list(domain.storms))
    tiles = None
    if tile_dir is not None:
        tiles = TileDepthSource(tile_dir, settings, root) if Path(tile_dir).is_dir() else None
    else:
        tiles = TileDepthSource.discover(root, domain.key, settings)
    return HybridDepthSource(grid, tiles, mode or str(settings.get("depth_source", "auto")))
