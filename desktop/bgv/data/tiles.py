"""Native-resolution solver depth tiles (COGs) as a service for the viewer: lookup, point / profile / footprint depth, and the
per-building tile statistics.  Pure CPU (no wgpu); thread safe; nothing here blocks the render thread for more than a cached
tile read (the heavy work runs in daemon worker threads: `start_index`, `building_depths`).

Files:  <base>/<scope>/<storm>/{cog_meta.json, max_depth.tif, depth_<seconds>s.tif}      base = BGV_COG_DIR | settings["cog_dir"] |
<data root>/layers/cog.  Scope 'county' = the county solver run (12 storms); 'city' = a city run (not produced yet).  The city view
falls back to the county run for the same storm name, labelled with scope 'county' (the county run covers the city).

    svc = TileService(root, domain, settings)
    svc.tileset("100yr")                          -> TileSet | None
    svc.depth_lnglat("100yr", lng, lat, "peak" | hour) -> DepthSample
    svc.depth_profile("100yr", lngs, lats, which) -> float64[n] metres (NaN = no data)
    svc.start_index(get_geojson_path)             background: footprint pixel index on the tile grid (cached in desktop/cache)
    svc.building_depth(bid, "100yr", which)       -> DepthSample (footprint maximum), status 'unavailable' until the index is ready
    svc.building_depths("100yr", which)           -> int32[n] centimetres (-1 = no data) or None while a worker computes it

Footprint rule = the desktop viewer's flow-grid rule applied on the tile grid: ALL-TOUCHED pixels, building depth = MAX over them.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cog as cogmod
from ..geo import GridFrame

TILE_LABEL = "solver depth (native tiles)"
GRID_LABEL = "flow-grid depth"
TIER_EDGES_CM = (10, 30, 60, 120)                          # shallow / moderate / deep / severe, same as the viewer (0.1 0.3 0.6 1.2 m)
FLOODED_CM = 10
INDEX_VERSION = "cogidx1"


@dataclass(frozen=True)
class DepthSample:
    status: str                      # 'depth' | 'dry' | 'nodata' | 'unavailable'
    m: float | None = None
    cm: float | None = None
    ft: float | None = None
    source: str = ""                 # 'tiles' | 'grid'  (what was actually used)
    label: str = ""
    storm: str = ""
    hour: object = None              # int hour or 'peak'
    scope: str = ""                  # 'county' | 'city' | ''
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("depth", "dry")

    def text(self, imperial: bool = False) -> str:
        if self.status == "nodata":
            return "no data (outside the modelled area)"
        if self.status == "unavailable":
            return self.note or "not available"
        if self.status == "dry":
            return "no water (0 cm)"
        return f"{self.cm:.0f} cm ({self.cm / 30.48:.2f} ft)"


def unavailable(source: str, storm: str, hour, note: str, scope: str = "") -> DepthSample:
    return DepthSample("unavailable", None, None, None, source, TILE_LABEL if source == "tiles" else GRID_LABEL, storm, hour, scope, note)


def tile_sample(raw, status_code: int, storm, hour, scope, note="", label=TILE_LABEL) -> DepthSample:
    if status_code != cogmod.ST_VALUE:
        return DepthSample("nodata", None, None, None, "tiles", label, storm, hour, scope, note)
    return DepthSample("dry" if raw == 0 else "depth", raw / 100.0, float(raw), raw / 100.0 * cogmod.M2FT, "tiles", label,
                       storm, hour, scope, note)


def norm_hour(hour):
    """'peak' / 'max' -> 'peak'; numbers -> float; None stays None."""
    if hour is None:
        return None
    if isinstance(hour, str):
        return "peak" if hour.lower() in ("peak", "max") else float(hour)
    return float(hour)


# ------------------------------------------------------------------------------------------------ tile set
class TileSet:
    """The files of one storm in one scope."""

    def __init__(self, scope: str, storm: str, folder: Path, meta: dict):
        self.scope, self.storm, self.folder, self.meta = scope, storm, Path(folder), meta
        self.hours = cogmod.hour_files(meta)                     # {hour: file name}
        self.has_peak = (self.folder / "max_depth.tif").is_file()
        self._cogs: dict = {}
        self._lock = threading.Lock()
        self.errors: dict = {}
        self.verified = False                                     # set by the service (settings / meta)
        try:
            self.res_m = float(meta.get("res_m") or 0.0)
        except (TypeError, ValueError):
            self.res_m = 0.0

    @property
    def label(self) -> str:
        """What to call this depth in the UI.  'verification pending' stays until WEB_COG_PROGRESS.md says verified and the
        setting `tiles_verified` is true (or the folder's cog_meta.json carries "verified": true)."""
        r = f"{self.res_m:g} m" if self.res_m else "tiles"
        if self.res_m and self.res_m <= 5.0:
            base = f"solver depth (native tiles, {r})"
        else:
            base = f"solver depth (tiles, {r}"+ ")" if self.res_m else "solver depth (tiles)"
        return base if self.verified else base[:-1] + ", verification pending)"
    def nearest_hour(self, t: float):
        if not self.hours:
            return None
        h = int(round(float(t)))
        if h in self.hours:
            return h
        best = min(self.hours, key=lambda k: abs(k - t))
        return best if abs(best - t) <= 1.01 else None

    def file_for(self, which):
        if which == "peak":
            return "max_depth.tif" if self.has_peak else None
        h = self.nearest_hour(which)
        return None if h is None else self.hours[h]

    def cog(self, which):
        """Opened Cog for 'peak' or an hour (None when that file is not there or unreadable; the reason is in `errors`)."""
        name = self.file_for(which)
        if name is None:
            return None
        with self._lock:
            c = self._cogs.get(name)
            if c is None and name not in self.errors:
                try:
                    c = self._cogs[name] = cogmod.open_cog(self.folder / name)
                except Exception as e:
                    self.errors[name] = f"{type(e).__name__}: {e}"
            return c

    def grid_signature(self):
        c = self.cog("peak") or (self.cog(next(iter(self.hours))) if self.hours else None)
        if c is None:
            return None
        return (c.width, c.height, round(c.res, 6), round(c.x0, 3), round(c.y0, 3))

    def close(self):
        with self._lock:
            for c in self._cogs.values():
                c.close()
            self._cogs.clear()


# ------------------------------------------------------------------------------------------------ service
class TileService:
    def __init__(self, root, domain=None, settings: dict | None = None, base=None):
        settings = settings or {}
        self.domain_key = getattr(domain, "key", "city")
        self.enabled = bool(settings.get("tiles_enabled", True))
        self.reason = ""
        b = base or os.environ.get("BGV_COG_DIR") or settings.get("cog_dir")
        folder = os.environ.get("BGV_COG_FOLDER") or settings.get("cog_folder") or "cog"      # e.g. "cog_native" when finer tiles land
        self.base = Path(b) if b else Path(root) / "layers" / folder
        self.verified = bool(settings.get("tiles_verified", False))
        if domain is not None and getattr(domain, "demo", False):
            self.enabled, self.reason = False, "the county demo data is synthetic and has no solver tiles"
        self._sets: dict = {}
        self._lock = threading.Lock()
        self.index = None                      # PixelIndex on the tile grid
        self.index_sig = None
        self.index_state = "idle"              # idle | building | ready | error
        self.index_msg = ""
        self._stats: OrderedDict = OrderedDict()   # (storm, which) -> int32[n] cm
        self._pending: dict = {}
        self.stats_ms: dict = {}
        mb = settings.get("tile_cache_mb")
        if mb:
            cogmod.TILE_CACHE.resize(int(float(mb) * 1048576))

    # ----------------------------------------------------------------------------- lookup
    def _scopes(self):
        return ("city", "county") if self.domain_key == "city" else ("county",)

    def tileset(self, storm: str):
        if not self.enabled:
            return None
        with self._lock:
            if storm in self._sets:
                return self._sets[storm]
        found = None
        for scope in self._scopes():
            folder = self.base / scope / storm
            mp = folder / "cog_meta.json"
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                if not isinstance(meta, dict) or not any(str(k).endswith(".tif") for k in (meta.get("files") or {})):
                    continue
                ts = TileSet(scope, storm, folder, meta)
                ts.verified = bool(self.verified or meta.get("verified"))
                if ts.has_peak or ts.hours:
                    found = ts
                    break
            except (OSError, ValueError):
                continue
        with self._lock:
            self._sets[storm] = found
        return found

    def refresh(self):
        """Forget what was found (files may have arrived since)."""
        with self._lock:
            for ts in self._sets.values():
                if ts is not None:
                    ts.close()
            self._sets.clear()

    def available_storms(self, storms) -> list:
        return [s for s in storms if self.tileset(s) is not None]

    def describe(self, storm: str) -> str:
        ts = self.tileset(storm)
        if ts is None:
            return self.reason or f"no solver tiles for {storm} in {self.base}"
        what = "county solver run" if ts.scope == "county" else "city solver run"
        return f"{what}, {len(ts.hours)} hourly tiles" + (" + peak" if ts.has_peak else "")

    # ----------------------------------------------------------------------------- points
    def depth_lnglat(self, storm: str, lng: float, lat: float, which) -> DepthSample:
        which = norm_hour(which)
        ts = self.tileset(storm)
        if ts is None:
            return unavailable("tiles", storm, which, self.reason or f"no solver tiles for {storm}")
        c = ts.cog(which)
        if c is None:
            err = ts.errors.get(ts.file_for(which) or "", "")
            return unavailable("tiles", storm, which, "tile file missing" + (f" ({err})" if err else f" for hour {which}"), ts.scope)
        try:
            x, y = cogmod.lnglat_to_merc(float(lng), float(lat))
            status, raw = c.sample_merc(float(x), float(y))
        except Exception as e:
            return unavailable("tiles", storm, which, f"tile read failed: {type(e).__name__}: {e}", ts.scope)
        h = which if which == "peak" else ts.nearest_hour(which)
        return tile_sample(raw if raw is not None else 0, cogmod.ST_VALUE if status == "value" else cogmod.ST_NODATA, storm, h, ts.scope,
                           label=ts.label)

    def depth_profile(self, storm: str, lngs, lats, which) -> np.ndarray:
        which = norm_hour(which)
        out = np.full(np.asarray(lngs).shape, np.nan)
        ts = self.tileset(storm)
        c = ts.cog(which) if ts is not None else None
        if c is None:
            return out
        x, y = cogmod.lnglat_to_merc(np.asarray(lngs, np.float64), np.asarray(lats, np.float64))
        raw, st = c.sample_many(x, y)
        ok = st == cogmod.ST_VALUE
        out[ok] = raw[ok] / 100.0
        return out

    # ----------------------------------------------------------------------------- footprint index
    def index_for(self, storm: str):
        """The footprint index when it is ready and matches this storm's tile grid, else None."""
        ts = self.tileset(storm)
        if ts is None or self.index is None:
            return None
        return self.index if ts.grid_signature() == self.index_sig else None

    def start_index(self, get_geojson, storm_hint: str | None = None):
        """Build (or load from desktop/cache) the footprint pixel index on the tile grid, in a daemon thread.  `get_geojson` is a
        callable returning the building GeoJSON path (called in the worker).  Idempotent."""
        if not self.enabled or self.index_state in ("building", "ready"):
            return
        ts = None
        for s in ([storm_hint] if storm_hint else []) + list(self._sets):
            ts = self.tileset(s)
            if ts is not None:
                break
        if ts is None:
            return
        sig = ts.grid_signature()
        if sig is None:
            return
        self.index_state, self.index_msg = "building", "footprint index on the tile grid"
        threading.Thread(target=self._build_index, args=(get_geojson, sig), name="bgv-tile-index", daemon=True).start()

    def _build_index(self, get_geojson, sig):
        try:
            from . import buildings as bmod, cachelib
            path = Path(get_geojson())
            W, H, res, x0, y0 = sig
            frame = GridFrame(W, H, res, res, x0, y0 - H * res, x0 + W * res, y0)
            cp = cachelib.cache_path("cogindex", frame, [path], INDEX_VERSION)
            hit = cachelib.load_npz(cp)
            if hit is not None:
                a, _ = hit
                idx = bmod.PixelIndex(a["offsets"], a["pixels"], W, H)
            else:
                t0 = time.perf_counter()
                features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
                idx = bmod.buildings_from_features(features, frame, index_only=True)
                cachelib.save_npz(cp, {"offsets": idx.offsets, "pixels": idx.pixels}, {"n": idx.n, "built_s": time.perf_counter() - t0})
            self.index, self.index_sig = idx, sig
            self.index_state, self.index_msg = "ready", ""
        except Exception as e:
            self.index_state, self.index_msg = "error", f"{type(e).__name__}: {e}"
            try:
                from .. import applog
                applog.log().exception("tile footprint index failed")
            except Exception:
                pass

    # ----------------------------------------------------------------------------- buildings
    def building_depth(self, bid: int, storm: str, which) -> DepthSample:
        which = norm_hour(which)
        ts = self.tileset(storm)
        if ts is None:
            return unavailable("tiles", storm, which, self.reason or f"no solver tiles for {storm}")
        c = ts.cog(which)
        if c is None:
            return unavailable("tiles", storm, which, f"tile file missing for hour {which}", ts.scope)
        idx = self.index_for(storm)
        if idx is None:
            msg = {"building": "footprint index is being prepared", "error": f"footprint index failed: {self.index_msg}"}.get(
                self.index_state, "footprint index not built yet")
            return unavailable("tiles", storm, which, msg, ts.scope)
        if not (0 <= bid < idx.n):
            return unavailable("tiles", storm, which, "building number out of range", ts.scope)
        pix = idx.pixels_of(int(bid))
        h = which if which == "peak" else ts.nearest_hour(which)
        if pix.size == 0:
            return DepthSample("nodata", None, None, None, "tiles", ts.label, storm, h, ts.scope, "footprint outside the tile grid")
        raw, st = c.sample_pixels(pix % idx.width, pix // idx.width)
        good = st == cogmod.ST_VALUE
        if not good.any():
            return DepthSample("nodata", None, None, None, "tiles", ts.label, storm, h, ts.scope, "footprint outside the modelled area")
        v = int(raw[good].max())
        return tile_sample(v, cogmod.ST_VALUE, storm, h, ts.scope, f"max over {int(good.sum())} footprint cells", label=ts.label)

    def building_depths(self, storm: str, which, block: bool = False):
        """int32[n] footprint-maximum depth in cm (-1 = no data) for every building, or None while it is being computed in a
        worker (or when tiles / index are missing).  Cached (8 most recent)."""
        which = norm_hour(which)
        ts = self.tileset(storm)
        if ts is None:
            return None
        hour = which if which == "peak" else ts.nearest_hour(which)
        if hour is None:
            return None
        key = (storm, hour)
        with self._lock:
            if key in self._stats:
                self._stats.move_to_end(key)
                return self._stats[key]
            pend = self._pending.get(key)
        idx = self.index_for(storm)
        if idx is None:
            return None
        if pend is None:
            ev = threading.Event()
            with self._lock:
                self._pending[key] = ev
            th = threading.Thread(target=self._compute_stats, args=(key, ts, hour, idx, ev), name="bgv-tile-stats", daemon=True)
            th.start()
            pend = ev
        if block:
            pend.wait(120.0)
            with self._lock:
                return self._stats.get(key)
        return None

    def _compute_stats(self, key, ts, hour, idx, ev):
        try:
            t0 = time.perf_counter()
            c = ts.cog(hour)
            if c is None:
                return
            arr = footprint_max(c, idx)
            with self._lock:
                self._stats[key] = arr
                while len(self._stats) > 8:
                    self._stats.popitem(last=False)
            self.stats_ms[key] = (time.perf_counter() - t0) * 1000.0
        except Exception:
            try:
                from .. import applog
                applog.log().exception("tile footprint statistics failed")
            except Exception:
                pass
        finally:
            with self._lock:
                self._pending.pop(key, None)
            ev.set()

    def close(self):
        self.refresh()


def footprint_max(c: cogmod.Cog, idx) -> np.ndarray:
    """Per-building maximum over its footprint cells of one tile raster: int32 cm, -1 where the building has no cell or all of its
    cells are no-data."""
    n = idx.n
    out = np.full(n, -1, np.int32)
    if idx.pixels.size == 0:
        return out
    raw, st = c.sample_pixels(idx.pixels % idx.width, idx.pixels // idx.width)
    valid = st == cogmod.ST_VALUE
    vals = np.where(valid, raw, 0).astype(np.int32)
    counts = np.diff(idx.offsets)
    has = counts > 0
    starts = idx.offsets[:-1][has]
    mx = np.maximum.reduceat(vals, starts)
    nv = np.add.reduceat(valid.astype(np.int32), starts)
    res = np.where(nv > 0, mx, -1)
    out[np.flatnonzero(has)] = res
    return out


def tier_counts_cm(cm: np.ndarray) -> dict:
    """Flooded count and tier counts of a per-building cm array (the viewer's tiers: >= 10, 30, 60, 120 cm)."""
    e = TIER_EDGES_CM
    return {"flooded": int((cm >= e[0]).sum()), "shallow": int(((cm >= e[0]) & (cm < e[1])).sum()),
            "moderate": int(((cm >= e[1]) & (cm < e[2])).sum()), "deep": int(((cm >= e[2]) & (cm < e[3])).sum()),
            "severe": int((cm >= e[3]).sum()), "nodata": int((cm < 0).sum())}
