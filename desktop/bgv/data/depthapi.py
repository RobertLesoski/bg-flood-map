"""Stable depth-sampling API shared by every feature (see desktop/docs/EXTENSION_API.md).  Also installed as methods of `App`:

    app.depth_at(lng, lat, storm=None, hour=None, source="auto") -> DepthSample
    app.building_depth(bid, storm=None, hour=None, source="auto") -> DepthSample
    app.depth_profile(lngs, lats, storm=None, hour=None, source="auto") -> float64[n] metres (NaN = no data)
    app.tile_info(storm=None) -> TileInfo | None
    app.tiles                                        the TileService of the current domain (created on first use)

storm None = displayed storm; hour None = displayed hour (round(app.t)), 'peak' = maximum over the run; source 'tiles' | 'grid' | 'auto'.
"""
from __future__ import annotations

import numpy as np

from . import flow as flowdata
from .tiles import (GRID_LABEL, DepthSample, TileService, norm_hour, unavailable)

GRID_DRY_M = 0.02                                   # below this the flow grid is reported dry (its wet threshold)


class TileInfo:
    def __init__(self, ts, label, text):
        self.scope, self.storm, self.folder, self.hours, self.has_peak = ts.scope, ts.storm, ts.folder, sorted(ts.hours), ts.has_peak
        self.label, self.text, self.res_m, self.verified = label, text, ts.res_m, ts.verified


def service(app) -> TileService:
    """The tile service of the app's current domain (rebuilt when the domain changes)."""
    svc = getattr(app, "_tile_service", None)
    if svc is None or getattr(app, "_tile_service_key", None) != (app.domain.key, id(app.domain)):
        if svc is not None:
            svc.close()
        svc = TileService(app.root, app.domain, app.settings)
        app._tile_service, app._tile_service_key = svc, (app.domain.key, id(app.domain))
    return svc


def _storm_name(app, storm):
    return storm or (app.storm.name if app.storm is not None else "")


def _hour(app, hour):
    h = norm_hour(hour)
    return int(round(app.t)) if h is None else h


def _grid_storm(app, name):
    if app.storm is not None and app.storm.name == name:
        return app.storm
    cache = getattr(app, "_grid_storms", None)
    if cache is None:
        cache = app._grid_storms = {}
    s = cache.get(name)
    if s is None:
        for old in list(cache):                          # keep at most one extra storm open
            cache.pop(old).close()
        s = cache[name] = app.domain.open_storm(name)
    return s


def grid_depth_at(app, lng, lat, storm, hour) -> DepthSample:
    try:
        st = _grid_storm(app, storm)
        X, Y = app.terrain.frame.lnglat_to_world(lng, lat)
        if hour == "peak":
            best, valid = 0.0, False
            for h in st.hours:
                r = st.sample_t(float(h), X, Y)
                valid = valid or bool(r["valid"])
                best = max(best, float(r["depth"]))
            d, ok = best, valid
        else:
            r = st.sample_t(float(hour), X, Y)
            d, ok = float(r["depth"]), bool(r["valid"])
    except Exception as e:
        return unavailable("grid", storm, hour, f"flow-grid read failed: {type(e).__name__}: {e}")
    note = "preview from depth tiers (approximate)" if getattr(st, "source", "") == "tiers" else ""
    if not ok:
        return DepthSample("nodata", None, None, None, "grid", GRID_LABEL, storm, hour, "", note)
    if d < GRID_DRY_M:
        d = 0.0
    return DepthSample("dry" if d == 0.0 else "depth", d, d * 100.0, d * 3.28084, "grid", GRID_LABEL, storm, hour, "", note)


def depth_at(app, lng, lat, storm=None, hour=None, source="auto") -> DepthSample:
    storm, hour = _storm_name(app, storm), _hour(app, hour)
    svc = service(app)
    if source in ("tiles", "auto"):
        r = svc.depth_lnglat(storm, lng, lat, hour)
        if r.status != "unavailable" or source == "tiles":
            return r
    return grid_depth_at(app, lng, lat, storm, hour)


def grid_building_depth(app, bid, storm, hour) -> DepthSample:
    from ..ui import bridge
    b = bridge.buildings(app)
    if b is None or not (0 <= bid < b.n):
        return unavailable("grid", storm, hour, "buildings are not loaded")
    try:
        st = _grid_storm(app, storm)
        pix = b.index.pixels_of(int(bid))
        if pix.size == 0:
            return DepthSample("nodata", None, None, None, "grid", GRID_LABEL, storm, hour, "", "footprint outside the flow grid")
        hours = list(st.hours) if hour == "peak" else [None]
        best = 0.0
        for h in hours:
            if h is None:
                h0, h1, mix = flowdata.hour_pair(float(hour))
                grids = [(st.load_hour(h0), 1.0 - mix)] + ([(st.load_hour(h1), mix)] if mix > 0 else [])
            else:
                grids = [(st.load_hour(int(h)), 1.0)]
            code = 0.0
            for g, w in grids:
                flat = np.asarray(g).reshape(-1, 4)[pix]
                ok = flat[:, 3] >= 128
                code += w * (float(flat[ok, 0].max()) if ok.any() else 0.0)
            best = max(best, float(flowdata.decode_depth(code, st.depth_max)))
    except Exception as e:
        return unavailable("grid", storm, hour, f"flow-grid read failed: {type(e).__name__}: {e}")
    note = "max over the footprint's flow-grid cells" + ("; preview from depth tiers (approximate)" if getattr(st, "source", "") == "tiers" else "")
    return DepthSample("dry" if best < GRID_DRY_M else "depth", best, best * 100.0, best * 3.28084, "grid", GRID_LABEL,
                       storm, hour, "", note)


def building_depth(app, bid, storm=None, hour=None, source="auto") -> DepthSample:
    storm, hour = _storm_name(app, storm), _hour(app, hour)
    svc = service(app)
    if source in ("tiles", "auto"):
        r = svc.building_depth(int(bid), storm, hour)
        if r.status != "unavailable" or source == "tiles":
            return r
    return grid_building_depth(app, int(bid), storm, hour)


def depth_profile(app, lngs, lats, storm=None, hour=None, source="auto") -> np.ndarray:
    storm, hour = _storm_name(app, storm), _hour(app, hour)
    lngs, lats = np.asarray(lngs, np.float64), np.asarray(lats, np.float64)
    svc = service(app)
    if source in ("tiles", "auto") and svc.tileset(storm) is not None:
        return svc.depth_profile(storm, lngs, lats, hour)
    if source == "tiles":
        return np.full(lngs.shape, np.nan)
    out = np.full(lngs.shape, np.nan)
    for i, (a, b) in enumerate(zip(lngs.ravel(), lats.ravel())):
        r = grid_depth_at(app, float(a), float(b), storm, hour)
        if r.m is not None:
            out.flat[i] = r.m
    return out


def tile_info(app, storm=None):
    storm = _storm_name(app, storm)
    svc = service(app)
    ts = svc.tileset(storm)
    return None if ts is None else TileInfo(ts, ts.label, svc.describe(storm))


def install(cls):
    """Add the API as methods / a property of the App class (called once from bgv/app.py)."""
    cls.depth_at = lambda self, lng, lat, storm=None, hour=None, source="auto": depth_at(self, lng, lat, storm, hour, source)
    cls.building_depth = lambda self, bid, storm=None, hour=None, source="auto": building_depth(self, bid, storm, hour, source)
    cls.depth_profile = lambda self, lngs, lats, storm=None, hour=None, source="auto": depth_profile(self, lngs, lats, storm, hour, source)
    cls.tile_info = lambda self, storm=None: tile_info(self, storm)
    cls.tiles = property(service)
