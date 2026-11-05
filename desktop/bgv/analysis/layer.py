"""AnalysisLayer: owns the state and the background services of the analysis tools (UI: bgv/ui/panels_analysis.py; GPU drawing of the
compare modes: bgv/gpu/layers/compare.py).  `draws = False`: nothing is drawn into the scene pass from here.

    L = app.get_layer("analysis")
    L.st                AnalysisState (persisted: units, tab, depth-source mode, compare settings)
    L.source            HybridDepthSource (tiles for the county, flow grid for the city / as fallback)
    L.measures          list of Measure (session only), L.tool the active map tool
    L.request_profile() / L.profile                 background cross-section of the drawn line
    L.series(target)                                background depth-vs-hour series of every storm
    L.timeline()                                    FloodTimeline of the selected storm (tiles: background full-resolution pass, cached)
    L.cmp               CompareController (storm B, swipe / difference settings, background grids + statistics)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..gpu.layers import Layer
from . import compare as cmpmod
from . import geodesy, profile as profmod, series as seriesmod, timeline as tlmod
from .compare import CompareJob as LatestJob
from .depth_source import make_depth_source

TOOLS = ("none", "measure_line", "measure_area", "profile", "point")
TABS = ("measure", "profile", "point", "timeline", "compare")
COMPARE_MODES = ("off", "swipe", "diff")


@dataclass
class Measure:
    kind: str                                  # 'line' | 'area'
    pts: list = field(default_factory=list)    # world (x, y)
    done: bool = False

    def lnglat(self, frame):
        a = np.asarray(self.pts, np.float64).reshape(-1, 2)
        return geodesy.world_to_lnglat(frame, a[:, 0], a[:, 1])

    def length_m(self, frame, closed: bool = False) -> float:
        if len(self.pts) < 2:
            return 0.0
        lng, lat = self.lnglat(frame)
        if closed:
            lng, lat = np.append(lng, lng[0]), np.append(lat, lat[0])
        return geodesy.path_length_m(lng, lat)

    def area_m2(self, frame) -> float:
        if len(self.pts) < 3:
            return 0.0
        lng, lat = self.lnglat(frame)
        return geodesy.polygon_area_m2(lng, lat)


@dataclass
class AnalysisState:
    imperial: bool = False
    tab: str = "measure"
    open: bool = False
    depth_source: str = "auto"                 # 'auto' (tiles where they exist) | 'grid'
    win: list = field(default_factory=lambda: [-1.0, -1.0, 640.0, 500.0])   # x, y (-1 = default placement), w, h
    mode: str = "off"                          # compare: off | swipe | diff
    b_storm: str = ""
    b_same_hour: bool = True
    b_hour: float = 12.0
    swipe: float = 0.5
    vertical: bool = True
    range_m: float = 2.0
    opacity: float = 0.88
    show_labels: bool = True

    PERSISTED = ("imperial", "tab", "open", "depth_source", "win", "mode", "b_storm", "b_same_hour", "b_hour", "swipe",
                 "vertical", "range_m", "opacity", "show_labels")

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in ((k, getattr(self, k)) for k in self.PERSISTED)}

    def load(self, d) -> "AnalysisState":
        if not isinstance(d, dict):
            return self

        def clamp(v, lo, hi, default):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            return default if v != v else min(max(v, lo), hi)
        for k in ("imperial", "open", "b_same_hour", "vertical", "show_labels"):
            if k in d:
                setattr(self, k, bool(d[k]))
        if d.get("tab") in TABS:
            self.tab = d["tab"]
        if d.get("depth_source") in ("auto", "grid"):
            self.depth_source = d["depth_source"]
        if d.get("mode") in COMPARE_MODES:
            self.mode = d["mode"]
        if isinstance(d.get("b_storm"), str):
            self.b_storm = d["b_storm"][:24]
        w = d.get("win")
        if isinstance(w, (list, tuple)) and len(w) == 4:
            self.win = [clamp(w[0], -1.0, 8000.0, -1.0), clamp(w[1], -1.0, 8000.0, -1.0), clamp(w[2], 380.0, 1400.0, 640.0),
                        clamp(w[3], 260.0, 1000.0, 500.0)]
        self.b_hour = clamp(d.get("b_hour", self.b_hour), 0.0, 24.0, self.b_hour)
        self.swipe = clamp(d.get("swipe", self.swipe), 0.02, 0.98, self.swipe)
        self.range_m = clamp(d.get("range_m", self.range_m), 0.25, 8.0, self.range_m)
        self.opacity = clamp(d.get("opacity", self.opacity), 0.2, 1.0, self.opacity)
        return self


class CompareController:
    """What the two compare modes show.  A = the viewer's storm at its hour, B = `b_storm` at the same hour or at its own hour."""

    def __init__(self, layer: "AnalysisLayer"):
        self.layer = layer
        self.job = LatestJob()                    # grids + packed delta
        self.stats_job = LatestJob()              # flooded buildings / roads of A and B
        self.area_job = LatestJob()               # native-resolution area / volume statistics (tiles)
        self.key = None
        self.pending = None                       # (key, float16[H, W, 2], bool chunk table) for the GPU layer to upload
        self.shown_key = None                     # what the GPU layer has on screen
        self.grid_stats = None                    # (key, CompareStats)
        self.status = ""
        self.error = ""
        self.view_rect = (0.0, 0.0, 1.0, 1.0)     # logical px of the 3D view (set by the UI every frame)
        self.win_size = (1.0, 1.0)
        self._prov = None
        self._prov_key = None

    # ------------------------------------------------------------------------------------------ pair
    @property
    def st(self):
        return self.layer.st

    def a_state(self, app):
        from ..ui.units import snapshot_hour
        return app.storm.name, snapshot_hour(app.t)

    def b_state(self, app):
        from ..ui.units import snapshot_hour
        hb = snapshot_hour(app.t) if self.st.b_same_hour else int(min(max(round(self.st.b_hour), 0), 24))
        return self.b_name(app), hb

    def b_name(self, app) -> str:
        names = list(app.storms_available)
        return self.st.b_storm if self.st.b_storm in names else (names[-1] if names else "")

    def b_time(self, app) -> float:
        return float(app.t) if self.st.b_same_hour else float(self.st.b_hour)

    def same_frame(self, app, b: str) -> bool:
        try:
            return self.layer.source.grid.storm_obj(b).frame == app.terrain.frame
        except Exception:
            return False

    # ------------------------------------------------------------------------------------------ per frame
    def update(self, app):
        if self.st.mode == "off":
            self.status = ""
            return
        a, ha = self.a_state(app)
        b, hb = self.b_state(app)
        src = self.layer.source
        key = (a, ha, b, hb, src.source_key(a), src.source_key(b), id(app.terrain))
        if self.st.mode == "diff" and key != self.key:
            self.key = key
            self.status = "computing difference..."
            self.error = ""
            prov = self._provider(app)
            frame = app.terrain.frame
            scene = self.layer.scene()

            def job():
                ga, gb = prov.grid(a, ha), prov.grid(b, hb)
                from ..data.flow import block_max
                from ..gpu.flowtex import CHUNK_TEXELS
                packed = cmpmod.delta_pack(ga, gb)
                top = np.maximum(ga, gb)
                wet = block_max((top > 0).view(np.uint8), CHUNK_TEXELS) > 0
                st = cmpmod.compare_grids(ga, gb, frame)
                return key, packed, wet, st
            self.job.request(key, job)
        r = self.job.result
        if r is not None and r[0] == self.key and self.pending is None and self.shown_key != r[0]:
            self.pending = (r[0], r[1], r[2])
            self.grid_stats = (r[0], r[3])
            self.status = ""
        if self.job.error and self.st.mode == "diff":
            self.error = self.job.error
        if self.st.mode == "diff" and src.uses_tiles(a) and src.uses_tiles(b):
            akey = ("area", a, ha, b, hb)
            if akey != self.area_job.key:
                ts_a, ts_b, frame = src.tiles.storm(a), src.tiles.storm(b), app.terrain.frame
                self.area_job.request(akey, lambda: (akey, cmpmod.tile_compare_stats(ts_a.raster(ha), ts_b.raster(hb), frame)))
        # building / road numbers (both modes): background, newest pair wins
        sc = self.layer.scene()
        if sc is not None and (self.st.mode in ("swipe", "diff")):
            skey = ("state", a, ha, b, hb, src.source_key(a), src.source_key(b))
            if skey != self.stats_job.key:
                pts = self._scene_points(sc)
                self.stats_job.request(skey, lambda: (skey, cmpmod.state_stats(src, a, float(ha), pts),
                                                      cmpmod.state_stats(src, b, float(hb), pts)))

    def _provider(self, app):
        k = (id(app.terrain), id(self.layer.source), self.layer.source.mode)
        if self._prov is None or self._prov_key != k:
            self._prov = cmpmod.DepthGridProvider(self.layer.source, app.terrain.frame)
            self._prov_key = k
        return self._prov

    def _scene_points(self, sc):
        b, r, f = sc
        k = (id(b), id(r))
        if getattr(self, "_pts_key", None) != k:
            self._pts = cmpmod.ScenePoints(b, r, f)
            self._pts_key = k
        return self._pts

    def area_stats(self):
        """(CompareStats, resolution text) of the shown pair: the native tile statistics when ready, else the flow-grid-resolution ones."""
        r = self.area_job.result
        if r is not None and r[0] == self.area_job.key and self.area_job.key[1:] == (
                self.key[0], self.key[1], self.key[2], self.key[3]):
            return r[1], "tile resolution"
        gs = self.grid_stats
        if gs is not None and gs[0] == self.shown_key and self.shown_key is not None:
            return gs[1], "flow-grid cell size"
        return None

    def building_stats(self):
        r = self.stats_job.result
        return r if r is not None and r[0] == self.stats_job.key else None

    def close(self):
        self.job.close()
        self.stats_job.close()
        self.area_job.close()


class AnalysisLayer(Layer):
    name = "analysis"
    title = "Analysis tools"
    draws = False
    has_ui = False

    def __init__(self):
        self.st = AnalysisState()
        self.app = None
        self.source = None
        self.measures: list[Measure] = []
        self.tool = "none"
        self.profile_line: list = []                 # world (x, y) vertices of the cross-section line
        self.profile = None                          # profile.Profile of the newest finished job
        self.profile_key = None
        self._profile_cache: dict = {}
        self._prof_job = None
        self.profile_error = ""
        self.target = None                           # series.Target of the picked point / building
        self.series_worker = None
        self.series_result = None
        self.timeline_worker = None
        self._tl_mem: dict = {}
        self._ground = None
        self.cmp: CompareController | None = None
        self.edit_vertex = None
        self.hover_world = None

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        self.app = app
        self.source = make_depth_source(app)
        self.source.mode = self.st.depth_source
        self.series_worker = seriesmod.SeriesWorker(lambda: self.source)
        self.timeline_worker = tlmod.TimelineWorker(self.scene, lambda: self.source)
        self._prof_job = LatestJob()
        self.cmp = CompareController(self)
        app.analysis = self

    def shutdown(self, app):
        for w in (self.series_worker, self.timeline_worker, self._prof_job, self.cmp):
            try:
                if w is not None:
                    w.close()
            except Exception:
                pass
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
        if getattr(app, "analysis", None) is self:
            app.analysis = None

    def on_storm(self, app):
        self.cmp.key = None
        self.cmp.shown_key = None if self.st.mode != "diff" else self.cmp.shown_key
        self.profile_key = None
        self.series_result = None

    def update(self, app, dt):
        self.cmp.update(app)

    # ------------------------------------------------------------------------------------------ persistence
    def save_state(self):
        return self.st.to_dict()

    def load_state(self, d):
        self.st.load(d)

    # ------------------------------------------------------------------------------------------ helpers
    @property
    def frame(self):
        return self.app.terrain.frame

    def scene(self):
        """(buildings, roads, frame) once the scene data is ready, else None."""
        sc = getattr(self.app, "_scene_data", None)
        if sc is None or getattr(sc, "state", "") != "ready":
            return None
        return sc.buildings, sc.roads, sc.frame

    def set_depth_mode(self, mode: str):
        if mode in ("auto", "grid") and mode != self.st.depth_source:
            self.st.depth_source = mode
            self.source.mode = mode
            self._profile_cache.clear()
            self.profile_key = None
            self.series_result = None
            self.cmp.key = None
            self.cmp.shown_key = None
            self.cmp.stats_job.key = None
            self.cmp.stats_job.result = None

    def depth_label(self, storm: str | None = None) -> str:
        return self.source.label(storm or self.app.storm.name)

    def ground(self):
        if self._ground is None or getattr(self._ground, "_terrain_id", None) != id(self.app.terrain):
            self._ground = profmod.make_ground(self.app)
            self._ground._terrain_id = id(self.app.terrain)
        return self._ground

    # ------------------------------------------------------------------------------------------ measure
    def finish_measure(self):
        if self.measures and not self.measures[-1].done:
            m = self.measures[-1]
            need = 2 if m.kind == "line" else 3
            if len(m.pts) >= need:
                m.done = True
            else:
                self.measures.pop()
        if self.tool in ("measure_line", "measure_area"):
            self.tool = "none"

    def clear_measures(self):
        self.measures.clear()

    def add_measure_point(self, x: float, y: float):
        kind = "line" if self.tool == "measure_line" else "area"
        if not self.measures or self.measures[-1].done or self.measures[-1].kind != kind:
            self.measures.append(Measure(kind))
        self.measures[-1].pts.append((float(x), float(y)))

    # ------------------------------------------------------------------------------------------ profile
    def request_profile(self, hour: int | None = None):
        """(Re)compute the cross-section of `profile_line` for the viewer's storm at an integer hour (background, cached)."""
        if len(self.profile_line) < 2:
            self.profile = None
            return
        from ..ui.units import snapshot_hour
        app = self.app
        h = snapshot_hour(app.t) if hour is None else int(hour)
        storm = app.storm.name
        key = (tuple(self.profile_line), storm, h, self.source.source_key(storm))
        if key == self.profile_key:
            return
        self.profile_key = key
        hit = self._profile_cache.get(key)
        if hit is not None:
            self.profile = hit
            return
        line = np.asarray(self.profile_line, np.float64)
        frame, src, ground = self.frame, self.source, self.ground()

        def job():
            return key, profmod.build_profile(line, frame, src, storm, float(h), ground)
        self._prof_job.request(key, job)

    def poll_profile(self):
        r = self._prof_job.result
        if r is not None and r[0] == self.profile_key and self.profile is not r[1]:
            self.profile = r[1]
            self._profile_cache[r[0]] = r[1]
            while len(self._profile_cache) > 40:
                self._profile_cache.pop(next(iter(self._profile_cache)))
        self.profile_error = self._prof_job.error

    # ------------------------------------------------------------------------------------------ series
    def set_target_point(self, lng: float, lat: float):
        self.target = seriesmod.point_target(lng, lat)
        self._start_series()

    def set_target_building(self, i: int):
        sc = self.scene()
        if sc is None:
            return
        b, _, f = sc
        idx = sig = None
        try:                                               # the building card's all-touched tile index, when ready
            from ..data import depthapi
            svc = depthapi.service(self.app)
            if svc.index_state == "ready":
                idx, sig = svc.index, svc.index_sig
        except Exception:
            pass
        self.target = seriesmod.building_target(b, int(i), f, tile_index=idx, tile_sig=sig)
        self._start_series()

    def _start_series(self):
        if self.target is None:
            return
        self.series_result = self.series_worker.request(self.target, list(self.app.storms_available), first=self.app.storm.name)

    def series(self):
        if self.target is not None and (self.series_result is None or self.series_result.target.key != self.target.key):
            self._start_series()
        return self.series_result

    # ------------------------------------------------------------------------------------------ timeline
    def timeline(self) -> "tlmod.FloodTimeline | None":
        app = self.app
        storm = app.storm.name
        if self.source.uses_tiles(storm):
            return self.timeline_worker.get(storm)
        hub = getattr(app, "_stats_hub", None)
        key = (storm, "grid", self.source.source_key(storm))
        mem = self._tl_mem.get(key)
        now = time.perf_counter()
        last = getattr(self, "_tl_last", (None, 0.0, None))
        if last[0] == key and now - last[1] < 0.25 and last[2] is not None:
            return last[2]
        live = tlmod.grid_timeline(hub, storm, self.source.label(storm))
        if live.complete:
            self._tl_mem[key] = live
        tl = live if (live.complete or mem is None or live.done >= mem.done) else mem
        self._tl_last = (key, now, tl)
        return tl
