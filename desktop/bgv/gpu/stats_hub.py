"""Live flood statistics: ONE place that computes the numbers for the displayed snapshot.

Install (idempotent; BuildingsLayer / RoadsLayer do it in init, a UI agent may call it too):

    hub = stats_hub.get_hub(app)          # sets app.stats_hook (chained with any hook that was there before)

Published every frame by the hook (main thread, <= 1 ms on a cache hit; a miss only reads the cache and asks the worker):

    app.flood = {
        "ready":        bool            True when the numbers below belong to a computed snapshot
        "pending":      bool            True while the snapshot for `wanted_hour` is still being computed (the numbers are
                                        then those of the previous snapshot of the SAME storm, `hour` says which)
        "storm":        str             storm name the numbers belong to
        "hour":         int             snapshot hour the numbers belong to  ("as of hour N")
        "wanted_hour":  int             floor(t + 0.5) of the displayed continuous time
        "buildings":    data.stats.BuildingStats | None   (depth_m, tier, flooded, order ... arrays, see data/stats.py)
        "roads":        data.stats.RoadStats | None
        "grid":         data.stats.GridSummary | None
        "tier_counts":  {"shallow": int, "moderate": int, "deep": int, "severe": int}
        "flooded_count": int,  "n_buildings": int,  "n_buildings_in_grid": int
        "wet_area_km2": float, "max_depth_m": float, "road_pct": float, "road_flooded_km": float
        "source":       "tiers" | "rasters"
        "version":      int             increases every time the published snapshot changes (cheap change detection)
    }
    app.timeline = {                                  # whole storm, precomputed in the background, NaN = not computed yet
        "storm": str, "hours": [0..24], "flooded": [..], "wet_area_km2": [..], "max_depth_m": [..], "roads_pct": [..],
        "tiers": [[shallow, moderate, deep, severe] per hour], "done": int (hours computed), "complete": bool }
    app.stats                                         # summary lines for the sidebar (only the keys listed in STAT_KEYS)

Rules (identical to fluid-stats.js, see data/stats.py): hour = floor(t + 0.5) (JavaScript Math.round, clamped 0..24; the
sidebar's python round() differs only for t exactly x.5), building depth = max over its all-touched pixels, flooded >=
0.1 m, tiers [0.1,0.3) [0.3,0.6) [0.6,1.2) >= 1.2, wet area = cells >= 0.02 m, roads = share of 10 m samples >= 0.1 m.

Threading
---------
* The expensive part (gunzip of the hour, numba kernels, ~0.4 s per hour) runs in ONE worker thread, which computes the
  hours of the current storm nearest to the displayed one first and then the rest (this is also the timeline).  The main
  thread therefore never decodes anything: a snapshot that is not ready yet leaves the previous numbers on screen with
  `pending = True` (a few hundred ms at worst; every hour is usually ready within ~10 s of loading a storm).
* Results are immutable once stored.  `_cache` is guarded by `_lock`; `app.timeline` / `app.flood` are replaced as a whole
  (copy on write), so a reader never sees a half-updated dict.  Only the main thread assigns `app.flood` / `app.stats`.
* The worker never touches wgpu.  It holds the numba parallel lock (data.stats) only inside the kernels.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .scene_data import get_scene

N_HOURS = 25
TIER_NAMES = ("shallow", "moderate", "deep", "severe")
STAT_KEYS = ("Snapshot", "Flooded buildings", "Wet area", "Max depth", "Roads flooded", "Scene data")
MAX_STORMS = 3


def snapshot_hour(t: float) -> int:
    """JS Math.round on the continuous time, clamped to the 0..24 hour layers."""
    return int(min(max(math.floor(float(t) + 0.5), 0), N_HOURS - 1))


@dataclass
class Snapshot:
    storm: str
    hour: int
    buildings: object
    roads: object
    grid: object
    t_compute: float = 0.0

    @property
    def tier_counts(self) -> dict:
        return {n: int(c) for n, c in zip(TIER_NAMES, self.buildings.tier_counts)}


def compute_snapshot(storm, hour: int, scene) -> Snapshot:
    """Pure function used by the worker (and the tests): statistics of one hour grid of `storm`."""
    from ..data import stats
    t0 = time.perf_counter()
    rgba = None if storm.is_zero(hour) else storm.load_hour(hour, cache=False)
    b = stats.evaluate_buildings(rgba, scene.buildings.index)
    time.sleep(0.002)                      # numba kernels hold the GIL until they return: let the render thread in between
    r = stats.evaluate_roads(rgba, scene.roads)
    time.sleep(0.002)
    g = stats.summarize_grid(rgba, scene.frame)
    return Snapshot(storm.name, int(hour), b, r, g, time.perf_counter() - t0)


class StatsHub:
    def __init__(self, app):
        self.app = app
        self.scene = get_scene(app)
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, dict[int, Snapshot]]" = OrderedDict()     # storm -> hour -> Snapshot
        self._timelines: dict[str, dict] = {}
        self._storm = None
        self._gen = 0
        self._bad_hours: dict = {}
        self._want = 12
        self._published = None            # Snapshot currently shown in app.flood
        self._version = 0
        self._wake = threading.Event()
        self._closed = False
        self.last_call_ms = 0.0
        self.max_call_ms = 0.0
        self.compute_ms: list[float] = []
        self.prev_hook = None
        self.throttle_s = 0.18
        self._warmed = False
        self._thread = threading.Thread(target=self._worker, name="bgv-stats", daemon=True)
        self._thread.start()
        app.flood = {"ready": False, "pending": True, "storm": getattr(app.storm, "name", ""), "hour": snapshot_hour(app.t),
                     "wanted_hour": snapshot_hour(app.t), "buildings": None, "roads": None, "grid": None,
                     "tier_counts": {n: 0 for n in TIER_NAMES}, "flooded_count": 0, "n_buildings": 0,
                     "n_buildings_in_grid": 0, "wet_area_km2": 0.0, "max_depth_m": 0.0, "road_pct": 0.0,
                     "road_flooded_km": 0.0, "source": getattr(app.storm, "source", ""), "version": 0}
        if getattr(app, "timeline", None) is None:
            app.timeline = self._empty_timeline(getattr(app.storm, "name", ""))
        from . import selection
        selection.ensure(app)

    # ------------------------------------------------------------------------------------------ main thread
    def __call__(self, app):
        t0 = time.perf_counter()
        if self.prev_hook is not None:
            try:
                self.prev_hook(app)
            except Exception as e:
                app.report_error("stats_hook(prev)", e)
                self.prev_hook = None
        storm = app.storm
        hour = snapshot_hour(app.t)
        if getattr(app, "_flight", None) is not None:
            from . import selection
            selection.update_flight(app)                # camera flight started by selection.fly_to_building
        if storm is not self._storm:
            self._switch(storm)
        self._want = hour
        scene = self.scene
        if scene.state == "ready":
            snap = self._get(storm.name, hour)
            if snap is None and storm.is_zero(hour):
                snap = self._store(compute_snapshot(storm, hour, scene))          # dry hour: no decode, ~1 ms
            if snap is not None:
                if snap is not self._published:
                    self._publish(app, snap, hour)
                elif app.flood.get("wanted_hour") != hour or app.flood.get("pending"):
                    app.flood = dict(app.flood, wanted_hour=hour, pending=False)
            else:
                f = app.flood
                if not f.get("pending") or f.get("wanted_hour") != hour:
                    app.flood = dict(f, pending=True, wanted_hour=hour)
                    self._stat_lines(app, f.get("hour"), True)
                self._wake.set()
        else:
            self._scene_lines(app)
        self.last_call_ms = (time.perf_counter() - t0) * 1000.0
        self.max_call_ms = max(self.max_call_ms, self.last_call_ms)

    def _switch(self, storm):
        self._storm = storm
        self._gen += 1
        self._published = None
        self._version += 1
        n = getattr(self.scene.buildings, "n", 0)
        app = self.app
        app.flood = {"ready": False, "pending": True, "storm": storm.name, "hour": snapshot_hour(app.t),
                     "wanted_hour": snapshot_hour(app.t), "buildings": None, "roads": None, "grid": None,
                     "tier_counts": {k: 0 for k in TIER_NAMES}, "flooded_count": 0, "n_buildings": n,
                     "n_buildings_in_grid": 0, "wet_area_km2": 0.0, "max_depth_m": 0.0, "road_pct": 0.0,
                     "road_flooded_km": 0.0, "source": getattr(storm, "source", ""), "version": self._version}
        app.timeline = self._timeline_for(storm.name)
        self._wake.set()

    def _publish(self, app, snap: Snapshot, wanted: int):
        self._published = snap
        self._version += 1
        b, r, g = snap.buildings, snap.roads, snap.grid
        sc = self.scene
        app.flood = {
            "ready": True, "pending": False, "storm": snap.storm, "hour": snap.hour, "wanted_hour": wanted,
            "buildings": b, "roads": r, "grid": g, "tier_counts": snap.tier_counts, "flooded_count": int(b.flooded_count),
            "n_buildings": int(sc.buildings.n), "n_buildings_in_grid": int(sc.buildings.in_grid.sum()),
            "wet_area_km2": float(g.wet_area_km2), "max_depth_m": float(g.max_depth_m), "road_pct": float(r.flooded_pct),
            "road_flooded_km": float(r.flooded_length_m) / 1000.0, "source": getattr(app.storm, "source", ""),
            "version": self._version}
        self._stat_lines(app, snap.hour, False)

    def _stat_lines(self, app, hour, pending: bool):
        f = app.flood
        st = app.stats
        st.pop("Scene data", None)
        if not f.get("ready") or f.get("buildings") is None:
            st["Snapshot"] = "computing..."
            return
        st["Snapshot"] = f"hour {hour}" + (" (updating)" if pending else "") + f"  [{f['storm']}]"
        st["Flooded buildings"] = f"{f['flooded_count']:,} of {f['n_buildings_in_grid']:,}"
        st["Wet area"] = f"{f['wet_area_km2']:.2f} km2"
        st["Max depth"] = f"{f['max_depth_m']:.2f} m" + (" (tiers)" if f["source"] == "tiers" else "")
        st["Roads flooded"] = f"{f['road_pct']:.1f} %"

    def _scene_lines(self, app):
        sc = self.scene
        if sc.state == "error":
            app.stats["Scene data"] = "failed: " + str(sc.error)[:60]
        else:
            app.stats["Scene data"] = f"loading {int(sc.frac * 100)}%  {sc.message}"[:60]

    # ------------------------------------------------------------------------------------------ cache
    def _get(self, storm: str, hour: int):
        with self._lock:
            d = self._cache.get(storm)
            return None if d is None else d.get(hour)

    def _store(self, snap: Snapshot) -> Snapshot:
        with self._lock:
            d = self._cache.setdefault(snap.storm, {})
            self._cache.move_to_end(snap.storm)
            d[snap.hour] = snap
            while len(self._cache) > MAX_STORMS:
                old, _ = self._cache.popitem(last=False)
                self._timelines.pop(old, None)
            tl = self._build_timeline(snap.storm, d)
            self._timelines[snap.storm] = tl
        if self._storm is not None and snap.storm == self._storm.name:
            self.app.timeline = tl                      # atomic reference swap, see module docstring
        return snap

    def get_snapshot(self, storm: str, hour: int):
        """Computed snapshot or None (any thread)."""
        return self._get(storm, hour)

    @staticmethod
    def _empty_timeline(name: str) -> dict:
        nan = [float("nan")] * N_HOURS
        return {"storm": name, "hours": list(range(N_HOURS)), "flooded": list(nan), "wet_area_km2": list(nan),
                "max_depth_m": list(nan), "roads_pct": list(nan), "tiers": [[float("nan")] * 4 for _ in range(N_HOURS)],
                "done": 0, "complete": False}

    def _timeline_for(self, name: str) -> dict:
        with self._lock:
            return self._timelines.get(name) or self._empty_timeline(name)

    def _build_timeline(self, storm: str, d: dict) -> dict:
        tl = self._empty_timeline(storm)
        for h, s in d.items():
            tl["flooded"][h] = float(s.buildings.flooded_count)
            tl["wet_area_km2"][h] = float(s.grid.wet_area_km2)
            tl["max_depth_m"][h] = float(s.grid.max_depth_m)
            tl["roads_pct"][h] = float(s.roads.flooded_pct)
            tl["tiers"][h] = [float(c) for c in s.buildings.tier_counts]
        tl["done"] = len(d)
        tl["complete"] = len(d) >= N_HOURS
        return tl

    # ------------------------------------------------------------------------------------------ worker
    @staticmethod
    def _lower_priority():
        """Windows: run the statistics worker below the render thread so a background timeline fill never competes with it
        for a core (no-op elsewhere / on failure)."""
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)          # THREAD_PRIORITY_BELOW_NORMAL
        except Exception:
            pass

    def _worker(self):
        self._lower_priority()
        while not self._closed:
            self._wake.wait(0.25)
            if self._closed:
                return
            sc = self.scene
            if sc.state != "ready" or self._storm is None:
                if sc.state == "error":
                    return
                # the event stays set while the scene loads (the render thread sets it every frame): without a real sleep
                # this loop spun ~9 million times in 20 s and starved every other thread of the GIL
                self._wake.clear()
                time.sleep(0.02)
                continue
            if not self._warmed:
                try:
                    from ..data import stats
                    stats.warmup()
                except Exception as e:
                    self.app.report_error("stats warmup", e)
                self._warmed = True
            storm, gen = self._storm, self._gen
            self._wake.clear()
            while not self._closed and gen == self._gen:
                with self._lock:
                    have = set(self._cache.get(storm.name, {}))
                have |= self._bad_hours.setdefault(id(storm), set())       # unreadable hours are tried once, not forever
                todo = [h for h in range(N_HOURS) if h not in have]
                if not todo:
                    break
                want = self._want
                h = min(todo, key=lambda x: (abs(x - want), x))
                try:
                    snap = compute_snapshot(storm, h, sc)
                except Exception as e:
                    self.app.report_message(f"stats hour {h}: {type(e).__name__}: {e}"[:200], toast=False)   # flow reports it
                    self._bad_hours.setdefault(id(storm), set()).add(h)
                    time.sleep(0.05)
                    continue
                if gen != self._gen:
                    break
                self.compute_ms.append(snap.t_compute * 1000.0)
                self._store(snap)
                if h != self._want:
                    time.sleep(self.throttle_s)      # background timeline hours trickle in: never a burst of GIL stalls

    # ------------------------------------------------------------------------------------------ misc
    def wait_ready(self, timeout: float = 60.0, hour: int | None = None) -> bool:
        """Block (tools / tests only) until the scene is loaded and the snapshot of `hour` (default: the wanted one)
        is computed.  Returns False on timeout."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            st = self.app.storm
            if self.scene.state == "error":
                return False
            self.__call__(self.app)                 # (re)registers the storm / wanted hour with the worker
            if self.scene.state == "ready":
                h = snapshot_hour(self.app.t) if hour is None else hour
                if self._get(st.name, h) is not None or st.is_zero(h):
                    return True
            time.sleep(0.01)
        return False

    def wait_timeline(self, timeout: float = 120.0) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.app.timeline.get("complete") and self.app.timeline.get("storm") == self.app.storm.name:
                return True
            self.__call__(self.app)
            time.sleep(0.05)
        return False

    def close(self):
        self._closed = True
        self._wake.set()


def get_hub(app) -> StatsHub:
    """Create (once per app) and install the hub as `app.stats_hook`."""
    hub = getattr(app, "_stats_hub", None)
    if hub is not None and hub.scene is get_scene(app) and not hub._closed:
        if app.stats_hook is not hub and app.stats_hook is not None:
            hub.prev_hook = app.stats_hook          # somebody installed a hook after us: run ours first, theirs after
            app.stats_hook = hub
        elif app.stats_hook is None:
            app.stats_hook = hub
        return hub
    if hub is not None:
        hub.close()
    hub = StatsHub(app)
    prev = app.stats_hook
    hub.prev_hook = prev if prev is not None and not isinstance(prev, StatsHub) else None
    app.stats_hook = hub
    app._stats_hub = hub
    return hub
