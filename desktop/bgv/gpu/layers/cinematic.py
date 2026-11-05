"""CinematicLayer: runtime controller of the cinematic tools (bookmarks, tour, time-lapse, clip export).

It draws nothing and has no sidebar section (`draws = False`, `has_ui = False`); its panel, menu and captions are registered with
the UI hooks (`bgv/ui/hooks.py`) by `bgv/ui/panels_cinematic.py`, which this layer imports in `init`.  State (bookmarks per
domain, tour / time-lapse / export options) persists through the layer state in user_settings.json.

Everything that moves the camera or the hour does it from `update()` (called every frame after the camera step), and stops when
the user takes the camera (drag, wheel, keys): a tour is cancelled, a flight is cancelled, the time-lapse keeps going.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import Layer
from ... import tools_cinematic as tc
from ...data import depthtiles


@dataclass
class TourOptions:
    max_facilities: int = 10
    max_roads: int = 4
    types: list = field(default_factory=lambda: list(tc.DEFAULT_TYPES))
    only_wet: bool = False
    dwell_s: float = 6.0
    speed: float = 1.0
    escalate: bool = False                 # visit storms small -> large, one share of the stops each
    escalate_n: int = 3
    time_mode: str = "sweep"               # tc.TIME_MODES
    hide_ui: bool = False


@dataclass
class ExportOptions:
    kind: str = "time"                     # time | orbit | flight | tour
    size_idx: int = 1
    fps: int = 30
    seconds: float = 10.0
    orbit_deg: float = 90.0
    loops: float = 1.0
    mp4: bool = True
    keep_png: bool = True
    flight_a: int = 0
    flight_b: int = 1


SIZES = ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160))
FPS_CHOICES = (24, 30, 60)
YEAR_ORDER = ("2yr", "5yr", "10yr", "25yr", "50yr", "100yr")


class TourRuntime:
    def __init__(self):
        self.path: tc.TourPath | None = None
        self.plan: tc.TourPlan | None = None
        self.building = False
        self.error = ""
        self.active = False
        self.paused = False
        self.T = 0.0
        self.buffering = False
        self.info: dict = {}
        self.restore: dict | None = None
        self.state: tc.TourState | None = None
        self.opts: TourOptions | None = None


class CinematicLayer(Layer):
    name = "cinematic"
    title = "Cinematic tools"
    draws = False
    has_ui = False

    def __init__(self):
        self.store = tc.BookmarkStore()
        self.flight = tc.ViewFlight()
        self.lapse = tc.TimeLapse()
        self.tour_opts = TourOptions()
        self.exp_opts = ExportOptions()
        self.tour = TourRuntime()
        self.export = None                     # export_cinematic.ExportJob while a clip is being rendered
        self.panel_open = False
        self.tab = 0
        self._tab_request = None
        self.restore_time = False
        self.series = depthtiles.SeriesCache()
        self._sources: dict = {}
        self._build_lock = threading.Lock()
        self._after_build = None
        self.app = None
        self.last_message = ""
        self.message_t = 0.0

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        self.app = app
        from ...ui import panels_cinematic            # registers the panel / menu / overlay / key hooks (idempotent)
        panels_cinematic.register()

    def on_storm(self, app):
        self._close_sources()

    def shutdown(self, app):
        if self.export is not None:
            self.export.cancel()
        self.series.close()
        self._close_sources()
        self.app = None

    def _close_sources(self):
        for s in self._sources.values():
            try:
                s.close()
            except Exception:
                pass
        self._sources = {}

    # ------------------------------------------------------------------------------------------ persistence
    def save_state(self):
        o, e = self.tour_opts, self.exp_opts
        return {"bookmarks": self.store.to_dict(), "lapse": self.lapse.to_dict(), "panel_open": self.panel_open, "tab": self.tab,
                "restore_time": self.restore_time,
                "tour": {k: getattr(o, k) for k in o.__dataclass_fields__},
                "export": {k: getattr(e, k) for k in e.__dataclass_fields__}}

    def load_state(self, d):
        if not isinstance(d, dict):
            return
        self.store.load(d.get("bookmarks"))
        self.lapse.load(d.get("lapse"))
        self.panel_open = bool(d.get("panel_open", False))
        self.tab = int(d.get("tab", 0)) if isinstance(d.get("tab", 0), int) else 0
        self.restore_time = bool(d.get("restore_time", False))
        for obj, key in ((self.tour_opts, "tour"), (self.exp_opts, "export")):
            src = d.get(key)
            if isinstance(src, dict):
                for k, v in src.items():
                    if k in obj.__dataclass_fields__ and type(v) is type(getattr(obj, k)):
                        setattr(obj, k, v)
        o = self.tour_opts
        o.max_facilities = min(max(o.max_facilities, 1), 40)
        o.max_roads = min(max(o.max_roads, 0), 12)
        o.dwell_s = min(max(o.dwell_s, 2.0), 20.0)
        o.speed = min(max(o.speed, 0.25), 4.0)
        if o.time_mode not in tc.TIME_MODES:
            o.time_mode = "sweep"
        e = self.exp_opts
        e.size_idx = min(max(e.size_idx, 0), len(SIZES) - 1)
        e.fps = e.fps if e.fps in FPS_CHOICES else 30
        e.seconds = min(max(e.seconds, 1.0), 300.0)
        if e.kind not in ("time", "orbit", "flight", "tour"):
            e.kind = "time"

    # ------------------------------------------------------------------------------------------ helpers
    @property
    def domain_key(self) -> str:
        return self.app.domain.key

    def note(self, msg: str):
        self.last_message, self.message_t = msg, time.perf_counter()
        try:
            ui = self.app.get_layer("overlay").manager
            from ...ui import theme
            ui.toasts.push(msg[:170], theme.ACCENT2, 5.0)
        except Exception:
            pass

    def current_view(self) -> tc.View:
        c = self.app.camera
        return (float(c.target[0]), float(c.target[1]), float(c.dist), float(c.yaw), float(c.pitch))

    def depth_source(self, storm: str | None = None):
        """Depth source for the storm (cached until the storm changes)."""
        app = self.app
        name = storm or app.storm.name
        key = (app.domain.key, name)
        s = self._sources.get(key)
        if s is None:
            s = depthtiles.open_depth_source(app, name)
            self._sources[key] = s
        return s

    def depth_series(self, lng, lat, storm=None, radius=depthtiles.RADIUS_DEFAULT_M):
        return self.series.get(self.depth_source(storm), lng, lat, radius)

    # ------------------------------------------------------------------------------------------ bookmarks
    def bookmarks(self) -> list:
        return self.store.list(self.domain_key)

    def save_bookmark(self, name: str | None = None, slot: int | None = None) -> int:
        app = self.app
        i = self.store.add(self.domain_key, name, self.current_view(), app.t, app.storm.name, slot)
        self.note(f"Bookmark saved: {self.bookmarks()[i].name}  (Ctrl+{i + 1} to return)" if i < 9 else
                  f"Bookmark saved: {self.bookmarks()[i].name}")
        return i

    def jump(self, i: int):
        bm = self.store.get(self.domain_key, i)
        if bm is None:
            return
        app = self.app
        self.stop_tour(quiet=True)
        self.flight.start(app.camera, tuple(bm.view))
        if self.restore_time and bm.t is not None:
            if bm.storm and bm.storm in app.storms_available and bm.storm != app.storm.name:
                app.set_storm(bm.storm)
            app.t = float(bm.t)

    # ------------------------------------------------------------------------------------------ time-lapse
    def toggle_lapse(self):
        self.lapse.playing = not self.lapse.playing
        if self.lapse.playing:
            self.app.playing = False
            lo, hi = sorted((self.lapse.t_in, self.lapse.t_out))
            if not (lo <= self.app.t <= hi):
                self.app.t = lo
            elif self.lapse.mode == "once" and self.app.t >= hi - 1e-6:
                self.app.t = lo

    # ------------------------------------------------------------------------------------------ tour
    def tour_storms(self) -> list:
        app, o = self.app, self.tour_opts
        if not o.escalate:
            return [app.storm.name]
        yrs = [s for s in YEAR_ORDER if s in app.storms_available] or list(app.storms_available)
        n = min(max(o.escalate_n, 2), len(yrs))
        if len(yrs) <= n:
            return yrs
        idx = sorted({round(i * (len(yrs) - 1) / (n - 1)) for i in range(n)})
        return [yrs[i] for i in idx]

    def build_tour(self, then_start: bool = True):
        """Choose the places and build the camera path in a worker thread (the depth ranking reads a few hundred small windows)."""
        rt = self.tour
        if rt.building:
            return
        app, o = self.app, self.tour_opts
        rt.building, rt.error = True, ""
        storms = self.tour_storms()
        ref = storms[-1]
        start = self.current_view()
        frame = app.terrain.frame
        root, dkey = app.root, app.domain.key
        src = self.depth_source(ref)
        demo = bool(getattr(app.domain, "demo", False))

        def work():
            try:
                peak = None if (src.kind == "none") else src.peak_many
                stops, info = tc.build_stops(root, dkey, ref, frame, peak_many=peak, types=tuple(o.types),
                                             max_facilities=o.max_facilities, max_roads=o.max_roads, only_wet=o.only_wet,
                                             roads=not demo)
                if not stops:
                    raise ValueError("no named places found for this area and storm (infrastructure.geojson / roads_flooded)")
                stops = tc.order_route(stops, start[:2])
                path = tc.TourPath(stops, start, dwell_s=o.dwell_s)
                plan = tc.TourPlan(storms=storms if o.escalate else [], time_mode=o.time_mode, hold_t=float(app.t))
                info.update({"storms": storms, "source": src.label, "ref_storm": ref})
                rt.path, rt.plan, rt.info, rt.opts = path, plan, info, TourOptions(**vars(o))
                rt.error = ""
            except Exception as e:
                rt.path, rt.error = None, f"{type(e).__name__}: {e}"
            finally:
                rt.building = False

        threading.Thread(target=work, name="bgv-tour-build", daemon=True).start()
        self._after_build = self._start_tour_now if then_start else None

    def _start_tour_now(self):
        rt, app = self.tour, self.app
        if rt.path is None:
            return
        self.flight.cancel()
        self.lapse.playing = False
        app.playing = False
        rt.restore = {"storm": app.storm.name, "t": app.t, "view": self.current_view()}
        rt.T, rt.active, rt.paused, rt.buffering = 0.0, True, False, False
        rt.state = None
        app.camera.stop_motion()
        # the path was built from the view at build time: rebuild from the view now when the camera moved since
        if np.hypot(rt.path.start_view[0] - app.camera.target[0], rt.path.start_view[1] - app.camera.target[1]) > 1.0:
            rt.path = tc.TourPath(rt.path.stops, self.current_view(), dwell_s=rt.path.dwell_s)
        if rt.opts and rt.opts.hide_ui:
            try:
                self.app.get_layer("overlay").manager.st.hidden = True
            except Exception:
                pass

    def tour_toggle(self):
        rt = self.tour
        if rt.active:
            self.stop_tour()
        elif rt.path is not None:
            self._start_tour_now()
        else:
            self.build_tour(True)

    def stop_tour(self, quiet: bool = False, finished: bool = False):
        rt = self.tour
        if not rt.active:
            return
        rt.active = False
        r = rt.restore or {}
        try:
            if r.get("storm") and r["storm"] != self.app.storm.name:
                self.app.set_storm(r["storm"])
            if "t" in r:
                self.app.t = r["t"]
            if rt.opts and rt.opts.hide_ui:
                self.app.get_layer("overlay").manager.st.hidden = False
        except Exception:
            pass
        if not quiet:
            self.note("Tour finished" if finished else "Tour stopped")

    def tour_skip(self, direction: int):
        rt = self.tour
        if not rt.active or rt.path is None:
            return
        rt.T = rt.path.next_start(rt.T) if direction > 0 else rt.path.prev_start(rt.T)
        rt.T = min(rt.T, rt.path.duration)

    def tour_caption(self) -> dict | None:
        """Place caption of the running tour: {title, subtitle, extra, alpha} (names from the data files only)."""
        rt = self.tour
        if not (rt.active and rt.state is not None and rt.state.caption_stop is not None):
            return None
        stop = rt.path.stops[rt.state.caption_stop]
        return {"title": stop.name, "subtitle": stop.label + (f" ({stop.detail})" if stop.detail else ""),
                "alpha": rt.state.caption_alpha, "extra": self.stop_depth_line(stop, self.app.storm.name, self.app.t)}

    def stop_depth_line(self, stop: tc.Stop, storm: str | None, t: float, blocking: bool = False) -> str:
        src = self.depth_source(storm)
        if src.kind == "none":
            return "depth read-out: no depth data for this storm"
        s = (self.series.get_blocking(src, stop.lng, stop.lat) if blocking else self.depth_series(stop.lng, stop.lat, storm))
        if s is None:
            return "reading depth..."
        pk = src.peak(stop.lng, stop.lat) if src.kind == "tiles" else None
        return tc.depth_text(s, t, pk, src.label)

    def _tour_update(self, app, dt):
        rt = self.tour
        path, plan = rt.path, rt.plan
        cam = app.camera
        if getattr(cam, "moving", False) and rt.state is not None:         # the user took the camera: end the tour
            self.stop_tour()
            return
        if rt.paused:
            return
        T_next = min(rt.T + dt * rt.opts.speed, path.duration)
        storm, hour = plan.at(path, T_next)
        if storm and storm != app.storm.name and storm in app.storms_available:
            app.set_storm(storm)
        app.flow.pump()
        if not app.flow.ready_for(hour):
            rt.buffering = True
            if rt.state is None:                                              # at the start: show the first pose, wait for data
                rt.state = path.at(0.0)
            return
        rt.buffering = False
        rt.T = T_next
        st = path.at(rt.T)
        rt.state = st
        app.t = float(hour)
        cam.set_view(*st.view)
        tc._keep_eye_above_ground(cam)
        if rt.T >= path.duration - 1e-6:
            self.stop_tour(finished=True)

    # ------------------------------------------------------------------------------------------ per frame
    def update(self, app, dt):
        if self.export is not None and self.export.state in ("prepare", "render", "encode"):
            return                                                          # the exporter owns camera and hour
        if self._after_build is not None and not self.tour.building:
            fn, self._after_build = self._after_build, None
            if self.tour.path is not None:
                fn()
            elif self.tour.error:
                self.note("Tour: " + self.tour.error)
        if self.tour.active:
            self._tour_update(app, dt)
        elif self.flight.active:
            self.flight.update(app.camera, dt)
        if self.lapse.playing and not self.tour.active and dt > 0:
            nt = self.lapse.step(app.t, dt)
            app.buffering = False
            if app.flow.ready_for(nt):
                app.t = nt
            else:
                app.buffering = True

    # ------------------------------------------------------------------------------------------ export
    def clip_dir(self, name: str):
        from ... import export_cinematic as ex
        return ex.default_dir(self.app) / f"{tc.safe_name(name)}_{time.strftime('%Y%m%d_%H%M%S')}"

    def make_shot(self) -> tuple[tc.Shot | None, str]:
        """(shot, name) for the current export options, or (None, reason)."""
        app, e, lp = self.app, self.exp_opts, self.lapse
        law = tc.TimeLaw("range", min(lp.t_in, lp.t_out), max(lp.t_in, lp.t_out), e.loops)
        view = self.current_view()
        if e.kind == "time":
            return tc.OrbitShot(view, e.seconds, law, 0.0), f"{app.storm.name}_timelapse"
        if e.kind == "orbit":
            return tc.OrbitShot(view, e.seconds, law, e.orbit_deg), f"{app.storm.name}_orbit"
        if e.kind == "flight":
            a, b = self.store.get(self.domain_key, e.flight_a), self.store.get(self.domain_key, e.flight_b)
            if a is None or b is None or a is b:
                return None, "pick two different bookmarks first"
            return tc.FlightShot(tuple(a.view), tuple(b.view), e.seconds, law), f"{app.storm.name}_flight"
        rt = self.tour
        if rt.path is None:
            return None, "build the tour route first (Tour tab)"
        cap = lambda stop, storm, t: self.stop_depth_line(stop, storm, t, blocking=True)       # noqa: E731
        shot = tc.TourShot(rt.path, rt.plan, rt.opts.speed if rt.opts else 1.0, cap)
        return shot, f"{app.storm.name}_tour"

    def start_export(self):
        from ... import export_cinematic as ex
        if self.export is not None and self.export.state in ("prepare", "render", "encode"):
            return
        shot, name = self.make_shot()
        if shot is None:
            self.note("Export: " + name)
            return
        self.stop_tour(quiet=True)
        self.flight.cancel()
        self.lapse.playing = False
        e = self.exp_opts
        w, h = SIZES[e.size_idx]
        src = self.depth_source()
        label = src.label if src.kind != "none" else ""
        spec = ex.ExportSpec(w, h, float(e.fps), shot, self.clip_dir(name), name=tc.safe_name(name), make_mp4=e.mp4,
                             keep_png=e.keep_png, depth_label=label)
        self.export = ex.ExportJob(self.app, spec, log=lambda m: None)
        self.note(f"Exporting {spec.n_frames()} frames at {w}x{h} to {spec.out_dir}")

    def export_active(self) -> bool:
        return self.export is not None and self.export.state in ("prepare", "render", "encode")

    def export_tick(self):
        """Called once per drawn frame by the UI hook: one exported frame."""
        job = self.export
        if job is None or job.state in ("done", "cancelled", "error"):
            return
        job.tick()
        if job.state in ("done", "cancelled", "error"):
            self.note(job.message)
