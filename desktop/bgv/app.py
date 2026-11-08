"""bgv application: window / offscreen driver, shared state, sidebar UI.

`App` owns the GPU context, terrain + flow textures, the frame uniforms (bind group 0), the orbit camera and the
ordered list of layers.  Hooks for later layers:
    app.layers          ordered list[Layer]            app.frame     FrameState (uniforms, bind group 0, layout)
    app.flow            FlowTextures                   app.terrain   TerrainTextures (height tex, sampler, frame)
    app.storm           data.flow.Storm                app.camera    OrbitCamera
    app.depth_format / app.colour_format / app.sample_count   (shared scene pass: build pipelines with these)
    app.domain          bgv.domain.Domain: 'city' (Bowling Green) or 'county' (Warren County); app.set_domain(key) switches at runtime
    app.t (hours) app.exag app.sun_dir() app.view_proj (float64 4x4)
    app.stats_hook      callable(app) run once per frame after the time / flow update (data layer computes stats here)
    app.stats           dict shown at the bottom of the sidebar when non-empty (stats_hook fills it)
    app.terrain_mesh    visible chunk lists of the terrain pass (set by TerrainLayer)
"""
from __future__ import annotations

import sys
import math
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import wgpu

from . import config
from . import domain as domainmod
from .data import flow as flowdata
from .gpu.camera import OrbitCamera
from .gpu.context import GpuContext, GpuTimer, tex_bytes
from .gpu.flowtex import FlowTextures, FrameState
from .gpu.layers import Layer
from .gpu.layers import registry
from .gpu.terraintex import TerrainTextures

SEC_PER_HOUR = 1.5            # playback speed 1x: one simulated hour per 1.5 real seconds
DEFAULT_SUN = (320.0, 36.0)   # compass azimuth of the sun, elevation (degrees)
DEPTH_FORMAT = "depth32float"
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


class App:
    def __init__(self, ctx: GpuContext, root, storm: str | None = None, msaa: int = 4, quiet: bool = False,
                 settings: dict | None = None, colour_format: str = "rgba8unorm-srgb", defer: bool = False,
                 prefetch=None, domain=None, chooser=None):
        """defer=True: only cheap state is created here; the heavy part (terrain, flow textures, every layer) runs in
        small steps through `step_init()` while the window shows a loading screen (see bgv/startup.py).  `prefetch` is an
        optional startup.Prefetch whose worker thread already opened the storm and prepared the terrain data.  `domain`:
        a bgv.domain.Domain or 'city' / 'county' (default: settings["domain"], else the city); an unavailable county falls back
        to the city with a message.  `chooser`: a ui.chooser.DomainChooser - the loading screen then offers City / County
        (see bgv/ui/chooser.py); the domain being built is the remembered one."""
        self.ctx = ctx
        self.root = Path(root)
        self.quiet = quiet
        self.settings = settings if settings is not None else config.load_settings()
        self.sample_count = msaa if msaa in (1, 2, 4) else 4
        self.depth_format = DEPTH_FORMAT
        self.colour_format = colour_format            # must equal the canvas format (see attach_canvas)
        self.camera = OrbitCamera()
        self.layers: list[Layer] = []
        self.layer_log: list[str] = []
        self.errors: deque = deque(maxlen=6)
        self._toast_seen: dict = {}
        self._pending_toasts: list = []
        self._flow_reported: set = set()
        self.stats: dict = {}
        self.stats_hook = None
        self.terrain_mesh = None
        s = self.settings
        if not s.get("restore_last"):                  # settings migration: stored storm / hour from older sessions never override the default
            s.pop("storm", None)
            for v in (s.get("domain_views") or {}).values():
                if isinstance(v, dict):
                    v.pop("storm", None)
        if isinstance(domain, domainmod.Domain):
            notice = ""
        else:
            try:
                domain, notice = domainmod.resolve_or_city(domain or s.get("domain") or domainmod.CITY, self.root, s)
            except domainmod.DomainUnavailable as e:
                raise FileNotFoundError(str(e)) from None
        self.domain = domain
        self._domain_notice = notice
        self.storms_available = list(domain.storms)
        # ---- user-visible state
        self.t = float(s.get("t", 12.0)) if s.get("restore_last") else domainmod.DEFAULT_HOUR
        self.playing = False
        self.speed_idx = SPEEDS.index(s.get("speed", 1.0)) if s.get("speed", 1.0) in SPEEDS else 2
        dv = (s.get("domain_views") or {}).get(domain.key) or {}
        self.exag = float(dv.get("exag", s.get("exag", 2.0) if domain.key == domainmod.CITY else domain.default_exag))
        self.ripple_strength = 1.0
        self.relief = 1.6
        self.water_boost = 1.0
        self.sun_az, self.sun_el = (float(v) for v in s.get("sun", DEFAULT_SUN))
        self.sun_intensity = 1.0
        self.haze_density = float(dv.get("haze", s.get("haze", domain.default_haze) if domain.key == domainmod.CITY
                                         else domain.default_haze))
        self.anim_time = 0.0
        self.palette = s.get("palette", "std") if s.get("palette") in ("std", "cvd") else "std"     # tier colours (bgv.palette)
        self.buffering = False
        # ---- timing
        self.frame_ms = deque(maxlen=240)
        self.fps = 0.0
        self.gpu_ms: float | None = None
        self._t_last = None
        self._t_fps = time.perf_counter()
        self._n_fps = 0
        self.frame_index = 0
        self.t_startup = {}
        # ---- GPU objects
        t0 = time.perf_counter()
        self.frame = FrameState(ctx)
        self.timer = GpuTimer(ctx)
        from .perf import PerfManager
        self.render_scale = 1.0                         # adaptive resolution (perf.QualityGovernor); 1 = native
        self.gov_lod = 1.0                              # terrain LOD distance multiplier set by the governor
        self.perf = PerfManager(self, s.get("quality_mode", "auto"))
        self.storm = None
        self.terrain: TerrainTextures | None = None
        self.flow: FlowTextures | None = None
        self.view_proj = np.eye(4)
        self._targets = None
        self._target_size = (0, 0)
        self.canvas = None
        self.imgui_renderer = None
        self._layer_classes = registry.load_layer_classes(self._log)
        if storm and storm not in self.storms_available:
            print(f"[bgv] unknown storm {storm!r}; available: {', '.join(self.storms_available)}", file=sys.stderr)
        first = domainmod.pick_storm(s, domain.key, self.storms_available, storm)
        self._first_storm = first
        self._prefetch = prefetch
        self._explicit_storm = bool(storm)
        self._t_init0 = t0
        self.chooser = chooser if defer else None        # start-up City / County chooser (loading screen only)
        self._build_done = False                         # the staged build finished, the chooser may still hold the screen
        self.loading = bool(defer)                      # True until step_init() has built everything
        self.load_frac, self.load_msg = 0.0, "Starting"
        self._init_gen = None
        if defer:
            self._init_gen = self._rebuild_steps(None, True)
        else:
            self._load_storm(first, initial=True)
            self._finish_init()

    def _finish_init(self):
        self.t_startup["init"] = time.perf_counter() - self._t_init0
        dv = (self.settings.get("domain_views") or {}).get(self.domain.key) or {}
        view = dv.get("camera") or (self.settings.get("camera") if self.domain.key == domainmod.CITY else None)
        if view and not self._explicit_storm:
            try:
                self.camera.set_view(*view)
            except Exception:
                pass
        self.loading = False
        self.load_frac, self.load_msg = 1.0, "Ready"
        self._build_done = False
        self.chooser = None
        self.settings["domain"] = self.domain.key         # the choice is remembered (saved when the window closes)
        self._announce_domain()

    def _announce_domain(self):
        """Window title + the one-off messages of a freshly shown domain (fallback notice, DEM unit correction)."""
        try:
            if self.canvas is not None:
                self.canvas.set_title(f"BG Flood Viewer - {self.domain.title}")
        except Exception:
            pass
        if self._domain_notice:
            self.report_message(self._domain_notice, toast=True)
            self._domain_notice = ""
        note = (getattr(self.terrain, "stats", None) or {}).get("note") if self.terrain is not None else None
        if note:
            self.report_message(str(note), toast=True)

    def step_init(self, budget_ms: float = 8.0) -> bool:
        """Advance the staged start-up for about `budget_ms` (a single step may take longer).  True when finished."""
        if not self.loading or (self._init_gen is None and not self._build_done):
            return True
        ch = self.chooser
        if ch is not None:
            want = ch.take_request()                     # the user picked the other domain: build that one instead
            if want is not None and want != self.domain.key:
                if self.set_domain(want):
                    ch.building = want
                else:
                    ch.chosen = False                    # unavailable after all: back to waiting for a choice
        t_end = time.perf_counter() + budget_ms * 1e-3
        while time.perf_counter() < t_end:
            if self._init_gen is not None:
                try:
                    self.load_frac, self.load_msg = next(self._init_gen)
                except StopIteration:
                    self._init_gen = None
                    self._build_done = True
                    continue
                if self.load_msg.startswith("Waiting"):  # a worker thread is still busy: give the loading screen a frame
                    break
            else:                                        # built: the chooser (when there is one) may still hold the screen
                if ch is not None and not ch.may_finish(time.perf_counter()):
                    self.load_msg = "Ready"
                    return False
                self._finish_init()
                return True
        return False

    # ------------------------------------------------------------------------------------------ logging
    def _log(self, msg: str):
        self.layer_log.append(msg)
        if not self.quiet:
            print(msg, flush=True)

    def report_error(self, where: str, exc: BaseException):
        self.report_message(f"{where}: {type(exc).__name__}: {exc}")

    def report_message(self, msg: str, toast: bool = True):
        """An error line for the sidebar (`errors`), the console / log and (once per 6 s per message) a red toast."""
        self.errors.append(msg)
        if not self.quiet:
            print("[bgv] " + msg, flush=True)
        try:
            from . import applog
            applog.log().error(msg)
        except Exception:
            pass
        if toast:
            now = time.perf_counter()
            if now - self._toast_seen.get(msg, -99.0) > 6.0:
                self._toast_seen[msg] = now
                if len(self._toast_seen) > 64:
                    self._toast_seen.clear()
                self._pending_toasts.append(msg)

    def _flush_toasts(self):
        if not self._pending_toasts:
            return
        ov = self.get_layer("overlay")
        m = getattr(ov, "manager", None) if ov is not None else None
        if m is None:
            return
        try:
            from .ui import theme
            for msg in self._pending_toasts[-3:]:
                m.toasts.push(msg[:160], theme.ERR, 8.0)
        except Exception:
            pass
        self._pending_toasts.clear()

    # ------------------------------------------------------------------------------------------ storm / data
    def sun_dir(self) -> np.ndarray:
        az, el = math.radians(self.sun_az), math.radians(self.sun_el)
        return np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])

    def _load_storm(self, name: str, initial: bool = False):
        storm = self.domain.open_storm(name)
        same_grid = (self.terrain is not None and self.flow is not None
                     and self.terrain.frame == storm.frame and (self.flow.width, self.flow.height) == (storm.width, storm.height))
        if self.storm is not None:
            self.storm.close()
        self.storm = storm
        if same_grid:
            self.flow.set_storm(storm, self.t)
            for l in self.layers:
                if l.enabled or True:
                    self._safe(l, "on_storm")
            return
        # first load, or the grid changed: rebuild terrain / flow textures and re-init every layer
        for _ in self._rebuild_steps(storm, initial):
            pass

    def _rebuild_steps(self, storm, initial: bool):
        """Generator behind `_load_storm` (grid changed / first load) and the deferred start-up: yields (fraction, message)
        between the heavy steps so a loading screen can be drawn in between.  storm=None: take the prefetched / first storm."""
        yield 0.02, "Opening storm data"
        pre = self._prefetch
        if storm is None:
            if pre is not None:
                while not pre.done:
                    yield 0.04 + 0.2 * float(pre.frac), "Waiting for the data" + (f" - {pre.message}" if pre.message else "")
                storm = pre.storm
            if storm is None:
                storm = self.domain.open_storm(self._first_storm)
            if self.storm is not None and self.storm is not storm:
                self.storm.close()
            self.storm = storm
        for l in self.layers:
            self._safe(l, "shutdown")
        self.layers = []
        if self.flow is not None:
            self.flow.close()
        if self.terrain is not None:
            self.terrain.close()
        yield 0.08, "Preparing terrain"
        data = pre.terrain_data if (pre is not None and pre.storm is storm) else None
        t0 = time.perf_counter()
        self.terrain = TerrainTextures(self.ctx, self.root, storm.frame, data=data, source=self.domain.terrain)
        self.t_startup["terrain"] = time.perf_counter() - t0
        yield 0.30, "Creating flow textures"
        self.flow = FlowTextures(self.ctx, storm.width, storm.height)
        self.flow.set_storm(storm, self.t)
        self.frame.rebuild(self.flow.view, self.terrain.height_view, self.terrain.slope_view)
        self.camera.DEFAULT = self.domain.overview(storm.frame)         # Home / reset: the whole domain in view
        self.camera.set_ground(self.terrain.sampler, self.exag)
        if initial:
            self.camera.reset()
        yield 0.36, "Starting layers"
        n = max(len(self._layer_classes), 1)
        for i, (spec, cls) in enumerate(self._layer_classes):
            self._init_one_layer(spec, cls)
            yield 0.36 + 0.62 * (i + 1) / n, "Starting " + spec.split(":")[-1].replace("Layer", "").lower()
        while pre is not None and not pre.warm_done:         # numba start-up runs in the prefetch thread: never on a frame
            yield 0.99, "Waiting for the numba kernels"
        self._prefetch = None
        yield 1.0, "Ready"

    def set_storm(self, name: str):
        """Switch storm (UI / CLI).  Cheap: meta + background decode; the frame never blocks."""
        if name == (self.storm.name if self.storm else None) or name not in self.storms_available:
            return
        t0 = time.perf_counter()
        try:
            self._load_storm(name)
        except Exception as e:
            self.report_error(f"storm {name}", e)
            return
        self.t_storm_switch_call = time.perf_counter() - t0

    # ------------------------------------------------------------------------------------------ domains
    def domain_infos(self, max_age: float = 3.0):
        """[bgv.domain.DomainInfo] for the Domain menu (file checks, cached for a few seconds)."""
        now = time.perf_counter()
        c = getattr(self, "_dom_info_cache", None)
        if c is None or now - c[0] > max_age:
            c = (now, domainmod.domain_infos(self.root, self.settings))
            self._dom_info_cache = c
        return c[1]

    def _remember_domain_view(self):
        dv = self.settings.setdefault("domain_views", {})
        cam = self.camera
        dv[self.domain.key] = {"camera": [float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw),
                                          float(cam.pitch)],
                               "exag": float(self.exag), "haze": float(self.haze_density),
                               "storm": self.storm.name if self.storm is not None else None}

    def set_domain(self, key: str, block: bool = False) -> bool:
        """Switch to the 'city' or 'county' domain without restarting: terrain, buildings, roads, statistics and the flow
        textures are rebuilt for the new grid and the old GPU memory is freed.  Windowed: a loading screen shows the steps
        (the heavy preparation runs in a worker thread); block=True (tools, tests) runs everything before returning.
        Returns False, with a message to the user, when the domain is unavailable."""
        key = (key or "").lower()
        starting = self.loading and self.chooser is not None       # the start-up chooser may restart the build
        if key == self.domain.key and not self.loading:
            return True
        try:
            dom = domainmod.resolve(key, self.root, self.settings)
        except domainmod.DomainUnavailable as e:
            self.report_message(str(e), toast=True)
            return False
        if self.loading and not starting:                 # a switch is already under way: not twice
            return False
        from .startup import Prefetch
        if not starting:
            self._remember_domain_view()
            self.settings = self.collect_settings()       # the live layer / UI state travels into the new layers
        self.settings["domain"] = dom.key
        old_pre = self._prefetch
        if old_pre is not None and hasattr(old_pre, "cancel"):
            old_pre.cancel()                              # the other domain's preparation is not needed any more
        self.playing = False
        self._drop_scene()
        self.domain = dom
        self._domain_notice = ""
        self.storms_available = list(dom.storms)
        dv = (self.settings.get("domain_views") or {}).get(dom.key) or {}
        self.exag = float(dv.get("exag", dom.default_exag if dom.key != domainmod.CITY else self.settings.get("exag", 2.0)))
        self.haze_density = float(dv.get("haze", dom.default_haze))
        self._explicit_storm = False
        self._first_storm = domainmod.pick_storm(dict(self.settings, restore_last=True), dom.key, self.storms_available)
        self._prefetch = Prefetch(self.root, self._first_storm, self.settings, domain=dom)
        self._t_init0 = time.perf_counter()
        self.load_frac, self.load_msg = 0.0, f"Switching to {dom.label}"
        self.loading = True
        self._build_done = False
        self._init_gen = self._rebuild_steps(None, True)
        self._dom_info_cache = None
        if block:
            while self.loading:
                if not self.step_init(1.0e6):
                    time.sleep(0.005)                     # a worker thread is still preparing the data
        return True

    def _drop_scene(self):
        """Forget everything that belongs to the old domain's buildings / roads / statistics (the GPU objects are released by
        the layers' shutdown inside `_rebuild_steps`)."""
        hub = getattr(self, "_stats_hub", None)
        if hub is not None:
            try:
                hub.close()
            except Exception:
                pass
        self._stats_hub = None
        self._scene_data = None
        for old in list((getattr(self, "_grid_storms", None) or {}).values()):
            try:
                old.close()
            except Exception:
                pass
        self._grid_storms = {}
        self.stats_hook = None
        self.flood = None
        self.timeline = None
        self.selection = None
        self._flight = None
        self.stats = {}
        self.ready_waiters = {}
        self._flow_reported = set()

    def _init_layers(self):
        for spec, cls in self._layer_classes:
            self._init_one_layer(spec, cls)

    def _init_one_layer(self, spec, cls):
        saved = self.settings.get("layers", {})
        try:
            layer = cls()
            st = saved.get(layer.name, {})
            layer.enabled = bool(st.get("enabled", layer.default_enabled))
            try:
                layer.load_state(st.get("state", {}))
            except Exception as e:
                self.report_error(f"{layer.name}.load_state", e)
            layer.init(self)
            self.layers.append(layer)
            if self._target_size != (0, 0):
                layer.resize(self, *self._target_size)
        except Exception as e:
            self.report_error(f"layer {spec} init", e)
            if not self.quiet:
                traceback.print_exc()

    def _safe(self, layer: Layer, method: str, *args):
        try:
            return getattr(layer, method)(self, *args)
        except Exception as e:
            self.report_error(f"{layer.name}.{method}", e)
            if method in ("draw", "update", "compute"):
                layer.enabled = False
            return None

    def get_layer(self, name: str):
        for l in self.layers:
            if l.name == name:
                return l
        return None

    # ------------------------------------------------------------------------------------------ canvas
    def attach_canvas(self, canvas, ui: bool = True):
        """Bind a rendercanvas canvas (window or offscreen).  Configures its context and the imgui renderer."""
        self.canvas = canvas
        cctx = canvas.get_wgpu_context()
        fmt = cctx.get_preferred_format(self.ctx.adapter)
        cctx.configure(device=self.ctx.device, format=fmt)
        self._canvas_ctx = cctx
        if fmt != self.colour_format:                  # layers were built for another format: rebuild them
            self.colour_format = fmt
            self._targets = None
            if not self.loading:                        # (a deferred start-up has not created any layer yet)
                self.rebuild_for_format()
        if ui:
            from wgpu.utils.imgui import ImguiRenderer
            self.imgui_renderer = ImguiRenderer(self.ctx.device, canvas)
            self.imgui_renderer.set_gui(self._gui)
            try:
                # imgui's default "trickle" queue applies at most one key transition per frame and holds the typed characters
                # behind it: fast typing (Shift + letter, several keys between two frames, slow county frames) arrived
                # frames late, and whatever was still queued when the field lost focus or Esc came was lost
                # ("Scottsville" -> "Scotllle").  Apply every queued event in the frame it arrives.
                self.imgui_renderer.backend.io.config_input_trickle_event_queue = False
            except Exception as e:
                self.report_error("imgui io", e)
            self._style_imgui()
        # Order: -100 releases (always seen, even when imgui swallows the event afterwards), -99 imgui's own handlers
        # (stop_propagation while it wants the mouse / keyboard), -98 our UI-rect gate (imgui's want_* flags are one
        # frame old), 0 the camera and the hotkeys.
        canvas.add_event_handler(self._on_release, "pointer_up", "key_up", order=-100)
        canvas.add_event_handler(self._gate_pointer, "pointer_down", "wheel", order=-98)
        canvas.add_event_handler(self._gate_key, "key_down", order=-98)
        canvas.add_event_handler(self._on_pointer_down, "pointer_down")
        canvas.add_event_handler(self._on_pointer_move, "pointer_move")
        canvas.add_event_handler(self._on_wheel, "wheel")
        canvas.add_event_handler(self._on_key, "key_down")

    def rebuild_for_format(self):
        """Colour format changed after layers were created (attach_canvas after construction): re-init layers."""
        for l in self.layers:
            self._safe(l, "shutdown")
        self.layers = []
        self._init_layers()

    # ------------------------------------------------------------------------------------------ input
    def pointer_over_ui(self, x: float, y: float) -> bool:
        """True when a press / wheel at logical pixel (x, y) belongs to the interface and not to the 3D view.  imgui's
        `want_capture_mouse` is computed at the start of a frame, so a fast move + click inside one frame would still
        see last frame's answer: also test the pointer against the layout of the last drawn frame."""
        try:
            io = self.imgui_renderer.backend.io if self.imgui_renderer is not None else None
            if io is not None and io.want_capture_mouse:
                return True
            ov = self.get_layer("overlay")
            m = getattr(ov, "manager", None) if ov is not None and ov.enabled else None
            if m is not None:
                return bool(m.over_ui(x, y))
            return self.imgui_renderer is not None and x < 340.0       # fallback sidebar
        except Exception:
            return False

    def _gate_pointer(self, ev):
        if self.pointer_over_ui(ev.get("x", 0.0), ev.get("y", 0.0)):
            ev["stop_propagation"] = True

    def _gate_key(self, ev):
        """While a text box has the focus every key belongs to it.  wgpu-py's imgui handler only swallows letters, digits
        and its named keys: space, [ ] , . - and friends would leak through to the hotkeys without this."""
        io = self.imgui_renderer.backend.io if self.imgui_renderer is not None else None
        if io is not None and (io.want_text_input or io.want_capture_keyboard):
            ev["stop_propagation"] = True

    def _on_release(self, ev):
        """Releases always reach the camera, even when imgui swallows them (a drag that ends over a window, a key that
        goes up while a text box has the focus): no sticky drag, no key stuck down."""
        if ev.get("event_type") == "pointer_up":
            self.camera.on_pointer_up(ev)
        else:
            self.camera.on_key(ev)

    def _on_pointer_down(self, ev):
        self.camera.on_pointer_down(ev)

    def _on_pointer_move(self, ev):
        self.camera.on_pointer_move(ev)

    def _on_wheel(self, ev):
        self.camera.on_wheel(ev)

    def _on_key(self, ev):
        ov = self.get_layer("overlay")
        m = getattr(ov, "manager", None) if ov is not None and ov.enabled else None
        if m is not None and ev.get("event_type") == "key_down" and m.text_focused():
            m.on_key(ev)                                # handles Esc / Ctrl+F; every other key stays in the text box
            return                  # UI hotkeys (F1 F3 F11 F12 H Esc [ ] F Home) get the first look
        if ov is not None and ov.enabled and ov.on_key(self, ev):
            return
        if ev.get("event_type") == "key_down":
            k = ev.get("key", "")
            if k == " ":
                self.playing = not self.playing
                return
            if k in (",", "<"):
                self.t = max(0.0, math.floor(self.t - 1e-6))
                return
            if k in (".", ">"):
                self.t = min(24.0, math.floor(self.t + 1.0))
                return
        self.camera.on_key(ev)

    # ------------------------------------------------------------------------------------------ per frame
    def set_exag(self, e: float):
        self.exag = float(e)
        self.camera.set_exag(self.exag)

    def step(self, dt: float):
        """Advance time, camera, flow uploads.  dt in real seconds."""
        self.anim_time += dt
        self.camera.update(dt)
        self.view_proj = self.camera.view_proj()
        self.buffering = False
        if self.playing and dt > 0:
            adv = dt * SPEEDS[self.speed_idx] / SEC_PER_HOUR
            nt = self.t + adv
            if nt >= 24.0:
                nt = 0.0 if self.flow.ready_for(0.0) else self.t
            if self.flow.ready_for(nt):
                self.t = nt
            else:
                self.buffering = True
        self.t = min(max(self.t, 0.0), 24.0)
        self.flow.pump()
        self._report_flow_failures()
        self._report_scene_error()
        if self.stats_hook is not None:
            try:
                self.stats_hook(self)
            except Exception as e:
                self.report_error("stats_hook", e)
                self.stats_hook = None

    def _report_flow_failures(self):
        """A missing / corrupt hour file reads as dry; say so (sidebar line + one toast naming every skipped hour) and
        keep every other hour."""
        failed = self.flow.failed if self.flow is not None else None
        if not failed:
            return
        key = (id(self.storm), tuple(sorted(failed)))
        if key in self._flow_reported:
            return
        self._flow_reported.add(key)
        hours = sorted(failed)
        names = ", ".join(str(h) for h in hours[:-1]) + (" and " if len(hours) > 1 else "") + str(hours[-1])
        why = "missing" if all("FileNotFoundError" in failed[h] for h in hours) else "missing or damaged"
        self.report_message(f"Hour{'s' if len(hours) > 1 else ''} {names} of the {self.storm.name} storm could not be read "
                            f"(file {why}) and {'are' if len(hours) > 1 else 'is'} shown dry; the other hours work. "
                            f"{failed[hours[0]][:100]}")

    def _report_scene_error(self):
        """buildings.geojson / roads.geojson missing or unreadable: say so once and switch those layers off (the terrain,
        the water and everything else keep working)."""
        sc = getattr(self, "_scene_data", None)
        if sc is None or getattr(sc, "state", "") != "error" or id(sc) in self._flow_reported:
            return
        self._flow_reported.add(id(sc))
        for name in ("buildings", "roads"):
            l = self.get_layer(name)
            if l is not None:
                l.enabled = False
        self.report_message(f"Buildings and roads could not be loaded, those layers are off: {getattr(sc, 'error', '') or sc.message}")

    def frustum(self):
        """The 5 frustum planes of `self.view_proj`, computed once per frame and shared by every culling layer."""
        vp = self.view_proj
        if getattr(self, "_fr_src", None) is not vp:
            from .gpu.cull import frustum_planes
            self._fr_planes = frustum_planes(vp)
            self._fr_src = vp
        return self._fr_planes

    def write_frame(self, pw: int, ph: int):
        h0, h1, mix = flowdata.hour_pair(self.t)
        st = self.storm
        sun = self.sun_dir()
        self.frame.write(
            view_proj=self.view_proj, eye=self.camera.eye(), anim_time=self.anim_time, sun=sun,
            sun_intensity=self.sun_intensity, grid=st.frame, enc=(st.depth_max, st.vmax, st.wet_threshold),
            t=self.t, mix=mix, h0=h0, h1=h1, exag=self.exag, water_exag=self.exag * self.water_boost,
            viewport=(pw, ph), zrange=self.terrain.zrange, ripple=self.ripple_strength,
            sun_elev=math.radians(self.sun_el), haze_density=self.haze_density, layer_map=self.flow.layer_map,
            relief=self.relief)

    def _ensure_targets(self, pw: int, ph: int):
        if self._targets is not None and self._target_size == (pw, ph):
            return
        dev = self.ctx.device
        if self._targets is not None:
            for t in self._targets.values():
                try:
                    t.destroy()
                except Exception:
                    pass
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT
        tg = {}
        tg["depth"] = dev.create_texture(label="scene-depth", size=(pw, ph, 1), format=self.depth_format,
                                         usage=usage, sample_count=self.sample_count)
        nb = tex_bytes(pw, ph, 1, 4) * self.sample_count
        if self.sample_count > 1:
            tg["msaa"] = dev.create_texture(label="scene-msaa", size=(pw, ph, 1), format=self.colour_format,
                                            usage=usage, sample_count=self.sample_count)
            nb += tex_bytes(pw, ph, 1, 4) * self.sample_count
        self._targets = tg
        self._target_size = (pw, ph)
        self._depth_view = tg["depth"].create_view()
        self._msaa_view = tg["msaa"].create_view() if "msaa" in tg else None
        self.ctx.track("render targets", nb)
        self.camera.resize(pw, ph)
        for l in self.layers:
            self._safe(l, "resize", pw, ph)

    def render_scene(self, target_view, pw: int, ph: int, dt: float = 0.0):
        """Update + record + submit the whole scene into `target_view` (a view of a pw x ph texture of
        `colour_format` with RENDER_ATTACHMENT).  Returns the CPU time spent recording (ms)."""
        t0 = time.perf_counter()
        fw, fh = pw, ph                                   # the final (window) size; the scene may be drawn smaller
        pw, ph = self.perf.scene_size(fw, fh)
        self._ensure_targets(pw, ph)
        self.write_frame(pw, ph)
        active = [l for l in self.layers if l.enabled]
        for l in active:
            self._safe(l, "update", dt)
        enc = self.ctx.device.create_command_encoder(label="frame")
        for l in active:
            if l.enabled:
                self._safe(l, "compute", enc)
        scene_target = target_view
        scaled = (pw, ph) != (fw, fh)
        if scaled:                                        # adaptive resolution: scene into an intermediate, blit to the window
            res = self.perf.resolve_for(pw, ph)
            scene_target = res.view
        if self._msaa_view is not None:
            color = {"view": self._msaa_view, "resolve_target": scene_target, "clear_value": (0.6, 0.72, 0.86, 1.0),
                     "load_op": "clear", "store_op": "discard"}
        else:
            color = {"view": scene_target, "clear_value": (0.6, 0.72, 0.86, 1.0), "load_op": "clear", "store_op": "store"}
        depth = {"view": self._depth_view, "depth_clear_value": 0.0, "depth_load_op": "clear", "depth_store_op": "discard"}
        tw = self.timer.pass_writes()
        kwargs = {"timestamp_writes": tw} if tw else {}
        p = enc.begin_render_pass(color_attachments=[color], depth_stencil_attachment=depth, **kwargs)
        for l in active:
            if l.enabled and l.draws:
                self._safe(l, "draw", p)
        p.end()
        self.timer.after_pass(enc)
        if scaled:
            self.perf.resolve.blit(enc, target_view)
        self.ctx.queue.submit([enc.finish()])
        self.timer.after_submit()
        if self.timer.last_ms is not None:
            self.gpu_ms = self.timer.last_ms
        return (time.perf_counter() - t0) * 1000.0

    FATAL_AFTER_FAILED_FRAMES = 120

    def draw_frame(self):
        """rendercanvas draw callback (window or offscreen canvas).  A frame that raises (lost surface / device, a bug
        outside the per-layer guards) is reported (sidebar line + toast + log) and the next frame tries again; only a
        long unbroken run of failures (about 2 s) closes the window with a message."""
        try:
            self._draw_frame_inner()
            self._frame_fail = 0
        except Exception as e:
            self._frame_fail = getattr(self, "_frame_fail", 0) + 1
            self.report_error("frame", e)
            if self._frame_fail == 1:
                try:
                    from . import applog
                    applog.log().exception("frame failed")
                except Exception:
                    pass
            if self._frame_fail % 20 == 0:                # lost / outdated surface: configure it again
                try:
                    self._canvas_ctx.configure(device=self.ctx.device, format=self.colour_format)
                    self._targets = None
                except Exception:
                    pass
            if self._frame_fail >= self.FATAL_AFTER_FAILED_FRAMES and not getattr(self, "fatal_error", None):
                self.fatal_error = e
                try:
                    self.canvas.close()
                except Exception:
                    pass

    def _draw_loading(self):
        """One frame of the loading screen: clear, imgui text + progress bar, then a slice of the staged start-up."""
        pw, ph = self.canvas.get_physical_size()
        if pw < 2 or ph < 2:
            return
        dev = self.ctx.device
        view = self._canvas_ctx.get_current_texture().create_view()
        enc = dev.create_command_encoder(label="loading")
        enc.begin_render_pass(color_attachments=[{"view": view, "clear_value": (0.012, 0.016, 0.028, 1.0),
                                                  "load_op": "clear", "store_op": "store"}]).end()
        self.ctx.queue.submit([enc.finish()])
        if self.imgui_renderer is not None:
            self.imgui_renderer.render()
        self.frame_index += 1
        if self.step_init(10.0) and self.imgui_renderer is not None:
            self._style_imgui()                         # the UI layer exists now: theme + fonts + fast draw replay
            self.imgui_renderer.set_gui(self._gui)
            self.perf.gc.enter_frames()
            self._t_last = None

    def _loading_gui(self):
        from imgui_bundle import imgui
        io = imgui.get_io()
        W, H = io.display_size.x, io.display_size.y
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((W, H))
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_background
                 | imgui.WindowFlags_.no_inputs | imgui.WindowFlags_.no_saved_settings)
        imgui.begin("##loading", None, flags)
        bw = min(420.0, W * 0.6)
        imgui.set_cursor_pos(((W - bw) * 0.5, H * 0.5 - 38.0))
        imgui.text_colored((0.62, 0.80, 1.0, 1.0), "BG Flood Viewer")
        imgui.set_cursor_pos(((W - bw) * 0.5, H * 0.5 - 8.0))
        imgui.push_style_color(imgui.Col_.plot_histogram, (0.30, 0.63, 0.92, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg, (0.157, 0.188, 0.259, 1.0))
        imgui.progress_bar(float(self.load_frac), (bw, 8), "")
        imgui.pop_style_color(2)
        imgui.set_cursor_pos(((W - bw) * 0.5, H * 0.5 + 10.0))
        imgui.text_disabled(f"Loading - {self.load_msg}")
        imgui.end()
        if self.chooser is not None:
            try:
                from .ui import chooser as chooser_ui
                chooser_ui.draw(self)
            except Exception as e:                       # the chooser must never stop the start-up
                self.chooser = None
                self.report_error("domain chooser", e); traceback.print_exc()

    def _draw_frame_inner(self):
        if self.loading:
            self._draw_loading()
            return
        now = time.perf_counter()
        dt = 0.0 if self._t_last is None else min(now - self._t_last, 0.1)
        if self._t_last is not None:
            self.frame_ms.append((now - self._t_last) * 1000.0)
        self._t_last = now
        self._n_fps += 1
        if now - self._t_fps >= 0.5:
            self.fps = self._n_fps / (now - self._t_fps)
            self._n_fps = 0
            self._t_fps = now
        pw, ph = self.canvas.get_physical_size()
        if pw < 2 or ph < 2:                            # minimised / degenerate: draw nothing, drop held input
            self.camera.release_keys()
            self.camera.end_drag()
            return
        pr = self.canvas.get_pixel_ratio()
        self._ensure_targets(*self.perf.scene_size(pw, ph))
        self.camera.resize(pw / pr, ph / pr)       # input coordinates are logical pixels
        self.step(dt)
        tex = self._canvas_ctx.get_current_texture()
        self.render_scene(tex.create_view(), pw, ph, dt)
        if self.imgui_renderer is not None:
            self._flush_toasts()
            self.imgui_renderer.render()
            ov = self.get_layer("overlay")
            if ov is not None and ov.enabled:
                ov.post_frame(self)                     # screenshot requests (needs the finished imgui draw data)
        self.frame_index += 1
        if self.frame_ms:
            self.perf.after_frame(self.frame_ms[-1])

    # ------------------------------------------------------------------------------------------ UI
    def _style_imgui(self):
        from imgui_bundle import imgui
        ov = self.get_layer("overlay")
        if ov is not None:                              # the UI layer brings its own theme, fonts and implot context
            try:
                from .ui import imgui_fast, theme
                theme.setup(imgui, float(ov.state.ui_scale))
                imgui_fast.patch(self.imgui_renderer, ov.lite_frame)    # wgpu-py's imgui backend rebuilds a bind group per draw command
                return
            except Exception as e:
                self.report_error("ui theme", e)
        st = imgui.get_style()
        st.window_rounding = 6.0
        st.frame_rounding = 4.0
        st.grab_rounding = 4.0
        st.window_border_size = 0.0
        st.item_spacing = (8.0, 5.0)
        st.window_padding = (10.0, 10.0)
        V = imgui.ImVec4
        for col, rgba in ((imgui.Col_.window_bg, (0.07, 0.09, 0.12, 0.92)), (imgui.Col_.header, (0.16, 0.26, 0.40, 0.8)),
                          (imgui.Col_.frame_bg, (0.14, 0.17, 0.22, 1.0)), (imgui.Col_.slider_grab, (0.30, 0.62, 0.95, 1.0)),
                          (imgui.Col_.button, (0.18, 0.36, 0.60, 1.0)), (imgui.Col_.check_mark, (0.40, 0.75, 1.0, 1.0))):
            st.set_color_(col, V(*rgba))

    def _gui(self):
        from imgui_bundle import imgui
        if self.loading:
            return self._loading_gui()
        ov = self.get_layer("overlay")
        if ov is not None and ov.enabled:
            try:
                if ov.gui(self):
                    return
            except Exception as e:                      # fall back to the minimal sidebar below
                self.report_error("ui", e)
                ov.enabled = False
        try:
            self._sidebar(imgui)
        except Exception as e:                          # a broken layer.ui() must not kill the frame
            self.report_error("ui", e)

    def _sidebar(self, imgui):
        io = imgui.get_io()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((340, io.display_size.y))
        flags = (imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
                 | imgui.WindowFlags_.no_title_bar)
        imgui.begin("sidebar", None, flags)
        imgui.text_colored((0.55, 0.80, 1.0, 1.0), "BG Flood Viewer")
        imgui.same_line()
        imgui.text_disabled("desktop")
        imgui.separator()

        # ---- storm
        st = self.storm
        cur = self.storms_available.index(st.name) if st.name in self.storms_available else 0
        imgui.set_next_item_width(-1)
        ch, idx = imgui.combo("##storm", cur, [self.storm_label(s) for s in self.storms_available])
        if ch:
            self.set_storm(self.storms_available[idx])
        self._source_badge(imgui)
        ready, total = self.flow.progress()
        if ready < total:
            imgui.progress_bar(ready / max(total, 1), (-1, 6))
            imgui.text_disabled(f"loading hours {ready}/{total}")

        # ---- time
        imgui.separator()
        imgui.set_next_item_width(-1)
        ch, t = imgui.slider_float("##t", self.t, 0.0, 24.0, f"t = %.2f h")
        if ch:
            self.t = t
        if imgui.button("Pause" if self.playing else "Play ", (70, 0)):
            self.playing = not self.playing
        imgui.same_line()
        imgui.set_next_item_width(-1)
        ch, si = imgui.slider_int("##speed", self.speed_idx, 0, len(SPEEDS) - 1, f"speed {SPEEDS[self.speed_idx]:g}x")
        if ch:
            self.speed_idx = si
        if self.buffering:
            imgui.text_colored((1.0, 0.8, 0.3, 1.0), "buffering next hour...")
        hr = round(self.t)
        imgui.text_disabled(f"snapshot hour {hr}" + ("  (dry: no file)" if st.is_zero(hr) else ""))

        # ---- view
        imgui.separator()
        imgui.set_next_item_width(-1)
        ch, e = imgui.slider_float("##exag", self.exag, 1.0, 10.0, "vertical exaggeration x%.1f")
        if ch:
            self.set_exag(e)
        imgui.set_next_item_width(-1)
        ch, az = imgui.slider_float("##saz", self.sun_az, 0.0, 360.0, "sun azimuth %.0f")
        if ch:
            self.sun_az = az
        imgui.set_next_item_width(-1)
        ch, el = imgui.slider_float("##sel", self.sun_el, 3.0, 85.0, "sun elevation %.0f")
        if ch:
            self.sun_el = el
        if imgui.button("Reset view"):
            self.camera.reset()

        # ---- layers
        imgui.separator()
        imgui.text("Layers")
        for l in self.layers:
            imgui.push_id(l.name)
            _, l.enabled = imgui.checkbox(l.label(), l.enabled)
            imgui.pop_id()
        for l in self.layers:
            if l.has_ui and l.enabled:
                imgui.push_id("ui" + l.name)
                if imgui.collapsing_header(l.label() + " options"):
                    try:
                        l.ui(self)
                    except Exception as e:
                        self.report_error(f"{l.name}.ui", e)
                imgui.pop_id()

        # ---- statistics from the data layer
        if self.stats:
            imgui.separator()
            imgui.text("Statistics")
            for k, v in self.stats.items():
                imgui.text_disabled(f"{k}: {v}")

        # ---- performance / honesty
        imgui.separator()
        fm = np.array(self.frame_ms, np.float64) if self.frame_ms else np.array([16.7])
        imgui.text(f"{self.fps:5.1f} fps   {fm.mean():.1f} ms   p99 {np.percentile(fm, 99):.1f} ms")
        gpu = "n/a" if self.gpu_ms is None else f"{self.gpu_ms:.2f} ms"
        imgui.text_disabled(f"GPU pass {gpu}   VRAM ~{self.ctx.vram_mb():.0f} MB")
        imgui.text_disabled(f"{self.ctx.adapter_name} ({self.ctx.backend})")
        imgui.text_wrapped("Velocity is DERIVED from the water-surface slope (Manning), not solver output.")
        for msg in list(self.errors)[-3:]:
            imgui.text_colored((1.0, 0.45, 0.4, 1.0), msg[:90])
        imgui.end()

    def storm_label(self, name: str) -> str:
        return self.domain.storm_label(name)

    def _source_badge(self, imgui):
        st = self.storm
        if st.source == "tiers":
            imgui.text_colored((1.0, 0.78, 0.35, 1.0), "Preview from depth tiers")
            imgui.text_disabled("depth is approximate; severe tier = >= 1.2 m")
        else:
            imgui.text_colored((0.5, 0.9, 0.55, 1.0), "Full-precision (solver depth rasters)")

    # ------------------------------------------------------------------------------------------ persistence
    def collect_settings(self) -> dict:
        cam = self.camera
        d = dict(self.settings)
        d.update({
            "quality_mode": self.perf.gov.mode, "t": round(self.t, 3), "speed": SPEEDS[self.speed_idx],
            "sun": [self.sun_az, self.sun_el], "domain": self.domain.key, "palette": self.palette,
            "layers": {l.name: {"enabled": bool(l.enabled), "state": self._safe_state(l)} for l in self.layers},
        })
        if self.domain.key == domainmod.CITY:             # the city keeps its legacy top-level keys
            d.update({"storm": self.storm.name, "exag": self.exag, "haze": self.haze_density,
                      "camera": [float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw), float(cam.pitch)]})
        dv = dict(d.get("domain_views") or {})
        dv[self.domain.key] = {"camera": [float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw),
                                          float(cam.pitch)], "exag": float(self.exag), "haze": float(self.haze_density),
                               "storm": self.storm.name}
        d["domain_views"] = dv
        return d

    def _safe_state(self, l: Layer) -> dict:
        try:
            return l.save_state()
        except Exception:
            return {}

    def save_settings(self):
        config.save_settings(self.collect_settings())

    def vram_mb(self) -> float:
        return self.ctx.vram_mb()

    def shutdown(self):
        hub = getattr(self, "_stats_hub", None)
        if hub is not None:                                   # stop the statistics worker before the data goes away
            try:
                hub.close()
            except Exception:
                pass
        if getattr(self, "_shut", False):                     # idempotent: tests and the window may both call it
            return
        self._shut = True
        for l in self.layers:
            self._safe(l, "shutdown")
        self.layers = []                                      # drop GPU buffers / textures held by the layers
        if self.flow:
            self.flow.close()
        if getattr(self, "terrain", None) is not None:
            try:
                self.terrain.close()
            except Exception:
                pass
        if self.storm:
            self.storm.close()


from .data import depthapi as _depthapi  # noqa: E402  (installs app.depth_at / building_depth / depth_profile / tile_info / tiles)
_depthapi.install(App)
