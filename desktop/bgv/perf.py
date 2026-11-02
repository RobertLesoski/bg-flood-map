"""Runtime performance management: garbage collector control and the automatic quality governor.

`app.perf` (PerfManager) is created by App; the window loop calls `app.perf.after_frame(frame_ms)` once per frame.

GC      `gc.freeze()` after start-up (the loaded data / modules are never scanned again) and `gc.disable()` while frames are
        running; a young-generation collection runs every ~2 s and a full one at quiet moments (storm switch, every minute),
        so a collection never lands in the middle of a busy frame.

Quality governor (mode "auto", the default; "manual" = the user picked a preset in the Quality section, nothing is touched)
        Frame-time driven with hysteresis.  Level 3 is the full look; every step down trades a little on the cheapest things
        first (shadow march steps, sub-texel shoreline range, particle / arrow caps, terrain LOD distance) and only the last two
        levels lower the render resolution (the scene is drawn at `render_scale` and bilinearly stretched, the interface stays
        sharp).  Down: the mean frame time stays above 1.25 x the target for 0.6 s.  Up: GPU time below 45 % of the target and
        no missed frames for 4 s, one level at a time, never within 3 s of the last change.
        MSAA is fixed at launch (pipelines are built for it); the render scale is the adaptive replacement.
"""
from __future__ import annotations

import gc
import time

import wgpu

# level -> (app.quality preset, shadow march steps, sub-texel shoreline range m/px, terrain LOD multiplier, render scale)
LEVELS = {
    3: ("high", 26.0, 8.0, 1.0, 1.0),
    2: ("medium", 20.0, 5.0, 0.85, 1.0),
    1: ("low", 14.0, 3.0, 0.7, 0.85),
    0: ("low", 8.0, 0.0, 0.55, 0.7),
}
LEVEL_NAMES = {3: "High", 2: "Medium", 1: "Low", 0: "Minimum"}
TARGET_MS = 1000.0 / 60.0


class GcControl:
    """See the module docstring.  `enter_frames()` once the app is up, `tick(frame_index)` every frame."""

    def __init__(self):
        self.active = False
        self.collections = 0
        self.last_ms = 0.0
        self._t_young = time.perf_counter()
        self._t_full = self._t_young

    def enter_frames(self):
        gc.collect()
        try:
            gc.freeze()
        except Exception:
            pass
        gc.disable()
        self.active = True
        self._t_young = self._t_full = time.perf_counter()

    def collect_now(self, generation: int = 2):
        t0 = time.perf_counter()
        gc.collect(generation)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.collections += 1

    def tick(self, busy: bool = False):
        """Call at the END of a frame.  Young collections are ~0.1 ms; the full one only when nothing is going on."""
        if not self.active:
            return
        now = time.perf_counter()
        if now - self._t_young >= 2.0:
            self._t_young = now
            self.collect_now(0)
        if not busy and now - self._t_full >= 60.0:
            self._t_full = now
            self.collect_now(1)

    def leave_frames(self):
        if self.active:
            gc.enable()
            self.active = False


class SceneResolve:
    """Intermediate colour texture + blit pipeline used when the scene is rendered below the window resolution."""

    WGSL = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
  return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> { return textureSample(src, samp, in.uv); }
"""

    def __init__(self, device, colour_format: str):
        self.device = device
        self.format = colour_format
        self.size = None
        self.texture = None
        self.view = None
        self.bind_group = None
        self.bgl = device.create_bind_group_layout(label="resolve-bgl", entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}}])
        self.sampler = device.create_sampler(label="resolve-samp", mag_filter="linear", min_filter="linear",
                                             address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        mod = device.create_shader_module(label="resolve", code=self.WGSL)
        self.pipeline = device.create_render_pipeline(
            label="resolve", layout=device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            vertex={"module": mod, "entry_point": "vs"}, primitive={"topology": "triangle-list"},
            fragment={"module": mod, "entry_point": "fs", "targets": [{"format": colour_format}]})

    def ensure(self, w: int, h: int):
        if self.size == (w, h):
            return self.view
        self.destroy()
        self.texture = self.device.create_texture(
            label="scene-resolve", size=(w, h, 1), format=self.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING)
        self.view = self.texture.create_view()
        self.bind_group = self.device.create_bind_group(label="resolve-bg", layout=self.bgl, entries=[
            {"binding": 0, "resource": self.view}, {"binding": 1, "resource": self.sampler}])
        self.size = (w, h)
        return self.view

    def blit(self, encoder, dst_view):
        p = encoder.begin_render_pass(color_attachments=[{"view": dst_view, "clear_value": (0, 0, 0, 1),
                                                           "load_op": "clear", "store_op": "store"}])
        p.set_pipeline(self.pipeline)
        p.set_bind_group(0, self.bind_group)
        p.draw(3)
        p.end()

    def destroy(self):
        if self.texture is not None:
            try:
                self.texture.destroy()
            except Exception:
                pass
        self.texture = self.view = self.bind_group = None
        self.size = None


class QualityGovernor:
    DOWN_FACTOR = 1.25            # mean frame time above this x target ...
    DOWN_AFTER_S = 0.6            # ... for this long -> one level down
    UP_GPU_FRACTION = 0.45        # GPU pass below this fraction of the target ...
    UP_AFTER_S = 4.0              # ... with no missed frames for this long -> one level up
    MIN_DWELL_S = 3.0             # never two changes within this time

    def __init__(self, app, mode: str = "auto", level: int = 3):
        self.app = app
        self.mode = mode if mode in ("auto", "manual") else "auto"
        self.level = level
        self.target_ms = TARGET_MS
        self._bad_since = None
        self._good_since = None
        self._last_change = time.perf_counter()
        self._ema = None
        self._skip = 30                       # ignore the first frames (shader warm-up, window creation)
        self.changes = 0
        self.apply(level, force=True)

    # ------------------------------------------------------------------ levels
    def apply(self, level: int, force: bool = False):
        level = max(0, min(3, int(level)))
        if level == self.level and not force:
            return
        self.level = level
        q, steps, smooth_ps, lod, scale = LEVELS[level]
        app = self.app
        if self.mode == "auto":
            app.quality = q                    # read by the particle / arrow layers
        app.render_scale = scale
        app.gov_lod = lod
        try:
            app.frame.q[0], app.frame.q[1] = steps, smooth_ps
        except Exception:
            pass

    def set_mode(self, mode: str):
        """'auto' (governor in charge, full look to start with) or 'manual' (the preset buttons own the settings)."""
        mode = mode if mode in ("auto", "manual") else "auto"
        if mode == self.mode:
            return
        self.mode = mode
        self._bad_since = self._good_since = None
        if mode == "auto":
            self.apply(3, force=True)
        else:                                  # manual: the user's preset decides, but never keep a reduced resolution
            self.level = 3
            self.apply(3, force=True)

    @property
    def label(self) -> str:
        return ("Auto: " if self.mode == "auto" else "Manual: ") + LEVEL_NAMES[self.level]

    # ------------------------------------------------------------------ per frame
    def observe(self, frame_ms: float, gpu_ms: float | None):
        """Feed one frame (wall time between two presented frames, GPU pass time when known)."""
        if self.mode != "auto":
            return
        q = LEVELS[self.level][0]
        if getattr(self.app, "quality", None) != q:     # the governor owns app.quality in auto mode (the UI sets it once at start)
            self.app.quality = q
        if self._skip > 0:
            self._skip -= 1
            return
        frame_ms = min(frame_ms, 80.0)          # a stall (window drag, alt-tab, storm decode) is not a reason to degrade for long
        a = 0.12
        self._ema = frame_ms if self._ema is None else self._ema + a * (frame_ms - self._ema)
        now = time.perf_counter()
        T = self.target_ms
        if self._ema > self.DOWN_FACTOR * T:
            self._good_since = None
            if self._bad_since is None:
                self._bad_since = now
            if now - self._bad_since >= self.DOWN_AFTER_S and self.level > 0 and now - self._last_change >= 1.0:
                self._change(self.level - 1, now)
            return
        self._bad_since = None
        ok_gpu = gpu_ms is not None and gpu_ms < self.UP_GPU_FRACTION * T
        if ok_gpu and self._ema < 1.1 * T and self.level < 3:
            if self._good_since is None:
                self._good_since = now
            if now - self._good_since >= self.UP_AFTER_S and now - self._last_change >= self.MIN_DWELL_S:
                self._change(self.level + 1, now)
        else:
            self._good_since = None

    def _change(self, level: int, now: float):
        self._last_change = now
        self._bad_since = self._good_since = None
        self._ema = None
        self._skip = 10
        self.changes += 1
        self.apply(level)


class PerfManager:
    def __init__(self, app, mode: str = "auto"):
        self.app = app
        self.gc = GcControl()
        self.gov = QualityGovernor(app, mode)
        self.resolve: SceneResolve | None = None
        self._last_storm = None

    def scene_size(self, pw: int, ph: int) -> tuple[int, int]:
        s = getattr(self.app, "render_scale", 1.0)
        if s >= 0.999:
            return pw, ph
        return max(16, int(pw * s) & ~1), max(16, int(ph * s) & ~1)

    def resolve_for(self, w: int, h: int) -> SceneResolve:
        app = self.app
        if self.resolve is None or self.resolve.format != app.colour_format:
            if self.resolve is not None:
                self.resolve.destroy()
            self.resolve = SceneResolve(app.ctx.device, app.colour_format)
        self.resolve.ensure(w, h)
        return self.resolve

    def after_frame(self, frame_ms: float):
        app = self.app
        self.gov.observe(frame_ms, getattr(app, "gpu_ms", None))
        storm = getattr(getattr(app, "storm", None), "name", None)
        busy = storm != self._last_storm
        self._last_storm = storm
        self.gc.tick(busy=app.flow is not None and app.flow.progress()[0] < app.flow.progress()[1])
        if busy and self.gc.active:                # a storm switch is the quietest good moment for a full collection
            self.gc.collect_now(1)
