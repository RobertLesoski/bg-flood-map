"""Flow-paths overlay: `flowpaths.rgba.gz` (summary over all hours) as a colour map draped on the terrain.

Modes (selectable in the layer options):
    Maximum depth       R  (sqrt code, 0 .. 8 m)
    Mean discharge      G  (sqrt code, 0 .. 2 m2/s: mean over the hours of depth x speed)
    Hours wet           B  (0 .. 24 h)
Off by default; nothing is loaded until the layer is enabled (the file is decoded in a worker thread and uploaded in row
slices, so enabling it never stalls a frame).  The overlay reuses the terrain chunk mesh, restricted to the chunks that hold
any flow-path pixel, is lit by the terrain slope map so the relief stays readable, and blends with a user alpha.  The
colour ramps are defined once here (`RAMPS`) and feed both the shader (uniform) and the legend drawn in the UI.
"""
from __future__ import annotations

import threading

import numpy as np
import wgpu

from ...data.flow import block_max
from ..flowtex import COMMON_WGSL, CHUNK_TEXELS
from . import Layer
from .terrain import CHUNK_WGSL, _vertex_state

# six-stop ramps (linear RGB-ish display colours); t = code / 255
RAMPS = {
    "depth": [(0.99, 0.86, 0.22), (0.98, 0.64, 0.14), (0.92, 0.38, 0.20), (0.76, 0.18, 0.36), (0.46, 0.09, 0.46), (0.18, 0.05, 0.34)],
    "discharge": [(0.10, 0.20, 0.55), (0.12, 0.55, 0.80), (0.20, 0.78, 0.55), (0.80, 0.88, 0.25), (0.98, 0.62, 0.15), (0.85, 0.12, 0.10)],
    "hours": [(0.97, 0.78, 0.40), (0.52, 0.86, 0.88), (0.22, 0.66, 0.92), (0.14, 0.40, 0.88), (0.32, 0.16, 0.70), (0.26, 0.05, 0.40)],
}
MODES = ("depth", "discharge", "hours")
MODE_LABELS = ("Maximum depth (m)", "Mean discharge (m2/s)", "Hours wet (h)")
LEGEND = {"depth": (0.0, 8.0, True), "discharge": (0.0, 2.0, True), "hours": (0.0, 24.0, False)}   # (min, max, sqrt law)
PARAM_BYTES = 32 + 3 * 6 * 16


def decode_value(mode: str, code: float) -> float:
    """Physical value of a flowpaths code (0..255) in a mode (matches FLOW_FORMAT.md)."""
    lo, hi, sq = LEGEND[mode]
    t = code / 255.0
    return lo + (t * t if sq else t) * (hi - lo)


def ramp_colour(mode: str, t: float):
    """CPU twin of the shader ramp (for the legend and tests)."""
    stops = RAMPS[mode]
    x = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    return tuple(stops[i][k] * (1 - f) + stops[i + 1][k] * f for k in range(3))


WGSL = COMMON_WGSL + CHUNK_WGSL + """
struct FP { a: vec4<f32>, b: vec4<f32>, ramps: array<vec4<f32>, 18> };
@group(1) @binding(0) var fp_tex: texture_2d<f32>;
@group(1) @binding(1) var fp_samp: sampler;
@group(1) @binding(2) var<uniform> FPU: FP;      // a: mode, alpha, 0, 0

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) dist: f32,
};
@vertex fn vs(@builtin(vertex_index) vi: u32, inst: Inst) -> VOut {
  let m = mesh_vertex(vi, inst.c);
  let zg = height_texel(m.texel.x, m.texel.y) * frame.grid.w;
  let dist = length(frame.eye.xyz - vec3<f32>(m.wxy, zg));
  let lift = 0.05 + dist * 3.0e-5 + f32(1u << m.lod) * 0.02;
  var o: VOut;
  o.pos = frame.view_proj * vec4<f32>(vec3<f32>(m.wxy, zg + lift), 1.0);
  o.uv = world_to_uv(m.wxy);
  o.dist = dist;
  return o;
}

fn ramp(mode: u32, t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0) * 5.0;
  let i = min(u32(floor(x)), 4u);
  let f = x - f32(i);
  return mix(FPU.ramps[mode * 6u + i].rgb, FPU.ramps[mode * 6u + i + 1u].rgb, f);
}

@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(fp_tex, fp_samp, in.uv, 0.0);
  let cov = smoothstep(1.5, 16.0 / 255.0 * 255.0, c.r * 255.0) * smoothstep(0.2, 0.9, c.a);
  if (cov <= 0.01) { discard; }
  let mode = u32(FPU.a.x + 0.5);
  var t = c.r;
  if (mode == 1u) { t = c.g; }
  if (mode == 2u) { t = c.b; }
  var col = ramp(mode, t);
  let s = textureSampleLevel(slope_tex, slope_samp, in.uv, 0.0);
  let g = s.xy * frame.grid.w * frame.opts.w;
  let N = normalize(vec3<f32>(-g.x, -g.y, 1.0));
  let diff = clamp(dot(N, frame.sun.xyz), 0.0, 1.0);
  col = col * (0.62 + 0.55 * diff);
  col = mix(col, vec3<f32>(0.75, 0.80, 0.86), smoothstep(8000.0, 40000.0, in.dist) * 0.5);
  let a = cov * FPU.a.y;
  return vec4<f32>(col * a, a);
}
"""


class FlowpathsLayer(Layer):
    name = "flowpaths"
    title = "Flow paths"
    has_ui = True
    default_enabled = False

    def __init__(self):
        self.mode = 0
        self.alpha = 0.75
        self.pipeline = None
        self.tex = None
        self._loaded_for = None
        self._pending = None            # (storm name, array | Exception) from the worker
        self._loading = None
        self._arr = None
        self._rows = 0
        self.error = ""
        self.chunk_mask = None
        self.inst_buf = None
        self.draws = []
        self._bg = None
        self.n_chunks = 0

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        dev = app.ctx.device
        vf = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.bgl = dev.create_bind_group_layout(label="flowpaths-bgl", entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}}])
        self.sampler = dev.create_sampler(label="flowpaths-samp", mag_filter="linear", min_filter="linear",
                                          address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        self.params = dev.create_buffer(label="flowpaths-params", size=PARAM_BYTES,
                                        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        mod = dev.create_shader_module(label="flowpaths", code=WGSL)
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, self.bgl])
        blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.pipeline = dev.create_render_pipeline(
            label="flowpaths", layout=pl, vertex=_vertex_state(mod, "vs"),
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "greater"},
            multisample={"count": app.sample_count},
            fragment={"module": mod, "entry_point": "fs", "targets": [{"format": app.colour_format, "blend": blend}]})
        from .terrain import INST_BYTES, MAX_INSTANCES
        self.inst_buf = dev.create_buffer(label="flowpaths-inst", size=MAX_INSTANCES * INST_BYTES,
                                          usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        self._write_params(app)

    def shutdown(self, app):
        for b in (self.tex, self.inst_buf, getattr(self, "params", None)):
            try:
                if b is not None:
                    b.destroy()
            except Exception:
                pass
        app.ctx.untrack("flowpaths texture")

    def on_storm(self, app):
        self._loaded_for = None
        self._arr = None
        self.chunk_mask = None
        self.error = ""

    # ------------------------------------------------------------------------------------------ data
    def _start_load(self, app):
        storm = app.storm
        name = storm.name
        self._loading = name
        self._arr = None
        self._rows = 0
        self.error = ""

        def work():
            try:
                arr = storm.load_flowpaths()
                self._pending = (name, np.ascontiguousarray(arr))
            except Exception as e:                       # missing / corrupt file: report in the UI, never crash
                self._pending = (name, e)
        threading.Thread(target=work, name="bgv-flowpaths", daemon=True).start()

    def _write_params(self, app):
        buf = np.zeros(PARAM_BYTES // 4, np.float32)
        buf[0:4] = (self.mode, self.alpha, 0.0, 0.0)
        r = np.zeros((18, 4), np.float32)
        for mi, m in enumerate(MODES):
            for i, c in enumerate(RAMPS[m]):
                r[mi * 6 + i, :3] = c
        buf[8:] = r.reshape(-1)
        app.ctx.queue.write_buffer(self.params, 0, buf)

    def update(self, app, dt):
        name = app.storm.name
        if self._loading != name and self._loaded_for != name:
            self._start_load(app)
        p = self._pending
        if p is not None and p[0] == name and self._loading == name:
            self._pending = None
            if isinstance(p[1], Exception):
                self.error = f"{type(p[1]).__name__}: {p[1]}"
                self._loaded_for = name
                self._loading = None
            else:
                self._arr = p[1]
                self._rows = 0
                self._make_texture(app)
        elif p is not None and p[0] != name:
            self._pending = None
        if self._arr is not None and self._rows < self._arr.shape[0]:
            self._upload_rows(app)
        self._write_params(app)
        m = app.terrain_mesh
        if self.chunk_mask is not None and self._loaded_for == name and m is not None:
            data, self.draws = m.pack_visible(self.chunk_mask & m.vis)
            self.n_chunks = len(data)
            if len(data):
                app.ctx.queue.write_buffer(self.inst_buf, 0, data)
        else:
            self.draws = []
            self.n_chunks = 0

    def _make_texture(self, app):
        h, w = self._arr.shape[:2]
        if self.tex is None or self.tex.size[:2] != (w, h):
            if self.tex is not None:
                self.tex.destroy()
            self.tex = app.ctx.device.create_texture(label="flowpaths", size=(w, h, 1), format="rgba8unorm",
                                                     usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self._bg = app.ctx.device.create_bind_group(label="flowpaths-bg", layout=self.bgl, entries=[
                {"binding": 0, "resource": self.tex.create_view()},
                {"binding": 1, "resource": self.sampler},
                {"binding": 2, "resource": {"buffer": self.params, "offset": 0, "size": PARAM_BYTES}}])
            app.ctx.track("flowpaths texture", w * h * 4)

    def _upload_rows(self, app, budget=6 << 20):
        arr = self._arr
        h, w = arr.shape[:2]
        rows = min(h - self._rows, max(1, budget // (w * 4)))
        app.ctx.queue.write_texture({"texture": self.tex, "mip_level": 0, "origin": (0, self._rows, 0)},
                                    arr[self._rows:self._rows + rows],
                                    {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": rows}, (w, rows, 1))
        self._rows += rows
        if self._rows >= h:
            self.chunk_mask = block_max(arr[..., 0], CHUNK_TEXELS) > 0
            self._loaded_for = self._loading
            self._loading = None

    def draw(self, app, pass_):
        m = app.terrain_mesh
        if not self.draws or m is None or self._bg is None or self._loaded_for != app.storm.name or self.error:
            return
        pass_.set_pipeline(self.pipeline)
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, self._bg)
        m.draw(pass_, self.draws, self.inst_buf)

    # ------------------------------------------------------------------------------------------ UI / state
    def ui(self, app):
        from imgui_bundle import imgui
        imgui.set_next_item_width(-120)
        ch, i = imgui.combo("Show##fp", self.mode, list(MODE_LABELS))
        if ch:
            self.mode = i
        imgui.set_next_item_width(-120)
        ch, v = imgui.slider_float("Opacity##fp", self.alpha, 0.1, 1.0, "%.2f")
        if ch:
            self.alpha = v
        mode = MODES[self.mode]
        lo, hi, sq = LEGEND[mode]
        dl = imgui.get_window_draw_list()
        p0 = imgui.get_cursor_screen_pos()
        wbar = imgui.get_content_region_avail().x
        n = 24
        for k in range(n):
            c = ramp_colour(mode, k / (n - 1))
            col = imgui.get_color_u32((c[0], c[1], c[2], 1.0))
            dl.add_rect_filled((p0.x + wbar * k / n, p0.y), (p0.x + wbar * (k + 1) / n + 1, p0.y + 10), col)
        imgui.dummy((wbar, 12))
        imgui.text_disabled(f"{lo:g}")
        imgui.same_line()
        imgui.text_disabled(f"... {decode_value(mode, 255 * 0.5):.3g} ... {hi:g}")
        if self.error:
            imgui.text_colored((1.0, 0.45, 0.4, 1.0), self.error[:80])
        elif self._loaded_for != app.storm.name:
            imgui.text_disabled("loading flow paths...")
        else:
            imgui.text_disabled(f"{self.n_chunks} chunks drawn")

    def save_state(self):
        return {"mode": self.mode, "alpha": self.alpha}

    def load_state(self, d):
        self.mode = int(d.get("mode", self.mode)) % len(MODES)
        self.alpha = float(d.get("alpha", self.alpha))
