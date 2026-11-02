"""Roads layer: the road network as flat ribbons draped on the (exaggerated) terrain.

Data comes from `bgv.gpu.scene_data` (same background load as the buildings): `RoadRibbon` vertices (x, y, across, along,
class) are uploaded in slices.  Z is sampled from the height texture in the vertex shader:
    z = ground_height(xy) * E + lift,   lift = (0.10 + 1.5e-4 * distance) * (0.5 + 0.5 E)   (no z-fighting with the terrain
    mesh at any LOD; the pipeline also has a small positive depth bias, reversed-Z means "closer" = larger depth)
Fragment: asphalt colour by class (dark grey, lighter for major roads), light edge lines, dashed centre line on wide roads,
and FLOODED stretches coloured orange -> red by the interpolated flow depth at the displayed time (>= 0.1 m, the same
threshold as the road statistics, `flow_at` = identical decoding to the water layer).
"""
from __future__ import annotations

import time

import numpy as np
import wgpu

from ...data.roads import ROAD_CLASSES, ROAD_WIDTH_M
from ..cull import planes_for
from ..flowtex import COMMON_WGSL
from ..scene_data import get_scene
from ..stats_hub import get_hub
from . import Layer
from .buildings import UPLOAD_BUDGET, TileCuller, srgb_to_linear, hex_rgb

# sRGB asphalt per class: motorway, motorway_link, trunk, primary, primary_link, secondary, tertiary, unclassified,
# residential, other
ASPHALT_HEX = ("#7b8089", "#6d727a", "#787d85", "#72777f", "#656a72", "#666b73", "#5a5f67", "#53575f", "#4d5159", "#484c53")


def _wgsl_arr(name, vals, ty="f32"):
    return f"const {name} = array<{ty}, {len(vals)}>({', '.join(f'{float(v):.5f}' for v in vals)});\n"


def _wgsl_col_arr(name, hexes):
    cols = [srgb_to_linear(hex_rgb(h)) for h in hexes]
    return (f"const {name} = array<vec3<f32>, {len(cols)}>("
            + ", ".join(f"vec3<f32>({c[0]:.5f}, {c[1]:.5f}, {c[2]:.5f})" for c in cols) + ");\n")


ROADS_WGSL = COMMON_WGSL + _wgsl_arr("ROAD_W", ROAD_WIDTH_M) + _wgsl_col_arr("ASPHALT", ASPHALT_HEX) + """
struct RParams {
  opt: vec4<f32>,       // x flood colour strength, y centre line, z lift scale, w unused
};
@group(1) @binding(0) var<uniform> R: RParams;

struct RIn {
  @location(0) xy: vec2<f32>,
  @location(1) across: f32,
  @location(2) along: f32,
  @location(3) cls: u32,
};
struct ROut {
  @builtin(position) pos: vec4<f32>,
  @location(0) wpos: vec3<f32>,
  @location(1) across: f32,
  @location(2) along: f32,
  @location(3) @interpolate(flat) cls: u32,
};

@vertex fn vs_road(v: RIn) -> ROut {
  let E = frame.grid.w;
  let gz = ground_height(v.xy) * E;
  let dist = length(frame.eye.xyz - vec3<f32>(v.xy, gz));
  let lift = (0.10 + 1.5e-4 * dist) * (0.5 + 0.5 * E) * R.opt.z;
  // a flooded stretch rides on the water surface (same Z law as the water mesh) so the colour is not hidden by the water
  let fl = flow_at(world_to_uv(v.xy));
  var dpt = 0.0;
  if (fl.depth > frame.enc.z && fl.valid > 0.5 && R.opt.x > 0.0) { dpt = fl.depth; }
  let surf = select(0.0, dpt * frame.enc.w + 0.10 + 3.0e-5 * dist, dpt > 0.0);
  var o: ROut;
  o.wpos = vec3<f32>(v.xy, gz + lift + surf);
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  o.across = v.across;
  o.along = v.along;
  o.cls = v.cls;
  return o;
}

@fragment fn fs_road(in: ROut) -> @location(0) vec4<f32> {
  let E = frame.grid.w;
  let to_eye = frame.eye.xyz - in.wpos;
  let dist = length(to_eye);
  let V = to_eye / max(dist, 1e-3);
  let ci = min(in.cls, 9u);
  let width = ROAD_W[ci];
  let hw = 0.5 * width;
  var base = ASPHALT[ci];
  // faint tyre-wear banding, only where it is resolvable
  let uv = world_to_uv(in.wpos.xy);
  let s = textureSample(slope_tex, slope_samp, uv);
  let N = normalize(vec3<f32>(-s.x * E, -s.y * E, 1.0));
  let L = frame.sun.xyz;
  let diff = max(dot(N, L), 0.0);
  let amb = mix(vec3<f32>(0.30, 0.29, 0.27), vec3<f32>(0.42, 0.52, 0.70), N.z * 0.5 + 0.5);
  let light = vec3<f32>(1.0, 0.95, 0.86) * frame.sun.w * diff * 1.05 + amb * 0.80;
  // pixel footprint in metres across the ribbon (for anti-aliased lines)
  let px_m = max(fwidth(in.across) * hw, 1e-4);
  let a = abs(in.across);
  let edge_m = (1.0 - a) * hw;                         // metres from the nearest edge
  // light edge line 0.35 m inside the kerb, 0.16 m wide; fades out when it is thinner than a pixel
  let aa = max(px_m, 0.04);
  var line = 1.0 - smoothstep(0.11, 0.11 + aa, abs(edge_m - 0.45));
  line = clamp(line, 0.0, 1.0) * (1.0 - smoothstep(0.25, 0.9, px_m));
  var col = base * light;
  col = mix(col, vec3<f32>(0.78, 0.80, 0.84) * light, line * 0.75);
  // dashed centre line on wide roads: 3 m dash every 12 m
  if (R.opt.y > 0.5 && width >= 9.0 && ci != 1u && ci != 4u) {
    let ph = fract(in.along / 12.0);
    let dash = smoothstep(0.0, 0.01, ph) * (1.0 - smoothstep(0.25, 0.26, ph));
    let cl = (1.0 - smoothstep(0.08, 0.08 + max(px_m, 0.04), abs(in.across) * hw)) * dash;
    col = mix(col, vec3<f32>(0.82, 0.68, 0.22) * light, cl * 0.8 * (1.0 - smoothstep(0.1, 0.4, px_m)));
  }
  // flooded stretch: orange -> red with depth
  let fl = flow_at(uv);
  let d = fl.depth;
  let fk = smoothstep(0.085, 0.115, d) * smoothstep(0.2, 0.9, fl.valid) * R.opt.x;
  if (fk > 0.0) {
    let orange = vec3<f32>(1.0, 0.30, 0.01);
    let red = vec3<f32>(0.80, 0.07, 0.04);
    let fc = mix(orange, red, smoothstep(0.1, 1.0, d));
    let lum = dot(light, vec3<f32>(0.3333));
    col = mix(col, fc * (0.55 + 0.75 * lum), fk * 0.92);
  }
  col = haze(col, dist, -V);
  return vec4<f32>(col, 1.0);
}
"""


class RoadsLayer(Layer):
    name = "roads"
    title = "Roads"
    has_ui = True

    def __init__(self):
        self.flood_colour = 1.0
        self.centre_line = True
        self.pipe = None
        self.scene = None
        self.vbuf = self.ibuf = self.params_buf = None
        self.bind_group = None
        self._up_v = self._up_i = 0
        self._uploaded = False
        self._draws: list[tuple[int, int]] = []
        self._culler = None
        self._last_p = b""
        self.tiles_visible = 0
        self.tris_drawn = 0
        self.cull_ms = 0.0
        self._p = np.zeros(4, np.float32)

    def init(self, app):
        dev = app.ctx.device
        get_hub(app)
        self.scene = get_scene(app)
        mod = dev.create_shader_module(label="roads", code=ROADS_WGSL)
        vis = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.bgl = dev.create_bind_group_layout(label="road-bgl", entries=[
            {"binding": 0, "visibility": vis, "buffer": {"type": "uniform"}}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, self.bgl])
        vl = {"array_stride": 20, "step_mode": "vertex", "attributes": [
            {"format": "float32x2", "offset": 0, "shader_location": 0},
            {"format": "float32", "offset": 8, "shader_location": 1},
            {"format": "float32", "offset": 12, "shader_location": 2},
            {"format": "uint32", "offset": 16, "shader_location": 3}]}
        self.pipe = dev.create_render_pipeline(
            label="roads", layout=pl, vertex={"module": mod, "entry_point": "vs_road", "buffers": [vl]},
            primitive={"topology": "triangle-list", "cull_mode": "none", "front_face": "ccw"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": True, "depth_compare": "greater",
                           "depth_bias": 2, "depth_bias_slope_scale": 1.0},
            multisample={"count": app.sample_count},
            fragment={"module": mod, "entry_point": "fs_road", "targets": [{"format": app.colour_format}]})
        self.params_buf = dev.create_buffer(label="road-params", size=16,
                                            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.bind_group = dev.create_bind_group(label="road-bg", layout=self.bgl, entries=[
            {"binding": 0, "resource": {"buffer": self.params_buf, "offset": 0, "size": 16}}])
        if not hasattr(app, "ready_waiters"):
            app.ready_waiters = {}
        app.ready_waiters[self.name] = lambda timeout: self._wait_ready(app, timeout)

    def _wait_ready(self, app, timeout: float = 120.0):
        """Headless tools only: block until the road mesh is loaded and fully uploaded."""
        if self.scene.wait(timeout):
            if self.vbuf is None:
                self._create_buffers(app)
            while not self._uploaded:
                self._upload_slice(app)

    def _create_buffers(self, app):
        dev = app.ctx.device
        m = self.scene.road_mesh
        self.vbuf = dev.create_buffer(label="road-vertices", size=max(m.vertices.nbytes, 16),
                                      usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        self.ibuf = dev.create_buffer(label="road-indices", size=max(m.indices.nbytes, 16),
                                      usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST)
        app.ctx.track("roads mesh", m.vertices.nbytes + m.indices.nbytes)
        self._up_v = self._up_i = 0
        self._uploaded = m.vertices.nbytes == 0

    def _upload_slice(self, app):
        m = self.scene.road_mesh
        vb = np.ascontiguousarray(m.vertices).view(np.uint8).reshape(-1)
        ib = np.ascontiguousarray(m.indices).view(np.uint8).reshape(-1)
        q = app.ctx.queue
        budget = UPLOAD_BUDGET
        if self._up_v < vb.size:
            n = min(budget, vb.size - self._up_v)
            q.write_buffer(self.vbuf, self._up_v, vb[self._up_v:self._up_v + n])
            self._up_v += n
            budget -= n
        if budget > 0 and self._up_i < ib.size:
            n = min(budget, ib.size - self._up_i)
            q.write_buffer(self.ibuf, self._up_i, ib[self._up_i:self._up_i + n])
            self._up_i += n
        self._uploaded = self._up_v >= vb.size and self._up_i >= ib.size

    def shutdown(self, app):
        for b in (self.vbuf, self.ibuf, self.params_buf):
            try:
                if b is not None:
                    b.destroy()
            except Exception:
                pass
        app.ctx.untrack("roads mesh")
        self.vbuf = self.ibuf = None

    def update(self, app, dt):
        sc = self.scene
        if sc.state != "ready":
            return
        if self.vbuf is None:
            self._create_buffers(app)
        if not self._uploaded:
            self._upload_slice(app)
            return
        t0 = time.perf_counter()
        if self._culler is None:
            self._culler = TileCuller(sc.road_mesh.tiles)
        self._draws = self._culler.draws(app.view_proj, app.exag, 1.0, 1.0, 6.0 * max(app.exag, 1.0), planes_for(app))
        self.tiles_visible = self._culler.visible_count
        self.tris_drawn = sum(c for _, c in self._draws) // 3
        self.cull_ms = (time.perf_counter() - t0) * 1000.0
        self._p[:] = (self.flood_colour, 1.0 if self.centre_line else 0.0, 1.0, 0.0)
        raw = self._p.tobytes()
        if raw != self._last_p:
            self._last_p = raw
            app.ctx.queue.write_buffer(self.params_buf, 0, self._p)

    def draw(self, app, pass_):
        if not self._uploaded or self.vbuf is None or not self._draws:
            return
        pass_.set_pipeline(self.pipe)
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, self.bind_group)
        pass_.set_vertex_buffer(0, self.vbuf)
        pass_.set_index_buffer(self.ibuf, "uint32")
        for first, count in self._draws:
            pass_.draw_indexed(count, 1, first, 0, 0)

    def ui(self, app):
        from imgui_bundle import imgui
        sc = self.scene
        if sc.state != "ready":
            imgui.text_disabled("road data loading (see Buildings)")
            return
        ch, v = imgui.slider_float("Flood colour", self.flood_colour, 0.0, 1.0, "%.2f")
        if ch:
            self.flood_colour = v
        _, self.centre_line = imgui.checkbox("Centre lines", self.centre_line)
        f = getattr(app, "flood", None) or {}
        if f.get("ready"):
            imgui.text_disabled(f"as of hour {f['hour']}: {f['road_pct']:.1f} % flooded ({f['road_flooded_km']:.1f} km)")
        imgui.text_disabled(f"tiles {self.tiles_visible}/{sc.road_mesh.tiles.n}  tris {self.tris_drawn / 1e3:.0f}k")

    def save_state(self):
        return {"flood_colour": self.flood_colour, "centre_line": self.centre_line}

    def load_state(self, d):
        self.flood_colour = float(d.get("flood_colour", self.flood_colour))
        self.centre_line = bool(d.get("centre_line", self.centre_line))
