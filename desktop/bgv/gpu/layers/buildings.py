"""Buildings layer: 20k extruded footprints standing on the exaggerated terrain, coloured by the flood tier of the displayed
snapshot, with a selection outline + emissive pulse.

Data: `bgv.gpu.scene_data` loads the cached mesh in a worker thread (never blocks the first frames; the sidebar shows a
progress bar and `app.scene_status`).  The mesh is uploaded in slices of <= UPLOAD_BUDGET bytes per frame.
Per-building state lives in ONE small storage buffer (`depth m, tier`) that is rewritten only when the snapshot published by
`bgv.gpu.stats_hub` (app.flood) changes - so the colours always agree with the statistics ("as of hour N").

Geometry: world Z = base_z[b] * E + local_z * height_scale  (E = frame.grid.w = vertical exaggeration; building height is NOT
exaggerated unless `height_scale` > 1).  Wall bottoms are pushed down by a skirt (0.8 + 0.35 E metres) so no building
floats over the terrain mesh (coarser LODs, slopes, exaggeration).  Back faces are culled (mesh is CCW from outside),
reversed-Z (`greater`), depth written, MSAA via app.sample_count.  Spatial tiles are frustum culled on the CPU and merged
into a few draw_indexed calls.

Selection: `app.selection["building"]` (id or None) is highlighted (emissive pulse + silhouette outline drawn as an expanded
back-face shell); `app.selection["hover_building"]` (optional) gets a soft highlight.
"""
from __future__ import annotations

import math
import time

import numpy as np
import wgpu

from ...data.stats import TIER_NAMES
from .. import fastcull
from ..cull import frustum_planes as _frustum_planes, planes_for
from ..flowtex import COMMON_WGSL
from ..scene_data import get_scene
from ..stats_hub import get_hub
from . import Layer

UPLOAD_BUDGET = 3 << 20
NONE_ID = 0xFFFFFFFF
PARAM_BYTES = 16 + 16 + 16 * 4 + 16


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# the web viewer's tier colours (shallow, moderate, deep, severe)
TIER_HEX = ("#4dabf7", "#00c9c9", "#ff8c00", "#e74c3c")
TIER_LINEAR = [tuple(float(v) for v in srgb_to_linear(hex_rgb(h))) for h in TIER_HEX]

BUILDINGS_WGSL = COMMON_WGSL + """
struct BParams {
  sel: vec4<u32>,                // x selected id, y hover id, z count, w flags (bit0 water line)
  opt: vec4<f32>,                // x height scale, y outline width (m), z flood tint, w skirt (m)
  pulse: vec4<f32>,              // x selection pulse 0..1
  tier: array<vec4<f32>, 4>,     // linear tier colours
};
@group(1) @binding(0) var<uniform> P: BParams;
@group(1) @binding(1) var<storage, read> dyn: array<vec2<f32>>;       // depth m, tier (0 dry, 1..4)
@group(1) @binding(2) var<storage, read> stat: array<vec4<f32>>;      // centroid x, y, base z (m), shadow flag

struct VIn {
  @location(0) pos: vec3<f32>,
  @location(1) nrm: vec4<f32>,
  @location(2) bid: u32,
};
struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) wpos: vec3<f32>,
  @location(1) nrm: vec3<f32>,
  @location(2) @interpolate(flat) bid: u32,
  @location(3) zl: f32,
};

fn place(v: VIn, expand: f32) -> vec3<f32> {
  let s = stat[v.bid];
  let E = frame.grid.w;
  var zl = v.pos.z * P.opt.x;
  if (v.nrm.z < 0.5 && v.pos.z < 0.01) { zl = -P.opt.w; }
  var xy = v.pos.xy;
  if (expand > 0.0) {
    let d = xy - s.xy;
    xy += d / max(length(d), 1e-3) * expand;
    if (v.pos.z > 0.01) { zl += expand; }
  }
  return vec3<f32>(xy, s.z * E + zl);
}

@vertex fn vs_main(v: VIn) -> VOut {
  var o: VOut;
  o.wpos = place(v, 0.0);
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  o.nrm = v.nrm.xyz;
  o.bid = v.bid;
  o.zl = o.wpos.z - stat[v.bid].z * frame.grid.w;
  return o;
}

fn hash11(x: u32) -> f32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  return f32(h & 0xffffffu) / 16777215.0;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let E = frame.grid.w;
  let N = normalize(in.nrm);
  let L = frame.sun.xyz;
  let to_eye = frame.eye.xyz - in.wpos;
  let dist = length(to_eye);
  let V = to_eye / max(dist, 1e-3);
  let roof = N.z > 0.5;
  let hv = hash11(in.bid);
  // neutral light grey / white; roofs a little darker and warmer, every building slightly different
  var alb = select(vec3<f32>(0.80, 0.79, 0.77), vec3<f32>(0.60, 0.58, 0.56), roof);
  alb *= 0.90 + 0.18 * hv;
  alb += (vec3<f32>(hv, 1.0 - hv, 0.5) - vec3<f32>(0.5)) * 0.025;
  // fake AO: walls darken toward the ground, roofs stay clean
  let ao = select(mix(0.56, 1.0, smoothstep(0.0, 5.0, in.zl)), 1.0, roof);
  let diff = max(dot(N, L), 0.0);
  let amb = mix(vec3<f32>(0.30, 0.29, 0.27), vec3<f32>(0.42, 0.52, 0.70), N.z * 0.5 + 0.5);
  let light = (vec3<f32>(1.0, 0.95, 0.86) * frame.sun.w * diff * 1.15 + amb * 0.88) * ao;
  var col = alb * light;

  let d = dyn[in.bid];
  let tier = i32(d.y + 0.5);
  if (tier >= 1) {
    let tc = P.tier[tier - 1].xyz;
    let wl = d.x * frame.enc.w;
    let under = 1.0 - smoothstep(wl - 0.12, wl + 0.12, in.zl);
    let lum = dot(light, vec3<f32>(0.3333));
    let k = P.opt.z * mix(0.94, 1.0, under);
    let tshade = clamp(0.34 + 0.60 * lum, 0.0, 1.0) * (0.96 + 0.08 * hv);
    col = mix(col, tc * tshade, k);
    col *= 1.0 - 0.22 * under * select(1.0, 0.0, roof);
    if ((P.sel.w & 1u) != 0u && !roof) {
      let line = 1.0 - smoothstep(0.0, 0.16, abs(in.zl - wl));
      col = mix(col, tc * 1.35, line * 0.65);
    }
  }
  // selection / hover
  if (in.bid == P.sel.x) {
    let rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.0);
    col = mix(col, vec3<f32>(1.0, 0.82, 0.22), 0.16 + 0.14 * P.pulse.x) + vec3<f32>(1.0, 0.7, 0.1) * rim * 0.30;
  } else if (in.bid == P.sel.y) {
    col = mix(col, vec3<f32>(1.0, 1.0, 1.0), 0.22);
  }
  col = haze(col, dist, -V);
  return vec4<f32>(col, 1.0);
}

// ---- planar sun shadows on the ground (see BuildingsLayer.draw): the extruded mesh is projected along the sun onto the
// terrain; fragments only write the ALPHA channel with the `min` blend op (idempotent where projections overlap), then one
// full-screen pass multiplies the colour by that alpha (dst.rgb * dst.a) and restores alpha to 1.
struct SOut { @builtin(position) pos: vec4<f32>, @location(0) xy: vec2<f32> };
@vertex fn vs_shadow(v: VIn) -> SOut {
  let s = stat[v.bid];
  let E = frame.grid.w;
  var zl = v.pos.z * P.opt.x;
  if (v.nrm.z < 0.5 && v.pos.z < 0.01) { zl = 0.0; }
  let L = frame.sun.xyz;
  let xy = v.pos.xy - L.xy / max(L.z, 0.14) * zl;
  let gz = ground_height(xy) * E;
  let dist = length(frame.eye.xyz - vec3<f32>(xy, gz));
  let lift = (0.10 + 1.5e-4 * dist) * (0.5 + 0.5 * E) + 0.05 + 1.0e-4 * dist;
  var o: SOut;
  o.pos = frame.view_proj * vec4<f32>(xy, gz + lift, 1.0);
  if (s.w < 0.5) { o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); }         // buried building: no shadow (collapsed triangle)
  o.xy = xy;
  return o;
}
@fragment fn fs_shadow(in: SOut) -> @location(0) vec4<f32> {
  let fl = flow_at(world_to_uv(in.xy));
  if (fl.depth > frame.enc.z) { discard; }            // under water: no ground shadow
  return vec4<f32>(0.0, 0.0, 0.0, P.pulse.y);
}
@vertex fn vs_apply(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.5, 1.0);
}
@fragment fn fs_apply() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }

@vertex fn vs_outline(v: VIn) -> VOut {
  var o: VOut;
  o.wpos = place(v, P.opt.y);
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  o.nrm = v.nrm.xyz;
  o.bid = v.bid;
  o.zl = 0.0;
  return o;
}
@vertex fn vs_fill(v: VIn) -> VOut {
  var o: VOut;
  o.wpos = place(v, 0.0);
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  o.nrm = v.nrm.xyz;
  o.bid = v.bid;
  o.zl = 0.0;
  return o;
}
@fragment fn fs_fill(in: VOut) -> @location(0) vec4<f32> {
  let N = normalize(in.nrm);
  let shade = 0.75 + 0.25 * max(dot(N, frame.sun.xyz), 0.0);
  let a = 0.20 + 0.14 * P.pulse.x;
  return vec4<f32>(vec3<f32>(1.0, 0.70, 0.10) * shade * a, a);        // premultiplied amber
}
@fragment fn fs_outline(in: VOut) -> @location(0) vec4<f32> {
  let pulse = 0.85 + 0.35 * P.pulse.x;
  return vec4<f32>(vec3<f32>(1.0, 0.62, 0.06) * pulse + vec3<f32>(0.2, 0.25, 0.0), 1.0);
}
"""


def frustum_planes(vp: np.ndarray) -> np.ndarray:
    return _frustum_planes(vp)                                               # reversed-Z, infinite far (bgv/gpu/cull.py)


class TileCuller:
    """Frustum culling of tile boxes (box centre / extent form, two small matmuls on (T, 3) arrays); the result and the merged
    draw list are cached while the camera and the Z scales do not change."""

    def __init__(self, tiles):
        self.tiles = tiles
        self._zkey = None
        self._vkey = None
        self._vpkey = None
        self._draws = []
        self.visible_count = 0
        self._ctr = np.empty((tiles.n, 3), np.float64)
        self._ext = np.empty((tiles.n, 3), np.float64)
        self._ctr[:, 0] = (tiles.x0 + tiles.x1) * 0.5
        self._ctr[:, 1] = (tiles.y0 + tiles.y1) * 0.5
        self._ext[:, 0] = (tiles.x1 - tiles.x0) * 0.5
        self._ext[:, 1] = (tiles.y1 - tiles.y0) * 0.5
        self._vis = np.zeros(tiles.n, bool)
        self._first = np.ascontiguousarray(tiles.first, np.int64)
        self._count = np.ascontiguousarray(tiles.count, np.int64)
        self._run_first = np.zeros(tiles.n + 1, np.int64)
        self._run_count = np.zeros(tiles.n + 1, np.int64)

    def visible(self, vp, exag: float, hscale: float, skirt: float, extra: float = 2.0, planes=None):
        t = self.tiles
        key = (exag, hscale, skirt, extra)
        if key != self._zkey:
            self.z0 = t.zlo * exag - skirt - extra
            self.z1 = t.zhi * exag + t.hmax * hscale + extra
            self._ctr[:, 2] = (self.z0 + self.z1) * 0.5
            self._ext[:, 2] = (self.z1 - self.z0) * 0.5
            self._zkey = key
        P = _frustum_planes(vp) if planes is None else planes
        if fastcull.available():
            fastcull.tile_visible(self._ctr, self._ext, P, self._vis)
            return self._vis
        S = self._ctr @ P[:, :3].T
        S += self._ext @ np.abs(P[:, :3]).T
        S += P[:, 3]
        return (S >= 0.0).all(axis=1)

    def draws(self, vp, exag, hscale, skirt, extra=2.0, planes=None):
        """[(first_index, index_count)] of the visible tiles (cached when the visible set did not change)."""
        vpk = (np.asarray(vp).tobytes(), exag, hscale, skirt, extra)
        if vpk == self._vpkey:
            return self._draws
        self._vpkey = vpk
        vis = self.visible(vp, exag, hscale, skirt, extra, planes)
        k = vis.tobytes()
        if k != self._vkey:
            self._vkey = k
            if fastcull.available():
                r = fastcull.tile_runs(vis, self._first, self._count, 48, self._run_first, self._run_count)
                self._draws = list(zip(self._run_first[:r].tolist(), self._run_count[:r].tolist()))
            else:
                self._draws = merge_runs(self.tiles, vis)
            self.visible_count = int(vis.sum())
        return self._draws


def cull_tiles(tiles, vp, exag: float, hscale: float, skirt: float, extra: float = 2.0, eye=None, max_dist: float = 0.0):
    """bool[T]: tile boxes (ground-metre footprint, exaggerated Z range) intersecting the view frustum."""
    vis = TileCuller(tiles).visible(vp, exag, hscale, skirt, extra)
    if eye is not None and max_dist > 0:
        dx = np.maximum(np.maximum(tiles.x0 - eye[0], eye[0] - tiles.x1), 0.0)
        dy = np.maximum(np.maximum(tiles.y0 - eye[1], eye[1] - tiles.y1), 0.0)
        vis &= (dx * dx + dy * dy) <= max_dist * max_dist
    return vis


def merge_runs(tiles, vis: np.ndarray, max_draws: int = 48):
    """[(first_index, index_count)] covering the visible tiles; adjacent tiles are one draw call (gaps merged when many)."""
    idx = np.flatnonzero(vis)
    if idx.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, np.diff(idx) > 1])
    run_lo = idx[starts]
    run_hi = idx[np.r_[starts[1:] - 1, idx.size - 1]]
    if run_lo.size > max_draws:                       # merge the cheapest gaps until the call budget is met
        gap = tiles.first[run_lo[1:]] - (tiles.first[run_hi[:-1]] + tiles.count[run_hi[:-1]])
        keep = np.ones(run_lo.size - 1, bool)
        for g in np.argsort(gap, kind="stable")[: run_lo.size - max_draws]:
            keep[g] = False
        new_lo = np.r_[run_lo[0], run_lo[1:][keep]]
        new_hi = np.r_[run_hi[:-1][keep], run_hi[-1]]
        run_lo, run_hi = new_lo, new_hi
    first = tiles.first[run_lo]
    end = tiles.first[run_hi] + tiles.count[run_hi]
    return [(int(a), int(b - a)) for a, b in zip(first, end)]


class BuildingsLayer(Layer):
    name = "buildings"
    title = "Buildings"
    has_ui = True

    def __init__(self):
        self.height_scale = 1.0
        self.flood_tint = 1.0
        self.water_line = True
        self.outline = True
        self.pipe = None
        self.pipe_shadow = None
        self.pipe_apply = None
        self.shadows = True
        self.shadow_strength = 0.38
        self.shadow_max_dist = 5000.0
        self._cam_dist = 0.0
        self.scene = None
        self.hub = None
        self.vbuf = self.ibuf = self.params_buf = self.dyn_buf = self.stat_buf = None
        self.bind_group = None
        self._up_v = 0
        self._up_i = 0
        self._uploaded = False
        self._state_version = -1
        self._draws: list[tuple[int, int]] = []
        self._culler = None
        self._last_params = b""
        self._pkey = None
        self._tiers_set = False
        self.tiles_visible = 0
        self.tris_drawn = 0
        self.cull_ms = 0.0
        self.state_upload_ms = 0.0
        self.mesh_upload_ms = 0.0
        self._params = np.zeros(PARAM_BYTES // 4, np.float32)
        self._params_u = self._params.view(np.uint32)

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        dev = app.ctx.device
        self.hub = get_hub(app)
        self.scene = get_scene(app)
        mod = dev.create_shader_module(label="buildings", code=BUILDINGS_WGSL)
        vis = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.bgl = dev.create_bind_group_layout(label="bld-bgl", entries=[
            {"binding": 0, "visibility": vis, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": vis, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": vis, "buffer": {"type": "read-only-storage"}}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, self.bgl])
        vbuf_layout = {"array_stride": 20, "step_mode": "vertex", "attributes": [
            {"format": "float32x3", "offset": 0, "shader_location": 0},
            {"format": "snorm8x4", "offset": 12, "shader_location": 1},
            {"format": "uint32", "offset": 16, "shader_location": 2}]}
        ms = {"count": app.sample_count}
        ds = {"format": app.depth_format, "depth_write_enabled": True, "depth_compare": "greater"}
        tgt = [{"format": app.colour_format}]
        self.pipe = dev.create_render_pipeline(
            label="buildings", layout=pl, vertex={"module": mod, "entry_point": "vs_main", "buffers": [vbuf_layout]},
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"}, depth_stencil=ds,
            multisample=ms, fragment={"module": mod, "entry_point": "fs_main", "targets": tgt})
        self.mod, self.vbuf_layout, self.pl = mod, vbuf_layout, pl
        one = {"src_factor": "one", "dst_factor": "zero", "operation": "add"}
        self.pipe_shadow = dev.create_render_pipeline(
            label="buildings-shadow", layout=pl,
            vertex={"module": mod, "entry_point": "vs_shadow", "buffers": [vbuf_layout]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil=dict(ds, depth_write_enabled=False), multisample=ms,
            fragment={"module": mod, "entry_point": "fs_shadow", "targets": [{
                "format": app.colour_format, "write_mask": wgpu.ColorWrite.ALPHA,
                "blend": {"color": one, "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "min"}}}]})
        self.pipe_apply = dev.create_render_pipeline(
            label="buildings-shadow-apply", layout=pl, vertex={"module": mod, "entry_point": "vs_apply"},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "always"},
            multisample=ms, fragment={"module": mod, "entry_point": "fs_apply", "targets": [{
                "format": app.colour_format,
                "blend": {"color": {"src_factor": "zero", "dst_factor": "dst-alpha", "operation": "add"}, "alpha": one}}]})
        self.params_buf = dev.create_buffer(label="bld-params", size=PARAM_BYTES,
                                            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._state_version = -1
        if not hasattr(app, "ready_waiters"):
            app.ready_waiters = {}
        app.ready_waiters[self.name] = lambda timeout: self._wait_ready(app, timeout)

    def _wait_ready(self, app, timeout: float = 120.0):
        """Headless tools only (tools.wait_for_flow): block until the scene is loaded, the mesh is fully uploaded and the
        statistics of the displayed hour exist, so a single --render frame shows everything."""
        sc = self.scene
        if sc.wait(timeout):
            if self.vbuf is None:
                self._create_buffers(app)
            while not self._uploaded:
                self._upload_slice(app)
            self.hub.wait_ready(timeout)

    def _create_buffers(self, app):
        """Called once the scene data is ready (main thread)."""
        dev = app.ctx.device
        sc = self.scene
        b, m = sc.buildings, sc.mesh
        n = b.n
        usage = wgpu.BufferUsage.COPY_DST
        self.vbuf = dev.create_buffer(label="bld-vertices", size=m.vertices.nbytes, usage=usage | wgpu.BufferUsage.VERTEX)
        self.ibuf = dev.create_buffer(label="bld-indices", size=m.indices.nbytes, usage=usage | wgpu.BufferUsage.INDEX)
        self.dyn_buf = dev.create_buffer(label="bld-dyn", size=max(n, 1) * 8, usage=usage | wgpu.BufferUsage.STORAGE)
        self.stat_buf = dev.create_buffer(label="bld-static", size=max(n, 1) * 16, usage=usage | wgpu.BufferUsage.STORAGE)
        stat = np.zeros((n, 4), np.float32)
        stat[:, 0] = np.nan_to_num(b.cx)
        stat[:, 1] = np.nan_to_num(b.cy)
        stat[:, 2] = sc.base_z
        stat[:, 3] = sc.shadow_ok.astype(np.float32)          # w: 1 = casts a ground shadow
        app.ctx.queue.write_buffer(self.stat_buf, 0, stat)
        app.ctx.queue.write_buffer(self.dyn_buf, 0, np.zeros((n, 2), np.float32))
        self.bind_group = dev.create_bind_group(label="bld-bg", layout=self.bgl, entries=[
            {"binding": 0, "resource": {"buffer": self.params_buf, "offset": 0, "size": PARAM_BYTES}},
            {"binding": 1, "resource": {"buffer": self.dyn_buf, "offset": 0, "size": max(n, 1) * 8}},
            {"binding": 2, "resource": {"buffer": self.stat_buf, "offset": 0, "size": max(n, 1) * 16}}])
        app.ctx.track("buildings mesh", m.vertices.nbytes + m.indices.nbytes + n * 24)
        self._up_v = self._up_i = 0
        self._uploaded = False
        self._state_version = -1

    def _upload_slice(self, app):
        t0 = time.perf_counter()
        m = self.scene.mesh
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
        self.mesh_upload_ms = max(self.mesh_upload_ms, (time.perf_counter() - t0) * 1000.0)

    def on_storm(self, app):
        self._state_version = -1

    def shutdown(self, app):
        for b in (self.vbuf, self.ibuf, self.params_buf, self.dyn_buf, self.stat_buf):
            try:
                if b is not None:
                    b.destroy()
            except Exception:
                pass
        app.ctx.untrack("buildings mesh")
        self.vbuf = self.ibuf = None

    # ------------------------------------------------------------------------------------------ per frame
    def update(self, app, dt):
        sc = self.scene
        app.scene_status = {"state": sc.state, "message": sc.message, "frac": sc.frac}
        if sc.state != "ready":
            return
        if self.vbuf is None:
            self._create_buffers(app)
        if not self._uploaded:
            self._upload_slice(app)
            return
        self._update_state(app)
        E = app.exag
        skirt = 0.8 + 0.35 * E
        t0 = time.perf_counter()
        if self._culler is None:
            self._culler = TileCuller(sc.mesh.tiles)
        self._draws = self._culler.draws(app.view_proj, E, self.height_scale, skirt, 6.0, planes_for(app))
        self.tiles_visible = self._culler.visible_count
        self.tris_drawn = sum(c for _, c in self._draws) // 3
        self.cull_ms = (time.perf_counter() - t0) * 1000.0
        sel = getattr(app, "selection", None) or {}
        sid = sel.get("building")
        hid = sel.get("hover_building")
        cam = app.camera
        eye = cam.eye()
        ow = 0.0
        if sid is not None and 0 <= int(sid) < sc.buildings.n:
            c = np.array([sc.buildings.cx[sid], sc.buildings.cy[sid], sc.base_z[sid] * E])
            ow = float(np.clip(np.linalg.norm(eye - c) * 0.0045, 0.7, 80.0))
        tg = cam.target
        self._cam_dist = math.sqrt((eye[0] - tg[0]) ** 2 + (eye[1] - tg[1]) ** 2 + (eye[2] - tg[2]) ** 2)
        pulse = 0.5 + 0.5 * math.sin(app.anim_time * 5.0) if sid is not None else 0.0
        # the uniform block only changes with the selection / sliders (and the pulse while something is selected): build it
        # only then; the tier colours are written once
        pkey = (sid, hid, sc.buildings.n, self.water_line, self.height_scale, ow, self.flood_tint, skirt, pulse,
                self.shadow_strength)
        if pkey == self._pkey:
            return
        self._pkey = pkey
        p, pu = self._params, self._params_u
        pu[0] = NONE_ID if sid is None else int(sid)
        pu[1] = NONE_ID if hid is None else int(hid)
        pu[2] = sc.buildings.n
        pu[3] = 1 if self.water_line else 0
        p[4:8] = (self.height_scale, ow, self.flood_tint, skirt)
        p[8:12] = (pulse, 1.0 - self.shadow_strength, 0.0, 0.0)
        if not self._tiers_set:
            for i, c in enumerate(TIER_LINEAR):
                p[12 + 4 * i:16 + 4 * i] = (c[0], c[1], c[2], 1.0)
            self._tiers_set = True
        raw = p.tobytes()
        if raw != self._last_params:
            self._last_params = raw
            app.ctx.queue.write_buffer(self.params_buf, 0, p)

    def _update_state(self, app):
        f = getattr(app, "flood", None)
        if f is None:
            return
        ver = f.get("version", 0)
        if ver == self._state_version:
            return
        t0 = time.perf_counter()
        n = self.scene.buildings.n
        st = np.zeros((n, 2), np.float32)
        b = f.get("buildings")
        if f.get("ready") and b is not None:
            st[:, 0] = b.depth_m
            st[:, 1] = b.tier
        app.ctx.queue.write_buffer(self.dyn_buf, 0, st)
        self._state_version = ver
        self.state_upload_ms = (time.perf_counter() - t0) * 1000.0

    def draw(self, app, pass_):
        if not self._uploaded or self.bind_group is None:
            return
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, self.bind_group)
        pass_.set_vertex_buffer(0, self.vbuf)
        pass_.set_index_buffer(self.ibuf, "uint32")
        if self.shadows and self.shadow_strength > 0.01 and self._draws and self._cam_dist < self.shadow_max_dist                 and app.sun_el > 4.0:
            pass_.set_pipeline(self.pipe_shadow)
            for first, count in self._draws:
                pass_.draw_indexed(count, 1, first, 0, 0)
            pass_.set_pipeline(self.pipe_apply)
            pass_.draw(3)
        pass_.set_pipeline(self.pipe)
        for first, count in self._draws:
            pass_.draw_indexed(count, 1, first, 0, 0)

    # ------------------------------------------------------------------------------------------ UI
    def ui(self, app):
        from imgui_bundle import imgui
        sc = self.scene
        if sc.state == "loading":
            imgui.progress_bar(float(sc.frac), (-1, 6))
            imgui.text_disabled(f"loading buildings & roads: {sc.message}")
            return
        if sc.state == "error":
            imgui.text_colored((1.0, 0.45, 0.4, 1.0), "scene data failed: " + str(sc.error)[:80])
            return
        if not self._uploaded:
            imgui.progress_bar(self._up_v / max(sc.mesh.vertices.nbytes, 1), (-1, 6))
            imgui.text_disabled("uploading building mesh")
        ch, v = imgui.slider_float("Height scale", self.height_scale, 0.5, 4.0, "x%.1f")
        if ch:
            self.height_scale = v
        ch, v = imgui.slider_float("Flood colour", self.flood_tint, 0.0, 1.0, "%.2f")
        if ch:
            self.flood_tint = v
        _, self.shadows = imgui.checkbox("Sun shadows", self.shadows)
        if self.shadows:
            ch, v = imgui.slider_float("Shadow strength", self.shadow_strength, 0.1, 0.7, "%.2f")
            if ch:
                self.shadow_strength = v
        _, self.water_line = imgui.checkbox("Water line on walls", self.water_line)
        _, self.outline = imgui.checkbox("Outline selection", self.outline)
        f = getattr(app, "flood", None) or {}
        if f.get("ready"):
            tc = f["tier_counts"]
            imgui.text_disabled(f"as of hour {f['hour']}: {f['flooded_count']:,} flooded")
            for name, hexc, in zip(TIER_NAMES, TIER_HEX):
                r, g, b = hex_rgb(hexc)
                imgui.color_button(f"##{name}", (r, g, b, 1.0), imgui.ColorEditFlags_.no_tooltip, (10, 10))
                imgui.same_line()
                imgui.text(f"{name}: {tc[name]:,}")
        imgui.text_disabled(f"tiles {self.tiles_visible}/{sc.mesh.tiles.n}  tris {self.tris_drawn / 1e3:.0f}k  "
                            f"cull {self.cull_ms:.2f} ms")

    def save_state(self):
        return {"height_scale": self.height_scale, "flood_tint": self.flood_tint, "water_line": self.water_line,
                "outline": self.outline, "shadows": self.shadows, "shadow_strength": self.shadow_strength}

    def load_state(self, d):
        self.height_scale = float(d.get("height_scale", self.height_scale))
        self.flood_tint = float(d.get("flood_tint", self.flood_tint))
        self.water_line = bool(d.get("water_line", self.water_line))
        self.outline = bool(d.get("outline", self.outline))
        self.shadows = bool(d.get("shadows", self.shadows))
        self.shadow_strength = float(d.get("shadow_strength", self.shadow_strength))


class SelectionOverlay(Layer):
    """Highlight of the selected / hovered building, drawn AFTER the water (registered behind it) so it stays visible when the
    building is submerged: an amber translucent fill on the building's own pixels (depth test against the depth the building
    wrote) plus the expanded back-face shell as silhouette outline."""
    name = "selection"
    title = "Selection highlight"
    has_ui = False

    def __init__(self):
        self.pipe_fill = None
        self.pipe_outline = None
        self.owner = None

    def init(self, app):
        self.owner = app.get_layer("buildings")
        if self.owner is None:
            self.enabled = False
            return
        dev = app.ctx.device
        o = self.owner
        ms = {"count": app.sample_count}
        ds = {"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "greater-equal",
              "depth_bias": 4, "depth_bias_slope_scale": 1.0}
        over = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.pipe_fill = dev.create_render_pipeline(
            label="selection-fill", layout=o.pl, vertex={"module": o.mod, "entry_point": "vs_fill", "buffers": [o.vbuf_layout]},
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"}, depth_stencil=ds, multisample=ms,
            fragment={"module": o.mod, "entry_point": "fs_fill", "targets": [{"format": app.colour_format, "blend": over}]})
        self.pipe_outline = dev.create_render_pipeline(
            label="selection-outline", layout=o.pl,
            vertex={"module": o.mod, "entry_point": "vs_outline", "buffers": [o.vbuf_layout]},
            primitive={"topology": "triangle-list", "cull_mode": "front", "front_face": "ccw"},
            depth_stencil=dict(ds, depth_compare="greater", depth_bias=0, depth_bias_slope_scale=0.0), multisample=ms,
            fragment={"module": o.mod, "entry_point": "fs_outline", "targets": [{"format": app.colour_format}]})

    def draw(self, app, pass_):
        o = self.owner
        if o is None or not o._uploaded or o.bind_group is None:
            return
        sel = (getattr(app, "selection", None) or {}).get("building")
        if sel is None or not (0 <= int(sel) < o.scene.buildings.n):
            return
        m = o.scene.mesh
        cnt = int(m.bld_count[sel])
        if cnt <= 0:
            return
        first = int(m.bld_first[sel])
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, o.bind_group)
        pass_.set_vertex_buffer(0, o.vbuf)
        pass_.set_index_buffer(o.ibuf, "uint32")
        pass_.set_pipeline(self.pipe_fill)
        pass_.draw_indexed(cnt, 1, first, 0, 0)
        if o.outline:
            pass_.set_pipeline(self.pipe_outline)
            pass_.draw_indexed(cnt, 1, first, 0, 0)
