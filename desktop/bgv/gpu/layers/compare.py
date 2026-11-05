"""CompareLayer: the two compare modes of the analysis tools (state and settings: bgv/analysis/layer.py `CompareController`).

SWIPE   storm A (the viewer's storm) is drawn by the normal water layer on one side of a divider, storm B (or the same storm at another
        hour) by this layer on the other side - the SAME water pipeline and look, a second bind group 0 that differs only in the flow
        array and the time words.  B's flow array is a small hour cache (5 hours + a dry layer, ~190 MB at county size, freed when the
        mode is left), not a second 26-layer array.  Buildings, roads, arrows and particles keep showing storm A on both sides.
DIFF    the water of storm A is hidden and a sheet at the higher of the two water surfaces is drawn, coloured by (depth B - depth A)
        with the sign-aware palette of bgv/analysis/compare.py.  The difference comes from a float16 texture (rg16float: delta, max depth)
        built in the background (tiles where they exist) and uploaded without stalling a frame.

Hooks used by shared files (documented in docs/ANALYSIS_HOOKS.md):  `app.compare_hook` (WaterLayer.draw calls water_begin / water_end) and
`app.compare_wet` (TerrainLayer.update ORs it into the wet-chunk table so the water pass covers the chunks wet in B as well).
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import wgpu

from ...analysis import compare as cmpmod
from ...data import flow as flowdata
from ..bgthread import lower_priority
from ..context import tex_bytes
from ..flowtex import COMMON_WGSL, CHUNK_TEXELS, FRAME_BYTES
from . import Layer
from .terrain import CHUNK_WGSL, NOISE_WGSL, _vertex_state

N_SLOTS = 5
ZERO_SLOT = N_SLOTS
DIFF_WGSL = COMMON_WGSL + NOISE_WGSL + CHUNK_WGSL + """
struct DP { a: vec4<f32>, };            // x = range (m), y = opacity, z = flat threshold (m)
@group(1) @binding(0) var<uniform> DPU: DP;
@group(1) @binding(1) var delta_tex: texture_2d<f32>;
@group(1) @binding(2) var delta_samp: sampler;

""" + cmpmod.wgsl_ramp() + """
struct DOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) wpos: vec3<f32>,
  @location(1) uv: vec2<f32>,
};
@vertex fn vs_diff(@builtin(vertex_index) vi: u32, inst: Inst) -> DOut {
  let m = mesh_vertex(vi, inst.c);
  let uv = world_to_uv(m.wxy);
  let dt = textureSampleLevel(delta_tex, delta_samp, uv, 0.0);
  let zg = height_texel(m.texel.x, m.texel.y) * frame.grid.w;
  var top = 0.0;
  if (dt.g > 0.02) { top = dt.g; }
  let dist = length(frame.eye.xyz - vec3<f32>(m.wxy, zg));
  let lift = 0.03 + dist * 2.5e-5 + f32(1u << m.lod) * 0.02;
  var o: DOut;
  o.wpos = vec3<f32>(m.wxy, zg + top * frame.enc.w + lift);
  o.uv = uv;
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  return o;
}
@fragment fn fs_diff(in: DOut) -> @location(0) vec4<f32> {
  let dt = textureSampleLevel(delta_tex, delta_samp, in.uv, 0.0);
  let edge = smoothstep(0.012, 0.045, dt.g);
  if (edge <= 0.01) { discard; }
  let d = dt.r;
  let x = clamp(abs(d) / DPU.a.x, 0.0, 1.0);
  var col = delta_color(x, d > 0.0);
  let flat_c = """ + cmpmod.wgsl_flat() + """;                  // neutral grey of compare.py FLAT_RGB (linear)
  col = mix(flat_c, col, smoothstep(DPU.a.z, DPU.a.z * 2.0, abs(d)));
  // hillshade so the relief stays readable under the colour
  let s = textureSampleLevel(slope_tex, slope_samp, in.uv, 0.0);
  let N = normalize(vec3<f32>(-s.x * frame.grid.w * frame.opts.w, -s.y * frame.grid.w * frame.opts.w, 1.0));
  let L = frame.sun.xyz;
  let lit = 0.62 + 0.38 * clamp(dot(N, L), 0.0, 1.0);
  let to_eye = frame.eye.xyz - in.wpos;
  let dist = length(to_eye);
  let V = to_eye / max(dist, 1e-3);
  let c = haze(col * lit, dist, -V);
  let a = edge * DPU.a.y;
  return vec4<f32>(c * a, a);
}
"""


class LeanFlow:
    """A tiny hour cache for storm B: 5 hour slots + a permanent zero layer (rgba8unorm array), filled by a worker thread and
    uploaded in small per-frame slices (same staged scheme as FlowTextures, so a frame never stalls)."""

    def __init__(self, ctx, width: int, height: int):
        self.ctx = ctx
        self.width, self.height = width, height
        self.texture = ctx.device.create_texture(
            label="compare-flow", size=(width, height, N_SLOTS + 1), format="rgba8unorm", dimension="2d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC)
        self.view = self.texture.create_view(dimension="2d-array")
        self.stage = ctx.device.create_texture(
            label="compare-stage", size=(width, height, 1), format="rgba8unorm", dimension="2d",
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC)
        ctx.track("compare flow", tex_bytes(width, height, N_SLOTS + 2, 4))
        self.storm = None
        self.slot_of: dict[int, int] = {}          # hour -> slot (uploaded and usable)
        self.wet: dict[int, np.ndarray] = {}
        self._pending: dict[int, int] = {}          # hour -> slot being decoded / uploaded
        self._lru: list[int] = []
        self._gen = 0
        self._jobs: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        self._cur = None
        self._closed = False
        self._free: list = []
        self.upload_ms = 2.0
        self.failed: dict[int, str] = {}
        self._th = threading.Thread(target=self._worker, name="bgv-compare-decode", daemon=True)
        self._th.start()

    def set_storm(self, storm):
        self._gen += 1
        self.storm = storm
        self.slot_of.clear()
        self.wet.clear()
        self._pending.clear()
        self._lru.clear()
        self.failed.clear()
        self._cur = None
        self._drain()

    def _drain(self):
        for q in (self._jobs, self._done):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def ready(self, h: int) -> bool:
        s = self.storm
        return s is None or s.is_zero(h) or h in self.slot_of or h in self.failed

    def request(self, hours):
        """Make these hours resident (decode + upload in the background); slots of other hours are recycled, LRU first."""
        s = self.storm
        if s is None:
            return
        want = [h for h in dict.fromkeys(int(h) for h in hours) if 0 <= h <= 24 and not s.is_zero(h)]
        for h in want:
            if h in self._lru:
                self._lru.remove(h)
            self._lru.append(h)
            if h in self.slot_of or h in self._pending or h in self.failed:
                continue
            used = set(self.slot_of.values()) | set(self._pending.values())
            free = [i for i in range(N_SLOTS) if i not in used]
            if not free:
                victim = next((v for v in self._lru if v in self.slot_of and v not in want), None)
                if victim is None:
                    continue
                slot = self.slot_of.pop(victim)
                self.wet.pop(victim, None)
                self._lru.remove(victim)
                free = [slot]
            self._pending[h] = free[0]
            self._jobs.put((self._gen, s, h, free[0]))

    def _worker(self):
        lower_priority()
        while not self._closed:
            try:
                gen, storm, h, slot = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                buf = self._free.pop() if self._free else np.empty((self.height, self.width, 4), np.uint8)
                storm.read_hour_into(h, buf)
                wet = flowdata.block_max(buf[..., 0], CHUNK_TEXELS) > 0
                self._done.put((gen, h, slot, buf, wet, None))
            except Exception as e:
                self._done.put((gen, h, slot, None, None, f"{type(e).__name__}: {e}"))

    def pump(self):
        t_end = time.perf_counter() + self.upload_ms * 1e-3
        rowbytes = self.width * 4
        chunk_rows = max(1, (1 << 20) // rowbytes)
        while time.perf_counter() < t_end:
            if self._cur is None:
                try:
                    gen, h, slot, buf, wet, err = self._done.get_nowait()
                except queue.Empty:
                    return
                if gen != self._gen:
                    if buf is not None:
                        self._free.append(buf)
                    continue
                if err is not None:
                    self.failed[h] = err
                    self._pending.pop(h, None)
                    continue
                self._cur = [h, slot, buf, wet, 0]
            h, slot, buf, wet, row = self._cur
            rows = min(self.height - row, chunk_rows)
            self.ctx.queue.write_texture({"texture": self.stage, "mip_level": 0, "origin": (0, row, 0)}, buf[row:row + rows],
                                         {"offset": 0, "bytes_per_row": rowbytes, "rows_per_image": rows},
                                         (self.width, rows, 1))
            self._cur[4] = row + rows
            if self._cur[4] >= self.height:
                enc = self.ctx.device.create_command_encoder(label="compare-layer-copy")
                enc.copy_texture_to_texture({"texture": self.stage, "mip_level": 0, "origin": (0, 0, 0)},
                                            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, slot)},
                                            (self.width, self.height, 1))
                self.ctx.queue.submit([enc.finish()])
                self.slot_of[h] = slot
                self.wet[h] = wet
                self._pending.pop(h, None)
                self._free.append(buf)
                self._cur = None

    def layer_map(self) -> np.ndarray:
        lm = np.full(28, ZERO_SLOT, np.uint32)
        for h, s in self.slot_of.items():
            lm[h] = s
        return lm

    def wet_for(self, h0: int, h1: int, mix: float):
        a = self.wet.get(h0)
        b = self.wet.get(h1) if mix > 0 else None
        if a is None and b is None:
            return None
        return a if b is None else (b if a is None else (a | b))

    def close(self):
        self._closed = True
        self._gen += 1
        self.ctx.untrack("compare flow")
        for t in (self.texture, self.stage):
            try:
                t.destroy()
            except Exception:
                pass


class CompareLayer(Layer):
    name = "compare"
    title = "Compare"
    draws = True
    has_ui = False

    def __init__(self):
        self.diff_pipeline = None
        self.lean: LeanFlow | None = None
        self.lean_key = None
        self.ubuf = None
        self.bgroup = None
        self.storm_b = None
        self.delta_tex = None
        self.delta_stage = None
        self.delta_view = None
        self.diff_bg = None
        self.diff_params = None
        self._pkey = None
        self._up = None                   # [array, row, key, wet]
        self.shown_key = None
        self.diff_wet = None
        self._disp = (12, 13, 0.0)
        self.error = ""
        self.mode_active = "off"

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        dev = app.ctx.device
        self.app = app
        bgl = dev.create_bind_group_layout(label="diff-bgl", entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX,
             "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX, "sampler": {"type": "filtering"}}])
        self.diff_bgl = bgl
        self.diff_params = dev.create_buffer(label="diff-params", size=16,
                                             usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.diff_sampler = dev.create_sampler(label="diff-samp", mag_filter="linear", min_filter="linear",
                                               address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, bgl])
        mod = dev.create_shader_module(label="diff", code=DIFF_WGSL)
        blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.diff_pipeline = dev.create_render_pipeline(
            label="diff", layout=pl, vertex=_vertex_state(mod, "vs_diff"),
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "greater"},
            multisample={"count": app.sample_count},
            fragment={"module": mod, "entry_point": "fs_diff", "targets": [{"format": app.colour_format, "blend": blend}]})
        self.ubuf = dev.create_buffer(label="compare-frame", size=FRAME_BYTES,
                                      usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        app.compare_hook = self
        app.compare_wet = None

    def shutdown(self, app):
        if getattr(app, "compare_hook", None) is self:
            app.compare_hook = None
            app.compare_wet = None
        self._free_lean()
        self._free_delta()
        for b in (self.ubuf, self.diff_params):
            try:
                if b is not None:
                    b.destroy()
            except Exception:
                pass

    def _free_lean(self):
        if self.lean is not None:
            self.lean.close()
        self.lean = None
        self.lean_key = None
        self.bgroup = None
        if self.storm_b is not None:
            try:
                self.storm_b.close()
            except Exception:
                pass
        self.storm_b = None

    def _free_delta(self):
        for t in (self.delta_tex, self.delta_stage):
            try:
                if t is not None:
                    t.destroy()
            except Exception:
                pass
        self.delta_tex = self.delta_stage = self.delta_view = self.diff_bg = None
        self.shown_key = None
        self._up = None
        self.app.ctx.untrack("compare delta")

    # ------------------------------------------------------------------------------------------ helpers
    @property
    def ctl(self):
        a = getattr(self.app, "analysis", None)
        return None if a is None else a.cmp

    def regions(self, app, which: str):
        """Scissor rectangle (x, y, w, h) in render-target pixels of side 'a' or 'b' of the divider."""
        cc, st = self.ctl, app.analysis.st
        pw, ph = app._target_size
        x0, y0, x1, y1 = cc.view_rect
        if st.vertical:
            x = int(round((x0 + st.swipe * (x1 - x0)) * pw))
            x = min(max(x, 0), pw)
            return (0, 0, x, ph) if which == "a" else (x, 0, pw - x, ph)
        y = int(round((y0 + st.swipe * (y1 - y0)) * ph))
        y = min(max(y, 0), ph)
        return (0, 0, pw, y) if which == "a" else (0, y, pw, ph - y)

    # ------------------------------------------------------------------------------------------ per frame
    def update(self, app, dt):
        an = getattr(app, "analysis", None)
        if an is None or an.cmp is None:
            return
        mode = an.st.mode
        self.mode_active = mode
        try:
            if mode == "swipe":
                self._update_swipe(app, an)
            else:
                if self.lean is not None:
                    self._free_lean()
            if mode == "diff":
                self._update_diff(app, an)
            elif self.delta_tex is not None:
                self._free_delta()
            if mode == "off":
                app.compare_wet = None
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            an.st.mode = "off"
            app.compare_wet = None
            app.report_error("compare", e)

    # ---- swipe
    def _update_swipe(self, app, an):
        cc = an.cmp
        b = cc.b_name(app)
        if not b or not cc.same_frame(app, b):
            app.compare_wet = None
            self.error = f"{b or 'storm B'} is on a different grid than the view: swipe needs the same grid (use the difference layer)"
            return
        self.error = ""
        key = (b, id(app.terrain))
        if self.lean is None or self.lean_key != key:
            self._free_lean()
            f = app.terrain.frame
            self.storm_b = app.domain.open_storm(b)
            self.lean = LeanFlow(app.ctx, f.width, f.height)
            self.lean.set_storm(self.storm_b)
            self.lean_key = key
            self.bgroup = app.ctx.device.create_bind_group(label="compare-bg", layout=app.frame.layout, entries=[
                {"binding": 0, "resource": {"buffer": self.ubuf, "offset": 0, "size": FRAME_BYTES}},
                {"binding": 1, "resource": self.lean.view},
                {"binding": 2, "resource": app.frame.flow_sampler},
                {"binding": 3, "resource": app.terrain.height_view},
                {"binding": 4, "resource": app.terrain.slope_view},
                {"binding": 5, "resource": app.frame.slope_sampler}])
        tb = min(max(cc.b_time(app), 0.0), 24.0)
        h0, h1, mix = flowdata.hour_pair(tb)
        want = [h0, h1]
        if app.playing:
            want += [min(h1 + 1, 24), min(h1 + 2, 24)]
        self.lean.request(want)
        self.lean.pump()
        f = app.frame.f.copy()
        sb = self.storm_b
        f[48:52] = (tb, mix, h0, h1)
        f[52:55] = (sb.depth_max, sb.vmax, sb.wet_threshold)
        u = f.view(np.uint32)
        u[64:92] = self.lean.layer_map()
        f[64 + 25] = app.frame.q[0]
        f[64 + 26] = app.frame.q[1]
        app.ctx.queue.write_buffer(self.ubuf, 0, f)
        app.compare_wet = self.lean.wet_for(h0, h1, mix)

    # ---- difference
    def _update_diff(self, app, an):
        cc = an.cmp
        pend = cc.pending
        if pend is not None and self._up is None:
            key, packed, wet = pend
            cc.pending = None
            f = app.terrain.frame
            if self.delta_tex is None:
                dev = app.ctx.device
                self.delta_tex = dev.create_texture(label="diff-delta", size=(f.width, f.height, 1), format="rg16float",
                                                    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
                                                    | wgpu.TextureUsage.COPY_SRC)
                self.delta_stage = dev.create_texture(label="diff-stage", size=(f.width, f.height, 1), format="rg16float",
                                                      usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC)
                self.delta_view = self.delta_tex.create_view()
                self.diff_bg = dev.create_bind_group(label="diff-bg", layout=self.diff_bgl, entries=[
                    {"binding": 0, "resource": {"buffer": self.diff_params, "offset": 0, "size": 16}},
                    {"binding": 1, "resource": self.delta_view},
                    {"binding": 2, "resource": self.diff_sampler}])
                app.ctx.track("compare delta", tex_bytes(f.width, f.height, 2, 4))
            self._up = [np.ascontiguousarray(packed), 0, key, wet]
        if self._up is not None:
            arr, row, key, wet = self._up
            H, W = arr.shape[:2]
            rowbytes = W * 4
            t_end = time.perf_counter() + 2.0e-3
            chunk_rows = max(1, (1 << 20) // rowbytes)
            while row < H and time.perf_counter() < t_end:
                rows = min(chunk_rows, H - row)
                app.ctx.queue.write_texture({"texture": self.delta_stage, "mip_level": 0, "origin": (0, row, 0)},
                                            arr[row:row + rows], {"offset": 0, "bytes_per_row": rowbytes, "rows_per_image": rows},
                                            (W, rows, 1))
                row += rows
            self._up[1] = row
            if row >= H:
                enc = app.ctx.device.create_command_encoder(label="diff-copy")
                enc.copy_texture_to_texture({"texture": self.delta_stage, "mip_level": 0, "origin": (0, 0, 0)},
                                            {"texture": self.delta_tex, "mip_level": 0, "origin": (0, 0, 0)}, (W, H, 1))
                app.ctx.queue.submit([enc.finish()])
                self.shown_key = key
                cc.shown_key = key
                self.diff_wet = wet
                self._up = None
        p = np.zeros(4, np.float32)
        p[:] = (an.st.range_m, an.st.opacity, cmpmod.FLAT_M, 0.0)
        k = tuple(float(v) for v in p)
        if k != self._pkey:
            self._pkey = k
            app.ctx.queue.write_buffer(self.diff_params, 0, p)
        if self.shown_key is not None and self.diff_wet is not None:
            app.compare_wet = self.diff_wet
        else:
            app.compare_wet = None

    # ------------------------------------------------------------------------------------------ WaterLayer hooks
    def water_begin(self, app, pass_) -> bool:
        """Called by WaterLayer.draw before the water: False = skip storm A's water (difference mode)."""
        if self.mode_active == "diff" and self.shown_key is not None:
            return False
        if self.mode_active == "swipe" and self.lean is not None and self.bgroup is not None:
            x, y, w, h = self.regions(app, "a")
            if w > 0 and h > 0:
                pass_.set_scissor_rect(x, y, w, h)
            else:
                return False
        return True

    def water_end(self, app, pass_):
        if self.mode_active == "swipe" and self.lean is not None:
            pw, ph = app._target_size
            pass_.set_scissor_rect(0, 0, pw, ph)

    # ------------------------------------------------------------------------------------------ draw
    def draw(self, app, pass_):
        m = getattr(app, "terrain_mesh", None)
        if m is None or not m.draws_water:
            return
        pw, ph = app._target_size
        if self.mode_active == "swipe" and self.lean is not None and self.bgroup is not None:
            wl = app.get_layer("water")
            if wl is None or wl.pipeline is None:
                return
            x, y, w, h = self.regions(app, "b")
            if w <= 0 or h <= 0:
                return
            pass_.set_scissor_rect(x, y, w, h)
            pass_.set_pipeline(wl.pipeline)
            pass_.set_bind_group(0, self.bgroup)
            pass_.set_bind_group(1, wl.bg)
            m.draw(pass_, m.draws_water, m.inst_water)
            pass_.set_scissor_rect(0, 0, pw, ph)
            pass_.set_bind_group(0, app.frame.bind_group)
        elif self.mode_active == "diff" and self.shown_key is not None and self.diff_bg is not None:
            pass_.set_pipeline(self.diff_pipeline)
            pass_.set_bind_group(0, app.frame.bind_group)
            pass_.set_bind_group(1, self.diff_bg)
            m.draw(pass_, m.draws_water, m.inst_water)
