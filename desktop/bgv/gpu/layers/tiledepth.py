"""Solver-depth overlay: the native solver tiles (bgv.data.tiles) draped on the terrain in the viewer's depth-tier colours.

A clipmap window of 2048 x 2048 tile cells is kept around the camera target: the pyramid level follows the camera distance (level 0 =
6.25 m cells while closer than about 5 km, then the 12.5 m / 25 m ... overviews), the window is re-centred when the target leaves its middle
half, and it is re-read when the displayed hour (or the storm) changes.  Reading, decoding and the float16 conversion run in a worker
thread; the upload goes into the idle one of two textures in 256-row slices (never more than ~1 MB per frame), then the textures swap, so
the frame never stalls and a half-written window is never shown.

The overlay is a film at ground level drawn after the water, with the tier colours of the active palette (>= 0.1 m, 0.3, 0.6, 1.2 m) and a
soft edge; its depth is the solver's own (see the label of the tile set), not the flow grid's.  Off by default.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import wgpu

from ... import palette as palmod
from ...data import cog as cogmod
from ..flowtex import COMMON_WGSL
from . import Layer
from .terrain import CHUNK_WGSL, NOISE_WGSL, _vertex_state

N = 1024                                     # window size in tile cells
SLICE = 128                                  # rows per upload step (1024 * 2 B * 128 = 256 KB)

TILEDEPTH_WGSL = COMMON_WGSL + NOISE_WGSL + CHUNK_WGSL + """
struct TP {
  win: vec4<f32>,                 // window top-left mercator x, y, mercator metres per cell, cells per side
  wm: vec4<f32>,                  // grid centre mercator x, y, world metres per mercator metre (k), opacity
  tier: array<vec4<f32>, 4>,      // linear tier colours
};
@group(1) @binding(0) var<uniform> TPU: TP;
@group(1) @binding(1) var depth_tex: texture_2d<f32>;
@group(1) @binding(2) var depth_samp: sampler;

struct TOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) wpos: vec3<f32>,
};
@vertex fn vs_tile(@builtin(vertex_index) vi: u32, inst: Inst) -> TOut {
  let m = mesh_vertex(vi, inst.c);
  let zg = height_texel(m.texel.x, m.texel.y) * frame.grid.w;
  let dist = length(frame.eye.xyz - vec3<f32>(m.wxy, zg));
  let lift = 0.06 + dist * 5.0e-5 + f32(1u << m.lod) * 0.04;
  var o: TOut;
  o.wpos = vec3<f32>(m.wxy, zg + lift);
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  return o;
}

@fragment fn fs_tile(in: TOut) -> @location(0) vec4<f32> {
  let mx = in.wpos.x / TPU.wm.z + TPU.wm.x;
  let my = in.wpos.y / TPU.wm.z + TPU.wm.y;
  let uv = vec2<f32>((mx - TPU.win.x) / TPU.win.z, (TPU.win.y - my) / TPU.win.z) / TPU.win.w;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { discard; }
  let d = textureSampleLevel(depth_tex, depth_samp, uv, 0.0).r;
  let edge = smoothstep(0.075, 0.115, d);
  if (edge <= 0.01) { discard; }
  var t = 0;
  if (d >= 1.2) { t = 3; } else if (d >= 0.6) { t = 2; } else if (d >= 0.3) { t = 1; }
  let to_eye = frame.eye.xyz - in.wpos;
  let dist = length(to_eye);
  let V = to_eye / max(dist, 1e-3);
  let lum = 0.55 + 0.45 * clamp(frame.sun.z * 1.3, 0.0, 1.0) * frame.sun.w;
  var col = TPU.tier[t].xyz * lum;
  col = haze(col, dist, -V);
  return vec4<f32>(col, TPU.wm.w * edge);
}
"""


class TileDepthLayer(Layer):
    name = "tiledepth"
    title = "Solver depth overlay (experimental)"
    default_enabled = False
    has_ui = True

    def __init__(self):
        self.opacity = 0.5
        self.mode = "hour"                    # "hour" (the displayed hour's file) | "peak" (max_depth.tif)
        self.pipe = None
        self.status = "off"
        self.tiles_label = ""  # NOT self.label: that would shadow Layer.label() (the sidebar PopID assert)
        self.level = None
        self.upload_ms = 0.0
        self.max_upload_ms = 0.0
        self._tex = [None, None]
        self._bg = [None, None]
        self._front = -1                      # index of the texture that is shown (-1 = nothing yet)
        self._key = None                      # (storm, which, level, c0, r0) of the shown window
        self._req_key = None
        self._ready = None                    # worker result waiting to be uploaded: (key, f16 array, params)
        self._up = None                       # running upload: dict(key, arr, params, row, tex)
        self._params = None
        self._lock = threading.Lock()
        self._busy = False
        self._uniform = np.zeros(24, np.float32)
        self._uniform_key = None

    # ------------------------------------------------------------------------------------------ gpu objects
    def init(self, app):
        dev = app.ctx.device
        mod = dev.create_shader_module(label="tiledepth", code=TILEDEPTH_WGSL)
        vis = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.bgl = dev.create_bind_group_layout(label="tiledepth-bgl", entries=[
            {"binding": 0, "visibility": vis, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, self.bgl])
        blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.pipe = dev.create_render_pipeline(
            label="tiledepth", layout=pl, vertex=_vertex_state(mod, "vs_tile"),
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "greater"},
            multisample={"count": app.sample_count},
            fragment={"module": mod, "entry_point": "fs_tile", "targets": [{"format": app.colour_format, "blend": blend}]})
        self.ubuf = dev.create_buffer(label="tiledepth-params", size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.samp = dev.create_sampler(mag_filter="linear", min_filter="linear", address_mode_u="clamp-to-edge",
                                       address_mode_v="clamp-to-edge")
        self._tex_ready = False

    def _ensure_textures(self, app):
        if self._tex_ready:
            return
        dev = app.ctx.device
        for i in range(2):
            t = dev.create_texture(label=f"tiledepth-{i}", size=(N, N, 1), format="r16float", dimension="2d",
                                   usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self._tex[i] = t
            self._bg[i] = dev.create_bind_group(label=f"tiledepth-bg{i}", layout=self.bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.ubuf, "offset": 0, "size": 96}},
                {"binding": 1, "resource": t.create_view()}, {"binding": 2, "resource": self.samp}])
        app.ctx.track("solver depth overlay", 2 * N * N * 2)
        self._tex_ready = True

    def shutdown(self, app):
        for t in self._tex:
            try:
                if t is not None:
                    t.destroy()
            except Exception:
                pass
        try:
            self.ubuf.destroy()
        except Exception:
            pass
        app.ctx.untrack("solver depth overlay")
        self._tex = [None, None]
        self._tex_ready = False

    def on_storm(self, app):
        self._key = self._req_key = None
        self._ready = self._up = None

    # ------------------------------------------------------------------------------------------ window choice
    def _wanted(self, app, ts):
        """(storm, which, level, c0, r0) for the camera now, or the current window when it still serves (hysteresis)."""
        c = ts.cog("peak") or ts.cog(next(iter(ts.hours)))
        if c is None:
            return None
        cam = app.camera
        mx, my = app.terrain.frame.world_to_merc(float(cam.target[0]), float(cam.target[1]))
        k = app.terrain.frame.k
        need = 3.0 * float(cam.dist) / k                                  # mercator metres that should be covered
        lv = 0
        while lv < len(c.levels) - 1 and N * c.level_res(lv)[0] < need:
            lv += 1
        which = "peak" if self.mode == "peak" else (ts.nearest_hour(app.t) if ts.nearest_hour(app.t) is not None else None)
        if which is None:
            return None
        cur = self._key
        if cur is not None and cur[0] == app.storm.name and cur[1] == which:
            cl = cur[2]
            rx, ry = c.level_res(cl)
            span = N * rx
            cx_ = c.x0 + (cur[3] + N / 2) * rx
            cy_ = c.y0 - (cur[4] + N / 2) * ry
            inside = abs(mx - cx_) < span * 0.25 and abs(my - cy_) < span * 0.25
            # keep the window while it is centred enough and at the wanted level (or one finer level that still covers the view)
            if inside and (cl == lv or (cl == lv - 1 and need < span * 0.9)):
                return cur
        rx, ry = c.level_res(lv)
        c0 = int(round(((mx - c.x0) / rx - N / 2) / 128.0)) * 128
        r0 = int(round(((c.y0 - my) / ry - N / 2) / 128.0)) * 128
        return (app.storm.name, which, lv, c0, r0)

    def _work(self, key, ts):
        try:
            storm, which, lv, c0, r0 = key
            c = ts.cog(which)
            if c is None:
                return
            raw = c.read_window(lv, c0, r0, c0 + N, r0 + N)
            m = raw.astype(np.float32)
            m[raw == (c.nodata if c.nodata is not None else -1)] = 0.0
            arr = (m * 0.01).astype(np.float16)
            rx, ry = c.level_res(lv)
            params = (c.x0 + c0 * rx, c.y0 - r0 * ry, rx)
            with self._lock:
                self._ready = (key, arr, params)
        except Exception:
            try:
                from ... import applog
                applog.log().exception("solver depth overlay window failed")
            except Exception:
                pass
        finally:
            self._busy = False

    # ------------------------------------------------------------------------------------------ frame
    def update(self, app, dt):
        from ...data import depthapi
        svc = depthapi.service(app)
        ts = svc.tileset(app.storm.name) if svc.enabled else None
        if ts is None:
            self.status = svc.reason or f"no solver tiles for {app.storm.name}"
            self._active = False
            return
        self._ensure_textures(app)
        self.tiles_label = ts.label
        want = self._wanted(app, ts)
        if want is None:
            self.status = "tile file missing for this hour"
            return
        if want != self._key and want != self._req_key and not self._busy and self._up is None and self._ready is None:
            self._req_key = want
            self._busy = True
            threading.Thread(target=self._work, args=(want, ts), name="bgv-tile-window", daemon=True).start()
        # start an upload when the worker finished
        if self._up is None:
            with self._lock:
                rdy, self._ready = self._ready, None
            if rdy is not None:
                back = 1 if self._front == 0 else 0
                self._up = {"key": rdy[0], "arr": rdy[1], "params": rdy[2], "row": 0, "tex": back}
        if self._up is not None:
            t0 = time.perf_counter()
            u = self._up
            rows = min(SLICE, N - u["row"])
            app.ctx.queue.write_texture(
                {"texture": self._tex[u["tex"]], "mip_level": 0, "origin": (0, u["row"], 0)},
                u["arr"][u["row"]:u["row"] + rows], {"offset": 0, "bytes_per_row": N * 2, "rows_per_image": rows}, (N, rows, 1))
            u["row"] += rows
            if u["row"] >= N:                                              # complete: swap, atomically
                self._front, self._key, self._params = u["tex"], u["key"], u["params"]
                self._req_key = None
                self._up = None
                self.level = u["key"][2]
                self._uniform_key = None
            ms = (time.perf_counter() - t0) * 1000.0
            self.upload_ms = ms
            self.max_upload_ms = max(self.max_upload_ms, ms)
        self._active = self._front >= 0
        self.status = "ready" if self._active else "reading the tiles..."
        if self._active:
            fr = app.terrain.frame
            tier = palmod.linear(getattr(app, "palette", "std"))
            key = (self._params, self.opacity, getattr(app, "palette", "std"), fr.xc, fr.yc, fr.k)
            if key != self._uniform_key:
                u = self._uniform
                u[0:4] = (self._params[0], self._params[1], self._params[2], N)
                u[4:8] = (fr.xc, fr.yc, fr.k, self.opacity)
                for i, c in enumerate(tier):
                    u[8 + 4 * i:12 + 4 * i] = (c[0], c[1], c[2], 1.0)
                app.ctx.queue.write_buffer(self.ubuf, 0, u)
                self._uniform_key = key

    def draw(self, app, pass_):
        if not getattr(self, "_active", False) or self._front < 0:
            return
        m = getattr(app, "terrain_mesh", None)
        if m is None:
            return
        lists, inst = (m.draws_terrain, m.inst_terrain) if self.mode == "peak" else (m.draws_water, m.inst_water)
        if not lists:
            return
        pass_.set_pipeline(self.pipe)
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, self._bg[self._front])
        m.draw(pass_, lists, inst)

    # ------------------------------------------------------------------------------------------ ui / state
    def ui(self, app):
        from imgui_bundle import imgui
        imgui.text_wrapped("The solver's own depth, 6.25 m cells (tier colours >= 0.1 m), drawn over the water. Not the flow grid.")
        ch, v = imgui.slider_float("Opacity", self.opacity, 0.15, 1.0, "%.2f")
        if ch:
            self.opacity = v
        for key, lab in (("hour", "Displayed hour"), ("peak", "Peak of the run")):
            if imgui.radio_button(lab, self.mode == key):
                self.mode = key
            imgui.same_line()
        imgui.new_line()
        imgui.text_disabled(f"{self.status}" + (f"  |  level {self.level} ({6.25 * 2 ** self.level:g} m cells)" if self.level is not None else ""))
        if self.tiles_label:
            imgui.text_disabled(self.tiles_label)

    def save_state(self):
        return {"opacity": self.opacity, "mode": self.mode}

    def load_state(self, d):
        self.opacity = float(d.get("opacity", self.opacity))
        self.mode = d.get("mode", self.mode) if d.get("mode") in ("hour", "peak") else self.mode
