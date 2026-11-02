"""GPU flow particles: compute-shader advection + camera-facing streak quads.

Pipeline per frame (all on the GPU, no read-back):
  compute  cs_estimate  1 workgroup, 4096 stratified screen samples -> fraction of the screen that is moving wet water ->
                        number of ACTIVE particles (smoothed), written into the indirect draw arguments.
  compute  cs_update    one thread per active particle: advect (midpoint rule) through `flow_at` (bilinear + hour mix), age,
                        fade; a particle that expired, left the camera frustum, or reached dry / still water RESPAWNS by
                        rejection sampling: random screen point -> ray -> terrain surface -> keep it when the cell is wet
                        (depth >= 0.05 m, speed > 0.03 m/s).  Sampling in screen space means the density per screen pixel is
                        uniform and independent of zoom, perspective or camera teleports (everything outside the view is
                        re-seeded inside it in the same frame).
  draw     vs/fs        indirect instanced quads (triangle strip), head -> tail, constant on-screen width, premultiplied
                        alpha, depth tested (no write) against terrain/buildings, sitting just above the water surface.

Particle state lives in world metres (storage buffer, 32 B each): pos, velocity (m/s, east/north), surface z, age, life,
fade.  Velocity is multiplied by a view-dependent factor (`fxcommon.velocity_multiplier`) so slow water stays visible
when zoomed out; `animation speed` scales it.  Streak length = speed x factor x trail time, clamped to [min, max] pixels.
A paused simulation (frozen hour) keeps animating: particles advect through the frozen flow with real time.
"""
from __future__ import annotations

import math

import numpy as np
import wgpu

from ..flowtex import COMMON_WGSL
from . import Layer
from . import fxcommon as fx

MAX_POOL = 250_000
PARTICLE_BYTES = 32
ARGS_BYTES = 32
PARAM_BYTES = 96

PARAMS_WGSL = """
struct Params {
  a: vec4<f32>,   // dt (s), velocity multiplier, trail time (s), alpha
  b: vec4<f32>,   // width px, min length px, max length px, lift (m)
  c: vec4<f32>,   // life min, life max (s), fade in, fade out (s)
  d: vec4<f32>,   // min depth (m), min speed (m/s), particles per wet screen pixel, max range (m)
  e: vec4<f32>,   // pool size, colour mode, spawn attempts, camera distance
  u: vec4<u32>,   // frame counter, 0, 0, 0
};
struct Particle { pos: vec2<f32>, vel: vec2<f32>, z: f32, age: f32, life: f32, fade: f32 };
struct Args { vcount: u32, icount: u32, first_v: u32, first_i: u32, smooth_n: f32, wet_frac: f32, ground_frac: f32, pad: f32 };
@group(1) @binding(0) var<uniform> P: Params;
"""

COMPUTE_WGSL = COMMON_WGSL + fx.FX_WGSL + PARAMS_WGSL + """
@group(1) @binding(1) var<storage, read_write> parts: array<Particle>;
@group(1) @binding(2) var<storage, read_write> args: Args;

var<workgroup> wg_ground: atomic<u32>;
var<workgroup> wg_wet: atomic<u32>;

@compute @workgroup_size(256)
fn cs_estimate(@builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&wg_ground, 0u);
    atomicStore(&wg_wet, 0u);
  }
  workgroupBarrier();
  var rng = fx_pcg(li * 9781u + P.u.x * 6271u + 17u);
  let gx = li % 16u;
  let gy = li / 16u;
  var nground = 0u;
  var nwet = 0u;
  for (var k = 0u; k < 16u; k = k + 1u) {
    let sx = k % 4u;
    let sy = k / 4u;
    let jx = fx_rnd(&rng);
    let jy = fx_rnd(&rng);
    let ndc = vec2<f32>((f32(gx * 4u + sx) + jx) / 64.0, (f32(gy * 4u + sy) + jy) / 64.0) * 2.0 - vec2<f32>(1.0);
    let h = screen_to_ground(ndc, P.d.w);
    if (h.ok) {
      nground = nground + 1u;
      let fl = flow_at(world_to_uv(h.p.xy));
      if (fl.valid > 0.5 && fl.depth >= P.d.x && length(fl.vel) >= P.d.y) { nwet = nwet + 1u; }
    }
  }
  atomicAdd(&wg_ground, nground);
  atomicAdd(&wg_wet, nwet);
  workgroupBarrier();
  if (li == 0u) {
    let wet_frac = f32(atomicLoad(&wg_wet)) / 4096.0;
    let far = 1.0 + 5.0 * smoothstep(600.0, 9000.0, P.e.w);
    var n = wet_frac * frame.view.x * frame.view.y * P.d.z * far;
    if (atomicLoad(&wg_wet) > 0u) { n = max(n, 250.0); } else { n = 0.0; }
    n = min(n, P.e.x);
    let prev = args.smooth_n;
    var ns = n;
    if (prev > 0.0 && n > 0.0) { ns = mix(prev, n, 0.12); }
    if (n == 0.0) { ns = prev * 0.7; if (ns < 50.0) { ns = 0.0; } }
    args.smooth_n = ns;
    args.wet_frac = wet_frac;
    args.ground_frac = f32(atomicLoad(&wg_ground)) / 4096.0;
    args.vcount = 4u;
    args.icount = u32(ns);
    args.first_v = 0u;
    args.first_i = 0u;
  }
}

@compute @workgroup_size(64)
fn cs_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= args.icount) { return; }
  var p = parts[idx];
  let dt = P.a.x;
  let m = P.a.y;
  let E = frame.grid.w;
  var rng = fx_pcg(idx * 2654435761u + P.u.x * 40503u + 1013u);
  var alive = p.life > 0.0 && p.age < p.life;
  var expired = p.life > 0.0 && p.age >= p.life;
  var depth = 0.0;
  if (alive) {
    let f0 = flow_at(world_to_uv(p.pos));
    let mid = p.pos + f0.vel * (m * dt * 0.5);
    let f1 = flow_at(world_to_uv(mid));
    let np = p.pos + f1.vel * (m * dt);
    let f2 = flow_at(world_to_uv(np));
    let z = ground_height(np) * E + f2.depth * frame.enc.w;
    p.age = p.age + dt;
    if (f2.valid < 0.5 || f2.depth < P.d.x * 0.6 || length(f2.vel) < P.d.y * 0.6 || !in_view(vec3<f32>(np, z), 1.06)) {
      alive = false;
    } else {
      p.pos = np;
      p.vel = f2.vel;
      p.z = z;
      depth = f2.depth;
    }
    if (p.age >= p.life) { alive = false; expired = true; }
  }
  if (!alive) {
    var found = false;
    let attempts = u32(P.e.z);
    for (var a = 0u; a < attempts; a = a + 1u) {
      let ndc = vec2<f32>(fx_rnd(&rng), fx_rnd(&rng)) * 2.0 - vec2<f32>(1.0);
      let h = screen_to_ground(ndc, P.d.w);
      if (!h.ok) { continue; }
      let fl = flow_at(world_to_uv(h.p.xy));
      if (fl.valid < 0.5 || fl.depth < P.d.x || length(fl.vel) < P.d.y) { continue; }
      p.pos = h.p.xy;
      p.vel = fl.vel;
      p.z = h.p.z + fl.depth * frame.enc.w;
      p.life = mix(P.c.x, P.c.y, fx_rnd(&rng));
      p.age = select(fx_rnd(&rng) * p.life, 0.0, expired);
      depth = fl.depth;
      found = true;
      break;
    }
    if (!found) {
      p.life = 0.0;
      p.age = 0.0;
      p.fade = 0.0;
      parts[idx] = p;
      return;
    }
  }
  let sp = length(p.vel);
  let fin = smoothstep(0.0, P.c.z, p.age);
  let fout = 1.0 - smoothstep(max(p.life - P.c.w, 0.0), p.life, p.age);
  let sfade = smoothstep(P.d.y, P.d.y * 3.0, sp);
  let dfade = smoothstep(P.d.x, P.d.x * 5.0, depth);
  p.fade = fin * fout * sfade * dfade;
  parts[idx] = p;
}
"""

RENDER_WGSL = COMMON_WGSL + fx.FX_WGSL + PARAMS_WGSL + """
@group(1) @binding(1) var<storage, read> parts: array<Particle>;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) px: vec2<f32>,
  @location(1) col: vec4<f32>,
  @location(2) @interpolate(flat) len: f32,
};

fn to_screen(c: vec4<f32>) -> vec2<f32> {
  return (c.xy / c.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * frame.view.xy;
}

fn hue(a: f32) -> vec3<f32> {
  return clamp(vec3<f32>(abs(a * 6.0 - 3.0) - 1.0, 2.0 - abs(a * 6.0 - 2.0), 2.0 - abs(a * 6.0 - 4.0)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  o.px = vec2<f32>(0.0);
  o.col = vec4<f32>(0.0);
  o.len = 1.0;
  let p = parts[ii];
  if (p.life <= 0.0 || p.fade < 0.01) { return o; }
  let speed = length(p.vel);
  let head0 = vec3<f32>(p.pos, p.z);
  let ch0 = frame.view_proj * vec4<f32>(head0, 1.0);
  if (ch0.w < 0.3) { return o; }
  let lift = P.b.w + ch0.w * 4.0e-4;
  let head = head0 + vec3<f32>(0.0, 0.0, lift);
  let ch = frame.view_proj * vec4<f32>(head, 1.0);
  let vdir = p.vel / max(speed, 1.0e-4);
  let trail_world = speed * P.a.y * P.a.z;
  let probe = max(trail_world, 0.5 + ch.w * 1.0e-3);
  let ct = frame.view_proj * vec4<f32>(head - vec3<f32>(vdir * probe, 0.0), 1.0);
  if (ct.w < 0.3) { return o; }
  let sh = to_screen(ch);
  let st = to_screen(ct);
  let dpx = st - sh;
  let lp = length(dpx);
  var dir = vec2<f32>(1.0, 0.0);
  if (lp > 1.0e-4) { dir = dpx / lp; }
  let len = clamp(lp * trail_world / probe, P.b.y, P.b.z);
  let halfw = P.b.x * 0.5 + 1.0;
  let perp = vec2<f32>(-dir.y, dir.x);
  let side = select(-1.0, 1.0, (vi & 1u) == 1u);
  let is_head = (vi >> 1u) == 1u;
  var base = sh + dir * len;
  var ax = len;
  var clipz = ct.z;
  var clipw = ct.w;
  if (is_head) {
    base = sh - dir * halfw;
    ax = -halfw;
    clipz = ch.z;
    clipw = ch.w;
  }
  let px = base + perp * (side * halfw);
  let ndc = px / frame.view.xy * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
  o.pos = vec4<f32>(ndc * clipw, clipz, clipw);
  o.px = vec2<f32>(ax, side * halfw);
  o.len = max(len, 1.0);
  var rgb = speed_ramp(speed);
  let mode = u32(P.e.y + 0.5);
  if (mode == 1u) {
    rgb = mix(vec3<f32>(0.75, 0.9, 1.0), hue(fract(atan2(p.vel.y, p.vel.x) / 6.2831853 + 1.0)), 0.8);
  } else if (mode == 2u) {
    rgb = vec3<f32>(0.92, 0.97, 1.0);
  } else if (mode == 3u) {
    let d = flow_at(world_to_uv(p.pos)).depth;
    rgb = mix(vec3<f32>(0.9, 1.0, 0.8), vec3<f32>(0.45, 0.8, 1.0), smoothstep(0.1, 2.5, d));
  }
  o.col = vec4<f32>(rgb, P.a.w * p.fade);
  return o;
}

@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
  let hw = P.b.x * 0.5;
  var r = abs(in.px.y);
  if (in.px.x < 0.0) { r = length(in.px); }
  let cover = clamp(hw + 0.5 - r, 0.0, 1.0);
  let g = pow(clamp(1.0 - in.px.x / in.len, 0.0, 1.0), 1.25);
  let a = cover * g * in.col.a;
  if (a <= 0.002) { discard; }
  return vec4<f32>(in.col.rgb * a, a * 0.9);
}
"""


class ParticlesLayer(Layer):
    name = "particles"
    title = "Flow particles"
    has_ui = True

    PRESETS = ("low", "medium", "high")
    COLOURS = ("Speed (cyan - white - warm)", "Direction (hue)", "Uniform white", "Depth")

    def __init__(self):
        self.preset = "high"
        self.density = 1.0               # user multiplier of the particle density
        self.anim_speed = 1.0            # animation speed (time scale of the streak motion)
        self.trail = 1.0                 # trail length multiplier
        self.width_px = 1.7
        self.alpha = 0.85
        self.colour = 0
        self.const_speed = True
        self.life_min, self.life_max = 2.5, 5.5
        self.min_depth = 0.05
        self.min_speed = 0.03
        self.base_density = 0.016        # particles per screen pixel of moving wet water
        self.min_len_px, self.max_len_px = 4.0, 56.0
        self.px_per_ms = 24.0            # apparent on-screen speed of 1 m/s of flow (px/s at animation speed 1)
        self._quality_seen = None
        self._frame = 0
        self._dt = 0.0
        self._pool_cap = MAX_POOL
        self.params = None
        self.n_active_est = 0.0

    # ------------------------------------------------------------------------------------------ lifecycle
    def init(self, app):
        dev = app.ctx.device
        vis_c = wgpu.ShaderStage.COMPUTE
        vis_r = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.params = dev.create_buffer(label="particle-params", size=PARAM_BYTES,
                                        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.buf = dev.create_buffer(label="particles", size=MAX_POOL * PARTICLE_BYTES,
                                     usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        self.args = dev.create_buffer(label="particle-args", size=ARGS_BYTES,
                                      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.INDIRECT
                                      | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        app.ctx.queue.write_buffer(self.buf, 0, np.zeros(MAX_POOL * PARTICLE_BYTES // 4, np.float32))
        app.ctx.queue.write_buffer(self.args, 0, np.zeros(ARGS_BYTES // 4, np.float32))
        app.ctx.track("particles", MAX_POOL * PARTICLE_BYTES)
        bgl_c = dev.create_bind_group_layout(label="particles-compute-bgl", entries=[
            {"binding": 0, "visibility": vis_c | vis_r, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": vis_c, "buffer": {"type": "storage"}},
            {"binding": 2, "visibility": vis_c, "buffer": {"type": "storage"}}])
        bgl_r = dev.create_bind_group_layout(label="particles-render-bgl", entries=[
            {"binding": 0, "visibility": vis_c | vis_r, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {"type": "read-only-storage"}}])
        self.bg_c = dev.create_bind_group(label="particles-compute-bg", layout=bgl_c, entries=[
            {"binding": 0, "resource": {"buffer": self.params, "offset": 0, "size": PARAM_BYTES}},
            {"binding": 1, "resource": {"buffer": self.buf, "offset": 0, "size": MAX_POOL * PARTICLE_BYTES}},
            {"binding": 2, "resource": {"buffer": self.args, "offset": 0, "size": ARGS_BYTES}}])
        self.bg_r = dev.create_bind_group(label="particles-render-bg", layout=bgl_r, entries=[
            {"binding": 0, "resource": {"buffer": self.params, "offset": 0, "size": PARAM_BYTES}},
            {"binding": 1, "resource": {"buffer": self.buf, "offset": 0, "size": MAX_POOL * PARTICLE_BYTES}}])
        cmod = dev.create_shader_module(label="particles-compute", code=COMPUTE_WGSL)
        pl_c = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, bgl_c])
        self.pipe_est = dev.create_compute_pipeline(label="particles-estimate", layout=pl_c,
                                                    compute={"module": cmod, "entry_point": "cs_estimate"})
        self.pipe_upd = dev.create_compute_pipeline(label="particles-update", layout=pl_c,
                                                    compute={"module": cmod, "entry_point": "cs_update"})
        rmod = dev.create_shader_module(label="particles-render", code=RENDER_WGSL)
        pl_r = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, bgl_r])
        blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.pipe_draw = dev.create_render_pipeline(
            label="particles-draw", layout=pl_r, vertex={"module": rmod, "entry_point": "vs"},
            primitive={"topology": "triangle-strip", "cull_mode": "none"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "greater"},
            multisample={"count": app.sample_count},
            fragment={"module": rmod, "entry_point": "fs", "targets": [{"format": app.colour_format, "blend": blend}]})
        self._apply_quality(app, force=True)

    def shutdown(self, app):
        for b in (getattr(self, "buf", None), getattr(self, "args", None), self.params):
            try:
                if b is not None:
                    b.destroy()
            except Exception:
                pass
        app.ctx.untrack("particles")

    def on_storm(self, app):
        pass              # particles that are on dry cells in the new storm respawn on wet ones by themselves

    # ------------------------------------------------------------------------------------------ settings
    def set_preset(self, name: str):
        if name not in fx.QUALITY:
            return
        self.preset = name
        q = fx.QUALITY[name]
        self._pool_cap = int(q["pool"])
        self._attempts = int(q["attempts"])
        self._preset_density = float(q["density"])

    def set_quality(self, preset):
        """Called by the sidebar quality presets ('Low' / 'Medium' / 'High' / 'Ultra'); same mapping as `app.quality`."""
        p = str(preset).lower()
        self.set_preset(p if p in fx.QUALITY else "high")

    def _apply_quality(self, app, force=False):
        qn = fx.quality_name(app)
        if qn is not None and (qn != self._quality_seen or force):
            self._quality_seen = qn
            self.set_preset(qn)
        elif force or not hasattr(self, "_attempts"):
            self.set_preset(self.preset)

    # ------------------------------------------------------------------------------------------ per frame
    def update(self, app, dt):
        self._apply_quality(app)
        cam = app.camera
        pw, ph = app._target_size if app._target_size != (0, 0) else (1920, 1080)
        ppm = fx.px_per_metre(cam.dist, cam.fov, ph)
        mult = fx.velocity_multiplier(ppm, self.anim_speed, self.px_per_ms, self.const_speed)
        self._dt = min(dt, 0.1)
        self._frame += 1
        eye_dist = float(np.linalg.norm(cam.eye() - cam.target))
        max_range = max(60000.0, 3.0 * cam.dist)
        a = np.zeros(PARAM_BYTES // 4, np.float32)
        a[0:4] = (self._dt, mult, 0.62 * self.trail, self.alpha * fx.far_alpha(eye_dist))
        a[4:8] = (self.width_px, self.min_len_px, self.max_len_px, 0.10)
        a[8:12] = (self.life_min, self.life_max, 0.45, 0.9)
        a[12:16] = (self.min_depth, self.min_speed, self.base_density * self.density * self._preset_density, max_range)
        a[16:20] = (self._pool_cap, self.colour, self._attempts, eye_dist)
        a.view(np.uint32)[20] = self._frame & 0xFFFFFFFF
        app.ctx.queue.write_buffer(self.params, 0, a)

    def compute(self, app, enc):
        cp = enc.begin_compute_pass(label="particles")
        cp.set_bind_group(0, app.frame.bind_group)
        cp.set_bind_group(1, self.bg_c)
        cp.set_pipeline(self.pipe_est)
        cp.dispatch_workgroups(1)
        cp.set_pipeline(self.pipe_upd)
        cp.dispatch_workgroups((self._pool_cap + 63) // 64)
        cp.end()

    def draw(self, app, pass_):
        pass_.set_pipeline(self.pipe_draw)
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, self.bg_r)
        pass_.draw_indirect(self.args, 0)

    # ------------------------------------------------------------------------------------------ debugging / tests
    def read_state(self, app, n: int | None = None):
        """Synchronous read-back (tests / tools only): (active_count, wet_frac, particles float32[n, 8])."""
        q = app.ctx.queue
        app.ctx.sync()
        a = np.frombuffer(q.read_buffer(self.args, 0, ARGS_BYTES), np.uint32).copy()
        af = a.view(np.float32)
        active = int(a[1])
        n = active if n is None else min(n, MAX_POOL)
        parts = np.frombuffer(q.read_buffer(self.buf, 0, max(n, 1) * PARTICLE_BYTES), np.float32).reshape(-1, 8).copy()
        return active, float(af[5]), parts

    # ------------------------------------------------------------------------------------------ UI / state
    def ui(self, app):
        from imgui_bundle import imgui
        imgui.set_next_item_width(-120)
        ch, i = imgui.combo("Quality##p", self.PRESETS.index(self.preset), ["Low", "Medium", "High"])
        if ch:
            self.set_preset(self.PRESETS[i])
        imgui.set_next_item_width(-120)
        ch, v = imgui.slider_float("Density##p", self.density, 0.1, 4.0, "%.2f x", imgui.SliderFlags_.logarithmic)
        if ch:
            self.density = v
        imgui.set_next_item_width(-120)
        ch, v = imgui.slider_float("Animation speed", self.anim_speed, 0.1, 5.0, "%.2f x", imgui.SliderFlags_.logarithmic)
        if ch:
            self.anim_speed = v
        imgui.set_next_item_width(-120)
        ch, v = imgui.slider_float("Trail length", self.trail, 0.2, 3.0, "%.2f x")
        if ch:
            self.trail = v
        imgui.set_next_item_width(-120)
        ch, v = imgui.slider_float("Line width (px)", self.width_px, 1.0, 4.0, "%.1f")
        if ch:
            self.width_px = v
        imgui.set_next_item_width(-120)
        ch, v = imgui.slider_float("Opacity##p", self.alpha, 0.1, 1.0, "%.2f")
        if ch:
            self.alpha = v
        imgui.set_next_item_width(-120)
        ch, i = imgui.combo("Colour by", self.colour, list(self.COLOURS))
        if ch:
            self.colour = i
        _, self.const_speed = imgui.checkbox("Constant on-screen speed", self.const_speed)
        imgui.text_disabled(f"pool {self._pool_cap // 1000}k particles max; count follows the wet screen area")

    def save_state(self):
        return {"preset": self.preset, "density": self.density, "anim_speed": self.anim_speed, "trail": self.trail,
                "width_px": self.width_px, "alpha": self.alpha, "colour": self.colour, "const_speed": self.const_speed}

    def load_state(self, d):
        self.preset = d.get("preset", self.preset) if d.get("preset") in fx.QUALITY else self.preset
        self.density = float(d.get("density", self.density))
        self.anim_speed = float(d.get("anim_speed", self.anim_speed))
        self.trail = float(d.get("trail", self.trail))
        self.width_px = float(d.get("width_px", self.width_px))
        self.alpha = float(d.get("alpha", self.alpha))
        self.colour = int(d.get("colour", self.colour))
        self.const_speed = bool(d.get("const_speed", self.const_speed))
