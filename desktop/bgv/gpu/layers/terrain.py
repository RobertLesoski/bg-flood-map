"""Terrain + water layers.

TerrainLayer  - hillshaded terrain drawn from the height texture (vertex pulling, no vertex buffers) as frustum-culled,
                distance-LOD'd 128x128-texel chunks, one instanced draw per LOD.  Also draws the sky (full-screen
                triangle, depth test off).  Owns `app.terrain_mesh` (visible chunk lists) that WaterLayer reuses.
WaterLayer    - second pass over the same mesh, only the chunks that hold water at the displayed hours.  Surface
                Z = ground*E + depth*E_w (+ a tiny lift), depth taken from the flow array AFTER interpolating the codes,
                blended between the two hours.  Fragment: depth colour ramp, ripples advected along the flow velocity,
                sun specular + fresnel sky reflection, foam where fast / at the shore, soft alpha at the wet edge.

Mesh: chunk = 128 x 128 texels (= flow-grid pixels); LOD l uses (128 >> l)^2 quads, l = 0..5.  Cracks between chunks
of different LOD are closed in the vertex shader: an edge vertex whose neighbour is coarser snaps to the neighbour's
vertex grid (instance word `.w` holds the four level differences).
"""
from __future__ import annotations

import math
import time

import numpy as np
import wgpu

from ...data.flow import block_max, hour_pair
from .. import fastcull
from ..cull import frustum_planes, planes_for
from ..flowtex import COMMON_WGSL, CHUNK_TEXELS
from . import Layer

CHUNK = CHUNK_TEXELS
MAX_LOD = 5
INST_BYTES = 16
MAX_INSTANCES = 4096


# ============================================================================================ WGSL
NOISE_WGSL = """
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash2(p: vec2<i32>) -> vec2<f32> {
  let h = pcg(u32(p.x) * 1597334677u ^ pcg(u32(p.y) * 3812015801u));
  return vec2<f32>(f32(h & 0xffffu), f32(h >> 16u)) / 32767.5 - vec2<f32>(1.0);
}
// gradient noise, returns (value, d/dx, d/dy)
fn noised(p: vec2<f32>) -> vec3<f32> {
  let i = vec2<i32>(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  let ga = hash2(i);
  let gb = hash2(i + vec2<i32>(1, 0));
  let gc = hash2(i + vec2<i32>(0, 1));
  let gd = hash2(i + vec2<i32>(1, 1));
  let va = dot(ga, f);
  let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
  let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
  let vd = dot(gd, f - vec2<f32>(1.0, 1.0));
  let k = va - vb - vc + vd;
  let v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
  let d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx * k + vec2<f32>(vb, vc) - vec2<f32>(va));
  return vec3<f32>(v * 1.4, d * 1.4);
}
"""

CHUNK_WGSL = """
struct Inst { @location(0) c: vec4<u32> };    // cx, cy, lod, snap word (W | E<<4 | N<<8 | S<<12)
struct MeshVert {
  texel: vec2<i32>,
  wxy: vec2<f32>,
  lod: u32,
};
fn mesh_vertex(vi: u32, inst: vec4<u32>) -> MeshVert {
  let lod = inst.z;
  let n = 128u >> lod;
  var a = vi % (n + 1u);
  var b = vi / (n + 1u);
  let sw = inst.w & 15u;
  let se = (inst.w >> 4u) & 15u;
  let sn = (inst.w >> 8u) & 15u;
  let ss = (inst.w >> 12u) & 15u;
  if (a == 0u) { b = (b >> sw) << sw; }
  if (a == n)  { b = (b >> se) << se; }
  if (b == 0u) { a = (a >> sn) << sn; }
  if (b == n)  { a = (a >> ss) << ss; }
  let tx = min(i32(inst.x * 128u + (a << lod)), i32(frame.gridpx.x) - 1);
  let ty = min(i32(inst.y * 128u + (b << lod)), i32(frame.gridpx.y) - 1);
  var o: MeshVert;
  o.texel = vec2<i32>(tx, ty);
  o.wxy = texel_to_world(f32(tx), f32(ty));
  o.lod = lod;
  return o;
}
"""

SKY_WGSL = COMMON_WGSL + """
struct SkyOut { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> };
@vertex fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: SkyOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.ndc = p[vi];
  return o;
}
@fragment fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
  let far = frame.inv_view_proj * vec4<f32>(in.ndc, 0.0001, 1.0);
  let near = frame.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - near.xyz / near.w);
  return vec4<f32>(sky_color(dir), 1.0);
}
"""

# used instead of basemap_wgsl(1) (bgv/gpu/layers/basemap.py) when app.basemap does not exist: same API, never draws
BASEMAP_OFF_WGSL = """
fn basemap_color(terrain_uv: vec2<f32>) -> vec4<f32> { return vec4<f32>(0.0); }
"""

TERRAIN_WGSL = COMMON_WGSL + NOISE_WGSL + CHUNK_WGSL + """
struct TOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) wpos: vec3<f32>,
  @location(1) uv: vec2<f32>,
};
@vertex fn vs_terrain(@builtin(vertex_index) vi: u32, inst: Inst) -> TOut {
  let m = mesh_vertex(vi, inst.c);
  let z = height_texel(m.texel.x, m.texel.y) * frame.grid.w;
  var o: TOut;
  o.wpos = vec3<f32>(m.wxy, z);
  o.uv = world_to_uv(m.wxy);
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  return o;
}

fn hypso(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.20, 0.34, 0.15);
  let c1 = vec3<f32>(0.33, 0.47, 0.19);
  let c2 = vec3<f32>(0.50, 0.53, 0.24);
  let c3 = vec3<f32>(0.60, 0.50, 0.30);
  let c4 = vec3<f32>(0.70, 0.60, 0.46);
  var c = mix(c0, c1, smoothstep(0.0, 0.25, t));
  c = mix(c, c2, smoothstep(0.25, 0.5, t));
  c = mix(c, c3, smoothstep(0.5, 0.75, t));
  return mix(c, c4, smoothstep(0.75, 1.0, t));
}

// heightfield shadow: march toward the sun through the height texture (exaggerated heights); 1 = lit
fn terrain_shadow(p: vec3<f32>, L: vec3<f32>, dist: f32) -> f32 {
  if (L.z <= 0.03) { return 0.0; }
  if (dist > 9000.0) { return 1.0; }
  let E = frame.grid.w;
  var res = 1.0;
  var t = 6.0;
  let nsteps = gov_shadow_steps();
  let grow = pow(144.2, 1.0 / f32(max(nsteps - 1, 1)));         // 1.22 per step at the full 26 steps: same range for fewer steps
  let zmax_e = frame.view.w * E;
  let can_stop = p.z <= zmax_e || L.z >= 0.1;                  // once the ray is above every terrain height no later step can lower res
  for (var i = 0; i < nsteps; i = i + 1) {
    let q = p + L * t;
    if (can_stop && 10.0 * (q.z - zmax_e) >= t) { break; }
    let uv = world_to_uv(q.xy);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) { break; }
    let ix = i32(uv.x * frame.gridpx.x);
    let iy = i32(uv.y * frame.gridpx.y);
    let h = height_texel(ix, iy) * E;
    res = min(res, 10.0 * (q.z - h) / t);
    if (res <= -1.0) { break; }                                // fully shadowed: the result below is 0 whatever follows
    t = t * grow;
  }
  return clamp(res * 0.5 + 0.5, 0.0, 1.0) * clamp(res + 1.0, 0.0, 1.0);
}

@fragment fn fs_terrain(in: TOut) -> @location(0) vec4<f32> {
  let E = frame.grid.w;
  let eye = frame.eye.xyz;
  let to_eye = eye - in.wpos;
  let dist = length(to_eye);
  let V = to_eye / max(dist, 1e-3);
  let s = textureSample(slope_tex, slope_samp, in.uv);
  let bmc = basemap_color(in.uv);                             // OSM drape (bgv/gpu/layers/basemap.py): rgb linear, a = opacity (0 = off)
  var g = s.xy * E * frame.opts.w;                            // slope of the exaggerated surface (+ relief boost)
  // grid-independent detail: small world-space noise bends the normal and varies the albedo near the camera
  let near_fade = 1.0 - smoothstep(150.0, 2500.0, dist);
  var d1 = vec3<f32>(0.0);
  var d2 = vec3<f32>(0.0);
  if (near_fade > 0.0) {                                      // beyond 2.5 km the detail terms are multiplied by 0: skip them
    d1 = noised(in.wpos.xy * 0.45);
    d2 = noised(in.wpos.xy * 2.3 + vec2<f32>(17.0, 3.0));
  }
  g += (d1.yz * 0.05 + d2.yz * 0.025) * near_fade * (1.0 - 0.85 * bmc.a);   // a printed map has no grain
  let N = normalize(vec3<f32>(-g.x, -g.y, 1.0));
  let L = frame.sun.xyz;
  let ndl = dot(N, L);
  let sh = terrain_shadow(in.wpos + vec3<f32>(0.0, 0.0, 0.5 + dist * 4e-4), L, dist);
  let diff = clamp(ndl, 0.0, 1.0) * sh;
  let sky_amb = mix(vec3<f32>(0.26, 0.25, 0.22), vec3<f32>(0.36, 0.48, 0.68), N.z * 0.5 + 0.5);
  let zt = clamp((in.wpos.z / E - frame.view.z) / max(frame.view.w - frame.view.z, 1.0), 0.0, 1.0);
  var albedo = hypso(zt);
  let n_lo = noised(in.wpos.xy * 0.004).x;
  let n_mid = noised(in.wpos.xy * 0.03 + vec2<f32>(40.0, 8.0)).x;
  albedo *= 1.0 + 0.16 * n_lo + 0.08 * n_mid + (d1.x * 0.08 + d2.x * 0.06) * near_fade;
  let slope_len = length(s.xy);
  albedo = mix(albedo, vec3<f32>(0.42, 0.37, 0.30), smoothstep(0.10, 0.40, slope_len) * 0.6);
  let cav = clamp(s.z * 60.0, -1.0, 1.0);
  let ao = 1.0 - 0.30 * max(cav, 0.0) + 0.10 * max(-cav, 0.0);
  albedo = mix(vec3<f32>(0.50, 0.53, 0.56), albedo, smoothstep(0.35, 0.9, s.w));
  let light = (vec3<f32>(1.0, 0.95, 0.85) * frame.sun.w * diff * 1.25 + sky_amb * 0.72) * ao;
  var col = albedo * light;
  if (bmc.a > 0.0) {
    // OSM drape: the map keeps its own colours (flat, sunlit ground shows the tile colour unchanged: light / lref = 1),
    // hillshade, cast shadows and AO from the terrain lighting modulate it so the relief is still readable
    let lref = vec3<f32>(1.0, 0.95, 0.85) * frame.sun.w * max(L.z, 0.0) * 1.25 + vec3<f32>(0.36, 0.48, 0.68) * 0.72;
    let r = light / lref;                                           // sun-facing slopes brighten, shade / shadow darken
    col = mix(col, bmc.rgb * (min(r, vec3<f32>(1.0)) + min(max(r - 1.0, vec3<f32>(0.0)), vec3<f32>(0.5)) * 0.35), bmc.a);
  }
  col = haze(col, dist, -V);
  return vec4<f32>(col, 1.0);
}
"""

WATER_WGSL = COMMON_WGSL + NOISE_WGSL + CHUNK_WGSL + """
struct WP {
  a: vec4<f32>,   // opacity, absorption (1/m), smooth-shore flag (1 = sub-texel shoreline), turbidity
  b: vec4<f32>,   // spare
};
@group(1) @binding(0) var<uniform> WPU: WP;

struct WOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) wpos: vec3<f32>,
  @location(1) uv: vec2<f32>,
};
@vertex fn vs_water(@builtin(vertex_index) vi: u32, inst: Inst) -> WOut {
  let m = mesh_vertex(vi, inst.c);
  let uv = world_to_uv(m.wxy);
  let fl = flow_at(uv);
  let zg = height_texel(m.texel.x, m.texel.y) * frame.grid.w;
  var dpt = 0.0;
  if (fl.depth > frame.enc.z) { dpt = fl.depth; }
  let dist = length(frame.eye.xyz - vec3<f32>(m.wxy, zg));
  let lift = 0.03 + dist * 2.5e-5 + f32(1u << m.lod) * 0.02;
  var o: WOut;
  o.wpos = vec3<f32>(m.wxy, zg + dpt * frame.enc.w + lift);
  o.uv = uv;
  o.pos = frame.view_proj * vec4<f32>(o.wpos, 1.0);
  return o;
}

// ---- sub-texel water depth ---------------------------------------------------------------------------------------
// The data holds depth only at 4-5 m cells, zero below 0.02 m, so thresholding the bilinear depth gives a stair-stepped
// shoreline.  Instead reconstruct the WATER SURFACE LEVEL (ground + depth, a smooth field: flat for ponds) from the wet
// cells around the point with a cubic B-spline weighted average (wet cells only), subtract a Catmull-Rom (bicubic)
// ground height, and use the result as depth: the shoreline is where the surface plane meets the terrain, at sub-texel
// accuracy and smooth by construction.
fn bspline_w(t: f32) -> vec4<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return vec4<f32>((1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0, (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
                   (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0, t3 / 6.0);
}
fn catmull_w(t: f32) -> vec4<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
fn hour_depth(es: f32, ws: f32, ground: f32) -> f32 {
  if (ws <= 1.0e-4) { return 0.0; }
  return max(es / ws - ground, 0.0) * smoothstep(0.03, 0.35, ws);
}
fn smooth_depth(uv: vec2<f32>) -> f32 {
  let f = uv * frame.gridpx.xy - vec2<f32>(0.5);
  let fl0 = floor(f);
  let t = f - fl0;
  let i0 = vec2<i32>(fl0) - vec2<i32>(1);
  let wbx = bspline_w(t.x);
  let wby = bspline_w(t.y);
  let wcx = catmull_w(t.x);
  let wcy = catmull_w(t.y);
  let la = flow_layer(u32(frame.flow.z));
  let lb = flow_layer(u32(frame.flow.w));
  let mixv = frame.flow.y;
  let dmax = frame.enc.x;
  let mx = vec2<i32>(i32(frame.gridpx.x) - 1, i32(frame.gridpx.y) - 1);
  var ground = 0.0;
  var wsa = 0.0; var esa = 0.0; var wsb = 0.0; var esb = 0.0;
  for (var jy = 0; jy < 4; jy = jy + 1) {
    for (var jx = 0; jx < 4; jx = jx + 1) {
      let c = clamp(i0 + vec2<i32>(jx, jy), vec2<i32>(0), mx);
      let z = textureLoad(height_tex, c, 0).r;
      ground = ground + z * (wcx[jx] * wcy[jy]);
      let wb = wbx[jx] * wby[jy];
      let ca = textureLoad(flow_tex, c, la, 0).r;
      if (ca > 0.0) { wsa = wsa + wb; esa = esa + wb * (z + ca * ca * dmax); }
      if (mixv > 0.0) {
        let cb = textureLoad(flow_tex, c, lb, 0).r;
        if (cb > 0.0) { wsb = wsb + wb; esb = esb + wb * (z + cb * cb * dmax); }
      }
    }
  }
  let da = hour_depth(esa, wsa, ground);
  if (mixv <= 0.0) { return da; }
  return mix(da, hour_depth(esb, wsb, ground), mixv);
}

fn depth_ramp(d: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.34, 0.62, 0.66);
  let c1 = vec3<f32>(0.15, 0.50, 0.74);
  let c2 = vec3<f32>(0.06, 0.33, 0.64);
  let c3 = vec3<f32>(0.03, 0.18, 0.46);
  let c4 = vec3<f32>(0.01, 0.07, 0.26);
  var c = mix(c0, c1, smoothstep(0.0, 0.3, d));
  c = mix(c, c2, smoothstep(0.3, 0.9, d));
  c = mix(c, c3, smoothstep(0.9, 2.2, d));
  return mix(c, c4, smoothstep(2.2, 5.5, d));
}

// ripple slope: two flow-map phases (half a cycle apart) of 4-octave gradient noise, each advected by the velocity.
// Calm where the water is at rest (ponds, the river 'at rest' of the tiers data), rougher along fast flow.
fn ripple_slope(wp: vec2<f32>, vel: vec2<f32>, time: f32, speed: f32, dist: f32) -> vec2<f32> {
  let cyc = 3.0;
  let t = time / cyc;
  let p0 = fract(t);
  let p1 = fract(t + 0.5);
  let w0 = 1.0 - abs(2.0 * p0 - 1.0);
  let w1 = 1.0 - abs(2.0 * p1 - 1.0);
  let still = 1.0 - smoothstep(0.0, 0.35, speed);
  let v = vel * 1.6 + vec2<f32>(0.16, 0.07) * still;
  let o0 = v * (p0 - 0.5) * cyc;
  let o1 = v * (p1 - 0.5) * cyc;
  var g0 = vec2<f32>(0.0); var g1 = vec2<f32>(0.0);
  let ps = dist * 0.83 / max(frame.view.y, 1.0);                 // ground metres per pixel here
  var f = 0.14; var a = 1.0;
  for (var i = 0; i < 4; i = i + 1) {
    let vis = 1.0 - smoothstep(0.07, 0.20, f * ps);            // octaves finer than ~5 px would only alias / shimmer
    g0 += noised((wp - o0) * f + vec2<f32>(f32(i) * 7.3, 1.7)).yz * (a * vis);
    g1 += noised((wp - o1) * f + vec2<f32>(f32(i) * 7.3 + 31.0, 9.1)).yz * (a * vis);
    f *= 2.35; a *= 0.62;
  }
  let g = (g0 * w0 + g1 * w1);
  let amp = (0.035 + 0.30 * smoothstep(0.04, 1.8, speed)) * frame.opts.x * (1.0 - smoothstep(4000.0, 14000.0, dist));
  return g * amp;
}

fn foam_pattern(wp: vec2<f32>, vel: vec2<f32>, time: f32) -> f32 {
  let cyc = 2.0;
  let t = time / cyc;
  let p0 = fract(t);
  let p1 = fract(t + 0.5);
  let w0 = 1.0 - abs(2.0 * p0 - 1.0);
  let w1 = 1.0 - abs(2.0 * p1 - 1.0);
  let n0 = noised((wp - vel * (p0 - 0.5) * cyc) * 0.9).x * 0.65 + noised((wp - vel * (p0 - 0.5) * cyc) * 2.7).x * 0.35;
  let n1 = noised((wp - vel * (p1 - 0.5) * cyc) * 0.9 + vec2<f32>(5.0, 11.0)).x * 0.65
         + noised((wp - vel * (p1 - 0.5) * cyc) * 2.7 + vec2<f32>(5.0, 11.0)).x * 0.35;
  return (n0 * w0 + n1 * w1) * 0.5 + 0.5;
}

@fragment fn fs_water(in: WOut) -> @location(0) vec4<f32> {
  let fl = flow_at(in.uv);
  var d = fl.depth;
  let smooth_on = WPU.a.z > 0.5;
  if (smooth_on) {
    // deep interior (all four nearest cells >= ~0.3 m in both hours): the bilinear depth is already smooth, skip the
    // 16-tap reconstruction; it only matters along the shoreline band
    let ga = textureGather(0, flow_tex, flow_samp, in.uv, flow_layer(u32(frame.flow.z)));
    var mn = min(min(ga.x, ga.y), min(ga.z, ga.w));
    if (frame.flow.y > 0.0) {
      let gb = textureGather(0, flow_tex, flow_samp, in.uv, flow_layer(u32(frame.flow.w)));
      mn = min(mn, min(min(gb.x, gb.y), min(gb.z, gb.w)));
    }
    // the 16-tap reconstruction only changes the picture where a 5 m cell spans more than ~1 px: beyond gov_smooth_ps()
    // metres per pixel the bilinear depth is used (cross-fade over the last two metres per pixel)
    let ps_s = length(frame.eye.xyz - in.wpos) * 0.83 / max(frame.view.y, 1.0);
    let sp = gov_smooth_ps();
    if (mn < 0.2 && ps_s < sp) {
      let ds = smooth_depth(in.uv);
      d = mix(ds, d, smoothstep(sp - 2.0, sp, ps_s));
    }
  }
  let fw = fwidth(d);
  let wet = frame.enc.z;
  var edge = smoothstep(wet * 0.8, wet + 0.09, d);
  if (smooth_on) { edge = smoothstep(0.0, max(0.04, 1.6 * fw), d); }
  edge = edge * smoothstep(0.2, 0.9, fl.valid);
  if (edge <= 0.004) { discard; }
  let dd = max(d, 0.0);
  let eye = frame.eye.xyz;
  let to_eye = eye - in.wpos;
  let dist = length(to_eye);
  let V = to_eye / max(dist, 1e-3);
  let speed = length(fl.vel);
  let time = frame.eye.w;
  let ps = dist * 0.83 / max(frame.view.y, 1.0);                // ground metres per pixel

  let sl = ripple_slope(in.wpos.xy, fl.vel, time, speed, dist);
  let N = normalize(vec3<f32>(-sl.x, -sl.y, 1.0));
  let L = frame.sun.xyz;
  let H = normalize(L + V);
  let ndv = max(dot(N, V), 0.0);
  let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);

  // body colour: depth ramp, slightly silty (browner, lighter) where it moves fast
  var body = depth_ramp(dd);
  let turb = WPU.a.w * (0.35 + 0.65 * smoothstep(0.1, 1.5, speed));
  body = mix(body, vec3<f32>(0.40, 0.36, 0.25) * (0.55 + 0.45 * smoothstep(0.0, 1.0, dd)), turb * 0.45);
  let diff = clamp(dot(N, L) * 0.5 + 0.55, 0.0, 1.0);
  let lit = body * (vec3<f32>(1.0, 0.97, 0.9) * frame.sun.w * diff * 0.75 + vec3<f32>(0.34, 0.40, 0.48));
  // Beer-Lambert style opacity: the ground shows through shallow water
  let a_body = mix(0.22, 0.97, 1.0 - exp(-dd * WPU.a.y));
  // sky reflection (Fresnel); rougher far away
  let R = reflect(-V, N);
  let refl = sky_base(vec3<f32>(R.x, R.y, abs(R.z)));
  let rough = smoothstep(0.5, 4.0, ps);
  let refl_amt = clamp(fres * mix(1.0, 0.75, rough) + 0.02, 0.0, 0.95);
  var rgb = refl * refl_amt + lit * (a_body * (1.0 - refl_amt));
  var a = refl_amt + (1.0 - refl_amt) * a_body;
  // sun glitter (highlight widens with distance so it does not sparkle)
  let nh = max(dot(N, H), 0.0);
  let gloss = 1.0 - smoothstep(0.25, 2.0, ps);
  let spec = pow(nh, mix(60.0, 280.0, gloss)) * mix(0.8, 2.2, gloss) + pow(nh, 30.0) * 0.12 + pow(nh, 8.0) * 0.04;
  // shallow sheet flow is turbid and rough: much less mirror-like than a river; the glint saturates smoothly (never flat white)
  let glossy = mix(0.30, 1.0, smoothstep(0.08, 1.2, dd));
  let spec_t = (1.0 - exp(-spec * 0.55)) * glossy * 1.25;
  rgb += vec3<f32>(1.0, 0.95, 0.82) * spec_t * frame.sun.w * smoothstep(0.0, 0.08, ndv);
  // highlight roll-off: a soft shoulder above 0.75 instead of clipping to flat white, so the sun's glare on a river keeps its gradient
  rgb = select(rgb, 0.75 + 0.30 * (1.0 - exp(-(rgb - vec3<f32>(0.75)) / 0.30)), rgb > vec3<f32>(0.75));

  // foam: fast water and the shoreline
  let fp = foam_pattern(in.wpos.xy, fl.vel * 1.3, time);
  let fast = smoothstep(0.7, 2.4, speed);
  let shore = 1.0 - smoothstep(0.02, 0.16, dd);
  var foam = smoothstep(0.55, 0.95, fp + fast * 0.30) * fast * 0.62;
  foam += smoothstep(0.55, 0.95, fp) * shore * 0.38;
  foam = clamp(foam, 0.0, 1.0) * (1.0 - smoothstep(600.0, 5000.0, dist) * 0.7);
  let foam_col = vec3<f32>(0.93, 0.97, 1.0) * (0.55 + 0.5 * diff);
  rgb = foam_col * foam + rgb * (1.0 - foam);
  a = foam + a * (1.0 - foam);
  rgb = rgb * mix(1.0, 1.15, fast * 0.4);

  // aerial perspective on the (un-premultiplied) colour
  let c = haze(rgb / max(a, 1.0e-3), dist, -V);
  let k = edge * WPU.a.x;
  return vec4<f32>(c * (a * k), a * k);
}
"""


# ============================================================================================ mesh
def _lod_indices(n: int) -> np.ndarray:
    """uint16 triangle list for an n x n quad chunk, CCW seen from above (+Z up, row 0 north)."""
    b, a = np.mgrid[0:n, 0:n]
    v00 = (b * (n + 1) + a).ravel()           # NW   (a = east, b = south)
    v10 = v00 + 1                             # NE
    v01 = v00 + (n + 1)                       # SW
    v11 = v01 + 1                             # SE
    idx = np.stack([v00, v01, v10, v10, v01, v11], axis=1).astype(np.uint16)
    return np.ascontiguousarray(idx.reshape(-1))


class TerrainMesh:
    """Chunk grid, per-frame visibility + LOD selection, instance buffers and index buffers."""

    def __init__(self, app):
        self.app = app
        dev = app.ctx.device
        t = app.terrain
        fr = t.frame
        self.W, self.H = fr.width, fr.height
        self.nx = -(-self.W // CHUNK)
        self.ny = -(-self.H // CHUNK)
        sx, sy = fr.size_m
        cx0 = np.arange(self.nx) * CHUNK
        cx1 = np.minimum(cx0 + CHUNK, self.W - 1)
        cy0 = np.arange(self.ny) * CHUNK
        cy1 = np.minimum(cy0 + CHUNK, self.H - 1)

        def wx(i): return ((i + 0.5) / self.W - 0.5) * sx
        def wy(j): return (0.5 - (j + 0.5) / self.H) * sy
        self.x_min = np.broadcast_to(wx(cx0)[None, :], (self.ny, self.nx)).astype(np.float64)
        self.x_max = np.broadcast_to(wx(cx1)[None, :], (self.ny, self.nx)).astype(np.float64)
        self.y_max = np.broadcast_to(wy(cy0)[:, None], (self.ny, self.nx)).astype(np.float64)   # row 0 = north
        self.y_min = np.broadcast_to(wy(cy1)[:, None], (self.ny, self.nx)).astype(np.float64)
        h = t.heights
        zmin = np.empty((self.ny, self.nx), np.float64)
        zmax = np.empty((self.ny, self.nx), np.float64)
        for j in range(self.ny):
            for i in range(self.nx):
                blk = h[cy0[j]:cy1[j] + 1, cx0[i]:cx1[i] + 1]
                zmin[j, i], zmax[j, i] = blk.min(), blk.max()
        self.z_min, self.z_max = zmin, zmax
        self.centers = np.stack([(self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5], -1).reshape(-1, 2)
        # index buffers
        self.index_bufs, self.index_counts = [], []
        for lod in range(MAX_LOD + 1):
            idx = _lod_indices(CHUNK >> lod)
            buf = dev.create_buffer(label=f"terrain-idx{lod}", size=idx.nbytes,
                                    usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST)
            app.ctx.queue.write_buffer(buf, 0, idx)
            self.index_bufs.append(buf)
            self.index_counts.append(int(idx.size))
        app.ctx.track("terrain indices", sum(c * 2 for c in self.index_counts))
        self.inst_terrain = dev.create_buffer(label="terrain-inst", size=MAX_INSTANCES * INST_BYTES,
                                              usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        self.inst_water = dev.create_buffer(label="water-inst", size=MAX_INSTANCES * INST_BYTES,
                                            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        # per-frame results
        self.draws_terrain: list[tuple[int, int, int]] = []     # (lod, first_instance, count)
        self.draws_water: list[tuple[int, int, int]] = []
        self.lod_scale = 1.0
        self.n_visible = 0
        self.n_water = 0
        self.tris = 0
        self.tris_water = 0
        self._lods = np.zeros((self.ny, self.nx), np.int32)
        self._vis = np.zeros((self.ny, self.nx), bool)
        self._insts = np.zeros((0, 4), np.uint32)
        self._order = None
        self.cull_ms = 0.0
        # ---- flat arrays for the fast update()
        n = self.nx * self.ny
        self._xmin, self._xmax = self.x_min.reshape(-1).copy(), self.x_max.reshape(-1).copy()
        self._ymin, self._ymax = self.y_min.reshape(-1).copy(), self.y_max.reshape(-1).copy()
        self._tmp_a, self._tmp_b, self._tmp_c, self._tmp_d = (np.empty(n, np.float64) for _ in range(4))
        self._ctr = np.empty((n, 3), np.float64)
        self._ext = np.empty((n, 3), np.float64)
        self._ctr[:, 0] = (self._xmin + self._xmax) * 0.5
        self._ctr[:, 1] = (self._ymin + self._ymax) * 0.5
        self._ext[:, 0] = (self._xmax - self._xmin) * 0.5
        self._ext[:, 1] = (self._ymax - self._ymin) * 0.5
        self._zkey = self._ckey = self._ckey_z = None
        self._wet_prev = None
        self._big = np.zeros((self.ny + 2, self.nx + 2), np.int32)
        self._word2 = np.zeros((self.ny, self.nx), np.int32)
        self._inst = np.zeros((n, 4), np.uint32)
        self._inst[:, 0] = np.tile(np.arange(self.nx, dtype=np.uint32), self.ny)
        self._inst[:, 1] = np.repeat(np.arange(self.ny, dtype=np.uint32), self.nx)
        self._raw_t = self._raw_w = None
        self.vis = np.zeros((self.ny, self.nx), bool)
        self._cx = np.ascontiguousarray(self._ctr[:, 0])
        self._cy = np.ascontiguousarray(self._ctr[:, 1])
        self._ex = np.ascontiguousarray(self._ext[:, 0])
        self._ey = np.ascontiguousarray(self._ext[:, 1])
        self._cz = np.zeros(n)
        self._ez = np.zeros(n)
        self._nowet = np.zeros(n, bool)
        self._lod = np.zeros(n, np.int32)
        self._word = np.zeros(n, np.int32)
        self._visf = np.zeros(n, bool)
        self._d2 = np.zeros(n, np.float64)
        self._data_t = np.zeros((MAX_INSTANCES, 4), np.uint32)
        self._data_w = np.zeros((MAX_INSTANCES, 4), np.uint32)
        self._cnt_t = np.zeros(MAX_LOD + 1, np.int64)
        self._cnt_w = np.zeros(MAX_LOD + 1, np.int64)
        self._key_all = np.zeros(n)
        self._lod_all = self._lod

    # --------------------------------------------------------------------------------------------
    frustum_planes = staticmethod(frustum_planes)

    def update(self, eye, vp, exag: float, lod_distance: float, flow_wet: np.ndarray | None, water_headroom: float,
               planes: np.ndarray | None = None):
        """Per-frame visibility / LOD / instance packing.  Same result as `update_reference` (tested) at a fraction of the
        cost: one numba kernel (bgv/gpu/fastcull.py; a numpy version when numba is missing) and nothing recomputed or
        re-uploaded when the inputs did not change."""
        t0 = time.perf_counter()
        wet = None if flow_wet is None else np.asarray(flow_wet, bool).reshape(-1)
        e0, e1, e2 = float(eye[0]), float(eye[1]), float(eye[2])
        if planes is None:
            planes = frustum_planes(vp)
        # ---- exaggeration dependent boxes (cached)
        zk = (float(exag), float(water_headroom))
        if zk != self._zkey:
            z0 = self.z_min.reshape(-1) * exag
            z1 = self.z_max.reshape(-1) * exag
            self._z0, self._z1 = z0, z1
            zt = z1 + water_headroom * exag
            self._ctr[:, 2] = (z0 + zt) * 0.5
            self._ext[:, 2] = (zt - z0) * 0.5
            self._cz = np.ascontiguousarray(self._ctr[:, 2])
            self._ez = np.ascontiguousarray(self._ext[:, 2])
            self._zkey = zk
        # ---- cache: identical camera / params / wet table -> the buffers written last frame are still right
        ckey = (e0, e1, e2, planes.tobytes(), float(lod_distance), zk)
        if ckey == self._ckey and ((wet is None and self._wet_prev is None) or (
                wet is not None and self._wet_prev is not None and np.array_equal(wet, self._wet_prev))):
            self.cull_ms = 0.0
            return
        self._ckey = ckey
        self._wet_prev = None if wet is None else wet.copy()
        ld = max(float(lod_distance), 1.0)
        # lod = floor(log2(dist / lod_distance)) + 1 clipped to 0..MAX_LOD = number of thresholds lod_d * 2^k (k < MAX_LOD) passed
        thr = (ld * (2.0 ** np.arange(MAX_LOD))) ** 2
        if fastcull.available():
            nvis, nwat = fastcull.terrain_kernel(
                self._xmin, self._xmax, self._ymin, self._ymax, self._z0, self._z1, self._cx, self._cy, self._cz,
                self._ex, self._ey, self._ez, planes, e0, e1, e2, thr, self.nx, self.ny,
                wet if wet is not None else self._nowet, wet is not None, MAX_INSTANCES, self._lod, self._word, self._visf,
                self._d2, self._data_t, self._cnt_t, self._data_w, self._cnt_w)
            self._lods = self._lod.reshape(self.ny, self.nx)
            self.vis = self._vis = self._visf.reshape(self.ny, self.nx)
            data, wdata = self._data_t[:nvis], self._data_w[:nwat]
            self.draws_terrain = self._draws_from(self._cnt_t)
            self.draws_water = self._draws_from(self._cnt_w)
            lod, key = self._lod, self._lod * 1.0e12 + self._d2
        else:
            lod, key, data, wdata = self._update_numpy(e0, e1, e2, planes, thr, wet)
        self.n_visible = len(data)
        self.n_water = len(wdata)
        q = self.app.ctx.queue
        if len(data):
            raw = data.tobytes()
            if raw != self._raw_t:
                self._raw_t = raw
                q.write_buffer(self.inst_terrain, 0, data)
        if len(wdata):
            raw = wdata.tobytes()
            if raw != self._raw_w:
                self._raw_w = raw
                q.write_buffer(self.inst_water, 0, wdata)
        self._key_all = key
        self._lod_all = lod
        ic = self.index_counts
        self.tris = sum(ic[l] // 3 * c for l, _, c in self.draws_terrain)
        self.tris_water = sum(ic[l] // 3 * c for l, _, c in self.draws_water)
        self.cull_ms = (time.perf_counter() - t0) * 1000.0

    def pack_visible(self, sel_mask: np.ndarray):
        """(instance data, draws) of an arbitrary chunk mask (e.g. `mask & mesh.vis`), same ordering as the main lists."""
        s = np.flatnonzero(np.asarray(sel_mask).reshape(-1))
        order = s[np.argsort(self._key_all[s], kind="stable")][:MAX_INSTANCES]
        if order.size == 0:
            return np.zeros((0, 4), np.uint32), []
        inst = self._inst
        inst[:, 2] = self._lod_all
        inst[:, 3] = self._word.reshape(-1)
        cnt = np.bincount(self._lod_all[order], minlength=MAX_LOD + 1)
        return inst[order], self._draws_from(cnt)

    @staticmethod
    def _draws_from(cnt) -> list:
        draws, start = [], 0
        for l in range(MAX_LOD + 1):
            c = int(cnt[l])
            if c:
                draws.append((l, start, c))
                start += c
        return draws

    def _update_numpy(self, e0, e1, e2, planes, thr, wet):
        """numpy twin of terrain_kernel (used when numba is missing)."""
        dx, dy, dz, tmp = self._tmp_a, self._tmp_b, self._tmp_c, self._tmp_d
        np.subtract(self._xmin, e0, out=dx)
        np.subtract(e0, self._xmax, out=tmp)
        np.maximum(dx, tmp, out=dx)
        np.maximum(dx, 0.0, out=dx)
        np.subtract(self._ymin, e1, out=dy)
        np.subtract(e1, self._ymax, out=tmp)
        np.maximum(dy, tmp, out=dy)
        np.maximum(dy, 0.0, out=dy)
        np.subtract(self._z0, e2, out=dz)
        np.subtract(e2, self._z1, out=tmp)
        np.maximum(dz, tmp, out=dz)
        np.maximum(dz, 0.0, out=dz)
        np.multiply(dx, dx, out=dx)
        np.multiply(dy, dy, out=dy)
        np.multiply(dz, dz, out=dz)
        d2 = dx
        np.add(d2, dy, out=d2)
        np.add(d2, dz, out=d2)
        lod = np.searchsorted(thr, d2, side="right").astype(np.int32)
        S = self._ctr @ planes[:, :3].T
        S += self._ext @ np.abs(planes[:, :3]).T
        S += planes[:, 3]
        vis = (S >= 0.0).all(axis=1)
        big = self._big
        l2 = lod.reshape(self.ny, self.nx)
        big[1:-1, 1:-1] = l2
        big[0, 1:-1] = l2[0]
        big[-1, 1:-1] = l2[-1]
        big[1:-1, 0] = l2[:, 0]
        big[1:-1, -1] = l2[:, -1]
        w2 = self._word2
        np.maximum(big[1:-1, :-2] - l2, 0, out=w2)
        t = np.maximum(big[1:-1, 2:] - l2, 0)
        t <<= 4
        w2 |= t
        t = np.maximum(big[:-2, 1:-1] - l2, 0)
        t <<= 8
        w2 |= t
        t = np.maximum(big[2:, 1:-1] - l2, 0)
        t <<= 12
        w2 |= t
        inst = self._inst
        inst[:, 2] = lod
        inst[:, 3] = w2.reshape(-1)
        key = lod * 1.0e12 + d2                     # group by LOD (near first), front to back inside a group
        self._lods = l2
        self.vis = self._vis = vis.reshape(self.ny, self.nx)
        self._word = w2
        sel = np.flatnonzero(vis)
        order = sel[np.argsort(key[sel], kind="stable")][:MAX_INSTANCES]
        data = inst[order]
        self.draws_terrain = self._draws_from(np.bincount(lod[order], minlength=MAX_LOD + 1)) if len(order) else []
        if wet is not None:
            worder = order[wet[order]]
            wdata = inst[worder]
            self.draws_water = self._draws_from(np.bincount(lod[worder], minlength=MAX_LOD + 1)) if len(worder) else []
        else:
            wdata, self.draws_water = inst[:0], []
        return lod, key, data, wdata

    def update_reference(self, eye, vp, exag: float, lod_distance: float, flow_wet: np.ndarray | None,
                         water_headroom: float):
        """The straightforward (slow) implementation, kept as the executable specification for the tests."""
        import time as _t
        t0 = _t.perf_counter()
        z0 = self.z_min * exag
        z1 = self.z_max * exag
        # LOD from the distance to the chunk box
        dx = np.maximum(np.maximum(self.x_min - eye[0], eye[0] - self.x_max), 0.0)
        dy = np.maximum(np.maximum(self.y_min - eye[1], eye[1] - self.y_max), 0.0)
        dz = np.maximum(np.maximum(z0 - eye[2], eye[2] - z1), 0.0)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        lod = np.clip(np.floor(np.log2(np.maximum(dist, 1e-3) / max(lod_distance, 1.0))) + 1, 0, MAX_LOD).astype(np.int32)
        # frustum
        planes = self.frustum_planes(np.asarray(vp, np.float64))
        vis = np.ones((self.ny, self.nx), bool)
        zt = z1 + water_headroom * exag
        for p in planes:
            px = np.where(p[0] > 0, self.x_max, self.x_min)
            py = np.where(p[1] > 0, self.y_max, self.y_min)
            pz = np.where(p[2] > 0, zt, z0)
            vis &= (p[0] * px + p[1] * py + p[2] * pz + p[3]) >= 0
        self._lods, self._vis = lod, vis
        # snap word: how much COARSER each neighbour is than this chunk
        big = np.pad(lod, 1, mode="edge")
        sw = np.maximum(big[1:-1, :-2] - lod, 0)
        se = np.maximum(big[1:-1, 2:] - lod, 0)
        sn = np.maximum(big[:-2, 1:-1] - lod, 0)
        ss = np.maximum(big[2:, 1:-1] - lod, 0)
        word = (sw | (se << 4) | (sn << 8) | (ss << 12)).astype(np.uint32)
        jj, ii = np.mgrid[0:self.ny, 0:self.nx]
        inst = np.stack([ii.astype(np.uint32), jj.astype(np.uint32), lod.astype(np.uint32), word], -1).reshape(-1, 4)
        flat_dist = dist.reshape(-1)
        flat_lod = lod.reshape(-1)

        def pack(sel_mask: np.ndarray):
            sel = np.flatnonzero(sel_mask.reshape(-1))
            if sel.size == 0:
                return np.zeros((0, 4), np.uint32), []
            # group by LOD (near first), front to back inside a group
            key = flat_lod[sel] * 1e6 + flat_dist[sel]
            sel = sel[np.argsort(key, kind="stable")]
            data = np.ascontiguousarray(inst[sel][:MAX_INSTANCES])
            lods = flat_lod[sel][:MAX_INSTANCES]
            draws, start = [], 0
            for l in range(MAX_LOD + 1):
                cnt = int((lods == l).sum())
                if cnt:
                    draws.append((l, start, cnt))
                    start += cnt
            return data, draws

        self.pack_visible = pack                 # other layers (flowpaths): pack_visible(mask & mesh.vis) -> (data, draws)
        self.vis = vis
        data, self.draws_terrain = pack(vis)
        self.n_visible = len(data)
        if len(data):
            self.app.ctx.queue.write_buffer(self.inst_terrain, 0, data)
        wmask = vis & flow_wet if flow_wet is not None else np.zeros_like(vis)
        wdata, self.draws_water = pack(wmask)
        self.n_water = len(wdata)
        if len(wdata):
            self.app.ctx.queue.write_buffer(self.inst_water, 0, wdata)
        self.tris = sum(self.index_counts[l] // 3 * c for l, _, c in self.draws_terrain)
        self.tris_water = sum(self.index_counts[l] // 3 * c for l, _, c in self.draws_water)
        self.cull_ms = (_t.perf_counter() - t0) * 1000.0

    def draw(self, pass_, draws, inst_buf):
        pass_.set_vertex_buffer(0, inst_buf)
        cur = -1
        for lod, first, cnt in draws:
            if lod != cur:
                pass_.set_index_buffer(self.index_bufs[lod], "uint16")
                cur = lod
            pass_.draw_indexed(self.index_counts[lod], cnt, 0, 0, first)

    def close(self):
        for b in (*self.index_bufs, self.inst_terrain, self.inst_water):
            try:
                b.destroy()
            except Exception:
                pass
        self.app.ctx.untrack("terrain indices")


def _vertex_state(module, entry):
    return {"module": module, "entry_point": entry,
            "buffers": [{"array_stride": INST_BYTES, "step_mode": "instance",
                         "attributes": [{"format": "uint32x4", "offset": 0, "shader_location": 0}]}]}


# ============================================================================================ layers
class TerrainLayer(Layer):
    name = "terrain"
    title = "Terrain"
    has_ui = True

    def __init__(self):
        self.lod_distance = 1100.0
        self.relief = 1.6
        self.pipeline = None
        self.sky_pipeline = None
        self.mesh: TerrainMesh | None = None
        self.show_sky = True
        self._basemap = None

    def init(self, app):
        dev = app.ctx.device
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout])
        sky = dev.create_shader_module(label="sky", code=SKY_WGSL)
        # optional OSM drape: group 1 = app.basemap (BasemapLayer is registered before this layer); without it a stub
        bmh = getattr(app, "basemap", None)
        self._basemap = bmh if (bmh is not None and getattr(bmh, "bind_group_layout", None) is not None) else None
        if self._basemap is not None:
            from .basemap import basemap_wgsl
            bm_code = basemap_wgsl(1)
            ter_pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, self._basemap.bind_group_layout])
        else:
            bm_code = BASEMAP_OFF_WGSL
            ter_pl = pl
        ter = dev.create_shader_module(label="terrain", code=bm_code + TERRAIN_WGSL)
        ms = {"count": app.sample_count}
        self.sky_pipeline = dev.create_render_pipeline(
            label="sky", layout=pl, vertex={"module": sky, "entry_point": "vs_sky"},
            primitive={"topology": "triangle-list"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "always"},
            multisample=ms, fragment={"module": sky, "entry_point": "fs_sky", "targets": [{"format": app.colour_format}]})
        self.pipeline = dev.create_render_pipeline(
            label="terrain", layout=ter_pl, vertex=_vertex_state(ter, "vs_terrain"),
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": True, "depth_compare": "greater"},
            multisample=ms, fragment={"module": ter, "entry_point": "fs_terrain", "targets": [{"format": app.colour_format}]})
        self.mesh = TerrainMesh(app)
        app.terrain_mesh = self.mesh

    def on_storm(self, app):
        pass                                  # same grid -> same mesh (the app rebuilds layers when the grid changes)

    def update(self, app, dt):
        m = self.mesh
        app.relief = self.relief
        water_head = 30.0
        wet = app.flow.chunk_wet(*self._pair(app))
        extra = getattr(app, "compare_wet", None)             # ANALYSIS hook: chunks that are wet in the compare storm B too
        if extra is not None and extra.shape == wet.shape:
            wet = wet | extra
        m.update(app.camera.eye(), app.view_proj, app.exag, self.lod_distance * getattr(app, "gov_lod", 1.0), wet, water_head,
                 planes_for(app))
        if self._basemap is not None:
            self._basemap.write_params(app.ctx.queue)     # opacity -> 0 when the (disabled) basemap layer is off; cached, ~free

    @staticmethod
    def _pair(app):
        h0, h1, mix = hour_pair(app.t)
        return h0, h1, mix

    def draw(self, app, pass_):
        if self.show_sky:
            pass_.set_pipeline(self.sky_pipeline)
            pass_.set_bind_group(0, app.frame.bind_group)
            pass_.draw(3)
        pass_.set_pipeline(self.pipeline)
        pass_.set_bind_group(0, app.frame.bind_group)
        if self._basemap is not None:
            pass_.set_bind_group(1, self._basemap.bind_group)
        self.mesh.draw(pass_, self.mesh.draws_terrain, self.mesh.inst_terrain)

    def ui(self, app):
        from imgui_bundle import imgui
        ch, v = imgui.slider_float("LOD distance (m)", self.lod_distance, 200.0, 3000.0, "%.0f")
        if ch:
            self.lod_distance = v
        ch, v = imgui.slider_float("Hillshade relief", self.relief, 1.0, 4.0, "%.1f")
        if ch:
            self.relief = v
        _, self.show_sky = imgui.checkbox("Sky", self.show_sky)
        m = self.mesh
        imgui.text_disabled(f"chunks {m.n_visible}/{m.nx * m.ny}  tris {m.tris / 1e6:.2f} M  cull {m.cull_ms:.2f} ms")

    def save_state(self):
        return {"lod_distance": self.lod_distance, "sky": self.show_sky, "relief": self.relief}

    def load_state(self, d):
        self.lod_distance = float(d.get("lod_distance", self.lod_distance))
        self.show_sky = bool(d.get("sky", self.show_sky))
        self.relief = float(d.get("relief", self.relief))

    def shutdown(self, app):
        if self.mesh:
            self.mesh.close()


class WaterLayer(Layer):
    """Water surface (see the module docstring).  Options: ripple strength, depth boost (Z scale of the water only),
    opacity, clarity (absorption: how fast the ground disappears with depth), turbidity (silty tint), smooth shoreline
    (sub-texel shore from a bicubic surface reconstruction; off = cheap bilinear threshold)."""
    name = "water"
    title = "Water"
    has_ui = True

    def __init__(self):
        self.pipeline = None
        self.ripple = 1.0
        self.depth_boost = 1.0
        self.opacity = 1.0
        self.absorption = 2.4          # 1/m: opacity 1 - exp(-absorption * depth) of the water body
        self.turbidity = 0.25
        self.smooth_shore = True
        self.params = None
        self._pkey = None

    def init(self, app):
        dev = app.ctx.device
        self._pkey = None
        self.params = dev.create_buffer(label="water-params", size=32,
                                        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        bgl = dev.create_bind_group_layout(label="water-bgl", entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}}])
        self.bg = dev.create_bind_group(label="water-bg", layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": self.params, "offset": 0, "size": 32}}])
        pl = dev.create_pipeline_layout(bind_group_layouts=[app.frame.layout, bgl])
        mod = dev.create_shader_module(label="water", code=WATER_WGSL)
        blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.pipeline = dev.create_render_pipeline(
            label="water", layout=pl, vertex=_vertex_state(mod, "vs_water"),
            primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"},
            depth_stencil={"format": app.depth_format, "depth_write_enabled": False, "depth_compare": "greater"},
            multisample={"count": app.sample_count},
            fragment={"module": mod, "entry_point": "fs_water", "targets": [{"format": app.colour_format, "blend": blend}]})

    def shutdown(self, app):
        try:
            if self.params is not None:
                self.params.destroy()
        except Exception:
            pass

    def update(self, app, dt):
        app.ripple_strength = self.ripple
        app.water_boost = self.depth_boost
        key = (self.opacity, self.absorption, 1.0 if self.smooth_shore else 0.0, self.turbidity)
        if key != self._pkey:                                   # the uniform block only changes with the sliders
            self._pkey = key
            a = np.zeros(8, np.float32)
            a[0:4] = key
            app.ctx.queue.write_buffer(self.params, 0, a)

    def draw(self, app, pass_):
        m = getattr(app, "terrain_mesh", None)
        if m is None or not m.draws_water:
            return
        cmp_hook = getattr(app, "compare_hook", None)         # ANALYSIS hook: storm compare (scissor for the swipe, hidden for the difference)
        if cmp_hook is not None and not cmp_hook.water_begin(app, pass_):
            return
        pass_.set_pipeline(self.pipeline)
        pass_.set_bind_group(0, app.frame.bind_group)
        pass_.set_bind_group(1, self.bg)
        m.draw(pass_, m.draws_water, m.inst_water)
        if cmp_hook is not None:
            cmp_hook.water_end(app, pass_)

    def ui(self, app):
        from imgui_bundle import imgui
        ch, v = imgui.slider_float("Ripples", self.ripple, 0.0, 3.0, "%.2f")
        if ch:
            self.ripple = v
        ch, v = imgui.slider_float("Depth boost", self.depth_boost, 0.5, 6.0, "%.1f")
        if ch:
            self.depth_boost = v
        ch, v = imgui.slider_float("Water opacity", self.opacity, 0.1, 1.0, "%.2f")
        if ch:
            self.opacity = v
        ch, v = imgui.slider_float("Clarity (absorption)", self.absorption, 0.6, 6.0, "%.1f /m")
        if ch:
            self.absorption = v
        ch, v = imgui.slider_float("Turbidity", self.turbidity, 0.0, 1.0, "%.2f")
        if ch:
            self.turbidity = v
        _, self.smooth_shore = imgui.checkbox("Smooth shoreline (bicubic)", self.smooth_shore)
        m = getattr(app, "terrain_mesh", None)
        if m:
            imgui.text_disabled(f"water chunks {m.n_water}  tris {m.tris_water / 1e6:.2f} M")

    def save_state(self):
        return {"ripple": self.ripple, "depth_boost": self.depth_boost, "opacity": self.opacity,
                "absorption": self.absorption, "turbidity": self.turbidity, "smooth_shore": self.smooth_shore}

    def load_state(self, d):
        self.ripple = float(d.get("ripple", self.ripple))
        self.depth_boost = float(d.get("depth_boost", self.depth_boost))
        self.opacity = float(d.get("opacity", self.opacity))
        self.absorption = float(d.get("absorption", self.absorption))
        self.turbidity = float(d.get("turbidity", self.turbidity))
        self.smooth_shore = bool(d.get("smooth_shore", self.smooth_shore))
