"""Shared helpers of the flow-visualisation layers (particles, arrows, flowpaths): WGSL snippets and CPU maths.

WGSL (`FX_WGSL`, prepend after `COMMON_WGSL`; every name starts with `fx_`/`scr_` so it never clashes with the terrain
noise helpers):
    fx_pcg / fx_rnd              hash RNG (u32 state, float in [0,1))
    screen_to_ground(ndc, range) camera ray through a screen point -> point on the terrain surface (fixed-point iteration on
                                 the height texture, exaggerated Z), `ok` = false for sky / beyond range / outside the grid
    in_view(p, margin)           world point inside the camera frustum (NDC +-margin)
    px_per_m_at(clip_w)          screen pixels per ground metre at a given view depth
    speed_ramp(s)                pale cyan -> white -> warm colour by speed (m/s)
CPU (pure functions, unit tested): px_per_metre, velocity_multiplier, particle_target, QUALITY presets.
"""
from __future__ import annotations

import math

FX_WGSL = """
fn fx_pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn fx_rnd(state: ptr<function, u32>) -> f32 {
  let v = fx_pcg(*state);
  *state = v;
  return f32(v >> 8u) * (1.0 / 16777216.0);
}

struct GroundHit { ok: bool, p: vec3<f32>, t: f32 };

// camera ray through NDC point -> terrain surface point (exaggerated Z).  Fixed-point iteration: intersect the ray with the
// plane at the current ground height, re-read the ground at the hit, repeat.
fn screen_to_ground(ndc: vec2<f32>, max_range: f32) -> GroundHit {
  var h: GroundHit;
  h.ok = false;
  h.p = vec3<f32>(0.0);
  h.t = 0.0;
  let f = frame.inv_view_proj * vec4<f32>(ndc, 0.0001, 1.0);
  let n = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
  let o = frame.eye.xyz;
  let d = normalize(f.xyz / f.w - n.xyz / n.w);
  if (d.z > -0.02) { return h; }
  let E = frame.grid.w;
  var zg = ground_height(o.xy) * E;
  var t = 0.0;
  var p = o;
  for (var i = 0; i < 5; i = i + 1) {
    t = (zg - o.z) / d.z;
    p = o + d * t;
    zg = ground_height(p.xy) * E;
  }
  let err = abs(p.z - zg);
  if (t <= 0.0 || t > max_range || err > 1.0 + 0.004 * t) { return h; }
  if (abs(p.x) > frame.grid.x * 0.5 || abs(p.y) > frame.grid.y * 0.5) { return h; }
  h.ok = true;
  h.p = vec3<f32>(p.xy, zg);
  h.t = t;
  return h;
}

fn in_view(p: vec3<f32>, margin: f32) -> bool {
  let c = frame.view_proj * vec4<f32>(p, 1.0);
  if (c.w < 0.2) { return false; }
  let q = c.xy / c.w;
  return abs(q.x) <= margin && abs(q.y) <= margin;
}

// screen pixels per ground metre for a surface at view depth `w` (clip.w)
fn px_per_m_at(w: f32) -> f32 {
  return frame.view_proj[1][1] * frame.view.y * 0.5 / max(w, 0.01);
}

fn speed_ramp(s: f32) -> vec3<f32> {
  var c = mix(vec3<f32>(0.52, 0.86, 1.0), vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.08, 1.1, s));
  c = mix(c, vec3<f32>(1.0, 0.82, 0.50), smoothstep(1.1, 2.6, s));
  return mix(c, vec3<f32>(1.0, 0.52, 0.32), smoothstep(2.6, 4.0, s));
}
"""

# quality presets shared by the layers: (particle pool, density multiplier, arrow cap, spawn attempts)
QUALITY = {
    "low": {"pool": 90_000, "density": 0.7, "arrows": 12_000, "attempts": 12},
    "medium": {"pool": 160_000, "density": 1.0, "arrows": 20_000, "attempts": 18},
    "high": {"pool": 250_000, "density": 1.0, "arrows": 30_000, "attempts": 24},
}


def quality_name(app) -> str | None:
    """Preset name from an optional app-wide `quality` attribute (str 'low'|'medium'|'high' or int 0|1|2)."""
    q = getattr(app, "quality", None)
    if isinstance(q, str) and q.lower() in QUALITY:
        return q.lower()
    if isinstance(q, (int, float)) and not isinstance(q, bool):
        return ("low", "medium", "high")[max(0, min(2, int(q)))]
    return None


def px_per_metre(dist: float, fov_rad: float, viewport_h: float) -> float:
    """Screen pixels per ground metre at view distance `dist` for a perspective camera."""
    return 0.5 * viewport_h / (math.tan(fov_rad * 0.5) * max(dist, 1e-3))


def velocity_multiplier(ppm: float, anim_speed: float, px_per_ms: float = 24.0, constant_screen_speed: bool = True,
                        max_scale: float = 160.0) -> float:
    """Factor applied to the real water velocity when advecting particles.

    Real flow is slow (0.1 - 2 m/s) and the camera is often kilometres away, so physical speed would look frozen.  With
    `constant_screen_speed` the factor makes 1 m/s of flow move `px_per_ms * anim_speed` pixels per second on screen
    (never below the physical speed x anim_speed, never above `max_scale` x anim_speed).  Relative speeds across the map
    stay correct (the factor is the same everywhere), only the time scale is stretched."""
    anim_speed = max(anim_speed, 0.0)
    if not constant_screen_speed:
        return anim_speed
    m = px_per_ms / max(ppm, 1e-6)
    return anim_speed * min(max(m, 1.0), max_scale)


def particle_target(wet_screen_frac: float, view_px: float, density_per_px: float, cam_dist: float, pool: int,
                    min_if_wet: int = 250) -> float:
    """Number of active particles: `density_per_px` particles per screen pixel of moving wet water, boosted when the
    camera is far (each streak covers fewer pixels there), clamped to the pool."""
    if wet_screen_frac <= 0.0:
        return 0.0
    far = 1.0 + 5.0 * _smoothstep(600.0, 9000.0, cam_dist)
    n = wet_screen_frac * view_px * density_per_px * far
    return float(min(max(n, min_if_wet), pool))


def _smoothstep(a: float, b: float, x: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def far_alpha(cam_dist: float) -> float:
    """Global alpha factor of the particle streaks: full when close, softer when the whole map is in view."""
    return 1.0 - 0.35 * _smoothstep(2500.0, 22000.0, cam_dist)
