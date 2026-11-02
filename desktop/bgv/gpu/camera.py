"""Orbit camera (pure numpy, float64 maths) with mouse / keyboard control and ground picking against the heightmap.

World frame: +X east, +Y north, +Z up (geo.py).  The camera orbits `target`:
    yaw    compass heading the camera LOOKS toward, degrees (0 = north, 90 = east)
    pitch  degrees the view is tilted DOWN from the horizon (90 = straight down)
    dist   metres from target to eye
Projection: right-handed view space (looks down -Z), REVERSED-Z with an infinite far plane and depth32float:
    clip.z = near, clip.w = -z_view  ->  depth = near / distance  (1 at the near plane, -> 0 far away)
so depth tests use compare 'greater' and the depth buffer clears to 0.

Mouse (viewer conventions): LMB drag = pan (the grabbed ground point follows the cursor), RMB/MMB drag = orbit,
wheel = zoom toward the cursor, WASD/arrows = move, Q/E = rotate, R/F = tilt, +/- = zoom, Shift = faster, Home = reset.
"""
from __future__ import annotations

import math
import time

import numpy as np

DEG = math.pi / 180.0


def perspective_reversed_inf(fov_y_rad: float, aspect: float, near: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y_rad * 0.5)
    m = np.zeros((4, 4), np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 3] = near
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    n = np.linalg.norm(s)
    if n < 1e-9:                                      # looking straight along up: pick an arbitrary right vector
        s = np.array([1.0, 0.0, 0.0])
    else:
        s /= n
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class HeightSampler:
    """Bilinear CPU sampler of the (filled) heightmap in WORLD metres, with vectorised ray marching."""

    def __init__(self, heights: np.ndarray, size_x: float, size_y: float):
        self.h = np.ascontiguousarray(heights, dtype=np.float32)
        self.H, self.W = self.h.shape
        self.size_x, self.size_y = float(size_x), float(size_y)
        self.zmin = float(self.h.min())
        self.zmax = float(self.h.max())

    def sample(self, x, y):
        """Ground height (metres) at world x, y (scalars or arrays); clamped at the grid border."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        fx = (x / self.size_x + 0.5) * self.W - 0.5
        fy = (0.5 - y / self.size_y) * self.H - 0.5
        fx = np.clip(fx, 0.0, self.W - 1.0)
        fy = np.clip(fy, 0.0, self.H - 1.0)
        x0 = np.minimum(np.floor(fx).astype(np.int64), self.W - 2) if self.W > 1 else np.zeros_like(fx, np.int64)
        y0 = np.minimum(np.floor(fy).astype(np.int64), self.H - 2) if self.H > 1 else np.zeros_like(fy, np.int64)
        tx, ty = fx - x0, fy - y0
        h = self.h
        x1 = np.minimum(x0 + 1, self.W - 1)
        y1 = np.minimum(y0 + 1, self.H - 1)
        v = (h[y0, x0] * (1 - tx) * (1 - ty) + h[y0, x1] * tx * (1 - ty)
             + h[y1, x0] * (1 - tx) * ty + h[y1, x1] * tx * ty)
        return v.astype(np.float64) if v.ndim else float(v)

    def inside(self, x, y) -> bool:
        return abs(x) <= self.size_x * 0.5 and abs(y) <= self.size_y * 0.5

    def raycast(self, origin, direction, max_dist: float = 60000.0, exag: float = 1.0):
        """First intersection of a ray with the terrain surface z = exag * h(x, y).  Returns (x, y, z, t) or None.
        Coarse vectorised march (step grows with distance), then bisection."""
        o = np.asarray(origin, np.float64)
        d = np.asarray(direction, np.float64)
        d = d / np.linalg.norm(d)
        # cheap reject: ray pointing up and above the highest ground
        zmax = self.zmax * exag
        if d[2] >= 0 and o[2] > zmax:
            return None
        # clip the march to the part of the ray below zmax (above that nothing can be hit)
        t_start = 0.0
        if o[2] > zmax and d[2] < 0:
            t_start = (o[2] - zmax) / -d[2]
        t_end = max_dist
        if d[2] < 0:
            zmin = self.zmin * exag
            t_end = min(t_end, (o[2] - zmin) / -d[2] + 1.0) if o[2] > zmin else t_end
        if t_end <= t_start:
            return None
        step0 = 0.5 * min(self.size_x / self.W, self.size_y / self.H)
        n = 2048
        # geometric spacing: fine near the start, coarser far away, covering [t_start, t_end]
        grow = (max(t_end - t_start, 1.0) / step0) ** (1.0 / n)
        ts = t_start + step0 * (np.power(grow, np.arange(n + 1)) - 1.0) / max(grow - 1.0, 1e-9)
        ts = ts[ts <= t_end]
        p = o[None, :] + ts[:, None] * d[None, :]
        gz = self.sample(p[:, 0], p[:, 1]) * exag
        below = p[:, 2] <= gz
        # ignore samples outside the grid footprint (the border height is clamped, not real terrain)
        inside = (np.abs(p[:, 0]) <= self.size_x * 0.5) & (np.abs(p[:, 1]) <= self.size_y * 0.5)
        hit = below & inside
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            return None
        i = int(idx[0])
        if i == 0:
            t = float(ts[0])
            return (float(p[0, 0]), float(p[0, 1]), float(p[0, 2]), t)
        lo, hi = float(ts[i - 1]), float(ts[i])
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            q = o + mid * d
            if q[2] <= self.sample(q[0], q[1]) * exag:
                hi = mid
            else:
                lo = mid
        q = o + hi * d
        return (float(q[0]), float(q[1]), float(q[2]), hi)


class OrbitCamera:
    DEFAULT = dict(x=0.0, y=0.0, dist=14000.0, yaw=0.0, pitch=55.0)

    def __init__(self, fov_deg: float = 45.0):
        self.fov = fov_deg * DEG
        self.target = np.zeros(3)
        self.dist = self.DEFAULT["dist"]
        self.yaw = self.DEFAULT["yaw"]          # degrees
        self.pitch = self.DEFAULT["pitch"]      # degrees
        self.width, self.height = 1920, 1080
        self.min_dist, self.max_dist = 20.0, 60000.0
        self.min_pitch, self.max_pitch = 4.0, 89.5
        self.ground: HeightSampler | None = None
        self.exag = 1.0
        self.limit_xy = (12000.0, 12000.0)       # target kept inside +-limit
        self._keys: set[str] = set()
        self._drag = None
        self._t_last = None
        self.moving = False                       # True while input is changing the view (for UI / governor)
        # smooth input (all exponential, integrated analytically over dt -> frame-rate independent, exact rest)
        self.tau_keys = 0.10                      # held keys ease in / out (s)
        self.tau_glide = 0.22                     # pan / orbit glide after a flick (s)
        self.tau_zoom = 0.07                      # wheel zoom easing (s)
        self.smooth = True
        self._kv = np.zeros(5)                    # eased key intent: right, forward, rotate, tilt, zoom
        self._pan_v = np.zeros(2)                 # glide velocity of the target (m/s, world x y)
        self._orb_v = np.zeros(2)                 # glide velocity of yaw / pitch (deg/s)
        self._zoom_pending = 0.0                  # wheel steps still to apply
        self._zoom_px = None
        self.reset()

    # ------------------------------------------------------------------ state
    def set_ground(self, sampler: HeightSampler | None, exag: float = 1.0):
        self.ground = sampler
        self.exag = exag
        if sampler is not None:
            self.limit_xy = (sampler.size_x * 0.5, sampler.size_y * 0.5)
            self.max_dist = max(2000.0, 1.8 * max(sampler.size_x, sampler.size_y))
        self.target[2] = self.ground_z(self.target[0], self.target[1])

    def ground_z(self, x, y) -> float:
        if self.ground is None:
            return 0.0
        return float(self.ground.sample(x, y)) * self.exag

    def stop_motion(self):
        """Kill every glide / eased motion (a flight, a reset or a new press takes over)."""
        self._kv[:] = 0.0
        self._pan_v[:] = 0.0
        self._orb_v[:] = 0.0
        self._zoom_pending = 0.0
        self.moving = False

    def reset(self):
        self.stop_motion()
        d = self.DEFAULT
        self.target[:] = (d["x"], d["y"], 0.0)
        self.dist, self.yaw, self.pitch = d["dist"], d["yaw"], d["pitch"]
        self.target[2] = self.ground_z(d["x"], d["y"])

    def set_view(self, x, y, dist, yaw, pitch):
        self._pan_v[:] = 0.0
        self._orb_v[:] = 0.0
        self._zoom_pending = 0.0
        self.target[0], self.target[1] = x, y
        self.target[2] = self.ground_z(x, y)
        self.dist = float(np.clip(dist, self.min_dist if dist > self.min_dist else 1.0, self.max_dist))
        self.yaw = float(yaw) % 360.0
        self.pitch = float(np.clip(pitch, self.min_pitch, self.max_pitch))

    def set_exag(self, exag: float):
        """Vertical exaggeration changed: keep the target on the (scaled) ground."""
        self.exag = exag
        self.target[2] = self.ground_z(self.target[0], self.target[1])
        self._keep_eye_above_ground()               # a taller terrain must not swallow the eye

    def resize(self, w: int, h: int):
        self.width, self.height = max(float(w), 1.0), max(float(h), 1.0)       # logical px, fractional at 125 / 150 % DPI

    @property
    def aspect(self) -> float:
        return self.width / self.height

    # ------------------------------------------------------------------ geometry
    def forward_dir(self) -> np.ndarray:
        y, p = self.yaw * DEG, self.pitch * DEG
        return np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), -math.sin(p)])

    def eye(self) -> np.ndarray:
        return self.target - self.forward_dir() * self.dist

    @property
    def near(self) -> float:
        return float(np.clip(self.dist * 0.01, 0.3, 40.0))

    def view_matrix(self) -> np.ndarray:
        return look_at(self.eye(), self.target)

    def proj_matrix(self) -> np.ndarray:
        return perspective_reversed_inf(self.fov, self.aspect, self.near)

    def view_proj(self) -> np.ndarray:
        return self.proj_matrix() @ self.view_matrix()

    def ray(self, px: float, py: float):
        """World-space ray (origin, unit direction) through pixel (px, py) (y down, window pixels)."""
        ndc_x = 2.0 * px / self.width - 1.0
        ndc_y = 1.0 - 2.0 * py / self.height
        th = math.tan(self.fov * 0.5)
        v = np.array([ndc_x * th * self.aspect, ndc_y * th, -1.0])
        view = self.view_matrix()
        rot = view[:3, :3]
        d = rot.T @ v
        d /= np.linalg.norm(d)
        return self.eye(), d

    def project(self, p) -> tuple[float, float, float]:
        """World point -> (px, py, depth) in window pixels; depth is the reversed-Z value (0..1)."""
        c = self.view_proj() @ np.array([p[0], p[1], p[2], 1.0])
        if c[3] <= 0:
            return (float("nan"), float("nan"), float("nan"))
        return ((c[0] / c[3] * 0.5 + 0.5) * self.width, (0.5 - c[1] / c[3] * 0.5) * self.height, c[2] / c[3])

    def pick(self, px: float, py: float):
        """Terrain point under the pixel -> (x, y, z, t) or None."""
        o, d = self.ray(px, py)
        if self.ground is not None:
            return self.ground.raycast(o, d, max_dist=self.max_dist * 3, exag=self.exag)
        if d[2] >= 0:
            return None
        t = (self.target[2] - o[2]) / d[2]
        q = o + t * d
        return (float(q[0]), float(q[1]), float(q[2]), float(t))

    # ------------------------------------------------------------------ manipulation
    def _clamp(self):
        lx, ly = self.limit_xy
        self.target[0] = float(np.clip(self.target[0], -lx, lx))
        self.target[1] = float(np.clip(self.target[1], -ly, ly))
        self.dist = float(np.clip(self.dist, self.min_dist, self.max_dist))
        self.pitch = float(np.clip(self.pitch, self.min_pitch, self.max_pitch))
        self.yaw %= 360.0
        self.target[2] = self.ground_z(self.target[0], self.target[1])
        self._keep_eye_above_ground()

    def _keep_eye_above_ground(self, clearance: float = 3.0):
        if self.ground is None:
            return
        e = self.eye()
        if e[2] >= self.ground_z(e[0], e[1]) + clearance:
            return
        # raise the pitch (look more downward) until the eye clears the ground; straight down always works because
        # the eye is then dist metres above the target which sits on the ground
        for _ in range(60):
            if self.pitch >= self.max_pitch:
                break
            self.pitch = min(self.max_pitch, self.pitch + 1.5)
            e = self.eye()
            if e[2] >= self.ground_z(e[0], e[1]) + clearance:
                return
        e = self.eye()
        if e[2] < self.ground_z(e[0], e[1]) + clearance:
            self.dist = float(np.clip(max(self.dist, clearance * 2.0), self.min_dist, self.max_dist))

    def pan_pixels(self, px0, py0, px1, py1, grab_z: float | None = None):
        """Drag the ground point under (px0,py0) to (px1,py1).  The plane through the grabbed point (height `grab_z`,
        default the target height) is used, so the point stays under the cursor on hilly ground too."""
        z = self.target[2] if grab_z is None else grab_z
        a = self._plane_point(px0, py0, z)
        if a is None:
            return
        self._pan_toward((a[0], a[1], z), px1, py1)

    def _pan_toward(self, grab, px, py):
        """Move the target so that the world point `grab` (x, y, z) ends up under pixel (px, py)."""
        for _ in range(3):                           # the eye height follows the terrain under the target: iterate
            b = self._plane_point(px, py, grab[2])
            if b is None or math.hypot(b[0] - self.target[0], b[1] - self.target[1]) > self.max_dist * 1.5:
                return                               # cursor at / above the horizon: ignore
            dx, dy = grab[0] - b[0], grab[1] - b[1]
            if abs(dx) + abs(dy) < 0.01:
                break
            self.target[0] += dx
            self.target[1] += dy
            self._clamp()

    def _plane_point(self, px, py, z):
        o, d = self.ray(px, py)
        if d[2] >= -1e-6:
            return None
        t = (z - o[2]) / d[2]
        return o + t * d

    def orbit(self, dx_px: float, dy_px: float):
        self.yaw += dx_px * 0.25
        self.pitch += dy_px * 0.25
        self._clamp()

    def zoom_at(self, px: float, py: float, steps: float):
        """Wheel zoom toward the pixel: the ground point under the cursor stays under the cursor (as far as the
        orbit model allows).  steps > 0 zooms in."""
        factor = 0.86 ** steps
        pick = self.pick(px, py)
        new_dist = float(np.clip(self.dist * factor, self.min_dist, self.max_dist))
        if pick is not None and abs(new_dist - self.dist) > 1e-6:
            q = np.array(pick[:3])
            # scale the target toward the picked point by the same factor as the distance
            k = new_dist / self.dist
            self.target[:2] = q[:2] + (self.target[:2] - q[:2]) * k
            self.dist = new_dist
            self._clamp()
            self._pan_toward(q, px, py)                  # hilly ground: put the picked point exactly back under the cursor
            return
        self.dist = new_dist
        self._clamp()

    def move_local(self, right: float, forward: float):
        """Move the target in the ground plane relative to the heading (metres)."""
        y = self.yaw * DEG
        fx, fy = math.sin(y), math.cos(y)
        self.target[0] += fx * forward + fy * right
        self.target[1] += fy * forward - fx * right
        self._clamp()

    # ------------------------------------------------------------------ events (rendercanvas event dicts)
    @staticmethod
    def _ev_time(ev) -> float:
        t = ev.get("time_stamp")
        return float(t) if isinstance(t, (int, float)) else time.perf_counter()

    def on_pointer_down(self, ev):
        b = ev.get("button", 1)
        self.stop_motion()
        x, y = ev["x"], ev["y"]
        self._drag = {"button": b, "x": x, "y": y, "t": self._ev_time(ev), "grab": None}
        if b == 1:                                           # LMB: remember the ground point under the cursor
            pk = self.pick(x, y)
            self._drag["grab"] = None if pk is None else (pk[0], pk[1], pk[2])

    def on_pointer_up(self, ev=None):
        d = self._drag
        if d is not None and ev is not None and ev.get("button") not in (None, 0, d["button"]):
            return                                           # another button went up: this drag goes on
        self._drag = None
        if d is None or ev is None or not self.smooth:
            self._pan_v[:] = 0.0
            self._orb_v[:] = 0.0
            return
        if self._ev_time(ev) - d["t"] > 0.08:                 # held still before releasing: no flick
            self._pan_v[:] = 0.0
            self._orb_v[:] = 0.0
        elif d["button"] == 1:
            self._orb_v[:] = 0.0
            if float(np.hypot(*self._pan_v)) < 0.002 * self.dist:
                self._pan_v[:] = 0.0
        else:
            self._pan_v[:] = 0.0
            if float(np.hypot(*self._orb_v)) < 1.0:
                self._orb_v[:] = 0.0
        self.moving = bool(self._pan_v.any() or self._orb_v.any())

    def end_drag(self):
        """Pointer released somewhere we never saw (over a UI window, outside the window, focus lost)."""
        self._drag = None
        self._pan_v[:] = 0.0
        self._orb_v[:] = 0.0

    def on_pointer_move(self, ev):
        d = self._drag
        if d is None:
            return
        buttons = ev.get("buttons")
        if buttons is not None and d["button"] not in buttons:       # the release was lost: stop dragging
            self.end_drag()
            return
        x, y = ev["x"], ev["y"]
        t = self._ev_time(ev)
        dt = max(t - d["t"], 1e-4)
        if d["button"] == 1:
            before = self.target[:2].copy()
            if d["grab"] is not None:
                self._pan_toward(d["grab"], x, y)
            else:
                self.pan_pixels(d["x"], d["y"], x, y)
            if self.smooth:
                v = (self.target[:2] - before) / dt
                lim = 6.0 * self.dist
                n = float(np.hypot(*v))
                if n > lim:
                    v *= lim / n
                self._pan_v[:] = 0.5 * self._pan_v + 0.5 * v
        else:
            before = np.array([self.yaw, self.pitch])
            self.orbit(x - d["x"], y - d["y"])
            if self.smooth:
                dv = np.array([((self.yaw - before[0] + 180.0) % 360.0) - 180.0, self.pitch - before[1]]) / dt
                self._orb_v[:] = 0.5 * self._orb_v + 0.5 * np.clip(dv, -720.0, 720.0)
        d["x"], d["y"], d["t"] = x, y, t

    def on_wheel(self, ev, x=None, y=None):
        dy = ev.get("dy", 0.0)
        steps = -dy / 100.0
        px = ev.get("x", x if x is not None else self.width * 0.5)
        py = ev.get("y", y if y is not None else self.height * 0.5)
        if not self.smooth:
            self.zoom_at(px, py, steps)
            return
        self._pan_v[:] = 0.0
        self._zoom_px = (px, py)
        self._zoom_pending = float(np.clip(self._zoom_pending + steps, -12.0, 12.0))

    def on_key(self, ev):
        k = ev.get("key", "")
        kk = k.lower() if len(k) == 1 else k
        if ev.get("event_type") == "key_down":
            self._keys.add(kk)
            self._pan_v[:] = 0.0
            self._orb_v[:] = 0.0
            if k == "Home":
                self.reset()
        else:
            self._keys.discard(kk)

    def release_keys(self):
        self._keys.clear()

    def _intent(self):
        keys = self._keys
        return np.array([
            (("d" in keys) or ("ArrowRight" in keys)) - (("a" in keys) or ("ArrowLeft" in keys)),
            (("w" in keys) or ("ArrowUp" in keys)) - (("s" in keys) or ("ArrowDown" in keys)),
            ("e" in keys) - ("q" in keys),
            ("f" in keys) - ("r" in keys),
            (("=" in keys) or ("+" in keys)) - ("-" in keys)], np.float64)

    @staticmethod
    def _ease_integral(v0, target, tau: float, dt: float):
        """(mean velocity over dt, velocity at the end) of v' = (target - v) / tau: exact, so any frame split gives
        the same travelled distance."""
        e = math.exp(-dt / tau)
        v1 = target + (v0 - target) * e
        dist = target * dt + (v0 - target) * tau * (1.0 - e)
        return dist / dt, v1

    def update(self, dt: float):
        """Apply held keys, key / pointer glide and wheel easing.  Returns True when the view changed."""
        if dt <= 0.0:
            return False
        changed = False
        # ---- wheel zoom easing (exponential approach: the remaining steps decay by exp(-dt / tau))
        if self._zoom_pending != 0.0:
            p = self._zoom_pending
            step = p * (1.0 - math.exp(-dt / self.tau_zoom))
            if abs(p - step) < 0.01:                            # the last sliver: finish exactly, no creeping
                step = p
            self._zoom_pending = p - step
            px, py = self._zoom_px if self._zoom_px is not None else (self.width * 0.5, self.height * 0.5)
            self.zoom_at(px, py, step)
            changed = True
        # ---- held keys, eased
        fast = 3.0 if ("Shift" in self._keys) else 1.0
        intent = self._intent() * fast
        kv = self._kv
        if intent.any() or kv.any():
            mv = np.zeros(5)
            for i in range(5):
                mv[i], kv[i] = self._ease_integral(kv[i], intent[i], self.tau_keys, dt)
            if not intent.any() and float(np.abs(kv).max()) < 0.01:
                kv[:] = 0.0                                     # at rest: no creeping
            speed = self.dist * 0.6 * dt
            if mv[0] or mv[1]:
                self.move_local(mv[0] * speed, mv[1] * speed)
            if mv[2]:
                self.yaw += mv[2] * 60.0 * dt
            if mv[3]:
                self.pitch += mv[3] * 40.0 * dt
            if mv[4]:
                self.dist *= 0.5 ** (mv[4] * dt)
            self._clamp()
            changed = True
        # ---- glide after a pan / orbit flick (only while no button is held)
        if self._drag is None and (self._pan_v.any() or self._orb_v.any()):
            tau = self.tau_glide
            e = math.exp(-dt / tau)
            if self._pan_v.any():
                self.target[:2] += self._pan_v * tau * (1.0 - e)
                self._pan_v *= e
                if float(np.hypot(*self._pan_v)) < 0.002 * self.dist:
                    self._pan_v[:] = 0.0
            if self._orb_v.any():
                self.yaw += self._orb_v[0] * tau * (1.0 - e)
                self.pitch += self._orb_v[1] * tau * (1.0 - e)
                self._orb_v *= e
                if float(np.hypot(*self._orb_v)) < 1.0:
                    self._orb_v[:] = 0.0
            self._clamp()
            changed = True
        self.moving = bool(changed or self._drag is not None)
        return changed
