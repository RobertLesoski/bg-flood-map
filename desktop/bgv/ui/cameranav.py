"""Smooth camera flights (presets, fly-to-selection).  Pure numpy / python: drive it from a layer's update().

    nav = CameraNav()
    nav.fly(camera, x, y, dist, yaw, pitch, seconds=1.0)
    nav.update(camera, dt) -> True while flying;   nav.cancel()
Presets are given as lng/lat; `preset_view(frame, name)` converts to world metres.
"""
from __future__ import annotations

import math

PRESETS = {
    # name: (lng, lat, distance m, yaw deg, pitch deg)   (None lng = grid centre)
    "Overview": (None, None, 14000.0, 0.0, 55.0),
    "Downtown": (-86.4436, 36.9903, 1700.0, 20.0, 50.0),
    "Barren River": (-86.4305, 37.0040, 1800.0, 330.0, 45.0),
}


def preset_view(frame, name: str):
    lng, lat, dist, yaw, pitch = PRESETS[name]
    if lng is None:
        return 0.0, 0.0, dist, yaw, pitch
    x, y = frame.lnglat_to_world(lng, lat)
    return x, y, dist, yaw, pitch


def _ease(k: float) -> float:
    k = min(max(k, 0.0), 1.0)
    return k * k * (3.0 - 2.0 * k)


def _lerp_angle(a: float, b: float, k: float) -> float:
    d = ((b - a + 180.0) % 360.0) - 180.0
    return (a + d * k) % 360.0


class CameraNav:
    def __init__(self):
        self.active = False
        self._a = None
        self._b = None
        self._k = 0.0
        self._dur = 1.0

    def fly(self, cam, x, y, dist, yaw, pitch, seconds: float = 1.0):
        self._a = (float(cam.target[0]), float(cam.target[1]), float(cam.dist), float(cam.yaw), float(cam.pitch))
        self._b = (float(x), float(y), float(dist), float(yaw) % 360.0, float(pitch))
        self._k = 0.0
        self._dur = max(float(seconds), 1e-3)
        self.active = True

    def cancel(self):
        self.active = False

    def update(self, cam, dt: float) -> bool:
        if not self.active:
            return False
        self._k += dt / self._dur
        e = _ease(self._k)
        ax, ay, ad, ayaw, ap = self._a
        bx, by, bd, byaw, bp = self._b
        # distance interpolated in log space so a long hop zooms out and back in naturally
        dist = math.exp(math.log(ad) + (math.log(bd) - math.log(ad)) * e)
        arc = 0.0
        if self._k < 1.0:
            hop = math.hypot(bx - ax, by - ay)
            arc = math.sin(math.pi * e) * min(hop * 0.25, 2500.0)       # rise during a long flight
        cam.set_view(ax + (bx - ax) * e, ay + (by - ay) * e, dist + arc, _lerp_angle(ayaw, byaw, e), ap + (bp - ap) * e)
        if self._k >= 1.0:
            self.active = False
        return True


def building_view(cx: float, cy: float, height_m: float, cam) -> tuple:
    """Target view for flying to a building: close, keeps the current heading, tilted for a good look."""
    dist = max(90.0, min(220.0, 90.0 + height_m * 4.0))
    return cx, cy, dist, float(cam.yaw), 52.0
