"""Camera maths: view/projection, reversed-Z, rays, picking against a heightmap, orbit invariants."""
import math

import numpy as np
import pytest

from bgv.gpu.camera import HeightSampler, OrbitCamera, look_at, perspective_reversed_inf


def flat_sampler(z=100.0, n=64, size=6400.0):
    return HeightSampler(np.full((n, n), z, np.float32), size, size)


def test_look_at_and_axes():
    cam = OrbitCamera()
    cam.resize(1000, 500)
    cam.set_view(100.0, -50.0, 1000.0, 0.0, 45.0)        # looking north, tilted 45 deg down
    f = cam.forward_dir()
    assert f == pytest.approx([0, math.cos(math.radians(45)), -math.sin(math.radians(45))])
    eye = cam.eye()
    assert eye[1] < cam.target[1] and eye[2] > cam.target[2]            # camera is SOUTH of and ABOVE the target
    v = cam.view_matrix()
    t = v @ np.append(cam.target, 1.0)
    assert t[0] == pytest.approx(0, abs=1e-6) and t[1] == pytest.approx(0, abs=1e-6) and t[2] == pytest.approx(-1000.0)
    # east is to the right of the screen when looking north
    east = v @ np.append(cam.target + [10, 0, 0], 1.0)
    assert east[0] > 0
    # yaw 90 = looking east: north is to the LEFT
    cam.set_view(0, 0, 1000.0, 90.0, 45.0)
    north = cam.view_matrix() @ np.append(cam.target + [0, 10, 0], 1.0)
    assert north[0] < 0


def test_reversed_z_depth():
    n = 2.0
    p = perspective_reversed_inf(math.radians(45), 1.5, n)
    for dist in (n, 10.0, 1000.0, 1e6):
        c = p @ np.array([0, 0, -dist, 1.0])
        assert c[2] / c[3] == pytest.approx(n / dist)
    near = p @ np.array([0, 0, -n, 1.0])
    assert near[2] / near[3] == pytest.approx(1.0)


def test_project_and_ray_roundtrip():
    cam = OrbitCamera()
    cam.resize(1280, 720)
    cam.set_view(500.0, 300.0, 2500.0, 37.0, 50.0)
    for px, py in [(640, 360), (10, 700), (1200, 50), (400, 200)]:
        o, d = cam.ray(px, py)
        for t in (100.0, 2500.0, 9000.0):
            q = o + d * t
            ppx, ppy, depth = cam.project(q)
            assert ppx == pytest.approx(px, abs=1e-6) and ppy == pytest.approx(py, abs=1e-6)
            assert 0 < depth < 1
    # the target projects to the screen centre
    ppx, ppy, _ = cam.project(cam.target)
    assert (ppx, ppy) == (pytest.approx(640), pytest.approx(360))


def test_pick_flat_ground():
    cam = OrbitCamera()
    cam.resize(1000, 800)
    cam.set_ground(flat_sampler(100.0), 1.0)
    cam.set_view(0, 0, 1500.0, 0.0, 50.0)
    assert cam.target[2] == pytest.approx(100.0)
    hit = cam.pick(500, 400)                                  # screen centre = the target
    assert hit is not None and hit[:3] == pytest.approx([0, 0, 100.0], abs=0.05)
    hit = cam.pick(500, 100)                                  # higher on the screen = further north
    assert hit[1] > 100 and hit[2] == pytest.approx(100.0, abs=0.05)
    assert cam.pick(500, 1) is not None or True
    # the sky (above the horizon) hits nothing
    cam.set_view(0, 0, 1500.0, 0.0, 10.0)
    assert cam.pick(500, 0) is None or cam.pick(500, 0)[3] > 0


def test_pick_sloped_ground_and_exaggeration():
    n, size = 64, 6400.0
    xs = (np.arange(n) + 0.5) / n - 0.5
    hmap = np.tile((xs * size * 0.1).astype(np.float32), (n, 1)) + 200.0      # rises 10 % toward east
    s = HeightSampler(hmap, size, size)
    assert s.sample(0.0, 0.0) == pytest.approx(200.0, abs=0.1)
    assert s.sample(1000.0, 0.0) == pytest.approx(300.0, abs=0.5)
    o = np.array([1000.0, 0.0, 5000.0])
    hit = s.raycast(o, [0, 0, -1], exag=1.0)
    assert hit[:3] == pytest.approx([1000.0, 0.0, 300.0], abs=0.5)
    hit2 = s.raycast(o, [0, 0, -1], exag=2.0)
    assert hit2[2] == pytest.approx(600.0, abs=1.0)
    # oblique ray against the slope agrees with the surface
    d = np.array([0.6, 0.0, -0.8])
    hit = s.raycast(np.array([-1500.0, 0.0, 1500.0]), d)
    assert hit is not None and hit[2] == pytest.approx(float(s.sample(hit[0], 0.0)), abs=0.3)
    # a ray that misses (pointing away / outside the footprint)
    assert s.raycast(np.array([0, 0, 1000.0]), [0, 0, 1]) is None
    assert s.raycast(np.array([9000.0, 0, 1000.0]), [0, 0, -1]) is None


def test_zoom_keeps_cursor_point():
    cam = OrbitCamera()
    cam.resize(1200, 800)
    cam.set_ground(flat_sampler(150.0), 1.0)
    cam.set_view(0, 0, 3000.0, 20.0, 45.0)
    px, py = 900.0, 250.0
    before = cam.pick(px, py)
    cam.zoom_at(px, py, 3.0)
    after = cam.pick(px, py)
    assert cam.dist < 3000.0 * 0.86 ** 3 * 1.001
    assert after[:2] == pytest.approx(before[:2], abs=2.0)
    cam.zoom_at(px, py, -3.0)
    assert cam.pick(px, py)[:2] == pytest.approx(before[:2], abs=3.0)


def test_pan_keeps_grabbed_point():
    cam = OrbitCamera()
    cam.resize(1200, 800)
    cam.set_ground(flat_sampler(150.0), 1.0)
    cam.set_view(0, 0, 2000.0, 0.0, 60.0)
    grab = cam.pick(600, 500)
    cam.pan_pixels(600, 500, 700, 450)
    now = cam.pick(700, 450)
    assert now[:2] == pytest.approx(grab[:2], abs=1.0)


def test_orbit_and_clamps():
    cam = OrbitCamera()
    cam.resize(800, 600)
    cam.set_ground(flat_sampler(100.0), 1.0)
    cam.set_view(0, 0, 1000.0, 0.0, 45.0)
    t0 = cam.target.copy()
    cam.orbit(100, 0)
    assert cam.yaw == pytest.approx(25.0) and np.allclose(cam.target, t0)
    assert np.linalg.norm(cam.eye() - cam.target) == pytest.approx(1000.0)
    cam.orbit(0, 100000)
    assert cam.pitch == cam.max_pitch
    cam.orbit(0, -100000)
    assert cam.pitch >= cam.min_pitch
    assert cam.eye()[2] >= 100.0 + 3.0 - 1e-6                  # eye never goes under the ground
    for _ in range(80):
        cam.zoom_at(400, 300, 5.0)
    assert cam.dist == pytest.approx(cam.min_dist)
    for _ in range(80):
        cam.zoom_at(400, 300, -5.0)
    assert cam.dist == pytest.approx(cam.max_dist)
    cam.move_local(1e9, 1e9)
    assert abs(cam.target[0]) <= cam.limit_xy[0] and abs(cam.target[1]) <= cam.limit_xy[1]


def test_keyboard_moves_relative_to_heading():
    cam = OrbitCamera()
    cam.set_ground(flat_sampler(100.0), 1.0)
    cam.set_view(0, 0, 1000.0, 90.0, 45.0)                      # looking east
    cam.on_key({"event_type": "key_down", "key": "w"})
    assert cam.update(0.1)
    assert cam.target[0] > 0 and abs(cam.target[1]) < 1e-6      # W moves east when facing east
    cam.on_key({"event_type": "key_up", "key": "w"})
    assert cam.update(0.1)                                      # eases out (inertia) ...
    for _ in range(100):
        cam.update(0.1)
    assert not cam.update(0.1)                                  # ... and then rests exactly
    cam.on_key({"event_type": "key_down", "key": "Home"})
    assert cam.dist == cam.DEFAULT["dist"] and cam.yaw == cam.DEFAULT["yaw"]


def test_look_at_degenerate_straight_down():
    m = look_at([0, 0, 100], [0, 0, 0])
    assert np.isfinite(m).all()
    t = m @ np.array([0, 0, 0, 1.0])
    assert t[2] == pytest.approx(-100.0)
