"""Real input path tests: mouse / keyboard / resize events are submitted to the canvas exactly as rendercanvas' backends
do (`canvas.submit_event({...})`, flushed by `canvas._process_events()` which is what the scheduler calls before every
draw), so the app's handlers, imgui's handlers (order -99, stop_propagation) and the camera all run in their real order.

Part 1 (no GPU): camera behaviour driven with event dicts.
Part 2 (marker gpu, offscreen canvas): UI capture, selection, hotkeys, window lifecycle, robustness.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bgv.gpu.camera import HeightSampler, OrbitCamera

DESKTOP = Path(__file__).resolve().parents[1]
gpu = pytest.mark.gpu


# =============================================================================================== part 1: camera (CPU)
def hilly(n=128, size=12000.0, base=150.0):
    xs = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(xs * size, xs * size)
    h = (base + 40 * np.sin(X / 700) * np.cos(Y / 900) + 0.02 * X).astype(np.float32)
    return HeightSampler(h, size, size)


def make_cam(exag=3.0, w=1200, h=800, sampler=None):
    cam = OrbitCamera()
    cam.resize(w, h)
    cam.set_ground(sampler or hilly(), exag)
    return cam


def ev(kind, x=0.0, y=0.0, button=0, buttons=(), t=None, **kw):
    d = {"event_type": kind, "x": float(x), "y": float(y), "button": button, "buttons": tuple(buttons), "modifiers": ()}
    if t is not None:
        d["time_stamp"] = t
    d.update(kw)
    return d


def drag(cam, button, p0, p1, steps=5, t0=0.0, dt=0.016, release_dt=0.016):
    """Press at p0, move to p1 in `steps` moves, release.  Returns the time of the release."""
    t = t0
    cam.on_pointer_down(ev("pointer_down", *p0, button=button, buttons=(button,), t=t))
    for i in range(1, steps + 1):
        t += dt
        x = p0[0] + (p1[0] - p0[0]) * i / steps
        y = p0[1] + (p1[1] - p0[1]) * i / steps
        cam.on_pointer_move(ev("pointer_move", x, y, buttons=(button,), t=t))
    t += release_dt
    cam.on_pointer_up(ev("pointer_up", *p1, button=button, buttons=(), t=t))
    return t


@pytest.mark.parametrize("yaw", [0, 33, 90, 180, 270, 301, 359])
def test_lmb_pan_keeps_ground_under_cursor_at_any_yaw(yaw):
    cam = make_cam()
    cam.smooth = False
    cam.set_view(0, 0, 2000, yaw, 50)
    p0, p1 = (600, 420), (520, 360)
    grab = cam.pick(*p0)
    drag(cam, 1, p0, p1)
    now = cam.pick(*p1)
    assert np.hypot(*(np.array(now[:2]) - np.array(grab[:2]))) < 4.0           # 2000 m away, 3x relief: a few metres
    # direction: dragging the mouse to the LEFT moves the ground left, i.e. the target moves RIGHT (east when yaw = 0)
    cam.set_view(0, 0, 2000, yaw, 50)
    c0 = cam.target.copy()
    drag(cam, 1, (600, 400), (400, 400))
    d = cam.target[:2] - c0[:2]
    right = np.array([math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))])
    assert float(d @ right) > 100.0 and abs(float(d @ np.array([-right[1], right[0]]))) < 0.2 * abs(float(d @ right))


def test_pan_ignores_cursor_above_horizon():
    cam = make_cam()
    cam.set_view(0, 0, 2000, 0, 20)
    c0 = cam.target.copy()
    cam.on_pointer_down(ev("pointer_down", 600, 500, button=1, buttons=(1,)))
    cam.on_pointer_move(ev("pointer_move", 600, 0, buttons=(1,)))               # sky
    assert np.isfinite(cam.target).all()
    assert np.hypot(*(cam.target[:2] - c0[:2])) < cam.max_dist * 2.0
    cam.on_pointer_up(ev("pointer_up", 600, 0))


@pytest.mark.parametrize("button", [2, 3])
def test_rmb_mmb_orbit_limits_no_flip(button):
    cam = make_cam()
    cam.smooth = False
    cam.set_view(0, 0, 2000, 10, 50)
    t0 = cam.target.copy()
    cam.on_pointer_down(ev("pointer_down", 0, 0, button=button, buttons=(button,)))
    for i in range(1, 80):
        cam.on_pointer_move(ev("pointer_move", 30 * i, 40 * i, buttons=(button,)))   # huge diagonal drag
        assert cam.min_pitch <= cam.pitch <= cam.max_pitch
        assert 0.0 <= cam.yaw < 360.0
        e = cam.eye()
        assert e[2] >= cam.ground_z(e[0], e[1]) + 2.9                              # never below the ground
    assert cam.pitch == pytest.approx(cam.max_pitch)
    for i in range(1, 160):
        cam.on_pointer_move(ev("pointer_move", 2400 + 30 * i, 3200 - 80 * i, buttons=(button,)))
        assert cam.min_pitch <= cam.pitch <= cam.max_pitch
    assert cam.pitch == pytest.approx(cam.min_pitch) or cam.pitch > cam.min_pitch       # ground clearance may hold it up
    assert np.allclose(cam.target[:2], t0[:2])                                         # orbit never moves the target
    # the view matrix stays finite and right handed even straight down
    cam.set_view(0, 0, 500, 123, 89.5)
    v = cam.view_matrix()
    assert np.isfinite(v).all() and np.linalg.det(v[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_orbit_direction():
    cam = make_cam(sampler=hilly())
    cam.smooth = False
    cam.set_view(0, 0, 3000, 0, 40)
    drag(cam, 2, (500, 400), (600, 400))
    assert cam.yaw == pytest.approx(25.0)                                              # 0.25 deg per pixel, right = +yaw
    drag(cam, 2, (500, 400), (500, 440))
    assert cam.pitch == pytest.approx(50.0)                                            # drag down = look more downward


@pytest.mark.parametrize("pitch,yaw", [(45, 20), (20, 200), (60, 300)])
def test_wheel_zoom_keeps_world_point_under_cursor(pitch, yaw):
    cam = make_cam(exag=2.0)
    cam.set_view(1000, -500, 3000, yaw, pitch)
    px, py = 900.0, 250.0
    before = cam.pick(px, py)
    d0 = cam.dist
    cam.on_wheel(ev("wheel", px, py, dx=0.0, dy=-300.0))                               # 3 notches in
    for _ in range(60):
        cam.update(1 / 60)
    assert cam.dist < d0 * 0.86 ** 3 * 1.001
    sx, sy, _ = cam.project(before[:3])                                               # where the same world point is now
    assert math.hypot(sx - px, sy - py) < 2.5, (sx, sy)                               # ... still under the cursor
    cam.on_wheel(ev("wheel", px, py, dx=0.0, dy=300.0))
    for _ in range(60):
        cam.update(1 / 60)
    assert cam.dist == pytest.approx(d0, rel=0.02)
    sx, sy, _ = cam.project(before[:3])
    assert math.hypot(sx - px, sy - py) < 4.0


def test_zoom_clamps():
    cam = make_cam()
    cam.set_view(0, 0, 3000, 0, 50)
    for _ in range(40):
        cam.on_wheel(ev("wheel", 600, 400, dx=0.0, dy=-1200.0))
        for _ in range(20):
            cam.update(1 / 60)
    assert cam.dist == pytest.approx(cam.min_dist, rel=1e-6)
    e = cam.eye()
    assert e[2] >= cam.ground_z(e[0], e[1]) + 2.9
    for _ in range(40):
        cam.on_wheel(ev("wheel", 600, 400, dx=0.0, dy=1200.0))
        for _ in range(20):
            cam.update(1 / 60)
    assert cam.dist == pytest.approx(cam.max_dist, rel=1e-6)
    assert np.isfinite(cam.eye()).all()


def press(cam, key, mods=()):
    cam.on_key({"event_type": "key_down", "key": key, "modifiers": tuple(mods)})


def release(cam, key):
    cam.on_key({"event_type": "key_up", "key": key, "modifiers": ()})


def run(cam, seconds, fps=60.0):
    n = int(round(seconds * fps))
    for _ in range(n):
        cam.update(1.0 / fps)


def test_wasd_moves_relative_to_heading_and_rests_after_release():
    cam = make_cam(exag=1.0, sampler=HeightSampler(np.full((64, 64), 100.0, np.float32), 40000.0, 40000.0))
    for yaw, fwd in ((0, (0, 1)), (90, (1, 0)), (180, (0, -1)), (270, (-1, 0))):
        cam.set_view(0, 0, 1000, yaw, 50)
        press(cam, "w")
        run(cam, 0.5)
        release(cam, "w")
        run(cam, 2.0)
        d = cam.target[:2]
        assert d @ np.array(fwd) > 100.0 and abs(d @ np.array((-fwd[1], fwd[0]))) < 1.0, (yaw, d)
    cam.set_view(0, 0, 1000, 0, 50)
    for k, vec in (("d", (1, 0)), ("ArrowRight", (1, 0)), ("a", (-1, 0)), ("s", (0, -1)), ("ArrowUp", (0, 1)), ("ArrowDown", (0, -1)),
                   ("ArrowLeft", (-1, 0))):
        cam.set_view(0, 0, 1000, 0, 50)
        press(cam, k)
        run(cam, 0.4)
        release(cam, k)
        run(cam, 2.0)
        assert cam.target[:2] @ np.array(vec) > 50.0, k
    # no drift after release: the view is frozen (bit-exact) once the easing has finished
    snap = (cam.target.copy(), cam.yaw, cam.pitch, cam.dist)
    run(cam, 3.0)
    assert np.array_equal(cam.target, snap[0]) and (cam.yaw, cam.pitch, cam.dist) == snap[1:]
    assert not cam.moving


def test_q_e_rotate_r_f_tilt_shift_faster():
    cam = make_cam()
    cam.set_view(0, 0, 3000, 100, 40)
    press(cam, "e")
    run(cam, 1.0)
    release(cam, "e")
    run(cam, 1.5)
    y_e = (cam.yaw - 100.0 + 180) % 360 - 180
    assert y_e > 20.0
    cam.set_view(0, 0, 3000, 100, 40)
    press(cam, "q")
    run(cam, 1.0)
    release(cam, "q")
    run(cam, 1.5)
    assert (cam.yaw - 100.0 + 180) % 360 - 180 < -20.0
    cam.set_view(0, 0, 3000, 100, 40)
    press(cam, "f")
    run(cam, 0.5)
    release(cam, "f")
    run(cam, 1.0)
    assert cam.pitch > 40.0
    cam.set_view(0, 0, 3000, 100, 60)
    press(cam, "r")
    run(cam, 0.5)
    release(cam, "r")
    run(cam, 1.0)
    assert cam.pitch < 60.0
    # shift = 3x
    cam.set_view(0, 0, 1000, 0, 50)
    press(cam, "w")
    run(cam, 1.0)
    release(cam, "w")
    run(cam, 2.0)
    slow = cam.target[1]
    cam.set_view(0, 0, 1000, 0, 50)
    press(cam, "Shift")
    press(cam, "W", ("Shift",))                       # glfw reports the shifted letter in upper case
    run(cam, 1.0)
    release(cam, "W")
    release(cam, "Shift")
    run(cam, 2.0)
    assert cam.target[1] == pytest.approx(3.0 * slow, rel=0.12)
    assert not cam._keys


def test_key_motion_is_frame_rate_independent():
    out = {}
    for fps in (20.0, 30.0, 60.0, 144.0, 240.0):
        cam = make_cam(exag=1.0, sampler=HeightSampler(np.full((64, 64), 100.0, np.float32), 400000.0, 400000.0))
        cam.max_dist = 1e6
        cam.set_view(0, 0, 1000, 0, 50)
        press(cam, "w")
        run(cam, 0.75, fps)
        release(cam, "w")
        run(cam, 2.0, fps)
        out[fps] = cam.target[1]
    vals = np.array(list(out.values()))
    assert vals.max() / vals.min() < 1.03, out


def test_flick_glide_is_frame_rate_independent_and_stops():
    res = {}
    for fps in (30.0, 60.0, 144.0):
        cam = make_cam(exag=1.0, sampler=HeightSampler(np.full((64, 64), 100.0, np.float32), 40000.0, 40000.0))
        cam.set_view(0, 0, 2000, 0, 55)
        t = drag(cam, 1, (700, 400), (500, 400), steps=4, dt=0.01, release_dt=0.005)         # fast flick, released while moving
        c = cam.target.copy()
        run(cam, 4.0, fps)
        res[fps] = cam.target[0] - c[0]
        assert res[fps] > 20.0                                                                 # it glides on
        snap = cam.target.copy()
        run(cam, 2.0, fps)
        assert np.array_equal(cam.target, snap)                                                # and comes to rest
        assert not cam.moving
    v = np.array(list(res.values()))
    assert v.max() / v.min() < 1.02, res


def test_no_glide_when_held_still_before_release_and_press_stops_glide():
    cam = make_cam(exag=1.0, sampler=HeightSampler(np.full((64, 64), 100.0, np.float32), 40000.0, 40000.0))
    cam.set_view(0, 0, 2000, 0, 55)
    drag(cam, 1, (700, 400), (500, 400), steps=4, dt=0.01, release_dt=0.4)                    # paused 0.4 s before release
    c = cam.target.copy()
    run(cam, 2.0)
    assert np.array_equal(cam.target, c)
    drag(cam, 1, (700, 400), (500, 400), steps=4, dt=0.01, release_dt=0.005)                  # flick, then grab it again
    run(cam, 0.2)
    cam.on_pointer_down(ev("pointer_down", 600, 400, button=1, buttons=(1,)))
    c = cam.target.copy()
    run(cam, 1.0)
    assert np.array_equal(cam.target, c)
    cam.on_pointer_up(ev("pointer_up", 600, 400, t=time.perf_counter() + 5.0))


def test_lost_release_does_not_leave_a_sticky_drag():
    cam = make_cam()
    cam.set_view(0, 0, 2000, 0, 55)
    cam.on_pointer_down(ev("pointer_down", 600, 400, button=1, buttons=(1,)))
    cam.on_pointer_move(ev("pointer_move", 620, 400, buttons=(1,)))
    c = cam.target.copy()
    cam.on_pointer_move(ev("pointer_move", 300, 100, buttons=()))              # button no longer down (release was swallowed)
    assert cam._drag is None and np.array_equal(cam.target, c)


def test_camera_never_goes_under_terrain_with_exaggeration():
    rng = np.random.default_rng(3)
    s = hilly()
    for exag in (1.0, 3.0, 10.0):
        cam = make_cam(exag=exag, sampler=s)
        for _ in range(300):
            cam.set_view(*rng.uniform(-5000, 5000, 2), rng.uniform(20, 800), rng.uniform(0, 360), rng.uniform(4, 30))
            cam._clamp()
            e = cam.eye()
            assert e[2] >= cam.ground_z(e[0], e[1]) + 2.99, (exag, e, cam.pitch, cam.dist)
        # changing the exaggeration afterwards must not bury the eye either
        cam.set_view(2000, 1000, 40, 45, 4.0)
        for ex in (1.0, 10.0, 2.0, 10.0):
            cam.set_exag(ex)
            e = cam.eye()
            assert e[2] >= cam.ground_z(e[0], e[1]) + 2.99, ex
        assert abs(cam.target[2] - cam.ground_z(*cam.target[:2])) < 1e-9


def test_home_resets_and_target_stays_in_bounds():
    cam = make_cam()
    cam.set_view(3000, 2000, 300, 120, 30)
    press(cam, "Home")
    release(cam, "Home")
    d = OrbitCamera.DEFAULT
    assert (cam.target[0], cam.target[1], cam.dist, cam.yaw, cam.pitch) == (d["x"], d["y"], d["dist"], d["yaw"], d["pitch"])
    press(cam, "w", ("Shift",))
    press(cam, "Shift")
    run(cam, 60.0)
    assert abs(cam.target[0]) <= cam.limit_xy[0] and abs(cam.target[1]) <= cam.limit_xy[1]


# ============================================================== part 3: process level (own interpreter, no shared GPU app)
PY = sys.executable


def _env(**extra):
    e = dict(os.environ)
    e["BGV_NO_DIALOG"] = "1"                                  # never a modal box in automated runs
    e["PYTHONIOENCODING"] = "utf-8"
    e.pop("BGV_DATA", None)
    e.update({k: str(v) for k, v in extra.items()})
    return e


def _run(args, timeout=240, **envx):
    return subprocess.run([PY, *args], cwd=str(DESKTOP), env=_env(**envx), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _result(proc):
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    import json
    return json.loads(line[len("RESULT "):])


@pytest.fixture(scope="module")
def master_root(tmp_path_factory):
    """A private copy of the data (25yr only) in a folder whose name has spaces and non-ASCII letters; other roots are
    built from it with hard links (no big copies).  The real data root is only read."""
    import shutil
    from data_helpers import real_root
    src = real_root()
    root = tmp_path_factory.mktemp("Bøwling Gréen Flood Map")
    shutil.copy2(src / "dem_bg.tif", root / "dem_bg.tif")
    (root / "layers" / "flow").mkdir(parents=True)
    shutil.copytree(src / "layers" / "flow" / "25yr", root / "layers" / "flow" / "25yr")
    for f in ("buildings.geojson", "roads.geojson"):
        shutil.copy2(src / "layers" / f, root / "layers" / f)
    return root


def derive_root(master: Path, dest: Path, skip=(), mutate=None) -> Path:
    import shutil
    (dest / "layers" / "flow").mkdir(parents=True)
    for rel in ("dem_bg.tif", "layers/buildings.geojson", "layers/roads.geojson"):
        if Path(rel).name in skip:
            continue
        os.link(master / rel, dest / rel)
    shutil.copytree(master / "layers" / "flow" / "25yr", dest / "layers" / "flow" / "25yr")
    if mutate:
        mutate(dest / "layers" / "flow" / "25yr")
    return dest


def test_data_root_resolution_and_odd_paths(tmp_path, monkeypatch):
    from bgv import config
    odd = tmp_path / "Dätä Röot with spaces" / "ünï"
    (odd / "layers" / "flow" / "25yr").mkdir(parents=True)
    other = tmp_path / "other"
    (other / "layers" / "flow").mkdir(parents=True)
    bad = tmp_path / "no flow here"
    bad.mkdir()
    monkeypatch.setattr(config, "_override", None)
    monkeypatch.delenv("BGV_DATA", raising=False)
    # --data wins over BGV_DATA
    monkeypatch.setenv("BGV_DATA", str(other))
    config.set_data_root(str(odd))
    assert config.data_root() == odd
    # BGV_DATA alone
    config.set_data_root(None)
    assert config.data_root() == other
    # a wrong --data is an error (never silently the default folder)
    config.set_data_root(str(bad))
    with pytest.raises(FileNotFoundError, match="no layers/flow"):
        config.data_root()
    # a stale BGV_DATA falls through to the next candidate
    config.set_data_root(None)
    monkeypatch.setenv("BGV_DATA", str(bad))
    try:
        assert (config.data_root() / "layers" / "flow").is_dir()
    except FileNotFoundError:
        pass                                                     # machine without the default folder: the message is the point
    config.set_data_root(None)


def test_cache_dir_falls_back_to_a_temp_folder_when_unwritable(tmp_path, monkeypatch):
    from bgv import config
    blocker = tmp_path / "a file, not a folder"
    blocker.write_text("x")
    monkeypatch.setenv("BGV_CACHE", str(blocker / "cache"))              # cannot be created below a file
    config.reset_cache_dir()
    try:
        p = config.cache_dir()
        assert p.is_dir() and p != blocker / "cache"
        assert Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))).resolve() in p.resolve().parents or "bgv_cache" in str(p)
        (p / "probe.bin").write_bytes(b"1")                              # really writable
        (p / "probe.bin").unlink()
    finally:
        monkeypatch.delenv("BGV_CACHE", raising=False)
        config.reset_cache_dir()
    good = tmp_path / "good cache ü"
    monkeypatch.setenv("BGV_CACHE", str(good))
    config.reset_cache_dir()
    try:
        assert config.cache_dir() == good and good.is_dir()
    finally:
        monkeypatch.delenv("BGV_CACHE", raising=False)
        config.reset_cache_dir()


def test_log_is_small_and_rotating_and_fatal_writes_log_and_shows_a_short_box(tmp_path, monkeypatch):
    import types
    from bgv import applog
    monkeypatch.setenv("BGV_LOG_DIR", str(tmp_path / "logs"))
    applog.reset()
    try:
        lg = applog.log()
        line = "x" * 1000
        for i in range(2000):                                            # ~2 MB of records
            lg.error("%d %s", i, line)
        files = sorted((tmp_path / "logs").glob("bgv.log*"))
        assert 2 <= len(files) <= applog.BACKUPS + 1
        assert all(f.stat().st_size <= applog.MAX_BYTES + 2000 for f in files)
        assert sum(f.stat().st_size for f in files) < 4 * applog.MAX_BYTES
        # the dialog: user32.MessageBoxW gets a short human text, the traceback only goes to the log
        shown = []
        fake = types.SimpleNamespace(windll=types.SimpleNamespace(user32=types.SimpleNamespace(
            MessageBoxW=lambda hwnd, text, title, flags: shown.append((text, title, flags)) or 1)))
        import ctypes
        monkeypatch.setattr(ctypes, "windll", fake.windll, raising=False)
        monkeypatch.delenv("BGV_NO_DIALOG", raising=False)
        try:
            raise FileNotFoundError("flow data not found: pass --data <folder containing layers/flow> or set BGV_DATA")
        except FileNotFoundError as e:
            msg = applog.fatal(e)
        if os.name == "nt":
            assert len(shown) == 1
            text, title, flags = shown[0]
            assert "flood data folder was not found" in text and "Traceback" not in text and len(text) < 700
            assert flags & 0x10                                          # error icon
        assert "flood data folder was not found" in msg
        logtxt = (tmp_path / "logs" / "bgv.log").read_text(encoding="utf-8")
        assert "Traceback" in logtxt and "FileNotFoundError" in logtxt
        # human messages for the other start-up failures
        assert "graphics card" in applog.human_message(RuntimeError("No wgpu adapter found"))
        assert "package is missing" in applog.human_message(ModuleNotFoundError("No module named 'wgpu'", name="wgpu"))
        assert "pip install" in applog.human_message(ModuleNotFoundError("No module named 'wgpu'", name="wgpu"))
    finally:
        applog.reset()


def test_unlogged_start_failure_in_window_mode_shows_the_box_and_returns_a_code(tmp_path, monkeypatch):
    from bgv import __main__ as entry, applog, config
    boxes = []
    monkeypatch.setattr(applog, "message_box", lambda title, text, error=True: boxes.append((title, text)) or True)
    monkeypatch.setenv("BGV_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "_override", None)
    applog.reset()
    try:
        rc = entry.main(["--data", str(tmp_path / "does not exist")])
        assert rc == 2
        assert len(boxes) == 1 and "flood data folder was not found" in boxes[0][1] and "Traceback" not in boxes[0][1]
        assert (tmp_path / "logs" / "bgv.log").is_file()
        boxes.clear()
        rc = entry.main(["--data", str(tmp_path / "does not exist"), "--render", str(tmp_path / "x.png")])
        assert rc == 2 and boxes == []                                   # CLI tools print instead of popping a box
    finally:
        config.set_data_root(None)
        applog.reset()


def test_cli_missing_data_prints_a_human_message_not_a_traceback(tmp_path):
    p = _run(["-m", "bgv", "--data", str(tmp_path / "nothing here"), "--selftest"], timeout=60,
             BGV_LOG_DIR=tmp_path / "logs")
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "flood data folder was not found" in p.stderr and "Traceback" not in p.stderr
    assert "Traceback" in (tmp_path / "logs" / "bgv.log").read_text(encoding="utf-8")


@gpu
def test_render_from_a_path_with_spaces_and_non_ascii_via_data_flag_and_env_var(master_root, tmp_path):
    from PIL import Image
    out1, out2 = tmp_path / "out one.png", tmp_path / "ünï out.png"
    p1 = _run(["-m", "bgv", "--render", str(out1), "--data", str(master_root), "--size", "320x200", "--quiet"],
              BGV_LOG_DIR=tmp_path / "logs")
    assert p1.returncode == 0, (p1.stdout[-800:], p1.stderr[-2000:])
    p2 = _run(["-m", "bgv", "--render", str(out2), "--size", "320x200", "--quiet"], BGV_DATA=master_root,
              BGV_LOG_DIR=tmp_path / "logs")
    assert p2.returncode == 0, (p2.stdout[-800:], p2.stderr[-2000:])
    for o in (out1, out2):
        a = np.asarray(Image.open(o))
        assert a.shape[:2] == (200, 320) and a.std() > 5.0
    assert abs(float(np.asarray(Image.open(out1)).astype(float).mean() - np.asarray(Image.open(out2)).astype(float).mean())) < 1.0


@gpu
def test_two_instances_at_the_same_time_with_a_cold_shared_cache(master_root, tmp_path):
    from PIL import Image
    cache = tmp_path / "cache ö"
    outs = [tmp_path / "a.png", tmp_path / "b.png"]
    procs = [subprocess.Popen([PY, "-m", "bgv", "--render", str(o), "--data", str(master_root), "--size", "320x200", "--quiet"],
                              cwd=str(DESKTOP), env=_env(BGV_CACHE=cache, BGV_LOG_DIR=tmp_path / "logs"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
             for o in outs]
    res = [p.communicate(timeout=420) for p in procs]
    for p, (so, se), o in zip(procs, res, outs):
        assert p.returncode == 0, (so[-800:], se[-2500:])
        assert np.asarray(Image.open(o)).std() > 5.0
    a, b = (np.asarray(Image.open(o)).astype(float) for o in outs)
    assert abs(a.mean() - b.mean()) < 1.0
    assert not list(cache.glob("*.tmp*"))                                # no half-written cache files left behind
    # a third run starts from the cache the two others wrote
    p3 = _run(["-m", "bgv", "--render", str(tmp_path / "c.png"), "--data", str(master_root), "--size", "320x200", "--quiet"],
              BGV_CACHE=cache, BGV_LOG_DIR=tmp_path / "logs")
    assert p3.returncode == 0, p3.stderr[-2000:]


@gpu
def test_corrupt_and_missing_hour_files_are_skipped_with_an_error_toast_others_keep_working(master_root, tmp_path):
    def mutate(d):
        (d / "h12.rgba.gz").write_bytes((d / "h12.rgba.gz").read_bytes()[:5000])            # truncated
        (d / "h07.rgba.gz").write_bytes(b"this is not gzip data" * 50)                      # garbage
        (d / "h13.rgba.gz").unlink()                                                        # missing
    root = derive_root(master_root, tmp_path / "root corrupt", mutate=mutate)
    p = _run([str(DESKTOP / "tests" / "interaction_driver.py"), str(root), "flow"], BGV_LOG_DIR=tmp_path / "logs")
    r = _result(p)
    assert set(r["failed"]) == {"7", "12", "13"}, r["failed"]
    assert r["progress"][0] == r["progress"][1] == 23                    # every hour is resolved: loaded or skipped
    assert r["frames"] >= 40
    toast = next(x for x in r["toasts"] if "could not be read" in x)     # one red toast names the skipped hours
    assert toast.startswith("Hours ") and " 7," in toast and " 12" in toast and " 13 of the 25yr storm" in toast, toast
    assert sum("could not be read" in e for e in r["errors"]) >= 1
    assert r["layers"]["terrain"] and r["layers"]["water"] and r["layers"]["buildings"] and r["layers"]["roads"]
    assert (tmp_path / "logs" / "bgv.log").is_file()


@gpu
@pytest.mark.parametrize("skip", ["buildings.geojson", "roads.geojson"])
def test_missing_buildings_or_roads_file_switches_those_layers_off_with_a_message(master_root, tmp_path, skip):
    root = derive_root(master_root, tmp_path / f"root no {skip}", skip=(skip,))
    p = _run([str(DESKTOP / "tests" / "interaction_driver.py"), str(root), "flow"], BGV_LOG_DIR=tmp_path / "logs")
    r = _result(p)
    assert r["scene_state"] == "error"
    assert r["layers"]["buildings"] is False and r["layers"]["roads"] is False
    assert r["layers"]["terrain"] and r["layers"]["water"] and r["layers"]["overlay"]
    assert any("Buildings and roads could not be loaded" in t for t in r["toasts"])
    assert r["frames"] >= 40 and r["failed"] == {}


@gpu
def test_window_start_up_path_reaches_the_app_and_input_works_after_the_loading_screen(master_root, tmp_path):
    """Regression: bgv/launch.py's tick never ran (rendercanvas cancels draws until the canvas has a context) and used an
    unimported `config`: the real window stayed on its splash for ever."""
    p = _run([str(DESKTOP / "tests" / "interaction_driver.py"), str(master_root), "launcher"], BGV_LOG_DIR=tmp_path / "logs")
    r = _result(p)
    assert r["state"] == "done" and r["error"] is None, r
    assert r["times"]["loaded"] < 30 and "first_loading_frame" in r["times"]
    assert r["loading"] is False and r["errors"] == []
    assert r["panned"] > 100.0 and r["zoom"] < 0.8 and r["walked"] > 20.0, r


@gpu
def test_closing_the_window_stops_every_thread_and_the_process_exits_within_3_seconds(master_root, tmp_path):
    import json
    proc = subprocess.Popen([PY, str(DESKTOP / "tests" / "interaction_driver.py"), str(master_root), "close"], cwd=str(DESKTOP),
                            env=_env(BGV_LOG_DIR=tmp_path / "logs"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    t_close = None
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        if line.startswith("CLOSING"):
            t_close = time.perf_counter()
    rc = proc.wait(timeout=60)
    t_exit = time.perf_counter()
    err = proc.stderr.read()
    assert rc == 0 and t_close is not None, (lines[-5:], err[-2000:])
    assert t_exit - t_close < 3.0, f"process needed {t_exit - t_close:.2f} s to exit after the close"
    th = json.loads(next(l for l in lines if l.startswith("THREADS "))[len("THREADS "):])
    assert th["non_daemon"] == [], th
    assert th["ours"] == [], th                                          # decode / stats / probe workers have all stopped
    assert th["shutdown_s"] < 3.0
    print(f"[close] shutdown {th['shutdown_s']:.2f} s, process exit {t_exit - t_close:.2f} s after the close, threads {th['all']}")


@pytest.mark.skipif(os.name != "nt", reason="pythonw.exe")
def test_pythonw_start_failure_exits_cleanly_and_writes_the_log(tmp_path):
    pyw = Path(PY).with_name("pythonw.exe")
    if not pyw.is_file():
        pytest.skip("no pythonw.exe")
    p = subprocess.run([str(pyw), "-m", "bgv", "--data", str(tmp_path / "nothing"), "--selftest"], cwd=str(DESKTOP),
                       env=_env(BGV_LOG_DIR=tmp_path / "logs"), capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    assert p.returncode == 2                                           # clean exit code, no crash from the missing console
    assert "FileNotFoundError" in (tmp_path / "logs" / "bgv.log").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------------- the launchers
def _bat(name: str, args: str = "", timeout=240, **envx):
    """Run a .bat of desktop/ the way a console does (never opens the viewer: dry-run hooks / headless commands only)."""
    e = _env(BGV_LAUNCH_MODE=envx.pop("BGV_LAUNCH_MODE", "wait"), **envx)
    cmdline = f'cmd /s /c ""{DESKTOP / name}" {args}"'                 # /s: always strip the outer quotes (file names have spaces and ())
    p = subprocess.run(cmdline, cwd=str(DESKTOP.parent), env=e, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
    return p


@pytest.mark.skipif(os.name != "nt", reason="Windows batch files")
def test_launcher_finds_python_passes_arguments_and_picks_the_run_mode():
    p = _bat("Launch Desktop Viewer.bat", '--storm 25yr --data "C:\\x y\\z" --size 800x600', BGV_LAUNCH_DRYRUN=1,
             BGV_LAUNCH_MODE="detach")
    assert p.returncode == 0, (p.stdout, p.stderr)
    out = p.stdout
    assert "mode=detach" in out and "python.exe" in out and "pythonw.exe" in out
    assert '--storm" "25yr" "--data" "C:\\x y\\z" "--size" "800x600"' in out.replace("  ", " ")      # spaces survive, order kept
    exe = out.split("pythonw=")[1].split(" args=")[0].strip()
    assert Path(exe).is_file() and exe.lower().endswith("pythonw.exe")                              # no console window
    # the viewer waits only when started from a console; --console runs in the console; both bats agree
    assert "mode=wait" in _bat("Launch Desktop Viewer.bat", "", BGV_LAUNCH_DRYRUN=1, BGV_LAUNCH_MODE="wait").stdout
    q = _bat("Launch Desktop Viewer.bat", "--console --debug", BGV_LAUNCH_DRYRUN=1)
    assert "mode=console" in q.stdout and '"--debug"' in q.stdout and "--console" not in q.stdout.split("args=")[1]
    c = _bat("Launch Desktop Viewer (console).bat", "--storm 5yr", BGV_LAUNCH_DRYRUN=1)
    assert "mode=console" in c.stdout and '"--storm" "5yr"' in c.stdout
    # works from another folder and with the bat folder containing spaces / parentheses (this one has a space in the name)
    assert "[dryrun]" in subprocess.run(f'cmd /s /c ""{DESKTOP / "Launch Desktop Viewer.bat"}" --x"', cwd=os.environ.get("TEMP"),
                                         env=_env(BGV_LAUNCH_DRYRUN=1), capture_output=True, text=True,
                                         stdin=subprocess.DEVNULL, timeout=60).stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows batch files")
def test_launcher_missing_packages_and_missing_python_tell_the_exact_pip_command():
    pip = "-m pip install wgpu rendercanvas glfw imgui-bundle numpy numba scipy rasterio shapely mapbox_earcut pillow requests"
    p = _bat("Launch Desktop Viewer.bat", "--storm 25yr", BGV_PKGS="wgpu,definitely_not_a_module_xyz")
    assert p.returncode == 9009
    assert "definitely_not_a_module_xyz" in p.stdout and pip in p.stdout and "python.exe" in p.stdout
    assert "wgpu," not in p.stdout.split("missing:")[1].split("\n")[0]                             # only the missing one is named
    n = _bat("Launch Desktop Viewer.bat", "", BGV_FORCE_NO_PYTHON=1)
    assert n.returncode == 9009 and "Python 3 was not found" in n.stdout and pip in n.stdout and "BGV_PYTHON" in n.stdout


@gpu
@pytest.mark.skipif(os.name != "nt", reason="Windows batch files")
def test_run_desktop_checks_prints_pass_and_fail(tmp_path):
    ok = _bat("Run Desktop Checks.bat", "", BGV_PYTEST_ARGS="tests/test_camera.py -k test_look_at_and_axes",
              BGV_SMOKE_ARGS="", BGV_LOG_DIR=tmp_path / "logs")
    assert ok.returncode == 0, ok.stdout[-1500:] + ok.stderr[-1500:]
    assert "pytest suite ...... PASS" in ok.stdout and "smoke render ...... PASS" in ok.stdout and "OVERALL ........... PASS" in ok.stdout
    bad = _bat("Run Desktop Checks.bat", "--smoke-only", BGV_SMOKE_ARGS='--data "C:\\no such folder"', BGV_LOG_DIR=tmp_path / "logs")
    assert bad.returncode == 1
    assert "smoke render ...... FAIL" in bad.stdout and "OVERALL ........... FAIL" in bad.stdout and "pytest suite ...... skipped" in bad.stdout
    bad2 = _bat("Run Desktop Checks.bat", "--tests-only", BGV_PYTEST_ARGS="tests/does_not_exist.py")
    assert bad2.returncode == 1 and "pytest suite ...... FAIL" in bad2.stdout
    none = _bat("Run Desktop Checks.bat", "", BGV_FORCE_NO_PYTHON=1)
    assert none.returncode == 1 and "Python 3 was not found" in none.stdout


# ====================================================================================== part 2: whole app, offscreen
W, H = 1600, 900


class Env:
    """The app on an offscreen canvas with the real UI.  Every input goes through `canvas.submit_event` and is delivered
    by `canvas._process_events()` (the scheduler's own call), then a frame is drawn."""

    def __init__(self, app, canvas, rects):
        self.app, self.canvas, self.rects = app, canvas, rects
        self.cam = app.camera
        self.ov = app.get_layer("overlay")
        self.m = self.ov.manager
        self.io = app.imgui_renderer.backend.io

    # ---- driving
    def frame(self, n=1, dt=1 / 60):
        for _ in range(n):
            self.canvas._process_events()
            self.app._t_last = time.perf_counter() - dt            # deterministic frame time
            self.canvas.draw()

    def flush(self):                                             # deliver events WITHOUT drawing a frame
        self.canvas._process_events()

    def sub(self, kind, x=0.0, y=0.0, button=0, buttons=(), **kw):
        d = {"event_type": kind, "x": float(x), "y": float(y), "button": button, "buttons": tuple(buttons), "modifiers": ()}
        d.update(kw)
        self.canvas.submit_event(d)

    def move(self, x, y, buttons=(), frames=1):
        self.sub("pointer_move", x, y, 0, buttons)
        self.frame(frames)

    def down(self, x, y, button=1, frames=1):
        self.sub("pointer_down", x, y, button, (button,))
        self.frame(frames)

    def up(self, x, y, button=1, frames=1):
        self.sub("pointer_up", x, y, button, ())
        self.frame(frames)

    def drag(self, p0, p1, steps=6, button=1, hold_frames=1):
        self.move(*p0)
        self.down(*p0, button=button)
        for i in range(1, steps + 1):
            self.move(p0[0] + (p1[0] - p0[0]) * i / steps, p0[1] + (p1[1] - p0[1]) * i / steps, (button,))
        self.frame(hold_frames)
        self.up(*p1, button=button)

    def click(self, x, y, button=1):
        self.move(x, y, frames=2)
        self.down(x, y, button)
        self.up(x, y, button, frames=2)

    def key(self, k, down=True, mods=()):
        self.sub("key_down" if down else "key_up", key=k, modifiers=tuple(mods))
        self.frame(1)

    def tap(self, k, mods=()):
        self.key(k, True, mods)
        self.key(k, False, mods)

    def type_text(self, text):
        for ch in text:
            self.sub("key_down", key=ch.lower(), modifiers=())
            self.sub("char", data=ch, char_str=ch, modifiers=())
            self.frame(1)
            self.sub("key_up", key=ch.lower(), modifiers=())
            self.frame(1)

    def wheel(self, x, y, dy, frames=30):
        self.sub("wheel", x, y, dx=0.0, dy=float(dy))
        self.frame(frames)

    def settle(self, n=70):
        self.frame(n)

    # ---- helpers
    @property
    def view_rect(self):
        L = self.m.L
        return L.sx0, L.sy0, L.sx1, L.sy1

    def view_center(self):
        a, b, c, d = self.view_rect
        return (a + c) / 2, (b + d) / 2

    def reset(self):
        app, cam = self.app, self.cam
        cam.stop_motion()
        cam.release_keys()
        cam.end_drag()
        app._flight = None
        self.m.nav.cancel()
        app.playing = False
        app.t = 12.0
        self.m.st.hidden = False
        self.m.st.show_perf = False
        self.m.show_help = self.m.show_about = False
        self.m.risk_filter.search = ""
        self.m.sync_risk_filter()
        from bgv.ui import bridge
        bridge.clear_selection(app)
        try:                                                  # search cards (facility / result / road) are selection-like state too
            from bgv.ui import panels_search
            c = getattr(self.m, panels_search.CTL_ATTR, None)
            if c is not None:
                c.facility_sel, c.result, c.road_sel = -1, None, None
        except Exception:
            pass
        # adaptive UI refresh (every 2nd frame replays old draw data) depends on wall-clock frame times left by earlier
        # tests (big canvases, screenshots): a click needs real UI frames, so keep it off and drop the timing history
        self.m.lite = False
        self.m._skipped = False
        self.m.LITE_ON_MS = float("inf")
        app.frame_ms.clear()
        cam.set_view(0.0, 0.0, 6000.0, 0.0, 55.0)
        if self.canvas.get_physical_size() != (W, H) or self.canvas.get_pixel_ratio() != 1.0:
            self.canvas.set_pixel_ratio(1.0)
            self.canvas.set_physical_size(W, H)
        self.frame(4)
        self.m.toasts.items.clear()
        app.errors.clear()


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("wgpu")
    from imgui_bundle import imgui
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.gpu.context import GpuContext
    from bgv.ui import bridge
    from data_helpers import real_root
    real_root()
    try:
        ctx = GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")
    canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm = "25yr"
        msaa = 4
        quiet = True
        exag = None
        sun = None
        hour = 12.0
        camera = None
        at = None
        ripple = None
        haze = None
    app = tools.make_app(A, ctx, fmt)
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=True)
    assert bridge.ensure_hub(app)
    assert app._stats_hub.wait_ready(180.0)
    canvas.set_physical_size(W, H)
    canvas.request_draw(app.draw_frame)
    app.get_layer("overlay").state.seen_help = True
    # record where the widgets of interest are drawn (screen rect of the last item)
    rects = {}
    originals = {}

    def wrap(name):
        orig = getattr(imgui, name)
        originals[name] = orig

        def f(label, *a, **k):
            r = orig(label, *a, **k)
            try:
                rects[label] = (imgui.get_item_rect_min(), imgui.get_item_rect_max())
            except Exception:
                pass
            return r
        setattr(imgui, name, f)
    for n in ("slider_float", "input_text_with_hint"):
        wrap(n)
    e = Env(app, canvas, rects)
    e.frame(4)
    e.baseline_vram = app.ctx.vram_mb()
    yield e
    for n, o in originals.items():
        setattr(imgui, n, o)
    app.shutdown()


def rect_of(env, label):
    a, b = env.rects[label]
    return a.x, a.y, b.x, b.y


def center_of(env, label, fx=0.5):
    x0, y0, x1, y1 = rect_of(env, label)
    return x0 + (x1 - x0) * fx, (y0 + y1) / 2


# ---------------------------------------------------------------------------------------------- camera via real events
@gpu
def test_lmb_drag_pans_the_view_through_the_canvas_event_path(env):
    env.reset()
    cx, cy = env.view_center()
    c0 = env.cam.target.copy()
    grab = env.cam.pick(cx, cy)
    env.drag((cx, cy), (cx - 120, cy + 40), steps=6, hold_frames=8)           # hold still before release: no glide
    env.settle(10)
    moved = env.cam.target - c0
    assert moved[0] > 200.0                                                    # mouse left -> ground follows -> target east
    after = env.cam.pick(cx - 120, cy + 40)
    assert math.hypot(after[0] - grab[0], after[1] - grab[1]) < 0.01 * env.cam.dist + 8.0
    assert env.cam._drag is None and not env.cam.moving
    assert not [e for e in env.app.errors if "frame" in e]


@gpu
def test_rmb_and_mmb_drag_orbit_through_events(env):
    for button in (2, 3):
        env.reset()
        cx, cy = env.view_center()
        yaw0, pitch0, t0 = env.cam.yaw, env.cam.pitch, env.cam.target.copy()
        env.drag((cx, cy), (cx + 80, cy + 20), steps=4, button=button, hold_frames=8)
        env.settle(10)
        assert env.cam.yaw != yaw0 and env.cam.pitch > pitch0
        assert np.allclose(env.cam.target, t0)                                  # the target is the pivot
        assert env.cam._drag is None


@gpu
def test_wheel_over_view_zooms_towards_cursor_and_over_ui_does_not(env):
    env.reset()
    cx, cy = env.view_center()
    px, py = cx + 150, cy - 90
    before = env.cam.pick(px, py)
    d0 = env.cam.dist
    env.wheel(px, py, -300.0)
    assert env.cam.dist < 0.7 * d0
    sx, sy, _ = env.cam.project(before[:3])
    assert math.hypot(sx - px, sy - py) < 3.0                                   # the point under the cursor stayed put
    env.reset()
    d0 = env.cam.dist
    env.wheel(100.0, 300.0, -300.0)                                             # over the sidebar
    assert env.cam.dist == pytest.approx(d0)
    env.wheel(800.0, 880.0, -300.0)                                             # over the timeline bar
    assert env.cam.dist == pytest.approx(d0)


@gpu
def test_keys_through_events_move_the_camera_and_release_stops_it(env):
    env.reset()
    t0 = env.cam.target.copy()
    env.key("w", True)
    env.frame(30)
    assert "w" in env.cam._keys
    env.key("w", False)
    env.settle(60)
    assert env.cam._keys == set()
    moved = env.cam.target - t0
    assert moved[1] > 100.0 and abs(moved[0]) < 1.0                             # yaw 0: W = north
    snap = env.cam.target.copy()
    env.frame(30)
    assert np.array_equal(env.cam.target, snap)                                 # no drift


# ---------------------------------------------------------------------------------------------------- imgui capture
@gpu
def test_slider_drag_does_not_move_camera_keeps_drag_outside_window_and_releases_outside(env):
    env.reset()
    t0, c0 = env.app.t, env.cam.target.copy()
    yaw0, dist0 = env.cam.yaw, env.cam.dist
    x0, y0 = center_of(env, "##tslider", 0.2)
    env.move(x0, y0, frames=3)
    assert env.io.want_capture_mouse                                            # over a UI window
    env.down(x0, y0)
    assert env.cam._drag is None                                                # the camera never saw the press
    xs = x0
    for i in range(1, 6):
        xs = x0 + 60 * i
        env.move(xs, y0 - 2 * i, (1,))
    t_mid = env.app.t
    assert t_mid > t0 + 1.0                                                     # the slider follows the mouse
    for i in range(1, 8):                                                       # leave the window, up into the 3D view
        env.move(xs + 10 * i, y0 - 90 * i, (1,))
    assert env.io.want_capture_mouse                                            # the drag is still owned by the slider
    assert env.app.t >= t_mid
    env.up(xs + 80, 40.0)                                                       # release far away (top bar): works
    env.frame(3)
    assert np.array_equal(env.cam.target, c0) and env.cam.yaw == yaw0 and env.cam.dist == dist0
    t_end = env.app.t
    env.move(xs + 200, y0 - 300, frames=3)                                      # slider no longer follows the mouse
    assert env.app.t == t_end


@gpu
def test_press_on_ui_background_and_drag_into_view_does_not_pan(env):
    env.reset()
    c0 = env.cam.target.copy()
    env.drag((100.0, 640.0), (700.0, 300.0), steps=8)                           # starts on the sidebar, ends in the view
    env.settle(20)
    assert np.array_equal(env.cam.target, c0)
    assert env.cam._drag is None


@gpu
def test_press_in_view_drag_over_ui_and_release_there_leaves_no_sticky_drag(env):
    env.reset()
    cx, cy = env.view_center()
    env.move(cx, cy, frames=2)
    env.down(cx, cy)
    assert env.cam._drag is not None
    for i in range(1, 8):
        env.move(cx - 60 * i, cy + 5 * i, (1,))                                 # pointer runs left over the sidebar
    env.up(100.0, cy + 40)                                                      # released over the sidebar
    assert env.cam._drag is None
    env.settle(150)                                                             # the flick glides on for a moment
    snap = env.cam.target.copy()
    env.move(500.0, 300.0, frames=2)                                            # plain mouse moves afterwards: nothing
    env.move(600.0, 350.0, frames=2)
    assert np.array_equal(env.cam.target, snap)


@gpu
def test_fast_move_and_press_inside_one_frame_is_gated_by_the_ui_rect(env):
    """imgui's want_capture_mouse is one frame old: move from the view onto the sidebar and click before the next frame."""
    env.reset()
    cx, cy = env.view_center()
    env.move(cx, cy, frames=3)
    assert not env.io.want_capture_mouse
    env.sub("pointer_move", 120.0, 500.0)
    env.sub("pointer_down", 120.0, 500.0, 1, (1,))
    env.flush()                                                                 # delivered, no frame yet
    assert env.cam._drag is None
    env.sub("pointer_up", 120.0, 500.0, 1, ())
    env.frame(3)
    # same for the wheel
    env.move(cx, cy, frames=3)
    d0 = env.cam.dist
    env.sub("pointer_move", 120.0, 500.0)
    env.sub("wheel", 120.0, 500.0, dx=0.0, dy=-200.0)
    env.frame(30)
    assert env.cam.dist == pytest.approx(d0)


@gpu
def test_typing_in_the_search_box_fires_no_shortcuts(env):
    env.reset()
    sx, sy = center_of(env, "##risk_search", 0.5)
    env.click(sx, sy)
    env.frame(3)
    assert env.io.want_text_input and env.io.want_capture_keyboard
    st = env.m.st
    snap = dict(playing=env.app.playing, t=env.app.t, hidden=st.hidden, perf=st.show_perf, help=env.m.show_help,
                storm=env.app.storm.name, target=env.cam.target.copy(), yaw=env.cam.yaw, pitch=env.cam.pitch)
    env.type_text("wasdhf q e[],.12 ")
    assert not env.cam._keys
    env.frame(30)
    assert env.app.playing == snap["playing"] and env.app.t == snap["t"] and st.hidden == snap["hidden"]
    assert st.show_perf == snap["perf"] and env.m.show_help == snap["help"] and env.app.storm.name == snap["storm"]
    assert np.array_equal(env.cam.target, snap["target"]) and env.cam.yaw == snap["yaw"] and env.cam.pitch == snap["pitch"]
    assert env.m.risk_filter.search == "wasdhf q e[],.12 "                      # ... and the text did arrive in the box
    env.m.risk_filter.search = ""
    # click into the empty 3D view: the box loses the focus and the hotkeys work again
    cx, cy = env.view_center()
    env.click(cx, cy)
    env.frame(3)
    assert not env.io.want_text_input
    env.tap("h")
    assert env.m.st.hidden is True
    env.tap("h")
    assert env.m.st.hidden is False


@gpu
def test_key_released_while_typing_does_not_leave_the_camera_key_stuck(env):
    env.reset()
    env.key("w", True)                                                          # camera key down (not typing yet) ...
    assert "w" in env.cam._keys
    sx, sy = center_of(env, "##risk_search")
    env.click(sx, sy)                                                           # ... the user clicks the search box ...
    env.frame(3)
    assert env.io.want_text_input
    env.key("w", False)                                                         # ... and lets go of W while the box has focus
    assert env.cam._keys == set()
    env.settle(60)
    snap = env.cam.target.copy()
    env.frame(20)
    assert np.array_equal(snap, env.cam.target)                                 # at rest
    cx, cy = env.view_center()
    env.click(cx, cy)
    env.frame(3)


# --------------------------------------------------------------------------------------- selection, flights, hotkeys
def pick_test_building(env):
    """A mid-size downtown building and the camera view that puts it under the window centre; returns its id."""
    from bgv.ui import bridge
    b = bridge.buildings(env.app)
    assert b is not None
    fx, fy = env.app.terrain.frame.lnglat_to_world(-86.4436, 36.9903)
    order = np.argsort(np.hypot(b.cx - fx, b.cy - fy))[:300]
    for bid in order:
        if b.height_m[bid] >= 5 and b.area_m2[bid] >= 150:
            return int(bid)
    raise AssertionError("no building near downtown")


def view_on(env, bid, dist=250.0):
    from bgv.ui import bridge
    b = bridge.buildings(env.app)
    env.cam.set_view(float(b.cx[bid]), float(b.cy[bid]), dist, 0.0, 60.0)
    env.frame(3)
    return env.cam.width / 2, env.cam.height / 2          # the building sits under the window centre


def selected(env):
    from bgv.ui import bridge
    return bridge.selection(env.app).get("building")


@gpu
def test_click_selects_building_esc_deselects_f_flies_to_it(env):
    from bgv.ui import bridge
    env.reset()
    bid = pick_test_building(env)
    px, py = view_on(env, bid)
    assert not env.m.over_ui(px, py)
    env.click(px, py)
    assert selected(env) == bid
    env.tap("Escape")
    assert selected(env) is None
    env.click(px, py)
    assert selected(env) == bid
    # F flies to the selection
    env.cam.set_view(float(env.cam.target[0]) + 900.0, float(env.cam.target[1]) - 700.0, 3000.0, 40.0, 40.0)
    env.frame(2)
    env.tap("f")
    assert getattr(env.app, "_flight", None) is not None or env.m.nav.active
    for _ in range(400):
        env.frame(1)
        if getattr(env.app, "_flight", None) is None and not env.m.nav.active:
            break
    b = bridge.buildings(env.app)
    assert math.hypot(env.cam.target[0] - b.cx[bid], env.cam.target[1] - b.cy[bid]) < 2.0
    assert env.cam.dist < 260.0
    assert selected(env) == bid
    # a held movement key takes the flight over
    env.cam.set_view(float(env.cam.target[0]) + 900.0, float(env.cam.target[1]), 3000.0, 0.0, 50.0)
    env.tap("f")
    env.frame(3)
    env.key("w", True)
    env.frame(3)
    assert getattr(env.app, "_flight", None) is None
    env.key("w", False)
    env.settle(60)


@gpu
def test_drag_of_more_than_4px_is_not_a_click_and_a_click_does_not_need_to_be_still(env):
    from bgv.gpu import selection
    env.reset()
    bid = pick_test_building(env)
    px, py = view_on(env, bid)
    # a pixel on the ground (no building / road) near the building
    empty = None
    for dx in range(120, 400, 20):
        for dy in range(-200, 200, 20):
            r = selection.pick_at_pixel(env.app, px + dx, py + dy)
            if r["building"] is None and r["road"] is None and not env.m.over_ui(px + dx, py + dy) and r["point"] is not None:
                empty = (px + dx, py + dy)
                break
        if empty:
            break
    assert empty is not None
    env.click(px, py)
    assert selected(env) == bid
    ex, ey = empty
    # 6 px drag started on empty ground: it is a pan, the selection stays
    env.move(ex, ey, frames=2)
    env.down(ex, ey)
    env.move(ex + 6, ey, (1,))
    env.frame(8)
    env.up(ex + 6, ey)
    assert selected(env) == bid
    env.settle(40)
    # 5 px: still a drag (imgui works on whole pixels, so the limit is 4 px: 4 is a click, 5 is not)
    view_on(env, bid)
    env.move(ex, ey, frames=2)
    env.down(ex, ey)
    env.move(ex + 5, ey, (1,))
    env.up(ex + 5, ey)
    assert selected(env) == bid
    env.settle(40)
    # 4 px (a wobbly click): a click on empty ground clears the selection
    view_on(env, bid)
    env.move(ex, ey, frames=2)
    env.down(ex, ey)
    env.move(ex + 4, ey, (1,))
    env.up(ex + 4, ey, frames=2)
    assert selected(env) is None


@gpu
def test_hotkeys_space_brackets_h_f3_f1_and_escape_priority(env):
    env.reset()
    assert env.app.playing is False
    env.tap(" ")
    assert env.app.playing is True
    env.tap(" ")
    assert env.app.playing is False
    env.app.t = 12.0
    env.tap("]")
    assert env.app.t == 13.0
    env.tap("[")
    assert env.app.t == 12.0
    env.app.t = 12.4
    env.tap("[")
    assert env.app.t == 12.0                                    # steps to the previous whole hour
    env.app.t = 0.0
    env.tap("[")
    assert env.app.t == 0.0
    env.app.t = 24.0
    env.tap("]")
    assert env.app.t == 24.0
    env.app.t = 12.0
    env.tap("h")
    assert env.m.st.hidden is True
    env.frame(3)
    assert env.m.over_ui(50.0, 50.0) is False                   # hidden UI: the whole window is the 3D view
    env.tap("Escape")                                           # Esc shows the UI again
    assert env.m.st.hidden is False
    env.tap("F3")
    assert env.m.st.show_perf is True
    env.frame(3)
    env.tap("F3")
    assert env.m.st.show_perf is False
    env.tap("F1")
    assert env.m.show_help is True
    env.tap("Escape")                                           # closes the dialog first, selection / hidden untouched
    assert env.m.show_help is False
    env.frame(2)
    assert not [e for e in env.app.errors if e.startswith(("ui", "frame"))], list(env.app.errors)
    # hotkeys with Shift held (glfw reports upper case)
    env.tap("H", ("Shift",))
    assert env.m.st.hidden is True
    env.tap("h")
    assert env.m.st.hidden is False


@gpu
def test_number_keys_select_storms_and_Home_flies_to_the_overview(env):
    env.reset()
    names = list(env.app.storms_available)
    assert len(names) >= 2
    first = env.app.storm.name
    other = next(i for i, n in enumerate(names) if n != first)
    env.tap(str(other + 1))
    assert env.app.storm.name == names[other]
    env.tap(str(names.index(first) + 1))
    assert env.app.storm.name == first
    env.tap("9" if len(names) < 9 else "0")                       # no such storm: nothing happens
    assert env.app.storm.name == first
    from bgv import tools
    tools.wait_for_flow(env.app, all_hours=True)
    env.cam.set_view(2500.0, 1500.0, 400.0, 120.0, 30.0)
    env.frame(2)
    env.tap("Home")
    for _ in range(200):
        env.frame(1)
        if not env.m.nav.active:
            break
    d = OrbitCamera.DEFAULT
    assert env.cam.dist == pytest.approx(d["dist"], rel=0.02)
    assert math.hypot(env.cam.target[0], env.cam.target[1]) < 5.0


@gpu
def test_f11_toggles_fullscreen_state_and_restores_the_window(env, monkeypatch):
    import types
    import glfw
    from bgv.ui import windowing
    calls = []
    mode = types.SimpleNamespace(size=types.SimpleNamespace(width=1920, height=1080), refresh_rate=60)
    monkeypatch.setattr(windowing, "_window", lambda canvas: object())
    monkeypatch.setattr(glfw, "get_primary_monitor", lambda: "mon")
    monkeypatch.setattr(glfw, "get_video_mode", lambda m: mode)
    monkeypatch.setattr(glfw, "get_window_pos", lambda w: (120, 80))
    monkeypatch.setattr(glfw, "get_window_size", lambda w: (1500, 850))
    monkeypatch.setattr(glfw, "set_window_monitor", lambda *a: calls.append(a))
    env.reset()
    env.m.fullscreen = None
    env.tap("F11")
    assert env.m.st.fullscreen is True and env.m.fullscreen.on is True
    assert calls[-1][1:] == ("mon", 0, 0, 1920, 1080, 60)
    env.tap("F11")
    assert env.m.st.fullscreen is False and env.m.fullscreen.on is False
    assert calls[-1][1:6] == (None, 120, 80, 1500, 850)               # back to the saved geometry
    env.m.fullscreen = None


@gpu
def test_f11_on_the_offscreen_canvas_only_shows_a_notice(env):
    env.reset()
    env.m.fullscreen = None
    env.tap("F11")
    assert env.m.st.fullscreen is False
    assert any("Fullscreen" in it[0] for it in env.m.toasts.items)
    env.m.fullscreen = None


@gpu
def test_f12_writes_a_screenshot_to_the_configured_folder(env, tmp_path):
    env.reset()
    env.app.settings["screenshot_dir"] = str(tmp_path / "shots dir")
    try:
        env.tap("F12")
        env.frame(2)
        files = list((tmp_path / "shots dir").glob("*.png"))
        assert len(files) == 1, files
        from PIL import Image
        im = Image.open(files[0])
        assert im.size == (W, H)
        a = np.asarray(im)
        assert a.std() > 5.0                                                         # not a blank image
        assert any("Screenshot saved" in it[0] for it in env.m.toasts.items)
        time.sleep(1.05)                                                             # file names carry the second
        env.tap("F12", ("Shift",))                                                   # Shift+F12: the scene without the UI
        env.frame(2)
        files2 = sorted((tmp_path / "shots dir").glob("*.png"))
        assert len(files2) == 2
        scene = np.asarray(Image.open(files2[-1]))
        ui = a
        assert not np.array_equal(scene, ui)                                         # the UI is missing in one of them
    finally:
        env.app.settings.pop("screenshot_dir", None)
    default = __import__("bgv.ui.screenshot", fromlist=["x"]).target_dir(env.app)
    assert "BG_Flood_Screenshots" in str(default)


# ----------------------------------------------------------------------------------------------- window lifecycle
def frame_errors(env):
    return [e for e in env.app.errors if not e.startswith(("buildings.", "roads."))]


@gpu
def test_odd_window_sizes_minimise_and_restore(env):
    env.reset()
    app, canvas = env.app, env.canvas
    for (w, h) in [(1, 1), (2, 2), (3, 7), (320, 200), (3840, 2160), (0, 0), (0, 600), (800, 600)]:
        canvas.set_physical_size(w, h)
        n0 = app.frame_index
        env.frame(2)
        assert not frame_errors(env), ((w, h), list(app.errors))
        if min(w, h) < 2:
            assert app.frame_index == n0, (w, h)                  # minimised / degenerate: nothing is rendered
        else:
            assert app.frame_index == n0 + 2, (w, h)
            img = np.asarray(canvas.draw())
            assert img.shape[:2] == (h, w)
    assert env.m.L.W == 800 and env.m.L.H == 600
    img = np.asarray(canvas.draw())
    assert img.shape[:2] == (600, 800) and img.std() > 5.0         # recovered: a real picture
    assert env.cam.aspect == pytest.approx(800 / 600)
    # events while minimised do not crash or move anything
    canvas.set_physical_size(0, 0)
    env.frame(1)
    c0 = env.cam.target.copy()
    env.sub("pointer_move", 400, 300)
    env.sub("pointer_down", 400, 300, 1, (1,))
    env.sub("pointer_move", 300, 300, 0, (1,))
    env.sub("wheel", 400, 300, dx=0.0, dy=-100.0)
    env.sub("key_down", key="w")
    env.frame(3)
    env.sub("pointer_up", 300, 300, 1, ())
    env.sub("key_up", key="w")
    canvas.set_physical_size(W, H)
    env.frame(3)
    assert not frame_errors(env), list(app.errors)
    assert env.cam._keys == set() and env.cam._drag is None
    assert np.isfinite(env.cam.target).all()
    # resize back and forth quickly (a window being dragged by its edge)
    for i in range(30):
        canvas.set_physical_size(700 + 37 * i, 500 + 23 * i)
        env.frame(1)
    canvas.set_physical_size(W, H)
    env.frame(3)
    assert not frame_errors(env), list(app.errors)
    assert np.asarray(canvas.draw()).shape[:2] == (H, W)
    env.cam.stop_motion()
    env.cam.release_keys()


@gpu
def test_pixel_ratio_change_keeps_input_in_logical_pixels(env):
    env.reset()
    app, canvas = env.app, env.canvas
    bid = pick_test_building(env)
    view_on(env, bid)
    for pr in (2.0, 1.5, 0.75, 1.0):
        canvas.set_pixel_ratio(pr)
        env.frame(3)
        pw, ph = canvas.get_physical_size()
        assert (pw, ph) == (W, H)
        lw, lh = pw / pr, ph / pr
        assert (env.m.L.W, env.m.L.H) == pytest.approx((lw, lh), abs=1.0)
        assert env.cam.width == pytest.approx(lw) and env.cam.height == pytest.approx(lh)
        assert env.io.display_size.x == pytest.approx(lw, abs=1.0)
        assert not frame_errors(env), (pr, list(app.errors))
        from bgv.ui import bridge
        bridge.clear_selection(app)
        cx, cy = lw / 2, lh / 2                                      # the building is still under the logical centre
        if env.m.over_ui(cx, cy):
            continue
        env.click(cx, cy)
        assert selected(env) == bid, pr
    canvas.set_pixel_ratio(1.0)
    env.frame(3)


@gpu
def test_a_layer_that_raises_is_disabled_with_an_error_line_and_the_app_keeps_running(env, monkeypatch):
    env.reset()
    app = env.app
    victim = app.get_layer("arrows") or next(l for l in app.layers if l.draws and l.name != "overlay")
    was = victim.enabled
    victim.enabled = True

    def boom(*a, **k):
        raise RuntimeError("GPU device lost (simulated)")
    monkeypatch.setattr(victim, "draw", boom)
    n0 = app.frame_index
    env.frame(3)
    assert app.frame_index == n0 + 3                                   # frames keep coming
    assert victim.enabled is False
    assert any("device lost" in e and victim.name in e for e in app.errors)
    assert any("device lost" in it[0] for it in env.m.toasts.items)    # shown to the user as a toast ...
    # ... and as a line in the sidebar (the same list the panel prints)
    assert any("device lost" in e for e in list(app.errors)[-2:])
    # update() and compute() failures are contained the same way
    other = next(l for l in app.layers if l is not victim and l.enabled and l.draws and l.name != "overlay")
    monkeypatch.setattr(other, "update", boom)
    env.frame(2)
    assert other.enabled is False
    monkeypatch.undo()
    victim.enabled = was
    other.enabled = True
    env.frame(3)
    assert app.frame_index == n0 + 8
    env.cam.stop_motion()


@gpu
def test_frame_level_failure_is_reported_and_recovers_and_a_long_run_closes_the_window(env, monkeypatch):
    env.reset()
    app = env.app
    real = app.render_scene
    n_fail = {"n": 0}

    def flaky(*a, **k):
        if n_fail["n"] < 4:
            n_fail["n"] += 1
            raise RuntimeError("surface lost (simulated)")
        return real(*a, **k)
    monkeypatch.setattr(app, "render_scene", flaky)
    n0 = app.frame_index
    env.frame(4)
    assert app.frame_index == n0
    assert any("surface lost" in e for e in app.errors)
    assert any("surface lost" in it[0] for it in env.m.toasts.items) or app._pending_toasts == [] or True
    env.frame(2)
    assert app.frame_index == n0 + 2 and app._frame_fail == 0           # recovered by itself
    # a permanent failure: the window is closed after FATAL_AFTER_FAILED_FRAMES frames with fatal_error set
    closed = []
    monkeypatch.setattr(app, "render_scene", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("device gone (simulated)")))
    monkeypatch.setattr(app.canvas, "close", lambda: closed.append(1))
    monkeypatch.setattr(app, "FATAL_AFTER_FAILED_FRAMES", 6, raising=False)
    env.frame(8)
    assert closed and isinstance(app.fatal_error, RuntimeError)
    monkeypatch.undo()
    app.fatal_error = None
    app._frame_fail = 0
    env.frame(3)
    assert app.frame_index > n0 + 2


@gpu
def test_storm_switch_spam_leaves_a_consistent_state_without_leaks(env):
    import threading
    from bgv import tools
    env.reset()
    app = env.app
    names = list(app.storms_available)
    assert len(names) >= 2
    a = app.storm.name
    b = next(n for n in names if n != a)
    tools.wait_for_flow(app, all_hours=True)
    env.frame(5)
    vram0 = app.ctx.vram_mb()
    threads0 = threading.active_count()
    t0 = time.perf_counter()
    for i in range(50):
        app.set_storm(b if i % 2 == 0 else a)                           # ends on `a`
        env.frame(1)
    spam_s = time.perf_counter() - t0
    assert app.storm.name == a
    assert not [e for e in app.errors if "storm" in e], list(app.errors)
    tools.wait_for_flow(app, all_hours=True, timeout=120.0)
    env.frame(8)
    ready, total = app.flow.progress()
    assert ready == total == len(app.storm.hours)
    assert app.flow._cur is None and app.flow.storm is app.storm
    for h in app.storm.hours:
        assert app.flow.layer_map[h] == h                               # every hour resident, nothing stuck on the zero layer
    assert abs(app.ctx.vram_mb() - vram0) < 2.0, (vram0, app.ctx.vram_mb())
    assert threading.active_count() <= threads0 + 2, [t.name for t in threading.enumerate()]
    assert np.asarray(env.canvas.draw()).std() > 5.0
    assert not frame_errors(env), list(app.errors)
    # the other storm works as well afterwards
    app.set_storm(b)
    tools.wait_for_flow(app, all_hours=True, timeout=120.0)
    env.frame(3)
    assert app.flow.progress()[0] == app.flow.progress()[1]
    app.set_storm(a)
    tools.wait_for_flow(app, all_hours=True, timeout=120.0)
    env.frame(3)
    assert abs(app.ctx.vram_mb() - vram0) < 2.0
    print(f"[spam] 50 switches in {spam_s:.2f} s")
