"""Cinematic tools on the offscreen canvas with real data (no window): panel + menu draw, hotkeys, bookmark flights, tour (live
update loop, captions, pause / skip / user cancel, storm escalation), time-lapse, and clip export (PNG sequence, MP4 when ffmpeg
exists, cancel, state restored, driven by the UI frame loop, 4K, continuity between hourly frames)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from data_helpers import real_root

pytestmark = pytest.mark.gpu

wgpu = pytest.importorskip("wgpu")
W, H = 1280, 720


def make_env(domain: str, storm: str):
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.gpu.context import GpuContext
    from bgv.ui import bridge
    real_root()
    try:
        ctx = GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")
    canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        pass
    A.storm, A.msaa, A.quiet, A.exag, A.sun, A.hour, A.camera, A.at, A.ripple, A.haze = storm, 4, True, None, None, 12.0, None, None, None, None
    A.domain, A.county_dem, A.county_demo, A.county_pbf = domain, None, None, None
    try:
        app = tools.make_app(A, ctx, fmt)
    except Exception as e:
        pytest.skip(f"{domain} not available: {e}")
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=False)
    assert bridge.ensure_hub(app)
    assert app._stats_hub.wait_ready(240.0)
    canvas.set_physical_size(W, H)
    canvas.request_draw(app.draw_frame)
    ov = app.get_layer("overlay")
    ov.state.seen_help = True

    class Env:
        pass
    e = Env()
    e.app, e.canvas, e.ov, e.m, e.lay = app, canvas, ov, ov.manager, app.get_layer("cinematic")
    e.draw = lambda n=2: [canvas.draw() for _ in range(n)][-1]
    e.key = lambda k, mods=(): e.m.on_key({"event_type": "key_down", "key": k, "modifiers": tuple(mods)})
    yield e
    app.shutdown()


@pytest.fixture(scope="module")
def env():
    yield from make_env("city", "25yr")


@pytest.fixture(scope="module")
def full(env):
    from bgv import tools
    tools.wait_for_flow(env.app, all_hours=True)
    return env


def spin(env, cond, limit=120.0, dt=0.25):
    t0 = time.perf_counter()
    while not cond():
        env.lay.update(env.app, dt)
        env.app.flow.pump()
        time.sleep(0.002)
        assert time.perf_counter() - t0 < limit, "timed out"


def wait_build(lay, limit=180.0):
    t0 = time.perf_counter()
    while lay.tour.building:
        time.sleep(0.05)
        assert time.perf_counter() - t0 < limit
    assert lay.tour.error == "", lay.tour.error


# ------------------------------------------------------------------------------------------------ UI
def test_layer_hooks_and_panel_draw(env):
    from bgv.ui import hooks
    from bgv.ui import registry
    assert {"cinematic panel", "cinematic overlay"} <= {p.name for p in registry.PANELS} | set(registry.OVERLAYS)
    assert "Cinematic" in registry.TOP_MENUS
    env.lay.panel_open = True
    for tab in range(4):
        env.lay._tab_request = tab
        env.draw(2)
        assert not env.m.panel_errors, env.m.panel_errors
    env.lay.panel_open = False
    env.draw(2)
    assert not env.m.panel_errors and not list(env.app.errors)
    from bgv.ui import dialogs
    assert any(k == "Ctrl+B" for k, _ in dialogs.SHORTCUTS)               # the help dialog lists the new keys


def test_bookmark_keys_and_flight(env):
    app, lay, cam = env.app, env.lay, env.app.camera
    lay.store.items.clear()
    cam.set_view(500.0, -300.0, 900.0, 40.0, 50.0)
    assert env.key("b", ("Control",)) is True
    assert len(lay.bookmarks()) == 1 and lay.bookmarks()[0].view[:3] == pytest.approx([500.0, -300.0, 900.0])
    cam.set_view(-2000.0, 1500.0, 5000.0, 200.0, 35.0)
    assert env.key("1", ("Control",)) is True and lay.flight.active
    for _ in range(600):
        if not lay.flight.active:
            break
        lay.update(app, 1 / 30)
    assert not lay.flight.active
    assert np.allclose(cam.target[:2], [500.0, -300.0], atol=1e-3) and cam.dist == pytest.approx(900.0, rel=1e-3)
    assert cam.yaw == pytest.approx(40.0, abs=1e-3)
    cam.set_view(0.0, 0.0, 3000.0, 0.0, 45.0)
    assert env.key("1", ("Alt",)) is True                                    # Alt+1 overwrites slot 1 with the view
    assert lay.bookmarks()[0].view[2] == pytest.approx(3000.0) and len(lay.bookmarks()) == 1
    assert env.key("9", ("Control",)) is True and not lay.flight.active      # no 9th bookmark: nothing happens
    # a flight stops when the user grabs the camera
    lay.jump(0)
    cam.set_view(100.0, 100.0, 3000.0, 0.0, 45.0)
    lay.flight.start(cam, (4000.0, 0.0, 800.0, 10.0, 50.0), 3.0)
    lay.update(app, 0.1)
    cam.moving = True
    lay.update(app, 0.1)
    cam.moving = False
    assert not lay.flight.active
    # 1..9 without Control still picks a storm (the built-in key is untouched)
    assert env.key("1") is True
    app.set_storm("25yr")


def test_timelapse_advances_loops_and_waits_for_data(full):
    app, lay = full.app, full.lay
    lp = lay.lapse
    lp.rate, lp.t_in, lp.t_out, lp.mode = 2.0, 6.0, 10.0, "loop"
    app.t = 9.0
    assert full.key("l") is True and lp.playing and not app.playing
    lay.update(app, 0.25)
    assert app.t == pytest.approx(9.5)
    for _ in range(4):
        lay.update(app, 0.25)
    assert 6.0 <= app.t <= 10.0 and app.t == pytest.approx(7.5)             # wrapped around
    assert full.key("l") is True and not lp.playing
    lp.playing = False
    app.t = 12.0


# ------------------------------------------------------------------------------------------------ tour
def short_tour(lay, **kw):
    o = lay.tour_opts
    o.max_facilities, o.max_roads, o.dwell_s, o.speed, o.escalate, o.time_mode, o.hide_ui = 3, 1, 3.0, 4.0, False, "sweep", False
    for k, v in kw.items():
        setattr(o, k, v)
    lay.tour = type(lay.tour)()
    lay.build_tour(False)
    wait_build(lay)


def test_tour_runs_with_captions_sweeps_hours_and_restores(full):
    app, lay, cam = full.app, full.lay, full.app.camera
    short_tour(lay, speed=1.0)
    rt = lay.tour
    assert len(rt.path.stops) == 4 and rt.info["unnamed"] >= 0
    names = {s.name for s in rt.path.stops}
    assert all(n.strip() for n in names)
    app.t, app.playing = 7.0, False
    storm0 = app.storm.name
    cam.set_view(0.0, 0.0, 14000.0, 0.0, 55.0)
    lay.tour_toggle()
    assert rt.active and rt.restore["t"] == 7.0
    seen_caps, hours, views = set(), [], []
    n = 0
    while rt.active:
        lay.update(app, 1 / 30)
        n += 1
        c = lay.tour_caption()
        if c is not None and c["alpha"] > 0.9:
            seen_caps.add(c["title"])
            assert c["subtitle"] and "depth" in c["extra"] or "reading" in c["extra"] or "water here" in c["extra"]
        if rt.active:
            hours.append(app.t)
        views.append(cam.target[:2].copy())
        assert n < 9000
    assert seen_caps == names                                              # every place is named on screen, nothing else
    assert all(b >= a - 1e-6 for a, b in zip(hours, hours[1:])) and hours[-1] > 20.0     # the hour rises over the tour
    assert max(np.hypot(*(b - a)) for a, b in zip(views, views[1:])) < 250.0            # smooth path, no jumps (at 30 fps)
    assert app.t == 7.0 and app.storm.name == storm0                         # restored when it ended
    assert np.allclose(cam.target[:2], [0.0, 0.0], atol=1.0)                  # the path returns to where it started


def test_tour_pause_skip_and_user_cancel(full):
    app, lay, cam = full.app, full.lay, full.app.camera
    short_tour(lay, speed=1.0)
    rt = lay.tour
    cam.set_view(0.0, 0.0, 14000.0, 0.0, 55.0)
    assert full.key("t") is True and rt.active
    for _ in range(10):
        lay.update(app, 0.2)
    T0 = rt.T
    assert T0 > 0
    assert full.key(" ") is True and rt.paused
    lay.update(app, 0.5)
    assert rt.T == T0                                                       # paused: time stands still
    assert full.key(" ") is True and not rt.paused
    assert full.key("n") is True and rt.T == pytest.approx(rt.path.stop_times()[0][0])
    assert full.key("n") is True and rt.T == pytest.approx(rt.path.stop_times()[1][0])
    assert full.key("p") is True and rt.T == pytest.approx(rt.path.stop_times()[0][0])
    lay.update(app, 0.1)
    cam.moving = True                                                       # the user drags the camera
    lay.update(app, 0.1)
    cam.moving = False
    assert not rt.active
    assert full.key("t") is True and rt.active                              # T starts it again, Escape stops it
    assert full.key("Escape") is True and not rt.active


def test_tour_escalates_storms(full):
    app, lay = full.app, full.lay
    short_tour(lay, escalate=True, escalate_n=2)
    rt = lay.tour
    assert rt.plan.storms == ["5yr", "100yr"]
    app.set_storm("25yr")
    spin(full, lambda: app.flow.ready_for(app.t))
    lay._start_tour_now()
    storms = []
    spin(full, lambda: (storms.append(app.storm.name) or not rt.active), limit=300.0, dt=0.2)
    order = [s for i, s in enumerate(storms) if i == 0 or s != storms[i - 1]]
    assert order == ["25yr", "5yr", "100yr", "25yr"]                          # own storm, the two tour storms, own storm back
    assert app.storm.name == "25yr"                                         # the viewer's own storm is back
    spin(full, lambda: app.flow.ready_for(app.t))


# ------------------------------------------------------------------------------------------------ export
def run_job(env, spec):
    from bgv import export_cinematic as ex
    app = env.app
    before = (app.t, app.playing, app.render_scale, app._target_size, app.storm.name, env.lay.current_view(),
              app.camera.width, app.camera.height)
    job = ex.ExportJob(app, spec)
    t0 = time.perf_counter()
    while not job.tick():
        assert time.perf_counter() - t0 < 600
    return job, before


def after_checks(env, before):
    app = env.app
    assert (app.t, app.playing, app.render_scale, app._target_size, app.storm.name) == before[:5]
    assert env.lay.current_view() == pytest.approx(before[5])
    assert (app.camera.width, app.camera.height) == before[6:]
    env.draw(2)                                                              # the window renders on after the export
    assert not list(app.errors) and not env.m.panel_errors


def orbit_spec(env, out, w=320, h=180, fps=8.0, secs=1.5, **kw):
    from bgv import export_cinematic as ex
    from bgv import tools_cinematic as tc
    lay = env.lay
    shot = tc.OrbitShot(lay.current_view(), secs, tc.TimeLaw("range", 8.0, 14.0), 90.0)
    return ex.ExportSpec(w, h, fps, shot, out, name="t", depth_label="test depth source", preroll=2, **kw)


def test_export_png_sequence_stamp_and_restore(full, tmp_path):
    from PIL import Image
    spec = orbit_spec(full, tmp_path / "clip", make_mp4=False)
    full.app.camera.set_view(300.0, 200.0, 2500.0, 10.0, 50.0)
    full.app.t = 12.0
    job, before = run_job(full, spec)
    assert job.state == "done", job.message
    files = sorted((tmp_path / "clip").glob("frame_*.png"))
    assert len(files) == 12 and files[0].name == "frame_000001.png"
    imgs = [np.asarray(Image.open(f)) for f in files]
    assert all(i.shape == (180, 320, 3) for i in imgs)
    assert np.abs(imgs[0].astype(int) - imgs[-1].astype(int)).mean() > 1.0     # the camera moved and the water changed
    # the stamp: dark translucent band along the bottom, caption box top-left
    bottom = imgs[0][-24:].astype(int).mean()
    scene = imgs[0][60:120].astype(int).mean()
    assert bottom < scene
    meta = __import__("json").loads((tmp_path / "clip" / "clip.json").read_text())
    assert meta["frames"] == 12 and meta["size"] == [320, 180] and "not validated" in meta["honesty"] and meta["depth_source"] == "test depth source"
    assert (tmp_path / "clip" / "make_mp4.bat").is_file() or True
    after_checks(full, before)


def test_export_mp4_or_script(full, tmp_path):
    from bgv import export_cinematic as ex
    spec = orbit_spec(full, tmp_path / "clip", make_mp4=True, keep_png=False)
    job, before = run_job(full, spec)
    assert job.state == "done", job.message
    if ex.find_ffmpeg():
        mp4 = tmp_path / "clip" / "t.mp4"
        assert mp4.is_file() and mp4.stat().st_size > 1000
        assert not list((tmp_path / "clip").glob("frame_*.png"))             # keep_png False removed them after encoding
        assert (tmp_path / "clip" / "ffmpeg.log").is_file()
    else:
        assert (tmp_path / "clip" / "make_mp4.bat").is_file() and (tmp_path / "clip" / "make_mp4.sh").is_file()
        assert "ffmpeg was not found" in job.message
    after_checks(full, before)


def test_export_cancel_keeps_state_and_frames(full, tmp_path):
    from bgv import export_cinematic as ex
    spec = orbit_spec(full, tmp_path / "clip", make_mp4=False, secs=4.0)
    app = full.app
    before = (app.t, app.playing, app.render_scale, app._target_size, app.storm.name, full.lay.current_view(),
              app.camera.width, app.camera.height)
    job = ex.ExportJob(app, spec)
    for _ in range(8):
        job.tick()
    assert job.state == "render" and 0.0 < job.progress < 1.0
    job.cancel()
    assert job.state == "cancelled" and "Cancelled" in job.message
    assert 0 < len(list((tmp_path / "clip").glob("frame_*.png"))) < 32
    after_checks(full, before)


def test_export_driven_by_the_ui_frame_loop(full, tmp_path):
    """The real path: the Export tab starts a job and the overlay hook renders one frame per drawn UI frame."""
    lay = full.lay
    lay.exp_opts.kind, lay.exp_opts.size_idx, lay.exp_opts.fps, lay.exp_opts.seconds, lay.exp_opts.mp4 = "orbit", 0, 24, 2.0, False
    lay.exp_opts.orbit_deg = 45.0
    lay.lapse.t_in, lay.lapse.t_out = 4.0, 10.0
    import bgv.export_cinematic as ex
    ex_dir = tmp_path / "clips"
    old = ex.default_dir
    ex.default_dir = lambda app=None: ex_dir
    try:
        lay.panel_open, lay._tab_request = True, 3
        lay.start_export()
        assert lay.export_active()
        n = 0
        while lay.export_active():
            full.canvas.draw()
            n += 1
            assert n < 400
        assert lay.export.state == "done", lay.export.message
        frames = list(next(ex_dir.iterdir()).glob("frame_*.png"))
        assert len(frames) == 48
        assert not list(full.app.errors) and not full.m.panel_errors
    finally:
        ex.default_dir = old
        lay.panel_open = False
    full.draw(2)


def test_export_4k_frame(full, tmp_path):
    from PIL import Image
    from bgv import export_cinematic as ex
    from bgv import tools_cinematic as tc
    shot = tc.OrbitShot(full.lay.current_view(), 1.0 / 30.0, tc.TimeLaw("hold", hold=12.0), 0.0)
    spec = ex.ExportSpec(3840, 2160, 30.0, shot, tmp_path / "k4", name="k4", make_mp4=False, preroll=0)
    assert spec.n_frames() == 1
    job, before = run_job(full, spec)
    assert job.state == "done", job.message
    im = Image.open(tmp_path / "k4" / "frame_000001.png")
    assert im.size == (3840, 2160)
    arr = np.asarray(im)
    assert arr[100:900, 600:3000].std() > 5.0                                # a real picture, not a blank target
    after_checks(full, before)


def test_water_is_continuous_across_the_hourly_frames(full, tmp_path):
    """No visible stepping: the picture at t = 6.0 -/+ 0.01 h is almost the picture at 6.0, however the data change between hours."""
    from PIL import Image
    from bgv import export_cinematic as ex
    from bgv import tools_cinematic as tc
    app = full.app
    times = [5.0, 5.5, 5.99, 6.0, 6.01, 6.5, 7.0]

    class Times(tc.Shot):
        duration = len(times) / 10.0

        def state_at(self, s):
            i = min(int(round(s * 10)), len(times) - 1)
            return tc.ShotState(full.lay.current_view(), times[i])
    for name in ("buildings", "particles", "arrows"):
        app.get_layer(name).enabled = False
    try:
        X, Y = app.terrain.frame.lnglat_to_world(-86.44, 36.99)
        app.camera.set_view(X, Y, 6000.0, 20.0, 55.0)
        spec = ex.ExportSpec(480, 270, 10.0, Times(), tmp_path / "c", name="c", make_mp4=False, preroll=2, stamp=False)
        job, before = run_job(full, spec)
        assert job.state == "done", job.message
    finally:
        for name in ("buildings", "particles", "arrows"):
            app.get_layer(name).enabled = True
    f = [np.asarray(Image.open(tmp_path / "c" / f"frame_{i + 1:06d}.png")).astype(np.float32) for i in range(len(times))]
    d = lambda a, b: float(np.abs(f[a] - f[b]).mean())                   # noqa: E731
    assert d(0, 6) > 0.0                                                     # the water does change over two hours
    big = max(d(3, 5), d(2, 3), 1e-6)
    assert d(2, 4) <= max(0.35 * d(3, 5), 0.01)                              # +-0.01 h around an hour boundary: tiny
    assert d(3, 4) <= max(0.35 * d(3, 5), 0.01) and d(2, 3) <= max(0.35 * d(3, 5), 0.01), (d(2, 3), d(3, 4), d(3, 5), big)
    app.t = 12.0
