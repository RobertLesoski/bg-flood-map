"""Flow visualisation tests: CPU logic (fxcommon, arrow lattice, flowpaths ramps, state round trips) and offscreen GPU smoke
tests of the particle, arrow, flow-path and water layers on the real data (skipped when it is not present)."""
import math
import time
from pathlib import Path

import numpy as np
import pytest

from bgv import config
from bgv.data import flow as flowdata
from bgv.gpu.layers import arrows, flowpaths, fxcommon as fx


# ------------------------------------------------------------------------------------------------- CPU: fxcommon
def test_px_per_metre_scales_inversely_with_distance():
    fov = math.radians(45)
    a = fx.px_per_metre(100.0, fov, 1080)
    b = fx.px_per_metre(200.0, fov, 1080)
    assert a == pytest.approx(2 * b)
    assert a == pytest.approx(540 / (math.tan(fov / 2) * 100))


def test_velocity_multiplier_limits_and_monotonic():
    # close up: physical speed (x animation speed); far: stretched but bounded
    assert fx.velocity_multiplier(50.0, 1.0) == pytest.approx(1.0)
    assert fx.velocity_multiplier(1e-6, 1.0) == pytest.approx(160.0)
    assert fx.velocity_multiplier(0.1, 2.0) == pytest.approx(2.0 * min(24.0 / 0.1, 160.0))
    ms = [fx.velocity_multiplier(p, 1.0) for p in (30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03)]
    assert ms == sorted(ms)
    assert fx.velocity_multiplier(0.1, 1.5, constant_screen_speed=False) == pytest.approx(1.5)
    # on-screen speed of 1 m/s of flow: mult * ppm = px_per_ms in the stretched range
    ppm = 0.5
    assert fx.velocity_multiplier(ppm, 1.0) * ppm == pytest.approx(24.0)


def test_particle_target_clamps():
    assert fx.particle_target(0.0, 2e6, 0.0125, 1000.0, 250_000) == 0.0
    assert fx.particle_target(1e-6, 2e6, 0.0125, 1000.0, 250_000) == 250.0            # floor when anything is wet
    assert fx.particle_target(1.0, 2e6, 0.0125, 600.0, 250_000) == pytest.approx(25_000.0)
    assert fx.particle_target(1.0, 2e6, 1.0, 1000.0, 90_000) == 90_000.0              # pool cap
    near = fx.particle_target(0.1, 2e6, 0.0125, 500.0, 250_000)
    far = fx.particle_target(0.1, 2e6, 0.0125, 20000.0, 250_000)
    assert far == pytest.approx(near * 6.0)                                         # boosted when the whole map is in view


def test_far_alpha_and_quality_names():
    assert fx.far_alpha(100.0) == 1.0
    assert 0.6 < fx.far_alpha(30000.0) < 0.7
    assert fx.far_alpha(5000.0) >= fx.far_alpha(10000.0)

    class A: pass
    a = A()
    assert fx.quality_name(a) is None
    a.quality = "Medium"
    assert fx.quality_name(a) == "medium"
    a.quality = 0
    assert fx.quality_name(a) == "low"
    a.quality = 7
    assert fx.quality_name(a) == "high"
    assert set(fx.QUALITY) == {"low", "medium", "high"}
    assert fx.QUALITY["high"]["pool"] == 250_000


# ------------------------------------------------------------------------------------------------- CPU: arrow lattice
def _density(L, jmin=-3, jmax=12):
    """Expected shown arrows per unit area (world m2) at wanted exponent L: sum over levels of weight x level density."""
    d = 0.0
    for j in range(jmin, jmax + 1):
        dens = 4.0 ** (-j) if j == jmax else 0.75 * 4.0 ** (-j)          # level j = lattice 2^j minus lattice 2^(j+1)
        d += arrows.arrow_weight(j, L) * dens
    return d


def test_arrow_weight_partition():
    assert arrows.arrow_weight(5, 5.0) == 1.0
    assert arrows.arrow_weight(5, 6.0) == 0.0
    assert arrows.arrow_weight(5, 5.5) == pytest.approx(0.5)
    assert arrows.arrow_weight(7, 5.0) == 1.0
    # at integer L the union is exactly lattice 2^L; in between it moves continuously and monotonically
    for k in range(-2, 10):
        assert _density(float(k)) == pytest.approx(4.0 ** (-k), rel=1e-9)
    Ls = np.linspace(-2.0, 10.0, 481)
    ds = np.array([_density(L) for L in Ls])
    assert (np.diff(ds) <= 1e-15).all()                                                # density never increases with L
    assert (ds[:-1] / ds[1:]).max() < 1.09                                             # no jumps (smooth cross-fade)


def test_arrow_windows_cover_all_visible_points_and_are_anchored():
    f, vh, D = 1.0 / math.tan(math.radians(45) * 0.5), 1080.0, 78.0
    n = arrows.window_points(vh, f, D)
    assert 80 < n < 200
    # a point at horizontal distance r from the eye with view depth w <= r has weight > 0 only when
    # ppm > D / 2^(j+1), i.e. r*cos(offaxis) < f*vh/2 * 2^(j+1) / D ; the window half size n/2 * 2^j must cover that
    for j in range(-3, 13):
        reach_max = 1.35 * (f * vh * 0.5) * 2.0 / D * 2.0 ** j                   # largest horizontal distance with weight > 0
        assert n * 0.5 * 2.0 ** j >= reach_max
    # anchoring: the window origin is an integer lattice index and moves only in whole steps
    w0 = arrows.level_windows(1234.5, -987.25, -2, 8, n)
    w1 = arrows.level_windows(1234.5 + 0.001, -987.25 - 0.001, -2, 8, n)
    assert w0 == w1 or sum(a != b for a, b in zip(w0, w1)) <= 1
    for j, ix0, iy0, top in w0:
        s = 2.0 ** j
        assert abs((ix0 + n // 2) * s - 1234.5) <= s and abs((iy0 + n // 2) * s + 987.25) <= s
        assert isinstance(ix0, int) and isinstance(iy0, int)
        assert top == (1 if j == 8 else 0)
    # moving the eye by exactly one cell of level j shifts that window by exactly one index
    s = 2.0 ** 3
    a = arrows.level_windows(100.0, 50.0, 3, 3, n)[0]
    b = arrows.level_windows(100.0 + s, 50.0, 3, 3, n)[0]
    assert b[1] == a[1] + 1 and b[2] == a[2]


def test_arrow_level_range():
    f = 1.0 / math.tan(math.radians(45) * 0.5)
    n = arrows.window_points(1080, f, 78.0)
    # high above the ground: no fine levels needed; low camera needs finer ones
    hi = arrows.level_range(78.0, f, 1080, eye_height=9000.0, pitch_deg=55.0, fov_rad=math.radians(45), max_dist=25000.0, n_window=n)
    lo = arrows.level_range(78.0, f, 1080, eye_height=30.0, pitch_deg=20.0, fov_rad=math.radians(45), max_dist=25000.0, n_window=n)
    assert lo[0] < hi[0]
    assert arrows.J_MIN <= lo[0] <= lo[1] <= arrows.J_MIN + arrows.MAX_LEVELS - 1
    assert n * 0.5 * 2.0 ** hi[1] >= 25000.0                                          # coarsest window reaches the far corner


# ------------------------------------------------------------------------------------------------- CPU: flowpaths + state
def test_flowpaths_decode_matches_data_layer():
    for code in (0, 1, 40, 128, 255):
        assert flowpaths.decode_value("depth", code) == pytest.approx(float(flowdata.decode_depth(code)), rel=1e-6)
        assert flowpaths.decode_value("discharge", code) == pytest.approx(float(flowdata.decode_q(code)), rel=1e-6)
        assert flowpaths.decode_value("hours", code) == pytest.approx(code / 255.0 * 24.0)


def test_flowpaths_ramps():
    for mode in flowpaths.MODES:
        stops = flowpaths.RAMPS[mode]
        assert len(stops) == 6 and all(len(c) == 3 and all(0 <= v <= 1 for v in c) for c in stops)
        assert flowpaths.ramp_colour(mode, 0.0) == pytest.approx(stops[0])
        assert flowpaths.ramp_colour(mode, 1.0) == pytest.approx(stops[-1])
        assert flowpaths.ramp_colour(mode, 2.0) == pytest.approx(stops[-1])
        mid = flowpaths.ramp_colour(mode, 0.5)
        assert all(min(a, b) - 1e-9 <= m <= max(a, b) + 1e-9 for m, a, b in zip(mid, stops[2], stops[3]))
    assert len(flowpaths.MODE_LABELS) == len(flowpaths.MODES) == 3


def test_layer_state_round_trips():
    from bgv.gpu.layers.particles import ParticlesLayer
    from bgv.gpu.layers.terrain import WaterLayer
    for cls, change in ((ParticlesLayer, {"preset": "low", "density": 2.5, "anim_speed": 3.0, "trail": 1.7, "width_px": 2.5,
                                            "alpha": 0.4, "colour": 2, "const_speed": False}),
                        (arrows.ArrowsLayer, {"spacing_px": 120.0, "size": 1.6, "alpha": 0.5, "colour": 1, "full_speed": 3.0}),
                        (flowpaths.FlowpathsLayer, {"mode": 2, "alpha": 0.3}),
                        (WaterLayer, {"ripple": 0.4, "depth_boost": 3.0, "opacity": 0.6, "absorption": 4.0, "turbidity": 0.8,
                                      "smooth_shore": False})):
        a, b = cls(), cls()
        a.load_state(change)
        st = a.save_state()
        for k, v in change.items():
            assert st[k] == v, (cls.__name__, k)
        b.load_state(st)
        assert b.save_state() == st
        b.load_state({})                                   # empty / partial state keeps the values
        assert b.save_state() == st
        b.load_state({"nonsense": 1})
        assert b.save_state() == st
    assert flowpaths.FlowpathsLayer.default_enabled is False


def test_registry_lists_the_fx_layers_in_draw_order():
    from bgv.gpu.layers import registry
    names = [s.split(":")[1] for s in registry.LAYERS]
    assert names.index("WaterLayer") < names.index("FlowpathsLayer") < names.index("ParticlesLayer") < names.index("ArrowsLayer")


# ------------------------------------------------------------------------------------------------- GPU smoke tests
wgpu = pytest.importorskip("wgpu")

HOT = (-3443.0, 1247.0)          # moving water in the 25yr storm (tiers data)


@pytest.fixture(scope="module")
def ctx():
    from bgv.gpu.context import GpuContext
    try:
        return GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")


@pytest.fixture(scope="module")
def scene(ctx):
    root = config.DEFAULT_DATA_ROOT
    if not (Path(root) / "layers" / "flow" / "25yr" / "meta.json").is_file():
        pytest.skip("real data not present")
    from bgv.app import App
    from bgv import tools
    app = App(ctx, config.data_root(), storm="25yr", msaa=4, quiet=True, settings={})
    app.t = 12.0
    app.camera.set_view(HOT[0], HOT[1], 600.0, 30.0, 45.0)
    tools.wait_for_flow(app)
    w, h = 480, 270
    tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    s = type("Scene", (), {})()
    s.app, s.tex, s.view, s.w, s.h, s.ctx = app, tex, tex.create_view(), w, h, ctx

    def frames(n, dt=1 / 60):
        for _ in range(n):
            app.anim_time += dt
            app.step(dt)
            app.render_scene(s.view, w, h, dt)
        ctx.sync()

    def grab():
        raw = ctx.queue.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                     {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3].astype(np.int32).copy()
    s.frames, s.grab = frames, grab
    yield s
    app.shutdown()


def _ndc(app, xyz):
    vp = np.asarray(app.view_proj, np.float64)
    c = np.concatenate([xyz, np.ones((len(xyz), 1))], 1) @ vp.T
    return c[:, :2] / c[:, 3:4], c[:, 3]


@pytest.mark.gpu
def test_particles_live_on_wet_moving_cells_inside_the_view(scene):
    app, P = scene.app, scene.app.get_layer("particles")
    assert P is not None and not app.errors
    scene.frames(60)
    active, wet_frac, parts = P.read_state(app, 20000)
    assert active >= 250 and 0.01 < wet_frac <= 1.0
    alive = parts[(parts[:, 6] > 0) & (parts[:, 7] > 0.05)]
    assert len(alive) > 100
    ndc, w = _ndc(app, np.column_stack([alive[:, 0], alive[:, 1], alive[:, 4]]))
    assert (w > 0).all() and (np.abs(ndc) <= 1.12).all()                        # respawned inside the camera footprint
    rng = np.random.default_rng(3)
    pick = rng.choice(len(alive), size=min(150, len(alive)), replace=False)
    bad = 0
    for i in pick:
        s = app.storm.sample_t(app.t, float(alive[i, 0]), float(alive[i, 1]))
        if not (s["valid"] and s["depth"] >= 0.03 and s["speed"] >= 0.015):       # GPU/CPU bilinear + particle step slack
            bad += 1
    assert bad <= 0.05 * len(pick)
    assert (np.hypot(alive[:, 2], alive[:, 3]) >= 0.02).all()
    assert (alive[:, 5] <= alive[:, 6] + 1e-3).all()                            # age <= life
    assert active <= fx.QUALITY["high"]["pool"]


@pytest.mark.gpu
def test_particles_keep_moving_when_time_is_frozen(scene):
    app, P = scene.app, scene.app.get_layer("particles")
    app.playing = False
    scene.frames(5)
    _, _, p0 = P.read_state(app, 4000)
    scene.frames(4)
    _, _, p1 = P.read_state(app, 4000)
    both = (p0[:, 6] > 0) & (p1[:, 6] > 0) & (p0[:, 5] < p1[:, 5])             # same particle, still alive, aged
    assert both.sum() > 100
    moved = np.hypot(p1[both, 0] - p0[both, 0], p1[both, 1] - p0[both, 1])
    assert np.median(moved) > 0.05                                              # frozen flow time, advecting particles


@pytest.mark.gpu
def test_particles_follow_camera_teleport_and_hour_change(scene):
    app, P = scene.app, scene.app.get_layer("particles")
    # teleport to another wet area: after ONE frame no live particle may remain outside the new view
    app.camera.set_view(-4752.0, 6483.0, 500.0, 200.0, 40.0)
    scene.frames(2)
    active, wet_frac, parts = P.read_state(app, 20000)
    alive = parts[(parts[:, 6] > 0) & (parts[:, 7] > 0.0)]
    assert len(alive) > 100
    ndc, w = _ndc(app, np.column_stack([alive[:, 0], alive[:, 1], alive[:, 4]]))
    inside = (w > 0) & (np.abs(ndc) <= 1.12).all(axis=1)
    assert inside.mean() > 0.98
    # hour change: particles are still on wet moving cells at the new hour
    app.t = 14.0
    scene.frames(30)
    _, _, parts = P.read_state(app, 20000)
    alive = parts[(parts[:, 6] > 0) & (parts[:, 7] > 0.05)]
    if len(alive):
        pick = np.random.default_rng(1).choice(len(alive), size=min(100, len(alive)), replace=False)
        bad = sum(1 for i in pick if not app.storm.sample_t(app.t, float(alive[i, 0]), float(alive[i, 1]))["depth"] > 0.02)
        assert bad <= 0.08 * len(pick)
    app.camera.set_view(HOT[0], HOT[1], 600.0, 30.0, 45.0)
    app.t = 12.0
    scene.frames(30)


@pytest.mark.gpu
def test_arrows_only_on_wet_flowing_water_and_capped(scene):
    app, A = scene.app, scene.app.get_layer("arrows")
    scene.frames(3)
    n = A.read_count(app)
    assert 1 <= n <= 30_000
    raw = np.frombuffer(app.ctx.queue.read_buffer(A.buf, 0, min(n, 30_000) * arrows.INST_BYTES), np.float32).reshape(-1, 8)
    ndc, w = _ndc(app, np.column_stack([raw[:, 0], raw[:, 1], raw[:, 3]]))
    assert (w > 0).all() and (np.abs(ndc) <= 1.12).all()
    assert (raw[:, 5] > 0).all() and (raw[:, 5] <= 1.0 + 1e-6).all()            # alpha
    assert (raw[:, 4] >= 0.05 - 1e-4).all()                                      # speed >= 0.05 m/s
    sel = np.random.default_rng(2).choice(len(raw), size=min(40, len(raw)), replace=False)
    for i in sel:
        s = app.storm.sample_t(app.t, float(raw[i, 0]), float(raw[i, 1]))
        assert s["depth"] >= 0.08 and s["speed"] >= 0.04


@pytest.mark.gpu
def test_arrow_lattice_is_anchored_when_panning(scene):
    """The same lattice point keeps its world position while the camera pans (positions are on the 2^j lattice)."""
    app, A = scene.app, scene.app.get_layer("arrows")
    scene.frames(2)
    n = A.read_count(app)
    raw = np.frombuffer(app.ctx.queue.read_buffer(A.buf, 0, n * arrows.INST_BYTES), np.float32).reshape(-1, 8)
    on_lattice = 0
    for x, y in raw[:, :2]:
        for j in range(-3, 13):
            s = 2.0 ** j
            if abs(x / s - round(x / s)) < 1e-4 and abs(y / s - round(y / s)) < 1e-4:
                on_lattice += 1
                break
    assert on_lattice == len(raw)


@pytest.mark.gpu
def test_flowpaths_overlay_loads_lazily_and_changes_the_picture(scene):
    app, FP = scene.app, scene.app.get_layer("flowpaths")
    assert FP is not None and FP.enabled is False                                # off by default
    app.camera.set_view(-1500.0, 2000.0, 7000.0, 20.0, 55.0)
    scene.frames(3)
    assert FP._loaded_for is None and FP.n_chunks == 0                          # nothing loaded while disabled
    base = scene.grab()
    FP.enabled = True
    t0 = time.time()
    while time.time() - t0 < 60:
        scene.frames(2)
        if FP._loaded_for == app.storm.name and FP.n_chunks > 0:
            break
        time.sleep(0.05)
    assert FP._loaded_for == app.storm.name and FP.n_chunks > 0 and not FP.error
    scene.frames(2)
    on = scene.grab()
    assert np.abs(on - base).mean() > 1.0
    for mode in (1, 2):
        FP.mode = mode
        scene.frames(2)
        assert np.abs(scene.grab() - on).mean() > 0.3                            # each mode has its own colours
    FP.mode = 0
    FP.enabled = False
    app.camera.set_view(HOT[0], HOT[1], 600.0, 30.0, 45.0)
    scene.frames(2)
    assert not app.errors


@pytest.mark.gpu
def test_water_shoreline_options_render_and_differ(scene):
    app, W = scene.app, scene.app.get_layer("water")
    P, A = app.get_layer("particles"), app.get_layer("arrows")
    P.enabled = A.enabled = False
    try:
        W.smooth_shore = True
        scene.frames(3)
        a = scene.grab()
        W.smooth_shore = False
        scene.frames(3)
        b = scene.grab()
        assert np.abs(a - b).mean() > 0.2
        assert a.std() > 8 and b.std() > 8
        W.smooth_shore = True
        W.opacity = 0.3
        scene.frames(3)
        c = scene.grab()
        assert np.abs(a - c).mean() > 0.5                                        # opacity option is wired
    finally:
        W.opacity = 1.0
        W.smooth_shore = True
        P.enabled = A.enabled = True
    assert not app.errors


@pytest.mark.gpu
def test_water_is_temporally_stable_between_consecutive_frames(scene):
    """Two renders 1/60 s apart in animation time may differ only a little (no shimmering): small mean change."""
    app = scene.app
    app.camera.set_view(HOT[0], HOT[1], 250.0, 30.0, 30.0)
    P, A = app.get_layer("particles"), app.get_layer("arrows")
    P.enabled = A.enabled = False
    try:
        app.anim_time = 4.0
        app.step(0.0)
        app.render_scene(scene.view, scene.w, scene.h, 0.0)
        a = scene.grab()
        app.anim_time = 4.0 + 1 / 60
        app.render_scene(scene.view, scene.w, scene.h, 0.0)
        b = scene.grab()
    finally:
        P.enabled = A.enabled = True
    d = np.abs(a - b)
    assert d.mean() < 0.5
    assert (d > 40).mean() < 0.01
    app.camera.set_view(HOT[0], HOT[1], 600.0, 30.0, 45.0)


@pytest.mark.gpu
def test_quality_attribute_limits_pool_and_arrows(scene):
    app = scene.app
    P, A = app.get_layer("particles"), app.get_layer("arrows")
    app.quality = "low"
    try:
        scene.frames(40)
        assert P._pool_cap == fx.QUALITY["low"]["pool"]
        active, _, _ = P.read_state(app, 4)
        assert active <= fx.QUALITY["low"]["pool"]
        assert A.read_count(app) <= fx.QUALITY["low"]["arrows"] + 1
    finally:
        del app.quality
    scene.frames(2)
    assert P._pool_cap == fx.QUALITY["low"]["pool"]                              # last applied preset stays until a new one
    assert not app.errors


@pytest.mark.gpu
def test_storm_change_keeps_particles_and_arrows_alive(scene):
    from bgv import tools
    app, P, A = scene.app, scene.app.get_layer("particles"), scene.app.get_layer("arrows")
    others = [s for s in app.storms_available if s != app.storm.name]
    if not others:
        pytest.skip("only one storm")
    first = app.storm.name
    app.set_storm(others[0])
    scene.frames(3)                                   # water is gone until the hours decode: must not crash or leave junk
    tools.wait_for_flow(app)
    scene.frames(80)
    active, wet_frac, parts = P.read_state(app, 20000)
    alive = parts[(parts[:, 6] > 0) & (parts[:, 7] > 0.05)]
    assert not app.errors
    if wet_frac > 0.005:
        assert len(alive) > 30
        pick = np.random.default_rng(5).choice(len(alive), size=min(80, len(alive)), replace=False)
        bad = sum(1 for i in pick if not app.storm.sample_t(app.t, float(alive[i, 0]), float(alive[i, 1]))["depth"] > 0.02)
        assert bad <= 0.08 * len(pick)
    app.set_storm(first)
    tools.wait_for_flow(app)
    scene.frames(30)
    assert not app.errors
