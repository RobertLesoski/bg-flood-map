"""Performance machinery (bgv/perf.py, bgv/gpu/fastcull.py, shared frustum, pooled decode, staged start-up).

Pure-CPU tests run anywhere; the `gpu` ones use the offscreen canvas and the real data root (skipped when absent)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from bgv import config

wgpu = pytest.importorskip("wgpu")


# ------------------------------------------------------------------------------------------------ pure CPU
def test_frustum_planes_match_the_stack_formula_and_contain_the_target():
    from bgv.gpu.camera import OrbitCamera
    from bgv.gpu.cull import frustum_planes
    cam = OrbitCamera()
    cam.resize(1600, 900)
    cam.set_view(100.0, 200.0, 3000.0, 30.0, 45.0)
    vp = cam.view_proj()
    r0, r1, r2, r3 = vp
    ref = np.stack([r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2])
    ref /= np.linalg.norm(ref[:, :3], axis=1, keepdims=True)
    assert np.allclose(frustum_planes(vp), ref, atol=1e-12)
    assert (frustum_planes(vp) @ np.append(cam.target, 1.0) >= -1e-6).all()


def test_planes_for_uses_the_app_cache_once_per_view_projection():
    from bgv.gpu.cull import planes_for

    class FakeApp:
        def __init__(self):
            self.view_proj = np.eye(4)
            self.calls = 0

        def frustum(self):
            self.calls += 1
            return np.zeros((5, 4))
    a = FakeApp()
    planes_for(a)
    planes_for(a, a.view_proj)
    assert a.calls == 2                                         # the app owns caching; the helper just delegates
    other = np.eye(4) * 2.0
    assert planes_for(a, other).shape == (5, 4) and a.calls == 2   # a different matrix is computed directly


def test_tile_kernels_match_the_numpy_reference():
    from bgv.gpu import fastcull
    from bgv.gpu.cull import frustum_planes
    from bgv.gpu.layers.buildings import merge_runs
    if not fastcull.available():
        pytest.skip("numba missing")
    rng = np.random.default_rng(7)
    n = 400

    class T:                                                    # the Tiles fields the culler / merge use
        pass
    t = T()
    t.n = n
    t.x0 = rng.uniform(-6000, 6000, n)
    t.x1 = t.x0 + rng.uniform(50, 400, n)
    t.y0 = rng.uniform(-6000, 6000, n)
    t.y1 = t.y0 + rng.uniform(50, 400, n)
    cnt = rng.integers(30, 3000, n).astype(np.int64)
    t.count = cnt
    t.first = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    from bgv.gpu.camera import OrbitCamera
    cam = OrbitCamera()
    cam.resize(1600, 900)
    for seed, (dist, yaw, pitch) in enumerate([(3000, 30, 45), (9000, 200, 60), (800, 90, 20), (14000, 0, 55)]):
        cam.set_view(float(rng.uniform(-2000, 2000)), float(rng.uniform(-2000, 2000)), dist, yaw, pitch)
        P = frustum_planes(cam.view_proj())
        ctr = np.stack([(t.x0 + t.x1) / 2, (t.y0 + t.y1) / 2, np.zeros(n)], 1)
        ext = np.stack([(t.x1 - t.x0) / 2, (t.y1 - t.y0) / 2, np.full(n, 30.0)], 1)
        vis = np.zeros(n, bool)
        fastcull.tile_visible(ctr, ext, P, vis)
        ref = ((ctr @ P[:, :3].T + ext @ np.abs(P[:, :3]).T + P[:, 3]) >= 0).all(axis=1)
        assert (vis == ref).all()
        for max_draws in (48, 5):
            of, oc = np.zeros(n + 1, np.int64), np.zeros(n + 1, np.int64)
            r = fastcull.tile_runs(vis, t.first, t.count, max_draws, of, oc)
            got = list(zip(of[:r].tolist(), oc[:r].tolist()))
            want = merge_runs(t, vis, max_draws=max_draws)
            assert got == want, (seed, max_draws)


def test_gc_control_freezes_disables_and_collects_young_generation_on_a_timer():
    import gc
    from bgv.perf import GcControl
    g = GcControl()
    was_enabled = gc.isenabled()
    try:
        g.enter_frames()
        assert g.active and not gc.isenabled()
        g._t_young = time.perf_counter() - 3.0
        n0 = g.collections
        g.tick()
        assert g.collections == n0 + 1                          # a young collection when two seconds passed
        g.tick()
        assert g.collections == n0 + 1                          # ... and not again at once
        g.leave_frames()
        assert gc.isenabled()
    finally:
        if was_enabled:
            gc.enable()
        try:
            gc.unfreeze()
        except Exception:
            pass


class _FakeApp:
    def __init__(self):
        from types import SimpleNamespace
        self.quality = None
        self.render_scale = 1.0
        self.gov_lod = 1.0
        self.frame = SimpleNamespace(q=[26.0, 8.0, 0.0])


def _drive(gov, frame_ms, gpu_ms, seconds):
    """Feed constant frames and move the governor's clock forward (it measures with perf_counter); the fake clock is
    continuous across calls so dwell times behave like real time."""
    import bgv.perf as perf
    real = perf.time.perf_counter
    t = [getattr(gov, "_fake_t", real())]
    perf.time.perf_counter = lambda: t[0]
    try:
        for _ in range(max(int(seconds * 1000.0 / frame_ms), 1)):
            t[0] += frame_ms / 1000.0
            gov.observe(frame_ms, gpu_ms)
    finally:
        perf.time.perf_counter = real
        gov._fake_t = t[0]


def test_governor_degrades_on_slow_frames_recovers_slowly_and_never_in_manual_mode():
    from bgv.perf import LEVELS, QualityGovernor
    app = _FakeApp()
    gov = QualityGovernor(app, "auto")
    assert gov.level == 3 and app.quality == "high" and app.render_scale == 1.0
    _drive(gov, 16.7, 6.0, 5.0)                                  # healthy: stays at the top
    assert gov.level == 3 and gov.changes == 0
    _drive(gov, 30.0, 25.0, 1.0)                                 # a second of 33 fps: one level down at least
    assert gov.level <= 2 and app.quality in ("medium", "low")
    _drive(gov, 30.0, 25.0, 6.0)
    assert gov.level == 0 and app.render_scale == LEVELS[0][4] < 1.0 and app.frame.q[0] == LEVELS[0][1]
    lvl = gov.level
    _drive(gov, 16.7, 20.0, 12.0)                                # fast frames but a heavy GPU: no promotion
    assert gov.level == lvl
    _drive(gov, 16.7, 4.0, 5.0)                                  # light GPU: up one level (not straight to the top)
    assert gov.level == lvl + 1
    _drive(gov, 16.7, 4.0, 40.0)
    assert gov.level == 3 and app.render_scale == 1.0 and app.quality == "high"
    # manual mode: the governor stands down and restores full resolution
    _drive(gov, 30.0, 25.0, 3.0)
    gov.set_mode("manual")
    assert gov.mode == "manual" and gov.level == 3 and app.render_scale == 1.0
    n = gov.changes
    _drive(gov, 40.0, 35.0, 8.0)
    assert gov.changes == n and gov.level == 3
    gov.set_mode("auto")
    assert gov.mode == "auto" and gov.level == 3


def test_governor_ignores_isolated_stalls():
    from bgv.perf import QualityGovernor
    app = _FakeApp()
    gov = QualityGovernor(app, "auto")
    _drive(gov, 16.7, 6.0, 2.0)
    for _ in range(5):                                           # five 70 ms hitches among healthy frames
        _drive(gov, 70.0, 6.0, 0.07)
        _drive(gov, 16.7, 6.0, 1.0)
    assert gov.level == 3


def test_gunzip_into_matches_gunzip_exact_and_rejects_bad_files(tmp_path):
    import gzip
    from bgv.data import flow
    rng = np.random.default_rng(3)
    n = 3 * 1024 * 1024 + 123                                    # not a multiple of the 1 MB piece size
    data = (rng.integers(0, 4, n, dtype=np.uint8) * 60).tobytes()
    p = tmp_path / "h.rgba.gz"
    p.write_bytes(gzip.compress(data, 1))
    out = np.empty(n, np.uint8)
    flow.gunzip_into(p, out)
    assert out.tobytes() == flow.gunzip_exact(p, n) == data
    with pytest.raises(flow.FlowFormatError):
        flow.gunzip_into(p, np.empty(n + 4, np.uint8))           # wrong size: caught by the gzip trailer
    bad = tmp_path / "bad.gz"
    bad.write_bytes(b"not gzip at all" * 4)
    with pytest.raises(flow.FlowFormatError):
        flow.gunzip_into(bad, np.empty(8, np.uint8))
    trunc = tmp_path / "trunc.gz"
    trunc.write_bytes(p.read_bytes()[:-200] + p.read_bytes()[-4:])
    with pytest.raises(flow.FlowFormatError):
        flow.gunzip_into(trunc, np.empty(n, np.uint8))


def test_cffi_type_parsing_speedup_is_safe_and_idempotent():
    from bgv.gpu import cffiwarm
    assert cffiwarm.speedup() in (True, False)
    import wgpu.backends.wgpu_native._api as api
    t0 = time.perf_counter()
    x = api.ffi.new("WGPUBufferDescriptor *")
    assert "WGPUBufferDescriptor" in repr(x)
    cffiwarm.warm()
    assert time.perf_counter() - t0 < 5.0
    assert cffiwarm.speedup() is cffiwarm._patched


# ------------------------------------------------------------------------------------------------ GPU
@pytest.fixture(scope="module")
def ctx():
    pytest.importorskip("numba")
    from bgv.gpu.context import GpuContext
    try:
        return GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")


@pytest.fixture(scope="module")
def app(ctx):
    from pathlib import Path
    from bgv import tools
    from bgv.app import App
    root = config.data_root()
    if not (Path(root) / "layers" / "flow" / "25yr" / "meta.json").is_file():
        pytest.skip("real data not present")
    a = App(ctx, root, storm="25yr", msaa=4, quiet=True, settings={})
    tools.wait_for_flow(a, all_hours=False)
    yield a
    a.shutdown()


class _Queue:
    """Records write_buffer calls of a TerrainMesh (offset 0 only) instead of sending them."""

    def __init__(self):
        self.writes = {}

    def write_buffer(self, buf, off, data):
        self.writes[id(buf)] = np.array(data, np.uint32).copy()


class _Ctx:
    def __init__(self, real, q):
        self.device, self.queue = real.device, q
        self._real = real

    def track(self, *a):
        pass

    def untrack(self, *a):
        pass


def test_terrain_fast_update_equals_the_reference_for_many_cameras(app):
    from bgv.gpu.camera import OrbitCamera
    from bgv.gpu.layers.terrain import TerrainMesh
    from bgv.data.flow import hour_pair
    mesh = app.terrain_mesh
    rng = np.random.default_rng(11)
    cam = OrbitCamera()
    cam.resize(1920, 1080)
    cam.set_ground(app.terrain.sampler, app.exag)
    real_ctx = app.ctx
    q = _Queue()
    mesh.app.ctx = _Ctx(real_ctx, q)
    try:
        for i in range(40):
            cam.set_view(float(rng.uniform(-4000, 4000)), float(rng.uniform(-4000, 4000)), float(rng.choice([300, 900, 3000, 9000, 16000])),
                         float(rng.uniform(0, 360)), float(rng.uniform(8, 80)))
            eye, vp = cam.eye(), cam.view_proj()
            wet = app.flow.chunk_wet(*hour_pair(float(rng.uniform(4, 20))))
            ld = float(rng.choice([450.0, 1100.0, 1800.0]))
            exag = float(rng.choice([1.0, 2.0, 5.0]))
            q.writes.clear()
            mesh.update(eye, vp, exag, ld, wet, 30.0)
            fast = (list(mesh.draws_terrain), list(mesh.draws_water), mesh.n_visible, mesh.n_water, mesh.vis.copy(),
                    q.writes.get(id(mesh.inst_terrain)), q.writes.get(id(mesh.inst_water)), mesh.tris, mesh.tris_water)
            pk = mesh.pack_visible(wet & mesh.vis)
            q.writes.clear()
            mesh._ckey = None                                    # force the reference to recompute (it has no cache)
            TerrainMesh.update_reference(mesh, eye, vp, exag, ld, wet, 30.0)
            ref = (list(mesh.draws_terrain), list(mesh.draws_water), mesh.n_visible, mesh.n_water, mesh.vis.copy(),
                   q.writes.get(id(mesh.inst_terrain)), q.writes.get(id(mesh.inst_water)), mesh.tris, mesh.tris_water)
            assert fast[0] == ref[0] and fast[1] == ref[1], i
            assert fast[2] == ref[2] and fast[3] == ref[3] and fast[7] == ref[7] and fast[8] == ref[8], i
            assert (fast[4] == ref[4]).all(), i
            for a, b in ((fast[5], ref[5]), (fast[6], ref[6])):
                assert (a is None) == (b is None)
                if a is not None:
                    assert np.array_equal(a.reshape(-1, 4), b.reshape(-1, 4)), i
            assert pk[1] == ref[1] and np.array_equal(pk[0], ref[6].reshape(-1, 4) if ref[6] is not None else np.zeros((0, 4), np.uint32)), i
            mesh.__dict__.pop("pack_visible", None)               # the reference installs an instance attribute; drop it
    finally:
        mesh.app.ctx = real_ctx
        mesh._ckey = None


def test_terrain_update_is_cached_when_nothing_changes_and_reuploads_when_it_does(app):
    mesh = app.terrain_mesh
    cam = app.camera
    cam.set_view(0.0, 0.0, 5000.0, 20.0, 50.0)
    app.step(0.0)
    lay = app.get_layer("terrain")
    lay.update(app, 0.0)
    first = mesh.cull_ms
    lay.update(app, 0.0)
    assert mesh.cull_ms == 0.0 and first > 0.0                   # identical camera / wet table: nothing recomputed
    cam.set_view(0.0, 0.0, 5200.0, 20.0, 50.0)
    app.step(0.0)
    lay.update(app, 0.0)
    assert mesh.cull_ms > 0.0


def test_frame_uniforms_are_stable_and_carry_the_governor_knobs(app):
    f = app.frame
    app.step(0.0)
    app.write_frame(1280, 720)
    a = f.f.copy()
    app.write_frame(1280, 720)
    assert np.array_equal(a, f.f)                                # cached view-projection / layer map: same words
    assert f.f[64 + 25] == pytest.approx(26.0) and f.f[64 + 26] == pytest.approx(8.0)
    f.q[0], f.q[1] = 12.0, 3.0
    app.write_frame(1280, 720)
    assert f.f[64 + 25] == pytest.approx(12.0) and f.f[64 + 26] == pytest.approx(3.0)
    f.q[0], f.q[1] = 26.0, 8.0
    app.write_frame(1280, 720)
    assert np.array_equal(f.u[64:64 + 25], app.flow.layer_map)    # the hour -> layer table itself is untouched


def _render(app, ctx, w, h, scale):
    app.render_scale = scale
    tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    for name in ("particles", "arrows"):
        lay = app.get_layer(name)
        if lay is not None:
            lay.enabled = False
    app.anim_time = 2.0
    app.step(0.0)
    app.render_scene(tex.create_view(), w, h, 0.0)
    raw = ctx.queue.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                 {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4).astype(np.int32)


def test_reduced_render_scale_blits_a_close_image_at_full_window_size(app, ctx):
    app.perf.gov.set_mode("manual")
    app.camera.set_view(0.0, 0.0, 6000.0, 30.0, 50.0)
    try:
        full = _render(app, ctx, 640, 360, 1.0)
        half = _render(app, ctx, 640, 360, 0.7)
        assert half.shape == full.shape == (360, 640, 4)
        assert app._target_size == app.perf.scene_size(640, 360) and app._target_size[0] < 640   # drawn at 70 %
        assert half[..., :3].std() > 8
        assert np.abs(full[..., :3] - half[..., :3]).mean() < 12.0   # same picture, a little softer
        assert not app.errors
    finally:
        app.render_scale = 1.0
        app.perf.gov.set_mode("auto")
        for name in ("particles", "arrows"):
            lay = app.get_layer(name)
            if lay is not None:
                lay.enabled = True


def test_deferred_start_up_builds_the_same_app_in_steps_behind_a_loading_screen(ctx):
    from pathlib import Path
    from PIL import Image  # noqa: F401  (pillow is a hard dependency of the viewer)
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import tools
    from bgv.app import App
    from bgv.startup import Prefetch
    root = config.data_root()
    if not (Path(root) / "layers" / "flow" / "25yr" / "meta.json").is_file():
        pytest.skip("real data not present")
    canvas = OffscreenRenderCanvas(size=(640, 360), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    pre = Prefetch(root, "25yr", {})
    a = App(ctx, root, storm="25yr", msaa=4, quiet=True, settings={}, colour_format=fmt, defer=True, prefetch=pre)
    try:
        assert a.loading and a.layers == [] and a.flow is None     # nothing heavy happened in the constructor
        a.attach_canvas(canvas, ui=True)
        canvas.set_physical_size(640, 360)
        canvas.request_draw(a.draw_frame)
        fracs, t0 = [], time.perf_counter()
        while a.loading and time.perf_counter() - t0 < 120:
            canvas.draw()
            fracs.append(a.load_frac)
        assert not a.loading and a.load_frac == 1.0
        assert fracs == sorted(fracs) and len(fracs) >= 5          # several frames, progress never goes back
        names = [l.name for l in a.layers]
        assert names[:2] == ["basemap", "terrain"] and "water" in names and "overlay" in names
        assert not a.errors, list(a.errors)
        tools.wait_for_flow(a, all_hours=False)
        a.camera.set_view(0.0, 0.0, 6000.0, 30.0, 50.0)
        canvas.draw()
        img = np.asarray(canvas.draw())
        assert img.shape[:2] == (360, 640) and img[..., :3].std() > 8 and not a.errors
    finally:
        a.shutdown()


def test_flow_decode_buffers_are_reused_and_hours_still_upload_correctly(ctx, tmp_path):
    from bgv.data import flow
    from bgv.gpu.flowtex import FlowTextures
    from test_flow import make_meta, write_storm
    w, h = 300, 200
    hours = (1, 2, 3, 4, 5, 6)
    meta = make_meta(w=w, h=h, hours=hours)
    rng = np.random.default_rng(5)
    grids = {hr: rng.integers(0, 256, (h, w, 4), dtype=np.uint8) for hr in hours}
    write_storm(tmp_path, meta, grids)
    storm = flow.Storm.open(tmp_path, "t")
    ft = FlowTextures(ctx, w, h)
    try:
        ft.set_storm(storm, t=3.0)
        t0 = time.perf_counter()
        while ft.progress()[0] < len(hours) and time.perf_counter() - t0 < 30:
            ft.pump()
            time.sleep(0.002)
        assert ft.progress()[0] == len(hours)
        assert len(ft._pooled) <= 3                                # six hours decoded through at most three buffers
        for hr in (1, 4, 6):
            raw = ctx.queue.read_texture({"texture": ft.texture, "mip_level": 0, "origin": (0, 0, int(ft.layer_map[hr]))},
                                         {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
            assert np.array_equal(np.frombuffer(raw, np.uint8).reshape(h, w, 4), grids[hr]), hr
        ft.set_storm(storm, t=2.0)                                 # switching mid-way releases every buffer again
        t0 = time.perf_counter()
        while ft.progress()[0] < len(hours) and time.perf_counter() - t0 < 30:
            ft.pump()
            time.sleep(0.002)
        assert ft.progress()[0] == len(hours) and len(ft._pooled) <= 3
    finally:
        ft.close()
