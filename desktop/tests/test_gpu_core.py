"""GPU core tests (offscreen, no window): flow texture array upload/readback, layer registry, scene smoke render."""
import numpy as np
import pytest

from bgv import config
from bgv.data import flow
from bgv.gpu.layers import Layer, registry

pytestmark = pytest.mark.gpu

wgpu = pytest.importorskip("wgpu")


@pytest.fixture(scope="module")
def ctx():
    from bgv.gpu.context import GpuContext
    try:
        return GpuContext.create(verbose=False)
    except Exception as e:                      # no adapter on this machine
        pytest.skip(f"no GPU adapter: {e}")


def test_adapter_is_discrete_vulkan(ctx):
    if ctx.adapter_type != "DiscreteGPU":
        pytest.skip("no discrete GPU here")
    assert ctx.backend == "Vulkan"
    assert ctx.limits["max-texture-array-layers"] >= 26


def test_layer_registry_orders_and_skips_missing():
    got = [spec.split(":")[1] for spec, _ in registry.load_layer_classes(lambda m: None)]
    assert got.index("TerrainLayer") < got.index("WaterLayer")
    assert all(issubclass(c, Layer) for _, c in registry.load_layer_classes(lambda m: None))


def _synthetic_storm(tmp_path, w=300, h=200):
    from test_flow import make_meta, write_storm
    meta = make_meta(w=w, h=h, hours=(2, 3, 5))
    rng = np.random.default_rng(1)
    grids = {}
    for hr in (2, 3, 5):
        g = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        g[..., 3] = 255
        grids[hr] = g
    write_storm(tmp_path, meta, grids)
    return flow.Storm.open(tmp_path, "t"), grids


def test_flow_texture_upload_matches_source(ctx, tmp_path):
    from bgv.gpu.flowtex import FlowTextures, ZERO_LAYER
    import time
    storm, grids = _synthetic_storm(tmp_path)
    ft = FlowTextures(ctx, storm.width, storm.height, upload_budget=100_000)    # many slices per hour
    try:
        ft.set_storm(storm, t=3.0)
        t0 = time.time()
        while ft.progress()[0] < 3 and time.time() - t0 < 20:
            ft.pump()
            time.sleep(0.005)
        assert ft.progress() == (3, 3)
        assert ft.layer_map[2] == 2 and ft.layer_map[3] == 3 and ft.layer_map[5] == 5
        assert ft.layer_map[1] == ZERO_LAYER and ft.layer_map[24] == ZERO_LAYER
        for hr in (2, 3, 5):                                                  # row-sliced uploads reassemble exactly
            back = np.frombuffer(ctx.queue.read_texture(
                {"texture": ft.texture, "mip_level": 0, "origin": (0, 0, hr)},
                {"offset": 0, "bytes_per_row": storm.width * 4, "rows_per_image": storm.height},
                (storm.width, storm.height, 1)), np.uint8).reshape(storm.height, storm.width, 4)
            assert np.array_equal(back, grids[hr])
        zero = np.frombuffer(ctx.queue.read_texture(
            {"texture": ft.texture, "mip_level": 0, "origin": (0, 0, ZERO_LAYER)},
            {"offset": 0, "bytes_per_row": storm.width * 4, "rows_per_image": storm.height},
            (storm.width, storm.height, 1)), np.uint8)
        assert not zero.any()                                                 # the shared dry layer stays zero
        # readiness: dry hours count as ready, hour 4 is dry (not in meta.hours)
        assert ft.ready_for(4.5) and ft.ready_for(2.3) and ft.ready_for(0.0)
        # chunk wet table follows the R channel
        wet = ft.chunk_wet(2, 3, 0.5)
        assert wet.shape == (ft.ny, ft.nx) and wet.any()
        # switching storm resets everything to dry first, then reloads
        ft.set_storm(storm, t=5.0)
        assert (ft.layer_map == ZERO_LAYER).all() and not ft.ready_for(2.5)
    finally:
        ft.close()


def test_scene_renders_deterministically_on_real_data(ctx):
    root = config.DEFAULT_DATA_ROOT
    from pathlib import Path
    if not (Path(root) / "layers" / "flow" / "25yr" / "meta.json").is_file():
        pytest.skip("real data not present")
    from bgv.app import App
    from bgv import tools
    app = App(ctx, config.data_root(), storm="25yr", msaa=4, quiet=True, settings={})
    try:
        app.t = 12.0
        app.camera.set_view(0, 0, 9000.0, 0.0, 55.0)
        tools.wait_for_flow(app)
        for name in ("particles", "arrows"):       # animated by design: a second frame legitimately differs
            lay = app.get_layer(name)
            if lay is not None:
                lay.enabled = False
        w, h = 320, 180
        tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        imgs = []
        for _ in range(2):
            app.anim_time = 2.0
            app.step(0.0)
            app.render_scene(tex.create_view(), w, h, 0.0)
            raw = ctx.queue.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                         {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
            imgs.append(np.frombuffer(raw, np.uint8).reshape(h, w, 4).copy())
        assert np.array_equal(imgs[0], imgs[1])
        a = imgs[0][..., :3].astype(int)
        assert a.std() > 8                                                    # not a flat colour
        assert (a[..., 2] > a[..., 0] + 20).any()                             # sky / water (blue) present
        assert app.terrain_mesh.n_visible > 0 and app.terrain_mesh.n_water > 0
        assert not app.errors
    finally:
        app.shutdown()


def test_frustum_planes_contain_target():
    from bgv.gpu.camera import OrbitCamera
    from bgv.gpu.layers.terrain import TerrainMesh
    cam = OrbitCamera()
    cam.resize(1600, 900)
    cam.set_view(100.0, 200.0, 3000.0, 30.0, 45.0)
    pl = TerrainMesh.frustum_planes(cam.view_proj())
    p = np.append(cam.target, 1.0)
    assert (pl @ p >= -1e-6).all()
    behind = np.append(cam.eye() - cam.forward_dir() * 500.0, 1.0)
    assert (pl @ behind < 0).any()                                            # the near plane rejects points behind
