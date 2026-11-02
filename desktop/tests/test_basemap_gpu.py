"""Basemap GPU smoke tests (offscreen, no window): checkerboard upload + texel / mip read-back, the WGSL snippet really
compiles and samples, layer life cycle with a stub app, no network anywhere."""
from types import SimpleNamespace

import numpy as np
import pytest

from bgv.data import basemap as bm

pytestmark = pytest.mark.gpu

wgpu = pytest.importorskip("wgpu")

from test_basemap import small_frame, png_bytes     # noqa: E402  (fixtures shared with the CPU tests)

RED, BLUE = (200, 40, 40), (40, 40, 200)


@pytest.fixture(scope="module")
def ctx():
    from bgv.gpu.context import GpuContext
    try:
        return GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")


def checker_comp(long_px=256, sq=16):
    comp = bm.BasemapComposite(small_frame(), long_px)
    yy, xx = np.mgrid[0:comp.height, 0:comp.width]
    chk = ((xx // sq + yy // sq) % 2 == 0)
    comp.rgba[..., 3] = 255
    comp.rgba[chk, :3] = RED
    comp.rgba[~chk, :3] = BLUE
    comp.rebuild_mips()
    return comp


def test_checkerboard_upload_and_readback(ctx):
    from bgv.gpu.layers.basemap import BasemapTexture
    comp = checker_comp(256, 16)
    tex = BasemapTexture(ctx, comp)
    try:
        assert not tex.initial_done and tex.pending() == comp.n_bands
        n = 0
        while tex.pending():
            assert tex.upload(30_000) > 0                 # small budget: bands stream in over several calls
            n += 1
        assert n > 2 and tex.initial_done
        # texels at the centres of two neighbouring squares, mip 0
        assert tex.read_texel(8, 8)[:3] == RED
        assert tex.read_texel(24, 8)[:3] == BLUE
        assert tex.read_texel(24, 24)[:3] == RED
        assert tex.read_texel(250, 250, 0)[3] == 255
        # mip 4 (16x16 texels, each = one 16 px square): same colours, nothing smeared
        assert tex.read_texel(0, 0, 4)[:3] == RED and tex.read_texel(1, 0, 4)[:3] == BLUE
        # mip 5: 2x2 box of red/blue squares = linear-light average (what the CPU box filter promises)
        got = np.array(tex.read_texel(0, 0, 5)[:3], int)
        lin = lambda c: ((np.array(c) / 255.0) ** 2.2)
        exp = (((lin(RED) + lin(BLUE)) / 2) ** (1 / 2.2)) * 255
        assert np.abs(got - exp).max() <= 4
        # final 1x1 mip exists
        assert tex.read_texel(0, 0, comp.n_levels - 1)[3] == 255
        # a later tile write marks only some bands dirty and is uploaded without touching the rest
        x0, _, y0, _ = bm.tile_range(comp.frame, comp.zoom)
        comp.put_tile(comp.zoom, x0, y0, np.full((256, 256, 3), 77, np.uint8))
        assert 0 < tex.pending() <= comp.n_bands
        tex.upload_all()
        assert tex.pending() == 0
        assert tex.read_texel(2, 2)[:3] == (77, 77, 77)
    finally:
        tex.destroy()


def test_fallback_is_cleared_on_gpu_and_tiles_still_upload(ctx):
    """Flat fallback costs no upload: a GPU clear stands in; a tile written afterwards is uploaded over it."""
    from bgv.gpu.layers.basemap import BasemapTexture
    comp = bm.BasemapComposite(small_frame(), 512)
    tex = BasemapTexture(ctx, comp)
    try:
        assert not tex.adopt_fallback() and not tex.initial_done            # nothing filled yet
        comp.fill_fallback()
        assert tex.adopt_fallback() and tex.initial_done and tex.pending() == 0
        assert tex.upload(10 ** 9) == 0                                      # nothing to upload: no 1 MB of paper
        for lvl in (0, 3, comp.n_levels - 1):
            got = tex.read_texel(0, 0, lvl)
            assert max(abs(a - b) for a, b in zip(got[:3], bm.FALLBACK_RGB)) <= 1, (lvl, got)
        x0, _, y0, _ = bm.tile_range(comp.frame, comp.zoom)
        comp.put_tile(comp.zoom, x0, y0, np.full((256, 256, 3), 40, np.uint8))
        assert tex.pending() > 0
        tex.upload_all()
        assert tex.read_texel(3, 3)[:3] == (40, 40, 40)
        assert max(abs(a - b) for a, b in zip(tex.read_texel(comp.width - 2, comp.height - 2)[:3], bm.FALLBACK_RGB)) <= 1
    finally:
        tex.destroy()


def test_upload_non_power_of_two_composite_all_bands_all_mips(ctx):
    """The real grid gives 3584x4096 / 5376x6144 composites: deep mips are 1 row high; every band must upload cleanly."""
    from bgv.gpu.layers.basemap import BasemapTexture
    from test_basemap import BG
    comp = bm.BasemapComposite(BG, 1536)                      # 1344 x 1536, 11 levels
    assert (comp.width, comp.height) == (1344, 1536) and comp.n_levels == 11
    comp.rgba[..., :3] = (10, 120, 200)
    comp.rgba[-5:, :, :3] = (250, 10, 10)                     # last rows differ: they reach the 1-row-high mips
    comp.rgba[..., 3] = 255
    comp.rebuild_mips()
    tex = BasemapTexture(ctx, comp)
    try:
        tex.upload_all()
        assert tex.initial_done and tex.pending() == 0
        assert tex.read_texel(1343, 1535)[:3] == (250, 10, 10)
        assert tex.read_texel(5, 5)[:3] == (10, 120, 200)
        top = tex.read_texel(0, 0, comp.n_levels - 1)
        assert top[3] == 255 and top[0] < 40                  # 1x1 mip is dominated by the blue field
    finally:
        tex.destroy()


def test_wgsl_snippet_compiles_and_samples_texels(ctx):
    """Render a full-screen triangle with BASEMAP_WGSL's basemap_color(): output pixels equal the uploaded texels."""
    from bgv.gpu.layers import basemap as L
    layer = L.BasemapLayer()
    app = SimpleNamespace(ctx=ctx)
    layer.init(app)
    comp = checker_comp(128, 8)
    tex = L.BasemapTexture(ctx, comp)
    tex.upload_all()
    layer.comp, layer.tex = comp, tex
    layer.enabled = True
    layer.opacity = 0.5
    try:
        assert app.basemap is layer.handle and layer.handle.version == 1
        layer.handle.set_texture(tex.view)
        assert layer.handle.version == 2 and layer.handle["texture_view"] is tex.view
        assert layer.handle.enabled and not layer.handle.has_tiles            # checker is not real tiles
        layer.handle.write_params(ctx.queue)
        dev = ctx.device
        mod = dev.create_shader_module(code=L.basemap_wgsl(0) + """
struct VO { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VO; o.pos = vec4<f32>(p[i], 0.0, 1.0); o.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5); return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4<f32> {
  let bm = basemap_color(in.uv);
  return vec4<f32>(bm.rgb, bm.a);     // alpha carries the opacity out for the check
}
""")
        pl = dev.create_pipeline_layout(bind_group_layouts=[layer.handle.bind_group_layout])
        pipe = dev.create_render_pipeline(layout=pl, vertex={"module": mod, "entry_point": "vs"},
                                          fragment={"module": mod, "entry_point": "fs",
                                                    "targets": [{"format": "rgba8unorm-srgb"}]},
                                          primitive={"topology": "triangle-list"})
        W = comp.width
        out = dev.create_texture(size=(W, W, 1), format="rgba8unorm-srgb",
                                 usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        enc = dev.create_command_encoder()
        p = enc.begin_render_pass(color_attachments=[{"view": out.create_view(), "clear_value": (0, 0, 0, 0),
                                                      "load_op": "clear", "store_op": "store"}])
        p.set_pipeline(pipe)
        p.set_bind_group(0, layer.handle.bind_group)
        p.draw(3)
        p.end()
        ctx.queue.submit([enc.finish()])
        raw = ctx.queue.read_texture({"texture": out, "mip_level": 0, "origin": (0, 0, 0)},
                                     {"offset": 0, "bytes_per_row": W * 4, "rows_per_image": W}, (W, W, 1))
        img = np.frombuffer(raw, np.uint8).reshape(W, W, 4)
        assert tuple(img[4, 4, :3]) in (RED, (RED[0] + 1, RED[1], RED[2])) or np.abs(img[4, 4, :3].astype(int) - RED).max() <= 1
        assert np.abs(img[4, 12, :3].astype(int) - BLUE).max() <= 1
        assert np.abs(img[12, 12, :3].astype(int) - RED).max() <= 1
        assert abs(int(img[4, 4, 3]) - 128) <= 1                              # opacity 0.5 reached the shader
        # layer disabled -> opacity param goes to 0 on the next write_params, no update() needed
        layer.enabled = False
        layer.handle.write_params(ctx.queue)
        enc = dev.create_command_encoder()
        p = enc.begin_render_pass(color_attachments=[{"view": out.create_view(), "clear_value": (0, 0, 0, 0),
                                                      "load_op": "clear", "store_op": "store"}])
        p.set_pipeline(pipe)
        p.set_bind_group(0, layer.handle.bind_group)
        p.draw(3)
        p.end()
        ctx.queue.submit([enc.finish()])
        raw = ctx.queue.read_texture({"texture": out, "mip_level": 0, "origin": (0, 0, 0)},
                                     {"offset": 0, "bytes_per_row": W * 4, "rows_per_image": W}, (W, W, 1))
        assert np.frombuffer(raw, np.uint8).reshape(W, W, 4)[4, 4, 3] == 0
    finally:
        layer.shutdown(app)


def test_layer_lifecycle_offline_cache_only(ctx, tmp_path):
    """Layer with a stub app: ticking the layer reads ONLY the disk cache (a fake server counts requests)."""
    from test_basemap import FakeServer
    from bgv.gpu.layers.basemap import BasemapLayer
    frame = small_frame()
    server = FakeServer()
    # pre-populate the cache for the standard preset's plan using the fake server, then verify the layer never asks again
    src = bm.TileSource(server.url, cache_root=tmp_path / "t", delay=0, min_interval=0, backoff=0.01)
    comp0 = bm.BasemapComposite(frame, 4096 // 16)
    ldr = bm.BasemapLoader(comp0, src)
    ldr.start(download=True)
    ldr.join(30)
    n_req = server.total
    assert n_req > 0

    class TestLayer(BasemapLayer):
        def make_source(self):
            return bm.TileSource(server.url, cache_root=tmp_path / "t", delay=0, min_interval=0)
    app = SimpleNamespace(ctx=ctx, terrain=SimpleNamespace(frame=frame), quiet=True)
    layer = TestLayer()
    layer.load_state({"opacity": 0.6, "detail": 0})
    layer.init(app)
    try:
        assert not layer.enabled and layer.default_enabled is False           # off by default
        assert not app.basemap.enabled and app.basemap["opacity"] == pytest.approx(0.6)
        layer.enabled = True
        import time
        t0 = time.time()
        # detail 0 = 4096 px composite of the tiny frame (z16..): plan tiles differ from the 256-px pre-population,
        # so some tiles are missing: they must stay missing (cache only), proving no request is made on enable
        while time.time() - t0 < 60:
            layer.update(app, 0.016)
            if layer.loader.status().phase in ("done", "unavailable") and layer.tex.pending() == 0:
                break
            time.sleep(0.01)
        assert server.total == n_req                                          # NO network traffic from ticking the layer
        assert layer.comp is not None and layer.tex is not None
        assert layer.handle.ready and app.basemap.enabled
        assert ctx.vram_breakdown().get("basemap", 0) > 50
        st = layer.save_state()
        assert st["opacity"] == pytest.approx(0.6) and st["detail"] == 0
        plan = layer.plan
        for _ in range(100):
            if layer.plan is not None:
                break
            time.sleep(0.02)
        assert layer.plan is not None and layer.plan.missing >= 0
    finally:
        layer.shutdown(app)
        server.close()
    assert "basemap" not in ctx.vram_breakdown() and app.basemap is None


# ====================================================================================================== terrain integration
def _project(app, X, Y, w, h):
    z = float(app.terrain.sampler.sample(X, Y)) * app.exag
    c = np.asarray(app.view_proj, np.float64) @ np.array([X, Y, z, 1.0])
    return (c[0] / c[3] * 0.5 + 0.5) * w, (0.5 - c[1] / c[3] * 0.5) * h


def _magenta(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 170) & (b > 150) & (g < 120)


def test_terrain_shader_compiles_with_and_without_basemap(ctx):
    from bgv.gpu.layers.terrain import TERRAIN_WGSL, BASEMAP_OFF_WGSL
    from bgv.gpu.layers.basemap import basemap_wgsl
    for code in (basemap_wgsl(1) + TERRAIN_WGSL, BASEMAP_OFF_WGSL + TERRAIN_WGSL):
        assert ctx.device.create_shader_module(code=code) is not None


def test_terrain_drape_alignment_toggle_and_relief(tmp_path):
    """The real terrain pipeline with a synthetic composite: a marker block lands on its world point (uv identity), the layer
    toggle / opacity reach the shader without a basemap update(), and the hillshade still modulates the map."""
    from bgv import config
    try:
        config.data_root()
    except FileNotFoundError:
        pytest.skip("flow data root not found")
    import basemap_synth as S
    w, h = 960, 540
    app, canvas, _ctx = S.make_app({"layers": {"basemap": {"enabled": True, "state": {"opacity": 1.0}}}}, (w, h))
    try:
        for l in app.layers:                                        # terrain + basemap only: nothing else may tint the block
            if l.name not in ("terrain", "basemap"):
                l.enabled = False
        fr = app.terrain.frame
        comp = bm.BasemapComposite(fr, 2048)
        comp.rgba[..., :3] = (240, 238, 232)
        comp.rgba[..., 3] = 255
        P = (1205.0, 1486.0)
        u, v = fr.world_to_uv(*P)
        cx, cy = int(u * comp.width), int(v * comp.height)
        comp.rgba[cy - 6:cy + 6, cx - 6:cx + 6, :3] = (255, 0, 255)          # 12 px = ~90 m
        comp.rebuild_mips()
        lay = S.install(app, comp)
        cam = (P[0], P[1], 700.0, 0.0, 88.0)
        on = S.render(app, canvas, tmp_path / "on.png", cam, 0.0, (w, h))
        m = _magenta(on)
        assert m.sum() > 100, "marker block not visible: drape missing"
        ys, xs = np.nonzero(m)
        px, py = _project(app, *P, w, h)
        assert abs(xs.mean() - px) < 3.0 and abs(ys.mean() - py) < 3.0, (xs.mean(), ys.mean(), px, py)
        # relief: lit ground varies in luminance (hillshade over the flat map), not a flat colour
        lum = on[..., :3].astype(float).mean(-1)
        ground = lum[~m][::7]
        assert ground.std() > 2.0
        # toggle: the basemap layer is disabled -> the terrain pass writes opacity 0 itself
        lay.enabled = False
        off = S.render(app, canvas, tmp_path / "off.png", cam, 0.0, (w, h))
        assert _magenta(off).sum() == 0
        # corr of luminance between the map-on and map-off renders (same lighting): relief survives under the drape
        lo = off[..., :3].astype(float).mean(-1)
        sel = ~m
        c = np.corrcoef(lum[sel][::5], lo[sel][::5])[0, 1]
        assert c > 0.2, c
        lay.enabled = True
        lay.opacity = 0.0
        zero = S.render(app, canvas, tmp_path / "zero.png", cam, 0.0, (w, h))
        assert _magenta(zero).sum() == 0
        lay.opacity = 0.5
        half = S.render(app, canvas, tmp_path / "half.png", cam, 0.0, (w, h))
        ix, iy = int(round(px)), int(round(py))
        sat = lambda im: int(im[iy, ix, 0]) - int(im[iy, ix, 1])             # magenta-ness at the marker centre
        assert sat(zero) < 40 < sat(half) < sat(on), (sat(zero), sat(half), sat(on))
        assert not app.errors
    finally:
        app.shutdown()
