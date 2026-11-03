"""County domain at app level (offscreen GPU, no window): a tiny synthetic county + a tiny OSM extract, domain switching with the
GPU memory returning to its baseline, the loading-screen City / County chooser.  The city part uses the real city data
(skipped when absent); the county data are tiny and fabricated by the tests."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bgv import config, domain as domainmod, tools

pytestmark = pytest.mark.gpu
wgpu = pytest.importorskip("wgpu")

REAL = Path(config.DEFAULT_DATA_ROOT)
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(scope="module")
def ctx():
    from bgv.gpu.context import GpuContext
    try:
        return GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"cannot create a junction: {r.stdout} {r.stderr}")
    else:
        os.link(src, dst)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """(root, settings): a data root that links the real CITY data (no real county grids in it) plus a tiny synthetic county
    demo folder and a tiny OSM extract (400 buildings, 2 roads) inside the demo grid."""
    if not (REAL / "layers" / "flow" / "25yr" / "meta.json").is_file() or sys.platform != "win32":
        pytest.skip("real city data not present")
    sys.path.insert(0, str(SCRIPTS))
    from rasterio.warp import transform
    from bgv.data import demo
    from pbf_writer import Block, write_pbf
    base = tmp_path_factory.mktemp("county_world")
    root = base / "root"
    _link(REAL / "layers" / "flow", root / "layers" / "flow")
    _link(REAL / "dem_bg.tif", root / "dem_bg.tif")
    _link(REAL / "layers" / "buildings.geojson", root / "layers" / "buildings.geojson")
    _link(REAL / "layers" / "roads.geojson", root / "layers" / "roads.geojson")
    x, y = transform("EPSG:4326", "EPSG:32616", [-86.4436], [36.9903])
    box = (x[0] - 3000, y[0] - 3000, x[0] + 3000, y[0] + 3000)
    dem_root = base / "demo"
    demo.make_county_demo(dem_root, utm_box=box, hours=[8, 10, 12, 14], storms=demo.DEMO_STORMS, workers=1, log=lambda m: None)
    nb, wb = Block(), Block()
    nodes, ways = [], []
    nid = 1
    for i in range(20):
        for j in range(20):
            lon0, lat0 = -86.4436 - 0.030 + i * 0.003, 36.9903 - 0.022 + j * 0.0022
            ids = []
            for dx, dy in ((0, 0), (0.0003, 0), (0.0003, 0.0002), (0, 0.0002)):
                nodes.append((nid, lon0 + dx, lat0 + dy))
                ids.append(nid)
                nid += 1
            ways.append((1000 + i * 20 + j, {"building": "house", "addr:housenumber": str(i * 20 + j), "addr:street": "Demo Rd"},
                         ids + [ids[0]]))
    road_ids = []
    for k in range(10):
        nodes.append((nid, -86.4436 - 0.03 + k * 0.006, 36.9903))
        road_ids.append(nid)
        nid += 1
    ways.append((9001, {"highway": "primary", "name": "Demo Rd"}, road_ids))
    ways.append((9002, {"highway": "residential", "name": "Side St"}, road_ids[:5]))
    nodes.sort()
    nb.add_nodes(nodes)
    wb.add_ways(ways)
    pbf = write_pbf(base / "tiny.pbf", [nb, wb])
    settings = {"county_demo": str(dem_root), "county_pbf": str(pbf)}
    return root, settings


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A private cache folder for the test (the county caches never touch desktop/cache).  The city's cache files are
    HARD-LINKED into it, so the city part of a test starts from its warm caches (copy-free) while everything the test
    writes lands in the private folder."""
    real = config.cache_dir()
    priv = tmp_path / "cache"
    priv.mkdir()
    for f in real.iterdir():
        if f.is_file():
            try:
                os.link(f, priv / f.name)
            except OSError:
                break
    monkeypatch.setenv("BGV_CACHE", str(priv))
    config.reset_cache_dir()
    yield
    monkeypatch.undo()
    config.reset_cache_dir()


def _wait(app, timeout=240.0):
    tools.wait_for_flow(app, timeout=timeout)


def _frame_ok(app, ctx, w=320, h=180):
    tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    try:
        app.step(0.0)
        app.render_scene(tex.create_view(), w, h, 0.0)
        raw = ctx.queue.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                     {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3].astype(int)
    finally:
        tex.destroy()


def test_county_app_runs_on_demo_data_with_osm_buildings(ctx, world, cache):
    from bgv.app import App
    root, settings = world
    app = App(ctx, root, storm="100yr", msaa=4, quiet=True, settings=dict(settings), domain="county")
    try:
        assert app.domain.key == "county" and app.domain.demo and app.storm.synthetic
        assert app.storms_available == ["10yr", "100yr"] and app.storm.name == "100yr"
        assert app.exag == app.domain.default_exag
        app.t = 12.0
        app.camera.set_view(0.0, 0.0, 9000.0, 0.0, 55.0)
        _wait(app)
        sc = app._scene_data
        assert sc.state == "ready", sc.error
        assert sc.buildings.n == 400 and int(sc.buildings.in_grid.sum()) == 400 and sc.roads.n_roads == 2
        assert sc.osm_info["buildings"] == 400
        hub = app._stats_hub
        assert hub.wait_ready(120.0)
        f = app.flood
        assert f["ready"] and f["n_buildings"] == 400 and f["flooded_count"] > 0 and f["source"] == "rasters"
        assert f["wet_area_km2"] > 1.0
        img = _frame_ok(app, ctx)
        assert img.std() > 8 and not app.errors
        # click-select a building of the county scene through the same picking code as the city
        from bgv.gpu import selection
        bid = int(np.flatnonzero(f["buildings"].flooded)[0])
        selection.select_building(app, bid)
        d = selection.describe_building(app, bid)
        assert d["flooded"] and d["depth_m"] >= 0.1 and d["address"].endswith("Demo Rd")
    finally:
        app.shutdown()


def test_switching_domains_frees_the_gpu_and_restores_the_baseline(ctx, world, cache):
    from bgv.app import App
    root, settings = world
    app = App(ctx, root, storm="25yr", msaa=4, quiet=True, settings=dict(settings))
    try:
        assert app.domain.key == "city"
        _wait(app)
        _frame_ok(app, ctx)
        base = dict(ctx.vram_breakdown())
        base_mb = ctx.vram_mb()
        w_city = app.flow.width
        # ---- city -> county
        assert app.set_domain("county", block=True) and not app.loading
        assert app.domain.key == "county" and app.storm.synthetic and app.storm.name == "100yr"
        assert app.flow.width != w_city and app.terrain.frame == app.storm.frame
        _wait(app)
        assert app._scene_data.frame == app.terrain.frame and app._scene_data.buildings.n == 400
        assert app._stats_hub.wait_ready(120.0) and app.flood["n_buildings"] == 400 and app.flood["storm"] == "100yr"
        _frame_ok(app, ctx)
        in_county = dict(ctx.vram_breakdown())
        assert in_county["terrain height"] != base["terrain height"] and "buildings mesh" in in_county
        assert abs(app.camera.dist - app.domain.overview_dist(app.terrain.frame)) < 1.0         # the camera fits the county
        assert app.camera.limit_xy[0] == pytest.approx(app.terrain.frame.size_m[0] / 2)
        assert not app.errors, list(app.errors)
        # a county storm switch inside the domain still works
        app.set_storm("10yr")
        _wait(app)
        assert app.storm.name == "10yr" and app.storm.label.startswith("10-year")
        # ---- county -> city: everything of the county is gone, the city baseline is back
        assert app.set_domain("city", block=True) and app.domain.key == "city"
        _wait(app)
        assert app._stats_hub.wait_ready(120.0)
        _frame_ok(app, ctx)
        back = dict(ctx.vram_breakdown())
        assert set(back) == set(base), (set(back) ^ set(base))
        for k in base:
            assert back[k] == pytest.approx(base[k], abs=0.15), (k, base[k], back[k])
        assert ctx.vram_mb() == pytest.approx(base_mb, abs=1.0)
        assert app.flood["n_buildings"] > 15000 and app.storm.name == "25yr" and app.exag == 2.0
        assert app.settings["domain"] == "city" and app.settings["domain_views"]["county"]["storm"] == "10yr"
        assert app.collect_settings()["storm"] == "25yr"                  # the city keeps its legacy settings keys
    finally:
        app.shutdown()


def test_unavailable_county_keeps_the_city_and_says_why(ctx, world, cache):
    from bgv.app import App
    root, settings = world
    app = App(ctx, root, storm="25yr", msaa=4, quiet=True, settings={"county_demo": str(Path(root) / "nowhere")})
    try:
        assert not app.set_domain("county", block=True)
        assert app.domain.key == "city" and not app.loading
        assert any("No county data yet" in m and "--make-county-demo" in m for m in app.errors)
        infos = {i.key: i for i in app.domain_infos(max_age=0.0)}
        assert infos["city"].available and not infos["county"].available and infos["county"].reason
        # a missing county request at construction falls back to the city with the same explanation
        a2 = App(ctx, root, storm="25yr", msaa=4, quiet=True, settings={"county_demo": str(Path(root) / "nowhere")}, domain="county")
        try:
            assert a2.domain.key == "city" and any("showing the city" in m for m in a2.errors)
        finally:
            a2.shutdown()
    finally:
        app.shutdown()


# ================================================================================================ the chooser
def _drive(app, canvas, until, timeout=60.0, on_frame=None):
    n = 0
    t0 = time.perf_counter()
    while not until():
        canvas.draw()
        n += 1
        if on_frame:
            on_frame(n)
        assert time.perf_counter() - t0 < timeout, f"loading did not finish: {app.load_msg} {[str(e)[:300] for e in list(app.errors)[:1]]}"
    return n


def _deferred_app(ctx, root, settings, chooser, domain="city"):
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv.app import App
    from bgv.startup import Prefetch
    canvas = OffscreenRenderCanvas(size=(900, 560), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)
    dom, _ = domainmod.resolve_or_city(domain, root, settings)
    pre = Prefetch(root, None, settings, domain=dom)
    app = App(ctx, root, quiet=True, settings=dict(settings), colour_format=fmt, defer=True, prefetch=pre, domain=dom, chooser=chooser)
    app.attach_canvas(canvas, ui=True)
    canvas.set_physical_size(900, 560)
    canvas.request_draw(app.draw_frame)
    return app, canvas


def test_chooser_continues_with_the_remembered_domain_after_the_countdown(ctx, world, cache):
    from bgv.ui.chooser import DomainChooser
    root, settings = world
    ch = DomainChooser(domainmod.domain_infos(root, settings), "city", countdown=0.8)
    assert ch.waits
    app, canvas = _deferred_app(ctx, root, settings, ch)
    try:
        seen = {}
        t0 = time.perf_counter()

        def watch(n):
            if app.loading and app.chooser is not None and "drawn" not in seen and ch.t0 is not None:
                seen["drawn"] = time.perf_counter()
        _drive(app, canvas, lambda: not app.loading, on_frame=watch)
        assert app.domain.key == "city" and not app.errors, list(app.errors)
        assert ch.t0 is not None and app.chooser is None and app.settings["domain"] == "city"
        assert time.perf_counter() - ch.t0 >= 0.8 - 0.05                    # it did not close before the countdown ran out
    finally:
        app.shutdown()


def test_chooser_click_switches_to_the_county_and_builds_only_that_one(ctx, world, cache):
    from bgv.ui.chooser import DomainChooser
    root, settings = world
    ch = DomainChooser(domainmod.domain_infos(root, settings), "city", countdown=30.0)       # would wait 30 s without a click
    app, canvas = _deferred_app(ctx, root, settings, ch)
    try:
        def click(n):
            if n == 4:
                assert ch.click("county") == "switch"
        _drive(app, canvas, lambda: not app.loading, on_frame=click)
        assert app.domain.key == "county" and app.storm.synthetic and app.settings["domain"] == "county"
        assert ch.building == "county" and not app.errors, list(app.errors)
        names = [l.name for l in app.layers]
        assert "terrain" in names and "buildings" in names
        _wait(app)
        assert app._scene_data.buildings.n == 400                         # only the county scene exists, built once
        assert app.flow.width == app.storm.width
    finally:
        app.shutdown()


def test_chooser_does_not_wait_when_the_county_has_no_data(ctx, world, cache):
    from bgv.ui.chooser import DomainChooser
    root, _ = world
    s = {"county_demo": str(Path(root) / "nowhere")}
    infos = domainmod.domain_infos(root, s)
    assert [i.available for i in infos] == [True, False] and infos[1].reason
    ch = DomainChooser(infos, "city", countdown=30.0)
    assert not ch.waits
    app, canvas = _deferred_app(ctx, root, s, ch)
    try:
        t0 = time.perf_counter()
        _drive(app, canvas, lambda: not app.loading)
        assert time.perf_counter() - t0 < 25.0 and app.domain.key == "city" and not app.errors, list(app.errors)
    finally:
        app.shutdown()


def test_chooser_cancel_waits_for_a_click(ctx, world, cache):
    from bgv.ui.chooser import DomainChooser
    root, settings = world
    ch = DomainChooser(domainmod.domain_infos(root, settings), "city", countdown=0.2)
    app, canvas = _deferred_app(ctx, root, settings, ch)
    try:
        def act(n):
            if ch.t0 is not None and not ch.cancelled and not ch.chosen:
                ch.cancel()
            if app.load_msg == "Ready" and app._build_done and n > 400 and not ch.chosen:
                pass
        for _ in range(0, 600):
            canvas.draw()
            act(0)
            if app._build_done:
                break
        assert app._build_done and app.loading                              # built, but held at the chooser
        for _ in range(30):
            canvas.draw()
        assert app.loading and ch.cancelled
        ch.click("city")
        _drive(app, canvas, lambda: not app.loading, timeout=30.0)
        assert app.domain.key == "city"
    finally:
        app.shutdown()
