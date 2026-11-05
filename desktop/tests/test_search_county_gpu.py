"""County + depth tiles end to end on the offscreen canvas (real data; skipped when the county flow grids, DEM, OSM cache or tiles are
absent): address search results, facility depths and the risk report use the tile depths and show tiles, index and their difference."""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pytest

from data_helpers import real_root

wgpu = pytest.importorskip("wgpu")
W, H = 1600, 900
DATA = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")


@pytest.fixture(scope="module")
def env():
    from rendercanvas.offscreen import OffscreenRenderCanvas
    from bgv import domain as domainmod, tools
    from bgv.gpu.context import GpuContext
    from bgv.search import tiles as T
    from bgv.ui import bridge, panels_search
    root = real_root()
    infos = {i.key: i for i in domainmod.domain_infos(root, {})}
    if not infos["county"].available or T.TileSet.discover(root, "county") is None:
        pytest.skip("county data or depth tiles not available")
    try:
        ctx = GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")
    canvas = OffscreenRenderCanvas(size=(W, H), pixel_ratio=1.0)
    fmt = canvas.get_wgpu_context().get_preferred_format(ctx.adapter)

    class A:
        storm = "100yr"
        msaa = 4
        quiet = True
        exag = None
        sun = None
        hour = 12.0
        camera = None
        at = None
        ripple = None
        haze = None
        domain = "county"
        county_dem = None
        county_demo = None
        county_pbf = None
    app = tools.make_app(A, ctx, fmt, {})
    try:
        app.attach_canvas(canvas, ui=True)
        tools.apply_view_args(app, A)
        tools.wait_for_flow(app, all_hours=False)
        assert bridge.ensure_hub(app) and app._stats_hub.wait_ready(240.0)
        canvas.set_physical_size(W, H)
        canvas.request_draw(app.draw_frame)
        ov = app.get_layer("overlay")
        ov.state.seen_help = True

        class Env:
            pass
        e = Env()
        e.app, e.canvas, e.ov, e.m = app, canvas, ov, ov.manager
        e.draw = lambda n=2: [canvas.draw() for _ in range(n)][-1]
        e.draw(3)
        e.c = panels_search.controller(e.m)
        e.c.onboarded = True
        for _ in range(600):
            e.draw(1)
            if all(e.c.status[k].state in ("ready", "missing", "error") for k in ("index", "facilities", "tiles")):
                break
            time.sleep(0.05)
        yield e
    finally:
        app.shutdown()


def _clean(env):
    assert not env.m.panel_errors, env.m.panel_errors
    assert not [x for x in env.app.errors if x.startswith(("ui", "screenshot", "overlay", "report"))]


def test_county_loaded_with_tiles_and_labels(env):
    c = env.c
    assert c.status["tiles"].state == "ready" and c.tiles is not None and len(c.tiles.storms) == 12
    assert c.index.domain == "county" and len(c.index.storm_keys) == 12 and c.facilities.n == 98
    assert c.facilities.source("100yr") == "solver depth (tiles, 6.25 m; verification pending)"
    assert c.facilities.has_tiles("ev6")
    env.draw(3)
    _clean(env)


def test_address_result_uses_tiles_and_shows_both(env):
    from bgv.ui import panels_search
    c, idx = env.c, env.c.index
    rnd = random.Random(2)
    cand = [i for i in range(idx.n) if idx.kind[i] == 0 and idx.depth_cm["100yr"][i] >= 20]
    for i in rnd.sample(cand, 3):
        c.select_record(i)
        for _ in range(400):
            env.draw(1)
            if c.result is not None and not c.result.pending:
                break
            time.sleep(0.02)
        env.m.nav.cancel()
        a = c.result.assessment
        assert a is not None and a.method == "footprint" and a.tile_label.startswith("solver depth (tiles, 6.25 m")
        r = a.row("100yr")
        assert r.tile_cm is not None and r.shown_cm == r.tile_cm and r.shown_source == a.tile_label
        assert r.index.depth_cm == int(idx.depth_cm["100yr"][i])               # the index number is kept next to it
        assert r.comparison.delta_cm == r.tile_cm - r.index.depth_cm
        env.draw(3)
        _clean(env)


def test_footprint_resolution_picks_the_named_osm_building(env):
    """The county record's OSM way id -> polygon -> centroid -> the viewer's building (not just the nearest one to the record point)."""
    from bgv.ui import bridge
    c, app = env.c, env.app
    idx = c.index
    fp = c.get_footprints()
    assert fp is not None
    b = bridge.buildings(app)
    rnd = random.Random(9)
    cand = [i for i in range(idx.n) if idx.kind[i] == 0 and fp.get(int(idx.bid[i])) is not None]
    dists = []
    for i in rnd.sample(cand, 200):
        rec = idx.record(i)
        bid = c.resolve_building(rec, fp)
        if bid is None:
            continue
        poly = fp.get(rec.bid)
        ring = np.asarray(poly[0], np.float64)
        x, y = app.terrain.frame.lnglat_to_world(float(ring[:, 0].mean()), float(ring[:, 1].mean()))
        dists.append(float(np.hypot(b.cx[bid] - x, b.cy[bid] - y)))
    assert len(dists) >= 190 and np.median(dists) < 3.0, (len(dists), np.median(dists))


def test_facility_card_and_markers_with_tiles(env):
    c, app = env.c, env.app
    fs = c.facilities
    i = int(np.argmax(fs.d_index["100yr"]))
    c.select_facility(i, fly=True)
    env.m.nav.cancel()
    c.show_fac = True
    env.draw(4)
    _clean(env)
    info = fs.info(i, "100yr")
    p = next(x for x in info["per_storm"] if x["storm"] == "100yr")
    assert p["tile_cm"] is not None and p["index_cm"] == int(fs.d_index["100yr"][i]) and p["near_cm"] is not None
    assert p["source"].startswith("solver depth (tiles")
    c.show_fac = False
    c.facility_sel = -1


def test_report_county_has_both_sources(env, tmp_path):
    c, idx = env.c, env.c.index
    env.app.settings["report_dir"] = str(tmp_path)
    i = next(int(k) for k in np.flatnonzero((idx.depth_cm["100yr"] >= 30) & (idx.kind == 0)) if int(idx.nb[k]) == 1)
    c.select_record(i)
    for _ in range(400):
        env.draw(1)
        if c.result is not None and not c.result.pending:
            break
        time.sleep(0.02)
    env.m.nav.cancel()
    c.request_report()
    env.draw(2)
    env.ov.post_frame(env.app)
    html = next(tmp_path.glob("risk_*.html")).read_text(encoding="utf-8")
    assert "solver depth (tiles, 6.25 m; verification pending)" in html and "index (grid)" in html
    assert "Difference" in html and "Tiles" in html and "Index" in html and "100-year storm" in html
    assert "validated against observed floods" in html.lower()
    assert len(list(tmp_path.glob("risk_*.png"))) == 1
    env.app.settings.pop("report_dir", None)


def test_roads_county_zoom(env):
    c = env.c
    c.show_roads = True
    t = None
    for _ in range(300):
        env.draw(1)
        t = c.road_table("100yr")
        if t is not None:
            break
        time.sleep(0.02)
    assert t is not None and not isinstance(t, str) and t.n == 8875
    c.ensure_geometry()
    for _ in range(400):
        env.draw(1)
        if c.geometry is not None:
            break
        time.sleep(0.02)
    view = t.view("flooded_pct", False)
    assert c.select_road(t, int(view[0]))
    env.m.nav.cancel()
    env.draw(3)
    _clean(env)
    c.show_roads = False
    c.road_sel = None
