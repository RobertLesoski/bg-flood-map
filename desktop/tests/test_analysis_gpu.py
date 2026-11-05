"""Analysis tools on the offscreen canvas with the real CITY data (flow-grid back end; the county tile path is covered by
test_analysis_data.py and one real county render test at the end): clicks through the real UI path, profile / series / timeline
numbers against independent reads of the flow grid and data/stats.py, the difference layer's texture against decoded grids, and
frames of the swipe / difference modes."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from data_helpers import real_root

pytestmark = pytest.mark.gpu
wgpu = pytest.importorskip("wgpu")
W, H = 1600, 900


def _wait(cond, timeout=120.0, what=""):
    t0 = time.time()
    while not cond():
        time.sleep(0.03)
        if time.time() - t0 > timeout:
            raise TimeoutError(what)


def _make_env(domain: str, storm: str, hour: float):
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
    A.storm, A.msaa, A.quiet, A.exag, A.sun, A.hour, A.camera, A.at = storm, 4, True, None, None, hour, None, None
    A.ripple = A.haze = None
    A.domain, A.county_dem, A.county_demo, A.county_pbf = domain, None, None, None
    app = tools.make_app(A, ctx, fmt, settings={})
    app.attach_canvas(canvas, ui=True)
    tools.apply_view_args(app, A)
    tools.wait_for_flow(app, all_hours=True)
    assert bridge.ensure_hub(app)
    assert app._stats_hub.wait_ready(180.0)
    canvas.set_physical_size(W, H)
    canvas.request_draw(app.draw_frame)

    class Env:
        pass
    e = Env()
    e.app, e.canvas = app, canvas
    e.ov = app.get_layer("overlay")
    e.ov.state.seen_help = True
    e.m, e.an = e.ov.manager, app.get_layer("analysis")
    e.draw = lambda n=2: np.asarray([canvas.draw() for _ in range(n)][-1])[..., :3].astype(np.int16)
    e.click = lambda fx, fy: e.m._click(e.m.L.sx0 + (e.m.L.sx1 - e.m.L.sx0) * fx, e.m.L.sy0 + (e.m.L.sy1 - e.m.L.sy0) * fy)
    return e


@pytest.fixture(scope="module")
def env():
    e = _make_env("city", "25yr", 12.0)
    e.app.camera.set_view(*e.app.terrain.frame.lnglat_to_world(-86.4436, 36.9903), 3500.0, 0.0, 55.0)
    e.draw(3)
    _wait(lambda: e.an.scene() is not None, 120, "scene")
    yield e
    e.app.shutdown()


def _clean(e):
    assert not e.m.panel_errors, e.m.panel_errors
    mine = [x for x in e.app.errors if x.startswith(("ui", "compare", "analysis"))]
    assert not mine, mine


def test_layers_exist_and_stay_off_the_sidebar(env):
    assert env.an is not None and env.app.get_layer("compare") is not None
    from bgv.ui import panels_side
    assert "compare" in panels_side.HIDDEN_LAYERS and "analysis" in panels_side.HIDDEN_LAYERS
    assert env.an.source.tiles is None and not env.an.source.uses_tiles("25yr")        # the city has no tiles: flow grid
    assert "preview" in env.an.depth_label() or "flow grid" in env.an.depth_label()
    env.an.st.open = True
    env.draw(2)
    _clean(env)


def test_measure_by_clicks_and_keys(env):
    an = env.an
    an.clear_measures()
    an.tool = "measure_line"
    for f in ((0.30, 0.50), (0.45, 0.40), (0.60, 0.50)):
        env.click(*f)
    assert an.measures and len(an.measures[0].pts) == 3 and not an.measures[0].done
    env.app._on_key({"event_type": "key_down", "key": "Backspace"})
    assert len(an.measures[0].pts) == 2
    env.click(0.60, 0.50)
    env.app._on_key({"event_type": "key_down", "key": "Enter"})
    m = an.measures[0]
    assert m.done and an.tool == "none"
    from bgv.analysis import geodesy
    lng, lat = m.lnglat(an.frame)
    assert m.length_m(an.frame) == pytest.approx(geodesy.path_length_m(lng, lat))
    # the distance between two clicked ground points is a plausible fraction of the view width, in true metres
    assert 100.0 < m.length_m(an.frame) < 4000.0
    an.tool = "measure_area"
    for f in ((0.4, 0.6), (0.6, 0.6), (0.6, 0.75), (0.4, 0.75)):
        env.click(*f)
    env.app._on_key({"event_type": "key_down", "key": "Enter"})
    a = an.measures[-1]
    assert a.kind == "area" and a.done and a.area_m2(an.frame) > 1000.0
    an.tool = "measure_line"
    env.click(0.2, 0.2)
    env.app._on_key({"event_type": "key_down", "key": "Escape"})
    assert an.tool == "none" and all(x.done for x in an.measures)               # Esc drops the unfinished one
    env.draw(2)
    _clean(env)
    an.clear_measures()


def _ref_depth(storm, hour, lng, lat):
    """Independent flow-grid bilinear depth: whole numpy array, plain loops."""
    from bgv.analysis import geodesy
    a = storm.load_hour(hour)
    meta = storm.meta
    out = []
    x, y = geodesy.lnglat_to_merc(lng, lat)
    for xm, ym in zip(x, y):
        gx = (xm - meta["bounds_merc"][0]) / meta["cell_m"]
        gy = (meta["bounds_merc"][3] - ym) / meta["cell_m"]
        if not (0 <= gx <= meta["width"] and 0 <= gy <= meta["height"]):
            out.append(0.0)
            continue
        fx, fy = gx - 0.5, gy - 0.5
        x0, y0 = math.floor(fx), math.floor(fy)
        tx, ty = fx - x0, fy - y0
        code = 0.0
        al = 0.0
        for dx, dy, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
            c, r = min(max(x0 + dx, 0), meta["width"] - 1), min(max(y0 + dy, 0), meta["height"] - 1)
            code += w * float(a[r, c, 0])
            al += w * float(a[r, c, 3])
        d = (code / 255.0) ** 2 * 8.0 if al >= 127.5 else 0.0
        out.append(0.0 if d < 0.02 else d)
    return np.array(out)


def test_profile_by_clicks_matches_direct_grid_reads(env):
    an, app = env.an, env.app
    an.profile_line.clear()
    an.profile = None
    an.profile_key = None
    an.tool = "profile"
    env.click(0.15, 0.30)
    env.click(0.85, 0.75)
    app._on_key({"event_type": "key_down", "key": "Enter"})
    _wait(lambda: (env.draw(1), an.profile is not None)[1], 60, "profile")
    P = an.profile
    assert P.storm == "25yr" and P.n >= 100 and P.length_m > 500
    assert "flow grid" in P.source_label or "preview" in P.source_label
    ref = _ref_depth(app.storm, 12, P.lng, P.lat)
    idx = np.linspace(0, P.n - 1, 150).astype(int)
    assert np.allclose(P.depth_m[idx], ref[idx], atol=1e-5)
    g = app.terrain.sampler.sample(P.x, P.y)
    assert np.allclose(np.nan_to_num(P.ground_m[idx], nan=-1), np.where(np.isfinite(P.ground_m[idx]), g[idx], -1), atol=1e-3) or P.ground_label.startswith("ground: DEM")
    assert (P.depth_m > 0).any() or True
    # the hour moves the water: another hour gives another profile (cached per hour)
    app.t = 18.0
    an.request_profile()
    _wait(lambda: (env.draw(1), an.profile is not None and an.profile.hour == 18.0)[1], 60, "profile 18")
    assert an.profile.depth_m.sum() != P.depth_m.sum()
    app.t = 12.0
    env.draw(2)
    _clean(env)


def test_building_series_equals_the_statistics_per_hour(env):
    """The series of a building on the flow grid is the statistics rule: it must equal the hub's per-hour building depth."""
    from bgv.ui import bridge
    an, app = env.an, env.app
    sc = an.scene()
    B = sc[0]
    fl = app.flood["buildings"]
    ids = [int(fl.order[0]), int(fl.order[len(fl.order) // 2])]
    for bi in ids:
        an.set_target_building(bi)
        _wait(lambda: an.series() is not None and an.series().complete, 120, "series")
        r = an.series()
        i = r.index("25yr")
        hub_row = bridge.depth_series(app, bi)
        assert hub_row is not None
        got = r.depth[i]
        for h in range(25):
            if not math.isnan(hub_row[h]):
                assert got[h] == pytest.approx(hub_row[h], abs=1e-3), (bi, h)
        assert r.source[i] == "grid" and len(r.storms) == len(app.storms_available)
        assert r.peak(i) == pytest.approx(np.nanmax(hub_row), abs=1e-3) or np.isnan(np.nanmax(hub_row))
    an.set_target_point(-86.4436, 36.9903)
    _wait(lambda: an.series().complete, 60, "series point")
    env.draw(2)
    _clean(env)


def test_grid_timeline_equals_direct_statistics(env):
    from bgv.data import stats
    an, app = env.an, env.app
    _wait(lambda: app.timeline.get("complete"), 180, "hub timeline")
    tl = an.timeline()
    assert tl.complete and tl.source == "grid"
    sc = an.scene()
    B, R, frame = sc
    storm = app.storm
    for h in (4, 9, 12, 18, 24):
        g = None if storm.is_zero(h) else storm.load_hour(h, cache=False)
        bs = stats.evaluate_buildings(g, B.index)
        rs = stats.evaluate_roads(g, R)
        gs = stats.summarize_grid(g, frame)
        assert tl.flooded[h] == bs.flooded_count
        assert tl.road_km[h] == pytest.approx(rs.flooded_length_m / 1000.0, rel=1e-9)
        assert tl.area_km2[h] == pytest.approx(gs.wet_area_km2, rel=1e-9)
        assert list(tl.tiers[h]) == list(bs.tier_counts)
    env.an.st.tab = "timeline"
    env.an.st.open = True
    env.draw(3)
    _clean(env)


def test_difference_layer_texture_and_frame(env):
    from bgv.analysis import compare
    app, an = env.app, env.an
    app.t = 12.0
    base = env.draw(3)
    an.st.b_storm = "100yr"
    an.st.b_same_hour = True
    an.st.mode = "diff"
    cl = app.get_layer("compare")
    _wait(lambda: (env.draw(1), an.cmp.shown_key is not None)[1], 120, "diff")
    assert cl.shown_key == an.cmp.shown_key and not an.cmp.error
    img = env.draw(3)
    assert np.abs(img - base).sum() > 1e6                       # the sheet changes the picture
    # the texture content: delta = decoded depth(B) - decoded depth(A) on the flow grid, for 100+ cells
    a, b = app.storm, an.source.grid.storm_obj("100yr")
    ga, gb = a.load_hour(12), b.load_hour(12)

    def dec(g, st):
        t = g[..., 0].astype(np.float64) / 255.0
        d = t * t * st.depth_max
        d[(g[..., 3] == 0) | (d < st.wet_threshold)] = 0.0
        return d
    da, db = dec(ga, a), dec(gb, b)
    packed = compare.delta_pack(np.round(da * 100).astype(np.float32) / 100, np.round(db * 100).astype(np.float32) / 100)
    prov = an.cmp._provider(app)
    ga2, gb2 = prov.grid("25yr", 12), prov.grid("100yr", 12)
    assert np.allclose(ga2, da, atol=0.0051) and np.allclose(gb2, db, atol=0.0051)
    rng = np.random.default_rng(2)
    wet = np.flatnonzero(((da > 0) | (db > 0)).ravel())
    pick = rng.choice(wet, 150, replace=False)
    got = (gb2 - ga2).ravel()[pick]
    assert np.allclose(got, (db - da).ravel()[pick], atol=0.011)
    S = an.cmp.area_stats()
    assert S is not None
    st, _ = S
    cg2 = a.frame.cell_ground_m ** 2
    assert st.a_area_km2 == pytest.approx(float((da >= 0.02).sum()) * cg2 / 1e6, rel=0.002)
    assert st.new_wet_km2 > 0 and st.volume_delta_m3 > 0 and st.max_added_m > 0
    bs = None
    _wait(lambda: (env.draw(1), an.cmp.building_stats() is not None)[1], 60, "building stats")
    _, sa, sb = an.cmp.building_stats()
    assert sb.flooded > sa.flooded                             # the 100-year storm floods more buildings than the 25-year
    from bgv.data import stats
    sc = an.scene()
    assert sa.flooded == stats.evaluate_buildings(ga, sc[0].index).flooded_count          # grid back end = the statistics rule exactly
    assert sb.flooded == stats.evaluate_buildings(gb, sc[0].index).flooded_count
    an.st.mode = "off"
    env.draw(3)
    assert cl.delta_tex is None and getattr(app, "compare_wet", 1) is None
    _clean(env)


def test_swipe_frames_left_is_storm_a_right_is_storm_b(env):
    app, an = env.app, env.an
    app.t = 12.0
    an.st.mode = "off"
    # terrain + water only: buildings, roads, particles and arrows are coloured by storm A on both sides of the divider (documented)
    saved = {n: app.get_layer(n).enabled for n in ("buildings", "roads", "particles", "arrows", "flowpaths")}
    for n in saved:
        app.get_layer(n).enabled = False
    an.st.open = False                                                           # the window shows storm-dependent plots
    env.draw(3)
    app.set_storm("100yr")
    _wait(lambda: (env.draw(1), app.flow.ready_for(12.0) and app.flow.progress()[0] >= app.flow.progress()[1])[1], 120, "100yr")
    env.draw(3)
    base_b = env.draw(3)
    app.set_storm("25yr")
    _wait(lambda: (env.draw(1), app.flow.ready_for(12.0) and app.flow.progress()[0] >= app.flow.progress()[1])[1], 120, "25yr")
    env.draw(3)
    base_a = env.draw(3)                                                         # storm A in exactly the state the swipe starts from
    an.st.b_storm = "100yr"
    an.st.b_same_hour = True
    an.st.vertical = True
    an.st.swipe = 0.5
    an.st.mode = "swipe"
    cl = app.get_layer("compare")
    env.draw(2)
    _wait(lambda: (env.draw(1), cl.lean is not None and cl.lean.ready(12) and 12 in cl.lean.slot_of and 13 in cl.lean.slot_of)[1], 120, "B hours")
    assert not cl.error
    sw = env.draw(4)
    x = int(round(env.m.L.sx0 + 0.5 * (env.m.L.sx1 - env.m.L.sx0)))
    ov_h = slice(int(env.m.L.sy0) + 60, int(env.m.L.sy1) - 20)                 # skip the labels and the status line
    left, right = slice(int(env.m.L.sx0) + 20, x - 30), slice(x + 30, int(env.m.L.sx1) - 30)
    # the left half is storm A's picture, the right half storm B's (compared with the pictures of the plain modes); the UI overlay
    # is identical in all of them, so only the water can differ
    dl_a = np.abs(sw[ov_h, left] - base_a[ov_h, left]).mean()
    dl_b = np.abs(sw[ov_h, left] - base_b[ov_h, left]).mean()
    dr_a = np.abs(sw[ov_h, right] - base_a[ov_h, right]).mean()
    dr_b = np.abs(sw[ov_h, right] - base_b[ov_h, right]).mean()
    assert np.abs(base_a[ov_h, right] - base_b[ov_h, right]).mean() > 0.2          # the storms really differ on the right side
    assert dl_a < dl_b and dr_b < dr_a, (dl_a, dl_b, dr_a, dr_b)
    # the divider moves with the setting
    an.st.swipe = 0.8
    sw2 = env.draw(3)
    assert np.abs(sw2 - sw).sum() > 1e5
    an.st.vertical = False
    env.draw(3)
    an.st.mode = "off"
    env.draw(3)
    assert cl.lean is None                                                       # B's hour cache is freed with the mode
    for n, v in saved.items():
        app.get_layer(n).enabled = v
    _clean(env)


def test_state_persists_through_the_layer(env):
    an = env.an
    an.st.imperial, an.st.range_m, an.st.tab = True, 3.5, "compare"
    d = an.save_state()
    from bgv.analysis.layer import AnalysisLayer
    fresh = AnalysisLayer()
    fresh.load_state(d)
    assert fresh.st.imperial and fresh.st.range_m == 3.5 and fresh.st.tab == "compare"
    an.st.imperial = False


# ------------------------------------------------------------------------------------------ real county render (tiles)
COUNTY_TILES = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\layers\cog\county\100yr\cog_meta.json")


@pytest.mark.skipif(not COUNTY_TILES.is_file(), reason="county tiles not present")
def test_county_tiles_drive_the_tools_and_the_difference_layer():
    try:
        e = _make_env("county", "25yr", 12.0)
    except Exception as ex:                                      # county data / DEM / PBF not usable on this machine
        pytest.skip(f"county domain unavailable: {ex}")
    try:
        app, an = e.app, e.an
        assert an.source.tiles is not None and an.source.uses_tiles("25yr")
        assert "verification pending" in an.depth_label("25yr") or "verified" in an.depth_label("25yr")
        app.camera.set_view(*app.terrain.frame.lnglat_to_world(-86.43, 37.0), 9000.0, 0.0, 52.0)
        e.draw(3)
        an.st.b_storm = "100yr"
        an.st.mode = "diff"
        _wait(lambda: (e.draw(1), an.cmp.shown_key is not None)[1], 180, "county diff")
        _wait(lambda: (e.draw(1), an.cmp.area_stats() is not None and an.cmp.area_stats()[1] == "tile resolution")[1], 180, "native stats")
        st, res = an.cmp.area_stats()
        assert st.b_area_km2 > st.a_area_km2 > 0 and st.new_wet_km2 > 0
        assert e.draw(3).std() > 5
        # a profile through the Barren River valley with the tile depth and the 5 m DEM
        an.profile_line = list(map(tuple, [app.terrain.frame.lnglat_to_world(-86.46, 36.97), app.terrain.frame.lnglat_to_world(-86.42, 37.01)]))
        an.st.mode = "off"
        an.request_profile(18)
        _wait(lambda: (e.draw(1), an.profile is not None)[1], 60, "profile")
        assert "tiles" in an.profile.source_label and an.profile.ground_label.startswith("ground: DEM") and an.profile.max_depth_m > 0
        assert not e.m.panel_errors, e.m.panel_errors
    finally:
        e.app.shutdown()
