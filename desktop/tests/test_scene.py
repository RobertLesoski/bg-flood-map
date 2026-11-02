"""Tests for the SCENE parts: scene_data (tiled meshes), stats hub, selection, buildings / roads layers.

CPU tests use tiny synthetic grids (fast); `real_*` tests use the real data root (skipped when it is absent);
tests marked `gpu` render offscreen (no window)."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from data_helpers import blank_grid, depth_to_code, line, polygon, real_root, rect, tiny_frame
from bgv.data import buildings as B
from bgv.data import roads as R
from bgv.data import stats as ST
from bgv.gpu import scene_data, selection, stats_hub
from bgv.gpu.camera import HeightSampler, OrbitCamera
from bgv.gpu.layers.buildings import TileCuller, frustum_planes, merge_runs


# ================================================================================================ synthetic scene
def make_scene(fr=None, base=100.0):
    fr = fr or tiny_frame(60, 60, 10.0)
    feats = [
        polygon(fr, rect(5, 5, 8, 8), height_m=6),                                   # 0 plain box
        polygon(fr, rect(20, 20, 30, 30), [(23, 23), (23, 27), (27, 27), (27, 23)], height_m=9),   # 1 box with a courtyard hole
        polygon(fr, rect(40, 10, 43, 13), height_m=30),                              # 2 tall
        polygon(fr, rect(12, 40, 15, 43), height_m=4),                               # 3
    ]
    b = B.buildings_from_features(feats, fr)
    rd = R.roads_from_features([line(fr, [(0, 15), (59, 15)], "primary"), line(fr, [(30, 0), (30, 59)], "residential")], fr)
    heights = np.full((fr.height, fr.width), base, np.float32)
    sampler = HeightSampler(heights, *fr.size_m)
    bz = B.attach_base_z(b.index, heights)
    sc = SimpleNamespace(state="ready", ready=True, frac=1.0, message="ready", error=None, frame=fr, heights=heights,
                         sampler=sampler, buildings=b, roads=rd, base_z=bz, mesh=scene_data.build_building_mesh(b, bz),
                         road_mesh=scene_data.build_road_mesh(rd, sampler), pick=scene_data.build_pick(b, rd))
    return sc


class FakeStorm:
    def __init__(self, name, frame, grids, slow=0.0):
        self.name, self.frame, self.grids, self.slow = name, frame, grids, slow
        self.hours = sorted(grids)
        self.source = "tiers"
        self.loads = 0

    def is_zero(self, h):
        return h not in self.grids

    def load_hour(self, h, cache=True):
        self.loads += 1
        if self.slow:
            time.sleep(self.slow)
        return self.grids[h]


def flood_grid(fr, cells_depth):
    g = blank_grid(fr)
    for (c, r), d in cells_depth.items():
        g[r, c, 0] = depth_to_code(d)
    return g


@pytest.fixture
def patched(monkeypatch):
    sc = make_scene()
    monkeypatch.setattr(stats_hub, "get_scene", lambda app: sc)
    monkeypatch.setattr(selection, "get_scene", lambda app: sc)
    return sc


def make_app(sc, storm, t=12.0):
    errors = []
    cam = OrbitCamera()
    cam.resize(1280, 720)
    cam.set_ground(sc.sampler, 1.0)
    app = SimpleNamespace(t=t, storm=storm, flood=None, stats={}, stats_hook=None, exag=1.0, camera=cam, errors=errors,
                          report_error=lambda w, e: errors.append((w, repr(e))), get_layer=lambda n: None, root=None)
    return app


def drain(hub, app, until, timeout=20.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        hub(app)
        if until():
            return True
        time.sleep(0.005)
    return False


# ================================================================================================ stats hub
def test_snapshot_hour_is_js_round():
    f = stats_hub.snapshot_hour
    assert [f(x) for x in (0.0, 0.49, 0.5, 11.5, 12.49, 12.5, 24.0, 30.0, -3.0)] == [0, 0, 1, 12, 12, 13, 24, 24, 0]


def test_hub_publishes_cached_snapshots_and_timeline(patched):
    sc = patched
    fr = sc.frame
    g12 = flood_grid(fr, {(6, 6): 0.5, (24, 24): 2.0, (41, 11): 0.05})     # box0 moderate, box1 (courtyard cell is a hole) , tall dry
    g13 = flood_grid(fr, {(6, 6): 1.5, (41, 11): 0.2})
    storm = FakeStorm("s1", fr, {12: g12, 13: g13})
    app = make_app(sc, storm, 12.0)
    hub = stats_hub.get_hub(app)
    assert app.stats_hook is hub and app.flood["ready"] is False
    assert drain(hub, app, lambda: app.flood["ready"] and not app.flood["pending"])
    f = app.flood
    ref = ST.evaluate_buildings(g12, sc.buildings)
    assert f["hour"] == 12 and f["storm"] == "s1"
    assert f["flooded_count"] == ref.flooded_count == 1                      # only box 0; 0.05 m is dry, (24,24) is not inside box1's ring? see below
    assert f["tier_counts"]["moderate"] == 1
    assert f["buildings"].depth_m[0] == pytest.approx(ref.depth_m[0])
    assert f["wet_area_km2"] == pytest.approx(ST.summarize_grid(g12, fr).wet_area_km2)
    assert f["road_pct"] == pytest.approx(ST.evaluate_roads(g12, sc.roads).flooded_pct)
    assert app.stats["Snapshot"].startswith("hour 12")
    assert app.stats["Flooded buildings"].startswith("1 of")
    v0 = f["version"]
    # a cache hit costs well under a millisecond and does not change the version
    t0 = time.perf_counter()
    for _ in range(200):
        hub(app)
    per = (time.perf_counter() - t0) / 200 * 1000
    assert per < 1.0 and app.flood["version"] == v0
    # rounding rule: t = 12.5 shows hour 13
    app.t = 12.5
    assert drain(hub, app, lambda: app.flood["hour"] == 13 and not app.flood["pending"])
    assert app.flood["buildings"].depth_m[0] == pytest.approx(ST.evaluate_buildings(g13, sc.buildings).depth_m[0])
    assert app.flood["version"] > v0
    # whole timeline in the background
    assert hub.wait_timeline(30.0)
    tl = app.timeline
    assert tl["complete"] and tl["done"] == 25 and tl["hours"] == list(range(25))
    assert tl["flooded"][12] == 1.0 and tl["flooded"][13] == ST.evaluate_buildings(g13, sc.buildings).flooded_count
    assert tl["flooded"][0] == 0.0 and tl["wet_area_km2"][5] == 0.0         # dry hours are exact zeros, not NaN
    assert storm.loads == 2                                                  # dry hours never decode a file
    assert not app.errors
    hub.close()


def test_hub_pending_keeps_previous_numbers_and_never_blocks(patched):
    sc = patched
    fr = sc.frame
    storm = FakeStorm("slow", fr, {12: flood_grid(fr, {(6, 6): 0.5}), 13: flood_grid(fr, {(6, 6): 1.5})}, slow=0.4)
    app = make_app(sc, storm, 12.0)
    hub = stats_hub.get_hub(app)
    assert drain(hub, app, lambda: app.flood["ready"] and app.flood["hour"] == 12, 30)
    app.t = 13.0
    worst = 0.0
    seen_pending = False
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 5.0:
        c0 = time.perf_counter()
        hub(app)
        worst = max(worst, time.perf_counter() - c0)
        seen_pending |= app.flood["pending"] and app.flood["hour"] != 13
        if app.flood["hour"] == 13 and not app.flood["pending"]:
            break
        time.sleep(0.01)
    assert app.flood["hour"] == 13
    assert worst < 0.015                                                      # a miss costs < 15 ms on the main thread
    assert seen_pending or storm.slow == 0                                    # numbers stayed those of hour 12 meanwhile
    hub.close()


def test_hub_storm_switch_resets(patched):
    sc = patched
    fr = sc.frame
    a = FakeStorm("a", fr, {12: flood_grid(fr, {(6, 6): 0.5})})
    b = FakeStorm("b", fr, {12: flood_grid(fr, {(6, 6): 2.5, (41, 11): 2.5})})
    app = make_app(sc, a, 12.0)
    hub = stats_hub.get_hub(app)
    assert drain(hub, app, lambda: app.flood["ready"] and app.flood["storm"] == "a")
    assert app.flood["flooded_count"] == 1
    app.storm = b
    hub(app)
    assert app.flood["storm"] == "b" and app.flood["ready"] is False         # never shows storm a's numbers under storm b
    assert drain(hub, app, lambda: app.flood["ready"] and app.flood["storm"] == "b")
    assert app.flood["flooded_count"] == 2 and app.flood["tier_counts"]["severe"] == 2
    assert app.timeline["storm"] == "b"
    hub.close()


def test_hub_chains_previous_hook(patched):
    sc = patched
    storm = FakeStorm("a", sc.frame, {12: flood_grid(sc.frame, {})})
    app = make_app(sc, storm)
    calls = []
    app.stats_hook = lambda a: calls.append(1)
    hub = stats_hub.get_hub(app)
    hub(app)
    assert calls and app.stats_hook is hub
    hub.close()


# ================================================================================================ meshes / tiles
def test_vertex_formats():
    assert scene_data.BLD_VERTEX_DTYPE.itemsize == 20 and scene_data.ROAD_VERTEX_DTYPE.itemsize == 20


def test_building_mesh_tiles_cover_every_triangle_once():
    sc = make_scene()
    m, b = sc.mesh, sc.buildings
    assert m.indices.size == b.mesh.indices.size
    t = m.tiles
    assert t.count.sum() == m.indices.size and (t.count % 3 == 0).all()
    assert (t.first[1:] == t.first[:-1] + t.count[:-1]).all() and t.first[0] == 0       # contiguous partition
    assert sorted(map(tuple, m.indices.reshape(-1, 3).tolist())) == sorted(map(tuple, b.mesh.indices.reshape(-1, 3).tolist()))
    # per building ranges: contiguous, only that building's vertices, and they add up
    tot = 0
    for i in range(b.n):
        assert m.bld_count[i] > 0
        rng = m.indices[m.bld_first[i]:m.bld_first[i] + m.bld_count[i]]
        assert (m.vertices["bid"][rng] == i).all()
        tot += int(m.bld_count[i])
    assert tot == m.indices.size
    # normals survive snorm8 packing; tile boxes contain their vertices
    nz = m.vertices["nrm"][:, 2] / 127.0
    assert set(np.round(nz[m.vertices["nrm"][:, 2] != 0], 3)) == {1.0}
    for k in range(t.n):
        v = m.vertices["pos"][np.unique(m.indices[t.first[k]:t.first[k] + t.count[k]])]
        assert v[:, 0].min() >= t.x0[k] - 1e-3 and v[:, 0].max() <= t.x1[k] + 1e-3
        assert v[:, 1].min() >= t.y0[k] - 1e-3 and v[:, 1].max() <= t.y1[k] + 1e-3
        assert t.hmax[k] >= v[:, 2].max() - 1e-3


def test_road_mesh_tiles_partition():
    sc = make_scene()
    rm, rd = sc.road_mesh, sc.roads
    assert rm.indices.size == rd.ribbon.indices.size
    assert rm.tiles.count.sum() == rm.indices.size
    assert sorted(rm.indices.tolist()) == sorted(rd.ribbon.indices.tolist())


def test_tile_culling_is_conservative_and_merges_runs():
    rng = np.random.default_rng(5)
    n = 400
    x0 = rng.uniform(-5000, 5000, n)
    y0 = rng.uniform(-5000, 5000, n)
    tiles = scene_data.Tiles(first=np.arange(n) * 30, count=np.full(n, 30), x0=x0, x1=x0 + 400, y0=y0, y1=y0 + 400,
                             zlo=np.full(n, 100.0), zhi=np.full(n, 140.0), hmax=np.full(n, 20.0))
    cam = OrbitCamera()
    cam.resize(1600, 900)
    for yaw in (0, 70, 200, 300):
        cam.set_view(0.0, 0.0, 2500.0, yaw, 40.0)
        vp = cam.view_proj()
        vis = TileCuller(tiles).visible(vp, 2.0, 1.0, 1.5, 2.0)
        assert 0 < vis.sum() < n
        for k in range(n):                                              # any tile with a corner on screen must be visible
            for x in (tiles.x0[k], tiles.x1[k]):
                for y in (tiles.y0[k], tiles.y1[k]):
                    for z in (tiles.zlo[k] * 2, tiles.zhi[k] * 2 + 20):
                        c = vp @ np.array([x, y, z, 1.0])
                        if c[3] > 0 and abs(c[0]) <= c[3] and abs(c[1]) <= c[3]:
                            assert vis[k]
        runs = merge_runs(tiles, vis, max_draws=8)
        assert len(runs) <= 8
        covered = np.zeros(n, bool)
        for first, cnt in runs:
            covered[first // 30:(first + cnt) // 30] = True
        assert covered[vis].all()
        runs_all = merge_runs(tiles, vis, max_draws=10_000)
        assert sum(c for _, c in runs_all) == int(vis.sum()) * 30         # without the cap: exactly the visible tiles


def test_frustum_planes_reject_behind():
    cam = OrbitCamera()
    cam.resize(800, 600)
    cam.set_view(0, 0, 1000.0, 0.0, 45.0)
    pl = frustum_planes(cam.view_proj())
    assert (pl @ np.append(cam.target, 1.0) >= -1e-6).all()
    assert (pl @ np.append(cam.eye() - cam.forward_dir() * 300.0, 1.0) < 0).any()


# ================================================================================================ selection
def aim(app, x, y, dist=200.0, yaw=0.0, pitch=50.0):
    app.camera.set_view(x, y, dist, yaw, pitch)


def pixel_of(app, p):
    return app.camera.project(np.asarray(p, float))[:2]


def test_pick_roof_wall_and_hole(patched):
    sc = patched
    b = sc.buildings
    app = make_app(sc, FakeStorm("a", sc.frame, {}))
    # a point on the roof (centre of its first roof triangle, so holes / L shapes are fine) -> that building
    for i in range(b.n):
        aim(app, float(b.cx[i]), float(b.cy[i]), 250.0, 20.0, 45.0)
        t = sc.pick.roof_tri[sc.pick.roof_off[i]].astype(float)
        roof = ((t[0] + t[2] + t[4]) / 3, (t[1] + t[3] + t[5]) / 3, sc.base_z[i] + b.height_m[i] - 0.05)
        px, py = pixel_of(app, roof)
        hit = selection.pick_building_ray(app, px, py)
        assert hit is not None and hit[0] == i, (i, hit)
    # looking straight through the courtyard hole of building 1 from above: no hit of building 1 (the ground is the target)
    aim(app, float(b.cx[1]), float(b.cy[1]), 120.0, 0.0, 89.0)
    px, py = pixel_of(app, (b.cx[1], b.cy[1], sc.base_z[1] + 0.5))
    hit = selection.pick_building_ray(app, px, py)
    assert hit is None or hit[0] != 1
    # a wall is hit as well: aim at the south wall of the tall tower from the south
    i = 2
    aim(app, float(b.cx[i]), float(b.cy[i] - 40), 120.0, 0.0, 25.0)
    px, py = pixel_of(app, (b.cx[i], b.cy[i] - math.sqrt(b.area_m2[i]) / 2 + 0.2, sc.base_z[i] + 15.0))
    hit = selection.pick_building_ray(app, px, py)
    assert hit is not None and hit[0] == 2


def test_pick_tall_building_occludes_ground_behind(patched):
    sc = patched
    b = sc.buildings
    app = make_app(sc, FakeStorm("a", sc.frame, {}))
    # camera south of the 30 m tower looking north at a low angle: the pixel of a point on the ground just behind it
    # is covered by the tower, so the click selects the tower (the old "ground ray only" logic would miss it)
    aim(app, float(b.cx[2]), float(b.cy[2]), 80.0, 0.0, 12.0)
    behind = (b.cx[2], b.cy[2] + 25.0, sc.base_z[2])
    px, py = pixel_of(app, behind)
    res = selection.pick_at_pixel(app, px, py)
    ray_ground = app.camera.pick(px, py)
    assert ray_ground is not None
    assert res["building"] == 2


def test_select_click_clears_on_miss_and_roads(patched):
    sc = patched
    b = sc.buildings
    app = make_app(sc, FakeStorm("a", sc.frame, {}))
    selection.ensure(app)
    aim(app, 0.0, 0.0, 600.0, 0.0, 70.0)
    px, py = pixel_of(app, (b.cx[0], b.cy[0], sc.base_z[0] + b.height_m[0]))
    sel = selection.select_at_pixel(app, px, py)
    assert sel["building"] == 0 and app.selection["building"] == 0
    r = sc.roads
    cand = np.flatnonzero(r.seg_class == 3)
    k = int(cand[np.argmin(np.abs(0.5 * (r.seg_x0[cand] + r.seg_x1[cand])))])   # a primary segment near the grid centre
    mx, my = 0.5 * (r.seg_x0[k] + r.seg_x1[k]), 0.5 * (r.seg_y0[k] + r.seg_y1[k])
    aim(app, float(mx), float(my), 150.0, 0.0, 85.0)
    px, py = pixel_of(app, (mx, my, sc.base_z[0]))
    sel = selection.select_at_pixel(app, px, py)
    assert sel["building"] is None and sel["road"] == int(r.seg_road[k])
    # empty ground clears both
    aim(app, 250.0, -250.0, 100.0, 0.0, 85.0)
    sel = selection.select_at_pixel(app, app.camera.width / 2, app.camera.height / 2)
    assert sel["building"] is None and sel["road"] is None


def test_tolerance_near_footprint(patched):
    sc = patched
    b = sc.buildings
    half = math.sqrt(b.area_m2[0]) / 2
    # a ground point 2 m outside the footprint edge counts as the building, 6 m outside does not
    app = make_app(sc, FakeStorm("a", sc.frame, {}))
    hit = selection.building_near_point(app, b.cx[0] + half + 2.0, b.cy[0], 3.0)
    assert hit is not None and hit[0] == 0 and 1.0 < hit[1] < 3.0
    assert selection.building_near_point(app, b.cx[0] + half + 6.0, b.cy[0], 3.0) is None
    assert selection.building_near_point(app, b.cx[0], b.cy[0], 3.0)[1] == 0.0


def test_fly_to_building_reaches_target(patched):
    sc = patched
    b = sc.buildings
    app = make_app(sc, FakeStorm("a", sc.frame, {}))
    aim(app, 0.0, 0.0, 5000.0, 123.0, 30.0)
    selection.fly_to_building(app, 3)
    assert app.selection["building"] == 3
    n = 0
    while selection.step_flight(app, 1 / 60):
        n += 1
        assert n < 400
    cam = app.camera
    assert 30 < n < 200
    assert cam.target[0] == pytest.approx(float(b.cx[3]), abs=0.01) and cam.target[1] == pytest.approx(float(b.cy[3]), abs=0.01)
    assert cam.dist == pytest.approx(120.0) and cam.pitch == pytest.approx(50.0) and cam.yaw == pytest.approx(123.0)
    # user input cancels a flight
    selection.fly_to_building(app, 0)
    selection.step_flight(app, 0.1)
    app.camera._drag = {"button": 1, "x": 0, "y": 0}
    assert selection.step_flight(app, 0.1) is False and getattr(app, "_flight", None) is None


def test_describe_building_uses_displayed_snapshot(patched):
    sc = patched
    fr = sc.frame
    g = flood_grid(fr, {(6, 6): 1.5})
    storm = FakeStorm("a", fr, {12: g})
    app = make_app(sc, storm, 12.0)
    hub = stats_hub.get_hub(app)
    assert drain(hub, app, lambda: app.flood["ready"])
    d = selection.describe_building(app, 0)
    assert d["flooded"] and d["tier"] == 4 and d["depth_m"] == pytest.approx(ST.decode_depth(depth_to_code(1.5)), abs=1e-4)
    assert d["rank"] == 1 and d["hour"] == 12
    assert selection.describe_building(app, 3)["flooded"] is False
    hub.close()


# ================================================================================================ real data
@pytest.fixture(scope="module")
def real_scene():
    from data_helpers import real_frame
    from bgv.data import terrain
    from bgv.gpu.camera import HeightSampler
    root = real_root()
    fr = real_frame()
    try:
        filled, _mips, _stats, _valid = terrain.load_terrain_gpu_data(root, fr)
    except Exception as e:
        pytest.skip(f"terrain not available: {e}")
    sc = scene_data.SceneData(root, fr, filled, HeightSampler(filled, *fr.size_m))
    assert sc.wait(120), sc.error
    return sc


def test_real_known_good_numbers_via_snapshot(real_scene):
    from bgv.data import flow
    root = real_root()
    st = flow.Storm.open(root, "25yr")
    s = stats_hub.compute_snapshot(st, 12, real_scene)
    assert s.buildings.flooded_count == 1309
    assert s.tier_counts == {"shallow": 831, "moderate": 363, "deep": 107, "severe": 8}
    assert s.grid.wet_area_km2 == pytest.approx(14.06, abs=0.01)
    assert s.grid.max_depth_m == pytest.approx(5.90, abs=0.01)
    assert s.roads.flooded_pct == pytest.approx(2.1, abs=0.05)
    st.close()


def test_real_base_z_spot_checks(real_scene):
    sc = real_scene
    b, hts = sc.buildings, sc.heights
    assert np.isfinite(sc.base_z).all()
    W = hts.shape[1]
    rng = np.random.default_rng(11)
    ids = rng.choice(np.flatnonzero(b.in_grid), 5, replace=False)
    for i in ids:                                                   # base == min of the terrain under the footprint pixels
        px = b.index.pixels_of(int(i))
        under = hts.reshape(-1)[px]
        assert sc.base_z[i] == pytest.approx(under.min(), abs=1e-4)
        # ... and the centroid's bilinear ground height is within the footprint's relief + 1 cm
        zc = float(sc.sampler.sample(b.cx[i], b.cy[i]))
        assert under.min() - 1.0 <= zc <= under.max() + 1.0
    # base (= min under the footprint) is within 1.5 m of the ground at the centroid for 98 % of buildings (the rest are big
    # buildings on slopes: they are buried on the uphill side, never floating)
    ok = np.isfinite(b.cx) & b.in_grid
    d = np.abs(sc.base_z[ok] - sc.sampler.sample(b.cx[ok], b.cy[ok]))
    assert (d < 1.5).mean() > 0.98
    assert (sc.base_z[ok] - sc.sampler.sample(b.cx[ok], b.cy[ok]) < 0.75).mean() > 0.995       # floating >0.75 m: <0.5 %


def test_real_scene_structures(real_scene):
    sc = real_scene
    b = sc.buildings
    tb = b.mesh.building_id[b.mesh.indices.reshape(-1, 3)[:, 0]]
    assert sc.mesh.triangle_count == int(b.in_grid[tb].sum()) < b.mesh.triangle_count == 320304   # off-grid buildings not drawn
    assert sc.mesh.tiles.count.sum() == sc.mesh.indices.size
    assert 0 < sc.road_mesh.indices.size < sc.roads.ribbon.indices.size                            # off-grid segments dropped
    rm = sc.road_mesh
    half = np.array(sc.frame.size_m) * 0.5
    seg_xy = rm.vertices["pos"][np.unique(rm.indices)]
    assert (np.abs(seg_xy[:, 0]) <= half[0] + 30).all() and (np.abs(seg_xy[:, 1]) <= half[1] + 30).all()
    # roof triangle area == footprint area for the first 200 buildings (exact footprint available for picking)
    pk = sc.pick
    for i in range(200):
        t = pk.roof_tri[pk.roof_off[i]:pk.roof_off[i + 1]].astype(np.float64)
        if not len(t):
            continue
        area = 0.5 * np.abs((t[:, 2] - t[:, 0]) * (t[:, 5] - t[:, 1]) - (t[:, 4] - t[:, 0]) * (t[:, 3] - t[:, 1])).sum()
        assert area == pytest.approx(float(sc.buildings.area_m2[i]), rel=0.02, abs=0.2)


# ================================================================================================ GPU smoke tests
gpu = pytest.mark.gpu


@pytest.fixture(scope="module")
def gctx():
    wgpu = pytest.importorskip("wgpu")
    from bgv.gpu.context import GpuContext
    try:
        return GpuContext.create(verbose=False)
    except Exception as e:
        pytest.skip(f"no GPU adapter: {e}")


def _render(app, ctx, w, h):
    import wgpu
    tex = ctx.device.create_texture(size=(w, h, 1), format=app.colour_format,
                                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    app.anim_time = 2.0
    app.step(0.0)
    app.render_scene(tex.create_view(), w, h, 0.0)
    app.render_scene(tex.create_view(), w, h, 0.0)
    raw = ctx.queue.read_texture({"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
                                 {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3].astype(np.float32) / 255.0


@pytest.fixture(scope="module")
def real_app(gctx):
    from bgv import config, tools
    from bgv.app import App
    real_root()
    app = App(gctx, config.data_root(), storm="25yr", msaa=4, quiet=True, settings={})
    app.t = 24.0
    tools.wait_for_flow(app)
    yield app
    app.shutdown()


@gpu
def test_layers_register_and_render(real_app, gctx):
    app = real_app
    names = [l.name for l in app.layers]
    assert "buildings" in names and "roads" in names
    assert names.index("roads") < names.index("buildings") < names.index("water")
    assert app.flood["ready"] and app.flood["flooded_count"] == 1827           # 25yr hour 24 (regression, same kernels)
    app.camera.set_view(-1400.0, 1000.0, 400.0, 30.0, 45.0)
    on = _render(app, gctx, 480, 270)
    for n in ("buildings", "roads"):
        app.get_layer(n).enabled = False
    off = _render(app, gctx, 480, 270)
    for n in ("buildings", "roads"):
        app.get_layer(n).enabled = True
    assert np.abs(on - off).mean() > 0.01                                      # buildings / roads really change the picture
    assert not app.errors
    b = app.get_layer("buildings")
    assert b._uploaded and b.tris_drawn > 0


@gpu
def test_flooded_buildings_show_their_tier_colour(real_app, gctx):
    """Top-down render with water / particles off: every large, unoccluded building's roof pixel is grey when the statistics
    say dry and tinted with the right tier hue when they say flooded."""
    import colorsys
    app = real_app
    sc = app._scene_data
    b = sc.buildings
    fl = app.flood["buildings"]
    hidden = [app.get_layer(n) for n in ("water", "particles", "arrows", "overlay") if app.get_layer(n) is not None]
    for l in hidden:
        l.enabled = False
    app.camera.set_view(-1400.0, 1000.0, 520.0, 0.0, 89.5)
    w, h = 1280, 720
    img = _render(app, gctx, w, h)
    for l in hidden:
        l.enabled = True
    hue_ref = {1: 205.0, 2: 180.0, 3: 33.0, 4: 6.0}
    n_fl = n_dry = ok_fl = ok_dry = 0
    pk = sc.pick
    for i in np.flatnonzero(np.isfinite(b.cx) & (b.area_m2 > 45)):
        t = pk.roof_tri[pk.roof_off[i]:pk.roof_off[i + 1]].astype(float)          # centre of the largest roof triangle: on the roof
        ar = np.abs((t[:, 2] - t[:, 0]) * (t[:, 5] - t[:, 1]) - (t[:, 4] - t[:, 0]) * (t[:, 3] - t[:, 1]))
        t = t[int(np.argmax(ar))]
        cx, cy = (t[0] + t[2] + t[4]) / 3, (t[1] + t[3] + t[5]) / 3
        px, py, _ = app.camera.project((cx, cy, sc.base_z[i] * app.exag + b.height_m[i]))
        if not (40 < px < w - 40 and 40 < py < h - 40):
            continue
        r, g, bl = img[int(py), int(px)]
        hh, ss, vv = colorsys.rgb_to_hsv(r, g, bl)
        if fl.flooded[i]:
            n_fl += 1
            tier = int(fl.tier[i])
            dh = min(abs(hh * 360 - hue_ref[tier]), 360 - abs(hh * 360 - hue_ref[tier]))
            ok_fl += int(ss > 0.25 and dh < 22)
        else:
            n_dry += 1
            ok_dry += int(ss < 0.2)
    print(f"tier-colour check: flooded {ok_fl}/{n_fl}, dry {ok_dry}/{n_dry}")
    assert n_fl >= 15 and n_dry >= 40, (n_fl, n_dry)
    assert ok_fl / n_fl > 0.9, (ok_fl, n_fl)
    assert ok_dry / n_dry > 0.97, (ok_dry, n_dry)


@gpu
def test_selection_highlight_and_outline(real_app, gctx):
    app = real_app
    sc = app._scene_data
    i = int(np.flatnonzero(app.flood["buildings"].flooded)[0])
    b = sc.buildings
    app.camera.set_view(float(b.cx[i]), float(b.cy[i]), 150.0, 20.0, 50.0)
    selection.clear_selection(app)
    base = _render(app, gctx, 400, 300)
    selection.select_building(app, i)
    sel = _render(app, gctx, 400, 300)
    selection.clear_selection(app)
    d = np.abs(sel - base).sum(axis=2)
    assert (d > 0.15).sum() > 150                                              # highlight + outline cover real pixels
    assert not app.errors
