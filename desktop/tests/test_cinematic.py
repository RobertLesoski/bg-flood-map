"""Cinematic tools, CPU side: bookmarks, flights, time-lapse, tour route / path / schedule, depth tiles (synthetic COGs and the
real ones when present), export helpers (stamp, scripts), UI hook registry, layer state."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bgv import tools_cinematic as tc
from bgv.data import depthtiles as dt

REAL = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")


# ------------------------------------------------------------------------------------------------ helpers
class FakeCam:
    def __init__(self):
        self.target = np.zeros(3)
        self.dist, self.yaw, self.pitch = 1000.0, 0.0, 50.0
        self.moving = False

    def set_view(self, x, y, d, yaw, pitch):
        self.target[:2] = (x, y)
        self.dist, self.yaw, self.pitch = d, yaw % 360.0, pitch


class FakeFrame:
    """lng/lat <-> world metres with a flat local projection."""
    def lnglat_to_world(self, lng, lat):
        return (lng + 86.44) * 90000.0, (lat - 36.99) * 111000.0

    def world_to_lnglat(self, x, y):
        return x / 90000.0 - 86.44, y / 111000.0 + 36.99


def write_root(tmp_path, names=True):
    """A tiny data root: infrastructure.geojson (named + unnamed), roads_flooded and the county road geometry."""
    L = tmp_path / "layers"
    (L / "roads_flooded" / "city").mkdir(parents=True)
    (L / "county").mkdir(parents=True)
    feats = []
    spec = [("Alpha Hospital", "hospital", -86.46, 36.98, 150), ("Beta Hospital", "hospital", -86.44, 37.00, 20),
            ("", "hospital", -86.42, 37.01, 400), ("Gamma School", "school", -86.45, 36.99, 80), ("Delta School", "school", -86.43, 36.97, 0),
            ("Epsilon Fire", "fire_station", -86.41, 36.99, 0), ("Zeta Police", "police", -86.47, 37.02, 30),
            ("Eta Shelter", "assembly_point", -86.40, 37.00, 0), ("Theta Clinic", "clinic", -86.39, 36.98, 5)]
    for n, k, lng, lat, cm in spec:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lng, lat]},
                      "properties": {"name": n, "type": k, "lng": lng, "lat": lat, "d_25yr": cm}})
    (L / "infrastructure.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    rows = [[11, "First Road", 3, 0, 100, 90, 90, 500], [12, "", 9, 0, 100, 90, 90, 480], [13, "Second Road", 8, 0, 100, 50, 50, 300],
            [14, "First Road", 8, 0, 50, 50, 50, 200], [15, "Third Road", 9, 0, 50, 50, 50, 100]]
    (L / "roads_flooded" / "city" / "25yr.json").write_text(json.dumps({"rows": rows}))
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"osm_id": i, "name": "x"},
         "geometry": {"type": "LineString", "coordinates": [[-86.44 + 0.001 * i, 36.99], [-86.44 + 0.001 * i + 0.004, 36.991]]}}
        for i in (11, 12, 13, 14, 15)]}
    with gzip.open(L / "county" / "roads.geojson.gz", "wt", encoding="utf-8") as fh:
        json.dump(geo, fh)
    return tmp_path


# ------------------------------------------------------------------------------------------------ bookmarks
def test_bookmarks_add_rename_delete_per_domain():
    s = tc.BookmarkStore()
    i = s.add("city", "", (1, 2, 300, 10, 45), t=5.0, storm="25yr")
    assert s.get("city", i).name == "View 1"
    s.add("city", "Downtown", (3, 4, 500, 20, 50))
    s.add("county", "Smiths Grove", (9, 9, 900, 0, 55))
    assert [b.name for b in s.list("city")] == ["View 1", "Downtown"]
    assert [b.name for b in s.list("county")] == ["Smiths Grove"]
    s.rename("city", 0, "  River  ")
    assert s.get("city", 0).name == "River"
    s.rename("city", 0, "   ")                             # a blank name is ignored
    assert s.get("city", 0).name == "River"
    s.add("city", None, (7, 7, 700, 0, 40), slot=1)         # overwrite slot 1, name kept
    assert s.get("city", 1).name == "Downtown" and s.get("city", 1).view[0] == 7
    s.move("city", 1, 0)
    assert [b.name for b in s.list("city")] == ["Downtown", "River"]
    s.delete("city", 0)
    s.delete("city", 99)                                    # out of range: ignored
    assert [b.name for b in s.list("city")] == ["River"]
    assert s.next_name("city") not in {b.name for b in s.list("city")}


def test_bookmarks_json_roundtrip_and_garbage():
    s = tc.BookmarkStore()
    s.add("city", "A", (1.5, -2.5, 300, 10, 45), t=3.25, storm="5yr")
    s.add("county", "B", (0, 0, 9000, 0, 55))
    d = json.loads(json.dumps(s.to_dict()))
    r = tc.BookmarkStore().load(d)
    assert r.get("city", 0).to_dict() == s.get("city", 0).to_dict()
    assert r.get("county", 0).t is None
    g = tc.BookmarkStore().load({"city": [{"name": "ok", "view": [1, 2, 3, 4, 5]}, {"view": [1, 2]}, "junk", {"view": [1, 2, 3, 4, float("nan")]}],
                                 "x": 5})
    assert [b.name for b in g.list("city")] == ["ok"]
    assert tc.BookmarkStore().load("not a dict").to_dict() == {}
    with pytest.raises(ValueError):
        t = tc.BookmarkStore()
        for _ in range(tc.MAX_BOOKMARKS + 1):
            t.add("city", None, (0, 0, 1, 0, 40))


# ------------------------------------------------------------------------------------------------ flight
def test_view_lerp_endpoints_and_rise():
    a, b = (0.0, 0.0, 1000.0, 350.0, 40.0), (6000.0, 0.0, 500.0, 10.0, 60.0)
    assert tc.view_lerp(a, b, 0.0) == pytest.approx(a)
    assert tc.view_lerp(a, b, 1.0) == pytest.approx(b)
    mid = tc.view_lerp(a, b, 0.5)
    assert mid[2] > math.sqrt(a[2] * b[2])                  # a long hop rises
    assert mid[3] == pytest.approx(0.0, abs=1e-9) or mid[3] == pytest.approx(360.0)      # yaw 350 -> 10 goes through north
    assert tc.flight_seconds(a, b) > tc.flight_seconds(a, (100.0, 0.0, 1000.0, 350.0, 40.0)) >= 0.9


def test_viewflight_reaches_target_and_yields_to_user():
    cam = FakeCam()
    f = tc.ViewFlight()
    f.start(cam, (500.0, 200.0, 800.0, 90.0, 55.0), seconds=1.0)
    xs = []
    while f.update(cam, 1 / 60):
        xs.append(cam.target[0])
    assert cam.target[:2] == pytest.approx([500.0, 200.0]) and cam.dist == pytest.approx(800.0) and cam.yaw == pytest.approx(90.0)
    assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))                     # monotone, eased
    d = np.diff(xs)
    assert d[0] < d.max() and d[-1] < d.max()                               # slow at both ends
    cam2 = FakeCam()
    f.start(cam2, (500.0, 0.0, 800.0, 0.0, 50.0), seconds=2.0)
    f.update(cam2, 0.1)
    cam2.moving = True                                                      # the user grabs the camera
    f.update(cam2, 0.1)
    assert not f.active


# ------------------------------------------------------------------------------------------------ time-lapse
def test_timelapse_modes():
    lp = tc.TimeLapse(rate=2.0, t_in=4.0, t_out=8.0, mode="loop")
    t = 7.0
    for _ in range(10):
        t = lp.step(t, 0.25)
        assert 4.0 <= t <= 8.0
    assert lp.step(7.9, 0.2) == pytest.approx(4.0 + 0.3)                    # wraps
    pp = tc.TimeLapse(rate=2.0, t_in=4.0, t_out=8.0, mode="pingpong")
    seq = []
    t = 4.0
    for _ in range(30):
        t = pp.step(t, 0.25)
        seq.append(t)
    assert max(seq) <= 8.0 and min(seq) >= 4.0 and pp.direction in (1, -1)
    assert any(b < a for a, b in zip(seq, seq[1:]))                         # turned around
    once = tc.TimeLapse(rate=4.0, t_in=0.0, t_out=2.0, mode="once", playing=True)
    t = 0.0
    for _ in range(10):
        t = once.step(t, 0.25)
    assert t == 2.0 and not once.playing


def test_rate_slider_roundtrip_and_load_clamp():
    for r in (0.05, 0.25, 1.0, 6.0, 12.0):
        assert tc.rate_from_slider(tc.slider_from_rate(r)) == pytest.approx(r)
    lp = tc.TimeLapse()
    lp.load({"rate": 999, "t_in": -5, "t_out": 90, "mode": "bogus"})
    assert lp.rate == tc.RATE_MAX and lp.t_in == 0.0 and lp.t_out == 24.0 and lp.mode == "loop"
    lp.load("garbage")


def test_timelaw():
    law = tc.TimeLaw("range", 6.0, 18.0, 1.0)
    assert law.at(0) == 6.0 and law.at(1) == 18.0 and law.at(0.5) == pytest.approx(12.0)
    two = tc.TimeLaw("range", 0.0, 24.0, 2.0)
    assert two.at(0.25) == pytest.approx(12.0) and two.at(0.5) == pytest.approx(0.0) and two.at(1.0) == 24.0
    assert tc.TimeLaw("hold", hold=9.0).at(0.7) == 9.0
    e = tc.TimeLaw("range", 0.0, 24.0, 1.0, ease=True)
    assert e.at(0.1) < 2.4 < e.at(0.9) + 24 * 0.2 and e.at(0.5) == pytest.approx(12.0)


# ------------------------------------------------------------------------------------------------ route
def test_build_stops_names_round_robin_and_unnamed(tmp_path):
    root = write_root(tmp_path)
    stops, info = tc.build_stops(root, "city", "25yr", FakeFrame(), max_facilities=5, max_roads=2)
    assert info["unnamed"] == 1                                             # the hospital without a name is never used
    assert all(s.name.strip() for s in stops)
    fac = [s for s in stops if s.kind != "road"]
    assert len(fac) == 5 and {s.kind for s in fac} >= {"hospital", "school", "fire_station", "police", "assembly_point"}
    assert [s.name for s in fac if s.kind == "hospital"] == ["Alpha Hospital"]  # round robin: one of each type first
    roads = [s for s in stops if s.kind == "road"]
    assert [r.name for r in roads] == ["First Road", "Second Road"]         # deepest first, one row per name, unnamed row skipped
    assert all(s.peak_m is not None for s in stops)
    assert roads[0].peak_m == pytest.approx(5.0)                            # from the file's max_cm without a depth source


def test_build_stops_uses_the_depth_source_for_ranking(tmp_path):
    root = write_root(tmp_path)
    calls = []

    def peak_many(pts):
        calls.append(len(pts))
        return [lat - 36.0 for lng, lat in pts]                             # deeper to the north, whatever the files say
    stops, _ = tc.build_stops(root, "city", "25yr", FakeFrame(), peak_many=peak_many, types=("school",), max_facilities=1, max_roads=0)
    assert stops[0].name == "Gamma School" and stops[0].peak_m == pytest.approx(0.99)      # lat 36.99 beats 36.97
    assert len(calls) == 1                                                  # one batch for all facilities
    stops, _ = tc.build_stops(root, "city", "25yr", FakeFrame(), peak_many=peak_many, types=("school",), only_wet=True,
                              max_facilities=5, max_roads=0, wet_m=0.995)
    assert stops == []


def test_order_route_is_short():
    rng = np.random.default_rng(3)
    stops = [tc.Stop(f"s{i}", "school", 0, 0, float(x), float(y)) for i, (x, y) in enumerate(rng.uniform(-5000, 5000, (14, 2)))]

    def length(seq):
        return sum(math.hypot(a.x - b.x, a.y - b.y) for a, b in zip(seq, seq[1:]))
    ordered = tc.order_route(stops, (0.0, 0.0))
    assert sorted(s.name for s in ordered) == sorted(s.name for s in stops)
    assert length(ordered) < 0.7 * length(stops)


def stops_line(n=4, step=1500.0):
    return [tc.Stop(f"P{i}", "hospital" if i % 2 else "road", 0, 0, i * step, (i % 2) * 700.0) for i in range(n)]


def test_tourpath_continuous_eased_and_closed():
    start = (-500.0, -300.0, 9000.0, 30.0, 55.0)
    p = tc.TourPath(stops_line(), start, dwell_s=5.0)
    assert p.at(0.0).view == pytest.approx(start)
    end = p.at(p.duration).view
    assert end[:3] == pytest.approx(start[:3]) and end[4] == pytest.approx(start[4])
    # no jump anywhere: neighbouring samples (1/30 s) differ by little in position, distance and heading
    prev = p.at(0.0).view
    worst_xy = 0.0
    for i in range(1, int(p.duration * 30) + 1):
        v = p.at(i / 30.0).view
        worst_xy = max(worst_xy, math.hypot(v[0] - prev[0], v[1] - prev[1]))
        assert abs(((v[3] - prev[3] + 180) % 360) - 180) < 3.0
        assert abs(v[2] - prev[2]) / prev[2] < 0.05
        prev = v
    assert worst_xy < 250.0
    # arrivals: at rest (speed ~ 0), the camera is over the stop, caption only while dwelling
    times = p.stop_times()
    assert len(times) == 4
    for i, (arr, dep) in enumerate(times):
        a, b = p.at(arr + 1e-3), p.at(arr + 0.4)
        assert math.hypot(a.view[0] - p.stops[i].x, a.view[1] - p.stops[i].y) < 1.0
        assert p.at((arr + dep) / 2).caption_stop == i and p.at((arr + dep) / 2).caption_alpha == pytest.approx(1.0)
        assert p.at(arr - 0.5).caption_stop is None
    assert p.at(times[0][0]).caption_alpha < 0.5                            # fades in
    # dwell orbit
    a, b = p.at(times[1][0] + 1e-3).view[3], p.at(times[1][1] - 1e-3).view[3]
    assert ((b - a + 180) % 360) - 180 == pytest.approx(tc.ORBIT_DEG, abs=0.5)
    # skipping
    assert p.next_start(0.0) == pytest.approx(times[0][0])
    assert p.next_start(times[0][0] + 1.0) == pytest.approx(times[1][0])
    assert p.next_start(times[-1][0] + 1.0) == pytest.approx(p.duration)
    assert p.prev_start(times[2][0] + 0.5) == pytest.approx(times[1][0])
    assert p.prev_start(times[2][0] + 3.0) == pytest.approx(times[2][0])


def test_tourpath_single_stop_and_no_return():
    p = tc.TourPath(stops_line(1), (0, 0, 5000, 0, 50), return_to_start=False)
    assert p.duration > 0 and p.at(p.duration).caption_stop == 0
    with pytest.raises(ValueError):
        tc.TourPath([], (0, 0, 5000, 0, 50))


def test_tourplan_storms_and_hours():
    p = tc.TourPath(stops_line(6), (0, 0, 9000, 0, 55), dwell_s=4.0)
    plan = tc.TourPlan(storms=["5yr", "25yr", "100yr"], time_mode="sweep")
    seen, hours = [], []
    for i in range(0, int(p.duration * 10) + 1):
        storm, h = plan.at(p, i / 10.0)
        seen.append(storm)
        hours.append(h)
        assert 0.0 <= h <= 24.0
    order = [s for k, s in enumerate(seen) if k == 0 or s != seen[k - 1]]
    assert order == ["5yr", "25yr", "100yr"]                                # escalates, each storm once
    drops = [k for k in range(1, len(hours)) if hours[k] < hours[k - 1] - 1]
    assert len(drops) == 2                                                  # the hour restarts at each new storm
    assert hours[0] == pytest.approx(0.0) and hours[-1] == pytest.approx(24.0, abs=0.2)
    hold = tc.TourPlan(storms=[], time_mode="hold", hold_t=7.5)
    assert hold.at(p, 3.0) == (None, 7.5)
    one = tc.TourPlan(storms=["25yr"], time_mode="sweep")
    assert one.at(p, 0.0) == ("25yr", 0.0) and one.at(p, p.duration)[1] == pytest.approx(24.0)


def test_shots():
    law = tc.TimeLaw("range", 0.0, 24.0)
    o = tc.OrbitShot((10, 20, 1000, 90, 50), 4.0, law, 180.0)
    s0, s1 = o.state_at(0), o.state_at(4.0)
    assert s0.view[3] == 90 and s1.view[3] == pytest.approx(270.0) and s0.t == 0.0 and s1.t == 24.0
    f = tc.FlightShot((0, 0, 1000, 0, 40), (1000, 0, 500, 90, 50), 2.0, law)
    assert f.state_at(1.0).view[0] == pytest.approx(500.0) and f.state_at(2.0).view[:3] == pytest.approx((1000, 0, 500))
    p = tc.TourPath(stops_line(3), (0, 0, 9000, 0, 55))
    shot = tc.TourShot(p, tc.TourPlan(storms=["25yr"]), speed=2.0, caption_fn=lambda stop, storm, t: f"depth of {stop.name}")
    assert shot.duration == pytest.approx(p.duration / 2)
    mid = p.stop_times()[1]
    st = shot.state_at((mid[0] + mid[1]) / 2 / 2.0)
    assert st.caption["title"] == "P1" and st.caption["extra"] == "depth of P1" and st.storm == "25yr"
    assert tc.clip_frames(2.5, 30) == 75 and tc.safe_name("25yr: tour/1") == "25yr_tour_1"


def test_real_route_names_come_from_the_data_files():
    if not (REAL / "layers" / "infrastructure.geojson").is_file():
        pytest.skip("real data not present")
    stops, info = tc.build_stops(REAL, "city", "25yr", FakeFrame(), max_facilities=12, max_roads=5)
    names = {str((f.get("properties") or {}).get("name") or "").strip() for f in
             json.loads((REAL / "layers" / "infrastructure.geojson").read_text(encoding="utf-8"))["features"]}
    rows = {str(r[1]).strip() for r in tc.flooded_road_rows(REAL, "city", "25yr")}
    for s in stops:
        assert s.name and (s.name in names if s.kind != "road" else s.name in rows)
    assert info["unnamed"] > 0 and any(s.kind == "road" for s in stops)


# ------------------------------------------------------------------------------------------------ depth tiles
def make_cog(path, arr, left=-9640000.0, top=4450000.0, res=6.25, nodata=65535):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint16", crs="EPSG:3857",
                       transform=from_origin(left, top, res, res), nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate") as d:
        d.write(arr.astype(np.uint16), 1)


def synth_tiles(tmp_path, hours=(0, 6, 12, 24)):
    folder = tmp_path / "layers" / "cog" / "county" / "100yr"
    folder.mkdir(parents=True)
    n = 400
    for h in hours:
        a = np.zeros((n, n), np.uint16)
        a[190:210, 190:210] = 10 * h                                       # cm: a 20 x 20 pixel pond that deepens with the hour
        a[0:50, 0:50] = 65535                                              # nodata corner
        make_cog(folder / f"depth_{h * 3600 + (4 if h == 6 else 0):06d}s.tif", a)
    mx = np.zeros((n, n), np.uint16)
    mx[190:210, 190:210] = 300
    make_cog(folder / "max_depth.tif", mx)
    return folder


def centre_lnglat(n=400, left=-9640000.0, top=4450000.0, res=6.25):
    x, y = left + 200 * res, top - 200 * res
    lng = math.degrees(x / dt.R_MERC)
    lat = math.degrees(2 * math.atan(math.exp(y / dt.R_MERC)) - math.pi / 2)
    return lng, lat


def test_tile_helpers(tmp_path, monkeypatch):
    assert dt.hour_of_seconds(15) == 0 and dt.hour_of_seconds(3604) == 1 and dt.hour_of_seconds(86400) == 24
    assert dt.tile_dir_name({}) == "cog"
    assert dt.tile_dir_name({"cinematic": {"tile_dir": "cog_native"}}) == "cog_native"
    monkeypatch.setenv("BGV_TILE_DIR", "fine")
    assert dt.tile_dir_name({}) == "fine" and dt.tile_dir_name({"cinematic": {"tile_dir": "x"}}) == "x"
    assert dt.tile_label(6.25, False) == "solver depth (tiles, 6.25 m; verification pending)"
    assert dt.tile_label(6.25, True) == "solver depth (tiles, 6.25 m; verified)"
    f = tmp_path / "p.md"
    assert dt.tiles_verified(f) is False                                     # missing file: not verified
    f.write_text("# log\n- Status: pending\nverified in the browser harness only\n")
    assert dt.tiles_verified(f) is False
    f.write_text("# log\nStatus: Verified\n")
    assert dt.tiles_verified(f) is True
    f.write_text("- Verification: verified (200 points per storm)\n")
    assert dt.tiles_verified(f) is True


def test_tiledepth_reads_windows_nodata_and_mixes(tmp_path):
    pytest.importorskip("rasterio")
    folder = synth_tiles(tmp_path)
    t = dt.TileDepth(folder, verified=False)
    assert t.kind == "tiles" and t.label == "solver depth (tiles, 6.25 m; verification pending)"
    assert sorted(t.files) == [0, 6, 12, 24]                                 # depth_021604s -> hour 6
    lng, lat = centre_lnglat()
    s = t.series(lng, lat, 10.0)
    assert s.shape == (25,)
    assert s[0] == 0.0 and s[6] == pytest.approx(0.6) and s[12] == pytest.approx(1.2) and s[24] == pytest.approx(2.4)
    assert np.isnan(s[3]) and np.isnan(s[20])                                # files that do not exist
    assert t.depth_hour(12, lng, lat, 10.0) == pytest.approx(1.2)
    assert t.depth_at_t(6.0, lng, lat, 10.0) == pytest.approx(0.6)
    assert t.peak(lng, lat, 10.0) == pytest.approx(3.0)                      # from max_depth.tif
    # radius: a point 100 m east of the pond sees it only with a big enough radius
    dlng = 100.0 / (111320.0 * math.cos(math.radians(lat)))
    assert t.depth_hour(12, lng + dlng, lat, 10.0) == 0.0
    assert t.depth_hour(12, lng + dlng, lat, 150.0) == pytest.approx(1.2)
    # nodata corner reads 0 (outside the model), outside the tile is NaN
    x0, y0 = -9640000.0 + 10 * 6.25, 4450000.0 - 10 * 6.25
    ll = (math.degrees(x0 / dt.R_MERC), math.degrees(2 * math.atan(math.exp(y0 / dt.R_MERC)) - math.pi / 2))
    assert t.depth_hour(12, ll[0], ll[1], 5.0) == 0.0
    assert math.isnan(t.depth_hour(12, -80.0, 30.0, 5.0))
    # batch + cache
    many = t.series_many([(lng, lat), (lng + dlng, lat)], 10.0)
    assert many[0][12] == pytest.approx(1.2) and many[1][12] == 0.0
    assert len(t._cache) > 0
    t.close()


def test_open_depth_source_prefers_tiles_then_flowgrid(tmp_path):
    pytest.importorskip("rasterio")
    synth_tiles(tmp_path)

    class Dom:
        key, demo = "county", False

    class Storm:
        name, synthetic, source = "100yr", False, "rasters"
        meta = {"cell_ground_m": 20.0}
        cell_ground_m = 20.0
        frame = FakeFrame()

        def load_hour(self, h):
            return np.zeros((4, 4, 4), np.uint8)

        def sample_world(self, a, b, mix, X, Y):
            return {"depth": 0.5, "valid": True}

    class App:
        domain, storm, root, settings = Dom(), Storm(), tmp_path, {}

        def report_message(self, *a, **k):
            pass
    src = dt.open_depth_source(App())
    assert src.kind == "tiles"
    App.domain = type("D", (), {"key": "county", "demo": True})()            # synthetic demo flood: never real tile depths
    fg = dt.open_depth_source(App())
    assert fg.kind == "flowgrid" and "DEMO" in fg.label
    App.domain = type("D", (), {"key": "city", "demo": False})()             # no city tiles: flow grid
    fg = dt.open_depth_source(App())
    assert fg.kind == "flowgrid" and "20 m flow grid" in fg.label
    s = fg.series(-86.44, 36.99, 25.0)
    assert s.shape == (25,) and np.allclose(s, 0.5)
    assert dt.open_depth_source(App(), "5yr").kind == "none"                 # another storm without tiles: no data


def test_flowgrid_batch_reads_each_hour_once():
    loads = []

    class Storm:
        name, synthetic, source = "25yr", False, "tiers"
        meta = {"cell_ground_m": 5.0}
        frame = FakeFrame()

        def load_hour(self, h):
            loads.append(h)
            return h

        def sample_world(self, a, b, mix, X, Y):
            return {"depth": float(a) * 0.1, "valid": True}
    fg = dt.FlowGridDepth(Storm())
    res = fg.series_many([(-86.44 + 0.001 * i, 36.99) for i in range(20)], 25.0)
    assert len(res) == 20 and loads == list(range(25))                       # 25 decodes for 20 points, not 500
    assert res[3][10] == pytest.approx(1.0) and fg.peak_many([(-86.44, 36.99)])[0] == pytest.approx(2.4)
    assert fg.series_many([(-86.44, 36.99)], 25.0)[0][5] == pytest.approx(0.5) and loads == list(range(25))   # cached


def test_series_cache_batches_requests():
    batches = []

    class Src(dt.DepthSource):
        kind = "tiles"

        def series_many(self, pts, radius_m=25.0):
            batches.append(len(pts))
            import time
            time.sleep(0.05)
            return [np.full(25, lng) for lng, lat in pts]
    c = dt.SeriesCache()
    s = Src()
    assert c.get(s, 1.0, 2.0) is None
    for i in range(2, 12):
        assert c.get(s, float(i), 2.0) is None
    v = c.get_blocking(s, 11.0, 2.0, timeout=10)
    assert v[0] == 11.0 and c.get(s, 1.0, 2.0)[0] == 1.0
    assert sum(batches) == 11 and len(batches) <= 3                          # the ten late requests shared a batch or two
    c.close()


@pytest.mark.skipif(not (REAL / "layers" / "cog" / "county" / "100yr" / "max_depth.tif").is_file(), reason="real tiles not present")
def test_real_tiles():
    pytest.importorskip("rasterio")
    t = dt.TileDepth(dt.tile_folder(REAL, "county", "100yr"))
    assert len(t.files) == 25 and t.resolution_m == pytest.approx(6.25)
    assert t.label.startswith("solver depth (tiles, 6.25 m; ")
    s = t.series(-86.4436, 36.9903, 25.0)
    assert s.shape == (25,) and np.isfinite(s).all() and (s >= 0).all() and s.max() < 30.0
    assert t.peak(-86.4436, 36.9903, 25.0) >= s.max() - 1e-6                 # the storm maximum is never below an hourly value
    assert math.isnan(t.depth_hour(12, -80.0, 30.0))
    t.close()


# ------------------------------------------------------------------------------------------------ export helpers
def test_export_helpers(tmp_path):
    from PIL import Image
    from bgv import export_cinematic as ex
    assert ex.even(1921) == 1920 and ex.even(99999) == ex.MAX_SIDE and ex.even(3) == 16
    scripts = ex.write_build_scripts(tmp_path, 30, "clip.mp4")
    bat = (tmp_path / "make_mp4.bat").read_text()
    assert "ffmpeg -y -framerate 30 -i frame_%06d.png" in bat.replace("%%", "%") and "clip.mp4" in bat and len(scripts) == 2
    sh = (tmp_path / "make_mp4.sh").read_bytes()
    assert b"\r" not in sh and sh.startswith(b"#!/bin/sh")
    cmd = ex.ffmpeg_command("ffmpeg", 24, tmp_path, tmp_path / "o.mp4")
    assert cmd[0] == "ffmpeg" and "libx264" in cmd and "yuv420p" in cmd and cmd[-1].endswith("o.mp4")
    # the stamp is burned into the bottom strip and top-left, the middle of the frame stays untouched
    arr = np.full((540, 960, 3), 90, np.uint8)
    img = Image.fromarray(arr)
    info = {"line_top": "Bowling Green | 25-year storm | Hour 12 of 24", "lines_bottom": [ex.HONESTY_LINE, "Water: Preview from depth tiers."],
            "caption": {"title": "Test Hospital", "subtitle": "Hospital", "extra": "water here 0.5 m", "alpha": 1.0}}
    ex.burn_stamps(img, info)
    out = np.asarray(img)
    assert (out[-30:] != 90).any() and (out[:30, :400] != 90).any()
    assert (out[240:300, 700:900] == 90).all()
    assert (out[380:480, 20:300] != 90).any()                                 # lower third
    ex.burn_stamps(Image.fromarray(arr), {"caption": {"title": "t", "subtitle": "s", "alpha": 0.0}})      # alpha 0: no crash
    # _save_frame writes a PNG; demo data adds the DEMO DATA marker
    a = np.full((360, 640, 4), 120, np.uint8)
    ex._save_frame(a.copy(), tmp_path / "f.png", info, False, True)
    ex._save_frame(a.copy(), tmp_path / "d.png", info, True, True)
    f, d = np.asarray(Image.open(tmp_path / "f.png")), np.asarray(Image.open(tmp_path / "d.png"))
    assert f.shape == (360, 640, 3) and (f != d).any()


def test_export_dir_setting_and_env(monkeypatch, tmp_path):
    from bgv import export_cinematic as ex
    monkeypatch.delenv("BGV_EXPORT_DIR", raising=False)
    assert ex.default_dir(None).name == "BG_Flood_Clips"
    monkeypatch.setenv("BGV_EXPORT_DIR", str(tmp_path / "e"))
    assert ex.default_dir(None) == tmp_path / "e"
    app = type("A", (), {"settings": {"cinematic": {"export_dir": str(tmp_path / "s")}}})()
    assert ex.default_dir(app) == tmp_path / "s"


# ------------------------------------------------------------------------------------------------ hooks + layer state
def test_hooks_registry(monkeypatch):
    from bgv.ui import hooks
    monkeypatch.setattr(hooks, "PANELS", {})
    monkeypatch.setattr(hooks, "OVERLAYS", {})
    monkeypatch.setattr(hooks, "MENUS", {})
    monkeypatch.setattr(hooks, "KEYS", {})
    calls = []

    class UI:
        error_logged = set()
        app = type("A", (), {"report_error": lambda self, w, e: calls.append(("err", w))})()

        def safe(self, name, fn, *a):
            calls.append(name)
            fn(self, *a)
    ui = UI()
    hooks.register_panel("p", lambda u: calls.append("panel"))
    hooks.register_panel("p", lambda u: calls.append("panel2"))                      # same name replaces
    hooks.register_overlay("o", lambda u: calls.append("overlay"))
    hooks.register_menu("M", lambda u: calls.append("menu"))
    hooks.draw_panels(ui)
    hooks.draw_overlays(ui)
    hooks.draw_menus(ui)
    assert calls == ["p", "panel2", "o", "overlay", "menu M", "menu"]
    hooks.register_key("bad", lambda u, ev: 1 / 0)
    hooks.register_key("eat", lambda u, ev: ev.get("key") == "x")
    assert hooks.on_key(ui, {"key": "x"}) is True and hooks.on_key(ui, {"key": "y"}) is False
    assert ("err", "key hook bad") in calls and calls.count(("err", "key hook bad")) == 1      # logged once


def test_layer_state_roundtrip_and_clamping():
    from bgv.gpu.layers.cinematic import CinematicLayer
    a = CinematicLayer()
    a.store.add("city", "One", (1, 2, 300, 4, 50), 3.0, "25yr")
    a.store.add("county", "Two", (5, 6, 700, 8, 55))
    a.lapse.rate, a.lapse.mode = 3.0, "pingpong"
    a.tour_opts.max_roads, a.tour_opts.escalate = 7, True
    a.exp_opts.kind, a.exp_opts.size_idx, a.exp_opts.fps = "orbit", 3, 60
    st = json.loads(json.dumps(a.save_state()))
    b = CinematicLayer()
    b.load_state(st)
    assert [x.name for x in b.store.list("city")] == ["One"] and [x.name for x in b.store.list("county")] == ["Two"]
    assert b.lapse.rate == 3.0 and b.lapse.mode == "pingpong" and b.tour_opts.max_roads == 7 and b.tour_opts.escalate
    assert (b.exp_opts.kind, b.exp_opts.size_idx, b.exp_opts.fps) == ("orbit", 3, 60)
    c = CinematicLayer()
    c.load_state({"tour": {"max_facilities": 9999, "dwell_s": -3.0, "time_mode": "nope", "speed": "fast"},
                  "export": {"size_idx": 99, "fps": 17, "seconds": 1e9, "kind": "warp"}, "tab": "x"})
    assert c.tour_opts.max_facilities == 40 and c.tour_opts.dwell_s == 2.0 and c.tour_opts.time_mode == "sweep"
    assert c.tour_opts.speed == 1.0 and c.exp_opts.size_idx == 3 and c.exp_opts.fps == 30 and c.exp_opts.seconds == 300.0
    assert c.exp_opts.kind == "time" and c.tab == 0
    c.load_state("garbage")                                                         # no crash
