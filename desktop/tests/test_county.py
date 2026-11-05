"""County domain, CPU side: OSM PBF reader, county scene cache, domain resolution, terrain from any GeoTIFF, the synthetic demo
data, the start-up chooser state machine.  (GPU / app level: test_county_gpu.py.)  Nothing here needs a GPU or the real data
except the two tests marked 'real' (skipped when the Kentucky extract is absent)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from bgv import config, domain as domainmod
from bgv.data import countyscene, flow as flowdata, osmpbf, terrain as terraindata
from bgv.geo import GridFrame
from bgv.ui.chooser import DomainChooser
from data_helpers import tiny_frame
from pbf_writer import Block, ring_nodes, write_pbf

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
KY_PBF = Path(config.DEFAULT_DATA_ROOT) / "kentucky-latest.osm.pbf"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A private cache folder (BGV_CACHE) for the test; the process-wide choice is restored afterwards."""
    monkeypatch.setenv("BGV_CACHE", str(tmp_path / "cache"))
    config.reset_cache_dir()
    yield tmp_path / "cache"
    monkeypatch.undo()
    config.reset_cache_dir()


# ================================================================================================ protobuf pieces
def test_varints_roundtrip_all_lengths():
    from pbf_writer import varint
    vals = [0, 1, 127, 128, 255, 16383, 16384, 2 ** 21 - 1, 2 ** 28, 2 ** 35 + 5, 2 ** 49, 2 ** 63 + 12345, 2 ** 64 - 1]
    buf = b"".join(varint(v) for v in vals)
    got = osmpbf.decode_varints(np.frombuffer(buf, np.uint8))
    assert [int(v) for v in got] == vals
    assert osmpbf.decode_varints(np.zeros(0, np.uint8)).size == 0
    with pytest.raises(ValueError):
        osmpbf.decode_varints(np.array([0x80, 0x81], np.uint8))               # no terminating byte


def test_unzigzag_and_delta():
    from pbf_writer import zigzag
    xs = [0, 1, -1, 2, -2, 1000, -1000, 2 ** 40, -2 ** 40]
    assert [int(v) for v in osmpbf.unzigzag(np.array([zigzag(x) for x in xs], np.uint64))] == xs


def test_parse_height_variants():
    ph = osmpbf.parse_height
    assert ph("12") == 12.0 and ph("12.5 m") == 12.5 and ph("12,5") == 12.5
    assert abs(ph("40 ft") - 12.192) < 1e-9 and abs(ph("40'") - 12.192) < 1e-9
    for bad in (None, "", "tall", "-3", "0", "1200", "12 stories"):
        assert np.isnan(ph(bad)), bad


def test_building_height_rules():
    bh = osmpbf.building_height
    assert bh({"height": "15"}) == 15.0
    assert bh({"building:levels": "3"}) == pytest.approx(3 * osmpbf.LEVEL_HEIGHT_M + 1.0)
    assert bh({"building": "garage"}) == 3.0
    assert bh({"building": "yes"}) == 6.0 and bh({}) == 6.0
    assert bh({"height": "x", "building:levels": "2"}) == pytest.approx(2 * osmpbf.LEVEL_HEIGHT_M + 1.0)
    assert osmpbf.address_of({"addr:housenumber": "12", "addr:street": "Main St"}) == "12 Main St"
    assert osmpbf.address_of({"addr:street": "Main St"}) == "Main St" and osmpbf.address_of({}) == ""


# ================================================================================================ a tiny PBF
LON0, LAT0 = -86.50, 36.95
BBOX = (-86.52, 36.93, -86.46, 36.99)


def make_pbf(path, compress="zlib", unsorted=False):
    """Block 1: nodes.  Block 2: ways.  Inside the box: two buildings (one with address / levels, one plain), a residential
    street, a tertiary road, a footway (ignored), a parking aisle (ignored); outside: one building."""
    nb = Block()
    nodes = []
    b1, r1 = ring_nodes(1, LON0, LAT0, 0.0003, 0.0002)                    # building A, ~25 x 22 m
    b2, r2 = ring_nodes(10, LON0 + 0.002, LAT0, 0.0002, 0.0002)           # building B
    b3, r3 = ring_nodes(20, -86.30, 36.95, 0.0002, 0.0002)                # building C: outside the box
    for n in (b1, b2, b3):
        nodes += n
    street = [(30 + i, LON0 - 0.001 + i * 0.0006, LAT0 - 0.0004) for i in range(8)]       # 8 nodes along y = LAT0 - 0.0004
    road = [(40 + i, LON0 - 0.001, LAT0 + 0.003 + i * 0.0005) for i in range(5)]
    foot = [(50 + i, LON0, LAT0 + 0.01 + i * 0.0003) for i in range(3)]
    aisle = [(60 + i, LON0 + 0.004, LAT0 + 0.01 + i * 0.0003) for i in range(3)]
    nodes += street + road + foot + aisle
    nodes.sort()
    nb.add_nodes(nodes)
    wb = Block()
    wb.add_ways([
        (101, {"building": "yes", "building:levels": "3", "addr:housenumber": "5", "addr:street": "Test Street"}, r1),
        (102, {"building": "house"}, r2),
        (103, {"building": "yes"}, r3),
        (104, {"building": "no"}, r2),
        (201, {"highway": "residential", "name": "Test Street"}, [n[0] for n in street]),
        (202, {"highway": "tertiary_link", "name": "Ramp"}, [n[0] for n in road]),
        (203, {"highway": "footway"}, [n[0] for n in foot]),
        (204, {"highway": "service", "service": "parking_aisle"}, [n[0] for n in aisle]),
        (205, {"amenity": "bench"}, [n[0] for n in foot]),
    ])
    blocks = [wb, nb] if unsorted else [nb, wb]
    return write_pbf(path, blocks, compress)


@pytest.mark.parametrize("compress", ["zlib", "raw"])
def test_read_pbf_extracts_buildings_and_roads(tmp_path, compress):
    p = make_pbf(tmp_path / "t.pbf", compress)
    r = osmpbf.read_pbf(p, BBOX)
    ids = sorted(f["properties"]["osm_id"] for f in r.buildings)
    assert ids == [101, 102]                                              # C is outside the box, 'building=no' is skipped
    a = next(f for f in r.buildings if f["properties"]["osm_id"] == 101)
    assert a["properties"]["ADDRESS"] == "5 Test Street"
    assert a["properties"]["height_m"] == pytest.approx(3 * osmpbf.LEVEL_HEIGHT_M + 1.0)
    ring = np.array(a["geometry"]["coordinates"][0])
    assert ring.shape == (5, 2) and np.allclose(ring[0], ring[-1])
    assert ring[:, 0].min() == pytest.approx(LON0, abs=2e-7) and ring[:, 1].max() == pytest.approx(LAT0 + 0.0002, abs=2e-7)
    roads = {f["properties"]["osm_id"]: f for f in r.roads}
    assert sorted(roads) == [201, 202]                                    # footway and parking aisle are dropped
    assert roads[201]["properties"]["highway"] == "residential"
    assert roads[202]["properties"]["highway"] == "primary_link"          # tertiary_link maps onto a known class
    assert len(roads[201]["geometry"]["coordinates"]) == 8
    from bgv.data.roads import ROAD_CLASSES
    assert all(f["properties"]["highway"] in ROAD_CLASSES for f in r.roads)
    assert r.stats["buildings"] == 2 and r.stats["roads"] == 2 and r.stats["nodes_seen"] > 0


def test_read_pbf_unsorted_and_unsupported(tmp_path):
    with pytest.raises(ValueError, match="not sorted"):
        osmpbf.read_pbf(make_pbf(tmp_path / "u.pbf", unsorted=True), BBOX)
    with pytest.raises(ValueError, match="unsupported compression"):
        osmpbf.read_pbf(make_pbf(tmp_path / "l.pbf", compress="lzma"), BBOX)
    (tmp_path / "trunc.pbf").write_bytes(make_pbf(tmp_path / "t2.pbf").read_bytes()[:-9])
    with pytest.raises(ValueError):
        osmpbf.read_pbf(tmp_path / "trunc.pbf", BBOX)


def test_way_crossing_the_box_edge_is_clipped_not_dropped(tmp_path):
    nb, wb = Block(), Block()
    nodes = [(i + 1, -86.53 + i * 0.01, 36.95) for i in range(8)]         # -86.53 ... -86.46: starts outside the box
    nb.add_nodes(nodes)
    wb.add_ways([(7, {"highway": "primary"}, [n[0] for n in nodes])])
    r = osmpbf.read_pbf(write_pbf(tmp_path / "c.pbf", [nb, wb]), BBOX)
    assert len(r.roads) == 1 and len(r.roads[0]["geometry"]["coordinates"]) == 8     # the margin keeps the whole way


def test_label_near_roads(tmp_path):
    r = osmpbf.read_pbf(make_pbf(tmp_path / "t.pbf"), BBOX)
    n = osmpbf.label_near_roads(r)
    by = {f["properties"]["osm_id"]: f["properties"]["ADDRESS"] for f in r.buildings}
    assert by[101] == "5 Test Street"                                     # a real address is never replaced
    assert by[102] == "near Test Street" and n == 1                       # B is ~45 m from Test Street, 330 m from Ramp


# ================================================================================================ county scene cache
def test_frame_bbox_is_padded_and_rounded_outward():
    fr = tiny_frame(100, 100, 25.0)
    w, s, e, n = countyscene.frame_bbox(fr)
    from bgv.geo import merc_to_lnglat
    lo0, la0 = merc_to_lnglat(fr.xmin, fr.ymin)
    lo1, la1 = merc_to_lnglat(fr.xmax, fr.ymax)
    assert w < lo0 and s < la0 and e > lo1 and n > la1
    assert all(abs(v * 100 - round(v * 100)) < 1e-6 for v in (w, s, e, n))


def test_ensure_osm_scene_builds_once_then_reads_the_cache(tmp_path, cache):
    pbf = make_pbf(tmp_path / "t.pbf")
    fr = tiny_frame(100, 100, 25.0)
    sc = countyscene.ensure_osm_scene(pbf, fr, bbox=BBOX)
    assert not sc.cached and sc.buildings_path.is_file() and sc.roads_path.is_file()
    assert sc.buildings_path.is_relative_to(cache)
    assert sc.info["buildings"] == 2 and sc.info["roads"] == 2 and sc.info["labelled_near_road"] == 1
    fc = json.loads(sc.buildings_path.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    sc2 = countyscene.ensure_osm_scene(pbf, fr, bbox=BBOX)
    assert sc2.cached and sc2.buildings_path == sc.buildings_path
    # another box / a changed PBF is another extraction
    sc3 = countyscene.ensure_osm_scene(pbf, fr, bbox=(-86.6, 36.9, -86.4, 37.0))
    assert not sc3.cached and sc3.buildings_path != sc.buildings_path
    pbf.write_bytes(pbf.read_bytes() + b"")
    os.utime(pbf, (1, 1))
    assert not countyscene.ensure_osm_scene(pbf, fr, bbox=BBOX).cached
    with pytest.raises(FileNotFoundError):
        countyscene.ensure_osm_scene(tmp_path / "nope.pbf", fr)


def test_osm_scene_feeds_the_city_loaders(tmp_path, cache):
    """The cached GeoJSON files go straight through the same building / road loaders as the city's."""
    from bgv.data import buildings as bmod, roads as rmod
    pbf = make_pbf(tmp_path / "t.pbf")
    # a frame (mercator, 10 m cells) that contains the tiny scene
    from bgv.geo import lnglat_to_merc
    x0, y0 = lnglat_to_merc(LON0 - 0.004, LAT0 - 0.002)
    fr = GridFrame(200, 150, 10.0, 8.0, x0, y0, x0 + 2000.0, y0 + 1500.0)
    sc = countyscene.ensure_osm_scene(pbf, fr, bbox=BBOX)
    b = bmod.load_buildings(tmp_path, fr, geojson=sc.buildings_path)
    r = rmod.load_roads(tmp_path, fr, geojson=sc.roads_path)
    assert b.n == 2 and int(b.in_grid.sum()) == 2
    assert float(b.height_m.max()) == pytest.approx(3 * osmpbf.LEVEL_HEIGHT_M + 1.0, abs=1e-3)
    assert 400.0 < float(b.area_m2[0]) < 800.0                          # 0.0003 x 0.0002 deg ~ 27 x 22 m = ~590 m2
    assert r.n_roads == 2 and r.n_samples > 10 and r.total_length_m > 300.0
    b2 = bmod.load_buildings(tmp_path, fr, geojson=sc.buildings_path)
    assert b2.timings.get("cached") is True                             # the second load is the npz cache


# ================================================================================================ domains
def _touch(p: Path, text: str = "{}"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def make_root(tmp, city=("25yr",), county=(), dem=False):
    root = tmp / "root"
    for s in city:
        _touch(root / "layers" / "flow" / s / "meta.json")
    for s in county:
        _touch(root / "layers" / "flow_county" / s / "meta.json")
    if dem:
        _touch(root / "layers" / "dem_warren_5m.tif", "x")
    return root


def test_storm_order_natural_and_labels(tmp_path):
    root = make_root(tmp_path, county=("ev10", "ev2", "100yr", "2yr", "10yr", "ev1"))
    assert flowdata.available_storms(root, "flow_county") == ["2yr", "10yr", "100yr", "ev1", "ev2", "ev10"]
    assert flowdata.available_storms(root) == ["25yr"]
    assert flowdata.available_storms(tmp_path / "nothing", "flow_county") == []
    _touch(root / "layers" / "flow_county" / "10yr" / "meta.json", json.dumps({"label": "10-year storm (county)"}))
    assert flowdata.storm_label(root, "10yr", "flow_county") == "10-year storm (county)"
    assert flowdata.storm_label(root, "ev1", "flow_county") == "ev1"


def test_city_domain_and_its_errors(tmp_path):
    root = make_root(tmp_path)
    d = domainmod.resolve("city", root)
    assert d.key == "city" and d.storms == ("25yr",) and d.flow_subdir == "flow" and not d.demo
    assert d.terrain.units == "feet" and d.terrain.path == root / "dem_bg.tif"
    assert d.storm_label("25yr") == "25-year storm" and d.storm_label("zz") == "zz"
    with pytest.raises(domainmod.DomainUnavailable, match="no storms found"):
        domainmod.resolve("city", tmp_path / "empty")
    with pytest.raises(domainmod.DomainUnavailable, match="unknown domain"):
        domainmod.resolve("moon", root)


def test_county_unavailable_says_what_to_do(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(domainmod.DomainUnavailable) as e:
        domainmod.resolve("county", root, {"county_demo": str(tmp_path / "nodemo")})
    msg = str(e.value)
    assert "flow_county" in msg and "dem_warren_5m.tif" in msg and "--make-county-demo" in msg and "COUNTY_RUN.md" in msg
    d, notice = domainmod.resolve_or_city("county", root, {"county_demo": str(tmp_path / "nodemo")})
    assert d.key == "city" and "unavailable" in notice and "flow_county" in notice
    infos = {i.key: i for i in domainmod.domain_infos(root, {"county_demo": str(tmp_path / "nodemo")})}
    assert infos["city"].available and not infos["county"].available and infos["county"].reason == msg


def test_county_real_data_and_dem_override(tmp_path):
    root = make_root(tmp_path, county=("100yr", "ev1"), dem=False)
    with pytest.raises(domainmod.DomainUnavailable, match="terrain is missing"):
        domainmod.resolve("county", root, {})                                       # flow grids but no DEM: unavailable
    own = _touch(tmp_path / "elsewhere" / "my_dem.tif", "x")
    d = domainmod.resolve("county", root, {"county_dem": str(own)})
    assert d.key == "county" and d.storms == ("100yr", "ev1") and d.terrain.path == own and d.terrain.units == "metres"
    assert d.terrain.lazy and not d.demo and d.flow_subdir == "flow_county" and d.title == "Warren County"
    _touch(root / "layers" / "dem_warren_5m.tif", "x")
    assert domainmod.resolve("county", root, {}).terrain.path == root / "layers" / "dem_warren_5m.tif"
    assert domainmod.resolve("county", root, {}).default_exag > 2.0


def test_county_demo_is_used_only_without_real_data(tmp_path):
    root = make_root(tmp_path)
    demo = tmp_path / "demo"
    _touch(demo / "layers" / "flow_county" / "10yr" / "meta.json")
    _touch(demo / "layers" / "dem_warren_demo.tif", "x")
    s = {"county_demo": str(demo)}
    d = domainmod.resolve("county", root, s)
    assert d.demo and d.flow_root == demo and d.title.endswith("(DEMO DATA)") and "SYNTHETIC" in d.notice
    assert domainmod.resolve_or_city("county", root, s) [1] == "" and domainmod.resolve_or_city("county", root, s)[0].demo
    infos = {i.key: i for i in domainmod.domain_infos(root, s)}
    assert infos["county"].available and infos["county"].demo and "DEMO DATA" in infos["county"].label
    # real flow grids arrive: they win, the demo folder is ignored
    _touch(root / "layers" / "flow_county" / "100yr" / "meta.json")
    _touch(root / "layers" / "dem_warren_5m.tif", "x")
    d2 = domainmod.resolve("county", root, s)
    assert not d2.demo and d2.flow_root == root and d2.storms == ("100yr",)


def test_pick_storm_and_cli_settings():
    pick = domainmod.pick_storm
    av = ["5yr", "25yr", "100yr"]
    assert pick({}, "city", av) == "25yr" and pick({}, "county", av) == "100yr"
    assert pick({"storm": "5yr"}, "city", av) == "5yr"                              # legacy city key
    assert pick({"storm": "5yr"}, "county", av) == "100yr"                          # ... not for the county
    assert pick({"domain_views": {"county": {"storm": "5yr"}}}, "county", av) == "5yr"
    assert pick({"storm": "5yr"}, "city", av, explicit="100yr") == "100yr"
    assert pick({"storm": "gone"}, "city", ["10yr", "50yr"]) == "10yr"
    class A:
        domain, county_dem, county_demo, county_pbf = "COUNTY", "D:/x.tif", None, "k.pbf"
    s = domainmod.apply_cli({"keep": 1}, A())
    assert s == {"keep": 1, "domain": "county", "county_dem": "D:/x.tif", "county_pbf": "k.pbf"}


def test_domain_camera_presets_and_overview():
    fr = GridFrame(2631, 3052, 25.0, 19.99, -9633395.2, 4400000.0, -9633395.2 + 65775.0, 4400000.0 + 76300.0)
    root_city = Path("x")
    d_city = domainmod.Domain("city", "Bowling Green", root_city, "flow", ("25yr",), terraindata.city_source(root_city), root_city,
                              presets=domainmod.CITY_PRESETS)
    d_cty = domainmod.Domain("county", "Warren County", root_city, "flow_county", ("100yr",),
                             terraindata.TerrainSource(Path("d.tif"), "metres", True), root_city, presets=domainmod.COUNTY_PRESETS)
    assert d_city.overview(fr)["dist"] == 14000.0
    sx, sy = fr.size_m
    assert d_cty.overview(fr)["dist"] == pytest.approx(1.1 * max(sx, sy))
    x, y, dist, yaw, pitch = d_cty.preset_view(fr, "Overview")
    assert (x, y) == (0.0, 0.0) and dist == pytest.approx(1.1 * max(sx, sy))
    assert set(d_cty.presets) >= {"Overview", "Bowling Green"} and set(d_city.presets) == {"Overview", "Downtown", "Barren River"}


# ================================================================================================ terrain from any GeoTIFF
def _write_dem(path, frame: GridFrame, fn, res=10.0, nodata=-9999.0, pad=600.0):
    """A UTM 16N GeoTIFF covering the frame, z = fn(easting, northing)."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import transform
    xs = [frame.xmin, frame.xmax, frame.xmin, frame.xmax]
    ys = [frame.ymin, frame.ymin, frame.ymax, frame.ymax]
    e, n = transform("EPSG:3857", "EPSG:32616", xs, ys)
    e0, e1, n0, n1 = min(e) - pad, max(e) + pad, min(n) - pad, max(n) + pad
    w, h = int((e1 - e0) / res) + 1, int((n1 - n0) / res) + 1
    E = e0 + (np.arange(w) + 0.5) * res
    N = n1 - (np.arange(h) + 0.5) * res
    z = fn(E[None, :], N[:, None]).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs="EPSG:32616",
                       transform=Affine(res, 0, e0, 0, -res, n1), nodata=nodata) as ds:
        ds.write(z, 1)
    return Path(path), (e0, e1, n0, n1)


FRAME = tiny_frame(60, 50, 25.0, 0.8)


def test_lazy_warp_reproduces_a_known_surface(tmp_path):
    dem, _ = _write_dem(tmp_path / "d.tif", FRAME, lambda e, n: 150.0 + 0.002 * (e - 549000.0) + 0.001 * (n - 4093000.0))
    z, note = terraindata.warp_dem_lazy(dem, FRAME, "metres")
    assert z.shape == (50, 60) and z.dtype == np.float32 and note == ""
    assert np.isfinite(z).all()
    from rasterio.warp import transform
    cx = FRAME.xmin + (np.arange(60) + 0.5) * FRAME.cell_m
    cy = FRAME.ymax - (np.arange(50) + 0.5) * FRAME.cell_m
    e, n = transform("EPSG:3857", "EPSG:32616", [cx[30]], [cy[25]])
    assert z[25, 30] == pytest.approx(150.0 + 0.002 * (e[0] - 549000.0) + 0.001 * (n[0] - 4093000.0), abs=0.15)
    # a plane has the same slope in feet and metres only after the conversion: feet x 0.3048
    zf, _ = terraindata.warp_dem_lazy(dem, FRAME, "feet")
    assert np.allclose(zf, z * 0.3048, atol=1e-3)


def test_lazy_warp_averages_fine_detail_instead_of_aliasing(tmp_path):
    # a 5 m checkerboard of +-10 m on a 25 m (mercator) / 20 m grid: area averaging gives ~0, bilinear sampling would not
    def check(e, n):
        return 200.0 + 10.0 * (((np.floor(e / 5.0) + np.floor(n / 5.0)) % 2) * 2 - 1)
    dem, _ = _write_dem(tmp_path / "c.tif", FRAME, check, res=5.0)
    z, _ = terraindata.warp_dem_lazy(dem, FRAME, "metres")
    assert abs(float(z.mean()) - 200.0) < 0.5 and float(np.abs(z - 200.0).max()) < 4.5


def test_lazy_warp_nodata_and_a_feet_dem_called_metres(tmp_path):
    from rasterio.warp import transform
    ex, _ = transform("EPSG:3857", "EPSG:32616", [FRAME.xmin, FRAME.xmax], [FRAME.ymin, FRAME.ymin])
    mid = 0.5 * (ex[0] + ex[1])

    def fn(e, n):
        v = 600.0 + 0 * e + 0 * n                                           # FEET in a file that is used as metres
        return np.where(e < mid, -9999.0, v)                                # the west half has no data
    dem, _ = _write_dem(tmp_path / "f.tif", FRAME, fn)
    z, note = terraindata.warp_dem_lazy(dem, FRAME, "metres")
    assert "FEET" in note and "converted" in note
    fin = z[np.isfinite(z)]
    assert fin.size and np.allclose(fin, 600.0 * 0.3048, atol=0.05)         # converted
    assert np.isnan(z).any() and np.isfinite(z).any()                       # nodata stays NaN
    ok, note2 = terraindata.warp_dem_lazy(_write_dem(tmp_path / "m.tif", FRAME, lambda e, n: 170.0 + 0 * e)[0], FRAME, "metres")
    assert note2 == "" and np.allclose(ok[np.isfinite(ok)], 170.0, atol=0.05)


def test_terrain_source_validation_and_city_cache_key_is_unchanged(tmp_path):
    with pytest.raises(ValueError):
        terraindata.TerrainSource(tmp_path / "x.tif", "yards")
    import hashlib
    dem = tmp_path / "dem_bg.tif"
    dem.write_bytes(b"1234")
    st = dem.stat()
    old = hashlib.sha1((f"{terraindata.CACHE_VERSION}|{st.st_size}|{int(st.st_mtime)}|{FRAME.width}x{FRAME.height}|{FRAME.cell_m!r}|"
                        f"{FRAME.xmin!r},{FRAME.ymin!r},{FRAME.xmax!r},{FRAME.ymax!r}").encode()).hexdigest()[:16]
    assert terraindata._cache_key(dem, FRAME) == old                                    # the city's cache files stay valid
    assert terraindata._cache_key(dem, FRAME, terraindata.city_source(tmp_path)) == old
    metres = terraindata.TerrainSource(dem, "metres", True)
    assert terraindata._cache_key(dem, FRAME, metres) != old


def test_gpu_data_is_cached_and_carries_the_unit_note(tmp_path, cache):
    def fn(e, n):
        return 520.0 + 0.001 * (e - 549000.0)                              # feet in a 'metres' file
    dem, _ = _write_dem(tmp_path / "f.tif", FRAME, fn)
    src = terraindata.TerrainSource(dem, "metres", True)
    msgs = []
    filled, mips, stats, valid = terraindata.load_terrain_gpu_data(tmp_path, FRAME, src, lambda m, f: msgs.append(m))
    assert filled.shape == (50, 60) and mips[0].shape == (50, 60, 4) and valid.all()
    assert "FEET" in stats["note"] and msgs
    assert 150.0 < stats["min"] < stats["max"] < 175.0                      # 520 ft = 158 m
    f2, m2, s2, v2 = terraindata.load_terrain_gpu_data(tmp_path, FRAME, src)
    assert np.array_equal(f2, filled) and s2["note"] == stats["note"]      # the second call is the npy cache, note included
    assert not np.shares_memory(f2, filled)


def test_missing_dem_is_a_clear_error(tmp_path, cache):
    with pytest.raises(FileNotFoundError, match="terrain not found"):
        terraindata.load_terrain_gpu_data(tmp_path, FRAME, terraindata.TerrainSource(tmp_path / "nope.tif", "metres", True))


# ================================================================================================ the synthetic demo
@pytest.fixture(scope="module")
def tiny_demo(tmp_path_factory):
    sys.path.insert(0, str(SCRIPTS))
    from rasterio.warp import transform
    from bgv.data import demo
    x, y = transform("EPSG:4326", "EPSG:32616", [-86.4436], [36.9903])
    box = (x[0] - 3000, y[0] - 3000, x[0] + 3000, y[0] + 3000)
    root = tmp_path_factory.mktemp("demo")
    res = demo.make_county_demo(root, utm_box=box, hours=[8, 10, 12, 14], storms=demo.DEMO_STORMS, workers=1, log=lambda m: None)
    return root, res, box


def test_demo_writes_valid_marked_storms_and_terrain(tiny_demo):
    root, res, box = tiny_demo
    import flowgrid as fg
    from bgv.data import demo
    assert (root / "layers" / demo.DEM_FILE).is_file() and (root / "README_DEMO.txt").is_file()
    assert flowdata.available_storms(root, "flow_county") == ["10yr", "100yr"]
    for name in ("10yr", "100yr"):
        d = root / "layers" / "flow_county" / name
        assert fg.validate_storm_dir(str(d)) == []
        st = flowdata.Storm.open(root, name, "flow_county")
        assert st.synthetic and st.source == "rasters" and "DEMO DATA" in st.label and st.hours == [8, 10, 12, 14]
        assert "DEMO DATA" in str(st.meta["source_note"]) and "DERIVED" in str(st.meta["source_note"])
        assert st.frame.cell_m == 25.0 and 15.0 < st.frame.cell_ground_m < 25.0
    st = flowdata.Storm.open(root, "100yr", "flow_county")
    rgba = st.load_hour(12)
    wet = rgba[..., 0] > 0
    assert wet.any() and wet.mean() < 0.9
    d100 = flowdata.decode_depth(rgba[..., 0][wet]).max()
    d10 = flowdata.decode_depth(flowdata.Storm.open(root, "10yr", "flow_county").load_hour(12)[..., 0]).max()
    assert d100 > d10 > 0.5


def test_demo_terrain_covers_the_county_extent_in_metres(tiny_demo):
    root, res, box = tiny_demo
    import rasterio
    from bgv.data import demo
    with rasterio.open(root / "layers" / demo.DEM_FILE) as ds:
        assert ds.crs.to_epsg() == 32616 and abs(ds.transform.a) == 20.0 and ds.nodata == -9999.0
        assert ds.bounds.left == box[0] and ds.bounds.top == box[3] and ds.bounds.right >= box[2] - 1e-6
        z = ds.read(1)
    assert 100.0 < float(z.min()) < float(z.max()) < 400.0                  # metres (Warren County is 110 - 330 m)


def test_demo_depth_model_properties():
    from bgv.data import demo
    z, hand, stage, sinks = demo.synth_terrain(260, 300, 20.0, seed=3)
    assert z.shape == hand.shape == stage.shape == (300, 260) and (hand >= 0).all() and sinks
    assert float(stage.max()) == 1.0 and float(stage.min()) >= 0.5
    areas = [float((demo.depth_field(z, hand, stage, sinks, float(h), 7.0, 15.0) > 0.02).mean()) for h in (2, 8, 15, 24)]
    assert areas[0] < areas[1] < areas[2] and areas[3] < areas[2]            # rises to the peak hour, then recedes
    d = demo.depth_field(z, hand, stage, sinks, 15.0, 7.0, 15.0)
    assert 3.0 < float(d.max()) <= 7.0 + 1e-3 and (d[(d > 0) & (d < 0.005)].size == 0)
    assert demo.hydrograph(0, 15) == 0.0 and demo.hydrograph(15, 15) == pytest.approx(1.0)
    z2, *_ = demo.synth_terrain(260, 300, 20.0, seed=3)
    assert np.array_equal(z, z2)                                            # deterministic for a seed


def test_mark_demo_stamps_the_banner():
    from bgv.ui.screenshot import mark_demo
    img = np.full((360, 640, 4), 90, np.uint8)
    out = mark_demo(img.copy())
    assert np.array_equal(out[:100, :100], img[:100, :100])                 # the top-left corner is untouched
    assert (out[-60:, -400:, :3] != 90).any()                               # the banner is bottom right
    assert out.shape == img.shape


# ================================================================================================ the start-up chooser
def infos(city=True, county=True, demo=False):
    return [domainmod.DomainInfo("city", "Bowling Green", city, "" if city else "no city"),
            domainmod.DomainInfo("county", "Warren County", county, "" if county else "no county data", demo)]


def test_chooser_waits_for_the_countdown_only_when_there_is_a_choice():
    c = DomainChooser(infos(), "city", countdown=3.0)
    assert c.waits and not c.may_finish(0.0)                    # not even drawn yet: never finish blind
    c.start(10.0)
    assert c.remaining(10.0) == 3.0 and c.counting(11.0) and not c.may_finish(12.9)
    assert c.may_finish(13.0) and not c.counting(13.0)
    one = DomainChooser(infos(county=False), "city")
    assert not one.waits and one.may_finish(0.0) and not one.counting(0.0)          # county missing: nothing to wait for
    off = DomainChooser(infos(), "city", auto=False)
    assert not off.waits and off.may_finish(0.0)


def test_chooser_cancel_holds_the_screen_until_a_click():
    c = DomainChooser(infos(), "city", countdown=3.0)
    c.start(0.0)
    c.cancel()
    assert not c.may_finish(100.0) and not c.counting(1.0)
    assert "cancelled" in c.status_text(1.0, "Bowling Green")
    assert c.click("city") == "continue" and c.may_finish(100.0)


def test_chooser_clicks():
    c = DomainChooser(infos(), "city")
    c.start(0.0)
    assert c.click("county") == "switch" and c.chosen and c.may_finish(0.1)
    assert c.take_request() == "county" and c.take_request() is None
    d = DomainChooser(infos(county=False), "city")
    assert d.click("county") == "unavailable" and not d.chosen and d.request is None
    assert d.click("moon") == "unavailable"
    assert d.click("city") == "continue"
    st = DomainChooser(infos(), "city")
    st.start(5.0)
    assert "in 3 s" in st.status_text(5.0, "Bowling Green") and "Continuing" in st.status_text(5.5, "Bowling Green")


# ================================================================================================ real data (skipped when absent)
@pytest.mark.real
def test_real_kentucky_county_extraction_and_alignment_with_the_city_layer():
    """The real PBF: counts are plausible and the county buildings sit on top of the city layer's (same OSM source) once
    both are put in the SAME world frame - proves the lng/lat -> grid projection of the county path."""
    if not KY_PBF.is_file():
        pytest.skip("Kentucky extract not present")
    from bgv.data import buildings as bmod
    from data_helpers import real_root, real_frame
    root = real_root()
    # county-sized frame: the Vista grid (2631 x 3052 cells of 25 m mercator) around the real county box
    from rasterio.warp import transform
    xs, ys = transform("EPSG:32616", "EPSG:3857", [524851.0, 576871.0], [4056081.0, 4116296.0])
    x0, x1, y0, y1 = min(xs) - 150, max(xs) + 150, min(ys) - 150, max(ys) + 150
    cell = 25.0
    w, h = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    county = GridFrame(w, h, cell, cell * 0.7997, x0, y0, x0 + w * cell, y0 + h * cell)
    assert abs(w - 2631) <= 25 and abs(h - 3052) <= 60                       # (the UTM box is rotated against mercator north)
    real_meta = Path(config.DEFAULT_DATA_ROOT) / "layers" / "flow_county" / "100yr" / "meta.json"
    if real_meta.is_file():                                                  # the grid of the real converted county storms
        county = GridFrame.from_meta(json.loads(real_meta.read_text()))
    sc = countyscene.ensure_osm_scene(KY_PBF, county)                       # ~1 minute once, then the cache
    assert 15_000 < sc.info["buildings"] < 60_000 and 15_000 < sc.info["roads"] < 60_000
    cb = bmod.load_buildings(root, county, geojson=sc.buildings_path)
    city = bmod.load_buildings(root, real_frame())
    assert cb.n == sc.info["buildings"] and int(cb.in_grid.sum()) > 0.95 * cb.n
    # both layers -> lng/lat through their own frames, then: is there a county building within 25 m of a city-layer building?
    from scipy.spatial import cKDTree
    R = 6378137.0

    def lnglat(fr, X, Y):
        xm, ym = X / fr.k + fr.xc, Y / fr.k + fr.yc
        return np.degrees(xm / R), np.degrees(2.0 * np.arctan(np.exp(ym / R)) - np.pi / 2.0)

    clon, clat = lnglat(county, cb.cx.astype(np.float64), cb.cy.astype(np.float64))
    tlon, tlat = lnglat(real_frame(), city.cx[::7].astype(np.float64), city.cy[::7].astype(np.float64))
    tree = cKDTree(np.c_[clon * 88800.0, clat * 111000.0])
    d, _ = tree.query(np.c_[tlon * 88800.0, tlat * 111000.0])
    close = float((d < 25.0).mean())
    assert close > 0.9, f"only {close:.0%} of the city layer's buildings have a county building within 25 m"


def test_roads_layer_uses_tier_palette_and_toggle():
    from bgv.gpu.layers import roads
    assert "R.tier[3]" in roads.ROADS_WGSL and "ci <= 6u" in roads.ROADS_WGSL      # tier colours come from the palette uniform
    layer = roads.RoadsLayer()
    assert layer.at_risk and layer.save_state()["at_risk"] is True
    layer.load_state({"at_risk": False})
    assert layer.at_risk is False
