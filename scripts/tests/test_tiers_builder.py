"""
Tests for scripts/build_flow_from_tiers.py (local preview of the flow grids from the depth-tier polygons).

  C:/Users/Minec/anaconda3/python.exe -m unittest discover -s C:/Users/Minec/Documents/BG_3D_Map/scripts/tests -v

Everything is synthetic and small (temp folders): rasterisation, the stream splitter against json.load, the Manning
raster, the smoothed surface (bands, footprint, halo, volume, ponds at rest), the velocity limits, the DEM unit, and a
whole tiny storm folder built end to end (validate_storm_dir, resume, force, position of a polygon in the output).
"""
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import build_flow_from_tiers as bf  # noqa: E402
import flowgrid as fg  # noqa: E402

from rasterio.transform import Affine  # noqa: E402

DX = 5.088392902741106
X0, Y0 = 549000.0, 4091000.0                    # NW corner of the synthetic DEM (UTM 16N, near Bowling Green)
FT = 1.0 / 0.3048


def fc(features):
    return {"type": "FeatureCollection", "features": features}


def poly_feature(x0, y0, x1, y1, tier, holes=()):
    ring = [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]
    rings = [ring] + [[[a, b] for a, b in h] for h in holes]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": rings},
            "properties": {"tier": tier, "color": "#4dabf7", "file": "depth_x", "run": "v2_x"}}


def write_compact(path, features):
    """The exact one-line shape tif_to_geojson.py writes."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(fc(features), separators=(",", ":")))


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def cell_box(r0, r1, c0, c1):
    """UTM box of DEM cells rows r0..r1-1, cols c0..c1-1 -> (xmin, ymin, xmax, ymax)."""
    return X0 + c0 * DX, Y0 - r1 * DX, X0 + c1 * DX, Y0 - r0 * DX


def write_dem(path, z_m, nodata=-9999.0, as_feet=True):
    import rasterio
    arr = (np.asarray(z_m, dtype=np.float64) * (FT if as_feet else 1.0)).astype(np.float32)
    H, W = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32", crs="EPSG:32616",
                       transform=Affine(DX, 0, X0, 0, -DX, Y0), nodata=nodata) as dst:
        dst.write(arr, 1)


def utm_to_lonlat(pts):
    pts = np.asarray(pts, dtype=np.float64)
    lon, lat = fg.utm16n_to_lnglat(pts[:, 0], pts[:, 1])
    return np.column_stack([lon, lat]).tolist()


def manning_dem(shape=(40, 50)):
    H, W = shape
    return {"transform": Affine(DX, 0, X0, 0, -DX, Y0), "shape": shape, "bounds": (X0, Y0 - H * DX, X0 + W * DX, Y0),
            "dx": DX}


# ------------------------------------------------------------------------------------------------ small helpers
class TestHoursAndPaths(unittest.TestCase):
    def test_parse_hours(self):
        self.assertEqual(bf.parse_hours("3-5"), [3, 4, 5])
        self.assertEqual(bf.parse_hours("1,2,5-7"), [1, 2, 5, 6, 7])
        self.assertEqual(bf.parse_hours("24"), [24])
        for bad in ("0", "25", "5-3", "a", ""):
            with self.assertRaises(ValueError):
                bf.parse_hours(bad)

    def test_tier_file_name(self):
        p = bf.tier_file("/app", "100yr", 13).replace("\\", "/")
        self.assertTrue(p.endswith("/BG FLOOD MAP VISTA SUPERCOMPUTER/v2/100yr/v2_100yr_depth_046800s.geojson"), p)
        self.assertTrue(bf.tier_file("/app", "5yr", 1).endswith("v2_5yr_depth_003600s.geojson"))


# ------------------------------------------------------------------------------------------------ stream splitter
class TestIterFeatures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_matches_json_load_across_chunk_boundaries(self):
        rnd = random.Random(7)
        feats = []
        for i in range(400):
            x, y = 550000 + rnd.randrange(0, 4000, 2), 4090000 + rnd.randrange(0, 4000, 2)
            tier = rnd.choice(bf.TIERS)
            feats.append(poly_feature(float(x), float(y), float(x + 2), float(y + rnd.choice((2, 4, 6))), tier))
        feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [
            [[[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [0.0, 0.0]]], [[[5.0, 5.0], [5.0, 7.0], [7.0, 7.0], [5.0, 5.0]]]]},
            "properties": {"tier": "deep", "color": "#000", "file": "f", "run": "r"}})
        path = os.path.join(self.tmp, "t.geojson")
        write_compact(path, feats)
        expect = load_json(path)["features"]
        for chunk in (97, 700, 4096, 1 << 20):            # 97 bytes cuts nearly every separator in two
            got = list(bf.iter_features(path, chunk=chunk))
            self.assertEqual(got, expect, "chunk=%d" % chunk)

    def test_empty_collection_and_single_feature(self):
        p1 = os.path.join(self.tmp, "e.geojson")
        write_compact(p1, [])
        self.assertEqual(os.path.getsize(p1), 42)         # the size of a dry-hour file in the real data
        self.assertEqual(list(bf.iter_features(p1)), [])
        p2 = os.path.join(self.tmp, "one.geojson")
        write_compact(p2, [poly_feature(0.0, 0.0, 2.0, 2.0, "shallow")])
        self.assertEqual(len(list(bf.iter_features(p2, chunk=10))), 1)

    def test_truncated_file_is_an_error(self):
        path = os.path.join(self.tmp, "t.geojson")
        write_compact(path, [poly_feature(float(i), 0.0, i + 2.0, 2.0, "deep") for i in range(20)])
        data = read_bytes(path)
        with open(path, "wb") as f:
            f.write(data[:-30])
        with self.assertRaises(ValueError):
            list(bf.iter_features(path, chunk=64))

    def test_other_shapes_fall_back_to_json_load_but_not_when_huge(self):
        path = os.path.join(self.tmp, "pretty.geojson")
        feats = [poly_feature(0.0, 0.0, 2.0, 2.0, "shallow"), poly_feature(4.0, 4.0, 6.0, 6.0, "severe")]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc(feats), f, indent=2)
        self.assertEqual(list(bf.iter_features(path)), feats)
        old = bf._JSON_FALLBACK_MAX
        bf._JSON_FALLBACK_MAX = 10
        try:
            with self.assertRaises(ValueError):
                list(bf.iter_features(path))
        finally:
            bf._JSON_FALLBACK_MAX = old


# ------------------------------------------------------------------------------------------------ rasterisation
class TestRasterise(unittest.TestCase):
    TR = Affine(5.0, 0, 1000.0, 0, -5.0, 2000.0)         # cell (r,c) centre = (1002.5 + 5c, 1997.5 - 5r)
    SHAPE = (20, 24)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def raster(self, feats):
        path = os.path.join(self.tmp, "t.geojson")
        write_compact(path, feats)
        return bf.read_tier_raster(path, self.TR, self.SHAPE)

    def test_each_tier_lands_at_its_cells_with_its_code(self):
        # x 1015..1035 -> centres 1017.5..1032.5 = cols 3..6 ; y 1975..1990 -> centres 1987.5..1977.5 = rows 2..4
        feats = [poly_feature(1015.0, 1975.0, 1035.0, 1990.0, "shallow"),
                 poly_feature(1050.0, 1960.0, 1070.0, 1975.0, "moderate"),      # cols 10..13, rows 5..7
                 poly_feature(1005.0, 1905.0, 1020.0, 1920.0, "deep"),          # cols 1..3, rows 16..18
                 poly_feature(1080.0, 1905.0, 1100.0, 1925.0, "severe")]        # cols 16..19, rows 15..18
        tier, info = self.raster(feats)
        want = np.zeros(self.SHAPE, np.uint8)
        want[2:5, 3:7] = 1
        want[5:8, 10:14] = 2
        want[16:19, 1:4] = 3
        want[15:19, 16:20] = 4
        np.testing.assert_array_equal(tier, want)
        self.assertEqual(info["features"], 4)
        self.assertEqual(info["by_tier"], {"shallow": 1, "moderate": 1, "deep": 1, "severe": 1})

    def test_cell_centre_rule_not_all_touched(self):
        # a polygon that covers only a corner of a cell (centre not inside) must not mark it
        tier, _ = self.raster([poly_feature(1000.0, 1990.0, 1004.0, 1999.0, "shallow")])    # covers x<1004: not the centre 1002.5? yes it is
        self.assertEqual(int(tier[0, 0]), 1)
        tier, _ = self.raster([poly_feature(1003.0, 1990.0, 1008.0, 1999.0, "shallow")])    # centre 1002.5 outside, 1007.5 inside col 1
        self.assertEqual(int(tier[0, 0]), 0)
        self.assertEqual(int(tier[0, 1]), 1)

    def test_deeper_tier_wins_on_overlap_hole_and_multipolygon(self):
        shallow = poly_feature(1010.0, 1950.0, 1060.0, 1990.0, "shallow")                 # cols 2..11, rows 2..9
        severe = poly_feature(1030.0, 1960.0, 1050.0, 1980.0, "severe")                   # inside it: cols 6..9, rows 4..7
        tier, _ = self.raster([shallow, severe])
        self.assertTrue((tier[4:8, 6:10] == 4).all())
        self.assertEqual(int(tier[2, 2]), 1)
        # hole
        holed = poly_feature(1010.0, 1950.0, 1060.0, 1990.0, "moderate",
                             holes=[[[1030.0, 1960.0], [1030.0, 1980.0], [1050.0, 1980.0], [1050.0, 1960.0], [1030.0, 1960.0]]])
        tier, _ = self.raster([holed])
        self.assertTrue((tier[4:8, 6:10] == 0).all())
        self.assertEqual(int(tier[2, 2]), 2)
        multi = {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [
            [[[1000.0, 1995.0], [1000.0, 1985.0], [1010.0, 1985.0], [1010.0, 1995.0], [1000.0, 1995.0]]],
            [[[1100.0, 1905.0], [1100.0, 1895.0], [1110.0, 1895.0], [1110.0, 1905.0], [1100.0, 1905.0]]]]},
            "properties": {"tier": "deep"}}
        tier, _ = self.raster([multi])
        self.assertTrue((tier[1:3, 0:2] == 3).all())        # centres y 1992.5, 1987.5 and x 1002.5, 1007.5
        self.assertTrue((tier[19, 20:22] == 3).all())       # the second part: row 19 only, the next row is off the grid
        self.assertEqual(int((tier == 3).sum()), 4 + 2)

    def test_unknown_tier_and_outside_polygons_are_ignored(self):
        feats = [poly_feature(1015.0, 1975.0, 1035.0, 1990.0, "bogus"), poly_feature(9000.0, 9000.0, 9010.0, 9010.0, "deep"),
                 {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1010.0, 1990.0]}, "properties": {"tier": "deep"}}]
        tier, info = self.raster(feats)
        self.assertEqual(int(tier.sum()), 0)
        self.assertEqual(info["skipped"], 2)               # bogus tier + point geometry; the far polygon is just outside

    def test_source_area_matches_raster_area_for_aligned_polygons(self):
        feats = [poly_feature(1015.0, 1975.0, 1035.0, 1990.0, "shallow"), poly_feature(1080.0, 1905.0, 1100.0, 1925.0, "severe")]
        path = os.path.join(self.tmp, "a.geojson")
        write_compact(path, feats)
        a = bf.source_area_km2(path)
        self.assertAlmostEqual(a["total"] * 1e6, 20 * 15 + 20 * 20, places=6)
        self.assertAlmostEqual(a["by_tier"]["shallow"] * 1e6, 300.0, places=6)
        tier, _ = bf.read_tier_raster(path, self.TR, self.SHAPE)
        self.assertAlmostEqual((tier > 0).sum() * 25.0, 700.0, places=6)          # aligned to the cells: exact
        clipped = bf.source_area_km2(path, (1000.0, 1900.0, 1090.0, 2000.0))     # cuts the severe polygon at x=1090
        self.assertAlmostEqual(clipped["inside"] * 1e6, 300.0 + 10 * 20, places=6)
        self.assertAlmostEqual(clipped["total"] * 1e6, 700.0, places=6)

    def test_holes_are_subtracted_in_the_area(self):
        holed = poly_feature(1010.0, 1950.0, 1060.0, 1990.0, "moderate",
                             holes=[[[1030.0, 1960.0], [1030.0, 1980.0], [1050.0, 1980.0], [1050.0, 1960.0], [1030.0, 1960.0]]])
        path = os.path.join(self.tmp, "h.geojson")
        write_compact(path, [holed])
        self.assertAlmostEqual(bf.source_area_km2(path)["total"] * 1e6, 50 * 40 - 20 * 20, places=6)


# ------------------------------------------------------------------------------------------------ Manning
class TestManning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "layers"))
        self.dem = manning_dem()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def geo(self, name, kind, geoms):
        feats = [{"type": "Feature", "properties": {}, "geometry": g} for g in geoms]
        path = os.path.join(self.tmp, "layers", name)
        dump_json(path, {"type": "FeatureCollection", "name": kind, "features": feats})
        return path

    def test_buildings_roads_buffer_and_priority(self):
        # row 10 centre y = Y0 - 10.5*DX. A road 3 m north of it, west to east across the whole raster.
        yc = Y0 - 10.5 * DX
        road_y = yc + 3.0
        x_lo, x_hi = X0 - 50.0, X0 + 50 * DX + 50.0
        road = {"type": "LineString", "coordinates": utm_to_lonlat([(x_lo, road_y), (x_hi, road_y)])}
        # a building over cols 20..24 and rows 8..13 (crosses the road rows 9..10)
        bx0, by0, bx1, by1 = cell_box(8, 14, 20, 25)
        bld = {"type": "Polygon", "coordinates": [utm_to_lonlat([(bx0, by0), (bx0, by1), (bx1, by1), (bx1, by0), (bx0, by0)])]}
        rpath = self.geo("roads.geojson", "roads", [road])
        bpath = self.geo("buildings.geojson", "buildings", [bld])
        cls = bf.manning_classes(self.dem, bpath, rpath)
        n = bf.manning_from_classes(cls)
        # distances from the road line to the row-r cell centres: row 10 -> 3 m, row 9 -> 2 m (road), row 8 -> 7 m, row 11 -> 8 m
        for r, want in ((9, 1), (10, 1), (8, 0), (11, 0), (12, 0)):
            self.assertEqual(int(cls[r, 5]), want, "row %d" % r)
        self.assertEqual(int(cls[10, 22]), 2)              # building wins over the road
        self.assertEqual(int(cls[9, 22]), 2)
        self.assertEqual(int(cls[12, 22]), 2)
        self.assertEqual(int(cls[7, 22]), 0)
        self.assertEqual(int(cls[10, 19]), 1)              # just west of the building the road is back
        self.assertAlmostEqual(float(n[10, 5]), 0.016, places=6)
        self.assertAlmostEqual(float(n[10, 22]), 0.08, places=6)
        self.assertAlmostEqual(float(n[30, 30]), 0.035, places=6)
        self.assertEqual(n.dtype, np.float32)

    def test_cache_roundtrip_and_invalidation(self):
        yc = Y0 - 10.5 * DX
        road = {"type": "LineString", "coordinates": utm_to_lonlat([(X0 - 10.0, yc), (X0 + 40 * DX, yc)])}
        self.geo("roads.geojson", "roads", [road])
        self.geo("buildings.geojson", "buildings", [])
        cache = os.path.join(self.tmp, "cache")
        n1 = bf.load_manning(self.tmp, self.dem, cache)
        self.assertTrue(os.path.isfile(os.path.join(cache, "manning_classes.npz")))
        n2 = bf.load_manning(self.tmp, self.dem, cache)
        np.testing.assert_array_equal(n1, n2)
        self.assertEqual(float(n1[10, 5]), np.float32(0.016))
        # change the roads file: the cache key changes, so the new raster is built
        self.geo("roads.geojson", "roads", [])
        n3 = bf.load_manning(self.tmp, self.dem, cache)
        self.assertEqual(float(n3[10, 5]), np.float32(0.035))

    def test_missing_files_give_open_ground(self):
        n = bf.load_manning(self.tmp, self.dem, None)
        self.assertTrue((n == np.float32(0.035)).all())


# ------------------------------------------------------------------------------------------------ DEM unit
class TestDemUnit(unittest.TestCase):
    def test_feet_are_converted_to_metres_and_nodata_is_nan(self):
        tmp = tempfile.mkdtemp()
        try:
            z_m = np.full((6, 8), 200.0)
            z_m[2, 3] = 250.0
            write_dem(os.path.join(tmp, "dem_bg.tif"), z_m)
            import rasterio
            with rasterio.open(os.path.join(tmp, "dem_bg.tif"), "r+") as dst:
                a = dst.read(1)
                a[0, 0] = -9999.0
                dst.write(a, 1)
            dem = bf.load_dem(tmp)
            self.assertAlmostEqual(float(dem["z"][3, 3]), 200.0, places=3)
            self.assertAlmostEqual(float(dem["z"][2, 3]), 250.0, places=3)
            self.assertTrue(np.isnan(dem["z"][0, 0]))
            self.assertFalse(dem["valid"][0, 0])
            self.assertTrue(dem["valid"][1, 1])
            self.assertAlmostEqual(dem["dx"], DX, places=6)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ the smoothed surface
def dilate4(m):
    d = m.copy()
    d[1:] |= m[:-1]
    d[:-1] |= m[1:]
    d[:, 1:] |= m[:, :-1]
    d[:, :-1] |= m[:, 1:]
    return d


def bowl_case(level=2.0, n=61, radius=14.0):
    """A round basin with a flat lake at `level` metres above its bottom; tiers from the TRUE depth of that lake."""
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n // 2, xx - n // 2)
    z = np.minimum((r / radius) ** 2 * level * 1.4, level * 3.0)            # parabolic bed, rim a little above the lake
    true = np.maximum(level - z, 0.0)
    tier = np.zeros(z.shape, np.uint8)
    for k, (lo, hi) in enumerate(zip(bf.TIER_LO, list(bf.TIER_HI[:3]) + [1e9])):
        tier[(true >= lo) & (true < hi)] = k + 1
    return z, tier, true


class TestSmoothSurface(unittest.TestCase):
    def test_midpoints_when_sweeps_zero(self):
        z = np.zeros((20, 20))
        tier = np.zeros((20, 20), np.uint8)
        tier[5:15, 5:15] = 3
        out = bf.smooth_surface(tier, z, sweeps=0)
        np.testing.assert_allclose(out["depth"][tier > 0], 0.9)
        self.assertEqual(out["volume_ratio"], 1.0)

    def test_bands_footprint_halo_and_volume_on_flat_bed(self):
        n = 41
        yy, xx = np.mgrid[0:n, 0:n]
        ring = np.maximum(np.abs(yy - n // 2), np.abs(xx - n // 2))
        tier = np.zeros((n, n), np.uint8)
        tier[ring <= 14] = 1
        tier[ring <= 10] = 2
        tier[ring <= 6] = 3
        tier[ring <= 3] = 4
        z = np.zeros((n, n))
        out = bf.smooth_surface(tier, z)
        d, halo = out["depth"], out["halo"]
        foot = tier > 0
        # footprint unchanged: depth > 0 exactly on the tiers, and wet cells stay wet
        np.testing.assert_array_equal(d > 0, foot)
        self.assertGreaterEqual(float(d[foot].min()), bf.MIN_DEPTH_M - 1e-6)
        # on a flat bed the bands are exact: the smoothing only moves depth inside each tier's interval
        lo = np.array((0,) + bf.TIER_LO)[tier]
        hi = np.array((0,) + bf.TIER_HI)[tier]
        self.assertTrue((d[foot] >= lo[foot] - 1e-6).all(), "depth below the tier floor")
        self.assertTrue((d[foot] <= hi[foot] + 1e-6).all(), "depth above the tier ceiling")
        # no water outside the footprint dilated by one cell; the halo is that one cell and as deep as the water beside it
        # (a halo thinner than that is a drop into the dry cell, which Manning turns into a jet along every rim)
        outside = ~dilate4(foot)
        self.assertEqual(float((d + halo)[outside].max()), 0.0)
        self.assertEqual(float(halo[foot].max()), 0.0)
        self.assertLessEqual(float(halo.max()), bf.HALO_MAX_M + 1e-6)
        self.assertGreater(float(halo.max()), 0.0)
        c = n // 2
        for r0, r1, c0, c1 in ((6, 5, c, c), (34, 35, c, c), (c, c, 6, 5), (c, c, 34, 35)):      # wet edge cell -> its dry neighbour
            self.assertAlmostEqual(float(halo[r1, c1]), float(d[r0, c0]), places=5)
        self.assertGreater(float(halo[5, c]), 0.1)                 # deeper than the 0.1 m cap of the first version
        # volume: tolerance stated here is 15 % of the midpoint volume for a staircase of every tier on a flat bed
        self.assertLess(abs(out["volume_ratio"] - 1.0), 0.15, out["volume_ratio"])
        mid = np.array((0,) + bf.TIER_MID)[tier]
        self.assertAlmostEqual(out["volume_ratio"], float(d[foot].sum() / mid[foot].sum()), places=6)

    def test_surface_is_smoother_than_the_staircase(self):
        n = 41
        yy, xx = np.mgrid[0:n, 0:n]
        ring = np.maximum(np.abs(yy - n // 2), np.abs(xx - n // 2))
        tier = np.zeros((n, n), np.uint8)
        tier[ring <= 14] = 1
        tier[ring <= 10] = 2
        tier[ring <= 6] = 3
        z = np.zeros((n, n))

        def energy(depth):
            e = depth.astype(np.float64)
            f = depth > 0
            tot = 0.0
            for a, b, fa, fb in ((e[1:], e[:-1], f[1:], f[:-1]), (e[:, 1:], e[:, :-1], f[:, 1:], f[:, :-1])):
                m = fa & fb
                tot += float(((a - b)[m] ** 2).sum())
            return tot
        raw = bf.smooth_surface(tier, z, sweeps=0)["depth"]
        sm = bf.smooth_surface(tier, z)["depth"]
        self.assertLess(energy(sm), 0.5 * energy(raw))

    def test_pond_at_rest_is_flat_and_calm(self):
        z, tier, true = bowl_case()
        zz = z + 100.0
        n_man = np.full(z.shape, 0.035, np.float32)
        tr = Affine(DX, 0, X0, 0, -DX, Y0)

        def run(sweeps):
            s = bf.smooth_surface(tier, zz, sweeps=sweeps)
            u, v = bf.velocity_fields(s["depth"], s["halo"], zz, n_man, DX, tr)
            eta = (zz + s["depth"])[tier > 0]
            return float(np.hypot(u, v).max()), float(eta.max() - eta.min()), s
        sp_raw, spread_raw, _ = run(0)
        sp_sm, spread_sm, s = run(bf.SWEEPS_DEFAULT)
        # the midpoint staircase reads a strong flow in a pond at rest, the smoothed surface does not
        self.assertGreater(sp_raw, 1.0)
        self.assertLess(sp_sm, 0.3)
        self.assertLess(spread_sm, 0.5 * spread_raw)
        self.assertLess(spread_sm, 0.25)                     # within the +-0.1 m the tiers allow, plus the severe tier's slack
        # and the depth it reconstructs is close to the true depth of the lake
        foot = tier > 0
        self.assertLess(float(np.abs(s["depth"][foot] - true[foot]).mean()), 0.12)

    def test_inclined_shallow_strip_flows_downslope(self):
        n_r, n_c = 12, 140
        xx = np.arange(n_c)[None, :] * DX
        z = np.repeat(200.0 - 0.01 * xx, n_r, axis=0)          # 1 % slope, down toward +column (east)
        tier = np.zeros((n_r, n_c), np.uint8)
        tier[4:8, 5:-5] = 1
        s = bf.smooth_surface(tier, z)
        # the shallow band cannot flatten a 1 % slope over 700 m: the surface keeps at least (drop - 0.2) m
        eta = z + s["depth"]
        drop = eta[5, 10] - eta[5, n_c - 11]
        self.assertGreater(drop, 0.9 * (z[5, 10] - z[5, n_c - 11]) - 0.2)
        tr = Affine(DX, 0, X0, 0, -DX, Y0)
        u, v = bf.velocity_fields(s["depth"], s["halo"], z, np.full(z.shape, 0.035, np.float32), DX, tr)
        mid = (slice(5, 7), slice(30, 110))
        self.assertGreater(float(np.median(u[mid])), 0.2)
        self.assertLess(abs(float(np.median(v[mid]))), 0.05 * float(np.median(u[mid])) + 0.01)
        # Manning at the bed slope is an upper bound for a surface that is at most as steep
        self.assertLess(float(u[mid].max()), 0.3 ** (2 / 3) * math.sqrt(0.0105) / 0.035)

    def test_steep_wall_bands_widen_but_stay_wet_and_bounded(self):
        # a deep tier next to a shallow tier 3 m higher: infeasible with exact bands, feasible with relief-widened ones
        z = np.zeros((16, 16))
        z[:, 8:] = 3.0
        tier = np.zeros((16, 16), np.uint8)
        tier[4:12, 2:8] = 4
        tier[4:12, 8:11] = 1
        out = bf.smooth_surface(tier, z)
        d = out["depth"]
        self.assertTrue(np.isfinite(d).all())
        self.assertGreaterEqual(float(d[tier > 0].min()), bf.MIN_DEPTH_M - 1e-6)
        self.assertLessEqual(float(d.max()), 6.0 + 1e-6)
        eta = (z + d)[tier > 0]
        self.assertLess(float(eta.max() - eta.min()), 3.5)

    def test_gaussian_polish_stays_inside_the_bands(self):
        n = 41
        yy, xx = np.mgrid[0:n, 0:n]
        ring = np.maximum(np.abs(yy - n // 2), np.abs(xx - n // 2))
        tier = np.zeros((n, n), np.uint8)
        tier[ring <= 14] = 1
        tier[ring <= 8] = 2
        z = np.zeros((n, n))
        d = bf.smooth_surface(tier, z, sigma=1.5)["depth"]
        lo = np.array((0,) + bf.TIER_LO)[tier]
        hi = np.array((0,) + bf.TIER_HI)[tier]
        self.assertTrue((d[tier > 0] >= lo[tier > 0] - 1e-6).all() and (d[tier > 0] <= hi[tier > 0] + 1e-6).all())

    def test_empty_tier_raster(self):
        out = bf.smooth_surface(np.zeros((8, 8), np.uint8), np.zeros((8, 8)))
        self.assertEqual(float(out["depth"].max()), 0.0)
        self.assertEqual(out["volume_ratio"], 1.0)

    def test_sor_matches_the_exact_solution_on_a_line(self):
        # 1-D chain of 9 cells, ends pinned to eta 0 and 8: the unconstrained minimiser is the straight line
        n = 9
        wet = np.zeros((3, n), bool)
        wet[1] = True
        ys, xs, nb, cnt, colour = bf._wet_graph(wet)
        lo = np.full(n, -100.0)
        hi = np.full(n, 100.0)
        lo[0] = hi[0] = 0.0
        lo[-1] = hi[-1] = 8.0
        eta = bf._sor(np.zeros(n), lo, hi, nb, cnt, colour, 200)
        np.testing.assert_allclose(eta, np.arange(n, dtype=float), atol=1e-6)
        # bounds are honoured: a cell forced above the line pulls its neighbours but never exceeds its own bounds
        lo2 = lo.copy()
        lo2[4] = 6.0
        hi2 = hi.copy()
        eta2 = bf._sor(np.zeros(n), lo2, hi2, nb, cnt, colour, 400)
        self.assertGreaterEqual(float(eta2[4]), 6.0 - 1e-9)
        self.assertTrue(np.all(np.diff(eta2[:5]) >= -1e-9))


# ------------------------------------------------------------------------------------------------ velocity limits
class TestVelocityFields(unittest.TestCase):
    TR = Affine(DX, 0, X0, 0, -DX, Y0)

    def test_unit_discharge_cap_and_zero_outside_footprint(self):
        n_r, n_c = 14, 40
        xx = np.arange(n_c)[None, :] * DX
        z = np.repeat(200.0 - 0.05 * xx, n_r, axis=0)               # 5 % slope
        depth = np.zeros((n_r, n_c), np.float32)
        depth[3:11, 4:36] = 4.0                                      # a 4 m deep block with the bed's slope in its surface
        halo = np.zeros_like(depth)
        n = np.full(z.shape, 0.035, np.float32)
        u, v = bf.velocity_fields(depth, halo, z, n, DX, self.TR)
        sp = np.hypot(u, v)
        self.assertLessEqual(float(sp.max()) * 4.0, fg.QMAX + 1e-3)
        self.assertEqual(float(sp[depth == 0].max()), 0.0)
        # without the cap the same field is much faster (the cap is what limits it)
        ug, vg = fg.face_velocity(depth, z, n, DX)
        self.assertGreater(float(np.hypot(ug, vg).max()), 2.0 * float(sp.max()))
        # shallow water is not touched by the cap (nor, with the dead band off, by anything else)
        d2 = np.zeros_like(depth)
        d2[3:11, 4:36] = 0.2
        u2, v2 = bf.velocity_fields(d2, halo, z, n, DX, self.TR, dead_band=0.0)
        ug2, vg2 = fg.face_velocity(d2, z, n, DX)
        c = (slice(5, 9), slice(10, 30))
        np.testing.assert_allclose(np.hypot(u2, v2)[c], np.hypot(ug2, vg2)[c], rtol=1e-4)
        self.assertLess(float(sp.max()), float(np.hypot(u2, v2).max()))
        # the default dead band scales that speed by sqrt(1 - dead_band / S), S = 5 % here (the surface follows the bed)
        u3, v3 = bf.velocity_fields(d2, halo, z, n, DX, self.TR)
        want = np.hypot(ug2, vg2)[c] * math.sqrt(1.0 - bf.SLOPE_DEAD_BAND / 0.05)
        np.testing.assert_allclose(np.hypot(u3, v3)[c], want, rtol=2e-3)

    def test_rotation_to_true_north(self):
        n_r, n_c = 14, 60
        xx = np.arange(n_c)[None, :] * DX
        z = np.repeat(200.0 - 0.01 * xx, n_r, axis=0)               # water runs along +column = UTM grid east
        depth = np.zeros((n_r, n_c), np.float32)
        depth[3:11, 4:56] = 0.2
        n = np.full(z.shape, 0.035, np.float32)
        u, v = bf.velocity_fields(depth, np.zeros_like(depth), z, n, DX, self.TR)
        c = (slice(5, 9), slice(20, 40))
        lon, lat = fg.utm16n_to_lnglat(X0 + 30 * DX, Y0 - 7 * DX)
        gamma = float(fg.convergence_angle_rad(lon, lat))
        self.assertGreater(gamma, 0.003)                             # about 0.3 degrees here
        ratio = float(np.median(v[c]) / np.median(u[c]))
        self.assertAlmostEqual(ratio, -math.tan(gamma), delta=0.1 * math.tan(gamma))


class TestSlopeDeadBand(unittest.TestCase):
    TR = Affine(DX, 0, X0, 0, -DX, Y0)

    @staticmethod
    def plane(slope, n_r=16, n_c=60, depth=0.5):
        xx = np.arange(n_c)[None, :] * DX
        z = np.repeat(200.0 - slope * xx, n_r, axis=0)               # down toward +column (east)
        h = np.zeros((n_r, n_c), np.float32)
        h[3:13, 4:56] = depth                                        # same depth: the surface has the bed's slope
        return z, h

    def test_surface_slope_of_a_plane_and_of_dry_cells(self):
        z, h = self.plane(0.01)
        s = bf.surface_slope(h, z, DX)
        c = (slice(5, 11), slice(8, 50))
        np.testing.assert_allclose(s[c], 0.01, rtol=1e-5)
        self.assertEqual(float(s[:2].max()), 0.0)                    # dry rows carry no flowing face
        self.assertEqual(float(s[:, :2].max()), 0.0)
        # flat water in a basin: zero slope even though the bed under it is not flat
        zb = np.zeros((12, 12))
        zb[4:8, 4:8] = -1.0
        hb = np.zeros((12, 12), np.float32)
        hb[4:8, 4:8] = 1.0                                           # lake level 0 everywhere over the wet cells
        sb = bf.surface_slope(hb, zb, DX)
        self.assertAlmostEqual(float(sb[5:7, 5:7].max()), 0.0, places=7)

    def test_strips_equal_the_whole_grid(self):
        rnd = np.random.default_rng(3)
        z = 200.0 + np.cumsum(rnd.normal(0, 0.05, (40, 30)), axis=0)
        h = np.where(rnd.random((40, 30)) < 0.7, rnd.uniform(0.0, 0.6, (40, 30)), 0.0).astype(np.float32)
        np.testing.assert_array_equal(bf.surface_slope(h, z, DX, strip_rows=7), bf.surface_slope(h, z, DX, strip_rows=1024))

    def test_gentle_surface_is_at_rest_and_steeper_surface_keeps_nearly_its_speed(self):
        n = np.full((16, 60), 0.035, np.float32)
        halo = np.zeros((16, 60), np.float32)
        c = (slice(5, 11), slice(10, 50))
        z, h = self.plane(0.0015)                                    # below the 0.002 band: not measurable
        u, v = bf.velocity_fields(h, halo, z, n, DX, self.TR)
        self.assertEqual(float(np.hypot(u, v)[c].max()), 0.0)         # (the block's own edge is a cliff: no halo here)
        u0, v0 = bf.velocity_fields(h, halo, z, n, DX, self.TR, dead_band=0.0)
        self.assertGreater(float(np.hypot(u0, v0)[c].min()), 0.3)    # without the band the same water runs at about 0.4 m/s
        z, h = self.plane(0.004)                                     # twice the band: speed x sqrt(0.5)
        u, v = bf.velocity_fields(h, halo, z, n, DX, self.TR)
        u0, v0 = bf.velocity_fields(h, halo, z, n, DX, self.TR, dead_band=0.0)
        np.testing.assert_allclose(np.hypot(u, v)[c], np.hypot(u0, v0)[c] * math.sqrt(0.5), rtol=2e-3)
        z, h = self.plane(0.04)                                      # 20 x the band: -2.5 %
        u, v = bf.velocity_fields(h, halo, z, n, DX, self.TR)
        u0, v0 = bf.velocity_fields(h, halo, z, n, DX, self.TR, dead_band=0.0)
        np.testing.assert_allclose(np.hypot(u, v)[c], np.hypot(u0, v0)[c] * math.sqrt(1 - 0.05), rtol=2e-3)
        # direction is unchanged by the band
        ang = np.arctan2(v, u)[c]
        ang0 = np.arctan2(v0, u0)[c]
        np.testing.assert_allclose(ang, ang0, atol=1e-5)


# ------------------------------------------------------------------------------------------------ whole storm folder
class SyntheticApp:
    """A tiny app folder: 140 x 160 cell DEM in feet (0.4 % slope toward east), one storm with 3 hours."""
    ROWS, COLS = 140, 160

    def __init__(self, root):
        self.root = root
        self.app = os.path.join(root, "app")
        os.makedirs(os.path.join(self.app, "layers"))
        xx = np.arange(self.COLS)[None, :] * DX
        self.z = np.repeat(200.0 - 0.004 * xx, self.ROWS, axis=0)
        # water sits where the ground is lower than its surroundings: a 0.5 m trough under the strip, a 2.5 m pit under the lake
        self.z[30:33, 20:120] -= 0.5
        self.z[70:100, 100:130] -= 2.5
        write_dem(os.path.join(self.app, "dem_bg.tif"), self.z)
        for name in ("buildings", "roads"):
            dump_json(os.path.join(self.app, "layers", name + ".geojson"), {"type": "FeatureCollection", "features": []})
        self.sdir = os.path.join(self.app, "BG FLOOD MAP VISTA SUPERCOMPUTER", "v2", "5yr")
        os.makedirs(self.sdir)
        self.block = cell_box(70, 100, 100, 130)             # severe lake, 30 x 30 cells (152.7 m square)
        self.strip = cell_box(30, 33, 20, 120)               # shallow strip 3 x 100 cells along the slope, NW of the lake
        self.small = cell_box(75, 85, 105, 115)              # hour 2: a small shallow patch inside hour 3's lake
        write_compact(self.hour_path(1), [])
        write_compact(self.hour_path(2), [poly_feature(*self.small, "shallow")])
        write_compact(self.hour_path(3), [poly_feature(*self.block, "severe"), poly_feature(*self.strip, "shallow")])
        dump_json(os.path.join(self.sdir, "run_summary.json"),
                  {"storm": "5yr", "solver": "v2", "total_rain_mm": 106.7, "final_mass_balance": {"stored_m3": 1.0}})
        self.out_root = os.path.join(root, "out")
        self.cache = os.path.join(root, "cache")
        self.dem = bf.load_dem(self.app)
        self.manning = bf.load_manning(self.app, self.dem, self.cache)

    def hour_path(self, h):
        return os.path.join(self.sdir, "v2_5yr_depth_%06ds.geojson" % (h * 3600))

    def build(self, hours=(1, 2, 3), **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = bf.build_storm(self.app, "5yr", list(hours), self.dem, self.manning, cache_dir=self.cache,
                                 out_root=self.out_root, **kw)
        return res, buf.getvalue()

    @property
    def out_dir(self):
        return os.path.join(self.out_root, "5yr")


def merc_centroid_of_box(box):
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    lon, lat = fg.utm16n_to_lnglat(cx, cy)
    mx, my = fg.lnglat_to_merc(lon, lat)
    return float(mx), float(my)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.syn = SyntheticApp(cls.tmp)
        cls.res, cls.log = cls.syn.build()
        with open(os.path.join(cls.syn.out_dir, "meta.json"), encoding="utf-8") as f:
            cls.meta = json.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def hour(self, h):
        return fg.read_rgba_gz(os.path.join(self.syn.out_dir, fg.hour_filename(h)), self.meta["width"], self.meta["height"])

    def test_valid_storm_dir_and_meta(self):
        self.assertEqual(self.res["problems"], [])
        self.assertEqual(fg.validate_storm_dir(self.syn.out_dir), [])
        m = self.meta
        self.assertEqual(m["source"], "tiers")
        self.assertIn("tier", m["source_note"].lower())
        self.assertEqual(m["storm"], "5yr")
        self.assertEqual(m["storm_depth_in"], 4.32)
        self.assertEqual(m["run_summary"]["total_rain_mm"], 106.7)
        self.assertEqual(m["hours"], [2, 3])
        self.assertEqual([e["hour"] for e in m["hourly"]], [2, 3])
        self.assertEqual(m["cell_m"], 6.4)
        self.assertAlmostEqual(m["cell_ground_m"], 6.4 * math.cos(math.radians(36.96)), delta=0.02)
        self.assertEqual(m["tiers_build"]["dry_hours"], [1])
        self.assertEqual(m["tiers_build"]["params"]["slope_dead_band"], bf.SLOPE_DEAD_BAND)
        self.assertIn("at rest", m["source_note"])
        self.assertIn("not solver output", m["source_note"])
        self.assertTrue(os.path.isfile(os.path.join(self.syn.out_dir, "flowpaths.rgba.gz")))

    def test_empty_hour_is_flagged_and_gives_no_file(self):
        self.assertEqual(self.res["dry_hours"], [1])
        self.assertFalse(os.path.exists(os.path.join(self.syn.out_dir, "h01.rgba.gz")))
        self.assertIn("h01 DRY", self.log)
        self.assertIn("flagged", bf.build_storm.__doc__)

    def test_valid_flag_marks_the_dem_footprint(self):
        rgba = self.hour(3)
        A = rgba[..., 3]
        self.assertEqual(set(np.unique(A).tolist()), {0, 255})
        self.assertEqual(int(A[0, 0]), 0)                    # the 150 m margin is outside the model domain
        self.assertEqual(int(A[A.shape[0] // 2, A.shape[1] // 2]), 255)
        # independent reference: project every pixel centre back to UTM; it is valid when it lies inside the DEM rectangle
        from rasterio.warp import transform
        m = self.meta
        cell = m["cell_m"]
        rr, cc = np.mgrid[0:A.shape[0], 0:A.shape[1]]
        mx = m["bounds_merc"][0] + (cc + 0.5) * cell
        my = m["bounds_merc"][3] - (rr + 0.5) * cell
        lon, lat = fg.merc_to_lnglat(mx.ravel(), my.ravel())
        e, n = transform("EPSG:4326", "EPSG:32616", lon.tolist(), lat.tolist())
        e = np.asarray(e).reshape(A.shape)
        n = np.asarray(n).reshape(A.shape)
        inside = (e >= X0) & (e <= X0 + self.syn.COLS * DX) & (n <= Y0) & (n >= Y0 - self.syn.ROWS * DX)
        self.assertLess(float(((A == 255) != inside).mean()), 0.005)
        self.assertGreater(int((A == 0).sum()), 0)
        self.assertGreater(int((A == 255).sum()), 0.5 * A.size)
        self.assertEqual(int(rgba[..., 0][A == 0].max()), 0)

    def test_polygon_position_in_the_output_is_within_one_pixel(self):
        rgba = self.hour(3)
        m = self.meta
        cell = m["cell_m"]
        xmin, ymax = m["bounds_merc"][0], m["bounds_merc"][3]
        R = rgba[..., 0]
        wet = R > 0
        rows, cols = np.nonzero(wet)
        # the lake block is the part of the footprint at the east/south: split from the strip by its row range
        lake_rows = rows > (rows.min() + rows.max()) // 2 - 5
        lake = np.zeros_like(wet)
        lake[rows[lake_rows], cols[lake_rows]] = True
        lr, lc = np.nonzero(lake)
        x = xmin + (lc.mean() + 0.5) * cell
        y = ymax - (lr.mean() + 0.5) * cell
        ex, ey = merc_centroid_of_box(self.syn.block)
        self.assertLessEqual(abs(x - ex), cell, "x off by %.1f m" % (x - ex))
        self.assertLessEqual(abs(y - ey), cell, "y off by %.1f m" % (y - ey))
        # extent of the lake in pixels: 152.7 m ground = 152.7 / cos(lat) mercator m
        side_px = 30 * DX / math.cos(math.radians(36.96)) / cell
        self.assertAlmostEqual(lc.max() - lc.min() + 1, side_px, delta=2.0)
        self.assertAlmostEqual(lr.max() - lr.min() + 1, side_px, delta=2.0)
        # north is up: the strip (rows 30..33 of the DEM) lies north of the lake (rows 70..100)
        sr = rows[~lake_rows]
        self.assertLess(sr.mean(), lr.mean())
        ex2, ey2 = merc_centroid_of_box(self.syn.strip)
        sc = cols[~lake_rows]
        self.assertLessEqual(abs(xmin + (sc.mean() + 0.5) * cell - ex2), cell)
        self.assertLessEqual(abs(ymax - (sr.mean() + 0.5) * cell - ey2), cell)

    def test_off_centre_hour_lands_at_its_own_pixels(self):
        # hour 2 is a 10 x 10 cell patch in the south-east part of the union grid: placed with its own sub-grid
        rgba = self.hour(2)
        m = self.meta
        cell = m["cell_m"]
        rows, cols = np.nonzero(rgba[..., 0] > 0)
        x = m["bounds_merc"][0] + (cols.mean() + 0.5) * cell
        y = m["bounds_merc"][3] - (rows.mean() + 0.5) * cell
        ex, ey = merc_centroid_of_box(self.syn.small)
        self.assertLessEqual(abs(x - ex), cell, "x off by %.1f m" % (x - ex))
        self.assertLessEqual(abs(y - ey), cell, "y off by %.1f m" % (y - ey))
        self.assertAlmostEqual(cols.max() - cols.min() + 1, 10 * DX / math.cos(math.radians(36.96)) / cell, delta=1.5)
        self.assertGreater(rows.mean(), 0.5 * rgba.shape[0])         # the south half of the grid
        self.assertGreater(cols.mean(), 0.5 * rgba.shape[1])         # the east half

    def test_strip_flows_east_lake_is_calm_and_depth_is_physical(self):
        rgba = self.hour(3)
        wet = rgba[..., 0] > 0
        rows, cols = np.nonzero(wet)
        split = (rows.min() + rows.max()) // 2 - 5
        strip = wet.copy()
        strip[split:] = False
        lake = wet.copy()
        lake[:split] = False
        u = fg.decode_vel(rgba[..., 1])
        v = fg.decode_vel(rgba[..., 2])
        d = fg.decode_depth(rgba[..., 0])
        ys, xs = np.nonzero(strip)
        mid = (xs > xs.min() + 20) & (xs < xs.max() - 20)
        self.assertGreater(float(np.median(u[ys[mid], xs[mid]])), 0.15)               # +u = toward east
        self.assertLess(abs(float(np.median(v[ys[mid], xs[mid]]))), 0.05)
        self.assertLess(float(np.hypot(u, v)[lake].max()), 0.5)                       # the flat lake is calm
        self.assertGreaterEqual(float(d[lake].min()), 1.0)                            # severe: at least about 1.2 m
        self.assertLessEqual(float(d[strip].max()), 0.45)                             # shallow: 0.1-0.3 m (+ pixel averaging)

    def test_flowpaths_summarise_the_hours(self):
        fp = fg.read_rgba_gz(os.path.join(self.syn.out_dir, "flowpaths.rgba.gz"), self.meta["width"], self.meta["height"])
        r2, r3 = self.hour(2)[..., 0], self.hour(3)[..., 0]
        np.testing.assert_array_equal(fp[..., 0], np.maximum(r2, r3))
        want_b = np.rint(255.0 * ((r2 > 0).astype(int) + (r3 > 0)) / 24.0)
        wet = fp[..., 0] > 0
        self.assertTrue((np.abs(fp[..., 2][wet].astype(float) - want_b[wet]) <= 1).all())
        self.assertTrue((fp[..., 1][~wet] == 0).all())

    def test_hourly_stats_and_volume_ratio(self):
        e3 = self.meta["hourly"][1]
        area_km2 = (30 * DX) ** 2 * 1e-6 + 3 * 100 * DX * DX * 1e-6
        self.assertAlmostEqual(e3["wet_area_km2"], area_km2, delta=0.08 * area_km2)
        self.assertGreater(e3["max_depth_m"], 1.2)
        ratio = self.meta["tiers_build"]["volume_ratio"]
        for h in ("2", "3"):
            self.assertLess(abs(ratio[h] - 1.0), 0.35, ratio)

    def test_resume_skips_written_hours_and_force_rebuilds_identical_bytes(self):
        files = ["h02.rgba.gz", "h03.rgba.gz", "flowpaths.rgba.gz"]
        before = {f: read_bytes(os.path.join(self.syn.out_dir, f)) for f in files}
        stamps = {f: os.stat(os.path.join(self.syn.out_dir, f)).st_mtime_ns for f in files[:2]}
        res, log = self.syn.build()
        self.assertEqual(res["skipped"], [2, 3])
        self.assertIn("h03 skipped", log)
        for f in files[:2]:
            self.assertEqual(os.stat(os.path.join(self.syn.out_dir, f)).st_mtime_ns, stamps[f])
        for f in files:
            self.assertEqual(read_bytes(os.path.join(self.syn.out_dir, f)), before[f], f)
        self.assertEqual(res["problems"], [])
        res, _ = self.syn.build(force=True)
        self.assertEqual(res["skipped"], [])
        for f in files:
            self.assertEqual(read_bytes(os.path.join(self.syn.out_dir, f)), before[f], "force: " + f)

    def test_subset_run_keeps_other_hours_when_the_grid_is_unchanged(self):
        res, _ = self.syn.build(hours=[3])
        self.assertEqual(res["hours"], [2, 3])
        self.assertEqual(res["problems"], [])
        self.assertTrue(os.path.isfile(os.path.join(self.syn.out_dir, "h02.rgba.gz")))

    def test_changed_method_needs_force(self):
        with self.assertRaises(SystemExit):
            self.syn.build(sigma=1.0)
        res, _ = self.syn.build(sigma=1.0, force=True)
        self.assertEqual(res["problems"], [])
        with self.assertRaises(SystemExit):                  # the dead band is part of the method too
            self.syn.build(dead_band=0.0)
        res, _ = self.syn.build(dead_band=0.0, force=True)
        self.assertEqual(res["problems"], [])
        self.assertNotIn("at rest", load_json(os.path.join(self.syn.out_dir, "meta.json"))["source_note"])
        res, _ = self.syn.build(force=True)                  # restore the default build for the other tests
        self.assertEqual(res["problems"], [])

    def test_crosscheck_polygon_area_against_grid(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = bf.crosscheck_storm(self.syn.app, "5yr", [3], self.syn.dem, self.syn.cache, self.syn.out_root)
        r = rows[0]
        self.assertAlmostEqual(r["polygons_km2"] * 1e6, (30 * DX) ** 2 + 3 * 100 * DX * DX, places=3)
        self.assertAlmostEqual(r["outside_dem_km2"], 0.0, places=9)
        self.assertLess(abs(r["gap_raster_pct"]), 0.001)      # aligned polygons: the DEM raster is exact
        self.assertLess(abs(r["gap_grid_pct"]), 8.0)          # edge pixels of a 150 m block: 1 pixel in 24 per side


class TestEndToEndGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.syn = SyntheticApp(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_dry_storm_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.syn.build(hours=[1])

    def test_never_overwrites_a_full_precision_build(self):
        os.makedirs(self.syn.out_dir)
        dump_json(os.path.join(self.syn.out_dir, "meta.json"), {"format": "bgflow", "source": "rasters"})
        for force in (False, True):
            with self.assertRaises(SystemExit):
                self.syn.build(force=force)
        self.assertEqual(load_json(os.path.join(self.syn.out_dir, "meta.json"))["source"], "rasters")

    def test_wet_cells_outside_the_dem_are_dropped_not_fatal(self):
        far = poly_feature(X0 - 500.0, Y0 + 400.0, X0 - 400.0, Y0 + 500.0, "deep")        # west and north of the DEM
        write_compact(self.syn.hour_path(2), [poly_feature(*self.syn.small, "shallow"), far])
        res, _ = self.syn.build(hours=[2, 3], force=True)
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["hours"], [2, 3])

    def test_cli_runs_on_the_synthetic_app(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = bf.main(["--app", self.syn.app, "--storms", "5yr", "--hours", "1-3", "--out", self.syn.out_root,
                            "--cache", self.syn.cache, "--crosscheck", "3"])
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn("validate_storm_dir: OK", text)
        self.assertIn("peak", text)
        self.assertIn("polygons", text)
        self.assertEqual(fg.validate_storm_dir(self.syn.out_dir), [])

    def test_cli_rejects_unknown_storm_and_bad_hours(self):
        for argv in (["--storms", "7yr"], ["--hours", "30"], ["--hours", "x"]):
            with self.assertRaises(SystemExit) as cm:
                with redirect_stdout(io.StringIO()), self._silence_stderr():
                    bf.main(argv + ["--app", self.syn.app])
            self.assertEqual(cm.exception.code, 2)

    @staticmethod
    def _silence_stderr():
        from contextlib import redirect_stderr
        return redirect_stderr(io.StringIO())


class TestHaloAtRest(unittest.TestCase):
    """A wet cell beside dry ground must not read a surface drop into it: ponds at rest show no velocity."""
    TR = Affine(DX, 0, X0, 0, -DX, Y0)

    def speed(self, n, tier_code, halo_max=None):
        size = n + 12
        z = np.full((size, size), 100.0)                      # perfectly flat bed
        tier = np.zeros((size, size), np.uint8)
        tier[6:6 + n, 6:6 + n] = tier_code
        kw = {} if halo_max is None else {"halo_max": halo_max}
        s = bf.smooth_surface(tier, z, **kw)
        u, v = bf.velocity_fields(s["depth"], s["halo"], z, np.full(z.shape, 0.035, np.float32), DX, self.TR)
        return float(np.hypot(u, v).max()), float(s["depth"][tier > 0].mean())

    def test_flat_bed_ponds_of_any_size_are_at_rest(self):
        for n in (2, 5, 9, 25):
            for code in (2, 3):
                sp, depth = self.speed(n, code)
                self.assertGreater(depth, 0.3, (n, code))      # a real pond, deeper than the old 0.1 m halo
                self.assertLess(sp, 0.01, (n, code, sp))

    def test_the_old_capped_halo_is_what_made_the_rim_jet(self):
        sp, _ = self.speed(5, 2, halo_max=0.1)
        self.assertGreater(sp, 1.0)

    def test_halo_follows_the_surface_but_not_above_it(self):
        z = np.zeros((9, 9))
        z[:, 6:] = 5.0                                         # a wall east of the pond, higher than its surface
        tier = np.zeros((9, 9), np.uint8)
        tier[3:6, 3:6] = 3
        s = bf.smooth_surface(tier, z)
        eta = (z + s["depth"])[4, 3]
        self.assertAlmostEqual(float(s["halo"][4, 2] + z[4, 2]), float(eta), delta=0.05)   # continues the surface west
        self.assertEqual(float(s["halo"][4, 6]), 0.0)          # bed above the surface: no water


class TestCacheRobustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.syn = SyntheticApp(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tier_cache(self):
        return os.path.join(self.syn.cache, "5yr_03.npz")

    @staticmethod
    def damage(path, how):
        data = read_bytes(path)
        with open(path, "wb") as f:
            f.write({"half": data[:len(data) // 2], "empty": b"", "garbage": b"not a zip file at all" * 50}[how])

    def test_damaged_tier_cache_is_rebuilt(self):
        ref, _ = bf.tier_hour_cached(self.syn.app, "5yr", 3, self.syn.dem, self.syn.cache)
        path = self.tier_cache()
        self.assertTrue(os.path.isfile(path))
        for how in ("half", "empty", "garbage"):
            self.damage(path, how)
            tier, info = bf.tier_hour_cached(self.syn.app, "5yr", 3, self.syn.dem, self.syn.cache)
            self.assertFalse(info.get("cached"), how)
            np.testing.assert_array_equal(tier, ref)
            tier, info = bf.tier_hour_cached(self.syn.app, "5yr", 3, self.syn.dem, self.syn.cache)
            self.assertTrue(info.get("cached"), how + ": the rebuilt cache is valid again")

    def test_damaged_manning_cache_is_rebuilt(self):
        path = os.path.join(self.syn.cache, "manning_classes.npz")
        self.assertTrue(os.path.isfile(path))
        for how in ("half", "empty", "garbage"):
            self.damage(path, how)
            n = bf.load_manning(self.syn.app, self.syn.dem, self.syn.cache)
            np.testing.assert_array_equal(n, self.syn.manning)
            with np.load(path) as z:                          # and the file is whole again
                self.assertEqual(z["cls"].shape, self.syn.dem["shape"])

    def test_cache_write_is_atomic_and_leaves_no_temp_file(self):
        bf.tier_hour_cached(self.syn.app, "5yr", 3, self.syn.dem, self.syn.cache)
        path = self.tier_cache()
        before = read_bytes(path)
        real = np.savez_compressed

        def broken(f, **arrays):
            f.write(b"PK partial")
            raise OSError("disk full")
        np.savez_compressed = broken
        try:
            bf._save_cache(path, tier=np.zeros(3), key=np.array("k"))
        finally:
            np.savez_compressed = real
        self.assertEqual(read_bytes(path), before)
        self.assertEqual([n for n in os.listdir(self.syn.cache) if n.endswith(".tmp")], [])

    def test_a_reader_in_the_middle_of_a_cache_write_still_sees_the_old_file(self):
        tier, _ = bf.tier_hour_cached(self.syn.app, "5yr", 3, self.syn.dem, self.syn.cache)
        path = self.tier_cache()
        with np.load(path) as z:
            key, info = str(z["key"]), str(z["info"])
        real, seen = np.savez_compressed, {}

        def half_way(f, **arrays):
            buf = io.BytesIO()
            real(buf, **arrays)
            data = buf.getvalue()
            f.write(data[:len(data) // 2])
            f.flush()
            try:                                              # another process loads the cache while this write is half done
                seen["hit"] = bf._load_cache(path, key, "tier")
            except BaseException as e:                        # noqa: BLE001
                seen["error"] = repr(e)
            f.write(data[len(data) // 2:])
        np.savez_compressed = half_way
        try:
            bf._save_cache(path, tier=tier, key=np.array(key), info=np.array(info))
        finally:
            np.savez_compressed = real
        self.assertNotIn("error", seen)
        np.testing.assert_array_equal(seen["hit"][0], tier)    # the previous file, whole
        np.testing.assert_array_equal(bf._load_cache(path, key, "tier")[0], tier)    # and afterwards the new one

    def test_concurrent_builders_on_an_empty_cache_do_not_crash(self):
        import threading
        shutil.rmtree(self.syn.cache)
        errors, results = [], []

        def work():
            try:
                for _ in range(3):
                    results.append(int(bf.tier_hour_cached(self.syn.app, "5yr", 3, self.syn.dem, self.syn.cache)[0].sum()))
                    bf.load_manning(self.syn.app, self.syn.dem, self.syn.cache)
            except Exception as e:                            # noqa: BLE001
                errors.append(repr(e))
        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), 1)


class TestBuildIsAtomic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.syn = SyntheticApp(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def snapshot(self, ignore_built=False):
        """name -> bytes of every file of the storm folder; ignore_built drops meta.json's rebuild time (1 s resolution)."""
        snap = {n: read_bytes(os.path.join(self.syn.out_dir, n)) for n in sorted(os.listdir(self.syn.out_dir))}
        if ignore_built and "meta.json" in snap:
            m = json.loads(snap["meta.json"])
            m.pop("built", None)
            snap["meta.json"] = json.dumps(m, sort_keys=True).encode()
        return snap

    def interrupt_in_hour(self, hour, **kw):
        real = bf.velocity_fields
        calls = []

        def boom(*a, **k):
            calls.append(1)
            if len(calls) == hour:
                raise KeyboardInterrupt()
            return real(*a, **k)
        bf.velocity_fields = boom
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.syn.build(**kw)
        finally:
            bf.velocity_fields = real

    def leftovers(self):
        return [n for n in os.listdir(self.tmp) if n.startswith(".out.")]

    def test_interrupted_forced_rebuild_keeps_the_working_storm(self):
        self.syn.build(hours=(2, 3))
        before = self.snapshot()
        self.interrupt_in_hour(2, hours=(2, 3), force=True, cell_m=8.0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(fg.validate_storm_dir(self.syn.out_dir), [])
        self.assertEqual(self.leftovers(), [])

    def test_interrupted_first_build_leaves_nothing_and_a_plain_rerun_works(self):
        self.interrupt_in_hour(2, hours=(2, 3))
        self.assertFalse(os.path.exists(self.syn.out_dir))
        self.assertEqual(self.leftovers(), [])
        res, _ = self.syn.build(hours=(2, 3))                 # no --force needed
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["hours"], [2, 3])

    def test_hour_files_without_meta_say_so(self):
        self.syn.build(hours=(2, 3))
        os.remove(os.path.join(self.syn.out_dir, "meta.json"))
        with self.assertRaises(SystemExit) as cm:
            self.syn.build(hours=(2, 3))
        self.assertIn("meta.json", str(cm.exception))
        self.assertNotIn("another grid", str(cm.exception))

    def test_failed_validation_does_not_replace_the_old_output(self):
        self.syn.build(hours=(2, 3))
        before = self.snapshot()
        real = fg.validate_storm_dir
        fg.validate_storm_dir = lambda p: ["synthetic problem"]
        try:
            res, _ = self.syn.build(hours=(2, 3), force=True)
        finally:
            fg.validate_storm_dir = real
        self.assertEqual(res["problems"], ["synthetic problem"])
        self.assertEqual(self.snapshot(), before)

    def test_subset_rerun_keeps_the_grid_of_the_full_build(self):
        self.syn.build(hours=(2, 3))
        before = self.snapshot(ignore_built=True)
        res, log = self.syn.build(hours=[2])                   # alone, hour 2 would plan a much smaller grid
        self.assertEqual(res["skipped"], [2])
        self.assertEqual(res["hours"], [2, 3])
        self.assertEqual(self.snapshot(ignore_built=True), before)
        res, _ = self.syn.build(hours=[2], force=True)
        self.assertEqual(res["hours"], [2, 3])
        self.assertEqual(self.snapshot(ignore_built=True), before)

    def test_subset_rerun_keeps_the_dry_hours_of_the_full_build(self):
        self.syn.build(hours=(1, 2, 3))
        self.assertEqual(load_json(os.path.join(self.syn.out_dir, "meta.json"))["tiers_build"]["dry_hours"], [1])
        res, _ = self.syn.build(hours=[3])
        self.assertEqual(res["hours"], [2, 3])
        tb = load_json(os.path.join(self.syn.out_dir, "meta.json"))["tiers_build"]
        self.assertEqual(tb["dry_hours"], [1])
        self.assertEqual(sorted(tb["sources"]), ["1", "2", "3"])
        res, _ = self.syn.build(hours=[1, 2, 3])
        self.assertEqual(load_json(os.path.join(self.syn.out_dir, "meta.json"))["tiers_build"]["dry_hours"], [1])

    def test_two_builders_of_the_same_storm_do_not_disturb_each_other(self):
        import threading
        errors, results = [], []

        def work():
            try:
                results.append(bf.build_storm(self.syn.app, "5yr", [2, 3], self.syn.dem, self.syn.manning, force=True,
                                              cache_dir=self.syn.cache, out_root=self.syn.out_root, log=lambda m: None))
            except BaseException as e:                         # noqa: BLE001
                errors.append(repr(e))
        threads = [threading.Thread(target=work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual([r["problems"] for r in results], [[], [], []])
        self.assertEqual(fg.validate_storm_dir(self.syn.out_dir), [])
        self.assertEqual(self.leftovers(), [])

    def plant_rivals(self, limit):
        """
        Patch fg._replace_retry so that after each of OUR first renames (out_dir moved aside) another builder gets its second
        rename in first: a copy of the folder just moved, marked with rival.txt, appears at out_dir. At most `limit` times.
        Returns (planted list, undo function).
        """
        real = fg._replace_retry
        planted = []

        def interleave(src, dst):
            real(src, dst)
            if ".old" in os.path.basename(dst) and len(planted) < limit:
                rival = os.path.join(self.tmp, "rival%d" % len(planted))
                shutil.copytree(dst, rival)
                with open(os.path.join(rival, "rival.txt"), "w") as f:
                    f.write("x")
                planted.append(rival)
                real(rival, self.syn.out_dir)
        fg._replace_retry = interleave
        return planted, lambda: setattr(fg, "_replace_retry", real)

    def test_a_builder_that_swaps_in_between_our_two_renames_is_replaced_not_fatal(self):
        # The interleaving the thread test above only hits by chance: our first rename empties out_dir, the other builder
        # gets ITS second rename in, so ours finds a finished folder at out_dir (twice here, the retry meets another one).
        self.syn.build(hours=(2, 3))
        planted, undo = self.plant_rivals(2)
        try:
            res, _ = self.syn.build(hours=(2, 3), force=True)
        finally:
            undo()
        self.assertEqual(len(planted), 2)
        self.assertEqual(res["problems"], [])
        self.assertEqual(fg.validate_storm_dir(self.syn.out_dir), [])
        self.assertFalse(os.path.exists(os.path.join(self.syn.out_dir, "rival.txt")))     # the last finished build wins
        self.assertEqual(self.leftovers(), [])

    def test_swap_gives_up_with_a_finished_build_in_place_when_rivals_never_stop(self):
        self.syn.build(hours=(2, 3))
        planted, undo = self.plant_rivals(10 ** 6)
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.syn.build(hours=(2, 3), force=True)
        finally:
            undo()
        self.assertEqual(len(planted), bf.SWAP_TRIES)
        self.assertIn(".building", str(cm.exception))
        self.assertEqual(fg.validate_storm_dir(self.syn.out_dir), [])                    # a rival's whole build stands there
        self.assertEqual([n for n in self.leftovers() if ".old" in n], [])                # no moved-aside folder is left behind

    def test_work_folders_of_killed_builds_are_cleaned_up_when_old(self):
        base = os.path.dirname(self.syn.out_root)
        old_dir = os.path.join(base, ".out.5yr.111-deadbeef.building")
        fresh_dir = os.path.join(base, ".out.5yr.222-cafef00d.building")
        other = os.path.join(base, ".out.10yr.333-deadbeef.building")        # another storm: not ours to touch
        for d in (old_dir, fresh_dir, other):
            os.makedirs(d)
            write_compact(os.path.join(d, "x.json"), [])
        long_ago = os.stat(old_dir).st_mtime - 2 * bf.WORK_STALE_S
        for d in (old_dir, other):
            os.utime(d, (long_ago, long_ago))
        self.syn.build(hours=(2, 3))
        self.assertFalse(os.path.exists(old_dir))
        self.assertTrue(os.path.exists(fresh_dir))
        self.assertTrue(os.path.exists(other))

    def test_swap_failure_leaves_the_old_output_and_names_the_new_folder(self):
        self.syn.build(hours=(2, 3))
        before = self.snapshot()
        real = fg._replace_retry

        def flaky(src, dst):
            if os.path.basename(src).endswith(".building"):    # the second rename, the new folder into place
                raise PermissionError("held open by a reader")
            return real(src, dst)
        fg._replace_retry = flaky
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.syn.build(hours=(2, 3), force=True)
        finally:
            fg._replace_retry = real
        self.assertIn(".building", str(cm.exception))
        self.assertEqual(self.snapshot(), before)              # the previous output was put back


class TestResumeSeesChangedInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.syn = SyntheticApp(self.tmp)
        self.syn.build(hours=(2, 3))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def h(self, n):
        return read_bytes(os.path.join(self.syn.out_dir, fg.hour_filename(n)))

    def test_changed_tier_file_is_rebuilt_and_the_others_are_skipped(self):
        h2, h3 = self.h(2), self.h(3)
        other = cell_box(72, 92, 102, 122)                     # a bigger patch than hour 2, inside the same lake footprint
        write_compact(self.syn.hour_path(2), [poly_feature(*other, "moderate")])
        res, log = self.syn.build(hours=(2, 3))
        self.assertEqual(res["skipped"], [3])
        self.assertNotIn("h02 skipped", log)
        self.assertNotEqual(self.h(2), h2)
        self.assertEqual(self.h(3), h3)
        meta = load_json(os.path.join(self.syn.out_dir, "meta.json"))
        self.assertEqual(sorted(meta["tiers_build"]["sources"]), ["2", "3"])
        res, _ = self.syn.build(hours=(2, 3))                  # and now it is up to date again
        self.assertEqual(res["skipped"], [2, 3])

    def test_a_build_with_the_old_capped_halo_is_not_resumed(self):
        # meta.json of the first version: no halo setting, method version 1. Its hours carry the rim jets, so a plain
        # rerun must refuse (and --force must rebuild every hour) instead of skipping them as "already written".
        path = os.path.join(self.syn.out_dir, "meta.json")
        meta = load_json(path)
        params = meta["tiers_build"]["params"]
        self.assertEqual(params["halo_max_m"], bf.HALO_MAX_M)
        del params["halo_max_m"]
        params["v"] = 1
        dump_json(path, meta)
        with self.assertRaises(SystemExit):
            self.syn.build(hours=(2, 3))
        res, _ = self.syn.build(hours=(2, 3), force=True)
        self.assertEqual(res["skipped"], [])
        self.assertEqual(load_json(path)["tiers_build"]["params"]["halo_max_m"], bf.HALO_MAX_M)

    def test_changed_building_or_terrain_file_needs_force(self):
        path = os.path.join(self.syn.app, "layers", "buildings.geojson")
        dump_json(path, {"type": "FeatureCollection", "features": [], "note": "edited"})
        with self.assertRaises(SystemExit):
            self.syn.build(hours=(2, 3))
        res, _ = self.syn.build(hours=(2, 3), force=True)
        self.assertEqual(res["skipped"], [])
        self.assertEqual(res["problems"], [])
        res, _ = self.syn.build(hours=(2, 3))
        self.assertEqual(res["skipped"], [2, 3])
        dem_path = os.path.join(self.syn.app, "dem_bg.tif")
        st = os.stat(dem_path)
        os.utime(dem_path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        with self.assertRaises(SystemExit):
            self.syn.build(hours=(2, 3))


if __name__ == "__main__":
    unittest.main()
