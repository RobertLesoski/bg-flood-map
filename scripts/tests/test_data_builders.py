"""
Tests for flowquery / build_address_index / build_infrastructure / make_cogs / fetch_svi on small synthetic data.

  python -m pytest scripts/tests/test_data_builders.py -q
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import flowgrid as fg  # noqa: E402
import flowquery as fq  # noqa: E402
import build_address_index as bai  # noqa: E402
import build_infrastructure as bi  # noqa: E402
import geodata as gd  # noqa: E402
import fetch_svi  # noqa: E402

W = H = 40
CELL = 10.0
X0, Y1 = -9600000.0, 4430000.0           # near Bowling Green in web mercator


def grid():
    return fq.Grid(W, H, CELL, X0, Y1 - H * CELL, X0 + W * CELL, Y1)


def lnglat(col, row):
    """lng/lat of the centre of pixel (col,row)."""
    lng, lat = fq.merc_to_lnglat(X0 + (col + 0.5) * CELL, Y1 - (row + 0.5) * CELL)
    return float(lng), float(lat)


def box_ll(c0, r0, c1, r1):
    a, b = lnglat(c0, r0)
    c, d = lnglat(c1, r1)
    return Polygon([(a, b), (c, b), (c, d), (a, d)])


def make_storm(folder, name, depth_by_hour):
    """depth_by_hour {hour: float32[H,W] metres} -> a storm folder in the flow format."""
    os.makedirs(folder, exist_ok=True)
    g = grid()
    mx = np.zeros((H, W))
    for h, d in depth_by_hour.items():
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 1:3] = 128
        rgba[..., 3] = 255
        rgba[..., 0] = fg.encode_depth(d)
        fg.write_rgba_gz(os.path.join(folder, "h%02d.rgba.gz" % h), rgba)
        mx = np.maximum(mx, d)
    fp = np.zeros((H, W, 4), np.uint8)
    fp[..., 0] = fg.encode_depth(mx)
    fp[..., 3] = 255
    fg.write_rgba_gz(os.path.join(folder, "flowpaths.rgba.gz"), fp)
    meta = {"format": "bgflow", "version": 1, "storm": name, "label": name + " storm", "source": "rasters", "width": W, "height": H,
            "cell_m": CELL, "bounds_merc": [g.xmin, g.ymin, g.xmax, g.ymax], "hours": sorted(depth_by_hour), "storm_depth_in": 3.0}
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f)


class FlowQuery(unittest.TestCase):
    def test_all_touched_and_max(self):
        g = grid()
        bld = box_ll(4.2, 4.2, 6.8, 5.8)               # touches columns 4..6, rows 4..5 (using centres as corners)
        idx = fq.building_pixels([bld, box_ll(30.1, 30.1, 30.4, 30.4)], g)
        px0 = set(idx.pixels[idx.offsets[0]:idx.offsets[1]].tolist())
        self.assertTrue({4 * W + 4, 5 * W + 6} <= px0)
        self.assertEqual(idx.offsets[2] - idx.offsets[1], 1)                      # tiny footprint still owns its pixel
        d = np.zeros((H, W)); d[5, 5] = 0.5; d[4, 4] = 1.0
        R = fg.encode_depth(d).astype(np.uint8).reshape(-1)
        A = np.full(H * W, 255, np.uint8)
        code = fq.max_code(R, A, idx)
        self.assertAlmostEqual(float(fq.code_depth(code[0])), 1.0, delta=0.02)
        self.assertEqual(int(code[1]), 0)
        A[4 * W + 4] = 0                                                          # outside the model domain: ignored
        self.assertAlmostEqual(float(fq.code_depth(fq.max_code(R, A, idx)[0])), 0.5, delta=0.02)

    def test_tiers(self):
        self.assertEqual(fq.tier_of([0.05, 0.1, 0.29, 0.3, 0.7, 1.2, 3]).tolist(), [0, 1, 1, 2, 3, 4, 4])

    def test_roads(self):
        g = grid()
        a = lnglat(2, 20)
        b = lnglat(37, 20)                                      # about 350 mercator m = 280 m ground
        rs = fq.road_samples([[list(a), list(b)]], g)
        L = rs.length_m[0]
        self.assertAlmostEqual(rs.weight_m.sum(), L, places=3)
        self.assertEqual(rs.pixel.size, int(np.ceil(L / 10.0)))
        d = np.zeros((H, W)); d[20, 2:12] = 0.5
        R = fg.encode_depth(d).astype(np.uint8).reshape(-1)
        wet, fl, mx = fq.road_stats(R, np.full(H * W, 255, np.uint8), rs)
        self.assertTrue(20 < wet[0] < 40 and abs(wet[0] - fl[0]) < 1e-9)
        self.assertAlmostEqual(float(fq.code_depth(mx[0])), 0.5, delta=0.02)

    def test_storm_discovery(self):
        with tempfile.TemporaryDirectory() as t:
            d = np.zeros((H, W)); d[10:20, 10:20] = 0.4
            make_storm(os.path.join(t, "25yr"), "25yr", {3: d * 0.5, 4: d})
            make_storm(os.path.join(t, "bad"), "bad", {3: d})
            os.remove(os.path.join(t, "bad", "h03.rgba.gz"))
            ss = fq.load_storms(t, log=None)
            self.assertEqual([s.name for s in ss], ["25yr"])
            R, A = ss[0].read_hour(4)
            self.assertEqual(R.size, H * W)


class AddressText(unittest.TestCase):
    def test_parse_normalize(self):
        self.assertEqual(bai.parse_address("592 INTERSTATE DRIVE"), ("592", "INTERSTATE DRIVE"))
        self.assertEqual(bai.parse_address("401,403,405,1101, 1103 PEARL STREET, BEAUTY AVENUE"), ("401", "PEARL STREET"))
        self.assertEqual(bai.parse_address("845 1/2 EAST 10TH AVENUE APT 3"), ("845", "EAST 10TH AVENUE"))
        self.assertEqual(bai.parse_address("near Main Street"), ("", ""))
        self.assertEqual(bai.normalize_address("568 West Villa Drive"), "568 W VILLA DR")
        self.assertEqual(bai.display_street("WEST VILLA DRIVE"), "West Villa Drive")
        self.assertEqual(bai.display_street("10TH AVENUE"), "10th Avenue")


class AddressIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        d1 = np.zeros((H, W)); d1[10:14, 10:14] = 0.2
        d2 = np.zeros((H, W)); d2[10:14, 10:14] = 0.8
        make_storm(os.path.join(t, "flow", "25yr"), "25yr", {3: d1, 4: d2, 5: d2})
        self.storms = fq.load_storms(os.path.join(t, "flow"), log=None)
        b_wet = box_ll(10.3, 10.3, 11.7, 11.7)
        b_dry = box_ll(25.3, 25.3, 26.7, 26.7)
        b_garage = box_ll(12.3, 12.3, 12.7, 12.7)
        geoms = np.array([b_wet, b_dry, b_garage], dtype=object)
        self.bset = gd.BuildingSet([0, 1, 2], geoms, ["100 MAIN STREET", "", ""], ["R", "R", "R"], [{}, {}, {}], "city")
        pg = np.array([box_ll(9, 9, 15, 15), box_ll(24, 24, 28, 28), box_ll(30, 30, 33, 33)], dtype=object)
        ps = gd.ParcelSet(pg, ["P1", "P2", "P3"], ["100 MAIN STREET", "7 ELM ROAD", "55 OAK LANE"], ["R", "C", "R"],
                          np.array([150000.0, 90000.0, 0.0]), np.zeros(3))
        ps.tree = shapely.STRtree(ps.geoms)
        self.parcels = ps
        w, s = fq.merc_to_lnglat(X0, Y1 - H * CELL)
        e, n = fq.merc_to_lnglat(X0 + W * CELL, Y1)
        self.dom = gd.Domain("city", "Test", os.path.join(t, "flow"), t, (float(w), float(s), float(e), float(n)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_and_flood(self):
        rec = bai.make_records(self.bset, self.parcels, self.dom, log=None)
        rec.sort()
        # P1 owns the wet building + the garage (one record), P2's building has no own address -> parcel address, P3 is a lot
        self.assertEqual(len(rec), 3)
        self.assertEqual(sorted(rec.cols["kind"]), [0, 0, 1])
        self.assertEqual(sorted(rec.cols["nb"]), [0, 1, 2])
        fl = bai.compute_flood(rec, self.bset, self.storms, log=None)["25yr"]
        i_main = rec.cols["num"].index("100")
        self.assertEqual(fl["idx"].tolist(), [i_main])                                  # only the wet record has an entry
        self.assertTrue(75 <= fl["cm"][0] <= 85)
        self.assertEqual((int(fl["tier"][0]), int(fl["first"][0]), int(fl["hours"][0])), (3, 3, 3))   # deep, flooded (>= 0.1 m) from hour 3, 3 hours

    def test_write_search_and_shards(self):
        rec = bai.make_records(self.bset, self.parcels, self.dom, log=None)
        rec.sort()
        flood = bai.compute_flood(rec, self.bset, self.storms, log=None)
        out = os.path.join(self.tmp.name, "address_index.json.gz")
        info = bai.write_index(self.dom, rec, flood, self.storms, out, log=None)
        self.assertFalse(info["sharded"])
        idx = bai.load_index(out)
        r = bai.search(idx, "100 main s")[0]
        self.assertEqual((r["address"], r["parcel"], r["value"]), ("100 Main Street", "P1", 150000))
        self.assertEqual(bai.flood_of(idx, "25yr")[r["i"]][1], 3)
        self.assertEqual(bai.search(idx, "elm")[0]["address"], "7 Elm Road")
        self.assertEqual(bai.search(idx, "55")[0]["address"], "55 Oak Lane")
        self.assertAlmostEqual(r["lng"], self.bset.geoms[0].representative_point().x, places=4)
        # forced sharding: manifest + one file per street initial, the reference loader merges them back identically
        out2 = os.path.join(self.tmp.name, "sub", "address_index.json.gz")
        info2 = bai.write_index(self.dom, rec, flood, self.storms, out2, max_single_mb=0.0, log=None)
        self.assertTrue(info2["sharded"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmp.name, "sub", "address", "M.json.gz")))
        idx2 = bai.load_index(out2)
        self.assertEqual(idx2["num"], idx["num"])
        self.assertEqual(idx2["flood"]["25yr"], idx["flood"]["25yr"])


class Infrastructure(unittest.TestCase):
    def test_classify_dedupe_merge(self):
        from osm_pbf import classify
        self.assertEqual(classify({"amenity": "hospital"}), "hospital")
        self.assertEqual(classify({"amenity": "shelter", "shelter_type": "picnic_shelter"}), None)
        self.assertEqual(classify({"emergency": "assembly_point"}), "assembly_point")
        a = {"osm_id": "w1", "type": "hospital", "name": "General", "lng": -86.4, "lat": 37.0, "ring": [[-86.4, 37], [-86.399, 37], [-86.399, 37.001], [-86.4, 37.001], [-86.4, 37]]}
        b = dict(a, osm_id="w2", ring=None, lng=-86.4005)
        c = dict(a, osm_id="w3", lng=-86.3, name="General")
        g = bi.dedupe_pois([a, b, c])
        self.assertEqual(len(g), 2)
        self.assertEqual(g[0]["main"]["osm_id"], "w1")
        groups, added, skipped = bi.merge_extra_points(g, [{"name": "X", "type": "hospital", "lng": -86.4001, "lat": 37.0, "source": "hifld"},
                                                           {"name": "Fire 1", "type": "fire_station", "lng": -86.5, "lat": 37.0, "source": "hifld"}])
        self.assertEqual((added, skipped), (1, 1))

    def test_depth_features(self):
        with tempfile.TemporaryDirectory() as t:
            d = np.zeros((H, W)); d[10:12, 10:12] = 0.35
            make_storm(os.path.join(t, "25yr"), "25yr", {4: d})
            make_storm(os.path.join(t, "2yr"), "2yr", {4: d * 0})
            ss = fq.load_storms(t, log=None)
            lng, lat = lnglat(10, 10)
            groups = [{"main": {"osm_id": "n1", "type": "school", "name": "S", "lng": lng, "lat": lat, "ring": None}, "members": []}]
            groups[0]["members"] = [groups[0]["main"]]
            dep = bi.depths_for_groups(groups, ss, log=None)
            feats = bi.build_features(groups, dep, ss)
            pr = feats[0]["properties"]
            self.assertTrue(30 <= pr["d_25yr"] <= 38 and pr["d_2yr"] == 0)
            self.assertEqual((pr["flooded_in"], pr["first_flooded"]), ("25yr", "25yr"))


class CogsAndSvi(unittest.TestCase):
    def test_cog(self):
        import rasterio
        from rasterio.transform import Affine
        import make_cogs
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "depth_003600s.tif")
            a = np.zeros((300, 300), np.float32); a[100:150, 100:150] = 1.234; a[:, :5] = -9999
            with rasterio.open(src, "w", driver="GTiff", width=300, height=300, count=1, dtype="float32", crs="EPSG:32616",
                               transform=Affine(5, 0, 560000, 0, -5, 4090000), nodata=-9999.0) as d:
                d.write(a, 1)
            out = os.path.join(t, "o.tif")
            make_cogs.convert(src, out, res_m=6.25)
            self.assertTrue(make_cogs.verify(out))
            with rasterio.open(out) as d:
                v = d.read(1)
                self.assertTrue(120 <= v[v < 65535].max() <= 124)                  # 1.234 m -> about 123 cm
                self.assertTrue((v == 65535).any())

    def test_svi_join(self):
        csv_text = "FIPS,LOCATION,E_TOTPOP,RPL_THEMES,RPL_THEME1,RPL_THEME2,RPL_THEME3,RPL_THEME4\n21227000100,T1,3000,0.8,0.7,0.6,0.5,-999\n21001000100,T2,10,0.1,0.1,0.1,0.1,0.1\n"
        svi = fetch_svi.read_svi(csv_text)
        self.assertEqual(list(svi), ["21227000100"])
        self.assertIsNone(svi["21227000100"]["RPL_THEME4"])
        fc = fetch_svi.join({"features": [{"type": "Feature", "properties": {"GEOID": "21227000100"}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]}, svi)
        self.assertEqual(len(fc["features"]), 1)


if __name__ == "__main__":
    unittest.main()
