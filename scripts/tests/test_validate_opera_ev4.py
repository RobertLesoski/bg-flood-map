"""Hand-checked tests for scripts/validate_opera_ev4.py (metrics, projections, sampling, sweep).

Run:  python -m pytest scripts/tests/test_validate_opera_ev4.py -q
  or: python -m unittest scripts/tests/test_validate_opera_ev4.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import validate_opera_ev4 as v  # noqa: E402


class Scores(unittest.TestCase):
    def test_hand_example(self):
        # 2 hits, 1 miss, 1 false alarm -> POD 2/3, FAR 1/3, CSI 2/4
        pod, far, csi = v.scores(2, 1, 1)
        self.assertAlmostEqual(pod, 2 / 3)
        self.assertAlmostEqual(far, 1 / 3)
        self.assertAlmostEqual(csi, 0.5)

    def test_zero_denominators_are_nan(self):
        pod, far, csi = v.scores(0, 0, 0)
        self.assertTrue(all(math.isnan(x) for x in (pod, far, csi)))

    def test_contingency_counts(self):
        obs = np.array([True, True, True, False, False, False, False])
        mod = np.array([True, True, False, True, False, False, True])
        valid = np.array([True] * 6 + [False])          # last cell excluded
        self.assertEqual(v.contingency(obs, mod, valid), (2, 1, 1, 2))

    def test_sweep_threshold_behaviour(self):
        # depths 0, 0.2, 0.6 m; observed wet only at the 0.6 m cell
        depth = np.array([0.0, 0.2, 0.6])
        obs = np.array([False, False, True])
        valid = np.ones(3, bool)
        rows = v.sweep(obs, depth, valid, cell_km2=1.0, thresholds=(0.1, 0.5))
        self.assertEqual((rows[0]["hits"], rows[0]["false_alarms"]), (1, 1))   # >=0.1 flags 0.2 and 0.6
        self.assertEqual((rows[1]["hits"], rows[1]["false_alarms"]), (1, 0))   # >=0.5 flags only 0.6
        self.assertAlmostEqual(rows[1]["csi"], 1.0)
        self.assertAlmostEqual(rows[0]["csi"], 0.5)


class Projections(unittest.TestCase):
    def test_mercator_known_values(self):
        x, y = v.merc_forward(0.0, 90.0)
        self.assertAlmostEqual(float(x), 6378137.0 * math.pi / 2, places=3)   # 10018754.17
        x, y = v.merc_forward(0.0, 0.0)
        self.assertAlmostEqual(float(x), 0.0)
        self.assertAlmostEqual(float(y), 0.0)

    def test_utm_central_meridian_origin(self):
        x, y = v.utm_forward(np.array([0.0]), np.array([-87.0]))
        self.assertAlmostEqual(float(x[0]), 500000.0, places=3)
        self.assertAlmostEqual(float(y[0]), 0.0, places=3)

    def test_utm_round_trip_warren_county(self):
        lat = np.array([36.8, 37.0, 37.1])
        lon = np.array([-86.7, -86.4, -86.0])
        x, y = v.utm_forward(lat, lon)
        la2, lo2 = v.utm_inverse(x, y)
        np.testing.assert_allclose(la2, lat, atol=1e-7)
        np.testing.assert_allclose(lo2, lon, atol=1e-7)

    def test_utm_matches_opera_tile_origin(self):
        # The OPERA tile T16SEF has its corner at (499980 E, 4100040 N); its lon/lat must sit near the
        # western edge of Warren County (about lon -86.9, lat 37.1).
        lat, lon = v.utm_inverse(np.array([499980.0]), np.array([4100040.0]))
        self.assertTrue(-87.2 < lon[0] < -86.8, lon)
        self.assertTrue(36.9 < lat[0] < 37.2, lat)


class Sampling(unittest.TestCase):
    def test_nearest_cell_and_outside(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)     # rows 0..2, cols 0..3
        x = np.array([0.5, 3.5, 10.0, -1.0])
        y = np.array([2.5, 0.5, 2.0, 2.0])
        vals, inside = v.sample_nearest(arr, x0=0.0, y0=3.0, sx=1.0, sy=1.0, x=x, y=y)
        # top-left corner (0, 3): point (0.5, 2.5) is row 0, col 0; point (3.5, 0.5) is row 2, col 3
        self.assertEqual(vals[0], 0.0)
        self.assertEqual(vals[1], 11.0)
        self.assertFalse(inside[2])
        self.assertFalse(inside[3])

    def test_county_mask_square(self):
        poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        mask = v.county_mask_in_tile(poly, x0=0.0, y0=4.0, sx=1.0, sy=1.0, shape=(4, 4))
        self.assertEqual(int(mask.sum()), 16)
        mask2 = v.county_mask_in_tile([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                                      x0=0.0, y0=4.0, sx=1.0, sy=1.0, shape=(4, 4))
        self.assertEqual(int(mask2.sum()), 4)


class Units(unittest.TestCase):
    def test_depth_scale_and_nodata(self):
        raw = np.array([[0, 10, 65535, 250]], dtype=np.uint16)
        depth = raw.astype(float) * 0.01
        depth[raw == 65535] = np.nan
        np.testing.assert_allclose(depth[0, :2], [0.0, 0.10])
        self.assertTrue(math.isnan(depth[0, 2]))
        self.assertAlmostEqual(depth[0, 3], 2.5)


class Blocked(unittest.TestCase):
    def test_missing_model_writes_blocked_status(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as td:
            code = v.main(["--opera-dir", td, "--boundary", td, "--model", os.path.join(td, "nope.tif"),
                           "--out", td])
            self.assertEqual(code, 2)
            with open(os.path.join(td, "opera_vs_model_BLOCKED.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["status"], "blocked")


if __name__ == "__main__":
    unittest.main()


class SyntheticOverlapOnRealTiles(unittest.TestCase):
    """End to end on the real OPERA tiles with a model whose wet area is a known UTM rectangle.
    Skipped if the tiles or the county boundary are not present."""

    def test_rectangle_model_recovers_exact_overlap(self):
        import glob
        tiles = sorted(glob.glob(os.path.join(HERE, "..", "..", "verification", "opera", "*.tif")))
        bnd = os.path.join(HERE, "..", "..", ".claude", "worktrees", "agent-ab76444f0e4e07897",
                           "layers", "county_boundary_21227.geojson")
        if not tiles or not os.path.exists(bnd):
            self.skipTest("OPERA tiles or boundary file not present")
        poly = v.boundary_ring_lonlat(bnd)
        lon = np.array([p[0] for p in poly]); lat = np.array([p[1] for p in poly])
        X, Y = v.utm_forward(lat, lon)
        ex, ny, cls, _ = v.collect_opera_cells(tiles, list(zip(X.tolist(), Y.tolist())))
        RX0, RX1, RY0, RY1 = 520000.0, 560000.0, 4075000.0, 4095000.0
        mx0, my0 = v.merc_forward(37.25, -87.0)
        mx1, my1 = v.merc_forward(36.70, -85.9)
        s = 50.0
        W = int((mx1 - mx0) / s) + 1
        H = int((my0 - my1) / s) + 1
        rr, cc = np.mgrid[0:H, 0:W]
        lon_c = np.degrees((mx0 + (cc + 0.5) * s) / v.R)
        lat_c = np.degrees(2 * np.arctan(np.exp((my0 - (rr + 0.5) * s) / v.R)) - np.pi / 2)
        UX, UY = v.utm_forward(lat_c, lon_c)
        arr = np.where((UX >= RX0) & (UX <= RX1) & (UY >= RY0) & (UY <= RY1), 50, 0).astype(np.uint16)
        model = {"array": arr, "x0": mx0, "y0": my0, "sx": s, "sy": s, "epsg": 3857,
                 "path": "synthetic", "nodata": None}
        depth = v.model_depth_m(model, ex, ny, 0.01, 65535)
        ok = (cls != 255) & (cls != 251) & np.isfinite(depth)
        obs = cls == 1
        r = v.sweep(obs, depth, ok, 0.0009, thresholds=(0.1,))[0]
        inr = (ex >= RX0) & (ex <= RX1) & (ny >= RY0) & (ny <= RY1)
        hits = int((obs & inr & ok).sum())
        misses = int((obs & ~inr & ok).sum())
        fas = int((~obs & inr & ok).sum())
        self.assertEqual(r["hits"], hits)
        self.assertEqual(r["misses"], misses)
        # false alarms differ only by 50 m grid-edge sampling (measured: 0.08%)
        self.assertLess(abs(r["false_alarms"] - fas) / fas, 0.005)
