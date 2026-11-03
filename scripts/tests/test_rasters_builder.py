"""
Tests for scripts/build_flow_from_rasters.py on small synthetic rasters (tmp dirs, a few hundred cells a side).

  python -m pytest scripts/tests/test_rasters_builder.py -q
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import flowgrid as fg  # noqa: E402
import build_flow_from_rasters as br  # noqa: E402

import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

H, W, DX = 200, 240, 5.0
E0, N1 = 564000.0, 4091000.0                      # near Bowling Green, UTM 16N
TR = Affine(DX, 0, E0, 0, -DX, N1)


def write_tif(path, arr, tr=TR, nodata=-9999.0):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                       crs="EPSG:32616", transform=tr, nodata=nodata, tiled=True, blockxsize=64, blockysize=64,
                       compress="lzw") as ds:
        ds.write(arr.astype(np.float32), 1)


def depth_field(amp):
    y, x = np.mgrid[0:H, 0:W]
    d = amp * np.exp(-(((x - 120) / 40.0) ** 2 + ((y - 90) / 30.0) ** 2))
    d[d < 0.003] = 0.0
    d[:, :4] = -9999.0                              # nodata strip along the west edge (outside the model domain)
    return d.astype(np.float32)


def terrain(units=1.0):
    x = np.arange(W)[None, :] * DX
    y = np.arange(H)[:, None] * DX
    return ((200.0 - 0.01 * x + 0.001 * y) / units + 0 * y).astype(np.float32)   # falls toward the east


def make_run(folder, hours=(1, 2, 3), amps=(0.2, 0.8, 1.5), dem=True):
    os.makedirs(folder, exist_ok=True)
    write_tif(os.path.join(folder, "depth_000000s.tif"), np.where(depth_field(1) < 0, -9999.0, 0.0))
    for h, a in zip(hours, amps):
        write_tif(os.path.join(folder, "depth_%06ds.tif" % (h * 3600)), depth_field(a))
    mx = np.maximum.reduce([depth_field(a) for a in amps])
    write_tif(os.path.join(folder, "max_depth.tif"), mx)
    with open(os.path.join(folder, "run_summary.json"), "w") as f:
        json.dump({"storm": "t", "total_rain_mm": 100.0}, f)
    dpath = None
    if dem:
        dpath = os.path.join(folder, "..", "dem_test_2m.tif")
        write_tif(dpath, terrain())
    return dpath


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bgras_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.run_dir = os.path.join(self.tmp, "run")
        self.dem = make_run(self.run_dir)
        self.dem = os.path.join(self.tmp, "dem_test_2m.tif")
        write_tif(self.dem, terrain())
        self.units = "metres"

    def build(self, name="s1", **kw):
        out = os.path.join(self.tmp, "out", name)
        kw.setdefault("dem", self.dem)
        kw.setdefault("dem_units_arg", "metres")
        kw.setdefault("cell_m", 25.0)
        res = br.build_storm(self.run_dir, out, name, log=lambda m: None, **kw)
        return out, res


class TestHelpers(unittest.TestCase):
    def test_hours_and_files(self):
        self.assertEqual(br.parse_hours("1-3,5"), [1, 2, 3, 5])
        with self.assertRaises(ValueError):
            br.parse_hours("0-3")
        with tempfile.TemporaryDirectory() as t:
            for n in ("depth_000000s.tif", "depth_003600s.tif", "depth_086400s.tif", "depth_001800s.tif", "max_depth.tif"):
                Path(t, n).write_bytes(b"x")
            self.assertEqual(sorted(br.find_hour_rasters(t)), [1, 24])


class TestBuild(Base):
    def test_validates_and_meta(self):
        out, res = self.build()
        self.assertEqual(res["problems"], [])
        self.assertEqual(fg.validate_storm_dir(out), [])
        meta = json.load(open(os.path.join(out, "meta.json")))
        self.assertEqual(meta["source"], "rasters")
        self.assertEqual(meta["hours"], [1, 2, 3])
        self.assertIn("DERIVED", meta["source_note"])
        self.assertIn("not solver output", meta["source_note"])
        self.assertEqual(meta["storm_depth_in"], 3.94)
        self.assertEqual(meta["run_summary"]["total_rain_mm"], 100.0)
        self.assertEqual(meta["rasters_build"]["block_k"], 4)
        self.assertEqual(meta["rasters_build"]["dem"], "dem_test_2m.tif")
        self.assertTrue(os.path.isfile(os.path.join(out, "flowpaths.rgba.gz")))
        self.assertEqual([f for f in os.listdir(out) if f.endswith(".tmp")], [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out", ".s1.building")))
        self.assertEqual(meta["label"], "s1")

    def test_volume_consistency(self):
        # output volume = raster volume minus the water thinner than the 0.02 m wet threshold (a big share only in the
        # faint hour 1: 0.2 m peak) and bilinear/edge effects: within 3 % for the deeper hours, never more than the raster
        out, res = self.build()
        meta = json.load(open(os.path.join(out, "meta.json")))
        for e in meta["hourly"]:
            raster = meta["rasters_build"]["raster_volume_m3"][str(e["hour"])]
            ratio = e["volume_m3"] / raster
            self.assertLess(ratio, 1.03)
            self.assertGreater(ratio, 0.97 if e["hour"] > 1 else 0.88, "hour %d" % e["hour"])
            self.assertAlmostEqual(meta["rasters_build"]["volume_ratio"][str(e["hour"])], ratio, delta=1e-3)

    def test_flow_goes_downhill_and_speed_bound(self):
        out, res = self.build()
        meta = json.load(open(os.path.join(out, "meta.json")))
        rgba = fg.read_rgba_gz(os.path.join(out, "h03.rgba.gz"), meta["width"], meta["height"])
        wet = rgba[..., 0] > 0
        u = fg.decode_vel(rgba[..., 1])[wet]
        self.assertGreater(float(u.mean()), 0.1)                      # terrain falls east: water moves east
        d = fg.decode_depth(rgba[..., 0])[wet]
        sp = np.hypot(u, fg.decode_vel(rgba[..., 2])[wet])
        self.assertTrue((sp <= np.minimum(np.sqrt(9.81 * d) * 1.05 + 0.05, 4.05)).all())

    def test_strips_equal_full_read(self):
        sg = br.solver_grid(os.path.join(self.run_dir, "depth_010800s.tif"))
        p = os.path.join(self.run_dir, "depth_010800s.tif")
        dem = br.RowReader(self.dem, sg)
        mann = br.RowReader(os.path.join(self.tmp, "dem_test_2m.tif"), sg, 0.0001)     # a valid positive n field
        res = []
        for sr in (8, 20, 64, 4096):
            res.append(br.hour_fields(p, sg, 4, sr, dem, mann))
        for r in res[1:]:
            for a, b in zip(res[0][:3], r[:3]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(res[0][3]["raster_wet_cells"], r[3]["raster_wet_cells"])
        # independent reference: whole raster, face_velocity + block_reduce
        with rasterio.open(p) as ds:
            d = ds.read(1)
        d = np.where(d <= -9000, np.nan, d)
        d = np.where(d < br.DRY_FLOOR_M, 0.0, d)
        z = dem.read_rows(0, H)
        n = mann.read_rows(0, H)
        ug, vg = fg.face_velocity(d, z, n, DX)
        db, ub, vb, _ = fg.block_reduce(d, ug, vg, 4, valid=np.isfinite(d))
        np.testing.assert_allclose(res[0][0], db, atol=1e-6)
        np.testing.assert_allclose(res[0][1], ub, atol=1e-5)
        dem.close()
        mann.close()

    def test_misaligned_dem_uses_warp(self):
        sg = br.solver_grid(os.path.join(self.run_dir, "depth_010800s.tif"))
        coarse = os.path.join(self.tmp, "dem_other.tif")
        write_tif(coarse, terrain()[::2, ::2][:90, :110], tr=Affine(10.0, 0, E0 - 20, 0, -10.0, N1 + 30))
        r = br.RowReader(coarse, sg)
        self.assertEqual(r.mode, "warp")
        z = r.read_rows(40, 60)
        mid = z[:, 100]
        self.assertTrue(np.isfinite(mid).all())
        r.close()
        out, res = self.build("s2", dem=coarse)
        self.assertEqual(res["problems"], [])

    def test_no_dem(self):
        out, res = self.build("nodem", dem=None)
        self.assertEqual(res["problems"], [])
        meta = json.load(open(os.path.join(out, "meta.json")))
        self.assertIn("no terrain", meta["source_note"])
        self.assertIsNone(meta["rasters_build"]["dem"])

    def test_dem_units_required(self):
        with self.assertRaises(ValueError):
            br.dem_units("somewhere/unknown.tif", "auto")
        self.assertEqual(br.dem_units("x/dem_bg.tif", "auto"), "feet")
        self.assertEqual(br.dem_units("x/dem_warren_5m.tif", "auto"), "metres")

    def test_feet_dem_gives_slower_flow(self):
        o1, _ = self.build("m", dem_units_arg="metres")
        o2, _ = self.build("f", dem_units_arg="feet")
        m1 = json.load(open(os.path.join(o1, "meta.json")))["hourly"][-1]["max_speed_ms"]
        m2 = json.load(open(os.path.join(o2, "meta.json")))["hourly"][-1]["max_speed_ms"]
        self.assertLess(m2, m1)

    def test_resume_and_skip(self):
        out, res = self.build()
        again = br.build_storm(self.run_dir, out, "s1", dem=self.dem, dem_units_arg="metres", cell_m=25.0, log=lambda m: None)
        self.assertTrue(again.get("skipped_all"))
        b0 = Path(out, "h02.rgba.gz").read_bytes()
        out2, res2 = self.build(force=True)
        self.assertEqual(Path(out2, "h02.rgba.gz").read_bytes(), b0)       # deterministic bytes

    def test_partial_resume(self):
        out = os.path.join(self.tmp, "out", "p")
        msgs = []
        br.build_storm(self.run_dir, out, "p", dem=self.dem, dem_units_arg="metres", cell_m=25.0, hours="1", log=lambda m: None)
        # a one-hour build is a different key: a full run converts all hours again but the result is complete and valid
        res = br.build_storm(self.run_dir, out, "p", dem=self.dem, dem_units_arg="metres", cell_m=25.0, log=msgs.append)
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["hours"], [1, 2, 3])

    def test_workers_identical(self):
        o1, _ = self.build("w1", workers=1)
        o2, _ = self.build("w2", workers=2)
        for f in sorted(os.listdir(o1)):
            if f != "meta.json":
                self.assertEqual(Path(o1, f).read_bytes(), Path(o2, f).read_bytes(), f)

    def test_dry_hour_omitted(self):
        write_tif(os.path.join(self.run_dir, "depth_014400s.tif"), np.where(depth_field(1) < 0, -9999.0, 0.0))
        out, res = self.build("dry")
        meta = json.load(open(os.path.join(out, "meta.json")))
        self.assertEqual(meta["hours"], [1, 2, 3])
        self.assertEqual(meta["rasters_build"]["dry_hours"], [4])
        self.assertEqual(res["problems"], [])

    def test_wrong_crs_rejected(self):
        p = os.path.join(self.tmp, "bad")
        os.makedirs(p)
        with rasterio.open(os.path.join(p, "depth_003600s.tif"), "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                           crs="EPSG:4326", transform=Affine(0.001, 0, -86, 0, -0.001, 37)) as ds:
            ds.write(np.zeros((8, 8), np.float32), 1)
        with self.assertRaises(ValueError):
            br.build_storm(p, os.path.join(self.tmp, "o2"), "x", log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
