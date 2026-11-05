"""
Tests for the real-depth COG pipeline (python side): make_cogs.py output, the synthetic fixture the browser harness reads, the
EPSG:4326 -> EPSG:3857 -> pixel maths the JavaScript reader uses (checked against PROJ through rasterio.warp), and HTTP Range service of a COG by serve.py.
The JavaScript reader itself is tested in the browser: cog/test/cog-harness.html (needs serve.py running; 30+ checks against this fixture).

  C:/Users/Minec/anaconda3/python.exe -m pytest scripts/tests/test_cog_reader.py -q -p no:cacheprovider
"""
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import make_cog_fixture as mf  # noqa: E402
import make_cogs  # noqa: E402

APP = Path(os.environ.get("BGMAP_APP", r"C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map"))
R = 6378137.0


def merc(lng, lat):
    """The formula cog-depth.js uses (spherical mercator, R = WGS84 semi-major axis)."""
    lng, lat = np.asarray(lng, float), np.asarray(lat, float)
    return R * np.radians(lng), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cogfx_")
        cls.exp = mf.build(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cog_structure(self):
        for name in ("a_direct.tif", "b_pipeline.tif"):
            p = os.path.join(self.tmp, name)
            self.assertTrue(make_cogs.verify(p), name)
            with rasterio.open(p) as d:
                self.assertEqual(d.crs.to_epsg(), 3857)
                self.assertEqual(d.dtypes[0], "uint16")
                self.assertEqual(d.block_shapes[0], (512, 512))
                self.assertEqual(d.nodata, 65535)
                self.assertGreaterEqual(len(d.overviews(1)), 1)
                self.assertAlmostEqual(d.transform.a, 6.25)

    def test_lnglat_to_mercator_matches_proj_100_plus_random_points(self):
        rng = np.random.default_rng(7)
        lng, lat = rng.uniform(-87.0, -86.0, 400), rng.uniform(36.5, 37.4, 400)           # the county area
        x, y = merc(lng, lat)
        px, py = warp_transform("EPSG:4326", "EPSG:3857", lng.tolist(), lat.tolist())
        err = np.hypot(x - np.array(px), y - np.array(py))
        self.assertLess(err.max(), 1e-6)                    # metres; half a pixel would be 3.1 m
        # same through the pixel grid of a real fixture file: lng/lat -> pixel differs from rasterio's by far less than half a pixel
        with rasterio.open(os.path.join(self.tmp, "a_direct.tif")) as d:
            tr = d.transform
        lng2, lat2 = warp_transform("EPSG:3857", "EPSG:4326", [tr.c + 100 * 6.25 + 0.3 * 6.25 + i * 3.17 for i in range(150)], [tr.f - 80 * 6.25 - i * 2.71 for i in range(150)])
        x2, y2 = merc(lng2, lat2)
        col, row = (x2 - tr.c) / tr.a, (tr.f - y2) / tr.a
        col_ref, row_ref = ~tr * (np.array([tr.c + 100 * 6.25 + 0.3 * 6.25 + i * 3.17 for i in range(150)]), np.array([tr.f - 80 * 6.25 - i * 2.71 for i in range(150)]))
        self.assertLess(np.abs(col - col_ref).max(), 0.01)
        self.assertLess(np.abs(row - row_ref).max(), 0.01)

    def test_expected_matches_gdal_read(self):
        with rasterio.open(os.path.join(self.tmp, "a_direct.tif")) as d:
            a = d.read(1)
            for rec in self.exp["files"]["a_direct.tif"]["points"][:200]:
                col, row = ~d.transform * (rec["x"], rec["y"])
                c, r = int(np.floor(col)), int(np.floor(row))
                want = int(a[r, c]) if 0 <= c < d.width and 0 <= r < d.height else None
                self.assertEqual(rec["v"][0], want)

    def test_pipeline_ramp_within_one_cm(self):
        """make_cogs.convert of a linear ramp (bilinear resample reproduces a linear function) stays within 1 cm of the analytic depth."""
        n, worst = 0, 0
        with rasterio.open(os.path.join(self.tmp, "b_pipeline.tif")) as d:
            a = d.read(1)
            for rec in self.exp["files"]["b_pipeline.tif"]["points"]:
                if rec.get("analytic_cm") is None:
                    continue
                col, row = ~d.transform * (rec["x"], rec["y"])
                v = int(a[int(np.floor(row)), int(np.floor(col))])
                worst = max(worst, abs(v - rec["analytic_cm"]))
                n += 1
        self.assertGreater(n, 100)
        self.assertLessEqual(worst, 1)

    def test_nodata_and_dry_survive(self):
        with rasterio.open(os.path.join(self.tmp, "b_pipeline.tif")) as d:
            a = d.read(1)
        self.assertTrue((a == 65535).any())                  # the -9999 strip and the area outside the warped raster
        self.assertTrue((a == 0).any())                      # dry block stays 0 (not nodata)

    def test_committed_fixture_is_current(self):
        """cog/test/fixture in the web app folder must be what the generator makes today (the browser harness reads that copy)."""
        p = APP / "cog" / "test" / "fixture" / "expected.json"
        if not p.is_file():
            self.skipTest("web app fixture folder not present")
        cur = json.loads(p.read_text())
        self.assertEqual(cur["files"]["a_direct.tif"]["points"], self.exp["files"]["a_direct.tif"]["points"])
        self.assertEqual([x["analytic_cm"] for x in cur["files"]["b_pipeline.tif"]["points"]], [x["analytic_cm"] for x in self.exp["files"]["b_pipeline.tif"]["points"]])


class HourMapTests(unittest.TestCase):
    def test_hour_is_nearest_not_floor(self):
        files = {n: 1 for n in ("depth_000015s.tif", "depth_003604s.tif", "depth_003602s.tif", "depth_010798s.tif", "depth_007200s.tif", "max_depth.tif")}
        hm = make_cogs.hour_map(files)
        self.assertEqual(hm["0"], "depth_000015s.tif")
        self.assertEqual(hm["1"], "depth_003602s.tif")        # closest to 3600
        self.assertEqual(hm["2"], "depth_007200s.tif")
        self.assertEqual(hm["3"], "depth_010798s.tif")        # a floor would have put it on hour 2
        self.assertNotIn("max", hm)

    def test_run_writes_meta_with_hours(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = os.path.join(t, "s"), os.path.join(t, "d")
            os.makedirs(src)
            for sec in (15, 3604, 7200, 10798):
                a = np.zeros((60, 60), np.float32); a[10:20, 10:20] = sec / 10000.0
                with rasterio.open(os.path.join(src, "depth_%06ds.tif" % sec), "w", driver="GTiff", width=60, height=60, count=1, dtype="float32",
                                   crs="EPSG:32616", transform=Affine(5, 0, 560000, 0, -5, 4090000), nodata=-9999.0) as d:
                    d.write(a, 1)
            meta = make_cogs.run(src, dst, 6.25, log=lambda *a: None)
            self.assertEqual(meta["hours"], {"0": "depth_000015s.tif", "1": "depth_003604s.tif", "2": "depth_007200s.tif", "3": "depth_010798s.tif"})
            self.assertIn("max_depth.tif", meta["files"])      # built from the hourly files when the source has none
            self.assertEqual(meta["units"], "cm")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def tiff_tile_table(path):
    """(offset, bytecount) of the first tile of IFD 0 and the tile count, read with plain struct (classic TIFF only)."""
    b = Path(path).read_bytes()
    assert b[:2] == b"II" and struct.unpack("<H", b[2:4])[0] == 42
    ifd = struct.unpack("<I", b[4:8])[0]
    n = struct.unpack("<H", b[ifd:ifd + 2])[0]
    tags = {}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack("<HHII", b[ifd + 2 + 12 * i: ifd + 14 + 12 * i])
        tags[tag] = (typ, cnt, val)
    def arr(tag):
        typ, cnt, val = tags[tag]
        size = {3: 2, 4: 4}[typ]
        return list(struct.unpack("<%d%s" % (cnt, "H" if typ == 3 else "I"), b[val:val + cnt * size])) if cnt * size > 4 else list(struct.unpack("<%d%s" % (cnt, "H" if typ == 3 else "I"), struct.pack("<I", val)[:cnt * size]))
    return arr(324), arr(325), b


class RangeServeTests(unittest.TestCase):
    """serve.py must answer the reader's Range requests for a COG with exact bytes (head, the last tile, a suffix range)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cogsrv_")
        shutil.copy2(str(APP / "serve.py"), os.path.join(cls.tmp, "serve.py"))
        os.makedirs(os.path.join(cls.tmp, "layers", "cog", "county", "t"))
        cls.exp = mf.build(os.path.join(cls.tmp, "fx"))
        shutil.copy2(os.path.join(cls.tmp, "fx", "a_direct.tif"), os.path.join(cls.tmp, "layers", "cog", "county", "t", "max_depth.tif"))
        cls.port = free_port()
        cls.proc = subprocess.Popen([sys.executable, os.path.join(cls.tmp, "serve.py"), "--no-browser", "--port", str(cls.port)],
                                    cwd=cls.tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", cls.port), 0.2).close(); break
            except OSError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(5)
        except Exception:
            cls.proc.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, rng=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/layers/cog/county/t/max_depth.tif", headers=({"Range": rng} if rng else {}))
        r = c.getresponse(); body = r.read(); h = {k.lower(): v for k, v in r.getheaders()}; c.close()
        return r.status, h, body

    def test_head_and_tiles_exact(self):
        offs, cnts, raw = tiff_tile_table(os.path.join(self.tmp, "layers", "cog", "county", "t", "max_depth.tif"))
        st, h, body = self.get("bytes=0-65535")
        self.assertEqual(st, 206)
        self.assertEqual(body, raw[:65536])
        self.assertEqual(h["content-type"], "image/tiff")
        self.assertEqual(h["accept-ranges"], "bytes")
        self.assertTrue(h["content-range"].startswith("bytes 0-65535/%d" % len(raw)))
        for k in (0, len(offs) // 2, len(offs) - 1):              # first, middle and last tile: exact bytes, exact Content-Range
            st, h, body = self.get("bytes=%d-%d" % (offs[k], offs[k] + cnts[k] - 1))
            self.assertEqual(st, 206)
            self.assertEqual(body, raw[offs[k]: offs[k] + cnts[k]])
            self.assertEqual(h["content-range"], "bytes %d-%d/%d" % (offs[k], offs[k] + cnts[k] - 1, len(raw)))

    def test_suffix_and_full_and_bad_range(self):
        _, _, raw = tiff_tile_table(os.path.join(self.tmp, "layers", "cog", "county", "t", "max_depth.tif"))
        st, h, body = self.get("bytes=-100")
        self.assertEqual((st, body), (206, raw[-100:]))
        st, h, body = self.get()
        self.assertEqual((st, len(body)), (200, len(raw)))
        st, h, body = self.get("bytes=%d-" % (len(raw) + 10))
        self.assertEqual(st, 416)


class ResampleAccuracyTests(unittest.TestCase):
    def test_noisy_field_matches_bilinear_within_one_cm(self):
        """Regression for GDAL's default 0.125 px warp tolerance (up to 7 cm on noisy real depths): make_cogs.convert must stay within 1 cm of the exact bilinear value."""
        rng = np.random.default_rng(5)
        n = 900
        a = (1.5 + rng.normal(0, 0.3, (n, n))).astype(np.float32)
        tr = Affine(5, 0, 524851.2, 0, -5, 4116296.3)
        with tempfile.TemporaryDirectory() as t:
            src, out = os.path.join(t, "s.tif"), os.path.join(t, "o.tif")
            with rasterio.open(src, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32", crs="EPSG:32616", transform=tr, nodata=-9999.0) as d:
                d.write(a, 1)
            make_cogs.convert(src, out, 6.25)
            worst = 0
            with rasterio.open(out) as d:
                v = d.read(1)
                for _ in range(400):
                    c, r = int(rng.integers(5, d.width - 5)), int(rng.integers(5, d.height - 5))
                    x, y = d.transform * (c + 0.5, r + 0.5)
                    e, nn = warp_transform("EPSG:3857", "EPSG:32616", [x], [y])
                    fc, fr = ~tr * (e[0], nn[0])
                    c0, r0 = int(np.floor(fc - 0.5)), int(np.floor(fr - 0.5))
                    if c0 < 0 or r0 < 0 or c0 + 1 >= n or r0 + 1 >= n:
                        continue
                    wx, wy = fc - 0.5 - c0, fr - 0.5 - r0
                    b = a[r0, c0] * (1 - wx) * (1 - wy) + a[r0, c0 + 1] * wx * (1 - wy) + a[r0 + 1, c0] * (1 - wx) * wy + a[r0 + 1, c0 + 1] * wx * wy
                    worst = max(worst, abs(int(v[r, c]) - int(round(max(float(b), 0) * 100))))
            self.assertLessEqual(worst, 1)


if __name__ == "__main__":
    unittest.main()
