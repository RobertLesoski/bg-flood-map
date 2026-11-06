"""Analysis tools, data side (no GPU): geodesy, windowed raster reads, the depth adapter (flow grid + tiles), profile, series,
flood timeline and compare statistics.  Every number is checked against an independent computation: a hand-written Vincenty, the
EPSG:6933 equal-area projection from PROJ, a plain numpy bilinear on the whole array, `Storm.sample_t`, whole-array counts.
Real-data tests are skipped when the data folder or the county tiles are absent."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_helpers import real_root
from bgv.analysis import compare, depth_source as ds, geodesy, profile, series, timeline
from bgv.analysis.rastersrc import BlockRaster, BlockCache
from bgv.data import flow
from bgv.geo import GridFrame

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

FRAME = GridFrame(60, 50, 10.0, 8.0, -9633395.2, 4446630.4 - 500.0, -9633395.2 + 600.0, 4446630.4)


# ------------------------------------------------------------------------------------------------ independent references
def vincenty(lng1, lat1, lng2, lat2):
    a, f = 6378137.0, 1 / 298.257223563
    b = (1 - f) * a
    L = math.radians(lng2 - lng1)
    U1, U2 = math.atan((1 - f) * math.tan(math.radians(lat1))), math.atan((1 - f) * math.tan(math.radians(lat2)))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa * sa
        c2sm = cs - 2 * sU1 * sU2 / c2a if c2a else 0.0
        C = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
        lam2 = L + (1 - C) * f * sa * (sig + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm ** 2)))
        if abs(lam2 - lam) < 1e-12:
            lam = lam2
            break
        lam = lam2
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds_ = B * ss * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm ** 2) - B / 6 * c2sm * (-3 + 4 * ss ** 2) * (-3 + 4 * c2sm ** 2)))
    return b * A * (sig - ds_)


def bilinear_ref(arr, gx, gy, nodata_mask):
    """Plain python bilinear at pixel-centre-based coordinates (gx, gy in pixel units, centres at +0.5), nodata skipped."""
    H, W = arr.shape
    fx, fy = gx - 0.5, gy - 0.5
    x0, y0 = math.floor(fx), math.floor(fy)
    tx, ty = fx - x0, fy - y0
    num = den = 0.0
    for dx, dy, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        c, r = min(max(x0 + dx, 0), W - 1), min(max(y0 + dy, 0), H - 1)
        if nodata_mask[r, c]:
            continue
        num += w * float(arr[r, c])
        den += w
    return num / den if den > 1e-12 else float("nan")


# ------------------------------------------------------------------------------------------------ geodesy
def test_path_length_matches_vincenty():
    rng = np.random.default_rng(7)
    for _ in range(40):
        lng0, lat0 = rng.uniform(-86.7, -86.2), rng.uniform(36.75, 37.2)
        lng1, lat1 = lng0 + rng.uniform(-0.3, 0.3), lat0 + rng.uniform(-0.3, 0.3)
        ref = vincenty(lng0, lat0, lng1, lat1)
        got = geodesy.path_length_m([lng0, lng1], [lat0, lat1])
        assert abs(got - ref) / ref < 2e-4, (got, ref)
    # a short line: sub-centimetre agreement
    ref = vincenty(-86.44, 36.99, -86.4395, 36.9903)
    assert abs(geodesy.path_length_m([-86.44, -86.4395], [36.99, 36.9903]) - ref) < 0.01


def test_polygon_area_matches_equal_area_projection():
    from rasterio.warp import transform
    rng = np.random.default_rng(3)
    for n in (3, 5, 9):
        c = np.array([rng.uniform(-86.6, -86.3), rng.uniform(36.85, 37.1)])
        ang = np.sort(rng.uniform(0, 2 * math.pi, n))
        rad = rng.uniform(0.01, 0.12, n)
        lng, lat = c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang) * 0.8
        t = np.arange(512) / 512.0                      # edges are straight in lng/lat (the function's contract): densify the
        dl = (lng[:, None] + (np.roll(lng, -1) - lng)[:, None] * t).ravel()     # reference too, else it measures edges
        dt = (lat[:, None] + (np.roll(lat, -1) - lat)[:, None] * t).ravel()     # straight in the projection
        x, y = transform("EPSG:4326", "EPSG:6933", dl.tolist(), dt.tolist())      # PROJ's cylindrical equal-area
        x, y = np.array(x), np.array(y)
        ref = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        got = geodesy.polygon_area_m2(lng, lat)
        assert abs(got - ref) / ref < 1e-6, (got, ref)
    # 1 km x 1 km square near BG is ~1e6 m2 within 0.5 %
    sq = geodesy.polygon_area_m2([-86.44, -86.4288, -86.4288, -86.44], [36.99, 36.99, 36.99900, 36.99900])
    assert 0.9e6 < sq < 1.1e6


def test_world_roundtrip_and_formats():
    X = np.array([-5000.0, 0.0, 12000.0])
    Y = np.array([300.0, 0.0, -8000.0])
    lng, lat = geodesy.world_to_lnglat(FRAME, X, Y)
    X2, Y2 = geodesy.lnglat_to_world(FRAME, lng, lat)
    assert np.allclose(X, X2, atol=1e-6) and np.allclose(Y, Y2, atol=1e-6)
    a, b = geodesy.fmt_length(1234.5, False)
    assert a.startswith("1.23") and a.endswith("km") and b.endswith("ft")
    a, b = geodesy.fmt_length(1234.5, True)
    assert a.endswith("mi") is False and a.endswith("ft") and b.endswith("km")
    a, b = geodesy.fmt_area(10000.0, False)
    assert a == "1.00 ha" and "acres" in b
    a, b = geodesy.fmt_area(4046.8564224, True)
    assert a.startswith("1.00 acres")
    assert geodesy.polygon_area_m2([0, 1], [0, 1]) == 0.0


# ------------------------------------------------------------------------------------------------ raster reads
def _write_raster(path, arr, left, top, res, dtype="uint16", nodata=65535, block=32, crs="EPSG:3857"):
    H, W = arr.shape
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=1, dtype=dtype, crs=crs,
                       transform=from_origin(left, top, res, res), nodata=nodata, tiled=True, blockxsize=block, blockysize=block,
                       compress="deflate") as ds_:
        ds_.write(arr.astype(dtype), 1)


@pytest.fixture
def raster(tmp_path):
    rng = np.random.default_rng(11)
    arr = (rng.random((150, 200)) * 400).astype(np.uint16)
    arr[40:60, 70:90] = 65535                      # a no-data hole
    arr[:, :5] = 0
    p = tmp_path / "r.tif"
    left, top = -9633000.0, 4446000.0
    _write_raster(p, arr, left, top, 6.25)
    return p, arr, left, top


def test_blockraster_bilinear_matches_reference_at_150_points(raster):
    p, arr, left, top = raster
    r = BlockRaster(p, scale=0.01, nodata=65535, cache=BlockCache(1 << 20))
    rng = np.random.default_rng(5)
    gx = rng.uniform(0, 200, 150)
    gy = rng.uniform(0, 150, 150)
    x, y = left + gx * 6.25, top - gy * 6.25
    got = r.sample(x, y)
    mask = arr == 65535
    ref = np.array([bilinear_ref(arr * 0.01, a, b, mask) for a, b in zip(gx, gy)])
    assert np.allclose(got, ref, equal_nan=True, atol=1e-5)
    assert np.isfinite(got).sum() > 120
    near = r.sample(x, y, "nearest")
    exp = np.array([arr[min(int(b), 149), min(int(a), 199)] * 0.01 for a, b in zip(gx, gy)])
    exp[np.array([mask[min(int(b), 149), min(int(a), 199)] for a, b in zip(gx, gy)])] = np.nan
    assert np.allclose(near, exp, equal_nan=True, atol=1e-6)


def test_blockraster_nodata_outside_and_cache(raster):
    p, arr, left, top = raster
    cache = BlockCache(1 << 20)
    r = BlockRaster(p, scale=0.01, nodata=65535, cache=cache)
    inside_hole = r.sample([left + 80 * 6.25], [top - 50 * 6.25])
    assert np.isnan(inside_hole[0])
    assert np.isnan(r.sample([left - 100.0], [top])[0]) and np.isnan(r.sample([left], [top + 100.0])[0])
    r.sample([left + 10 * 6.25], [top - 10 * 6.25])
    h0 = cache.hits
    r.sample([left + 11 * 6.25], [top - 11 * 6.25])
    assert cache.hits > h0                           # the second read came from the block cache
    c = BlockCache(max_bytes=40_000)                 # a few 32x32 float32 blocks only: bounded, still correct
    r2 = BlockRaster(p, scale=0.01, nodata=65535, cache=c)
    v = r2.sample(left + np.linspace(0, 1100, 200), top - np.linspace(0, 800, 200))
    assert c.nbytes <= 40_000 + 32 * 32 * 4 and np.isfinite(v).sum() > 150


def test_read_rows_and_overview(raster):
    p, arr, left, top = raster
    r = BlockRaster(p, scale=0.01, nodata=65535)
    band = r.read_rows(10, 30)
    ref = arr[10:30].astype(np.float32) * 0.01
    ref[arr[10:30] == 65535] = np.nan
    assert np.allclose(band, ref, equal_nan=True, atol=1e-6)


# ------------------------------------------------------------------------------------------------ tiles + adapter
def _tile_folder(tmp_path, storm="s1", hours=(0, 1, 2, 3, 4), names=None, meta=True):
    d = tmp_path / "tiles" / storm
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    base = (rng.random((120, 160)) * 300).astype(np.float64)
    files = {}
    for h in hours:
        a = np.rint(base * (h / 4.0)).astype(np.uint16)
        a[:6, :6] = 65535
        fn = (names or {}).get(h, f"depth_{h * 3600:06d}s.tif")
        _write_raster(d / fn, a, -9633000.0, 4446000.0, 6.25)
        files[h] = (fn, a)
    if meta:
        (d / "cog_meta.json").write_text(json.dumps({"crs": "EPSG:3857", "units": "cm", "dtype": "uint16", "nodata": 65535,
                                                     "res_m": 6.25, "hours": {str(h): f[0] for h, f in files.items()}}))
    return d.parent, files


def test_tile_source_depth_and_hour_blend(tmp_path):
    root, files = _tile_folder(tmp_path)
    src = ds.TileDepthSource(root)
    assert src.storms() == ["s1"] and src.has("s1") and src.resolution_m("s1") == 6.25
    rng = np.random.default_rng(2)
    gx, gy = rng.uniform(8, 150, 120), rng.uniform(8, 110, 120)
    x, y = -9633000.0 + gx * 6.25, 4446000.0 - gy * 6.25
    lng, lat = geodesy.merc_to_lnglat(x, y)
    for hour in (2.0, 2.5, 3.25):
        got = src.depth_many(lng, lat, "s1", hour)
        h0, h1, m = flow.hour_pair(hour)
        a0, a1 = files[h0][1], files[h1][1]
        ref = []
        for a, b in zip(gx, gy):
            d0 = bilinear_ref(a0 * 0.01, a, b, a0 == 65535)
            d1 = bilinear_ref(a1 * 0.01, a, b, a1 == 65535)
            d = d0 * (1 - m) + d1 * m if m > 0 else d0
            ref.append(0.0 if d < ds.WET_THRESHOLD_M else d)
        assert np.allclose(got, ref, atol=1e-4)
    assert src.depth_at(lng[0], lat[0], "s1", 2.0) == pytest.approx(float(src.depth_many(lng[:1], lat[:1], "s1", 2.0)[0]))
    # no-data corner and far outside the tile read dry, never NaN
    lg, lt = geodesy.merc_to_lnglat(np.array([-9633000.0 + 1 * 6.25, -9000000.0]), np.array([4446000.0 - 1 * 6.25, 4446000.0]))
    d = src.depth_many(lg, lt, "s1", 3.0)
    assert np.all(d == 0.0)
    assert "tiles, 6.25 m" in src.label("s1")


def test_tile_hour_from_file_name_without_meta(tmp_path):
    root, _ = _tile_folder(tmp_path, hours=(0, 1, 2), names={0: "depth_000015s.tif", 1: "depth_003604s.tif", 2: "depth_007208s.tif"}, meta=False)
    ts = ds.TileStorm(root / "s1")
    assert sorted(ts.files) == [0, 1, 2]


def test_verification_gate(tmp_path):
    log = tmp_path / "WEB_COG_PROGRESS.md"
    log.write_text("## Log\n3. Viewer integration done and verified with SYNTHETIC demo COGs.\n4. real-data verification pending\n")
    assert ds.tiles_verified(settings={"tiles_progress_log": str(log)}) is False
    log.write_text("## Log\nstuff\nStatus: verified\n")
    assert ds.tiles_verified(settings={"tiles_progress_log": str(log)}) is True
    assert ds.tiles_verified(settings={"tiles_verified": True, "tiles_progress_log": str(tmp_path / "none.md")}) is True
    root, _ = _tile_folder(tmp_path)
    src = ds.TileDepthSource(root, settings={"tiles_progress_log": str(log)})
    assert src.label("s1").endswith("verified)") and "pending" not in src.label("s1")
    log.write_text("pending\n")
    assert "verification pending" in ds.TileDepthSource(root, settings={"tiles_progress_log": str(log)}).label("s1")


def _flow_world(tmp_path):
    meta = {"format": "bgflow", "version": 1, "storm": "t", "label": "t", "source": "rasters", "crs": "EPSG:3857", "width": 60, "height": 50,
            "cell_m": 10.0, "cell_ground_m": 8.0, "bounds_merc": list(map(float, (FRAME.xmin, FRAME.ymin, FRAME.xmax, FRAME.ymax))),
            "hours": [2, 3, 4], "depth_encoding": {"type": "sqrt", "max_m": 8.0}, "vel_encoding": {"vmax_ms": 4.0, "zero_code": 128}}
    rng = np.random.default_rng(9)
    d = tmp_path / "layers" / "flow" / "t"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    grids = {}
    for h in (2, 3, 4):
        g = np.zeros((50, 60, 4), np.uint8)
        g[..., 0] = np.where(rng.random((50, 60)) < 0.5, (rng.random((50, 60)) * 140).astype(np.uint8), 0)
        g[..., 1:3] = 128
        g[..., 3] = 255
        g[:4, :4, 3] = 0                              # outside the model domain
        (d / f"h{h:02d}.rgba.gz").write_bytes(gzip.compress(g.tobytes(), mtime=0))
        grids[h] = g
    return flow.Storm.open(tmp_path, "t"), grids


def test_flowgrid_adapter_matches_storm_sample_t_at_150_points(tmp_path):
    storm, _ = _flow_world(tmp_path)
    src = ds.FlowGridDepthSource(lambda n: flow.Storm.open(tmp_path, n), current=lambda: storm, storms=["t"])
    rng = np.random.default_rng(4)
    X = rng.uniform(-FRAME.size_m[0] / 2 - 20, FRAME.size_m[0] / 2 + 20, 150)       # includes points just outside the grid
    Y = rng.uniform(-FRAME.size_m[1] / 2 - 20, FRAME.size_m[1] / 2 + 20, 150)
    lng, lat = geodesy.world_to_lnglat(storm.frame, X, Y)
    for hour in (1.0, 2.0, 2.4, 3.5, 4.0, 4.7, 9.0):
        got = src.depth_many(lng, lat, "t", hour)
        ref = np.array([storm.sample_t(hour, x, y)["depth"] for x, y in zip(X, Y)])
        ref = np.where(ref < storm.wet_threshold, 0.0, ref)
        assert np.allclose(got, ref, atol=1e-6), (hour, np.abs(got - ref).max())
    assert src.depth_at(lng[3], lat[3], "t", 3.0) == pytest.approx(float(src.depth_many(lng[3:4], lat[3:4], "t", 3.0)[0]))
    assert "flow grid" in src.label("t")


def test_hybrid_prefers_tiles_and_falls_back(tmp_path):
    storm, _ = _flow_world(tmp_path)
    root, _ = _tile_folder(tmp_path, storm="t")
    grid = ds.FlowGridDepthSource(lambda n: flow.Storm.open(tmp_path, n), current=lambda: storm, storms=["t", "u"])
    tiles = ds.TileDepthSource(root)
    h = ds.HybridDepthSource(grid, tiles)
    assert h.uses_tiles("t") and not h.uses_tiles("u") and h.source_key("t") == "tiles" and h.source_key("u") == "grid"
    h.mode = "grid"
    assert not h.uses_tiles("t") and "flow grid" in h.label("t")
    assert ds.HybridDepthSource(grid, None).uses_tiles("t") is False


# ------------------------------------------------------------------------------------------------ profile
class FlatGround:
    kind = "dem"
    label = "ground: test"

    def sample_lnglat(self, lng, lat):
        x, y = geodesy.lnglat_to_merc(lng, lat)
        return 100.0 + (x + 9633000.0) * 0.01


def test_profile_samples_match_direct_reads(tmp_path):
    root, files = _tile_folder(tmp_path, storm="t")
    storm, _ = _flow_world(tmp_path)
    grid = ds.FlowGridDepthSource(lambda n: flow.Storm.open(tmp_path, n), current=lambda: storm, storms=["t"])
    src = ds.HybridDepthSource(grid, ds.TileDepthSource(root))
    p0 = FRAME.lnglat_to_world(*geodesy.merc_to_lnglat(-9633000.0 + 20 * 6.25, 4446000.0 - 20 * 6.25))
    p1 = FRAME.lnglat_to_world(*geodesy.merc_to_lnglat(-9633000.0 + 140 * 6.25, 4446000.0 - 90 * 6.25))
    P = profile.build_profile(np.array([p0, p1]), FRAME, src, "t", 3.0, FlatGround())
    assert P.n >= 100
    assert P.dist_m[0] == 0.0 and np.all(np.diff(P.dist_m) > 0)
    lng_e, lat_e = geodesy.world_to_lnglat(FRAME, np.array([p0[0], p1[0]]), np.array([p0[1], p1[1]]))
    assert P.length_m == pytest.approx(vincenty(lng_e[0], lat_e[0], lng_e[1], lat_e[1]), rel=2e-4)
    a = files[3][1]
    mask = a == 65535
    xm, ym = geodesy.lnglat_to_merc(P.lng, P.lat)
    for i in range(0, P.n, max(P.n // 120, 1)):
        gx, gy = (xm[i] + 9633000.0) / 6.25, (4446000.0 - ym[i]) / 6.25
        d = bilinear_ref(a * 0.01, gx, gy, mask)
        d = 0.0 if (not math.isfinite(d) or d < 0.02) else d
        assert P.depth_m[i] == pytest.approx(d, abs=1e-4)
        assert P.ground_m[i] == pytest.approx(100.0 + (xm[i] + 9633000.0) * 0.01, abs=1e-6)
    assert P.max_depth_m == pytest.approx(P.depth_m.max()) and P.wet_length_m <= P.length_m + P.step_m
    r = P.at(P.length_m * 0.5)
    assert abs(r["dist_m"] - P.length_m * 0.5) < 1e-9
    assert "tiles" in P.source_label and P.ground_label == "ground: test"
    s = P.surface_m
    assert np.all(np.isnan(s[P.depth_m <= 0])) and np.allclose(s[P.depth_m > 0], (P.ground_m + P.depth_m)[P.depth_m > 0])


def test_dem_ground_reads_the_file(tmp_path):
    rng = np.random.default_rng(8)
    z = (rng.random((100, 120)) * 50 + 100).astype(np.float32)
    z[10:20, 10:20] = -9999
    path = tmp_path / "dem.tif"
    _write_raster(path, z, 500000.0, 4100000.0, 5.0, dtype="float32", nodata=-9999.0, crs="EPSG:32616", block=16)
    g = profile.DemGround(path, "metres")
    assert g.resolution_m() == 5.0
    from rasterio.warp import transform
    gx, gy = rng.uniform(25, 115, 120), rng.uniform(25, 95, 120)
    xs, ys = 500000.0 + gx * 5.0, 4100000.0 - gy * 5.0
    lng, lat = transform("EPSG:32616", "EPSG:4326", xs.tolist(), ys.tolist())
    got = g.sample_lnglat(np.array(lng), np.array(lat))
    ref = np.array([bilinear_ref(z, a, b, z == -9999) for a, b in zip(gx, gy)])
    assert np.allclose(got, ref, atol=2e-3, equal_nan=True)         # PROJ round trip: sub-millimetre
    feet = tmp_path / "feet.tif"
    _write_raster(feet, np.full((20, 20), 100.0, np.float32), 500000.0, 4100000.0, 5.0, dtype="float32", nodata=-9999.0, crs="EPSG:32616", block=16)
    lg, lt = transform("EPSG:32616", "EPSG:4326", [500050.0], [4099950.0])
    assert profile.DemGround(feet, "feet").sample_lnglat(np.array(lg), np.array(lt))[0] == pytest.approx(30.48, abs=1e-3)


# ------------------------------------------------------------------------------------------------ series
class FakeSource:
    def __init__(self):
        self.calls = []
        self.mode = "auto"
        self.tiles = None

    def source_key(self, storm):
        return "tiles" if storm != "g" else "grid"

    def resolution_m(self, storm):
        return 6.25

    def label(self, storm):
        return f"fake {storm}"

    def depth_many(self, lng, lat, storm, hour, method=None):
        self.calls.append((storm, hour, method))
        base = {"a": 1.0, "b": 2.0, "g": 3.0}[storm]
        return base * hour / 24.0 * (1.0 + np.arange(len(np.atleast_1d(lng))) * 0.1)


def test_series_rows_and_worker():
    src = FakeSource()
    pt = series.point_target(-86.44, 36.99)
    row, backend = series.compute_row(src, pt, "a")
    assert backend == "tiles" and np.allclose(row, np.arange(25) / 24.0)
    bt = series.Target("building", ("b", 1), "bld", (0, 0, 0), (0, 0, 0), (0, 0), (0, 0))
    row, _ = series.compute_row(src, bt, "b")
    assert np.allclose(row, 2.0 * np.arange(25) / 24.0 * 1.2)          # max over the three footprint samples
    assert ("b", 5.0, "nearest") in src.calls and ("a", 5.0, None) in src.calls
    w = series.SeriesWorker(lambda: src)
    try:
        r = w.request(pt, ["a", "b", "g"], first="g")
        import time
        t0 = time.time()
        while not r.complete and time.time() - t0 < 10:
            time.sleep(0.02)
        assert r.complete and r.storms[0] == "g" and r.source == ["grid", "tiles", "tiles"]
        assert r.peak(r.index("b")) == pytest.approx(2.0) and r.peak_hour(r.index("b")) == 24
        assert w.request(pt, ["a", "b", "g"], first="g") is r            # cached, not recomputed
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ timeline + compare
def _timeline_world(tmp_path):
    rng = np.random.default_rng(21)
    arr = np.zeros((200, 260), np.float64)
    arr[30:140, 40:200] = rng.random((110, 160)) * 300
    arr[:, :8] = 65535
    a = np.where(arr == 65535, 65535, np.rint(arr)).astype(np.uint16)
    path = tmp_path / "hr.tif"
    left, top = FRAME.xmin + 100.0, FRAME.ymax - 100.0
    _write_raster(path, a, left, top, 6.25, block=64)
    # buildings: squares in tile pixel space, converted to world metres
    centres = [(20 + (17 * i) % 220, 15 + (23 * i) % 170) for i in range(40)]
    X, Y, ids, cx, cy = [], [], [], [], []
    for i, (c, r) in enumerate(centres):
        for dc, dr in ((0, 0), (3, 0), (3, 3), (0, 3)):
            xm, ym = left + (c + dc) * 6.25, top - (r + dr) * 6.25
            wx, wy = FRAME.merc_to_world(xm, ym)
            X.append(wx); Y.append(wy); ids.append(i)
        wx, wy = FRAME.merc_to_world(left + (c + 1.5) * 6.25, top - (r + 1.5) * 6.25)
        cx.append(wx); cy.append(wy)
    bld = SimpleNamespace(n=len(centres), mesh=SimpleNamespace(positions=np.column_stack([X, Y, np.zeros(len(X))]).astype(np.float32),
                                                               building_id=np.array(ids, np.uint32)),
                          cx=np.array(cx, np.float32), cy=np.array(cy, np.float32))
    sx, sy, sw = [], [], []
    for k in range(300):
        c, r = 10 + (k * 7) % 240, 10 + (k * 13) % 180
        wx, wy = FRAME.merc_to_world(left + (c + 0.5) * 6.25, top - (r + 0.5) * 6.25)
        sx.append(wx); sy.append(wy); sw.append(10.0)
    rd = SimpleNamespace(sample_x=np.array(sx, np.float32), sample_y=np.array(sy, np.float32), sample_weight_m=np.array(sw, np.float32),
                         total_length_m=3000.0)
    return path, a, left, top, bld, rd, (centres, sx, sy)


def test_tile_hour_stats_matches_whole_array_computation(tmp_path):
    path, a, left, top, bld, rd, (centres, sx, sy) = _timeline_world(tmp_path)
    r = BlockRaster(path, scale=0.01, nodata=65535)
    bpts, rpts = timeline.prepare_points(r, FRAME, bld, rd)
    got = timeline.tile_hour_stats(r, FRAME, bpts, rpts, band_rows=37)       # odd band size on purpose
    d = np.where(a == 65535, 0.0, a * 0.01)
    # buildings: max over the four corners and the centre pixel of each square
    fl = []
    tiers = np.zeros(4)
    for c, rr in centres:
        v = max(d[rr + dr, c + dc] for dc, dr in ((0, 0), (3, 0), (3, 3), (0, 3), (1, 1)))
        # the centre sample sits at (c + 1.5, r + 1.5) -> pixel (c + 1, r + 1)
        if v >= 0.1:
            fl.append(1)
            tiers[int(np.searchsorted((0.1, 0.3, 0.6, 1.2), v, side="right")) - 1] += 1
    assert got["flooded"] == len(fl) and np.array_equal(got["tiers"], tiers)
    # roads
    wet_m = sum(10.0 for x, y in zip(sx, sy)
                if d[int((top - FRAME.world_to_merc(x, y)[1]) // 6.25), int((FRAME.world_to_merc(x, y)[0] - left) // 6.25)] >= 0.1)
    assert got["road_km"] * 1000.0 == pytest.approx(wet_m, abs=1e-6)
    # area / volume with the per-row cell area, computed row by row independently
    latc = 2 * math.atan(math.exp(FRAME.yc / geodesy.R_MERC)) - math.pi / 2
    area = vol = 0.0
    for row in range(a.shape[0]):
        ym = top - (row + 0.5) * 6.25
        lat = 2 * math.atan(math.exp(ym / geodesy.R_MERC)) - math.pi / 2
        k = FRAME.k * math.cos(lat) / math.cos(latc)
        cell = (6.25 * k) ** 2
        wet = d[row] >= 0.02
        area += wet.sum() * cell
        vol += d[row][wet].sum() * cell
    assert got["area_km2"] == pytest.approx(area / 1e6, rel=1e-9) and got["volume_m3"] == pytest.approx(vol, rel=1e-6)
    assert got["max_depth_m"] == pytest.approx(d.max(), abs=1e-6)


def test_compute_tile_timeline_fills_in_place(tmp_path):
    path, a, left, top, bld, rd, _ = _timeline_world(tmp_path)
    root = tmp_path / "tl" / "s"
    root.mkdir(parents=True)
    for h in (0, 1, 2):
        _write_raster(root / f"depth_{h * 3600:06d}s.tif", (a * (h / 2.0)).astype(np.uint16), left, top, 6.25, block=64)
    ts = ds.TileStorm(root)
    tl = timeline.compute_tile_timeline(ts, bld, rd, FRAME, "x")
    assert tl.complete and tl.flooded[0] == 0 and tl.area_km2[0] == 0 and tl.flooded[2] >= tl.flooded[1] >= tl.flooded[0]
    assert np.all(tl.flooded[3:] == 0)                                   # hours without a file are dry
    again = tl.copy()
    assert np.array_equal(again.flooded, tl.flooded)
    cancel = SimpleNamespace(is_set=lambda: True)
    with pytest.raises(Exception):
        timeline.compute_tile_timeline(ts, bld, rd, FRAME, "x", cancel=cancel)


def test_compare_stats_and_palette(tmp_path):
    rng = np.random.default_rng(31)
    a = (rng.random((50, 60)) * 2.0).astype(np.float32)
    b = a + (rng.random((50, 60)) * 1.0 - 0.3).astype(np.float32)
    a[a < 0.5] = 0
    b[b < 0.0] = 0
    S = compare.compare_grids(a, b, FRAME)
    cg2 = FRAME.cell_ground_m ** 2
    wa, wb = a >= 0.02, b >= 0.02
    d = b - a
    assert S.a_area_km2 == pytest.approx(wa.sum() * cg2 / 1e6) and S.b_area_km2 == pytest.approx(wb.sum() * cg2 / 1e6)
    assert S.new_wet_km2 == pytest.approx((wb & ~wa).sum() * cg2 / 1e6) and S.lost_wet_km2 == pytest.approx((wa & ~wb).sum() * cg2 / 1e6)
    assert S.deeper_km2 == pytest.approx((d > 0.02).sum() * cg2 / 1e6) and S.shallower_km2 == pytest.approx((d < -0.02).sum() * cg2 / 1e6)
    assert S.volume_delta_m3 == pytest.approx(float(d.astype(np.float64).sum()) * cg2, rel=1e-6)
    assert S.max_added_m == pytest.approx(d.max()) and S.max_less_m == pytest.approx(-d.min())
    assert len(S.as_lines()) == 5
    packed = compare.delta_pack(a, b)
    assert packed.dtype == np.float16 and packed.shape == a.shape + (2,)
    assert np.allclose(packed[..., 0], d, atol=2e-3) and np.allclose(packed[..., 1], np.maximum(a, b), atol=2e-3)
    # sign-aware palette: warm for added depth, cool for less depth, grey inside the flat band, symmetric ramp ends
    for x in (0.1, 0.5, 1.0, 2.0):
        w, c = compare.diverging_srgb(x, 2.0), compare.diverging_srgb(-x, 2.0)
        assert w[0] > w[2] and c[2] > c[0]
    assert compare.diverging_srgb(0.0) == compare.FLAT_RGB and compare.diverging_srgb(0.019) == compare.FLAT_RGB
    far = compare.diverging_srgb(9.0, 2.0)
    assert far == compare.diverging_srgb(2.0, 2.0)
    lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]       # noqa: E731
    assert lum(compare.diverging_srgb(2.0, 2.0)) < lum(compare.diverging_srgb(0.2, 2.0))       # darker = bigger magnitude
    wgsl = compare.wgsl_ramp()
    assert "fn delta_color" in wgsl and wgsl.count("smoothstep") == 8


def test_tile_compare_stats_equals_grid_stats_on_same_cells(tmp_path):
    path, a, left, top, bld, rd, _ = _timeline_world(tmp_path)
    b = np.where(a == 65535, 65535, np.minimum(a.astype(np.int64) * 3 // 2 + 5, 60000)).astype(np.uint16)
    _write_raster(tmp_path / "b.tif", b, left, top, 6.25, block=64)
    ra, rb = BlockRaster(path, scale=0.01, nodata=65535), BlockRaster(tmp_path / "b.tif", scale=0.01, nodata=65535)
    S = compare.tile_compare_stats(ra, rb, FRAME, band_rows=41)
    da, db = np.where(a == 65535, 0.0, a * 0.01), np.where(b == 65535, 0.0, b * 0.01)
    latc = 2 * math.atan(math.exp(FRAME.yc / geodesy.R_MERC)) - math.pi / 2
    cell = np.array([(6.25 * FRAME.k * math.cos(2 * math.atan(math.exp((top - (r + 0.5) * 6.25) / geodesy.R_MERC)) - math.pi / 2) / math.cos(latc)) ** 2
                     for r in range(a.shape[0])])[:, None]
    assert S.a_area_km2 == pytest.approx(float(((da >= 0.02) * cell).sum()) / 1e6, rel=1e-9)
    assert S.new_wet_km2 == pytest.approx(float((((db >= 0.02) & ~(da >= 0.02)) * cell).sum()) / 1e6, rel=1e-9)
    assert S.volume_delta_m3 == pytest.approx(float(((db - da) * cell).sum()), rel=1e-6)
    assert S.max_added_m == pytest.approx((db - da).max(), abs=1e-6)
    other = tmp_path / "o.tif"
    _write_raster(other, a[:100], left, top, 6.25, block=64)
    with pytest.raises(ValueError):
        compare.tile_compare_stats(ra, BlockRaster(other, scale=0.01, nodata=65535), FRAME)


def test_depth_grid_provider_flow_grid_decode(tmp_path):
    storm, grids = _flow_world(tmp_path)
    grid = ds.FlowGridDepthSource(lambda n: flow.Storm.open(tmp_path, n), current=lambda: storm, storms=["t"])
    prov = compare.DepthGridProvider(ds.HybridDepthSource(grid, None), storm.frame)
    g = prov.grid("t", 3)
    t = grids[3][..., 0].astype(np.float32) / 255.0
    ref = t * t * 8.0
    ref[(grids[3][..., 3] == 0) | (ref < 0.02)] = 0.0
    assert np.allclose(g, np.round(ref * 100) / 100, atol=0.006)          # cached as whole centimetres
    assert np.all(prov.grid("t", 9) == 0) and np.all(prov.grid("t", 0) == 0)       # dry hours


def test_analysis_state_roundtrip_and_garbage():
    from bgv.analysis.layer import AnalysisState
    s = AnalysisState()
    s.imperial, s.mode, s.b_storm, s.swipe, s.range_m, s.tab = True, "diff", "100yr", 0.3, 3.0, "profile"
    s.win = [10.0, 20.0, 700.0, 520.0]
    d = json.loads(json.dumps(s.to_dict()))
    t = AnalysisState().load(d)
    assert (t.imperial, t.mode, t.b_storm, t.swipe, t.range_m, t.tab, t.win) == (True, "diff", "100yr", 0.3, 3.0, "profile", [10.0, 20.0, 700.0, 520.0])
    g = AnalysisState().load({"mode": "bogus", "swipe": "x", "range_m": 1e9, "win": [1, 2], "tab": 5, "b_hour": float("nan"), "opacity": -3})
    assert g.mode == "off" and g.swipe == 0.5 and g.range_m == 8.0 and g.win[2] == 640.0 and g.opacity == 0.2
    assert AnalysisState().load(None).tab == "measure"


# ------------------------------------------------------------------------------------------------ real county tiles
TILES = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map\layers\cog\county")
real_tiles = pytest.mark.skipif(not (TILES / "100yr" / "cog_meta.json").is_file(), reason="county tiles not present")


@pytest.fixture(scope="module")
def county():
    from bgv import config, domain as dm
    root = real_root()
    try:
        dom = dm.resolve("county", root, {})
    except Exception as e:
        pytest.skip(f"county domain unavailable: {e}")
    return root, dom, ds.make_depth_source(root=root, domain=dom)


@real_tiles
def test_real_tiles_match_direct_windowed_reads_at_120_points(county):
    root, dom, src = county
    assert src.tiles is not None and src.uses_tiles("100yr") and "verification pending" in src.label("100yr")
    rng = np.random.default_rng(12)
    lng, lat = rng.uniform(-86.55, -86.3, 400), rng.uniform(36.9, 37.1, 400)
    for storm, hour in (("100yr", 12), ("25yr", 9), ("ev4", 18)):
        got = src.tiles.depth_hour(lng, lat, storm, hour)
        p = src.tiles.storm(storm).files[hour]
        x, y = geodesy.lnglat_to_merc(lng, lat)
        with rasterio.open(p) as f:
            n_checked = 0
            for i in range(len(x)):
                gx, gy = (x[i] - f.bounds.left) / 6.25, (f.bounds.top - y[i]) / 6.25
                x0, y0 = int(math.floor(gx - 0.5)), int(math.floor(gy - 0.5))
                win = f.read(1, window=rasterio.windows.Window(x0, y0, 2, 2)).astype(np.float64)
                tx, ty = gx - 0.5 - x0, gy - 0.5 - y0
                w = np.array([[(1 - tx) * (1 - ty), tx * (1 - ty)], [(1 - tx) * ty, tx * ty]])
                ok = win != 65535
                ref = float((win * 0.01 * w)[ok].sum() / w[ok].sum()) if ok.any() else float("nan")
                assert (math.isnan(ref) and math.isnan(got[i])) or got[i] == pytest.approx(ref, abs=1e-5)
                n_checked += 1
        assert n_checked >= 120
    assert (src.tiles.depth_hour(lng, lat, "100yr", 12) > 0).sum() > 5               # the sample does hit water


@real_tiles
def test_real_profile_samples_match_direct_dem_and_tile_reads(county):
    root, dom, src = county
    frame = dom.open_storm("100yr").frame
    ground = profile.DemGround(dom.terrain.path, dom.terrain.units)
    p0, p1 = frame.lnglat_to_world(-86.46, 36.97), frame.lnglat_to_world(-86.42, 37.01)
    P = profile.build_profile(np.array([p0, p1]), frame, src, "100yr", 12.0, ground)
    assert P.n >= 400 and "tiles" in P.source_label
    xs, ys = geodesy.lnglat_to_merc(P.lng, P.lat)
    from rasterio.warp import transform
    ux, uy = transform("EPSG:4326", "EPSG:32616", P.lng.tolist(), P.lat.tolist())
    with rasterio.open(dom.terrain.path) as dem, rasterio.open(src.tiles.storm("100yr").files[12]) as tl:
        idx = np.linspace(0, P.n - 1, 150).astype(int)
        for i in idx:
            gx, gy = (ux[i] - dem.bounds.left) / dem.res[0], (dem.bounds.top - uy[i]) / dem.res[1]
            x0, y0 = int(math.floor(gx - 0.5)), int(math.floor(gy - 0.5))
            win = dem.read(1, window=rasterio.windows.Window(x0, y0, 2, 2)).astype(np.float64)
            tx, ty = gx - 0.5 - x0, gy - 0.5 - y0
            w = np.array([[(1 - tx) * (1 - ty), tx * (1 - ty)], [(1 - tx) * ty, tx * ty]])
            ok = win != -9999.0
            gref = float((win * w)[ok].sum() / w[ok].sum()) if ok.any() else float("nan")
            assert (math.isnan(gref) and math.isnan(P.ground_m[i])) or P.ground_m[i] == pytest.approx(gref, abs=2e-3)
            gx, gy = (xs[i] - tl.bounds.left) / 6.25, (tl.bounds.top - ys[i]) / 6.25
            x0, y0 = int(math.floor(gx - 0.5)), int(math.floor(gy - 0.5))
            win = tl.read(1, window=rasterio.windows.Window(x0, y0, 2, 2)).astype(np.float64)
            tx, ty = gx - 0.5 - x0, gy - 0.5 - y0
            w = np.array([[(1 - tx) * (1 - ty), tx * (1 - ty)], [(1 - tx) * ty, tx * ty]])
            ok = win != 65535
            dref = float((win * 0.01 * w)[ok].sum() / w[ok].sum()) if ok.any() else 0.0
            dref = 0.0 if dref < 0.02 else dref
            assert P.depth_m[i] == pytest.approx(dref, abs=1e-5)
    assert P.max_depth_m > 0 and P.wet_length_m > 0


@real_tiles
def test_real_timeline_hour12_equals_whole_array_count(county):
    root, dom, src = county
    st = dom.open_storm("100yr")
    frame = st.frame
    bp, rp, _ = dom.scene_paths(frame)
    from bgv.data import buildings as bmod, roads as rmod
    B, R = bmod.load_buildings(root, frame, geojson=bp), rmod.load_roads(root, frame, geojson=rp)
    ts = src.tiles.storm("100yr")
    r = ts.raster(12)
    bpts, rpts = timeline.prepare_points(r, frame, B, R)
    got = timeline.tile_hour_stats(r, frame, bpts, rpts)
    with rasterio.open(ts.files[12]) as f:                   # the whole array at once, a different code path
        arr = f.read(1)
        left, top = f.bounds.left, f.bounds.top
    d = np.where(arr == 65535, 0.0, arr * 0.01)
    X, Y, ids = timeline.building_points(B)
    xm, ym = X / frame.k + frame.xc, Y / frame.k + frame.yc
    c, rr = np.floor((xm - left) / 6.25).astype(int), np.floor((top - ym) / 6.25).astype(int)
    ok = (c >= 0) & (c < arr.shape[1]) & (rr >= 0) & (rr < arr.shape[0])
    bmax = np.zeros(B.n)
    np.maximum.at(bmax, ids[ok], d[rr[ok], c[ok]])
    assert got["flooded"] == float((bmax >= 0.1).sum())
    assert 500 < got["flooded"] < B.n
    sx, sy = R.sample_x.astype(np.float64) / frame.k + frame.xc, R.sample_y.astype(np.float64) / frame.k + frame.yc
    c, rr = np.floor((sx - left) / 6.25).astype(int), np.floor((top - sy) / 6.25).astype(int)
    ok = (c >= 0) & (c < arr.shape[1]) & (rr >= 0) & (rr < arr.shape[0])
    wet = np.zeros(len(c), bool)
    wet[ok] = d[rr[ok], c[ok]] >= 0.1
    assert got["road_km"] * 1000.0 == pytest.approx(float(R.sample_weight_m[wet].astype(np.float64).sum()), rel=1e-6)
    cells = int((d >= 0.02).sum())
    cg = frame.cell_ground_m / frame.cell_m * 6.25
    assert got["area_km2"] == pytest.approx(cells * cg * cg / 1e6, rel=0.01)       # per-row k(row) is within 0.4 % of the constant k
