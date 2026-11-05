"""Native solver tiles: the COG reader against GDAL's decode (fixture written by scripts/make_cog_fixture.py), the tile service
(statuses, labels, fallbacks), the footprint maximum (all-touched, vs rasterio) and the tile-vs-grid comparison numbers.
No GPU, no real data folder needed (the fixture is in the repo: cog/test/fixture)."""
from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from bgv.data import cog as cogmod
from bgv.data import tiles
from bgv.geo import GridFrame

FIX = Path(__file__).resolve().parents[2] / "cog" / "test" / "fixture"
pytestmark = pytest.mark.skipif(not (FIX / "a_direct.tif").is_file(), reason="COG fixture missing (scripts/make_cog_fixture.py)")


@pytest.fixture(scope="module")
def expected():
    return json.loads((FIX / "expected.json").read_text())["files"]["a_direct.tif"]


@pytest.fixture()
def cog():
    c = cogmod.open_cog(FIX / "a_direct.tif")
    yield c
    c.close()


def test_geometry_and_levels(cog, expected):
    assert (cog.width, cog.height) == (expected["width"], expected["height"])
    assert [(L.width, L.height) for L in cog.levels] == [tuple(s) for s in expected["level_sizes"]]
    assert cog.res == pytest.approx(expected["res"])
    assert (cog.x0, cog.y0) == pytest.approx(tuple(expected["origin"]))
    assert cog.nodata == 65535


def test_every_level_equals_gdal(cog, expected):
    n = 0
    for p in expected["points"]:
        for lv, v in enumerate(p["v"]):
            st, raw = cog.sample_merc(p["x"], p["y"], lv)
            if v is None:
                continue
            assert st in ("value", "nodata") and raw == v, (p, lv)
            n += 1
    assert n >= 300


def test_vectorised_matches_scalar(cog, expected):
    xs = np.array([p["x"] for p in expected["points"]])
    ys = np.array([p["y"] for p in expected["points"]])
    raw, st = cog.sample_many(xs, ys)
    for i, p in enumerate(expected["points"]):
        if p["v"][0] is None:
            assert st[i] == cogmod.ST_OUTSIDE
        else:
            assert int(raw[i]) == p["v"][0]
    far, st2 = cog.sample_many([xs[0] + 1e7], [ys[0]])
    assert st2[0] == cogmod.ST_OUTSIDE


def test_dry_and_nodata_are_different(cog):
    c, r = 360, 230                                    # inside the dry block (rows 200..260, cols 300..420)
    x, y = cog.merc_of_pixel_centre(c, r)
    assert cog.sample_merc(float(x), float(y)) == ("value", 0)
    c, r = 950, 730                                    # inside the nodata block
    x, y = cog.merc_of_pixel_centre(c, r)
    assert cog.sample_merc(float(x), float(y))[0] == "nodata"
    assert cog.sample_merc(cog.x0 - 100.0, cog.y0)[0] == "outside"
    assert cogmod.depth_fields(0, "value")[0] == "dry"
    assert cogmod.depth_fields(None, "nodata")[0] == "nodata"
    assert cogmod.depth_fields(123, "value")[1:3] == (123.0, 1.23)


def test_read_window_matches_pixels(cog):
    w = cog.read_window(0, 290, 190, 430, 270)
    assert w.shape == (80, 140)
    assert (w[10:70, 10:130] == 0).all()               # the dry block
    out = cog.read_window(0, -5, -5, 3, 3)             # partly outside: nodata fill
    assert out[0, 0] == 65535 and out.shape == (8, 8)


def test_lnglat_mercator_roundtrip_vs_proj(expected):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.warp import transform
    rng = np.random.default_rng(3)
    lng, lat = rng.uniform(-87, -86, 50), rng.uniform(36.5, 37.5, 50)
    xr, yr = transform("EPSG:4326", "EPSG:3857", list(lng), list(lat))
    x, y = cogmod.lnglat_to_merc(lng, lat)
    assert np.abs(x - xr).max() < 1e-4 and np.abs(y - yr).max() < 1e-4
    l2, a2 = cogmod.merc_to_lnglat(x, y)
    assert np.abs(l2 - lng).max() < 1e-9 and np.abs(a2 - lat).max() < 1e-9


def test_tile_cache_is_lru_and_byte_capped(cog):
    cache = cogmod.TileCache(max_bytes=3 * 512 * 512 * 2)
    for i in range(6):
        cache.put(("k", i), np.zeros((512, 512), np.uint16))
    s = cache.stats()
    assert s["tiles"] == 3 and cache.get(("k", 0)) is None and cache.get(("k", 5)) is not None
    cache.get(("k", 3))
    cache.put(("k", 9), np.zeros((512, 512), np.uint16))
    assert cache.get(("k", 3)) is not None and cache.get(("k", 4)) is None      # 4 was the least recently used


def test_hour_files_nearest_save_time():
    meta = {"files": {n: {} for n in ("max_depth.tif", "depth_000015s.tif", "depth_003604s.tif", "depth_007200s.tif", "depth_086400s.tif")}}
    hf = cogmod.hour_files(meta)
    assert hf == {0: "depth_000015s.tif", 1: "depth_003604s.tif", 2: "depth_007200s.tif", 24: "depth_086400s.tif"}


# ------------------------------------------------------------------------------------------------ service
def make_root(tmp_path, scope="county", storm="t1", meta_extra=None):
    folder = tmp_path / "layers" / "cog" / scope / storm
    folder.mkdir(parents=True)
    for name in ("max_depth.tif", "depth_000015s.tif", "depth_043200s.tif"):
        shutil.copy(FIX / "a_direct.tif", folder / name)
    meta = {"crs": "EPSG:3857", "units": "cm", "res_m": 6.25,
            "files": {n: {"width": 1300, "height": 1100} for n in ("max_depth.tif", "depth_000015s.tif", "depth_043200s.tif")}}
    meta.update(meta_extra or {})
    (folder / "cog_meta.json").write_text(json.dumps(meta))
    return tmp_path


def lnglat_of_pixel(cog, c, r):
    x, y = cog.merc_of_pixel_centre(c, r)
    lng, lat = cogmod.merc_to_lnglat(float(x), float(y))
    return float(lng), float(lat)


def test_service_statuses_labels_and_hours(tmp_path, cog):
    root = make_root(tmp_path)
    svc = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {})
    ts = svc.tileset("t1")
    assert ts is not None and ts.scope == "county" and sorted(ts.hours) == [0, 12]
    lng, lat = lnglat_of_pixel(cog, 360, 230)
    s = svc.depth_lnglat("t1", lng, lat, "peak")
    assert (s.status, s.cm, s.m, s.source, s.scope) == ("dry", 0.0, 0.0, "tiles", "county")
    assert "verification pending" in s.label and "6.25 m" in s.label
    lng, lat = lnglat_of_pixel(cog, 950, 730)
    assert svc.depth_lnglat("t1", lng, lat, 12).status == "nodata"
    lng, lat = lnglat_of_pixel(cog, 100, 100)
    d = svc.depth_lnglat("t1", lng, lat, 12)
    st, raw = cog.sample_merc(*[float(v) for v in cog.merc_of_pixel_centre(100, 100)])
    assert d.status == "depth" and d.cm == raw and d.hour == 12 and d.ft == pytest.approx(raw / 100 * 3.28084)
    assert svc.depth_lnglat("t1", lng, lat, 11).hour == 12 or svc.depth_lnglat("t1", lng, lat, 11).hour == 0   # nearest file within an hour
    assert svc.depth_lnglat("t1", lng, lat, 5).status == "unavailable"                                          # no file near hour 5
    assert svc.depth_lnglat("nope", lng, lat, "peak").status == "unavailable"
    prof = svc.depth_profile("t1", np.array([lng, lng]), np.array([lat, lat + 5.0]), "peak")
    assert prof[0] == pytest.approx(raw / 100) and np.isnan(prof[1])


def test_label_after_verification_and_native(tmp_path):
    root = make_root(tmp_path, meta_extra={"verified": True})
    ts = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {}).tileset("t1")
    assert "verification pending" not in ts.label and "tiles, 6.25 m" in ts.label
    root2 = make_root(tmp_path / "n", meta_extra={"res_m": 5.0})
    ts2 = tiles.TileService(root2, types.SimpleNamespace(key="county", demo=False), {"tiles_verified": True}).tileset("t1")
    assert "native tiles" in ts2.label and "pending" not in ts2.label


def test_city_falls_back_to_county_run_and_folder_switch(tmp_path):
    root = make_root(tmp_path)
    city = types.SimpleNamespace(key="city", demo=False)
    ts = tiles.TileService(root, city, {}).tileset("t1")
    assert ts.scope == "county"                                       # labelled as the county run, never as a city run
    assert tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {"cog_folder": "cog_native"}).tileset("t1") is None
    shutil.move(str(root / "layers" / "cog"), str(root / "layers" / "cog_native"))
    assert tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {"cog_folder": "cog_native"}).tileset("t1") is not None


def test_disabled_for_demo_and_by_setting(tmp_path):
    root = make_root(tmp_path)
    demo = tiles.TileService(root, types.SimpleNamespace(key="county", demo=True), {})
    assert not demo.enabled and demo.tileset("t1") is None and "synthetic" in demo.reason
    off = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {"tiles_enabled": False})
    assert off.tileset("t1") is None
    assert off.depth_lnglat("t1", -86.5, 37.0, 12).status == "unavailable"


def test_damaged_and_missing_files_are_reported_not_raised(tmp_path, cog):
    root = make_root(tmp_path)
    folder = root / "layers" / "cog" / "county" / "t1"
    (folder / "depth_043200s.tif").write_bytes(b"not a tiff at all")
    svc = tiles.TileService(root, types.SimpleNamespace(key="county", demo=False), {})
    lng, lat = lnglat_of_pixel(cog, 100, 100)
    r = svc.depth_lnglat("t1", lng, lat, 12)
    assert r.status == "unavailable" and "missing" in r.note
    (folder / "max_depth.tif").unlink()
    svc.refresh()
    assert svc.depth_lnglat("t1", lng, lat, "peak").status == "unavailable"
    (folder / "cog_meta.json").write_text("{broken")
    svc.refresh()
    assert svc.tileset("t1") is None


# ------------------------------------------------------------------------------------------------ footprints
def _features(expected_all):
    return [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [f["ring"]]}} for f in expected_all]


def test_footprint_max_equals_rasterio_all_touched(cog):
    from bgv.data import buildings as bmod
    fp = json.loads((FIX / "expected.json").read_text())["footprints"]
    W, H, res = cog.width, cog.height, cog.res
    frame = GridFrame(W, H, res, res, cog.x0, cog.y0 - H * res, cog.x0 + W * res, cog.y0)
    idx = bmod.buildings_from_features(_features(fp), frame, index_only=True)
    assert idx.n == len(fp)
    got = tiles.footprint_max(cog, idx)
    for i, f in enumerate(fp):
        if f["touched"] is None or f["touched"]["max"] is None:
            assert got[i] == -1 or f["touched"] is None               # outside the raster / all no-data: no value
            continue
        assert idx.pixels_of(i).size >= f["center"]["n"]               # all-touched covers at least the centre-inside cells
        # the web viewer's rule (centre inside + outline touches) is rasterio's all_touched=True: same maximum
        assert got[i] == f["touched"]["max"], (i, got[i], f)


# ------------------------------------------------------------------------------------------------ comparison numbers
def test_compare_numbers_overlap_and_tiers():
    from bgv.ui.tilepanel import compare_numbers
    bs = types.SimpleNamespace(flooded=np.array([1, 1, 0, 0, 1], bool), tier=np.array([1, 2, 0, 0, 4]),
                               depth_m=np.array([0.2, 0.4, 0.0, 0.01, 2.0]))
    cm = np.array([15, 5, 40, -1, 130], np.int32)                       # tiles: flooded, dry-ish, flooded(shallow..), no data, severe
    r = compare_numbers({"buildings": bs}, cm, np.ones(5, bool))
    assert r["n"] == 4 and r["nodata"] == 1
    assert (r["grid_flooded"], r["tile_flooded"]) == (3, 3)             # building 3 (no tile data) is left out of both
    assert (r["both"], r["grid_only"], r["tile_only"]) == (2, 1, 1)
    assert list(r["tile_tiers"]) == [1, 1, 0, 1]
    assert tiles.tier_counts_cm(cm)["flooded"] == 3
