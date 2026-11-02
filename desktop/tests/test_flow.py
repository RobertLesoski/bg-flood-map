"""Data layer: decode laws, hour clamp, frame/pixel centres, gzip validation, terrain NaN + feet->m."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from bgv import config
from bgv.data import flow, terrain
from bgv.geo import GridFrame


def make_meta(w=8, h=6, cell=10.0, hours=(2, 3)):
    xmin, ymax = -9_600_000.0, 4_440_000.0
    return {"format": "bgflow", "version": 1, "storm": "t", "label": "t", "source": "tiers", "crs": "EPSG:3857",
            "width": w, "height": h, "cell_m": cell, "cell_ground_m": cell * 0.8,
            "bounds_merc": [xmin, ymax - h * cell, xmin + w * cell, ymax], "hours": list(hours),
            "depth_encoding": {"type": "sqrt", "max_m": 8.0}, "vel_encoding": {"vmax_ms": 4.0, "zero_code": 128}}


def write_storm(root, meta, grids):
    d = root / "layers" / "flow" / meta["storm"]
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    for h, g in grids.items():
        (d / f"h{h:02d}.rgba.gz").write_bytes(gzip.compress(g.tobytes(), mtime=0))
    return d


@pytest.fixture
def storm(tmp_path):
    meta = make_meta()
    g2 = np.zeros((6, 8, 4), np.uint8)
    g2[..., 1:3] = 128
    g2[..., 3] = 255
    g2[2, 3] = (flow.encode_depth(1.0), 128 + 127, 128, 255)       # 1 m, u = +4 m/s east
    g3 = g2.copy()
    g3[2, 3] = (flow.encode_depth(2.0), 128, 128 + 127, 255)       # 2 m, v = +4 m/s north
    write_storm(tmp_path, meta, {2: g2, 3: g3})
    return flow.Storm.open(tmp_path, "t"), g2, g3


# ------------------------------------------------------------------------------------------- decode laws
def test_depth_law():
    assert flow.decode_depth(0) == 0.0
    assert flow.decode_depth(255) == pytest.approx(8.0)
    assert flow.decode_depth(128) == pytest.approx((128 / 255) ** 2 * 8.0, rel=1e-6)
    r = np.arange(256, dtype=np.uint8)
    d = flow.decode_depth(r)
    assert np.all(np.diff(d) > 0)
    assert np.array_equal(flow.encode_depth(d), r)          # encode(decode) is the identity on codes
    assert flow.encode_depth(20.0) == 255 and flow.encode_depth(-1.0) == 0


def test_vel_law():
    assert flow.decode_vel(128) == 0.0
    assert flow.decode_vel(255) == pytest.approx(4.0)
    assert flow.decode_vel(1) == pytest.approx(-4.0)
    assert flow.decode_vel(0) == pytest.approx(-128 / 127 * 4.0)    # the zero grid read naively: why R==0 gates it
    assert flow.decode_q(255) == pytest.approx(2.0)


def test_hour_pair_clamp():
    assert flow.hour_pair(12.25) == (12, 13, pytest.approx(0.25))
    assert flow.hour_pair(0.0) == (0, 1, 0.0)
    assert flow.hour_pair(-3.0) == (0, 1, 0.0)
    assert flow.hour_pair(24.0) == (24, 24, 0.0)
    assert flow.hour_pair(99.0) == (24, 24, 0.0)
    h0, h1, m = flow.hour_pair(23.999)
    assert (h0, h1) == (23, 24) and 0.99 < m < 1.0


# ------------------------------------------------------------------------------------- frame / pixel centres
def test_pixel_centres_and_uv():
    f = GridFrame.from_meta(make_meta(w=8, h=6, cell=10.0))
    for col, row in [(0, 0), (7, 5), (3, 2)]:
        x = f.xmin + (col + 0.5) * f.cell_m
        y = f.ymax - (row + 0.5) * f.cell_m
        X, Y = f.merc_to_world(x, y)
        u, v = f.world_to_uv(X, Y)
        assert u == pytest.approx((col + 0.5) / 8) and v == pytest.approx((row + 0.5) / 6)
        c, r = f.world_to_pixel(X, Y)
        assert (c, r) == (pytest.approx(col + 0.5), pytest.approx(row + 0.5))
    assert f.world_to_uv(0, 0) == (pytest.approx(0.5), pytest.approx(0.5))
    assert f.world_to_uv(0, 1.0)[1] < 0.5                       # +Y north = smaller row
    sx, sy = f.size_m
    assert sx == pytest.approx(8 * f.cell_ground_m) and sy == pytest.approx(6 * f.cell_ground_m)


# ------------------------------------------------------------------------------------------------- storm
def test_open_meta_and_missing(storm):
    s, g2, g3 = storm
    assert s.hours == [2, 3] and s.source == "tiers" and (s.width, s.height) == (8, 6)
    assert flow.available_storms(s.root) == ["t"]
    with pytest.raises(flow.StormNotFoundError):
        flow.Storm.open(s.root, "nope")


def test_load_hour_and_zero(storm):
    s, g2, g3 = storm
    a = s.load_hour(2)
    assert a.shape == (6, 8, 4) and a.dtype == np.uint8 and np.array_equal(a, g2)
    z1, z2 = s.load_hour(1), s.load_hour(24)
    assert z1 is z2 and not z1.flags.writeable and not z1.any()
    assert s.is_zero(1) and s.is_zero(0) and s.is_zero(24) and not s.is_zero(2)


def test_prefetch(storm):
    s, g2, g3 = storm
    futs = s.prefetch([2, 3, 9])
    assert len(futs) == 2
    for f in futs:
        f.result(timeout=10)
    assert sorted(s.cached_hours()) == [2, 3]


def test_size_validation(tmp_path):
    d = write_storm(tmp_path, make_meta(), {2: np.zeros((6, 8, 4), np.uint8)})
    (d / "h03.rgba.gz").write_bytes(gzip.compress(bytes(10)))
    s = flow.Storm.open(tmp_path, "t")
    with pytest.raises(flow.FlowFormatError):
        s.load_hour(3)
    (d / "h02.rgba.gz").write_bytes(b"not gzip at all, really not gzip at all")
    with pytest.raises(flow.FlowFormatError):
        s.load_hour(2)


@pytest.mark.parametrize("patch", [{"format": "x"}, {"version": 2}, {"width": 0}, {"width": 99999},
                                   {"hours": [0]}, {"hours": [25]}, {"bounds_merc": [1, 2, 3]},
                                   {"vel_encoding": {"vmax_ms": 4.0, "zero_code": 100}}])
def test_meta_validation(patch):
    meta = make_meta()
    meta.update(patch)
    with pytest.raises(flow.FlowFormatError):
        flow.validate_meta(meta)


def test_sample_blend_after_decode(storm):
    s, g2, g3 = storm
    f = s.frame
    x = f.xmin + 3.5 * f.cell_m                    # centre of pixel (3, 2)
    y = f.ymax - 2.5 * f.cell_m
    a, b = s.load_hour(2), s.load_hour(3)
    p = s.sample(a, b, 0.0, x, y)
    assert p["valid"] and p["depth"] == pytest.approx(1.0, abs=0.02) and p["u"] == pytest.approx(4.0)
    assert p["v"] == pytest.approx(0.0, abs=1e-6)
    p = s.sample(a, b, 1.0, x, y)
    assert p["depth"] == pytest.approx(2.0, abs=0.02) and p["v"] == pytest.approx(4.0)
    assert p["u"] == pytest.approx(0, abs=1e-6)
    p = s.sample(a, b, 0.5, x, y)                  # blend AFTER decode: mean of decoded depths
    da, db = flow.decode_depth(g2[2, 3, 0]), flow.decode_depth(g3[2, 3, 0])
    assert p["depth"] == pytest.approx(0.5 * (da + db), rel=1e-5)
    assert p["u"] == pytest.approx(2.0) and p["v"] == pytest.approx(2.0)
    p = s.sample(a, None, 0.0, x + 0.5 * f.cell_m, y)   # half way between pixel centres: code-space average
    code = 0.5 * g2[2, 3, 0]
    assert p["depth"] == pytest.approx((code / 255) ** 2 * 8.0, rel=1e-5)
    assert not s.sample(a, b, 0.0, f.xmin - 1, y)["valid"]
    assert s.sample_t(99.0, 0.0, 0.0)["depth"] == 0.0


def test_sample_dry_grid_has_zero_velocity(storm):
    s, _, _ = storm
    z = s.load_hour(1)                              # all-zero grid, A = 0
    p = s.sample(z, z, 0.3, s.frame.xmin + 5, s.frame.ymax - 5)
    assert p["u"] == 0 and p["v"] == 0 and p["depth"] == 0 and not p["valid"]
    a = s.load_hour(2)                              # in-domain but dry: valid, depth 0, velocity 0
    p = s.sample(a, z, 0.5, s.frame.xmin + 5, s.frame.ymax - 5)
    assert p["valid"] and p["depth"] == 0 and p["speed"] == 0


def test_block_max():
    a = np.arange(35, dtype=np.uint8).reshape(5, 7)
    m = flow.block_max(a, 4)
    assert m.shape == (2, 2) and m[0, 0] == a[:4, :4].max() and m[1, 1] == a[4:, 4:].max()


# ----------------------------------------------------------------------------------------------- terrain
def test_dem_feet_to_metres_and_nan():
    raw = np.array([[541.0, -9999.0], [np.nan, 1000.0]], np.float32)
    z = terrain.dem_to_metres(raw)
    assert z[0, 0] == pytest.approx(541.0 * 0.3048, rel=1e-6)
    assert z[1, 1] == pytest.approx(304.8, rel=1e-6)
    assert np.isnan(z[0, 1]) and np.isnan(z[1, 0])
    assert z.dtype == np.float32


def test_fill_nan_and_stats():
    z = np.full((5, 6), 100.0, np.float32)
    z[0, 0] = np.nan
    z[2:4, 3:5] = np.nan
    z[4, 5] = 7.0
    st = terrain.terrain_stats(z)
    assert st["nan_count"] == 5 and st["min"] == 7.0 and st["max"] == 100.0
    f, valid = terrain.fill_nan(z)
    assert np.isfinite(f).all() and valid.sum() == z.size - 5
    assert f[0, 0] == 100.0
    assert np.array_equal(f[valid], z[valid])
    f2, v2 = terrain.fill_nan(np.full((3, 3), np.nan, np.float32))
    assert (f2 == 0).all() and not v2.any()


def test_slope_map_orientation():
    H, W = 8, 8
    col = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)       # rises toward EAST (+col): gx > 0, gy = 0
    nm = terrain.build_normal_map(col * 1.0, np.ones((H, W), bool), 1.0)
    assert nm.dtype == np.float16
    assert nm[4, 4, 0] == pytest.approx(1.0) and abs(float(nm[4, 4, 1])) < 1e-3 and nm[4, 4, 3] == 1
    row = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)       # rises toward SOUTH (+row): gy < 0 (north slope)
    nm = terrain.build_normal_map(row * 2.0, np.ones((H, W), bool), 1.0)
    assert nm[4, 4, 1] == pytest.approx(-2.0) and abs(float(nm[4, 4, 0])) < 1e-3
    bowl = ((np.arange(W) - 3.5) ** 2)[None, :] + ((np.arange(H) - 3.5) ** 2)[:, None]
    nm = terrain.build_normal_map(bowl.astype(np.float32), np.ones((H, W), bool), 1.0)
    assert nm[4, 4, 2] > 0                                            # hollow: positive curvature
    chain = terrain.normal_mip_chain(nm)
    assert chain[0].shape == (8, 8, 4) and chain[-1].shape == (1, 1, 4) and len(chain) == 4


REAL = Path(config.DEFAULT_DATA_ROOT)


@pytest.mark.skipif(not (REAL / "dem_bg.tif").is_file(), reason="real data not present")
def test_real_terrain_and_storm():
    root = config.data_root()
    s = flow.Storm.open(root, "25yr")
    z = terrain.load_terrain(root, s.frame)
    assert z.shape == (s.height, s.width) and z.dtype == np.float32
    st = terrain.terrain_stats(z)
    # Bowling Green is ~165 m; the file is in FEET (mean 541 ft) - read as metres it would give > 400 m
    assert 100 < st["min"] < 140 and 220 < st["max"] < 260 and 150 < st["mean"] < 180
    assert 0.005 < st["nan_fraction"] < 0.1
    a = s.load_hour(12)
    assert a.shape == (s.height, s.width, 4)
    wet = a[..., 0] > 0
    assert wet.sum() > 0
    d = flow.decode_depth(a[..., 0])
    area = float((d >= 0.02).sum()) * s.frame.cell_ground_m ** 2 / 1e6
    assert area == pytest.approx(14.06, abs=0.15)           # known-good number from DESIGN.md
    assert d.max() == pytest.approx(5.90, abs=0.1)
    assert not (wet & ~np.isfinite(z)).any()                # no water where there is no terrain
