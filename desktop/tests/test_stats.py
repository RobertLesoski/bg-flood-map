"""Tests for bgv.data.stats: hand-made grids + the known-good numbers of DESIGN.md on the real data."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from data_helpers import (blank_grid, depth_to_code, polygon, real_buildings, real_frame, real_hour, real_roads, tiny_frame,
                          rect)
from bgv.data import buildings as B
from bgv.data import stats as ST


def make(fr, feats):
    return B.buildings_from_features(feats, fr)


def one_pixel_buildings(fr, cells):
    """One sub-pixel building at the centre of each (col,row) in cells."""
    return make(fr, [polygon(fr, rect(c + 0.4, r + 0.4, c + 0.6, r + 0.6), height_m=5) for c, r in cells])


# ------------------------------------------------------------------------------------------------ decode / constants
def test_decode_laws():
    assert ST.decode_depth(0) == 0 and ST.decode_depth(255) == 8.0
    assert ST.decode_depth(100) == pytest.approx((100 / 255) ** 2 * 8)
    assert ST.decode_velocity(128) == 0 and ST.decode_velocity(255) == pytest.approx(4.0)
    assert ST.decode_velocity(1) == pytest.approx(-127 / 127 * 4)
    assert ST.min_code_for(0.1) == 29 and ST.min_code_for(0.02) == 13 and ST.min_code_for(99) == 256


# ------------------------------------------------------------------------------------------------ buildings
@pytest.mark.parametrize("code,tier,flooded", [(28, 0, False), (29, 1, True), (49, 1, True), (50, 2, True), (69, 2, True),
                                               (70, 3, True), (98, 3, True), (99, 4, True), (255, 4, True), (0, 0, False)])
def test_tier_boundaries(code, tier, flooded):
    fr = tiny_frame()
    b = one_pixel_buildings(fr, [(5, 5)])
    g = blank_grid(fr)
    g[5, 5, 0] = code
    s = ST.evaluate_buildings(g, b)
    assert s.depth_m[0] == pytest.approx((code / 255) ** 2 * 8, abs=1e-6)
    assert bool(s.flooded[0]) == flooded and s.tier[0] == (tier if flooded else (tier))
    assert s.flooded_count == int(flooded)
    assert s.tier_counts.sum() == int(flooded and tier > 0)
    if flooded:
        assert s.tier_counts[tier - 1] == 1
    assert s.below_tier == 0


def test_max_over_pixels_and_all_touched_semantics():
    fr = tiny_frame()
    # one building covering pixels (4..6, 4..5): max over them
    b = make(fr, [polygon(fr, rect(4.2, 4.2, 6.8, 5.8), height_m=5)])
    g = blank_grid(fr)
    g[4, 4, 0] = 40
    g[5, 6, 0] = 120
    g[3, 6, 0] = 255                                       # outside the footprint, ignored
    s = ST.evaluate_buildings(g, b)
    assert s.depth_m[0] == pytest.approx((120 / 255) ** 2 * 8, abs=1e-6)


def test_a0_pixels_are_ignored():
    fr = tiny_frame()
    b = make(fr, [polygon(fr, rect(4.2, 4.2, 5.8, 4.8), height_m=5)])      # pixels (4,4) and (5,4)
    g = blank_grid(fr)
    g[4, 4, 0] = 200
    g[4, 4, 3] = 0                                         # invalid domain, code must not count
    g[4, 5, 0] = 60
    s = ST.evaluate_buildings(g, b)
    assert s.depth_m[0] == pytest.approx((60 / 255) ** 2 * 8, abs=1e-6)
    g[4, 5, 3] = 0
    assert ST.evaluate_buildings(g, b).depth_m[0] == 0
    assert ST.evaluate_buildings(g, b).flooded_count == 0


def test_outside_grid_building_is_dry():
    fr = tiny_frame()
    b = make(fr, [polygon(fr, rect(40, 40, 42, 42)), polygon(fr, rect(5.2, 5.2, 5.8, 5.8))])
    g = blank_grid(fr)
    g[:, :, 0] = 200
    s = ST.evaluate_buildings(g, b)
    assert s.depth_m[0] == 0 and not s.flooded[0] and s.flooded[1]
    assert list(s.order) == [1]


def test_zero_grid_fast_path():
    fr = tiny_frame()
    b = one_pixel_buildings(fr, [(2, 2), (3, 3)])
    for z in (None, SimpleNamespace(isZero=True)):
        s = ST.evaluate_buildings(z, b)
        assert s.flooded_count == 0 and s.depth_m.shape == (2,) and s.depth_m.dtype == np.float32
        assert s.order.shape == (0,) and s.order.dtype == np.int32 and s.tier_counts.tolist() == [0, 0, 0, 0]
        assert s.speed_ms.shape == (2,) and not s.flooded.any()
    g0 = blank_grid(fr)                                    # an all-dry real array
    assert ST.evaluate_buildings(g0, b).flooded_count == 0
    sm = ST.summarize_grid(None, fr)
    assert sm.wet_cells == 0 and sm.wet_area_km2 == 0 and sm.max_depth_m == 0 and sm.volume_m3 == 0
    sm = ST.summarize_grid(g0, fr)
    assert sm.wet_cells == 0 and sm.max_speed_ms == 0


def test_ranking_deepest_first_ties_by_id():
    fr = tiny_frame()
    cells = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    b = one_pixel_buildings(fr, cells)
    g = blank_grid(fr)
    codes = [80, 120, 80, 0, 120, 255]
    for (c, r), v in zip(cells, codes):
        g[r, c, 0] = v
    s = ST.evaluate_buildings(g, b)
    assert list(s.order) == [5, 1, 4, 0, 2]                # 255, 120(id1), 120(id4), 80(id0), 80(id2); id3 dry
    assert s.flooded_count == 5
    assert s.tier_counts.tolist() == [0, 0, 2, 3] and s.tier_dict() == {"shallow": 0, "moderate": 0, "deep": 2, "severe": 3}


def test_below_tier_count_with_lower_min_depth():
    fr = tiny_frame()
    b = one_pixel_buildings(fr, [(1, 1), (2, 1), (3, 1)])
    g = blank_grid(fr)
    g[1, 1, 0] = 17        # 0.0356 m: flooded at min 0.02 but below the first tier
    g[1, 2, 0] = 35        # 0.15 m: shallow
    g[1, 3, 0] = 12        # 0.0177 m: not flooded even at 0.02
    s = ST.evaluate_buildings(g, b, min_depth_m=0.02)
    assert s.flooded_count == 2 and s.below_tier == 1 and s.tier_counts.tolist() == [1, 0, 0, 0]
    assert s.tier[0] == 0 and s.tier[1] == 1
    s = ST.evaluate_buildings(g, b)
    assert s.flooded_count == 1 and s.below_tier == 0


def test_speed_at_deepest_pixel_first_in_row_major_on_ties():
    fr = tiny_frame()
    b = make(fr, [polygon(fr, rect(4.2, 4.2, 6.8, 4.8), height_m=5)])     # pixels (4,4) (5,4) (6,4)
    g = blank_grid(fr)
    g[4, 4, :3] = (100, 255, 128)      # u = +4, v = 0     -> speed 4
    g[4, 5, :3] = (100, 128, 255)      # same depth, speed 4 (north) - tie goes to the first pixel
    g[4, 6, :3] = (90, 255, 255)       # shallower, speed 4*sqrt2 - must be ignored
    s = ST.evaluate_buildings(g, b)
    assert s.speed_ms[0] == pytest.approx(4.0)
    g[4, 4, :3] = (100, 128 + 64, 128)                                     # u = 64/127*4
    s = ST.evaluate_buildings(g, b)
    assert s.speed_ms[0] == pytest.approx(64 / 127 * 4, abs=1e-5)
    g[4, 5, 0] = 101                                                       # now the second pixel is the deepest
    g[4, 5, 1:3] = (128 - 127, 128)
    assert ST.evaluate_buildings(g, b).speed_ms[0] == pytest.approx(4.0, abs=0.04)    # code 1 = -127/127*4
    g[4, 5, 0] = 0
    assert ST.evaluate_buildings(blank_grid(fr), b).speed_ms[0] == 0


def test_hour_mix():
    fr = tiny_frame()
    b = one_pixel_buildings(fr, [(5, 5)])
    a, c = blank_grid(fr), blank_grid(fr)
    c[5, 5, 0] = 100
    c[5, 5, 1] = 255
    d = lambda code: (code / 255) ** 2 * 8
    s = ST.evaluate_buildings(a, b, c, 0.5)
    assert s.depth_m[0] == pytest.approx(d(50), abs=1e-6)                 # codes are mixed, then decoded
    assert s.speed_ms[0] == pytest.approx(0.5 * 4.0, abs=1e-6)            # velocity mixed linearly (dry hour -> 0)
    assert ST.evaluate_buildings(a, b, c, 0.0).depth_m[0] == 0
    assert ST.evaluate_buildings(a, b, c, 1.0).depth_m[0] == pytest.approx(d(100), abs=1e-6)
    assert ST.evaluate_buildings(a, b, c, 0.25).depth_m[0] == pytest.approx(d(25), abs=1e-6)
    assert ST.evaluate_buildings(a, b, c, 0.25).flooded_count == 0       # 25 -> 0.077 m < 0.1
    assert ST.evaluate_buildings(None, b, c, 0.5).depth_m[0] == pytest.approx(d(50), abs=1e-6)
    assert ST.evaluate_buildings(c, b, None, 0.5).depth_m[0] == pytest.approx(d(50), abs=1e-6)
    assert ST.evaluate_buildings(None, b, None, 0.5).flooded_count == 0
    # mixing a grid with itself is the same as the grid
    s1 = ST.evaluate_buildings(c, b)
    s2 = ST.evaluate_buildings(c, b, c, 0.37)
    assert s1.depth_m[0] == pytest.approx(s2.depth_m[0], abs=1e-6) and s1.speed_ms[0] == pytest.approx(s2.speed_ms[0], abs=1e-6)
    # flooded/tier from the mixed (float) code
    s = ST.evaluate_buildings(a, b, c, 0.6)                               # code 60 = 0.443 m -> moderate
    assert s.tier[0] == 2 and s.flooded_count == 1


def test_accepts_buildings_object_flat_and_noncontiguous_inputs():
    fr = tiny_frame()
    b = one_pixel_buildings(fr, [(5, 5)])
    g = blank_grid(fr)
    g[5, 5, 0] = 90
    ref = ST.evaluate_buildings(g, b).depth_m[0]
    assert ST.evaluate_buildings(g, b.index).depth_m[0] == ref
    assert ST.evaluate_buildings(g.reshape(-1), b).depth_m[0] == ref
    big = np.zeros((fr.height, fr.width * 2, 4), dtype=np.uint8)
    big[:, ::2] = g
    assert ST.evaluate_buildings(big[:, ::2], b).depth_m[0] == ref        # non-contiguous view
    with pytest.raises(ValueError):
        ST.evaluate_buildings(np.zeros((3, 3, 4), dtype=np.uint8), b)


# ------------------------------------------------------------------------------------------------ grid summary
def test_summarize_grid_hand_made():
    fr = tiny_frame(w=10, h=10, cell_m=10, k=0.8)                        # 8 m ground cells -> 64 m2
    g = blank_grid(fr)
    g[2, 2, 0] = 255                                                       # 8 m
    g[2, 3, :3] = (100, 255, 255)                                          # 1.2303 m, speed 4*sqrt2
    g[3, 3, 0] = 12                                                        # below the 0.02 m wet threshold
    g[4, 4, 0] = 200
    g[4, 4, 3] = 0                                                         # A = 0 ignored
    g[5, 5, 0] = 13                                                        # 0.0208 m: wet
    s = ST.summarize_grid(g, fr)
    d = ST.decode_depth
    assert s.wet_cells == 3
    assert s.wet_area_km2 == pytest.approx(3 * 64 / 1e6)
    assert s.max_depth_m == pytest.approx(8.0)
    assert s.mean_depth_m == pytest.approx((8.0 + d(100) + d(13)) / 3)
    assert s.volume_m3 == pytest.approx((8.0 + d(100) + d(13)) * 64)
    assert s.max_speed_ms == pytest.approx(4 * np.sqrt(2))


# ------------------------------------------------------------------------------------------------ real data, known-good numbers
@pytest.mark.parametrize("storm,flooded,tiers,wet_km2,maxd,roads_pct", [
    ("25yr", 1309, [831, 363, 107, 8], 14.06, 5.90, 2.1),
    ("5yr", 908, None, 6.70, 5.79, 1.2),
])
def test_known_good_numbers_hour_12(storm, flooded, tiers, wet_km2, maxd, roads_pct):
    b, r, fr = real_buildings(), real_roads(), real_frame()
    g = real_hour(storm, 12)
    s = ST.evaluate_buildings(g, b)
    assert s.flooded_count == flooded
    if tiers:
        assert s.tier_counts.tolist() == tiers
    assert s.below_tier == 0 and s.tier_counts.sum() == flooded
    assert len(s.order) == flooded and (np.diff(s.depth_m[s.order]) <= 0).all()
    sm = ST.summarize_grid(g, fr)
    assert round(sm.wet_area_km2, 2) == wet_km2
    assert round(sm.max_depth_m, 2) == maxd
    assert round(ST.evaluate_roads(g, r).flooded_pct, 1) == roads_pct
    assert 0 < sm.mean_depth_m < 2 and sm.max_speed_ms <= 4.05 and sm.volume_m3 > 1e5


def test_real_ranking_is_deepest_first_with_id_ties():
    b = real_buildings()
    s = ST.evaluate_buildings(real_hour("25yr", 12), b)
    o = s.order
    d = s.depth_m[o]
    assert (np.diff(d) <= 0).all()
    ties = np.diff(d) == 0
    assert (np.diff(o)[ties] > 0).all()
    assert s.flooded[o].all() and s.flooded.sum() == len(o)


def test_real_dry_hour_and_wet_hour_consistency():
    b = real_buildings()
    g = real_hour("25yr", 12)
    g2 = real_hour("25yr", 13)
    s0 = ST.evaluate_buildings(g, b)
    # mix=0 and mix=1 through the two-grid API equal the single-grid results
    s_a = ST.evaluate_buildings(g, b, g2, 0.0)
    s_b = ST.evaluate_buildings(g, b, g2, 1.0)
    s13 = ST.evaluate_buildings(g2, b)
    assert np.array_equal(s0.depth_m, s_a.depth_m) and np.array_equal(s13.depth_m, s_b.depth_m)
    # mixing a grid with itself reproduces it (to float32 rounding)
    sm = ST.evaluate_buildings(g, b, g, 0.3)
    assert np.allclose(sm.depth_m, s0.depth_m, atol=1e-5)
    assert sm.flooded_count == s0.flooded_count
    # the half mix never exceeds the deeper of its two ends (it can be below both when the two maxima sit on different pixels)
    half = ST.evaluate_buildings(g, b, g2, 0.5)
    hi = np.maximum(s0.depth_m, s13.depth_m) + 1e-4
    assert (half.depth_m <= hi).all()
    # roads mix at 0/1
    r = real_roads()
    assert ST.evaluate_roads(g, r, g2, 0.0).flooded_pct == ST.evaluate_roads(g, r).flooded_pct
    assert ST.evaluate_roads(g, r, g2, 1.0).flooded_pct == ST.evaluate_roads(g2, r).flooded_pct
    assert ST.evaluate_roads(g, r, g, 0.4).flooded_pct == pytest.approx(ST.evaluate_roads(g, r).flooded_pct, abs=0.01)


def test_real_evaluate_speed_targets():
    b, r, fr = real_buildings(), real_roads(), real_frame()
    g = real_hour("25yr", 12)
    ST.warmup()
    ST.evaluate_buildings(g, b)
    ST.evaluate_roads(g, r)

    def best(fn, n=40):
        t = 1e9
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            t = min(t, time.perf_counter() - t0)
        return t * 1000

    tb = best(lambda: ST.evaluate_buildings(g, b))
    tr = best(lambda: ST.evaluate_roads(g, r))
    g13 = real_hour("25yr", 13)
    tbm = best(lambda: ST.evaluate_buildings(g, b, g13, 0.5))
    trm = best(lambda: ST.evaluate_roads(g, r, g13, 0.5))
    ts = best(lambda: ST.summarize_grid(g, fr), 10)
    print(f"evaluate_buildings {tb:.2f} ms (mix {tbm:.2f}), evaluate_roads {tr:.3f} ms (mix {trm:.3f}), summarize_grid {ts:.2f} ms")
    import threading
    busy = sorted(t.name for t in threading.enumerate() if t is not threading.main_thread())
    assert tb < 5 and tr < 2, (tb, tr, busy)
    assert tbm < 5 and trm < 2, (tbm, trm, busy)


def test_warmup_runs():
    assert ST.warmup() >= 0
