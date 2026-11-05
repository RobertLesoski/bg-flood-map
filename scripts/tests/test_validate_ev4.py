import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import validate_ev4 as v  # noqa: E402


def test_merc_origin_and_known_point():
    x, y = v.merc(0, 0)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6
    x, y = v.merc(-86.5, 36.9)
    assert abs(float(x) - (-9629000)) < 2000  # about -9.63e6 m


def test_sample_line_spacing_and_ends():
    xs, ys, tot = v.sample_line([(0, 0), (100, 0), (100, 50)], step=10)
    assert math.isclose(tot, 150)
    assert (xs[0], ys[0]) == (0, 0) and (xs[-1], ys[-1]) == (100, 50)
    steps = np.hypot(np.diff(xs), np.diff(ys))
    assert steps.max() <= 10.0 + 1e-9 and len(xs) >= 16


def test_sample_line_degenerate():
    xs, ys, tot = v.sample_line([(5, 5)], 1)
    assert tot == 0 and len(xs) == 1


def _arr():
    a = np.zeros((10, 10), np.uint16)
    a[2, 3] = 50    # 0.50 m
    a[5, :] = 20    # 0.20 m row
    a[0, 0] = 65535  # nodata
    return a


def test_depth_at_value_nodata_and_outside():
    a = _arr()
    # pixel size 1, origin top-left (0, 10): col 3,row 2 centre = (3.5, 7.5)
    d = v.depth_at(a, 0, 10, 1, [3.5, 0.5, -5, 3.5], [7.5, 9.5, 5, 4.5])
    assert d[0] == 0.5
    assert np.isnan(d[1]) and np.isnan(d[2])  # nodata, outside
    assert d[3] == 0.2


def test_depth_at_window_takes_max():
    a = _arr()
    d = v.depth_at(a, 0, 10, 1, [4.5], [6.5], win=1)  # neighbourhood includes (2,3)
    assert d[0] == 0.5


def test_road_stats_and_flag():
    st = v.road_stats([0.0, 0.05, 0.12, 0.35, np.nan])
    assert st["n"] == 4 and st["max"] == 0.35
    assert st["frac_0.1"] == 0.5 and st["frac_0.3"] == 0.25
    assert v.flagged(st, 0.3) and v.flagged(st, 0.1, 0.5) and not v.flagged(st, 0.3, 0.5)
    assert v.road_stats([np.nan])["max"] is None and not v.flagged(v.road_stats([]), 0.1)


def test_wilson_and_contingency():
    lo, hi = v.wilson(5, 10)
    assert lo < 0.5 < hi and 0.2 < lo and hi < 0.8
    assert all(math.isnan(x) for x in v.wilson(0, 0))
    assert v.wilson(10, 10)[1] == 1.0
    c = v.contingency([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert c == dict(hits=2, misses=1, false_alarms=1, correct_negatives=1)


def test_auc():
    assert v.auc([3, 4], [1, 2]) == 1.0
    assert v.auc([1, 2], [3, 4]) == 0.0
    assert v.auc([1], [1]) == 0.5
    assert math.isnan(v.auc([], [1]))


def test_in_bbox_and_norm():
    assert v.in_bbox([(-86.45, 36.98), (-86.46, 36.99)])
    assert not v.in_bbox([(-86.9, 36.98)])
    assert v.norm("Boyce-Fairview  Road") == "boyce fairview road"


def test_end_to_end_synthetic_road():
    a = np.zeros((100, 100), np.uint16)
    a[:, 50:] = 40  # right half 0.40 m
    xs, ys, _ = v.sample_line([(10, 50), (90, 50)], 1)
    vals = v.depth_at(a, 0, 100, 1, xs, ys)
    st = v.road_stats(vals)
    assert st["max"] == 0.4 and 0.4 < st["frac_0.3"] < 0.6
