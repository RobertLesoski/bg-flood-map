"""Tests for bgv.data.roads (segments, ~10 m samples, ribbon mesh) and the road flood evaluation."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from data_helpers import (blank_grid, line, px_to_lnglat, real_frame, real_hour, real_root, real_roads, tiny_frame)
from bgv.data import roads as RD
from bgv.data import stats as ST
from bgv.geo import R_MERC, lnglat_to_merc


def build(features, fr=None, **kw):
    fr = fr or tiny_frame()
    return RD.roads_from_features(features, fr, **kw), fr


def test_straight_road_samples_and_segments():
    fr = tiny_frame(w=40, h=10, cell_m=10, k=0.8)               # 1 px = 8 ground m
    r, _ = build([line(fr, [(2.5, 5.5), (22.5, 5.5)])], fr)     # 20 px = 160 ground m  -> 16 samples
    assert r.n_roads == 1 and r.n_samples == 16 and r.n_segments == 16
    assert r.total_length_m == pytest.approx(160.0, rel=1e-4)
    assert r.sample_weight_m.sum() == pytest.approx(160.0, rel=1e-4)
    assert np.allclose(r.sample_weight_m, 10.0, rtol=1e-4)
    assert (r.sample_pixel >= 0).all()
    # samples are the midpoints of 16 equal parts: first at 5 m, last at 155 m from the start
    X0, _ = fr.merc_to_world(fr.xmin + 2.5 * fr.cell_m, fr.ymax - 5.5 * fr.cell_m)
    assert r.sample_x[0] == pytest.approx(X0 + 5.0, abs=0.01) and r.sample_x[-1] == pytest.approx(X0 + 155.0, abs=0.01)
    assert list(r.road_off) == [0, 16]
    # segments tile the polyline, each at most 10 m long, linked end to start, one per sample
    seg_len = np.hypot(r.seg_x1 - r.seg_x0, r.seg_y1 - r.seg_y0)
    assert seg_len.sum() == pytest.approx(160.0, rel=1e-4) and seg_len.max() <= 10.0 + 1e-3
    assert np.allclose(r.seg_x1[:-1], r.seg_x0[1:], atol=1e-3)
    assert list(r.seg_sample) == list(range(16))
    assert (r.seg_road == 0).all()


def test_bent_road_segments_are_cut_at_the_vertex_and_follow_the_polyline():
    fr = tiny_frame(w=40, h=40, cell_m=10, k=0.8)
    pts = [(2.0, 2.0), (12.0, 2.0), (12.0, 14.0)]               # 80 m east then 96 m south, ground metres
    r, _ = build([line(fr, pts)], fr)
    L = 80 + 96
    assert r.total_length_m == pytest.approx(L, rel=1e-4)
    n = math.ceil(L / 10 - 1e-9)                               # 18 samples of 9.78 m
    assert r.n_samples == n
    xs = np.concatenate([r.seg_x0, r.seg_x1[-1:]])
    ys = np.concatenate([r.seg_y0, r.seg_y1[-1:]])
    # the corner of the polyline appears exactly as a segment end point
    cx, cy = fr.merc_to_world(fr.xmin + 12.0 * fr.cell_m, fr.ymax - 2.0 * fr.cell_m)
    assert np.min(np.hypot(xs - cx, ys - cy)) < 0.01
    # a part that contains the corner is rendered as two segments belonging to the same sample
    ids, counts = np.unique(r.seg_sample, return_counts=True)
    assert counts.max() == 2 and (counts == 2).sum() == 1 and len(ids) == n
    seg_len = np.hypot(r.seg_x1 - r.seg_x0, r.seg_y1 - r.seg_y0)
    assert seg_len.sum() == pytest.approx(L, rel=1e-4)


def test_outside_grid_samples_are_minus_one_and_total_length_keeps_them():
    fr = tiny_frame(w=20, h=10, cell_m=10, k=0.8)
    r, _ = build([line(fr, [(15.0, 5.5), (35.0, 5.5)])], fr)     # 20 px = 160 m, only the first 5 px (40 m) on the grid
    assert r.total_length_m == pytest.approx(160.0, rel=1e-4)
    assert (r.sample_pixel < 0).sum() == 12 and (r.sample_pixel >= 0).sum() == 4
    assert (r.sample_pixel[:4] >= 0).all()
    assert r.in_grid_length_m == pytest.approx(40.0, rel=1e-3)


def test_classes_widths_and_bad_geometry():
    fr = tiny_frame(w=40, h=10)
    feats = [line(fr, [(1, 1), (5, 1)], "motorway"), line(fr, [(1, 2), (5, 2)], "weird_new_class"),
             {"type": "Feature", "properties": {"highway": "primary"}, "geometry": None},
             {"type": "Feature", "properties": {"highway": "primary"}, "geometry": {"type": "Point", "coordinates": [0, 0]}},
             line(fr, [(3, 3), (3, 3)], "tertiary"),                                     # zero length
             {"type": "Feature", "properties": {"highway": "tertiary"},
              "geometry": {"type": "LineString", "coordinates": [list(px_to_lnglat(fr, 1, 4)), list(px_to_lnglat(fr, 6, 4))]}}]
    r, _ = build(feats, fr)
    assert r.n_roads == 6
    assert RD.ROAD_CLASSES[r.road_class[0]] == "motorway" and RD.ROAD_CLASSES[r.road_class[1]] == "other"
    assert r.road_length_m[2] == 0 and r.road_length_m[3] == 0 and r.road_length_m[4] == 0 and r.road_length_m[5] > 0
    assert r.road_off[3] == r.road_off[2] == r.road_off[4]
    assert set(np.unique(r.seg_road)) == {0, 1, 5}
    assert r.seg_width_m[r.seg_road == 0][0] == RD.ROAD_WIDTH_M[0]
    assert len(RD.ROAD_WIDTH_M) == len(RD.ROAD_CLASSES)


def test_ribbon_geometry_straight():
    fr = tiny_frame(w=40, h=10)
    r, _ = build([line(fr, [(2, 5), (12, 5)], "primary")], fr)           # 80 m -> 8 segments of 10 m
    rb = r.ribbon
    S = r.n_segments
    assert rb.vertex_count == 4 * S and rb.triangle_count == 2 * S
    assert rb.positions.dtype == np.float32 and rb.indices.dtype == np.uint32 and rb.sample_id.dtype == np.uint32
    P = rb.positions.reshape(S, 4, 3)
    w = RD.ROAD_WIDTH_M[RD.ROAD_CLASSES.index("primary")]
    assert np.allclose(np.hypot(P[:, 1, 0] - P[:, 0, 0], P[:, 1, 1] - P[:, 0, 1]), w, atol=1e-3)
    assert np.allclose(P[:, :, 2], 0)
    # neighbouring quads share their joint edge
    assert np.allclose(P[:-1, 2:, :], P[1:, :2, :], atol=1e-3)
    assert list(rb.across[:4]) == [-1, 1, -1, 1]
    # CCW seen from above
    t = rb.indices.reshape(-1, 3)
    a, b, c = (rb.positions[t[:, i], :2].astype(np.float64) for i in range(3))
    assert (((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) > 0).all()
    assert rb.indices.max() < rb.vertex_count
    assert rb.sample_id.max() < r.n_samples
    assert rb.along_m[0] == 0 and rb.along_m[-1] == pytest.approx(80, rel=1e-3)
    # width scale
    rb2 = RD.make_ribbon(r, 2.0)
    P2 = rb2.positions.reshape(S, 4, 3)
    assert np.allclose(np.hypot(P2[:, 1, 0] - P2[:, 0, 0], P2[:, 1, 1] - P2[:, 0, 1]), 2 * w, atol=1e-3)


def test_ribbon_miter_at_a_bend_has_no_gap():
    fr = tiny_frame(w=40, h=40)
    r, _ = build([line(fr, [(2, 2), (12, 2), (12, 12)], "residential")], fr)
    P = r.ribbon.positions.reshape(-1, 4, 3)
    assert np.allclose(P[:-1, 2:, :], P[1:, :2, :], atol=1e-3)           # all joints shared, including the corner


def test_zero_road_set():
    r, _ = build([], tiny_frame())
    assert r.n_roads == 0 and r.n_samples == 0 and r.n_segments == 0 and r.total_length_m == 0
    assert r.ribbon.triangle_count == 0
    s = ST.evaluate_roads(blank_grid(tiny_frame()), r)
    assert s.flooded_pct == 0 and s.flooded_roads == 0


# ------------------------------------------------------------------------------------------------ flood evaluation (tiny)
def _road_grid(fr):
    g = blank_grid(fr)
    return g


def test_evaluate_roads_share_a_equals_fraction_of_flooded_samples():
    fr = tiny_frame(w=40, h=10, cell_m=10, k=0.8)
    r, _ = build([line(fr, [(2.5, 5.5), (22.5, 5.5)]), line(fr, [(2.5, 8.5), (12.5, 8.5)])], fr)   # 160 m + 80 m
    g = blank_grid(fr)
    code = 100                                                                                      # 1.23 m
    g[5, 2:12, 0] = code                                    # first 10 columns of road 0 are wet (the first 80 m)
    s = ST.evaluate_roads(g, r)
    assert s.total_length_m == pytest.approx(240, rel=1e-4)
    assert s.flooded_length_m == pytest.approx(80, rel=0.02)
    assert s.flooded_pct == pytest.approx(100 * s.flooded_length_m / 240)
    assert s.flooded_roads == 1
    assert s.per_road_pct[0] == pytest.approx(50, abs=2) and s.per_road_pct[1] == 0
    assert s.sample_flooded.sum() == round(s.flooded_length_m / 10)
    assert s.sample_flooded.dtype == np.uint8 and s.per_road_pct.dtype == np.float32


def test_evaluate_roads_threshold_a0_and_zero_grid():
    fr = tiny_frame(w=40, h=10)
    r, _ = build([line(fr, [(2.5, 5.5), (22.5, 5.5)])], fr)
    g = blank_grid(fr)
    g[5, :, 0] = 28                                          # 0.0965 m: just below 0.1
    assert ST.evaluate_roads(g, r).flooded_pct == 0
    g[5, :, 0] = 29                                          # 0.1035 m: flooded
    s = ST.evaluate_roads(g, r)
    assert s.flooded_pct == pytest.approx(100.0) and s.per_road_pct[0] == pytest.approx(100.0)
    g[5, :, 3] = 0                                           # A = 0 -> outside the model domain, never flooded
    assert ST.evaluate_roads(g, r).flooded_pct == 0
    assert ST.evaluate_roads(None, r).flooded_pct == 0
    z = ST.evaluate_roads(None, r)
    assert z.total_length_m == r.total_length_m and z.sample_flooded.sum() == 0
    g[5, :, 3] = 255
    s = ST.evaluate_roads(g, r, min_depth_m=0.02)
    assert s.flooded_pct == pytest.approx(100.0)


def test_evaluate_roads_hour_mix():
    fr = tiny_frame(w=40, h=10)
    r, _ = build([line(fr, [(2.5, 5.5), (22.5, 5.5)])], fr)
    a = blank_grid(fr)
    b = blank_grid(fr)
    b[5, :, 0] = 100
    assert ST.evaluate_roads(a, r, b, 0.0).flooded_pct == 0
    assert ST.evaluate_roads(a, r, b, 1.0).flooded_pct == pytest.approx(100)
    assert ST.evaluate_roads(a, r, b, 0.5).flooded_pct == pytest.approx(100)             # code 50 = 0.31 m
    assert ST.evaluate_roads(a, r, b, 0.1).flooded_pct == 0                               # code 10 = 0.012 m
    assert ST.evaluate_roads(None, r, b, 0.5).flooded_pct == pytest.approx(100)           # None = dry grid
    assert ST.evaluate_roads(b, r, None, 0.5).flooded_pct == pytest.approx(100)           # (a only: b None = dry)
    assert ST.evaluate_roads(None, r, None, 0.5).flooded_pct == 0


# ------------------------------------------------------------------------------------------------ real data
def test_real_roads_basic():
    r = real_roads()
    root = real_root()
    gj = json.load(open(root / "layers" / "roads.geojson", encoding="utf-8"))
    assert r.n_roads == 516 == len(gj["features"])
    prop_len = sum(f["properties"]["length"] for f in gj["features"])
    assert r.total_length_m == pytest.approx(prop_len, rel=0.005)
    assert r.n_samples == r.sample_pixel.shape[0] > 30000
    assert abs(r.sample_weight_m.astype(np.float64).sum() - r.total_length_m) < 1.0
    assert r.sample_weight_m.max() <= 10.0 + 1e-3 and r.sample_weight_m.min() > 0
    assert r.road_off[0] == 0 and r.road_off[-1] == r.n_samples and (np.diff(r.road_off) >= 0).all()
    seg_len = np.hypot(r.seg_x1 - r.seg_x0, r.seg_y1 - r.seg_y0)
    assert seg_len.sum() == pytest.approx(r.total_length_m, rel=1e-3)
    assert np.isfinite(r.ribbon.positions).all() and r.ribbon.indices.max() < r.ribbon.vertex_count
    assert set(np.unique(r.seg_class)) <= set(range(len(RD.ROAD_CLASSES)))
    # per-road sample/segment bookkeeping
    assert (np.bincount(r.seg_road, weights=seg_len, minlength=r.n_roads) - r.road_length_m).max() < 0.5
    # the sample of every segment lies on the same road
    sample_road = np.repeat(np.arange(r.n_roads), np.diff(r.road_off))
    assert (sample_road[r.seg_sample] == r.seg_road).all()


def test_real_roads_cache_load_time():
    root, fr = real_root(), real_frame()
    a = RD.load_roads(root, fr)
    t0 = time.perf_counter()
    b = RD.load_roads(root, fr)
    dt = time.perf_counter() - t0
    assert b.timings.get("cached") and dt < 0.5
    assert np.array_equal(a.sample_pixel, b.sample_pixel) and np.array_equal(a.ribbon.indices, b.ribbon.indices)
    assert np.array_equal(a.ribbon.positions, b.ribbon.positions)


def _js_reference(root, fr, hour_grid, spacing=10.0, min_code=29):
    """Port of fluid-stats.js buildRoadIndex + evaluateRoads (per-segment cosine of the latitude), pure python."""
    gj = json.load(open(root / "layers" / "roads.geojson", encoding="utf-8"))
    total = 0.0
    flooded = 0.0
    nsamples = 0
    R = hour_grid[..., 0]
    A = hour_grid[..., 3]
    for f in gj["features"]:
        geom = f["geometry"]
        lines = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        for ln in lines:
            pts = [lnglat_to_merc(p[0], p[1]) for p in ln]
            if len(pts) < 2:
                continue
            cum = [0.0]
            for j in range(1, len(pts)):
                cos_lat = 1.0 / math.cosh((pts[j][1] + pts[j - 1][1]) / 2 / R_MERC)
                cum.append(cum[-1] + math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1]) * cos_lat)
            L = cum[-1]
            if not L > 0:
                continue
            cnt = max(1, math.ceil(L / spacing - 1e-9))
            step = L / cnt
            k = 0
            for s in range(cnt):
                pos = (s + 0.5) * step
                while k < len(pts) - 2 and cum[k + 1] < pos:
                    k += 1
                d = cum[k + 1] - cum[k]
                t = min(max((pos - cum[k]) / d if d > 0 else 0, 0), 1)
                gx = (pts[k][0] + (pts[k + 1][0] - pts[k][0]) * t - fr.xmin) / fr.cell_m
                gy = (fr.ymax - (pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t)) / fr.cell_m
                col, row = math.floor(gx), math.floor(gy)
                if 0 <= col < fr.width and 0 <= row < fr.height and A[row, col] != 0 and R[row, col] >= min_code:
                    flooded += step
            total += step * cnt
            nsamples += cnt
    return total, flooded, nsamples


@pytest.mark.parametrize("storm,expected_pct", [("25yr", 2.1), ("5yr", 1.2)])
def test_real_roads_known_good_share(storm, expected_pct):
    r = real_roads()
    s = ST.evaluate_roads(real_hour(storm, 12), r)
    assert round(s.flooded_pct, 1) == expected_pct, s.flooded_pct
    assert s.flooded_roads > 50


def test_real_roads_match_the_js_algorithm():
    root, fr, r = real_root(), real_frame(), real_roads()
    g = real_hour("25yr", 12)
    total, flooded, ns = _js_reference(root, fr, g)
    s = ST.evaluate_roads(g, r)
    js_pct = 100 * flooded / total
    print(f"JS-reference pct {js_pct:.4f} vs bgv {s.flooded_pct:.4f}; samples {ns} vs {r.n_samples}; "
          f"length {total:.1f} vs {r.total_length_m:.1f}")
    assert abs(js_pct - s.flooded_pct) < 0.1                  # the two differ only by the cosine model (< 0.15 % of length)
    assert abs(ns - r.n_samples) / ns < 0.01
    assert abs(total - r.total_length_m) / total < 0.002


def test_real_roads_evaluate_speed():
    r = real_roads()
    g = real_hour("25yr", 12)
    ST.evaluate_roads(g, r)
    best = 1e9
    for _ in range(30):
        t0 = time.perf_counter()
        ST.evaluate_roads(g, r)
        best = min(best, time.perf_counter() - t0)
    print(f"evaluate_roads best {best * 1000:.3f} ms")
    assert best < 0.002
