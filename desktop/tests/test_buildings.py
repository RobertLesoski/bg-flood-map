"""Tests for bgv.data.buildings: hand-made cases + the real 20,284 footprints."""
from __future__ import annotations

import time

import numpy as np
import pytest

from data_helpers import (blank_grid, polygon, rect, real_buildings, real_frame, real_hour, real_root, tiny_frame)
from bgv.data import buildings as B


def build(features, fr=None, **kw):
    fr = fr or tiny_frame()
    return B.buildings_from_features(features, fr, **kw), fr


def px_set(b, i):
    W = b.index.width
    return {(int(p) % W, int(p) // W) for p in b.index.pixels_of(i)}


def tri_geometry(mesh):
    P = mesh.positions.astype(np.float64)
    t = mesh.indices.reshape(-1, 3)
    a, bb, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
    n = np.cross(bb - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n, area, t


def check_mesh_outward(mesh):
    """Every triangle's geometric normal agrees with the stored vertex normal (=> CCW from outside, culling is safe)."""
    n, area, t = tri_geometry(mesh)
    good = area > 0
    ng = n[good] / (2 * area[good])[:, None]
    stored = mesh.normals[t[good, 0]].astype(np.float64)
    return float((ng * stored).sum(axis=1).min())


# ------------------------------------------------------------------------------------------------ all-touched rule
def test_subpixel_building_owns_its_pixel():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(2.3, 2.3, 2.6, 2.6))], fr)
    assert px_set(b, 0) == {(2, 2)}
    assert b.in_grid[0]


def test_subpixel_building_on_pixel_corner_owns_four():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(2.8, 2.8, 3.2, 3.2))], fr)
    assert px_set(b, 0) == {(2, 2), (3, 2), (2, 3), (3, 3)}      # no centre inside, but all four are touched


def test_all_touched_includes_edge_pixels_not_only_centres():
    fr = tiny_frame()
    # thin diagonal triangle: few pixel centres inside, many pixels touched
    b, _ = build([polygon(fr, [(1.2, 1.1), (9.1, 8.7), (9.3, 8.3)])], fr)
    s = px_set(b, 0)
    assert (1, 1) in s and (9, 8) in s and (5, 4) in s
    assert len(s) >= 12


def test_polygon_with_hole_respects_hole_but_keeps_hole_edge_pixels():
    fr = tiny_frame()
    f = polygon(fr, rect(1.5, 1.5, 18.5, 18.5), rect(6.5, 6.5, 13.5, 13.5))
    b, _ = build([f], fr)
    s = px_set(b, 0)
    assert len(s) == 18 * 18 - 6 * 6                              # pixels 7..12 are wholly inside the hole
    assert (9, 9) not in s and (7, 7) not in s and (12, 12) not in s
    assert (6, 6) in s and (13, 13) in s                          # crossed by the hole's edge


def test_clockwise_and_ccw_give_identical_results():
    fr = tiny_frame()
    ccw = rect(3.2, 3.4, 8.7, 7.6)
    cw = ccw[::-1]
    b, _ = build([polygon(fr, ccw), polygon(fr, cw)], fr)
    assert px_set(b, 0) == px_set(b, 1)
    assert b.area_m2[0] == pytest.approx(b.area_m2[1], rel=1e-5) and b.area_m2[0] > 0
    assert b.cx[0] == pytest.approx(b.cx[1], abs=1e-3) and b.cy[0] == pytest.approx(b.cy[1], abs=1e-3)
    assert check_mesh_outward(b.mesh) > 0.999


def test_outside_grid_and_partially_outside():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(30, 30, 33, 33)), polygon(fr, rect(-3, 5.2, 2.5, 7.8)), polygon(fr, rect(-9, -9, -5, -5))], fr)
    assert px_set(b, 0) == set() and not b.in_grid[0]
    assert px_set(b, 2) == set() and not b.in_grid[2]
    assert px_set(b, 1) == {(c, r) for c in range(0, 3) for r in range(5, 8)}
    assert b.index.offsets[-1] == len(b.index.pixels)
    assert b.mesh.vertex_count > 0 and set(np.unique(b.mesh.building_id)) == {0, 1, 2}   # mesh keeps off-grid buildings


def test_area_and_centroid_in_ground_metres():
    fr = tiny_frame(k=0.8)                                         # 1 px = 10 mercator m = 8 ground m
    b, _ = build([polygon(fr, rect(2, 2, 6, 4))], fr)             # 4 x 2 px = 32 x 16 ground m
    assert b.area_m2[0] == pytest.approx(32 * 16, rel=1e-4)
    X, Y = fr.merc_to_world(fr.xmin + 4 * fr.cell_m, fr.ymax - 3 * fr.cell_m)
    assert b.cx[0] == pytest.approx(X, abs=0.01) and b.cy[0] == pytest.approx(Y, abs=0.01)


def test_degenerate_and_invalid_geometry_never_crashes():
    fr = tiny_frame()
    feats = [
        polygon(fr, [(2, 2), (4, 4), (6, 6)]),                    # collinear: zero area
        {"type": "Feature", "properties": {}, "geometry": None},
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [-86.4, 37.0]}},
        polygon(fr, rect(3, 3, 5, 5)),                            # valid one at the end
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-86.4, 37.0]] * 2]}},
    ]
    b, _ = build(feats, fr)
    assert b.n == 5
    assert list(b.in_grid) == [False, False, False, True, False]
    assert b.area_m2[0] == 0 and np.isnan(b.cx[0]) and np.isnan(b.cx[1])
    assert set(np.unique(b.mesh.building_id)) == {3}


def test_multipolygon_parts_are_both_rasterised_and_extruded():
    fr = tiny_frame()
    f = polygon(fr, rect(2.2, 2.2, 3.8, 3.8))
    g2 = polygon(fr, rect(10.2, 10.2, 11.8, 11.8))["geometry"]["coordinates"]
    f["geometry"] = {"type": "MultiPolygon", "coordinates": [f["geometry"]["coordinates"], g2]}
    b, _ = build([f], fr)
    assert px_set(b, 0) == {(c, r) for c in (2, 3) for r in (2, 3)} | {(c, r) for c in (10, 11) for r in (10, 11)}
    assert b.area_m2[0] == pytest.approx(2 * (1.6 * 8) ** 2, rel=1e-4)


# ------------------------------------------------------------------------------------------------ height rule
def test_height_defaults_and_clamp():
    fr = tiny_frame()
    sq = rect(3, 3, 5, 5)
    feats = [polygon(fr, sq, height_m=None), polygon(fr, sq, height_m=0), polygon(fr, sq, height_m=1.0),
             polygon(fr, sq, height_m=100), polygon(fr, sq, height_m=12.5), polygon(fr, sq), polygon(fr, sq, height_m="x")]
    b, _ = build(feats, fr)
    assert list(b.height_m) == [6.0, 6.0, 2.0, 60.0, 12.5, 6.0, 6.0]


def test_property_tables():
    fr = tiny_frame()
    f = polygon(fr, rect(3, 3, 5, 5), ADDRESS="1 MAIN ST", LAND_USE="C", sale_price=123456.0, name="Hall", amenity="townhall",
                building="civic")
    g = polygon(fr, rect(8, 8, 9, 9))
    b, _ = build([f, g], fr)
    assert str(b.address[0]) == "1 MAIN ST" and str(b.land_use[0]) == "C" and str(b.name[0]) == "Hall"
    assert str(b.amenity[0]) == "townhall" and str(b.building_type[0]) == "civic"
    assert b.sale_price[0] == 123456.0 and np.isnan(b.sale_price[1]) and str(b.address[1]) == ""


# ------------------------------------------------------------------------------------------------ mesh
def test_box_mesh_counts_and_normals():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(3, 3, 5, 6), height_m=10)], fr)
    m = b.mesh
    assert m.triangle_count == 2 + 8 and m.vertex_count == 4 + 16          # roof + 4 wall quads, no bottom
    assert m.positions.dtype == np.float32 and m.normals.dtype == np.float32
    assert m.building_id.dtype == np.uint32 and m.indices.dtype == np.uint32
    assert m.positions[:, 2].min() == 0 and m.positions[:, 2].max() == 10
    assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-6)
    assert check_mesh_outward(m) > 0.999
    n, area, _ = tri_geometry(m)
    assert (n[:, 2] <= 1e-9).sum() == 8 and (n[:, 2] > 0).sum() == 2        # only the 2 roof triangles face up
    roof = area[n[:, 2] > 0].sum()
    assert roof == pytest.approx(b.area_m2[0], rel=1e-4)
    assert m.indices.max() < m.vertex_count


def test_mesh_with_hole_has_hole_walls_facing_into_the_hole():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(2, 2, 18, 18), rect(7, 7, 13, 13), height_m=8)], fr)
    m = b.mesh
    n, area, t = tri_geometry(m)
    assert check_mesh_outward(m) > 0.999
    roof_area = area[n[:, 2] > 0].sum()
    assert roof_area == pytest.approx(b.area_m2[0], rel=1e-4)
    assert b.area_m2[0] == pytest.approx((16 * 16 - 6 * 6) * 64, rel=1e-4)
    assert m.triangle_count - (n[:, 2] > 0).sum() == 16                    # 8 walls (4 outer + 4 inner) x 2 triangles
    # inner wall normals point toward the hole centre
    cx, cy = (b.cx[0], b.cy[0])
    wall = np.flatnonzero(m.normals[:, 2] == 0)
    P = m.positions[wall].astype(np.float64)
    inner = (np.abs(P[:, 0] - cx) < 6 * 8 / 2 + 0.01) & (np.abs(P[:, 1] - cy) < 6 * 8 / 2 + 0.01)
    assert inner.any()
    to_center = np.stack([cx - P[inner, 0], cy - P[inner, 1]], axis=1)
    assert (np.sum(m.normals[wall][inner][:, :2] * to_center, axis=1) > 0).all()


def test_skirt_extends_walls_below_base():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(3, 3, 5, 5), height_m=10)], fr, skirt_m=1.5)
    assert b.mesh.positions[:, 2].min() == pytest.approx(-1.5) and b.mesh.positions[:, 2].max() == 10


def test_concave_polygon_roof_covers_exact_area():
    fr = tiny_frame()
    L = [(2, 2), (10, 2), (10, 5), (5, 5), (5, 12), (2, 12)]
    b, _ = build([polygon(fr, L, height_m=5)], fr)
    n, area, _ = tri_geometry(b.mesh)
    assert area[n[:, 2] > 0].sum() == pytest.approx(b.area_m2[0], rel=1e-4)
    assert check_mesh_outward(b.mesh) > 0.999


# ------------------------------------------------------------------------------------------------ base z
def test_attach_base_z_hand_made():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(2.2, 2.2, 4.8, 3.8)), polygon(fr, rect(10.2, 10.2, 10.8, 10.8)),
                  polygon(fr, rect(30, 30, 32, 32)), polygon(fr, rect(14.2, 14.2, 14.8, 14.8))], fr)
    t = np.full((fr.height, fr.width), 100.0, dtype=np.float32)
    t[2, 3] = 90.0
    t[10, 10] = np.nan                                  # building 1 has only this (NaN) pixel
    t[14, 14] = 95.5
    z = B.attach_base_z(b.index, t)
    assert z.dtype == np.float32 and z.shape == (4,)
    assert z[0] == 90.0 and np.isnan(z[1]) and np.isnan(z[2]) and z[3] == 95.5
    z2 = B.attach_base_z(b, t, fill=-1)
    assert z2[1] == -1 and z2[2] == -1 and z2[0] == 90.0
    with pytest.raises(ValueError):
        B.attach_base_z(b.index, np.zeros((3, 3), dtype=np.float32))
    # pure: input untouched
    assert np.isnan(t[10, 10]) and t[2, 3] == 90.0


def test_csr_invariants_tiny():
    fr = tiny_frame()
    b, _ = build([polygon(fr, rect(1.5, 1.5, 4.5, 3.5)), polygon(fr, rect(20, 20, 22, 22)), polygon(fr, rect(8, 8, 9.5, 9.5))], fr)
    o = b.index.offsets
    assert o.dtype == np.int32 and b.index.pixels.dtype == np.int32 and o[0] == 0 and (np.diff(o) >= 0).all()
    for i in range(3):
        p = b.index.pixels_of(i)
        assert (np.diff(p) > 0).all()


# ------------------------------------------------------------------------------------------------ real data
def test_real_buildings_basic():
    b = real_buildings()
    fr = real_frame()
    assert b.n == 20284
    assert b.index.width == fr.width and b.index.height == fr.height
    assert b.mesh.triangle_count < 1_200_000
    assert np.isfinite(b.mesh.positions).all() and np.isfinite(b.mesh.normals).all()
    assert b.mesh.indices.max() < b.mesh.vertex_count and b.mesh.building_id.max() < b.n
    assert b.height_m.min() >= 2 and b.height_m.max() <= 60
    assert b.mesh.positions[:, 2].min() >= 0
    ok = np.isfinite(b.cx)
    assert ok.sum() >= 20200
    assert (b.area_m2[ok] > 0).all()
    assert b.in_grid.sum() > 20000
    # world positions fall inside the grid extent (+ margin) for buildings that are on the grid
    sx, sy = fr.size_m
    outside = b.in_grid & ((np.abs(b.cx) > sx / 2 + 1) | (np.abs(b.cy) > sy / 2 + 1))
    assert outside.sum() <= 3          # only huge footprints straddling the grid edge (one 16 ha polygon is known)
    assert (b.area_m2[outside] > 50_000).all()
    # pixel index within bounds and ascending per building
    assert b.index.pixels.min() >= 0 and b.index.pixels.max() < fr.width * fr.height
    pix, off = b.index.pixels, b.index.offsets
    d = np.diff(pix.astype(np.int64))
    boundaries = np.zeros(len(pix), dtype=bool)
    boundaries[off[1:-1][off[1:-1] < len(pix)]] = True
    assert (d[~boundaries[1:]] > 0).all()


def test_real_mesh_is_outward_and_roofs_cover_footprints():
    b = real_buildings()
    m = b.mesh
    assert check_mesh_outward(m) > 0.99
    n, area, t = tri_geometry(m)
    up = n[:, 2] > 1e-9
    roof_area = np.bincount(m.building_id[t[up, 0]], weights=area[up], minlength=b.n)
    ok = b.area_m2 > 0
    rel = np.abs(roof_area[ok] - b.area_m2[ok]) / b.area_m2[ok]
    assert (rel > 0.02).sum() <= 5, f"{(rel > 0.02).sum()} buildings whose roof triangulation misses > 2% of the area"
    # no triangle faces down (no bottom faces emitted)
    assert not (n[:, 2] < -1e-9).any()


def test_real_holes_exist():
    import json
    root = real_root()
    gj = json.load(open(root / "layers" / "buildings.geojson", encoding="utf-8"))
    with_holes = [i for i, f in enumerate(gj["features"]) if len(f["geometry"]["coordinates"]) > 1]
    assert len(with_holes) == 9
    b = real_buildings()
    n, area, t = tri_geometry(b.mesh)
    up = n[:, 2] > 1e-9
    roof_area = np.bincount(b.mesh.building_id[t[up, 0]], weights=area[up], minlength=b.n)
    for i in with_holes:
        assert roof_area[i] == pytest.approx(b.area_m2[i], rel=0.02), i


def test_real_base_z_with_terrain():
    from bgv.data import terrain
    fr = real_frame()
    try:
        t = terrain.load_terrain(real_root(), fr)
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"terrain unavailable: {e}")
    b = real_buildings()
    z = B.attach_base_z(b.index, t)
    on = b.in_grid
    assert np.isfinite(z[on]).mean() > 0.97
    assert np.nanmin(z) > 100 and np.nanmax(z) < 300          # Bowling Green is at about 165 m
    # min over pixels <= the value at any single pixel of the building
    i = int(np.flatnonzero(on)[100])
    px = b.index.pixels_of(i)
    assert z[i] <= np.nanmin(t.reshape(-1)[px]) + 1e-6


def test_cache_roundtrip_and_load_time():
    root, fr = real_root(), real_frame()
    first = B.load_buildings(root, fr)                      # builds when missing
    t0 = time.perf_counter()
    again = B.load_buildings(root, fr)
    dt = time.perf_counter() - t0
    assert again.timings.get("cached") is True
    assert dt < 0.5, f"cached load took {dt:.3f}s"
    assert np.array_equal(first.index.pixels, again.index.pixels)
    assert np.array_equal(first.mesh.indices, again.mesh.indices)
    assert np.array_equal(first.mesh.positions, again.mesh.positions)
    assert np.array_equal(first.address, again.address) and np.array_equal(first.sale_price, again.sale_price, equal_nan=True)
    assert np.array_equal(first.cx, again.cx, equal_nan=True)


def test_all_touched_matches_exact_geometry_and_rasterio_on_real_data():
    """Exact reference (shapely closed-square intersection): 0 mismatches. rasterio/GDAL: a handful of sliver cases."""
    from rasterio_crosscheck import crosscheck, crosscheck_shapely
    root, fr, b = real_root(), real_frame(), real_buildings()
    grid = real_hour("25yr", 12)
    res = crosscheck(root, fr, b, grid)
    print("rasterio crosscheck:", {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()})
    assert len(res["depth_code_differs"]) == 0
    assert len(res["pixel_set_differs"]) <= 10
    assert res["pixels_only_mine"] + res["pixels_only_rasterio"] <= 20
    bad = crosscheck_shapely(root, fr, b)
    assert bad == [], f"{len(bad)} buildings differ from the exact intersection rule: {bad[:10]}"
