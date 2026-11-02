"""
Tests for scripts/flowgrid.py (the numerics and the bgflow format library).

  C:/Users/Minec/anaconda3/python.exe -m unittest discover -s C:/Users/Minec/Documents/BG_3D_Map/scripts/tests -v

Numbers refer to the checklist in the task: 1 inclined plane, 2 lake at rest, 3 mound, 4 wet/dry fronts and closed
basin, 5 Froude cap, 6 dry/nodata, 7 strips, 8 block_reduce, 9 convergence/rotation, 10 encode/decode/pack/gzip,
11 terrain units (read_dem_m), 12 warp and grid, 13 validate_storm_dir, 14 performance and memory. Everything is
analytic or an exact identity.
"""
import datetime
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import tracemalloc
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import flowgrid as fg  # noqa: E402

G = 9.81


def speed(u, v):
    return np.hypot(u.astype(np.float64), v.astype(np.float64))


def plane(shape, dx, gx_east=0.0, gy_south=0.0, z0=200.0):
    """float32 ground z0 + gx*x_east + gy*y_south; x = col*dx, y = row*dx (metres)."""
    H, W = shape
    x = np.arange(W)[None, :] * dx
    y = np.arange(H)[:, None] * dx
    return (z0 + gx_east * x + gy_south * y).astype(np.float32)


def manning_speed(h, s, n):
    return h ** (2.0 / 3.0) * math.sqrt(s) / n


# ------------------------------------------------------------------------------------------------ 1
class TestInclinedPlane(unittest.TestCase):
    H, W, DX, N = 40, 60, 2.0, 0.035

    def run_plane(self, gx, gy, depth=0.2):
        z = plane((self.H, self.W), self.DX, gx, gy)
        h = np.full((self.H, self.W), depth, np.float32)
        return fg.face_velocity(h, z, self.N, self.DX)

    def test_downslope_magnitude_and_no_perpendicular_component(self):
        s = 0.01
        u, v = self.run_plane(s, 0.0)               # ground rises toward the east, water runs west
        want = manning_speed(0.2, s, self.N)
        self.assertEqual(u.dtype, np.float32)
        self.assertEqual(u.shape, (self.H, self.W))
        inner = (slice(2, -2), slice(2, -2))
        self.assertTrue(np.all(u[inner] < 0))
        self.assertLess(np.abs(-u[inner] / want - 1).max(), 0.01)
        self.assertLess(np.abs(v).max(), 1e-6)
        # the edge cells have one face only but the same slope, so they agree too
        self.assertLess(np.abs(-u / want - 1).max(), 0.01)

    def test_axis_sign_conventions(self):
        s = 0.01
        want = manning_speed(0.2, s, self.N)
        u, v = self.run_plane(-s, 0.0)              # rises toward the west: water runs east
        self.assertAlmostEqual(float(u[20, 30]), want, delta=0.01 * want)
        u, v = self.run_plane(0.0, s)               # rises toward the south (higher row numbers): water runs NORTH
        self.assertAlmostEqual(float(v[20, 30]), want, delta=0.01 * want)
        self.assertLess(abs(float(u[20, 30])), 1e-6)
        u, v = self.run_plane(0.0, -s)              # rises toward the north: water runs south, v_north < 0
        self.assertAlmostEqual(float(v[20, 30]), -want, delta=0.01 * want)

    def test_diagonal_slope_has_manning_speed_and_follows_the_gradient(self):
        """A per-axis sqrt(|S|) would give 1.19x the speed at 45 degrees and bend the direction by up to 12 degrees."""
        s_total = 0.01
        want = manning_speed(0.2, s_total, self.N)
        for deg in (0.0, 10.0, 20.0, 45.0, 70.0, 90.0):
            th = math.radians(deg)
            gx, gy = s_total * math.cos(th), s_total * math.sin(th)     # gradient toward (east, south)
            u, v = self.run_plane(gx, gy)
            uc, vc = float(u[20, 30]), float(v[20, 30])
            self.assertAlmostEqual(math.hypot(uc, vc), want, delta=0.01 * want, msg="speed at %g deg" % deg)
            # water moves down-gradient: toward (west, north) = (-cos, +sin) in (east, north)
            cosang = (uc * -math.cos(th) + vc * math.sin(th)) / math.hypot(uc, vc)
            self.assertGreater(cosang, math.cos(math.radians(0.5)), msg="direction at %g deg" % deg)

    def test_manning_n_array_uses_face_mean(self):
        z = plane((self.H, self.W), self.DX, 0.01, 0.0)
        h = np.full((self.H, self.W), 0.2, np.float32)
        n = np.full((self.H, self.W), 0.03, np.float32)
        n[:, 30:] = 0.05                                       # faces: 0.03 | 0.04 (between cols 29 and 30) | 0.05
        u, v = fg.face_velocity(h, z, n, self.DX)
        m = manning_speed(0.2, 0.01, 1.0)                      # speed for n = 1
        self.assertAlmostEqual(float(-u[20, 10]) * 0.03 / m, 1.0, delta=1e-3)
        self.assertAlmostEqual(float(-u[20, 50]) * 0.05 / m, 1.0, delta=1e-3)
        self.assertAlmostEqual(float(-u[20, 29]) / m, 0.5 * (1 / 0.03 + 1 / 0.04), delta=1e-3 / 0.03)                  # relative 1e-3 of ~1/n
        self.assertAlmostEqual(float(-u[20, 30]) / m, 0.5 * (1 / 0.04 + 1 / 0.05), delta=1e-3 / 0.03)                  # relative 1e-3 of ~1/n
        n[:, :] = 0.035
        u2, _ = fg.face_velocity(h, z, n, self.DX)
        u3, _ = fg.face_velocity(h, z, 0.035, self.DX)
        np.testing.assert_allclose(u2, u3, rtol=1e-6)          # array n == scalar n when uniform (float32 n is 0.035 +- 4e-9)


# ------------------------------------------------------------------------------------------------ 2
class TestLakeAtRest(unittest.TestCase):
    def test_flat_surface_has_zero_velocity(self):
        rng = np.random.default_rng(3)
        H, W = 120, 150
        x = np.arange(W)[None, :]
        y = np.arange(H)[:, None]
        z = (200 + 0.5 * np.sin(x / 11.0) * np.cos(y / 13.0) + 0.01 * x + 0.1 * rng.standard_normal((H, W))).astype(np.float32)
        level = 200.45
        h = np.maximum(level - z.astype(np.float64), 0).astype(np.float32)
        self.assertGreater(float((h > 0.05).mean()), 0.2)
        self.assertLess(float((h > 0.05).mean()), 0.95)             # the shore is inside the domain
        for dx in (1.0, 2.0, 5.088):
            u, v = fg.face_velocity(h, z, 0.035, dx)
            self.assertLess(float(np.abs(u).max()), 1e-6)
            self.assertLess(float(np.abs(v).max()), 1e-6)

    def test_lake_level_on_a_float32_rounding_midpoint_stays_at_rest(self):
        """
        A water level exactly half an ulp above a float32 value makes eta = z + h round either way from cell to cell
        when the sum is done in float32 (1.5e-5 m at 200 m), which at dx = 1 m is a slope above the 1e-5 threshold.
        The face arithmetic is float64, so the lake must still be perfectly still.
        """
        rng = np.random.default_rng(9)
        H, W = 80, 90
        z = (200 + 0.3 * rng.standard_normal((H, W))).astype(np.float32)
        base = np.float32(200.45)
        level = float(base) + 0.5 * float(np.spacing(base))
        h = np.maximum(level - z.astype(np.float64), 0).astype(np.float32)
        # a solver's float32 depths carry independent last-bit noise: nudge each depth by one ulp up, down or not at all
        nudge = rng.integers(-1, 2, size=h.shape)
        h = np.where(nudge > 0, np.nextafter(h, np.float32(np.inf)),
                     np.where(nudge < 0, np.nextafter(h, np.float32(-np.inf)), h)).astype(np.float32)
        eta32 = z + h                                                     # what a float32 implementation would see
        wet = h > 0.05
        self.assertGreater(float(np.ptp(eta32[wet])), 1e-5)               # the float32 surface really is noisy
        eta64 = z.astype(np.float64) + h
        self.assertLess(float(np.ptp(eta64[wet])), 1e-6)                  # while the true surface is flat
        u, v = fg.face_velocity(h, z, 0.035, 1.0)
        self.assertEqual(float(np.abs(u).max()) + float(np.abs(v).max()), 0.0)

    def test_a_small_surface_step_does_move_water(self):
        """Positive control for the test above."""
        H, W = 60, 80
        z = plane((H, W), 2.0, 0.02, 0.0, z0=250.0)             # float32 at 250 m: ulp 3e-5 must not matter
        h = np.maximum(251.0 - z.astype(np.float64), 0).astype(np.float32)
        u0, _ = fg.face_velocity(h, z, 0.035, 2.0)
        self.assertEqual(float(np.abs(u0).max()), 0.0)
        h2 = h.copy()
        h2[:, :40] += 0.03
        u1, _ = fg.face_velocity(h2, z, 0.035, 2.0)
        self.assertGreater(float(np.abs(u1).max()), 0.05)


# ------------------------------------------------------------------------------------------------ 3
class TestMound(unittest.TestCase):
    def test_water_spreads_away_from_the_mound(self):
        N, dx, sig = 161, 2.0, 24.0                                  # sigma 24 m = 12 cells
        c = N // 2
        row, col = np.mgrid[0:N, 0:N]
        r = np.hypot(row - c, col - c) * dx
        h = (0.3 + 0.3 * np.exp(-r ** 2 / (2 * sig ** 2))).astype(np.float32)
        z = np.full((N, N), 200.0, np.float32)
        u, v = fg.face_velocity(h, z, 0.035, dx)
        flank = (r > sig) & (r < 2.5 * sig)
        ex = (col - c).astype(np.float64)
        en = -(row - c).astype(np.float64)                           # row 0 is north
        norm = np.hypot(ex, en)
        norm[norm == 0] = 1.0
        cosang = (u * ex / norm + v * en / norm)[flank] / speed(u, v)[flank]
        print("\n  mound: mean cos %.5f, min cos %.5f over %d flank cells" % (cosang.mean(), cosang.min(), cosang.size))
        self.assertGreater(cosang.mean(), 0.95)
        self.assertGreater(cosang.mean(), 0.995)                     # direction follows the surface gradient
        self.assertGreater(cosang.min(), 0.95)
        # magnitude at r = 1.5 sigma: Manning speed of the local surface slope within 10 %
        ring = (np.abs(r - 1.5 * sig) < dx) & (np.abs(row - c) > 3) & (np.abs(col - c) > 3)
        slope = 0.3 * (1.5 * sig) / sig ** 2 * math.exp(-1.5 ** 2 / 2)
        want = manning_speed(0.3 + 0.3 * math.exp(-1.5 ** 2 / 2), slope, 0.035)
        got = speed(u, v)[ring].mean()
        self.assertAlmostEqual(got / want, 1.0, delta=0.10)
        # the crest is calm, far field is dry-flat and calm
        self.assertLess(float(speed(u, v)[c, c]), 1e-3)
        self.assertLess(float(speed(u, v)[0, 0]), 1e-3)

    def test_symmetry(self):
        N, dx = 81, 2.0
        c = N // 2
        row, col = np.mgrid[0:N, 0:N]
        r = np.hypot(row - c, col - c) * dx
        h = (0.2 + 0.3 * np.exp(-r ** 2 / (2 * 16.0 ** 2))).astype(np.float32)
        z = np.full((N, N), 120.0, np.float32)
        u, v = fg.face_velocity(h, z, 0.03, dx)
        np.testing.assert_allclose(u, -u[:, ::-1], atol=1e-6)        # mirror in east-west: u flips
        np.testing.assert_allclose(v, -v[::-1, :], atol=1e-6)        # mirror in north-south: v flips
        np.testing.assert_allclose(u, u[::-1, :], atol=1e-6)


# ------------------------------------------------------------------------------------------------ 4
class TestWetDryFronts(unittest.TestCase):
    H, W, DX, N = 9, 40, 2.0, 0.035

    def pond(self, z_right, depth=0.5):
        z = np.full((self.H, self.W), 10.0, np.float32)
        z[:, 20:] = z_right
        h = np.zeros((self.H, self.W), np.float32)
        h[:, :20] = depth
        return h, z

    def test_pond_flows_onto_a_lower_dry_floor(self):
        h, z = self.pond(9.5)
        u, v = fg.face_velocity(h, z, self.N, self.DX)
        # face (19|20): h_f = max(eta) - max(z) = 10.5 - 10.0 = 0.5, speed capped at sqrt(g h_f);
        # cell 19 averages that face with its calm west face (equal weights)
        want = 0.5 * math.sqrt(G * 0.5)
        np.testing.assert_allclose(u[:, 19], want, rtol=1e-4)
        self.assertTrue(np.all(u[:, :19] == 0))
        self.assertTrue(np.all(u[:, 20:] == 0))           # the dry floor has no depth, hence no velocity
        self.assertLess(float(np.abs(v).max()), 1e-6)

    def test_front_over_a_ledge_uses_the_partial_face_depth(self):
        h, z = self.pond(10.2)                            # dry cell above the pond bed, below its surface
        u, _ = fg.face_velocity(h, z, self.N, self.DX)
        hf = 10.5 - 10.2
        want = hf * math.sqrt(G * hf) / (0.5 + hf)        # weights 0.5 (calm face) and h_f = 0.3
        np.testing.assert_allclose(u[:, 19], want, rtol=1e-4)

    def test_films_thinner_than_h_face_min_do_not_flow(self):
        for zr, flows in ((10.49, True), (10.498, False)):        # face depth 0.010 m (open) and 0.002 m (closed)
            h, z = self.pond(zr)
            u, _ = fg.face_velocity(h, z, self.N, self.DX)
            self.assertEqual(bool(np.abs(u).max() > 0), flows, msg="bed at %g" % zr)
        h, z = self.pond(10.498)
        u, _ = fg.face_velocity(h, z, self.N, self.DX, h_face_min=0.001)       # the threshold is a parameter
        self.assertGreater(float(u[:, 19].min()), 0.0)

    def test_cells_shallower_than_h_cell_min_have_no_velocity(self):
        h, z = self.pond(9.5, depth=0.5)
        h[:, 19] = 0.009                                              # a nearly dry cell at the front, below h_cell_min
        z[:, 19] = 10.0
        u, _ = fg.face_velocity(h, z, self.N, self.DX)
        self.assertTrue(np.all(u[:, 19] == 0))
        u, _ = fg.face_velocity(h, z, self.N, self.DX, h_cell_min=0.005)
        self.assertTrue(np.all(u[:, 19] != 0))

    def test_pond_never_climbs_onto_higher_dry_ground(self):
        for zr in (10.5, 10.6, 14.0):                     # at the surface, just above, far above
            h, z = self.pond(zr)
            u, v = fg.face_velocity(h, z, self.N, self.DX)
            self.assertEqual(float(np.abs(u).max()), 0.0, msg="rim at %g" % zr)
            self.assertEqual(float(np.abs(v).max()), 0.0)

    def basin(self, tilt):
        H, W = 11, 30
        z = np.full((H, W), 12.0, np.float32)
        z[3:8, 5:25] = 10.0
        h = np.zeros((H, W), np.float32)
        cols = np.arange(5, 25)
        h[3:8, 5:25] = (0.9 + tilt * (cols - 5))[None, :]       # surface tilts along x; rim at 12 m stays dry
        return h, z

    def test_closed_basin_water_stays_inside(self):
        for tilt in (-0.02, 0.02):
            h, z = self.basin(tilt)
            u, v = fg.face_velocity(h, z, self.N, self.DX)
            inside = np.zeros(h.shape, bool)
            inside[3:8, 5:25] = True
            self.assertTrue(np.all(u[~inside] == 0) and np.all(v[~inside] == 0))     # nothing outside the rim
            # water runs from the high end to the low end; each end cell has a single open face
            sign = -1.0 if tilt > 0 else 1.0
            self.assertTrue(np.all(sign * u[3:8, 5:25] > 0))
            self.assertLess(float(np.abs(v).max()), 1e-6)
            # the rim behaves exactly like nodata: closed faces, no leak
            zn = z.copy()
            zn[~inside] = np.nan
            un, vn = fg.face_velocity(h, zn, self.N, self.DX)
            np.testing.assert_array_equal(u, un)
            np.testing.assert_array_equal(v, vn)
            # high-end cell velocity points away from its wall, never into it
            if tilt < 0:
                self.assertTrue(np.all(u[3:8, 5] >= 0))
            else:
                self.assertTrue(np.all(u[3:8, 24] <= 0))


# ------------------------------------------------------------------------------------------------ 5
class TestFroudeCap(unittest.TestCase):
    def test_shallow_steep_slope_is_limited_by_sqrt_gh(self):
        z = plane((30, 50), 2.0, 0.3, 0.0)
        h = np.full((30, 50), 0.02, np.float32)
        uncapped = manning_speed(0.02, 0.3, 0.03)
        cap = math.sqrt(G * 0.02)
        self.assertGreater(uncapped, 2 * cap)
        u, v = fg.face_velocity(h, z, 0.03, 2.0)
        sp = speed(u, v)
        self.assertLessEqual(float(sp.max()), cap + 1e-6)
        self.assertAlmostEqual(float(sp[15, 25]), cap, delta=1e-4)     # the cap is what is active

    def test_deep_steep_slope_is_limited_by_vmax(self):
        z = plane((30, 50), 2.0, 0.3, 0.0)
        h = np.full((30, 50), 3.0, np.float32)
        u, v = fg.face_velocity(h, z, 0.03, 2.0)
        self.assertAlmostEqual(float(speed(u, v).max()), 4.0, delta=1e-5)
        u, v = fg.face_velocity(h, z, 0.03, 2.0, vmax=1.5)
        self.assertAlmostEqual(float(speed(u, v).max()), 1.5, delta=1e-5)

    def test_diagonal_does_not_exceed_the_cap(self):
        """Two axes at the cap each would give sqrt(2) times the cap: the vector must be limited."""
        z = plane((30, 50), 2.0, 0.3, 0.3)
        h = np.full((30, 50), 0.02, np.float32)
        u, v = fg.face_velocity(h, z, 0.03, 2.0)
        cap = math.sqrt(G * 0.02)
        self.assertLessEqual(float(speed(u, v).max()), cap + 1e-6)
        self.assertAlmostEqual(float(speed(u, v)[15, 25]), cap, delta=1e-4)

    def test_rough_steep_terrain_never_exceeds_the_bound(self):
        rng = np.random.default_rng(11)
        H, W = 200, 220
        z = (200 + np.cumsum(rng.standard_normal((H, W)), axis=1) * 0.4 + rng.standard_normal((H, W)) * 0.3).astype(np.float32)
        h = (rng.random((H, W)) * 0.6).astype(np.float32)
        h[rng.random((H, W)) < 0.2] = 0
        u, v = fg.face_velocity(h, z, 0.04, 2.0)
        bound = np.minimum(np.sqrt(G * h.astype(np.float64)), 4.0)
        self.assertLessEqual(float((speed(u, v) - bound).max()), 2e-6)       # float32 output: 1 ulp at 4 m/s is 5e-7
        self.assertGreater(float(speed(u, v).max()), 0.3)               # the field is actually moving


# ------------------------------------------------------------------------------------------------ 6
class TestDryAndNodata(unittest.TestCase):
    def field(self):
        rng = np.random.default_rng(5)
        H, W = 90, 100
        z = (200 + 2 * np.sin(np.arange(W) / 15.0)[None, :] + 0.05 * rng.standard_normal((H, W))).astype(np.float32)
        h = np.maximum(201.0 - z, 0).astype(np.float32)
        h[:, 30:34] = 0.5
        return h, z

    def test_dry_domain_is_zero(self):
        z = plane((20, 20), 2.0, 0.1, 0.1)
        u, v = fg.face_velocity(np.zeros((20, 20), np.float32), z, 0.035, 2.0)
        self.assertEqual(float(np.abs(u).max()) + float(np.abs(v).max()), 0.0)

    def test_nan_and_nodata_cells_are_zero_and_do_not_poison_neighbours(self):
        h, z = self.field()
        h_nan, z_nan = h.copy(), z.copy()
        z_nan[10:14, 28:36] = np.nan                  # all holes sit in the flowing strip (cols 28-36), next to moving water
        h_nan[50:53, 29:37] = np.nan
        z_nan[70, 31] = np.inf
        h_nan[20, 32] = np.nan
        h_nan[30, 33] = np.inf
        z_nan[40, 30] = np.nan
        hole = np.isnan(h_nan) | np.isnan(z_nan) | np.isinf(z_nan) | np.isinf(h_nan)
        h0, z0 = h_nan.copy(), z_nan.copy()
        ref_u, _ = fg.face_velocity(h, z, 0.035, 2.0)
        self.assertGreater(float(np.abs(ref_u[8:16, 26:38]).max()), 0.3)       # the water around the holes really moves
        with warnings.catch_warnings():
            warnings.simplefilter("error")             # no RuntimeWarning from NaN / inf input
            u, v = fg.face_velocity(h_nan, z_nan, 0.035, 2.0)
        self.assertTrue(np.isfinite(u).all() and np.isfinite(v).all())
        self.assertTrue(np.all(u[hole] == 0) and np.all(v[hole] == 0))
        dry = np.nan_to_num(h_nan, nan=0.0, posinf=0.0) < 0.01
        self.assertTrue(np.all(u[dry] == 0) and np.all(v[dry] == 0))
        np.testing.assert_array_equal(h_nan, h0)                        # inputs untouched
        np.testing.assert_array_equal(z_nan, z0)
        self.assertGreater(float(speed(u, v).max()), 0.1)               # and the rest still flows

        # equals the same domain with the holes replaced by tall dry walls
        hw = np.where(hole, 0.0, np.nan_to_num(h_nan, nan=0.0, posinf=0.0)).astype(np.float32)
        zw = np.where(hole, 1.0e4, z_nan).astype(np.float32)
        uw, vw = fg.face_velocity(hw, zw, 0.035, 2.0)
        np.testing.assert_array_equal(u[~hole], uw[~hole])
        np.testing.assert_array_equal(v[~hole], vw[~hole])

        # raster nodata (-9999) behaves like NaN
        z_nd = np.where(np.isnan(z_nan) | np.isinf(z_nan), -9999.0, z_nan).astype(np.float32)
        h_nd = np.where(np.isnan(h_nan) | np.isinf(h_nan), -9999.0, h_nan).astype(np.float32)
        un, vn = fg.face_velocity(h_nd, z_nd, 0.035, 2.0)
        np.testing.assert_array_equal(u, un)
        np.testing.assert_array_equal(v, vn)

    def test_bad_manning_cells_fall_back_to_the_default(self):
        h, z = self.field()
        n = np.full(h.shape, 0.035, np.float32)
        nb = n.copy()
        nb[40:50, 28:37] = np.nan                      # inside the flowing strip, otherwise the repair would not show
        nb[60, 31] = 0.0
        nb[61, 32] = -1.0
        nb[62, 33] = np.inf
        u, v = fg.face_velocity(h, z, nb, 2.0)
        u2, v2 = fg.face_velocity(h, z, n, 2.0)
        self.assertGreater(float(np.abs(u2[40:50, 28:37]).max()), 0.3)
        self.assertTrue(np.isfinite(u).all())
        np.testing.assert_array_equal(u, u2)
        np.testing.assert_array_equal(v, v2)

    def test_float64_and_shape_errors(self):
        h, z = self.field()
        u, _ = fg.face_velocity(h.astype(np.float64), z.astype(np.float64), 0.035, 2.0)
        u32, _ = fg.face_velocity(h, z, 0.035, 2.0)
        self.assertEqual(u.dtype, np.float32)
        np.testing.assert_allclose(u, u32, atol=1e-5)
        with self.assertRaises(ValueError):
            fg.face_velocity(h, z[:-1], 0.035, 2.0)
        with self.assertRaises(ValueError):
            fg.face_velocity(h, z, 0.035, 0.0)
        with self.assertRaises(ValueError):
            fg.face_velocity(h, z, np.ones((3, 3)), 2.0)

    def test_tiny_arrays(self):
        for shape in ((1, 1), (1, 7), (7, 1), (2, 2)):
            z = plane(shape, 2.0, 0.01, 0.01)
            h = np.full(shape, 0.3, np.float32)
            u, v = fg.face_velocity(h, z, 0.035, 2.0)
            self.assertEqual(u.shape, shape)
            self.assertTrue(np.isfinite(u).all() and np.isfinite(v).all())
        self.assertEqual(fg.face_velocity(np.zeros((1, 1)), np.zeros((1, 1)), 0.03, 2.0)[0].shape, (1, 1))


# ------------------------------------------------------------------------------------------------ 7
class TestStrips(unittest.TestCase):
    def test_strips_equal_whole_array_exactly(self):
        rng = np.random.default_rng(1)
        H, W = 157, 131
        z = (200 + 3 * np.sin(np.arange(W) / 20.0)[None, :] + 2 * np.cos(np.arange(H) / 17.0)[:, None]
             + 0.05 * rng.standard_normal((H, W))).astype(np.float32)
        h = np.maximum(201.0 - z, 0).astype(np.float32)
        h += ((rng.random((H, W)) < 0.3) * rng.random((H, W)) * 0.1).astype(np.float32)
        z[50:55, 60:70] = np.nan
        h[100:104, 90:100] = np.nan
        z[120, 20] = -9999.0
        n = (0.03 + 0.02 * rng.random((H, W))).astype(np.float32)
        n[10, 10] = np.nan
        u0, v0 = fg.face_velocity(h, z, n, 2.0)
        self.assertGreater(float(speed(u0, v0).max()), 0.2)
        self.assertGreater(float((speed(u0, v0) > 0).mean()), 0.2)
        for rows in (1, 2, 3, 4, 7, 50, 156, 157, 1000):
            u1, v1 = fg.face_velocity_strips(h, z, n, 2.0, strip_rows=rows)
            self.assertEqual(float(np.abs(u1 - u0).max()), 0.0, msg="strip_rows=%d" % rows)
            self.assertEqual(float(np.abs(v1 - v0).max()), 0.0, msg="strip_rows=%d" % rows)
        # scalar n and non-default options
        u0, v0 = fg.face_velocity(h, z, 0.04, 2.0, vmax=2.0, h_cell_min=0.02, slope_min=1e-4)
        u1, v1 = fg.face_velocity_strips(h, z, 0.04, 2.0, strip_rows=5, vmax=2.0, h_cell_min=0.02, slope_min=1e-4)
        np.testing.assert_array_equal(u0, u1)
        np.testing.assert_array_equal(v0, v1)

    def test_result_does_not_depend_on_the_internal_batch_size(self):
        """float32 accumulation order once depended on where batch boundaries fell (found by the 4000x4000 test)."""
        rng = np.random.default_rng(6)
        H, W = 120, 140
        z = (200 + 2 * np.sin(np.arange(W) / 9.0)[None, :] + 0.2 * rng.standard_normal((H, W))).astype(np.float32)
        h = np.maximum(200.8 - z, 0).astype(np.float32) + np.float32(0.05)
        ref = fg.face_velocity(h, z, 0.035, 2.0)
        self.assertGreater(float(speed(*ref).max()), 0.3)
        old = fg._CHUNK
        try:
            for chunk in (1, 7, 100, 1001, 4099):
                fg._CHUNK = chunk
                u, v = fg.face_velocity(h, z, 0.035, 2.0)
                np.testing.assert_array_equal(u, ref[0], err_msg="chunk=%d" % chunk)
                np.testing.assert_array_equal(v, ref[1], err_msg="chunk=%d" % chunk)
                us, vs = fg.face_velocity_strips(h, z, 0.035, 2.0, strip_rows=9)
                np.testing.assert_array_equal(us, ref[0], err_msg="strips, chunk=%d" % chunk)
                np.testing.assert_array_equal(vs, ref[1], err_msg="strips, chunk=%d" % chunk)
        finally:
            fg._CHUNK = old

    def test_strips_skip_dry_strips_and_handle_a_single_row(self):
        h = np.zeros((300, 40), np.float32)
        z = plane((300, 40), 2.0, 0.01, 0.0)
        h[290:, :] = 0.3
        u1, v1 = fg.face_velocity_strips(h, z, 0.035, 2.0, strip_rows=16)
        u0, v0 = fg.face_velocity(h, z, 0.035, 2.0)
        np.testing.assert_array_equal(u0, u1)
        self.assertGreater(float(np.abs(u1).max()), 0.5)
        hr, zr = h[-1:], z[-1:]
        one_row = fg.face_velocity(hr, zr, 0.035, 2.0)[0]
        self.assertGreater(float(np.abs(one_row).max()), 0.5)
        np.testing.assert_array_equal(one_row, fg.face_velocity_strips(hr, zr, 0.035, 2.0, strip_rows=1)[0])
        with self.assertRaises(ValueError):
            fg.face_velocity_strips(h, z, 0.035, 2.0, strip_rows=0)


# ------------------------------------------------------------------------------------------------ 8
class TestBlockReduce(unittest.TestCase):
    def hand(self):
        depth = np.array([[1, 0, 1, 3],
                          [0, 0, 0, 0],
                          [0, 0, 2, 2],
                          [4, 0, 2, 2]], np.float32)
        u = np.array([[1, 5, 1, 3],
                      [9, 9, 7, 7],
                      [8, 8, 3, 1],
                      [2, 6, 1, 3]], np.float32)
        v = np.array([[1, 2, 0, 0],
                      [1, 1, 1, 1],
                      [0, 0, -1, 1],
                      [-2, 7, 1, -1]], np.float32)
        valid = np.ones((4, 4), bool)
        valid[2:4, 0:2] = False
        return depth, u, v, valid

    def test_hand_made_4x4(self):
        depth, u, v, valid = self.hand()
        dk, uk, vk, ok = fg.block_reduce(depth, u, v, 2, valid)
        np.testing.assert_allclose(dk, [[0.25, 1.0], [1.0, 2.0]], atol=1e-6)
        # B00: only the 1 m cell is wet -> its velocity; B01: (1*1 + 3*3)/4; B10: the 4 m cell; B11: equal weights
        np.testing.assert_allclose(uk, [[1.0, 2.5], [2.0, 2.0]], atol=1e-6)
        np.testing.assert_allclose(vk, [[1.0, 0.0], [-2.0, 0.0]], atol=1e-6)
        self.assertEqual(ok.tolist(), [[True, True], [False, True]])
        self.assertEqual((dk.dtype, uk.dtype, ok.dtype), (np.float32, np.float32, np.dtype(bool)))
        # a dry block has zero velocity
        d2 = depth.copy()
        d2[0:2, 0:2] = 0
        _, uk2, vk2, _ = fg.block_reduce(d2, u, v, 2)
        self.assertEqual((float(uk2[0, 0]), float(vk2[0, 0])), (0.0, 0.0))

    def test_volume_and_discharge_are_conserved_and_remainder_trimmed(self):
        rng = np.random.default_rng(2)
        H, W, k = 103, 97, 5
        depth = (rng.random((H, W)) * 2.0 * (rng.random((H, W)) < 0.5)).astype(np.float32)
        u = (0.3 + rng.standard_normal((H, W)) * 0.5).astype(np.float32)
        v = (rng.standard_normal((H, W)) * 0.5).astype(np.float32)
        dk, uk, vk, ok = fg.block_reduce(depth, u, v, k, froude_g=None)
        self.assertEqual(dk.shape, (20, 19))
        tr = (slice(0, 100), slice(0, 95))
        vol = float(depth[tr].astype(np.float64).sum())
        self.assertAlmostEqual(float(dk.astype(np.float64).sum()) * k * k / vol, 1.0, places=6)
        qx = float((depth[tr].astype(np.float64) * u[tr]).sum())
        qxk = float((dk.astype(np.float64) * uk).sum()) * k * k
        self.assertAlmostEqual(qxk / qx, 1.0, places=4)
        # a 7.5 M cell array needs several internal batches; the blocks inside the first copy must be unchanged
        reps = (25, 30)
        big = fg.block_reduce(np.tile(depth, reps), np.tile(u, reps), np.tile(v, reps), k, froude_g=None)
        self.assertEqual(big[0].shape, (515, 582))
        np.testing.assert_array_equal(big[0][:20, :19], dk)
        np.testing.assert_array_equal(big[1][:20, :19], uk)

    def test_sparse_block_speed_is_limited_to_the_froude_bound(self):
        depth = np.zeros((5, 5), np.float32)
        depth[2, 2] = 1.0
        u = np.zeros((5, 5), np.float32)
        u[2, 2] = 3.0
        v = np.zeros((5, 5), np.float32)
        _, uk_pure, _, _ = fg.block_reduce(depth, u, v, 5, froude_g=None)
        dk, uk, vk, _ = fg.block_reduce(depth, u, v, 5)
        self.assertAlmostEqual(float(uk_pure[0, 0]), 3.0, places=6)          # plain weighted mean: breaks the bound
        self.assertAlmostEqual(float(dk[0, 0]), 0.04, places=7)
        self.assertAlmostEqual(float(uk[0, 0]), math.sqrt(G * 0.04), places=5)
        rng = np.random.default_rng(8)
        d = (rng.random((100, 100)) * (rng.random((100, 100)) < 0.2)).astype(np.float32)
        sp = np.minimum(np.sqrt(G * d), 4.0) * rng.random((100, 100)).astype(np.float32)
        ang = rng.random((100, 100)) * 2 * np.pi
        uu, vv = (sp * np.cos(ang)).astype(np.float32), (sp * np.sin(ang)).astype(np.float32)
        d2, u2, v2, _ = fg.block_reduce(d, uu, vv, 4)
        self.assertLessEqual(float((speed(u2, v2) - np.sqrt(G * d2.astype(np.float64))).max()), 2e-6)
        d3, u3, v3, _ = fg.block_reduce(d, uu, vv, 4, froude_g=None)
        self.assertGreater(float((speed(u3, v3) - np.sqrt(G * d3.astype(np.float64))).max()), 0.1)

    def test_nan_input_and_errors(self):
        depth = np.full((4, 4), 0.5, np.float32)
        depth[0, 0] = np.nan
        u = np.ones((4, 4), np.float32)
        u[0, 1] = np.nan
        dk, uk, _, ok = fg.block_reduce(depth, u, np.zeros((4, 4), np.float32), 2)
        self.assertTrue(np.isfinite(dk).all() and np.isfinite(uk).all())
        self.assertAlmostEqual(float(dk[0, 0]), 0.375, places=6)
        self.assertTrue(ok.all())
        with self.assertRaises(ValueError):
            fg.block_reduce(depth, u, u, 5)
        with self.assertRaises(ValueError):
            fg.block_reduce(depth, u, u, 0)


# ------------------------------------------------------------------------------------------------ 9
class TestConvergence(unittest.TestCase):
    def test_rotation_matches_the_true_bearing_of_grid_north(self):
        from rasterio.warp import transform
        lon0, lat0 = -86.4, 36.97
        (e0,), (n0,) = transform("EPSG:4326", "EPSG:32616", [lon0], [lat0])

        a_wgs, e2 = 6378137.0, 0.00669437999014

        def bearing(de, dn):
            # true bearing of a short UTM segment: east/north ground distances from the lon/lat differences
            (la, lb), (pa, pb) = transform("EPSG:32616", "EPSG:4326", [e0, e0 + de], [n0, n0 + dn])
            mid = math.radians(0.5 * (pa + pb))
            w = math.sqrt(1 - e2 * math.sin(mid) ** 2)
            prime = a_wgs / w                                   # radius of the parallel's normal section
            merid = a_wgs * (1 - e2) / w ** 3                   # meridian radius of curvature
            return math.atan2(prime * math.cos(mid) * math.radians(lb - la), merid * math.radians(pb - pa))

        true_bearing_north = bearing(0.0, 1000.0)
        gamma = float(fg.convergence_angle_rad(lon0, lat0))
        print("\n  gamma formula %.6f rad, rasterio bearing of grid north %.6f rad" % (gamma, true_bearing_north))
        self.assertGreater(gamma, 0)                                     # east of the central meridian
        self.assertLess(abs(gamma - true_bearing_north), 1e-3)
        self.assertLess(abs(gamma - true_bearing_north), 2e-5)           # in fact much closer
        ue, vn = fg.rotate_to_true(np.float64(0.0), np.float64(1.0), gamma)
        self.assertLess(abs(math.atan2(ue, vn) - true_bearing_north), 1e-3)
        self.assertAlmostEqual(math.hypot(ue, vn), 1.0, places=12)
        # grid east is turned by the same angle: bearing 90 deg + gamma
        ue, vn = fg.rotate_to_true(np.float64(1.0), np.float64(0.0), gamma)
        self.assertLess(abs(math.atan2(ue, vn) - bearing(1000.0, 0.0)), 1e-3)
        self.assertLess(abs(math.atan2(ue, vn) - (math.pi / 2 + gamma)), 1e-12)

    def test_sign_flips_west_of_the_central_meridian_and_arrays_work(self):
        self.assertLess(float(fg.convergence_angle_rad(-88.0, 37.0)), 0)
        self.assertAlmostEqual(float(fg.convergence_angle_rad(-87.0, 37.0)), 0.0, places=15)
        self.assertAlmostEqual(float(fg.convergence_angle_rad(-86.0, 0.0)), 0.0, places=15)          # none on the equator
        gam = fg.convergence_angle_rad(np.array([-86.5, -86.3]), np.array([37.0, 37.0]))
        u = np.array([[1.0, 0.0]], np.float32)
        v = np.array([[0.0, 1.0]], np.float32)
        ue, vn = fg.rotate_to_true(u, v, gam[None, :])
        self.assertEqual(ue.dtype, np.float32)
        np.testing.assert_allclose(np.hypot(ue, vn), 1.0, atol=1e-6)
        ue0, vn0 = fg.rotate_to_true(u, v, 0.0)
        np.testing.assert_array_equal(ue0, u)

    def test_convergence_field_matches_the_exact_formula(self):
        from rasterio.transform import from_origin
        tr = from_origin(548000.0, 4096000.0, 2.0, 2.0)
        H, W = 700, 900
        field = fg.convergence_field_utm16n(tr, H, W, step=100)
        self.assertEqual(field.shape, (H, W))
        rr = np.array([0, 123, 350, 699, 650])
        cc = np.array([0, 777, 450, 899, 12])
        e = tr.c + tr.a * (cc + 0.5)
        n = tr.f + tr.e * (rr + 0.5)
        lon, lat = fg.utm16n_to_lnglat(e, n)
        exact = fg.convergence_angle_rad(lon, lat)
        self.assertLess(float(np.abs(field[rr, cc] - exact).max()), 1e-6)


# ------------------------------------------------------------------------------------------------ 10
class TestEncoding(unittest.TestCase):
    def test_depth_round_trip_bounds(self):
        for d in (0.02, 0.1, 0.5, 2.0, 8.0):
            back = float(fg.decode_depth(fg.encode_depth(d)))
            half_step = 0.5 * (16.0 / 255.0) * math.sqrt(d / 8.0)          # d depth / d code, halved
            self.assertLessEqual(abs(back - d), half_step * 1.05 + 1e-6, msg="depth %g" % d)
        self.assertEqual(int(fg.encode_depth(0.02)), 13)
        self.assertEqual(int(fg.encode_depth(8.0)), 255)
        self.assertEqual(float(fg.decode_depth(255)), 8.0)
        self.assertEqual(float(fg.decode_depth(0)), 0.0)
        self.assertEqual(fg.encode_depth(np.array([0.0, 100.0, -3.0, np.nan, np.inf])).tolist(), [0, 255, 0, 0, 255])
        self.assertEqual(fg.encode_depth(np.zeros(3)).dtype, np.uint8)
        self.assertEqual(fg.decode_depth(np.array([1, 2], np.uint8)).dtype, np.float32)

    def test_velocity_round_trip_and_clipping(self):
        half = 0.5 * 4.0 / 127.0
        for x in (0.05, 1.0, 4.0, -0.05, -1.0, -4.0, 0.0):
            back = float(fg.decode_vel(fg.encode_vel(x)))
            self.assertLessEqual(abs(back - x), half + 1e-6, msg="vel %g" % x)
        self.assertEqual(int(fg.encode_vel(0.0)), 128)
        self.assertEqual(float(fg.decode_vel(128)), 0.0)
        self.assertEqual(fg.encode_vel(np.array([10.0, -10.0, 4.0, -4.0, np.nan])).tolist(), [255, 1, 255, 1, 128])
        self.assertEqual(float(fg.decode_vel(255)), 4.0)
        self.assertEqual(float(fg.decode_vel(1)), -4.0)
        self.assertEqual(float(fg.decode_vel(0)), -4.0)                    # code 0 is never written; clamped
        codes = fg.encode_vel(np.linspace(0, 4, 50))
        np.testing.assert_array_equal(fg.encode_vel(-np.linspace(0, 4, 50)).astype(int), 256 - codes.astype(int))

    def test_discharge_code(self):
        for q in (0.001, 0.05, 0.5, 2.0):
            back = float(fg.decode_q(fg.encode_q(q)))
            half = 0.5 * 2 * math.sqrt(q * fg.QMAX) / 255.0 * 1.1 + 1e-6
            self.assertLessEqual(abs(back - q), max(half, 1e-5))
        self.assertEqual(int(fg.encode_q(5.0)), 255)
        self.assertEqual(float(fg.decode_q(255)), 2.0)

    def test_pack_rgba_byte_layout_2x2(self):
        depth = np.array([[0.5, 0.01], [8.5, 2.0]], np.float32)
        u = np.array([[1.0, 3.0], [-4.0, 0.0]], np.float32)
        v = np.array([[-2.1, 3.0], [0.5, 2.0]], np.float32)
        valid = np.array([[True, True], [True, False]])
        px = fg.pack_rgba(depth, u, v, valid)
        self.assertEqual(px.dtype, np.uint8)
        self.assertEqual(px.shape, (2, 2, 4))
        # (0,0): R = round(255 sqrt(0.5/8)) = 64, G = 128 + round(0.25*127) = 160, B = 128 - round(0.525*127) = 61
        # (0,1): 0.01 m is below the 0.02 m threshold -> dry: R = 0, G = B = 128, still inside the domain
        # (1,0): 8.5 m clips to 255; u = -4 -> 1; v = 0.5 -> 128 + round(0.125*127) = 144
        # (1,1): outside the model domain -> everything zero but G = B = 128, A = 0
        want = [64, 160, 61, 255, 0, 128, 128, 255, 255, 1, 144, 255, 0, 128, 128, 0]
        self.assertEqual(px.reshape(-1).tolist(), want)
        self.assertEqual(px.tobytes(), bytes(want))
        px2 = fg.pack_rgba(depth, u, v, None)
        self.assertEqual(px2[1, 1].tolist(), [int(fg.encode_depth(2.0)), 128, int(fg.encode_vel(2.0)), 255])

    def test_pack_flowpaths_layout(self):
        maxd = np.array([[0.5, 0.01], [8.0, 3.0]], np.float32)
        q = np.array([[0.4, 1.0], [2.0, 4.0]], np.float32)
        hrs = np.array([[3, 1], [24, 10]])
        valid = np.array([[True, True], [True, True]])
        px = fg.pack_flowpaths(maxd, q, hrs, valid)
        # R: 64, dry, 255, round(255 sqrt(3/8)) = 156;  G: round(255 sqrt(q/2)) = 114, 0, 255, 255;
        # B: round(255 n/24) = 32, 0, 255, 106
        want = [64, 114, 32, 255, 0, 0, 0, 255, 255, 255, 255, 255, 156, 255, 106, 255]
        self.assertEqual(px.reshape(-1).tolist(), want)
        px = fg.pack_flowpaths(maxd, q, hrs, np.array([[True, True], [True, False]]))
        self.assertEqual(px[1, 1].tolist(), [0, 0, 0, 0])

    def test_gzip_round_trip_and_reproducible_bytes(self):
        rng = np.random.default_rng(4)
        rgba = rng.integers(0, 256, (17, 23, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a.gz"), os.path.join(td, "b.gz")
            fg.write_rgba_gz(a, rgba)
            time.sleep(1.1)                                              # a different mtime must not matter
            fg.write_rgba_gz(b, rgba)
            raw_a, raw_b = Path(a).read_bytes(), Path(b).read_bytes()
            self.assertEqual(raw_a, raw_b)
            self.assertEqual(raw_a[:2], b"\x1f\x8b")                    # RFC 1952 magic
            self.assertEqual(raw_a[3] & 0x08, 0)                        # no file name stored
            self.assertEqual(raw_a[4:8], b"\x00\x00\x00\x00")           # mtime 0
            self.assertEqual(gzip.decompress(raw_a), rgba.tobytes())    # a plain standard-library reader sees the bytes
            back = fg.read_rgba_gz(a, 23, 17)
            np.testing.assert_array_equal(back, rgba)
            self.assertTrue(back.flags.writeable)
            self.assertEqual(sorted(os.listdir(td)), ["a.gz", "b.gz"])      # no stray temp files
            with self.assertRaises(ValueError):
                fg.read_rgba_gz(a, 22, 17)
            with self.assertRaises(ValueError):
                fg.read_rgba_gz(a, 24, 17)
            with open(a, "wb") as f:
                f.write(raw_a[:len(raw_a) // 2])
            with self.assertRaises((EOFError, ValueError)):
                fg.read_rgba_gz(a, 23, 17)
            with self.assertRaises(ValueError):
                fg.write_rgba_gz(a, rgba.astype(np.int16))
            with self.assertRaises(ValueError):
                fg.write_rgba_gz(a, rgba[..., :3])


# ------------------------------------------------------------------------------------------------ 10b
class TestAtomicWrites(unittest.TestCase):
    """
    write_rgba_gz and write_meta go through a temp file and os.replace. On Windows os.replace raises PermissionError
    while the target is open in a reader without FILE_SHARE_DELETE (serve.py streams files with open(path, 'rb')) or
    while another writer is replacing it, so the replace is retried; and a failed write must not leave its temp file.
    """
    ROWS, COLS = 6, 9

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="bgflow_atomic_")
        self.addCleanup(shutil.rmtree, self.td, ignore_errors=True)

    def put(self, kind, path, k, rows=None, cols=None):
        if kind == "rgba":
            fg.write_rgba_gz(path, np.full((rows or self.ROWS, cols or self.COLS, 4), k, np.uint8))
        else:
            fg.write_meta(path, {"k": k})

    def get(self, kind, path):
        """The k a put() stored; fails unless the whole file is one consistent put."""
        if kind == "meta":
            with open(path, encoding="utf-8") as f:
                return json.load(f)["k"]
        arr = fg.read_rgba_gz(path, self.COLS, self.ROWS)
        self.assertTrue((arr == arr[0, 0, 0]).all())
        return int(arr[0, 0, 0])

    def target(self, kind):
        """A target path in a fresh folder, so one failing subtest cannot disturb the next."""
        return os.path.join(tempfile.mkdtemp(dir=self.td), "h01.rgba.gz" if kind == "rgba" else "meta.json")

    @staticmethod
    def listing(dest):
        return sorted(os.listdir(os.path.dirname(dest)))

    def test_a_transient_permission_error_on_replace_is_retried(self):
        real_replace = os.replace
        for kind in ("rgba", "meta"):
            with self.subTest(kind=kind):
                dest = self.target(kind)
                self.put(kind, dest, 1)
                calls = []

                def flaky(src, dst):
                    calls.append(src)
                    if len(calls) <= 3:
                        raise PermissionError(13, "Access is denied", src)
                    real_replace(src, dst)

                with mock.patch.object(os, "replace", side_effect=flaky), mock.patch.object(time, "sleep") as nap:
                    self.put(kind, dest, 2)
                self.assertEqual(len(calls), 4)
                self.assertEqual(self.get(kind, dest), 2)
                delays = [c.args[0] for c in nap.call_args_list]
                self.assertEqual(len(delays), 3)
                self.assertEqual(delays, sorted(delays))                # backs off, never sleeps long
                self.assertTrue(all(0.05 <= d <= 0.5 for d in delays), delays)
                self.assertEqual(self.listing(dest), [os.path.basename(dest)])

    def test_gives_up_after_many_retries_and_leaves_neither_a_temp_file_nor_a_changed_target(self):
        for kind in ("rgba", "meta"):
            with self.subTest(kind=kind):
                dest = self.target(kind)
                self.put(kind, dest, 1)
                with mock.patch.object(os, "replace", side_effect=PermissionError(13, "Access is denied")) as rep, \
                        mock.patch.object(time, "sleep") as nap:
                    with self.assertRaises(PermissionError):
                        self.put(kind, dest, 2)
                self.assertGreaterEqual(rep.call_count, 8)
                self.assertEqual(nap.call_count, rep.call_count - 1)
                self.assertLessEqual(sum(c.args[0] for c in nap.call_args_list), 10.0)     # a stuck write fails in seconds
                self.assertEqual(self.get(kind, dest), 1)
                self.assertEqual(self.listing(dest), [os.path.basename(dest)])

    def test_other_errors_from_replace_are_not_retried(self):
        dest = self.target("meta")
        with mock.patch.object(os, "replace", side_effect=FileNotFoundError(2, "gone")) as rep, \
                mock.patch.object(time, "sleep") as nap:
            with self.assertRaises(FileNotFoundError):
                self.put("meta", dest, 1)
        self.assertEqual((rep.call_count, nap.call_count), (1, 0))
        self.assertEqual(self.listing(dest), [])

    def test_a_reader_holding_the_target_open_only_delays_the_write(self):
        # the real thing on Windows (a reader without FILE_SHARE_DELETE blocks os.replace); elsewhere it passes trivially
        for kind in ("rgba", "meta"):
            with self.subTest(kind=kind):
                dest = self.target(kind)
                self.put(kind, dest, 1)
                reader = open(dest, "rb")
                self.addCleanup(reader.close)
                timer = threading.Timer(0.4, reader.close)
                timer.start()
                self.addCleanup(timer.join)
                t0 = time.time()
                self.put(kind, dest, 2)
                self.assertLess(time.time() - t0, 5.0)
                timer.join()
                self.assertEqual(self.get(kind, dest), 2)
                self.assertEqual(self.listing(dest), [os.path.basename(dest)])

    def test_a_failed_write_removes_its_temp_file_and_keeps_the_old_target(self):
        for kind in ("rgba", "meta"):
            for exc in (OSError(28, "No space left on device"), KeyboardInterrupt()):
                for existing in (False, True):
                    with self.subTest(kind=kind, exc=type(exc).__name__, existing=existing):
                        dest = self.target(kind)
                        if existing:
                            self.put(kind, dest, 1)
                        victim = (mock.patch.object(gzip.GzipFile, "write", side_effect=exc) if kind == "rgba"
                                  else mock.patch.object(json, "dump", side_effect=exc))
                        with victim:
                            with self.assertRaises(type(exc)):
                                self.put(kind, dest, 2)
                        if existing:
                            self.assertEqual(self.get(kind, dest), 1)
                            self.assertEqual(self.listing(dest), [os.path.basename(dest)])
                        else:
                            self.assertEqual(self.listing(dest), [])

    def test_meta_that_cannot_be_serialised_leaves_nothing_behind(self):
        dest = self.target("meta")
        fg.write_meta(dest, {"k": 1})
        with self.assertRaises(TypeError):
            fg.write_meta(dest, {"k": 2, "bad": object()})
        self.assertEqual(self.get("meta", dest), 1)
        self.assertEqual(self.listing(dest), ["meta.json"])

    def test_every_call_uses_its_own_temp_file_next_to_the_target(self):
        real_replace = os.replace
        for kind in ("rgba", "meta"):
            with self.subTest(kind=kind):
                dest = self.target(kind)
                srcs = []

                def spy(src, dst):
                    srcs.append(src)
                    real_replace(src, dst)

                with mock.patch.object(os, "replace", side_effect=spy):
                    for k in (1, 2, 3):
                        self.put(kind, dest, k)
                self.assertEqual(len(set(srcs)), 3, srcs)               # two writers can never share a temp file
                for s in srcs:
                    self.assertEqual(os.path.dirname(os.path.abspath(s)), os.path.dirname(os.path.abspath(dest)))
                    self.assertTrue(s.endswith(".tmp"))
                    self.assertTrue(os.path.basename(s).startswith(os.path.basename(dest)))

    def test_concurrent_writers_to_one_path_all_succeed_and_the_file_is_never_torn(self):
        for kind in ("rgba", "meta"):
            with self.subTest(kind=kind):
                dest = self.target(kind)
                errors, written = [], []
                gate = threading.Barrier(4)

                def work(tid):
                    gate.wait()
                    for i in range(15):
                        k = tid * 15 + i
                        try:
                            self.put(kind, dest, k, rows=64, cols=64)
                            written.append(k)
                        except Exception as e:                           # noqa: BLE001 - the test counts them
                            errors.append("%s: %s" % (type(e).__name__, e))

                threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.assertEqual(errors, [])
                self.assertEqual(len(written), 60)
                self.assertEqual(self.listing(dest), [os.path.basename(dest)])
                if kind == "rgba":
                    arr = fg.read_rgba_gz(dest, 64, 64)
                    self.assertTrue((arr == arr[0, 0, 0]).all())
                    self.assertIn(int(arr[0, 0, 0]), written)
                else:
                    self.assertIn(self.get(kind, dest), written)


# ------------------------------------------------------------------------------------------------ 12
# Independent UTM 16N inverse (Krueger series) and Mercator forward, used to place expected pixels.
_A, _F, _K0 = 6378137.0, 1 / 298.257223563, 0.9996
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_BETA = [_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96 - _N ** 4 / 360,
         _N ** 2 / 48 + _N ** 3 / 15 - 437 * _N ** 4 / 1440,
         17 * _N ** 3 / 480 - 37 * _N ** 4 / 840,
         4397 * _N ** 4 / 161280]
_DELTA = [2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3 + 116 * _N ** 4 / 45,
          7 * _N ** 2 / 3 - 8 * _N ** 3 / 5 - 227 * _N ** 4 / 45,
          56 * _N ** 3 / 15 - 136 * _N ** 4 / 35,
          4279 * _N ** 4 / 630]


def utm16_inverse(e, n):
    xi = np.asarray(n, float) / (_K0 * _AA)
    eta = (np.asarray(e, float) - 500000.0) / (_K0 * _AA)
    xi_p = xi - sum(b * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, b in enumerate(_BETA, 1))
    eta_p = eta - sum(b * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, b in enumerate(_BETA, 1))
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    lat = chi + sum(d * np.sin(2 * j * chi) for j, d in enumerate(_DELTA, 1))
    lon = math.radians(-87.0) + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lon), np.degrees(lat)


def merc_forward(lon, lat):
    lam, phi = np.radians(lon), np.radians(lat)
    return 6378137.0 * lam, 6378137.0 * np.log(np.tan(np.pi / 4 + phi / 2))


class TestWarpAndGrid(unittest.TestCase):
    E0, N0, PX, ROWS, COLS = 549000.0, 4095000.0, 5.0, 240, 320

    def setUp(self):
        from rasterio.transform import from_origin
        self.tr = from_origin(self.E0, self.N0, self.PX, self.PX)
        e = np.array([self.E0, self.E0 + self.COLS * self.PX] * 2)
        n = np.array([self.N0, self.N0, self.N0 - self.ROWS * self.PX, self.N0 - self.ROWS * self.PX])
        x, y = merc_forward(*utm16_inverse(e, n))
        self.grid = fg.grid_from_wet_bounds((x.min(), y.min(), x.max(), y.max()), 5.0, margin_m=40.0)

    def centres(self):
        g = self.grid
        x0, y0, x1, y1 = g["bounds_merc"]
        xs = x0 + (np.arange(g["width"]) + 0.5) * g["cell_m"]
        ys = y1 - (np.arange(g["height"]) + 0.5) * g["cell_m"]
        return np.meshgrid(xs, ys)

    def test_independent_projection_agrees_with_gdal(self):
        """Guards the independent reference used below."""
        from rasterio.warp import transform
        e, n = np.array([549321.0, 560000.0, 480000.0]), np.array([4095123.0, 4100000.0, 4000000.0])
        lon, lat = utm16_inverse(e, n)
        lo2, la2 = transform("EPSG:32616", "EPSG:4326", e.tolist(), n.tolist())
        self.assertLess(float(np.abs(lon - lo2).max()), 1e-8)
        self.assertLess(float(np.abs(lat - la2).max()), 1e-8)
        x, y = merc_forward(lon, lat)
        x2, y2 = fg.lnglat_to_merc(lon, lat)
        np.testing.assert_allclose(x, x2, atol=1e-6)
        np.testing.assert_allclose(y, y2, atol=1e-6)
        xr, yr = transform("EPSG:4326", "EPSG:3857", lon.tolist(), lat.tolist())
        self.assertLess(float(np.abs(np.array(xr) - x).max()), 1e-3)
        self.assertLess(float(np.abs(np.array(yr) - y).max()), 1e-3)
        lb, pb = fg.merc_to_lnglat(x, y)
        np.testing.assert_allclose(lb, lon, atol=1e-10)
        np.testing.assert_allclose(pb, lat, atol=1e-10)
        lo3, la3 = fg.utm16n_to_lnglat(e, n)
        np.testing.assert_allclose(lo3, lon, atol=1e-8)
        self.assertEqual(fg.utm16n_to_lnglat(549321.0, 4095123.0)[0].shape, ())

    def test_constant_field_stays_constant(self):
        arr = np.full((self.ROWS, self.COLS), 3.7, np.float32)
        out = fg.warp_to_grid(arr, self.tr, "EPSG:32616", self.grid, dst_fill=-1.0)
        self.assertEqual(out.shape, (self.grid["height"], self.grid["width"]))
        self.assertEqual(out.dtype, np.float32)
        vals = np.unique(out)
        self.assertEqual(len(vals), 2, msg=str(vals))                  # only the fill value and the constant
        self.assertLess(float(np.abs(vals[vals != -1.0] - 3.7).max()), 1e-5)
        self.assertEqual(float(out[0, 0]), -1.0)                       # the margin outside the source got the fill value
        inner = out[12:-12, 12:-12]
        self.assertLess(float(np.abs(inner - 3.7).max()), 1e-5)

    def test_linear_ramp_in_easting_stays_linear(self):
        b = 0.002
        cols = (np.arange(self.COLS) + 0.5) * self.PX
        arr = np.tile((10.0 + b * cols)[None, :], (self.ROWS, 1)).astype(np.float32)
        out = fg.warp_to_grid(arr, self.tr, "EPSG:32616", self.grid, dst_fill=np.nan)
        X, Y = self.centres()
        # expected: the field is 10 + b * (E - E0) at each pixel centre, E from the independent Mercator -> UTM
        from rasterio.warp import transform
        lon, lat = fg.merc_to_lnglat(X[20:-20:9, 20:-20:9], Y[20:-20:9, 20:-20:9])
        e, _ = transform("EPSG:4326", "EPSG:32616", lon.ravel().tolist(), lat.ravel().tolist())
        want = 10.0 + b * (np.array(e).reshape(lon.shape) - self.E0)
        got = out[20:-20:9, 20:-20:9]
        inside = np.isfinite(got)
        self.assertGreater(int(inside.sum()), 500)
        self.assertLess(float(np.abs(got[inside] - want[inside]).max()), 5e-3)     # b * 2.5 m = 5e-3 is half a source cell
        # exactly linear along a row of the output (a ramp in easting is not constant along it only by the rotation)
        row = out[self.grid["height"] // 2, 30:-30].astype(np.float64)
        d2 = np.diff(row, 2)
        self.assertLess(float(np.abs(d2).max()), 5e-3)

    def test_marker_cell_lands_on_the_expected_mercator_pixel(self):
        for method, tol_cells in (("bilinear", 1.0), ("nearest", 1.0)):
            r, c = 130, 170
            arr = np.zeros((self.ROWS, self.COLS), np.float32)
            arr[r, c] = 1.0
            out = fg.warp_to_grid(arr, self.tr, "EPSG:32616", self.grid, resampling=method)
            e = self.E0 + (c + 0.5) * self.PX
            n = self.N0 - (r + 0.5) * self.PX
            xe, ye = merc_forward(*utm16_inverse(e, n))
            X, Y = self.centres()
            wsum = float(out.sum())
            self.assertGreater(wsum, 0.5, msg=method)
            cx = float((out * X).sum()) / wsum
            cy = float((out * Y).sum()) / wsum
            cell = self.grid["cell_m"]
            self.assertLess(math.hypot(cx - xe, cy - ye), tol_cells * cell, msg=method)
            if method == "bilinear":
                self.assertLess(math.hypot(cx - xe, cy - ye), 0.35 * cell)
            i, j = np.unravel_index(np.argmax(out), out.shape)
            self.assertLess(math.hypot(X[i, j] - xe, Y[i, j] - ye), 1.5 * cell)

    def test_exact_zeros_stay_exact_zeros(self):
        """GDAL turns a computed 0.0 into 1.4e-45 when the destination nodata is 0.0; masks like depth > 0 must not see that."""
        zeros = np.zeros((self.ROWS, self.COLS), np.float32)
        half = zeros.copy()
        half[:, 160:] = 2.0
        for kw in ({}, {"src_nodata": -9999.0}, {"dst_fill": -1.0}, {"resampling": "nearest"}, {"resampling": "average"}):
            out = fg.warp_to_grid(zeros, self.tr, "EPSG:32616", self.grid, **kw)
            fill = kw.get("dst_fill", 0.0)
            inner = out[15:-15, 15:-15]
            self.assertEqual(float(np.abs(inner).max()), 0.0, msg=str(kw))
            self.assertEqual(float(out[0, 0]), fill, msg=str(kw))
            out = fg.warp_to_grid(half, self.tr, "EPSG:32616", self.grid, **kw)
            row = out[out.shape[0] // 2, 15:-15]
            self.assertEqual(float(row.min()), 0.0, msg=str(kw))
            self.assertEqual(float(row.max()), 2.0, msg=str(kw))
            nonzero = row[row != 0]
            self.assertGreater(float(nonzero.min()), 1e-6, msg=str(kw))       # no denormals between 0 and the first real value

    def test_source_nodata_is_ignored(self):
        arr = np.full((self.ROWS, self.COLS), 3.7, np.float32)
        arr[100:120, 100:130] = -9999.0
        out = fg.warp_to_grid(arr, self.tr, "EPSG:32616", self.grid, src_nodata=-9999.0, dst_fill=0.0)
        vals = out[out != 0.0]
        self.assertLess(float(np.abs(vals - 3.7).max()), 1e-4)        # no -9999 bleeding into the neighbours
        e = self.E0 + 115.0 * self.PX                                 # centre of the hole (rows 100-119, cols 100-129)
        n = self.N0 - 110.0 * self.PX
        xh, yh = merc_forward(*utm16_inverse(e, n))
        x0, _, _, y1 = self.grid["bounds_merc"]
        j, i = int((xh - x0) / self.grid["cell_m"]), int((y1 - yh) / self.grid["cell_m"])
        self.assertTrue(np.all(out[i - 5:i + 6, j - 5:j + 6] == 0.0))  # the hole itself is empty
        self.assertAlmostEqual(float(out[i - 25, j]), 3.7, places=4)    # and its surroundings are not

    def test_grid_from_wet_bounds_snaps_outward_to_whole_cells(self):
        b = (-9620000.3, 4400000.7, -9610000.1, 4410000.9)
        g = fg.grid_from_wet_bounds(b, 5.0, margin_m=150.0)
        x0, y0, x1, y1 = g["bounds_merc"]
        for v in (x0, y0, x1, y1):
            self.assertEqual(v % 5.0, 0.0)
        self.assertLessEqual(x0, b[0] - 150.0)
        self.assertLess(x0, b[0] - 150.0 + 5.0)
        self.assertGreaterEqual(x1, b[2] + 150.0)
        self.assertGreater(x1, b[2] + 150.0 - 5.0)
        self.assertLessEqual(y0, b[1] - 150.0)
        self.assertGreaterEqual(y1, b[3] + 150.0)
        self.assertEqual(g["width"] * 5.0, x1 - x0)
        self.assertEqual(g["height"] * 5.0, y1 - y0)
        self.assertEqual(g["cell_m"], 5.0)
        t = g["transform"]
        self.assertEqual((t.a, t.b, t.c, t.d, t.e, t.f), (5.0, 0.0, x0, 0.0, -5.0, y1))          # row 0 is the north edge
        self.assertEqual((g["width"], g["height"]), (2061, 2061))
        # a grid already on the cell lattice gains exactly the margin
        g2 = fg.grid_from_wet_bounds((0.0, 0.0, 1000.0, 500.0), 10.0, margin_m=100.0)
        self.assertEqual(g2["bounds_merc"], [-100.0, -100.0, 1100.0, 600.0])
        self.assertEqual((g2["width"], g2["height"]), (120, 70))

    def test_grid_from_wet_bounds_refuses_oversize(self):
        b = (-9650000.0, 4400000.0, -9600000.0, 4440000.0)             # 50 km x 40 km
        with self.assertRaises(ValueError) as cm:
            fg.grid_from_wet_bounds(b, 5.0)
        self.assertIn("cell_m", str(cm.exception))
        g = fg.grid_from_wet_bounds(b, 10.0)                            # 5030 wide: fits
        self.assertLessEqual(max(g["width"], g["height"]), 6000)
        with self.assertRaises(ValueError):
            fg.grid_from_wet_bounds(b, 8.0)                            # 6288 wide
        g = fg.grid_from_wet_bounds(b, 8.0, max_dim=7000)
        self.assertEqual(g["width"], 6288)
        with self.assertRaises(ValueError):
            fg.grid_from_wet_bounds((1, 1, 1, 5), 5.0)


# ------------------------------------------------------------------------------------------------ stats and meta
class TestStatsAndMeta(unittest.TestCase):
    def test_hour_stats(self):
        depth = np.array([[0.0, 0.01, 0.5], [1.0, 9.0, np.nan]], np.float32)
        u = np.array([[0, 50.0, 1.0], [2.0, 0, 7.0]], np.float32)
        v = np.array([[0, 0, 0.0], [0.0, 3.0, 7.0]], np.float32)
        s = fg.hour_stats(depth, u, v, 4.0)
        self.assertEqual(s["wet_cells"], 3)
        self.assertAlmostEqual(s["wet_area_km2"], 3 * 16 / 1e6, places=9)
        self.assertEqual(s["max_depth_m"], 9.0)
        self.assertAlmostEqual(s["mean_depth_m"], 3.5, places=6)
        self.assertEqual(s["max_speed_ms"], 3.0)             # the 50 m/s on a dry cell and the NaN cell are ignored
        self.assertAlmostEqual(s["volume_m3"], 10.5 * 16, places=2)
        self.assertEqual(s["clipped_cells"], 1)
        z = fg.hour_stats(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), 4.0)
        self.assertEqual(z, {"wet_cells": 0, "wet_area_km2": 0.0, "max_depth_m": 0.0, "mean_depth_m": 0.0,
                             "max_speed_ms": 0.0, "volume_m3": 0.0, "clipped_cells": 0})
        e = fg.hourly_entry(3, depth, u, v, 4.0)
        self.assertEqual((e["hour"], e["sim_seconds"]), (3, 10800))
        json.dumps(e)

    def test_build_meta_matches_the_documented_layout(self):
        grid = fg.grid_from_wet_bounds((-9622000.0, 4411000.0, -9619000.0, 4413000.0), 5.0)
        hourly = [fg.hourly_entry(h, np.full((4, 4), 0.1 * h, np.float32), None, None, 4.0) for h in (3, 4)]
        hourly[0]["wet_cells"] = np.int64(hourly[0]["wet_cells"])            # numpy scalars must serialise
        meta = fg.build_meta("5yr", "5-year storm", "rasters", grid, np.array([3, 4]), hourly, 4.2, run_summary={"a": 1})
        want_keys = ["format", "version", "storm", "label", "source", "source_note", "velocity_method", "crs", "width",
                     "height", "cell_m", "cell_ground_m", "bounds_merc", "bounds_lnglat", "depth_encoding", "vel_encoding",
                     "wet_threshold_m", "qmax_m2s", "hours", "storm_depth_in", "run_summary", "hourly", "built"]
        self.assertEqual(list(meta), want_keys)
        self.assertEqual((meta["format"], meta["version"], meta["crs"]), ("bgflow", 1, "EPSG:3857"))
        self.assertEqual(meta["depth_encoding"], {"type": "sqrt", "max_m": 8.0})
        self.assertEqual(meta["vel_encoding"], {"vmax_ms": 4.0, "zero_code": 128})
        self.assertEqual((meta["wet_threshold_m"], meta["qmax_m2s"]), (0.02, 2.0))
        self.assertEqual(meta["hours"], [3, 4])
        self.assertEqual(meta["width"], grid["width"])
        x0, y0, x1, y1 = meta["bounds_merc"]
        w, s = fg.merc_to_lnglat(x0, y0)
        e, n = fg.merc_to_lnglat(x1, y1)
        np.testing.assert_allclose(meta["bounds_lnglat"], [w, s, e, n], atol=1e-6)
        lat = float(fg.merc_to_lnglat(0, 0.5 * (y0 + y1))[1])
        self.assertAlmostEqual(meta["cell_ground_m"], 5.0 * math.cos(math.radians(lat)), places=3)
        self.assertAlmostEqual(meta["cell_ground_m"], 4.0, delta=0.05)
        self.assertIn("derived", meta["source_note"])
        self.assertIn("not solver output", meta["velocity_method"])
        t = datetime.datetime.fromisoformat(meta["built"].replace("Z", "+00:00"))
        self.assertLess(abs((datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()), 60)
        self.assertEqual(meta["hourly"][1]["sim_seconds"], 14400)
        tiers = fg.build_meta("5yr", "x", "tiers", grid, [], [], None)
        self.assertIn("tier", tiers["source_note"])
        self.assertIsNone(tiers["run_summary"])
        with self.assertRaises(ValueError):
            fg.build_meta("5yr", "x", "bogus", grid, [], [], None)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "meta.json")
            fg.write_meta(p, meta)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(json.load(f), meta)
            self.assertEqual(sorted(os.listdir(td)), ["meta.json"])


# ------------------------------------------------------------------------------------------------ 11
class TestReadDem(unittest.TestCase):
    """dem_bg.tif stores FEET (cells in metres); face_velocity needs metres. read_dem_m is the one place that converts."""
    APP_DEM = Path("C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map/dem_bg.tif")

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgflow_dem_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def write_dem(self, name, arr, nodata=-9999.0):
        import rasterio
        from rasterio.transform import Affine
        p = os.path.join(self.tmp, name)
        with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                           crs="EPSG:32616", transform=Affine(5.0, 0.0, 500000.0, 0.0, -5.0, 4100000.0), nodata=nodata) as ds:
            ds.write(arr.astype(np.float32), 1)
        return p

    def test_feet_become_metres_and_nodata_becomes_nan(self):
        arr = np.array([[100.0, 200.0, -9999.0], [540.0, -9999.0, 1000.0]], np.float32)
        z, tr, crs = fg.read_dem_m(self.write_dem("ft.tif", arr), units="feet")
        self.assertEqual(z.dtype, np.float32)
        self.assertEqual(z.shape, (2, 3))
        want = np.where(arr == -9999.0, np.nan, arr * 0.3048)
        np.testing.assert_allclose(z, want, rtol=1e-6)
        self.assertTrue(np.isnan(z[0, 2]) and np.isnan(z[1, 1]))             # not -3048 m
        self.assertEqual((tr.a, tr.e, tr.c, tr.f), (5.0, -5.0, 500000.0, 4100000.0))
        self.assertEqual(crs.to_epsg(), 32616)
        self.assertAlmostEqual(fg.FEET_TO_M, 0.3048, places=12)

    def test_metres_pass_through_and_the_known_file_names_pick_the_units(self):
        arr = np.array([[160.0, 165.0], [170.0, -9999.0]], np.float32)
        z, _, _ = fg.read_dem_m(self.write_dem("dem_bg_2m.tif", arr))          # Vista terrain: already metres
        np.testing.assert_array_equal(z[:1], arr[:1])
        self.assertTrue(np.isnan(z[1, 1]))
        z, _, _ = fg.read_dem_m(self.write_dem("dem_bg.tif", arr))             # local terrain: feet
        np.testing.assert_allclose(z[0, 0], 160.0 * 0.3048, rtol=1e-6)

    def test_unknown_units_are_refused_instead_of_guessed(self):
        p = self.write_dem("something_else.tif", np.ones((2, 2), np.float32))
        with self.assertRaises(ValueError):
            fg.read_dem_m(p)
        with self.assertRaises(ValueError):
            fg.read_dem_m(p, units="yards")
        z, _, _ = fg.read_dem_m(p, units="metres")
        np.testing.assert_array_equal(z, np.ones((2, 2), np.float32))

    def test_a_feet_dem_read_as_metres_would_give_the_wrong_speed(self):
        """Why it matters: the same lake-edge surface step, run with the raw feet values, is 3.28x too steep."""
        arr = (540.0 + 0.05 * np.arange(30, dtype=np.float32))[None, :].repeat(10, axis=0)      # 0.05 ft per cell: below the Froude cap
        z, _, _ = fg.read_dem_m(self.write_dem("ramp.tif", arr), units="feet")
        h = np.full(arr.shape, 0.2, np.float32)
        u_ok, _ = fg.face_velocity(h, z, 0.035, 5.0)
        u_raw, _ = fg.face_velocity(h, arr, 0.035, 5.0)
        self.assertAlmostEqual(float(u_raw[5, 15] / u_ok[5, 15]), math.sqrt(1 / 0.3048), delta=0.01)   # sqrt of the slope ratio

    @unittest.skipUnless(APP_DEM.exists(), "dem_bg.tif is not on this machine")
    def test_real_dem_bg_is_at_bowling_green_height_in_metres(self):
        z, tr, crs = fg.read_dem_m(str(self.APP_DEM))
        self.assertEqual(z.shape, (3500, 4096))
        self.assertEqual(crs.to_epsg(), 32616)
        self.assertEqual(int(np.isnan(z).sum()), 0)
        mean = float(np.mean(z, dtype=np.float64))
        self.assertTrue(120.0 < mean < 220.0, "mean elevation %.1f m" % mean)          # about 165 m; the raw file says 541


# ------------------------------------------------------------------------------------------------ 13
class SyntheticStorm:
    """A small but internally consistent storm folder written with the library itself."""
    W, H = 48, 36
    HOURS = (2, 3, 5)

    @classmethod
    def build(cls, path, w=None, h=None, peak=0.15):
        w = w or cls.W
        h = h or cls.H
        os.makedirs(path, exist_ok=True)
        x0, y0 = -9621000.0, 4410000.0
        grid = fg.grid_from_wet_bounds((x0, y0, x0 + w * 5.0, y0 + h * 5.0), 5.0, margin_m=0.0)
        assert (grid["width"], grid["height"]) == (w, h)
        rng = np.random.default_rng(21)
        row, col = np.mgrid[0:h, 0:w]
        valid = col < w - 3
        cg = 5.0 * math.cos(math.radians(36.9))
        hourly, d_all, q_all = [], [], []
        for hr in cls.HOURS:
            blob = np.exp(-((row - 18) ** 2 + (col - 20) ** 2) / (2 * (6.0 + hr) ** 2))
            depth = np.where(valid, (peak + 0.3 * hr) * blob, 0.0).astype(np.float32)
            depth[depth < 0.02] = 0.0
            bound = np.minimum(np.sqrt(G * depth.astype(np.float64)), 4.0)
            sp = bound * rng.random((h, w)) * 0.8
            ang = rng.random((h, w)) * 2 * np.pi
            u, v = (sp * np.cos(ang)).astype(np.float32), (sp * np.sin(ang)).astype(np.float32)
            fg.write_rgba_gz(os.path.join(path, fg.hour_filename(hr)), fg.pack_rgba(depth, u, v, valid))
            hourly.append(fg.hourly_entry(hr, depth, u, v, cg))
            d_all.append(depth)
            q_all.append(depth * np.hypot(u, v) * (depth >= fg.WET_THRESHOLD))
        max_d = np.max(d_all, axis=0)
        mean_q = np.mean(q_all, axis=0)
        count = np.sum([d >= fg.WET_THRESHOLD for d in d_all], axis=0)
        fg.write_rgba_gz(os.path.join(path, "flowpaths.rgba.gz"), fg.pack_flowpaths(max_d, mean_q, count, valid))
        meta = fg.build_meta("5yr", "5-year storm", "rasters", grid, cls.HOURS, hourly, 4.2, run_summary={"mass_error_pct": 0.1})
        fg.write_meta(os.path.join(path, "meta.json"), meta)
        return meta


def build_pit_storm(path, pit_depth, pit_speed, w=40, h=30):
    """
    One hour of 0.3 m of water drifting east at 0.1 m/s with a 5 x 5 pit of pit_depth m moving at pit_speed m/s.
    The flowpaths summary is built the way the real builders do it, from the TRUE depth * speed (FLOW_FORMAT.md),
    while the hour file clips the depth at 8 m.
    """
    os.makedirs(path)
    x0, y0 = -9621000.0, 4410000.0
    grid = fg.grid_from_wet_bounds((x0, y0, x0 + w * 5.0, y0 + h * 5.0), 5.0, margin_m=0.0)
    cg = 5.0 * math.cos(math.radians(36.9))
    depth = np.full((h, w), 0.3, np.float32)
    u = np.full((h, w), 0.1, np.float32)
    v = np.zeros((h, w), np.float32)
    pit = (slice(10, 15), slice(10, 15))
    depth[pit] = pit_depth
    u[pit] = pit_speed
    valid = np.ones((h, w), bool)
    fg.write_rgba_gz(os.path.join(path, "h01.rgba.gz"), fg.pack_rgba(depth, u, v, valid))
    fg.write_rgba_gz(os.path.join(path, "flowpaths.rgba.gz"),
                     fg.pack_flowpaths(depth, depth * np.hypot(u, v), np.ones((h, w), int), valid))
    meta = fg.build_meta("5yr", "pit", "rasters", grid, [1], [fg.hourly_entry(1, depth, u, v, cg)], 4.2)
    fg.write_meta(os.path.join(path, "meta.json"), meta)
    return meta


class TestValidateStormDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgflow_test_")
        cls.good = os.path.join(cls.tmp, "good")
        SyntheticStorm.build(cls.good)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def corrupt_copy(self, name):
        d = os.path.join(self.tmp, name)
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(self.good, d)
        return d

    @staticmethod
    def load_meta(d):
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def edit_hour(self, d, hour, fn):
        meta = self.load_meta(d)
        p = os.path.join(d, fg.hour_filename(hour) if hour else "flowpaths.rgba.gz")
        rgba = fg.read_rgba_gz(p, meta["width"], meta["height"]).copy()
        fn(rgba)
        fg.write_rgba_gz(p, rgba)

    def edit_meta(self, d, fn):
        meta = self.load_meta(d)
        fn(meta)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def assertReported(self, problems, *words):
        text = "\n".join(problems).lower()
        self.assertTrue(problems, "nothing reported")
        for w in words:
            self.assertIn(w.lower(), text, msg="expected %r in:\n%s" % (w, "\n".join(problems)))

    def test_stray_temp_files_of_an_interrupted_write_are_reported(self):
        d = self.corrupt_copy("stray_tmp")
        self.assertEqual(fg.validate_storm_dir(d), [])
        for name in ("h03.rgba.gz.tmp", "meta.json.4242.0123456789ab.tmp"):
            Path(d, name).write_bytes(b"partial")
            problems = fg.validate_storm_dir(d)
            self.assertReported(problems, name, "temporary")
            self.assertEqual(len([p for p in problems if name in p]), 1)
            os.remove(os.path.join(d, name))
        self.assertEqual(fg.validate_storm_dir(d), [])
        # also when nothing else in the folder is usable: a crashed first write leaves only the temp file
        empty = os.path.join(self.tmp, "only_tmp")
        os.makedirs(empty)
        Path(empty, "meta.json.1.abc.tmp").write_bytes(b"{")
        self.assertReported(fg.validate_storm_dir(empty), "meta.json.1.abc.tmp", "meta.json: missing")

    def test_depth_above_8_m_is_clipped_in_the_pixels_but_not_in_the_stats(self):
        d = os.path.join(self.tmp, "deep")
        meta = SyntheticStorm.build(d, peak=14.0)                        # centre depths of 14-15 m: the pixels clip at 8 m
        self.assertGreater(max(e["clipped_cells"] for e in meta["hourly"]), 20)
        self.assertGreater(max(e["max_depth_m"] for e in meta["hourly"]), 8.0)
        self.assertEqual(fg.validate_storm_dir(d), [])
        # but a volume that is far too LARGE for the pixels is still reported
        self.edit_meta(d, lambda m: m["hourly"][1].__setitem__("volume_m3", m["hourly"][1]["volume_m3"] * 0.3))
        self.assertReported(fg.validate_storm_dir(d), "volume")

    def test_flowpaths_of_a_pit_deeper_than_8_m_with_slow_flow(self):
        """
        The hour file clips depth at 8 m but flowpaths G is q = true depth * speed. 12 m of water creeping at 0.10 m/s
        carries 1.2 m2/s, more than the clipped 8 m pixel and its slow decoded speed can account for, yet it is valid.
        """
        for pit_depth, pit_speed in [(12, 0.10), (20, 0.05), (12, 0.05), (7.5, 0.10), (20, 0.02), (20, 0.0), (30, 0.2)]:
            with self.subTest(depth=pit_depth, speed=pit_speed):
                d = os.path.join(self.tmp, "pit_%g_%g" % (pit_depth, pit_speed))
                meta = build_pit_storm(d, pit_depth, pit_speed)
                self.assertEqual(meta["hourly"][0]["clipped_cells"], 25 if pit_depth > 8 else 0)
                self.assertEqual(fg.validate_storm_dir(d), [])

    def test_flowpaths_discharge_is_still_checked_beside_a_clipped_pit(self):
        """Only pixels at the depth clip (R = 255) have an unknown true depth; the 0.3 m water around the pit does not."""
        d = os.path.join(self.tmp, "pit_neg")
        build_pit_storm(d, 12, 0.10)
        self.assertEqual(fg.validate_storm_dir(d), [])
        self.edit_hour(d, 0, lambda a: a.__setitem__((2, 2, 1), 255))      # 0.3 m of water claiming QMAX = 2 m2/s
        self.assertReported(fg.validate_storm_dir(d), "flowpaths.rgba.gz", "unit discharge", "at 1 pixels")

    def test_data_exactly_at_the_froude_bound_survives_quantisation(self):
        """
        Capped flow sits exactly on min(sqrt(g d), 4); rounding the depth code down and the velocity codes up must
        not make it look invalid (found on real terrain, where most wet cells are at the cap).
        """
        w, h = 90, 70
        d = os.path.join(self.tmp, "atbound")
        os.makedirs(d)
        rng = np.random.default_rng(13)
        depth = (0.02 + rng.random((h, w)) ** 2 * 3.0).astype(np.float32)
        depth[:5] = 0.02 + rng.random((5, w)).astype(np.float32) * 0.01         # many thin films, where the bound is steepest
        sp = np.minimum(np.sqrt(G * depth.astype(np.float64)), 4.0)
        ang = rng.random((h, w)) * 2 * np.pi
        u, v = (sp * np.cos(ang)).astype(np.float32), (sp * np.sin(ang)).astype(np.float32)
        valid = np.ones((h, w), bool)
        x0, y0 = -9621000.0, 4410000.0
        grid = fg.grid_from_wet_bounds((x0, y0, x0 + w * 5.0, y0 + h * 5.0), 5.0, margin_m=0.0)
        cg = 5.0 * math.cos(math.radians(36.9))
        fg.write_rgba_gz(os.path.join(d, "h04.rgba.gz"), fg.pack_rgba(depth, u, v, valid))
        wet = depth >= fg.WET_THRESHOLD
        fg.write_rgba_gz(os.path.join(d, "flowpaths.rgba.gz"),
                         fg.pack_flowpaths(depth, depth * np.hypot(u, v), wet.astype(int), valid))
        fg.write_meta(os.path.join(d, "meta.json"),
                      fg.build_meta("5yr", "at the bound", "rasters", grid, [4], [fg.hourly_entry(4, depth, u, v, cg)], 4.2))
        self.assertEqual(fg.validate_storm_dir(d), [])
        # it is the quantised file that sits above the bound at some pixels, so the check must have real slack
        px = fg.read_rgba_gz(os.path.join(d, "h04.rgba.gz"), w, h)
        dec = fg.decode_depth(px[..., 0]).astype(np.float64)
        spd = np.hypot(fg.decode_vel(px[..., 1]), fg.decode_vel(px[..., 2])).astype(np.float64)
        self.assertGreater(float((spd - np.sqrt(G * dec)).max()), 0.01)

    def test_valid_folder_has_no_problems(self):
        self.assertEqual(fg.validate_storm_dir(self.good), [])
        meta = self.load_meta(self.good)
        self.assertEqual(meta["hours"], [2, 3, 5])
        rgba = fg.read_rgba_gz(os.path.join(self.good, "h03.rgba.gz"), meta["width"], meta["height"])
        self.assertGreater(int((rgba[..., 0] > 0).sum()), 100)          # the fixture is not trivially empty
        self.assertGreater(int((rgba[..., 1] != 128).sum()), 50)

    def test_truncated_gzip(self):
        d = self.corrupt_copy("c_trunc")
        p = os.path.join(d, "h03.rgba.gz")
        raw = Path(p).read_bytes()
        Path(p).write_bytes(raw[: len(raw) // 2])
        self.assertReported(fg.validate_storm_dir(d), "h03.rgba.gz")

    def test_wrong_width_in_meta(self):
        d = self.corrupt_copy("c_width")
        self.edit_meta(d, lambda m: m.__setitem__("width", m["width"] - 1))
        pr = fg.validate_storm_dir(d)
        self.assertReported(pr, "meta.width", "width*height*4")

    def test_alpha_channel_value_7(self):
        d = self.corrupt_copy("c_alpha")
        self.edit_hour(d, 2, lambda a: a.__setitem__((0, 0, 3), 7))
        self.assertReported(fg.validate_storm_dir(d), "h02.rgba.gz", "alpha")

    def test_speed_above_the_froude_bound(self):
        d = self.corrupt_copy("c_froude")

        def fn(a):
            a[18, 20] = [60, 255, 128, 255]          # 0.44 m deep, 4 m/s east
        self.edit_hour(d, 3, fn)
        self.assertReported(fg.validate_storm_dir(d), "h03.rgba.gz", "faster")

    def test_speed_above_vmax_in_deep_water(self):
        """At 4 m depth sqrt(g d) = 6.3 m/s, so only the 4 m/s limit can catch a 5.7 m/s pixel."""
        d = self.corrupt_copy("c_vmax")

        def fn(a):
            a[18, 20] = [181, 255, 255, 255]          # 4.03 m deep, u = v = 4 m/s
        self.edit_hour(d, 3, fn)
        self.assertReported(fg.validate_storm_dir(d), "h03.rgba.gz", "faster", "4 m/s")

    def test_flowpaths_discharge_above_what_the_hours_allow(self):
        d = self.corrupt_copy("c_fpq")
        self.edit_hour(d, 0, lambda a: a.__setitem__((a[..., 0] > 0, 1), 255))
        self.assertReported(fg.validate_storm_dir(d), "flowpaths.rgba.gz", "unit discharge")

    def test_valid_domain_differs_between_hours(self):
        d = self.corrupt_copy("c_adiff")
        self.edit_hour(d, 5, lambda a: a.__setitem__((2, 2, 3), 0))
        self.assertReported(fg.validate_storm_dir(d), "h05.rgba.gz", "valid-domain")
        d = self.corrupt_copy("c_adiff_fp")
        self.edit_hour(d, 0, lambda a: a.__setitem__((2, 2, 3), 0))
        self.assertReported(fg.validate_storm_dir(d), "flowpaths.rgba.gz", "valid-domain")

    def test_missing_hour_file(self):
        d = self.corrupt_copy("c_missing")
        os.remove(os.path.join(d, "h05.rgba.gz"))
        self.assertReported(fg.validate_storm_dir(d), "h05.rgba.gz", "missing")

    def test_bad_flowpaths_size(self):
        d = self.corrupt_copy("c_fpsize")
        meta = self.load_meta(d)
        fg.write_rgba_gz(os.path.join(d, "flowpaths.rgba.gz"), np.zeros((meta["height"], meta["width"] - 1, 4), np.uint8))
        self.assertReported(fg.validate_storm_dir(d), "flowpaths.rgba.gz", "width*height*4")

    # ---- further constraints
    def test_extra_hour_file_not_in_meta(self):
        d = self.corrupt_copy("c_extra")
        shutil.copy(os.path.join(d, "h02.rgba.gz"), os.path.join(d, "h07.rgba.gz"))
        self.assertReported(fg.validate_storm_dir(d), "h07.rgba.gz", "not in meta.hours")

    def test_velocity_on_a_dry_pixel(self):
        d = self.corrupt_copy("c_dryvel")
        self.edit_hour(d, 2, lambda a: a.__setitem__((1, 1), [0, 150, 128, 255]))
        self.assertReported(fg.validate_storm_dir(d), "dry pixels", "velocity")

    def test_depth_below_wet_threshold(self):
        d = self.corrupt_copy("c_thresh")
        self.edit_hour(d, 2, lambda a: a.__setitem__((1, 1), [5, 128, 128, 255]))
        self.assertReported(fg.validate_storm_dir(d), "below the wet threshold")

    def test_water_outside_the_domain(self):
        d = self.corrupt_copy("c_outside")
        self.edit_hour(d, 2, lambda a: a.__setitem__((5, 46), [40, 128, 128, 0]))
        self.assertReported(fg.validate_storm_dir(d), "outside the valid domain")

    def test_flowpaths_depth_below_an_hour(self):
        d = self.corrupt_copy("c_fpdepth")
        self.edit_hour(d, 0, lambda a: a.__setitem__((18, 20, 0), 5))
        self.assertReported(fg.validate_storm_dir(d), "flowpaths.rgba.gz")

    def test_flowpaths_hours_wet_wrong(self):
        d = self.corrupt_copy("c_fphours")
        self.edit_hour(d, 0, lambda a: a.__setitem__((18, 20, 2), 255))
        self.assertReported(fg.validate_storm_dir(d), "hours-wet")

    def test_hourly_stats_that_do_not_match_the_pixels(self):
        d = self.corrupt_copy("c_stats")

        def fn(m):
            for e in m["hourly"]:
                if e["hour"] == 3:
                    e["volume_m3"] *= 3.0
        self.edit_meta(d, fn)
        self.assertReported(fg.validate_storm_dir(d), "h03.rgba.gz", "volume")

    def test_meta_field_problems(self):
        cases = [
            ("m_format", lambda m: m.__setitem__("format", "other"), "meta.format"),
            ("m_source", lambda m: m.__setitem__("source", "magic"), "meta.source"),
            ("m_crs", lambda m: m.__setitem__("crs", "EPSG:4326"), "meta.crs"),
            ("m_enc", lambda m: m["depth_encoding"].__setitem__("max_m", 10.0), "meta.depth_encoding"),
            ("m_hours", lambda m: m.__setitem__("hours", [3, 2, 5]), "meta.hours"),
            ("m_built", lambda m: m.__setitem__("built", "yesterday"), "meta.built"),
            ("m_bounds", lambda m: m.__setitem__("bounds_lnglat", [0, 0, 1, 1]), "meta.bounds_lnglat"),
            ("m_hourly", lambda m: m["hourly"].pop(), "meta.hourly"),
            ("m_missing", lambda m: m.pop("qmax_m2s"), "meta.qmax_m2s"),
            ("m_type", lambda m: m.__setitem__("height", "36"), "meta.height"),
            ("m_cg", lambda m: m.__setitem__("cell_ground_m", 9.0), "cell_ground_m"),
        ]
        for name, fn, word in cases:
            d = self.corrupt_copy(name)
            self.edit_meta(d, fn)
            self.assertReported(fg.validate_storm_dir(d), word)

    def test_missing_or_unreadable_meta_and_flowpaths(self):
        d = self.corrupt_copy("c_nometa")
        os.remove(os.path.join(d, "meta.json"))
        self.assertReported(fg.validate_storm_dir(d), "meta.json")
        Path(d, "meta.json").write_text("{not json")
        self.assertReported(fg.validate_storm_dir(d), "unreadable")
        d = self.corrupt_copy("c_nofp")
        os.remove(os.path.join(d, "flowpaths.rgba.gz"))
        self.assertReported(fg.validate_storm_dir(d), "flowpaths.rgba.gz", "missing")
        self.assertReported(fg.validate_storm_dir(os.path.join(self.tmp, "nope")), "not a directory")

    def check_reported_not_raised(self, name, fn, *words):
        d = self.corrupt_copy(name)
        self.edit_meta(d, fn)
        try:
            problems = fg.validate_storm_dir(d)
        except Exception as e:                  # the point of the validator is to REPORT, never to crash
            self.fail("%s: validate_storm_dir raised %s: %s" % (name, type(e).__name__, e))
        self.assertIsInstance(problems, list)
        self.assertTrue(all(isinstance(p, str) for p in problems))
        self.assertFalse([p for p in problems if p.startswith("validator error")],
                         "%s: only the catch-all caught it, a specific check is missing: %s" % (name, problems))
        self.assertReported(problems, *words)

    def test_incomplete_hourly_entries_are_reported_not_raised(self):
        for k in list(self.load_meta(self.good)["hourly"][0]):
            self.check_reported_not_raised("h_drop_" + k, lambda m, k=k: m["hourly"][1].pop(k), "meta.hourly", k)
            for i, val in enumerate((None, "x", [1], {"a": 1}, True, -1)):
                self.check_reported_not_raised("h_set_%s_%d" % (k, i), lambda m, k=k, val=val: m["hourly"][1].__setitem__(k, val),
                                               "meta.hourly")

    def test_odd_meta_values_are_reported_not_raised(self):
        cases = [
            ("hourly_3", lambda m: m.__setitem__("hourly", 3), "meta.hourly"),
            ("hourly_null", lambda m: m.__setitem__("hourly", None), "meta.hourly"),
            ("hourly_str", lambda m: m.__setitem__("hourly", "abc"), "meta.hourly"),
            ("hourly_dict", lambda m: m.__setitem__("hourly", {"hour": 2}), "meta.hourly"),
            ("hourly_nulls", lambda m: m.__setitem__("hourly", [None, 1, "x"]), "meta.hourly"),
            ("hour_list", lambda m: m["hourly"][0].__setitem__("hour", [2]), "meta.hourly"),
            ("hour_dict", lambda m: m["hourly"][0].__setitem__("hour", {"a": 1}), "meta.hourly"),
            ("source_list", lambda m: m.__setitem__("source", ["rasters"]), "meta.source"),
            ("source_dict", lambda m: m.__setitem__("source", {"a": 1}), "meta.source"),
            ("hours_3", lambda m: m.__setitem__("hours", 3), "meta.hours"),
            ("hours_null", lambda m: m.__setitem__("hours", None), "meta.hours"),
            ("hours_nested", lambda m: m.__setitem__("hours", [[2], 3, 5]), "meta.hours"),
            ("hours_huge", lambda m: m.__setitem__("hours", [2, 3, 10 ** 400]), "meta.hours"),
            ("wh_1e9", lambda m: (m.__setitem__("width", 10 ** 9), m.__setitem__("height", 10 ** 9)), "meta.width"),
            ("w_6001", lambda m: m.__setitem__("width", 6001), "meta.width"),
            ("w_huge", lambda m: m.__setitem__("width", 10 ** 400), "meta.width"),
            ("cell_tiny", lambda m: m.__setitem__("cell_m", 1e-320), "meta.width"),
            ("cell_huge_int", lambda m: m.__setitem__("cell_m", 10 ** 400), "meta.cell_m"),
            ("bounds_1e308", lambda m: m.__setitem__("bounds_merc", [-1e308, -1e308, 1e308, 1e308]), "bounds_merc"),
            ("bounds_huge_int", lambda m: m.__setitem__("bounds_merc", [0, 0, 10 ** 400, 1]), "meta.bounds_merc"),
            ("lnglat_huge_int", lambda m: m.__setitem__("bounds_lnglat", [0, 0, 10 ** 400, 1]), "meta.bounds_lnglat"),
            ("depth_in_huge", lambda m: m.__setitem__("storm_depth_in", 10 ** 400), "meta.storm_depth_in"),
        ]
        for name, fn, word in cases:
            self.check_reported_not_raised(name, fn, word)

    def test_absurd_grid_size_is_refused_before_anything_is_allocated(self):
        d = self.corrupt_copy("c_huge_grid")
        self.edit_meta(d, lambda m: (m.__setitem__("width", 10 ** 9), m.__setitem__("height", 10 ** 9)))
        t0 = time.perf_counter()
        problems = fg.validate_storm_dir(d)
        self.assertLess(time.perf_counter() - t0, 2.0)
        self.assertReported(problems, "meta.width", "meta.height", "not checked")

    def test_random_damage_to_meta_and_gzip_never_raises(self):
        """A seeded fuzz: every outcome must be a list of strings, whatever the damage."""
        rng = np.random.default_rng(2024)
        pool = [None, True, False, 0, -1, 1, 7, 6000, 6001, 1.5, -2.5, 1e308, -1e308, 1e-320, 10 ** 9, 10 ** 400, "", "x",
                "2020-01-01T00:00:00Z", [], {}, [1], [[]], [1, 2, 3], {"a": 1}, [None], ["x"], float("nan"), float("inf")]
        base = self.load_meta(self.good)
        d = self.corrupt_copy("fuzz")
        files = sorted(f for f in os.listdir(d) if f.endswith(".gz"))
        originals = {f: Path(d, f).read_bytes() for f in files}

        def paths(o, prefix=()):
            yield prefix
            if isinstance(o, dict):
                for k, x in o.items():
                    yield from paths(x, prefix + (k,))
            elif isinstance(o, list):
                for i, x in enumerate(o):
                    yield from paths(x, prefix + (i,))

        all_paths = [p for p in paths(base) if p]
        for case in range(250):
            meta = json.loads(json.dumps(base))
            for _ in range(int(rng.integers(1, 4))):
                p = all_paths[int(rng.integers(len(all_paths)))]
                try:
                    parent = meta
                    for k in p[:-1]:
                        parent = parent[k]
                    if rng.random() < 0.3:
                        del parent[p[-1]]
                    else:
                        parent[p[-1]] = pool[int(rng.integers(len(pool)))]
                except (KeyError, IndexError, TypeError):
                    pass                                    # an earlier edit already removed this spot
            Path(d, "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            for f in files:
                data = originals[f]
                if rng.random() < 0.15:
                    n = int(rng.integers(0, len(data)))
                    data = data[:n] if rng.random() < 0.5 else data[:n] + bytes([data[n] ^ 0xFF]) + data[n + 1:]
                Path(d, f).write_bytes(data)
            try:
                problems = fg.validate_storm_dir(d)
            except Exception as e:
                self.fail("case %d raised %s: %s\nmeta=%s" % (case, type(e).__name__, e, json.dumps(meta)[:600]))
            self.assertIsInstance(problems, list, msg="case %d" % case)
            self.assertTrue(all(isinstance(x, str) for x in problems), msg="case %d" % case)
            self.assertFalse([x for x in problems if x.startswith("validator error")],
                             "case %d: only the catch-all caught it, a specific check is missing: %s\nmeta=%s"
                             % (case, problems, json.dumps(meta)[:600]))

    def test_row_batching_does_not_change_the_findings(self):
        d1 = self.corrupt_copy("b_speed")
        d2 = self.corrupt_copy("b_stats")
        d3 = self.corrupt_copy("b_fphours")
        self.edit_hour(d1, 3, lambda a: a.__setitem__((18, 20), [60, 255, 128, 255]))
        self.edit_hour(d1, 2, lambda a: a.__setitem__((30, 5), [0, 140, 128, 255]))
        self.edit_hour(d3, 0, lambda a: a.__setitem__((18, 20, 2), 255))

        def fn(m):
            m["hourly"][0]["volume_m3"] *= 2.5
        self.edit_meta(d2, fn)
        old = fg._SCAN_PIXELS
        try:
            for d in (self.good, d1, d2, d3):
                fg._SCAN_PIXELS = old
                ref = fg.validate_storm_dir(d)
                for px in (1, 50, 97, 1000):
                    fg._SCAN_PIXELS = px                       # batches of 1 row up to a few rows
                    self.assertEqual(fg.validate_storm_dir(d), ref, msg="%s with %d pixels per batch" % (d, px))
            self.assertEqual(fg.validate_storm_dir(self.good), [])
            self.assertTrue(fg.validate_storm_dir(d1) and fg.validate_storm_dir(d2) and fg.validate_storm_dir(d3))
        finally:
            fg._SCAN_PIXELS = old

    def test_command_line(self):
        py = sys.executable
        r = subprocess.run([py, str(SCRIPTS / "flowgrid.py"), "validate", self.good], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("OK", r.stdout)
        d = self.corrupt_copy("c_cli")
        os.remove(os.path.join(d, "h05.rgba.gz"))
        r = subprocess.run([py, str(SCRIPTS / "flowgrid.py"), "validate", self.good, d], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("h05.rgba.gz", r.stdout)
        self.assertIn("OK", r.stdout)
        r = subprocess.run([py, str(SCRIPTS / "flowgrid.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_command_line_lists_problems_of_an_incomplete_meta_instead_of_a_traceback(self):
        d = self.corrupt_copy("c_cli_meta")
        self.edit_meta(d, lambda m: m["hourly"][0].pop("volume_m3"))
        r = subprocess.run([sys.executable, str(SCRIPTS / "flowgrid.py"), "validate", d], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("INVALID", r.stdout)
        self.assertIn("volume_m3", r.stdout)


# ------------------------------------------------------------------------------------------------ 14
class TestPerformance(unittest.TestCase):
    N = 4000

    @staticmethod
    def field(n, wet):
        rng = np.random.default_rng(0)
        x = np.arange(n, dtype=np.float32)
        z = (200 + 0.002 * x[None, :] + np.outer(np.sin(x / 37), np.cos(x / 53)).astype(np.float32)
             + 0.02 * rng.standard_normal((n, n), dtype=np.float32)).astype(np.float32)
        if wet == "full":
            h = (0.3 + 0.05 * rng.random((n, n), dtype=np.float32)).astype(np.float32)
        else:                                                           # a basin: about a third of the cells are wet
            h = np.maximum(np.float32(203.0) - z, 0).astype(np.float32)
        return h, z

    def test_4000x4000_time_and_memory(self):
        n = self.N
        h, z = self.field(n, "full")
        inputs_mb = (h.nbytes + z.nbytes) / 2 ** 20

        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        u0, v0 = fg.face_velocity(h, z, 0.035, 4.0)
        t_whole = time.perf_counter() - t0
        peak_whole = tracemalloc.get_traced_memory()[1] / 2 ** 20        # includes the 128 MB of outputs
        tracemalloc.stop()

        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        u1, v1 = fg.face_velocity_strips(h, z, 0.035, 4.0, strip_rows=250)
        t_strips = time.perf_counter() - t0
        peak_strips = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()

        print("\n  PERF %dx%d float32, fully wet (every face open, worst case), measured with tracemalloc (numpy allocations):"
              % (n, n))
        print("  PERF face_velocity        %5.1f s   peak %6.0f MB (+%.0f MB inputs = %.0f MB)"
              % (t_whole, peak_whole, inputs_mb, peak_whole + inputs_mb))
        print("  PERF face_velocity_strips %5.1f s   peak %6.0f MB (+%.0f MB inputs = %.0f MB), strip_rows=250"
              % (t_strips, peak_strips, inputs_mb, peak_strips + inputs_mb))
        self.assertLess(t_whole, 40.0)
        self.assertLess(peak_whole + inputs_mb, 2.5 * 1024)
        self.assertLess(peak_strips, 0.6 * peak_whole)
        np.testing.assert_array_equal(u0, u1)                             # strips == whole array on the big field too
        np.testing.assert_array_equal(v0, v1)
        sp = speed(u0, v0)
        self.assertGreater(float(sp.max()), 0.5)
        self.assertLessEqual(float((sp - np.minimum(np.sqrt(G * h.astype(np.float64)), 4.0)).max()), 2e-6)
        del u0, v0, u1, v1, sp

        h2, z2 = self.field(n, "basin")
        wet = float((h2 > 0.01).mean())
        t0 = time.perf_counter()
        u2, v2 = fg.face_velocity(h2, z2, 0.035, 4.0)
        t_basin = time.perf_counter() - t0
        print("  PERF face_velocity on a partly wet field (%.0f%% wet): %.1f s" % (100 * wet, t_basin))
        self.assertLess(t_basin, 40.0)
        self.assertGreater(wet, 0.1)
        self.assertLess(wet, 0.9)
        # lake at rest on the rough 4000 x 4000 bed: still motionless
        self.assertLess(float(speed(u2, v2).max()), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
