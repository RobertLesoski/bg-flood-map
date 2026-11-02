r"""
flowgrid.py - depth-averaged velocity from water-surface slope, and the "bgflow" pixel format.

Implements FLOW_FORMAT.md (the contract between the Python builders and the browser renderer):

  face_velocity / face_velocity_strips   Manning velocity from the simulated water-surface slope
  block_reduce                           k x k aggregation (depth mean, depth-weighted velocity)
  convergence_angle_rad / rotate_to_true UTM grid axes -> true east/north
  encode_* / decode_* / pack_*           8-bit pixel encoding of depth, velocity, flow-paths summary
  write_rgba_gz / read_rgba_gz           reproducible gzip files
  grid_from_wet_bounds / warp_to_grid    the shared Web Mercator output grid
  read_dem_m                             terrain GeoTIFF in metres (dem_bg.tif is in feet), nodata as NaN
  hour_stats / build_meta / write_meta   meta.json
  validate_storm_dir                     checks a finished layers/flow/<storm> folder

CLI:   python flowgrid.py validate <storm_dir> [<storm_dir> ...]     exit status 1 on any problem

numpy only; rasterio is imported lazily inside the warp / grid helpers. No scipy, no pyproj, so it also runs on Vista.

Units: "ground m" are real metres on the ground, "merc m" are Web Mercator metres (stretched by 1/cos(lat), about
1.25 at Bowling Green). Velocities are m/s, depths m, unit discharge m^2/s.
"""
from __future__ import annotations

import datetime
import gzip
import json
import math
import os
import re
import sys
import time
import uuid
import zlib

import numpy as np

# ---- format constants (FLOW_FORMAT.md) ----
DEPTH_MAX = 8.0          # m, depth code 255 (deeper water clips)
VMAX = 4.0               # m/s, |u|,|v| code limit and the face-speed cap
WET_THRESHOLD = 0.02     # m, shallower water is written as dry
QMAX = 2.0               # m^2/s, unit-discharge code 255
ZERO_CODE = 128          # velocity code for 0 m/s
HOURS_TOTAL = 24
FORMAT_NAME = "bgflow"
FORMAT_VERSION = 1
MAX_GRID_DIM = 6000

GRAV = 9.81              # m/s^2
G = GRAV                 # alias required by the task API (constants DEPTH_MAX, VMAX, WET_THRESHOLD, QMAX, G)
N_DEFAULT = 0.035        # Manning n used where a Manning raster is nodata / non-positive
NODATA = -9999.0         # raster nodata of the DEM and the depth rasters

MERC_R = 6378137.0       # m, EPSG:3857 sphere radius
UTM16_LON0_DEG = -87.0   # central meridian of UTM zone 16N
VELOCITY_METHOD = "face-Manning from simulated water-surface slope (not solver output)"
SOURCE_NOTES = {
    "rasters": "depth rasters from solver v2 (2 m), velocity derived from water-surface slope",
    "tiers": "preview from depth-tier polygons (depth quantised to tier midpoints, then smoothed), "
             "velocity derived from water-surface slope",
}

_VEL_STEP = VMAX / 127.0           # m/s per velocity code
_CHUNK = 1 << 21                   # faces / cells processed per numpy batch inside face_velocity


# =====================================================================================================
# velocity from water-surface slope
# =====================================================================================================

def _prep_n(n, shape):
    """Manning n as a python float (scalar input) or a flat float32 array. Non-finite or non-positive -> N_DEFAULT."""
    if np.ndim(n) == 0:
        nv = float(n)
        if not nv > 0.0:
            raise ValueError("Manning n must be positive")
        return nv
    na = np.asarray(n, dtype=np.float32)
    if na.shape != tuple(shape):
        raise ValueError("n must be a scalar or have the same shape as h")
    bad = ~(np.isfinite(na) & (na > 0))
    if bad.any():
        na = np.where(bad, np.float32(N_DEFAULT), na)
    return np.ascontiguousarray(na).reshape(-1)


def _axis_pass(hf, zf, nf, wet, valid, shape, axis, dx, h_face_min, slope_min, vmax, g):
    """
    One axis of the face algorithm. axis=1: faces between (i,j) and (i,j+1) (toward +column = east);
    axis=0: faces between (i,j) and (i+1,j) (toward +row = south). hf, zf are flat views of the (H,W) arrays.
    Returns flat float32 arrays (vel, slope), both along +axis, per cell:
      vel   = sum(h_f * v_f) / sum(h_f)      over the cell's open faces on that axis (m/s)
      slope = sum(h_f * S_f) / sum(h_f)      signed surface drop per metre along +axis (S_f > 0: water moves +axis)
    """
    H, W = shape
    if axis == 1:
        if W < 2:
            return np.zeros(H * W, np.float32), np.zeros(H * W, np.float32)
        # h_f <= max(h_a, h_b), so a face can only be open if one side is wet: the candidate list is a superset
        a_all = np.flatnonzero((wet[:, :-1] | wet[:, 1:]) & (valid[:, :-1] & valid[:, 1:]))
        row_of = a_all // (W - 1)           # flat index in the (H, W-1) face layout -> index of cell a in (H, W)
        a_all += row_of
        del row_of
        step = 1
    else:
        if H < 2:
            return np.zeros(H * W, np.float32), np.zeros(H * W, np.float32)
        a_all = np.flatnonzero((wet[:-1] | wet[1:]) & (valid[:-1] & valid[1:]))
        step = W
    if a_all.size == 0:
        return np.zeros(H * W, np.float32), np.zeros(H * W, np.float32)

    wsum = np.zeros(H * W, np.float32)
    nsum = np.zeros(H * W, np.float32)
    ssum = np.zeros(H * W, np.float32)
    f8 = np.float64
    for s in range(0, a_all.size, _CHUNK):
        a = a_all[s:s + _CHUNK]
        b = a + step
        za = zf[a].astype(f8)
        zb = zf[b].astype(f8)
        ea = za + np.maximum(hf[a], 0)           # water surface, float64 so a float32 lake at rest has no noise
        eb = zb + np.maximum(hf[b], 0)
        hface = np.maximum(ea, eb) - np.maximum(za, zb)       # m, depth on the face
        drop = (ea - eb) / dx                                 # > 0: water moves from a to b (+axis)
        del za, zb, ea, eb
        opn = hface > h_face_min
        w = np.where(opn, hface, 0.0)
        ad = np.abs(drop)
        act = opn & (ad >= slope_min)
        hpos = np.maximum(hface, 0.0)
        if isinstance(nf, float):
            nfa = nf
        else:
            nfa = 0.5 * (nf[a].astype(f8) + nf[b].astype(f8))
        sp = np.cbrt(hpos) ** 2 * np.sqrt(ad) / nfa           # h^(2/3) sqrt(|S|) / n
        sp = np.minimum(sp, np.minimum(vmax, np.sqrt(g * hpos)))      # Froude <= 1 and vcap
        vf = np.where(act, np.copysign(sp, drop), 0.0)
        sd = np.where(act, drop, 0.0)
        wv = w * vf
        ws = w * sd
        # a and b are each unique inside one pass (one +face and one -face per cell), so plain fancy += is safe.
        # float32 addition is not associative: a cell's two contributions must always be added in the same order
        # (minus face first). Faces are visited in increasing index and the minus face of a cell has the lower
        # index, so "b before a" inside every batch is the same order for any batch size, and for strips.
        wsum[b] += w
        wsum[a] += w
        nsum[b] += wv
        nsum[a] += wv
        ssum[b] += ws
        ssum[a] += ws
    pos = wsum > 0
    np.divide(nsum, wsum, out=nsum, where=pos)      # nsum, ssum are exactly 0 wherever wsum is 0
    np.divide(ssum, wsum, out=ssum, where=pos)
    return nsum, ssum


def face_velocity(h, z, n, dx, *, h_face_min=0.005, h_cell_min=0.01, slope_min=1e-5, vmax=4.0, g=9.81,
                  nodata=NODATA):
    """
    Depth-averaged velocity from the water-surface slope (FLOW_FORMAT.md "Velocity method").

    h        (H,W) water depth, m                 z   (H,W) ground elevation, m
    n        Manning n, scalar or (H,W)           dx  cell spacing in GROUND metres
    nodata   value treated as "no data" in h and z (NaN and +-inf always are); None to disable
    Returns (u_grid_east, v_grid_north), float32 (H,W), m/s, in UTM GRID axes: +column = east, row 0 = north, so
    grid-north velocity is minus the row direction. Rotate to true east/north with rotate_to_true.

    Per cell face: h_f = max(eta_a, eta_b) - max(z_a, z_b) (eta = z + h); open if h_f > h_face_min; speed
    h_f^(2/3) sqrt(|S|) / n_f toward the lower surface, 0 if |S| < slope_min, limited to min(vmax, sqrt(g h_f)).
    Cell velocity per axis = depth-weighted mean over its open faces. Two refinements beyond the per-axis
    formula (FLOW_FORMAT.md explains why): the per-axis sqrt(|S|) is rescaled by sqrt(|Sx|/|S|), sqrt(|Sy|/|S|)
    so a diagonal flow has the Manning speed and points along the surface gradient; and the cell vector is
    finally limited to min(vmax, sqrt(g h)) with h the cell's own depth.
    Invalid cells (NaN / nodata) are dry walls: no flow into or out of them, zero velocity. Cells with
    h < h_cell_min get zero velocity. Face arithmetic is float64 internally; the output is float32.
    Each output cell depends only on its 3x3 neighbourhood, which is what face_velocity_strips relies on.
    """
    h = np.asarray(h)
    z = np.asarray(z)
    if h.ndim != 2 or h.shape != z.shape:
        raise ValueError("h and z must be 2-D arrays of the same shape")
    if not dx > 0:
        raise ValueError("dx must be positive")
    if h.dtype.kind != "f":
        h = h.astype(np.float32)
    if z.dtype.kind != "f":
        z = z.astype(np.float32)
    shape = h.shape
    H, W = shape
    nf = _prep_n(n, shape)

    valid = np.isfinite(h) & np.isfinite(z)
    if nodata is not None:
        valid &= (h != nodata) & (z != nodata)
    ue = np.zeros(shape, np.float32)
    vn = np.zeros(shape, np.float32)
    hf = np.ascontiguousarray(h).reshape(-1)
    zf = np.ascontiguousarray(z).reshape(-1)
    vflat = valid.reshape(-1)
    mask = vflat & (hf >= h_cell_min)
    if not mask.any():
        return ue, vn

    wet = valid & (h > h_face_min)
    ux, sx = _axis_pass(hf, zf, nf, wet, valid, shape, 1, dx, h_face_min, slope_min, vmax, g)
    vy, sy = _axis_pass(hf, zf, nf, wet, valid, shape, 0, dx, h_face_min, slope_min, vmax, g)
    del wet

    f8 = np.float64
    uef = ue.reshape(-1)
    vnf = vn.reshape(-1)
    cells_all = np.flatnonzero(mask)
    for s0 in range(0, cells_all.size, _CHUNK):
        cells = cells_all[s0:s0 + _CHUNK]
        u = ux[cells].astype(f8)
        v = vy[cells].astype(f8)                    # along +row = toward the south
        sxc = sx[cells].astype(f8)
        syc = sy[cells].astype(f8)
        smag = np.hypot(sxc, syc)
        ok = smag > 0
        den = np.where(ok, smag, 1.0)
        u *= np.where(ok, np.sqrt(np.abs(sxc) / den), 1.0)
        v *= np.where(ok, np.sqrt(np.abs(syc) / den), 1.0)
        hh = np.maximum(hf[cells].astype(f8), 0.0)
        lim = np.minimum(vmax, np.sqrt(g * hh))
        sp = np.hypot(u, v)
        scale = np.where(sp > lim, lim / np.where(sp > 0, sp, 1.0), 1.0)
        uef[cells] = u * scale
        vnf[cells] = 0.0 - v * scale                # grid north = minus the row direction
    return ue, vn


def face_velocity_strips(h, z, n, dx, *, strip_rows=1024, h_face_min=0.005, h_cell_min=0.01, slope_min=1e-5,
                         vmax=4.0, g=9.81, nodata=NODATA):
    """
    Same result as face_velocity (bit for bit) but processes strip_rows output rows at a time with a one-cell
    halo, so the working memory is bounded by the strip. Strips without any wet cell are skipped.
    h, z may be any 2-D array that supports row slicing (ndarray, np.memmap); n scalar or (H,W).
    """
    if strip_rows < 1:
        raise ValueError("strip_rows must be >= 1")
    if np.ndim(h) != 2 or np.shape(h) != np.shape(z):
        raise ValueError("h and z must be 2-D arrays of the same shape")
    H, W = np.shape(h)
    if np.ndim(n) != 0 and np.shape(n) != (H, W):
        raise ValueError("n must be a scalar or have the same shape as h")
    ue = np.zeros((H, W), np.float32)
    vn = np.zeros((H, W), np.float32)
    for r0 in range(0, H, strip_rows):
        r1 = min(r0 + strip_rows, H)
        a = max(r0 - 1, 0)
        b = min(r1 + 1, H)
        hs = np.asarray(h[a:b])
        if not (hs[r0 - a:r1 - a] >= h_cell_min).any():
            continue
        zs = np.asarray(z[a:b])
        ns = n if np.ndim(n) == 0 else np.asarray(n[a:b])
        us, vs = face_velocity(hs, zs, ns, dx, h_face_min=h_face_min, h_cell_min=h_cell_min, slope_min=slope_min,
                               vmax=vmax, g=g, nodata=nodata)
        ue[r0:r1] = us[r0 - a:r1 - a]
        vn[r0:r1] = vs[r0 - a:r1 - a]
    return ue, vn


def block_reduce(depth, u, v, k, valid=None, *, froude_g=GRAV):
    """
    Aggregate k x k blocks (remainder rows/columns at the south/east edge are trimmed).
    depth_k = mean depth (so sum(depth*area) is conserved); u_k, v_k = depth-weighted mean velocity
    (0 where the block is dry), so unit discharge is conserved as well; valid_k = any valid cell in the block.
    Because the block depth also averages over dry sub-cells, the weighted-mean speed can exceed
    sqrt(g * depth_k) in sparsely wet blocks; with froude_g (default 9.81) the speed is limited to that bound so
    the output meets the format's sanity bound. froude_g=None gives the plain weighted mean.
    valid=None means "finite depth". Non-finite values count as 0. Returns float32 arrays and a bool array.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be 2-D")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    H, W = depth.shape
    hk, wk = H // k, W // k
    if hk == 0 or wk == 0:
        raise ValueError("array smaller than one block")
    wt = wk * k
    depth_k = np.zeros((hk, wk), np.float32)
    u_k = np.zeros((hk, wk), np.float32)
    v_k = np.zeros((hk, wk), np.float32)
    valid_k = np.zeros((hk, wk), bool)
    step = max(1, (1 << 21) // max(wt * k, 1))          # block rows per batch (about 2 M cells)
    for r0 in range(0, hk, step):
        r1 = min(hk, r0 + step)
        sl = (slice(r0 * k, r1 * k), slice(0, wt))
        nr = r1 - r0
        d_raw = np.asarray(depth[sl], dtype=np.float64)
        fin = np.isfinite(d_raw)
        d = np.where(fin, np.maximum(d_raw, 0.0), 0.0)
        uu = np.nan_to_num(np.asarray(u[sl], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        vv = np.nan_to_num(np.asarray(v[sl], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        ok = fin if valid is None else (np.asarray(valid[sl], dtype=bool))
        sd = d.reshape(nr, k, wk, k).sum(axis=(1, 3))
        su = (d * uu).reshape(nr, k, wk, k).sum(axis=(1, 3))
        sv = (d * vv).reshape(nr, k, wk, k).sum(axis=(1, 3))
        wet = sd > 0
        den = np.where(wet, sd, 1.0)
        ub = np.where(wet, su / den, 0.0)
        vb = np.where(wet, sv / den, 0.0)
        db = sd / float(k * k)
        if froude_g is not None:
            lim = np.sqrt(froude_g * db)
            sp = np.hypot(ub, vb)
            sc = np.where(sp > lim, lim / np.where(sp > 0, sp, 1.0), 1.0)
            ub = ub * sc
            vb = vb * sc
        depth_k[r0:r1] = db
        u_k[r0:r1] = ub
        v_k[r0:r1] = vb
        valid_k[r0:r1] = ok.reshape(nr, k, wk, k).any(axis=(1, 3))
    return depth_k, u_k, v_k, valid_k


# =====================================================================================================
# geometry: convergence, projections, grids
# =====================================================================================================

def convergence_angle_rad(lon_deg, lat_deg):
    """
    UTM grid convergence gamma = atan(tan(lon - lon0) sin(lat)), lon0 = -87 deg (zone 16N), radians.
    gamma > 0 east of the central meridian: grid north is rotated clockwise (toward east) from true north.
    """
    dl = np.radians(np.asarray(lon_deg, dtype=np.float64) - UTM16_LON0_DEG)
    return np.arctan(np.tan(dl) * np.sin(np.radians(np.asarray(lat_deg, dtype=np.float64))))


def rotate_to_true(u_grid, v_grid, gamma_rad):
    """
    UTM grid components (east, north of the grid) -> true east/north components.
    The grid is rotated clockwise by gamma from true north, so a grid-north vector has bearing gamma:
        u_east = u cos(g) + v sin(g),   v_north = -u sin(g) + v cos(g).
    gamma_rad is a scalar or an array broadcastable to u_grid. Output keeps the dtype of u_grid.
    """
    u_grid = np.asarray(u_grid)
    v_grid = np.asarray(v_grid)
    gam = np.asarray(gamma_rad, dtype=np.float64)
    c = np.cos(gam)
    s = np.sin(gam)
    dt = u_grid.dtype if u_grid.dtype.kind == "f" else np.float32
    ue = (u_grid * c + v_grid * s).astype(dt, copy=False)
    vn = (v_grid * c - u_grid * s).astype(dt, copy=False)
    return ue, vn


def merc_to_lnglat(x, y):
    """EPSG:3857 metres -> (lon_deg, lat_deg), exact sphere formulas. Arrays or scalars."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = np.degrees(x / MERC_R)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / MERC_R)) - np.pi / 2.0)
    return lon, lat


def lnglat_to_merc(lon, lat):
    """(lon_deg, lat_deg) -> EPSG:3857 metres (x, y). Arrays or scalars."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x = MERC_R * np.radians(lon)
    y = MERC_R * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def utm16n_to_lnglat(e, n):
    """EPSG:32616 (easting, northing in metres) -> (lon_deg, lat_deg) with rasterio/GDAL. Arrays or scalars."""
    from rasterio.warp import transform
    e = np.asarray(e, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    e, n = np.broadcast_arrays(e, n)
    lon, lat = transform("EPSG:32616", "EPSG:4326", e.reshape(-1).tolist(), n.reshape(-1).tolist())
    return np.asarray(lon).reshape(e.shape), np.asarray(lat).reshape(e.shape)


def convergence_field_utm16n(transform, height, width, step=64):
    """
    Grid convergence (radians, float32 (height,width)) for every pixel centre of a UTM-16N raster, from a coarse
    lattice every `step` pixels interpolated bilinearly (gamma is smooth: the error is far below 1e-6 rad).
    """
    ri = np.unique(np.append(np.arange(0, height, step), height - 1))
    ci = np.unique(np.append(np.arange(0, width, step), width - 1))
    cols, rows = np.meshgrid(ci + 0.5, ri + 0.5)
    e = transform.c + transform.a * cols + transform.b * rows
    n = transform.f + transform.d * cols + transform.e * rows
    lon, lat = utm16n_to_lnglat(e, n)
    coarse = convergence_angle_rad(lon, lat)

    def axis(size, idx):
        if len(idx) == 1:
            return np.zeros(size, np.intp), np.zeros(size)
        pos = np.interp(np.arange(size), idx, np.arange(len(idx)))
        i0 = np.minimum(np.floor(pos).astype(np.intp), len(idx) - 2)
        return i0, pos - i0

    r0, tr = axis(height, ri)
    c0, tc = axis(width, ci)
    r1 = np.minimum(r0 + 1, len(ri) - 1)
    c1 = np.minimum(c0 + 1, len(ci) - 1)
    out = np.empty((height, width), np.float32)
    for s in range(0, height, 256):                  # row batches keep the float64 temporaries small
        e = min(s + 256, height)
        rows_interp = coarse[r0[s:e]] * (1 - tr[s:e])[:, None] + coarse[r1[s:e]] * tr[s:e][:, None]   # (rows, ncoarse)
        out[s:e] = rows_interp[:, c0] * (1 - tc)[None, :] + rows_interp[:, c1] * tc[None, :]
    return out


def grid_from_wet_bounds(bounds_merc, cell_m, margin_m=150.0, max_dim=MAX_GRID_DIM):
    """
    The shared output grid: wet-area bounds (xmin, ymin, xmax, ymax in merc m) grown by margin_m and snapped
    outward to multiples of cell_m (merc m). Returns dict(bounds_merc, width, height, cell_m, transform) with a
    rasterio Affine (north-up, row 0 = ymax). Raises ValueError if width or height would exceed max_dim.
    """
    from rasterio.transform import Affine
    xmin, ymin, xmax, ymax = (float(b) for b in bounds_merc)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("bounds_merc must be (xmin, ymin, xmax, ymax) with positive extent")
    cell = float(cell_m)
    if not cell > 0:
        raise ValueError("cell_m must be positive")
    eps = 1e-9
    x0 = math.floor((xmin - margin_m) / cell + eps) * cell
    y0 = math.floor((ymin - margin_m) / cell + eps) * cell
    x1 = math.ceil((xmax + margin_m) / cell - eps) * cell
    y1 = math.ceil((ymax + margin_m) / cell - eps) * cell
    width = int(round((x1 - x0) / cell))
    height = int(round((y1 - y0) / cell))
    if width > max_dim or height > max_dim:
        need = max(x1 - x0, y1 - y0) / max_dim
        raise ValueError("grid %dx%d exceeds max_dim=%d at cell_m=%g: increase cell_m to at least %.2f"
                         % (width, height, max_dim, cell, need))
    return {"bounds_merc": [x0, y0, x1, y1], "width": width, "height": height, "cell_m": cell,
            "transform": Affine(cell, 0.0, x0, 0.0, -cell, y1)}


def warp_to_grid(arr, src_transform, src_crs, grid, resampling="bilinear", src_nodata=None, dst_fill=0.0):
    """
    Reproject a 2-D array (any CRS, e.g. UTM 16N) onto the EPSG:3857 grid from grid_from_wet_bounds.
    resampling is a rasterio.enums.Resampling name ('bilinear', 'nearest', 'average', ...). Source pixels equal
    to src_nodata are ignored; destination pixels with no source data get dst_fill. Returns float32 (height,width).
    Downsampling by more than about 2x with 'bilinear' aliases: use 'average' (or block_reduce first).
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    src = np.asarray(arr, dtype=np.float32)
    dst = np.full((grid["height"], grid["width"]), dst_fill, np.float32)
    # dst is pre-filled and GDAL only writes pixels it can compute, so a destination nodata is only needed to keep
    # invalid source pixels out of the result; declaring one always would make GDAL replace a computed 0.0 by 1.4e-45
    reproject(source=src, destination=dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=grid["transform"], dst_crs="EPSG:3857", resampling=Resampling[resampling],
              src_nodata=src_nodata, dst_nodata=None if src_nodata is None else dst_fill, init_dest_nodata=False)
    if src_nodata is not None:
        dst[(dst != 0) & (np.abs(dst) < 1e-30)] = 0.0         # with dst_fill = 0 that nudge shows as a denormal: undo it
    return dst


FEET_TO_M = 0.3048
# The local dem_bg.tif stores elevations in FEET (its cells are metres); the Vista solver DEM is in metres. Neither file
# says so in its metadata, so the units are looked up by file name and anything else must be told explicitly.
DEM_UNITS_BY_NAME = {"dem_bg.tif": "feet", "dem_bg_2m.tif": "metres"}


def read_dem_m(path, units=None):
    """
    Terrain GeoTIFF as ELEVATION IN METRES, the unit face_velocity needs (eta = z + h with h in metres).
    units 'feet' or 'metres'; None picks it from the file name (DEM_UNITS_BY_NAME) and raises ValueError for any other
    file instead of guessing: reading dem_bg.tif as metres makes slopes 3.28x too steep and speeds about 1.8x too high.
    Returns (z, transform, crs): z float32 (H, W) with NaN where the file has no data (its nodata value, or -9999, or
    non-finite), which face_velocity treats as a wall; rasterio Affine; rasterio CRS.
    """
    import rasterio
    path = str(path)
    if units is None:
        units = DEM_UNITS_BY_NAME.get(os.path.basename(path).lower())
        if units is None:
            raise ValueError("elevation units of %s are unknown: pass units='feet' or units='metres'" % path)
    if units not in ("feet", "metres"):
        raise ValueError("units must be 'feet' or 'metres', got %r" % (units,))
    with rasterio.open(path) as ds:
        raw = ds.read(1)
        nodata, transform, crs = ds.nodata, ds.transform, ds.crs
    invalid = ~np.isfinite(raw) | (raw == NODATA)
    if nodata is not None:
        invalid |= raw == nodata
    if units == "feet":
        z = (raw.astype(np.float64) * FEET_TO_M).astype(np.float32)
    else:
        z = raw.astype(np.float32)
    z[invalid] = np.nan
    return z, transform, crs


# =====================================================================================================
# pixel encoding
# =====================================================================================================

def encode_depth(d):
    """depth m -> uint8, R = round(255 sqrt(clamp(d/8, 0, 1))). No wet threshold here (pack_rgba applies it)."""
    d = np.nan_to_num(np.asarray(d, dtype=np.float64), nan=0.0, posinf=DEPTH_MAX, neginf=0.0)
    return np.rint(255.0 * np.sqrt(np.clip(d / DEPTH_MAX, 0.0, 1.0))).astype(np.uint8)


def decode_depth(code):
    """uint8 depth code -> float32 metres, (R/255)^2 * 8."""
    c = np.asarray(code).astype(np.float32)
    return ((c / np.float32(255.0)) ** 2 * np.float32(DEPTH_MAX)).astype(np.float32)


def encode_vel(x):
    """velocity m/s (one component) -> uint8, 128 + round(clamp(x/4, -1, 1) * 127)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=VMAX, neginf=-VMAX)
    return (ZERO_CODE + np.rint(np.clip(x / VMAX, -1.0, 1.0) * 127.0)).astype(np.uint8)


def decode_vel(code):
    """uint8 velocity code -> float32 m/s, (code - 128) / 127 * 4 (code 0 is clamped to -4)."""
    c = np.asarray(code).astype(np.float32)
    return np.clip((c - np.float32(ZERO_CODE)) / np.float32(127.0) * np.float32(VMAX), -VMAX, VMAX).astype(np.float32)


def encode_q(q):
    """unit discharge m^2/s -> uint8, round(255 sqrt(clamp(q/QMAX, 0, 1)))."""
    q = np.nan_to_num(np.asarray(q, dtype=np.float64), nan=0.0, posinf=QMAX, neginf=0.0)
    return np.rint(255.0 * np.sqrt(np.clip(q / QMAX, 0.0, 1.0))).astype(np.uint8)


def decode_q(code):
    """uint8 unit-discharge code -> float32 m^2/s, (G/255)^2 * 2."""
    c = np.asarray(code).astype(np.float32)
    return ((c / np.float32(255.0)) ** 2 * np.float32(QMAX)).astype(np.float32)


def _valid_mask(valid, shape):
    if valid is None:
        return np.ones(shape, bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != tuple(shape):
        raise ValueError("valid must have the same shape as depth")
    return valid


def pack_rgba(depth, u_east, v_north, valid):
    """
    One hour as uint8 (H,W,4): R depth code, G u code, B v code, A 255 inside the domain / 0 outside.
    depth m, u_east / v_north m/s (true east / true north); valid bool or None (all valid).
    Depth below WET_THRESHOLD, and everything outside the domain, is written dry: R = 0, G = B = 128.
    """
    depth = np.asarray(depth)
    ok = _valid_mask(valid, depth.shape)
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0)
    wet = ok & (d >= WET_THRESHOLD)
    out = np.empty(depth.shape + (4,), np.uint8)
    out[..., 0] = np.where(wet, encode_depth(d), 0)
    out[..., 1] = np.where(wet, encode_vel(u_east), ZERO_CODE)
    out[..., 2] = np.where(wet, encode_vel(v_north), ZERO_CODE)
    out[..., 3] = np.where(ok, 255, 0)
    return out


def pack_flowpaths(max_depth, mean_q, hours_wet, valid, n_hours_total=HOURS_TOTAL):
    """
    flowpaths.rgba.gz content, uint8 (H,W,4): R max depth over all hours (depth encoding), G mean unit
    discharge (m^2/s, sqrt encoding, QMAX), B hours wet (count) as round(255 count / n_hours_total), A valid.
    Pixels never wet at the threshold (max_depth < WET_THRESHOLD) or outside the domain have R = G = B = 0.
    """
    max_depth = np.asarray(max_depth)
    ok = _valid_mask(valid, max_depth.shape)
    d = np.nan_to_num(np.asarray(max_depth, dtype=np.float32), nan=0.0)
    wet = ok & (d >= WET_THRESHOLD)
    cnt = np.nan_to_num(np.asarray(hours_wet, dtype=np.float64), nan=0.0)
    out = np.empty(max_depth.shape + (4,), np.uint8)
    out[..., 0] = np.where(wet, encode_depth(d), 0)
    out[..., 1] = np.where(wet, encode_q(mean_q), 0)
    out[..., 2] = np.where(wet, np.rint(255.0 * np.clip(cnt / float(n_hours_total), 0.0, 1.0)), 0).astype(np.uint8)
    out[..., 3] = np.where(ok, 255, 0)
    return out


def hour_filename(hour):
    """hNN.rgba.gz for hour 1..24."""
    return "h%02d.rgba.gz" % int(hour)


_REPLACE_TRIES = 10          # os.replace attempts before PermissionError is let through
_REPLACE_DELAY_S = 0.05      # first pause between attempts, doubled each time ...
_REPLACE_DELAY_MAX_S = 0.5   # ... up to this (all ten attempts together wait about 3 s)


def _replace_retry(src, dst):
    """
    os.replace that waits out a PermissionError. On Windows the replace fails while dst is open in a reader that
    did not share DELETE access (serve.py streams files with open(path, 'rb')) or while another writer is
    replacing it, and both go away within moments. Any other error is raised at once.
    """
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(min(_REPLACE_DELAY_MAX_S, _REPLACE_DELAY_S * 2 ** attempt))


def _atomic_write(path, write, mode="wb", **open_kw):
    """
    Call write(file) on a new file and put it at path with one os.replace, so a reader sees the old file or the
    whole new one. The temp file sits in the same folder (the replace must not cross volumes) under a name no other
    call can pick, because two writers sharing one temp name corrupt each other's output, and it is removed on
    any failure, KeyboardInterrupt included. A stray *.tmp is therefore left only by a killed process, which
    validate_storm_dir reports.
    """
    path = str(path)
    tmp = "%s.%d.%s.tmp" % (path, os.getpid(), uuid.uuid4().hex[:12])
    try:
        with open(tmp, mode, **open_kw) as f:
            write(f)
        _replace_retry(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_rgba_gz(path, rgba, level=6):
    """Write uint8 (H,W,4) as a standard gzip file with mtime=0 and no file name, so the bytes are reproducible."""
    rgba = np.ascontiguousarray(rgba)
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must be uint8 with shape (H, W, 4)")

    def write(raw):
        with gzip.GzipFile(filename="", mode="wb", compresslevel=level, fileobj=raw, mtime=0) as gz:
            for r in range(0, rgba.shape[0], 256):
                gz.write(rgba[r:r + 256].tobytes())

    _atomic_write(path, write)


def read_rgba_gz(path, width, height):
    """Read a file written by write_rgba_gz as uint8 (height,width,4). ValueError unless it is exactly that size."""
    n = int(width) * int(height) * 4
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    with gzip.open(str(path), "rb") as f:
        while got < n:
            m = f.readinto(view[got:])
            if not m:
                break
            got += m
        extra = len(f.read(1)) if got == n else 0
    if got != n or extra:
        raise ValueError("decompressed size %s%d bytes, expected width*height*4 = %d"
                         % ("" if got == n else "only ", got + extra, n))
    return np.frombuffer(buf, np.uint8).reshape(int(height), int(width), 4)


# =====================================================================================================
# statistics and meta.json
# =====================================================================================================

def hour_stats(depth_m, u, v, cell_ground_m, wet_threshold=WET_THRESHOLD):
    """
    Per-hour statistics over cells with depth >= wet_threshold (NaN counts as dry). depth_m (m); u, v (m/s) may be
    None; cell_ground_m = side of one cell in GROUND metres.
    wet_cells, wet_area_km2, max_depth_m, mean_depth_m (over wet cells), max_speed_ms, volume_m3 (sum of unclipped
    depth * cell area over wet cells), clipped_cells (depth > DEPTH_MAX).
    """
    d = np.asarray(depth_m, dtype=np.float32)
    wet = np.isfinite(d) & (d >= wet_threshold)
    nwet = int(wet.sum())
    area = float(cell_ground_m) ** 2
    out = {"wet_cells": nwet, "wet_area_km2": 0.0, "max_depth_m": 0.0, "mean_depth_m": 0.0,
           "max_speed_ms": 0.0, "volume_m3": 0.0, "clipped_cells": 0}
    if nwet == 0:
        return out
    dw = d[wet].astype(np.float64)
    out["wet_area_km2"] = round(nwet * area / 1e6, 6)
    out["max_depth_m"] = round(float(dw.max()), 4)
    out["mean_depth_m"] = round(float(dw.mean()), 4)
    out["volume_m3"] = round(float(dw.sum()) * area, 2)
    out["clipped_cells"] = int((dw > DEPTH_MAX).sum())
    if u is not None and v is not None:
        sp = np.hypot(np.asarray(u, dtype=np.float32)[wet].astype(np.float64),
                      np.asarray(v, dtype=np.float32)[wet].astype(np.float64))
        out["max_speed_ms"] = round(float(np.nanmax(sp)), 4)
    return out


def hourly_entry(hour, depth_m, u, v, cell_ground_m):
    """One meta.hourly element: hour, sim_seconds (= hour * 3600) and hour_stats."""
    e = {"hour": int(hour), "sim_seconds": int(hour) * 3600}
    e.update(hour_stats(depth_m, u, v, cell_ground_m))
    return e


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def build_meta(storm, label, source, grid, hours, hourly, storm_depth_in, run_summary=None, source_note=None):
    """
    meta.json content as a dict (FLOW_FORMAT.md). grid is the dict from grid_from_wet_bounds; hours the list of hour
    numbers written; hourly the list of hourly_entry dicts (sim_seconds filled in if missing); run_summary the
    solver's run_summary.json (or None). cell_ground_m = cell_m * cos(latitude of the grid centre).
    """
    if source not in SOURCE_NOTES:
        raise ValueError("source must be 'rasters' or 'tiers'")
    x0, y0, x1, y1 = (float(b) for b in grid["bounds_merc"])
    west, south = merc_to_lnglat(x0, y0)
    east, north = merc_to_lnglat(x1, y1)
    lat_c = float(merc_to_lnglat(0.5 * (x0 + x1), 0.5 * (y0 + y1))[1])
    cell = float(grid["cell_m"])
    hr = []
    for e in hourly:
        e = dict(e)
        e.setdefault("sim_seconds", int(e["hour"]) * 3600)
        hr.append(e)
    meta = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "storm": storm,
        "label": label,
        "source": source,
        "source_note": source_note if source_note is not None else SOURCE_NOTES[source],
        "velocity_method": VELOCITY_METHOD,
        "crs": "EPSG:3857",
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "cell_m": cell,
        "cell_ground_m": round(cell * math.cos(math.radians(lat_c)), 4),
        "bounds_merc": [x0, y0, x1, y1],
        "bounds_lnglat": [round(float(west), 7), round(float(south), 7), round(float(east), 7), round(float(north), 7)],
        "depth_encoding": {"type": "sqrt", "max_m": DEPTH_MAX},
        "vel_encoding": {"vmax_ms": VMAX, "zero_code": ZERO_CODE},
        "wet_threshold_m": WET_THRESHOLD,
        "qmax_m2s": QMAX,
        "hours": [int(h) for h in hours],
        "storm_depth_in": storm_depth_in,
        "run_summary": run_summary,
        "hourly": hr,
        "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return _jsonable(meta)


def write_meta(path, meta):
    """Write meta.json (UTF-8, LF, indented, key order kept)."""
    def write(f):
        json.dump(_jsonable(meta), f, indent=2)
        f.write("\n")

    _atomic_write(path, write, "w", encoding="utf-8", newline="\n")


# =====================================================================================================
# validation of a finished storm folder
# =====================================================================================================

_HOURLY_KEYS = ("hour", "sim_seconds", "wet_cells", "wet_area_km2", "max_depth_m", "mean_depth_m", "max_speed_ms",
                "volume_m3", "clipped_cells")


def _is_num(x):
    """A finite JSON number that float() can hold (a 400-digit integer is not)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _hourly_ok(e):
    """A meta.hourly entry whose every field is present and a finite number >= 0 (_check_meta reports the others)."""
    return isinstance(e, dict) and all(k in e and _is_num(e[k]) and e[k] >= 0 for k in _HOURLY_KEYS)


def _cells_across(lo, hi, cell):
    """round((hi - lo) / cell), or None when that is not a finite number (absurd bounds or cell size)."""
    try:
        q = (hi - lo) / cell
        return int(round(q)) if math.isfinite(q) else None
    except (OverflowError, ZeroDivisionError):
        return None


def _check_meta(meta, bad):
    """Field presence, types and internal consistency of meta.json. Appends problem strings via bad()."""
    def expect(key, cond, what):
        if key not in meta:
            bad("meta.%s: missing" % key)
            return False
        if not cond(meta[key]):
            bad("meta.%s: %s (got %r)" % (key, what, meta[key]))
            return False
        return True

    expect("format", lambda x: x == FORMAT_NAME, "must be %r" % FORMAT_NAME)
    expect("version", lambda x: _is_int(x) and x == FORMAT_VERSION, "must be %d" % FORMAT_VERSION)
    expect("storm", lambda x: isinstance(x, str) and x != "", "must be a non-empty string")
    expect("label", lambda x: isinstance(x, str), "must be a string")
    expect("source", lambda x: isinstance(x, str) and x in SOURCE_NOTES, "must be 'rasters' or 'tiers'")
    expect("source_note", lambda x: isinstance(x, str), "must be a string")
    expect("velocity_method", lambda x: isinstance(x, str) and x != "", "must be a non-empty string")
    expect("crs", lambda x: x == "EPSG:3857", "must be 'EPSG:3857'")
    ok_w = expect("width", lambda x: _is_int(x) and 1 <= x <= MAX_GRID_DIM, "must be an integer in 1..%d" % MAX_GRID_DIM)
    ok_h = expect("height", lambda x: _is_int(x) and 1 <= x <= MAX_GRID_DIM, "must be an integer in 1..%d" % MAX_GRID_DIM)
    ok_c = expect("cell_m", lambda x: _is_num(x) and x > 0, "must be a positive number")
    if expect("cell_ground_m", lambda x: _is_num(x) and x > 0, "must be a positive number") and ok_c:
        if meta["cell_ground_m"] > meta["cell_m"] * 1.0001:
            bad("meta.cell_ground_m: larger than cell_m (ground metres cannot exceed mercator metres)")
    bm_ok = expect("bounds_merc", lambda x: isinstance(x, list) and len(x) == 4 and all(_is_num(t) for t in x)
                   and x[0] < x[2] and x[1] < x[3], "must be [xmin, ymin, xmax, ymax] numbers, min < max")
    if bm_ok and ok_c:
        x0, y0, x1, y1 = meta["bounds_merc"]
        cell = meta["cell_m"]
        for key, ok, lo, hi in (("width", ok_w, x0, x1), ("height", ok_h, y0, y1)):
            if not ok:
                continue
            n = _cells_across(lo, hi, cell)
            if n is None:
                bad("meta.%s: bounds_merc / cell_m is not a finite number of cells" % key)
            elif n != meta[key]:
                bad("meta.%s: %s does not match bounds_merc / cell_m = %d" % (key, meta[key], n))
    if expect("bounds_lnglat", lambda x: isinstance(x, list) and len(x) == 4 and all(_is_num(t) for t in x)
              and x[0] < x[2] and x[1] < x[3], "must be [west, south, east, north] numbers") and bm_ok:
        x0, y0, x1, y1 = meta["bounds_merc"]
        with np.errstate(all="ignore"):                  # absurd bounds overflow the projection: still just a mismatch
            w, s = merc_to_lnglat(x0, y0)
            e, n = merc_to_lnglat(x1, y1)
        if not max(abs(a - float(b)) for a, b in zip(meta["bounds_lnglat"], (w, s, e, n))) <= 1e-5:
            bad("meta.bounds_lnglat: does not match bounds_merc")
    expect("depth_encoding", lambda x: isinstance(x, dict) and x.get("type") == "sqrt" and x.get("max_m") == DEPTH_MAX,
           "must be {'type': 'sqrt', 'max_m': 8.0}")
    expect("vel_encoding", lambda x: isinstance(x, dict) and x.get("vmax_ms") == VMAX and x.get("zero_code") == ZERO_CODE,
           "must be {'vmax_ms': 4.0, 'zero_code': 128}")
    expect("wet_threshold_m", lambda x: x == WET_THRESHOLD, "must be %g" % WET_THRESHOLD)
    expect("qmax_m2s", lambda x: x == QMAX, "must be %g" % QMAX)
    hours_ok = expect("hours", lambda x: isinstance(x, list) and all(_is_int(t) and 1 <= t <= HOURS_TOTAL for t in x)
                      and x == sorted(set(x)), "must be a sorted list of unique integers in 1..24")
    expect("storm_depth_in", lambda x: x is None or (_is_num(x) and x >= 0), "must be a number or null")
    expect("run_summary", lambda x: x is None or isinstance(x, dict), "must be an object or null")
    if "built" not in meta:
        bad("meta.built: missing")
    else:
        try:
            datetime.datetime.fromisoformat(str(meta["built"]).replace("Z", "+00:00"))
        except ValueError:
            bad("meta.built: not an ISO-8601 time (got %r)" % (meta["built"],))

    if not expect("hourly", lambda x: isinstance(x, list) and all(isinstance(e, dict) for e in x), "must be a list of objects"):
        return
    seen = []
    for e in meta["hourly"]:
        miss = [k for k in _HOURLY_KEYS if k not in e]
        if miss:
            bad("meta.hourly entry %r: missing %s" % (e.get("hour"), ", ".join(miss)))
            continue
        if not _hourly_ok(e):
            bad("meta.hourly hour %r: every value must be a finite number >= 0" % e["hour"])
            continue
        seen.append(e["hour"])
        if e["sim_seconds"] != e["hour"] * 3600:
            bad("meta.hourly hour %d: sim_seconds %s != hour*3600" % (e["hour"], e["sim_seconds"]))
        if e["max_speed_ms"] > VMAX + 1e-6:
            bad("meta.hourly hour %d: max_speed_ms %s exceeds vmax %g" % (e["hour"], e["max_speed_ms"], VMAX))
        if e["wet_cells"] > 0 and e["max_depth_m"] < WET_THRESHOLD:
            bad("meta.hourly hour %d: wet cells but max_depth_m below the wet threshold" % e["hour"])
        if e["mean_depth_m"] > e["max_depth_m"] + 1e-6:
            bad("meta.hourly hour %d: mean_depth_m exceeds max_depth_m" % e["hour"])
    if hours_ok and sorted(seen) != meta["hours"]:
        bad("meta.hourly: hours %s do not match meta.hours %s" % (sorted(seen), meta["hours"]))


_HOUR_FILE = re.compile(r"^h(\d\d)\.rgba\.gz$")


def _read_checked(path, w, h, label, bad):
    try:
        return read_rgba_gz(path, w, h)
    except (EOFError, OSError, ValueError, zlib.error) as e:
        bad("%s: %s" % (label, e))
        return None


_SCAN_PIXELS = 1 << 20             # pixels per row batch when scanning a grid (bounds the temporaries)


def _row_batches(h, w):
    step = max(1, _SCAN_PIXELS // max(w, 1))
    for r0 in range(0, h, step):
        yield slice(r0, min(h, r0 + step))


def _scan_hour(rgba, min_code, a_ref, first, rmax, count, qcode_max):
    """
    Row-batched pixel checks of one hour file. Returns a dict of totals; updates rmax (max depth code), count (hours
    wet) and qcode_max (upper bound of depth*speed as a q code) in place, and fills a_ref (valid mask) when `first`.
    """
    h, w = rgba.shape[:2]
    t = {"n_alpha": 0, "a_diff": 0, "n_low": 0, "n_out": 0, "n_dry": 0, "n_wet": 0, "n_over": 0,
         "worst": (-1e9, 0.0, 0.0), "dec_sum": 0.0, "dec_max": 0.0}
    rounding = 0.5 * _VEL_STEP * math.sqrt(2.0)          # two velocity components, each off by up to half a step
    for sl in _row_batches(h, w):
        px = rgba[sl]
        R, G, B, A = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
        t["n_alpha"] += int(((A != 0) & (A != 255)).sum())
        valid = A == 255
        if first:
            a_ref[sl] = valid
        else:
            t["a_diff"] += int((valid != a_ref[sl]).sum())
        wet = R > 0
        t["n_low"] += int((wet & (R < min_code)).sum())
        t["n_out"] += int((wet & ~valid).sum())
        t["n_dry"] += int((~wet & ((G != ZERO_CODE) | (B != ZERO_CODE))).sum())
        Rw = R[wet]
        if not Rw.size:
            continue
        speed = np.hypot(decode_vel(G[wet]).astype(np.float64), decode_vel(B[wet]).astype(np.float64))
        # The code only says the true depth lies within half a code step of decode(R), so the Froude bound has to be
        # evaluated at the UPPER end of that interval; the velocity codes add at most 0.71 step of rounding.
        d_up = (np.minimum(Rw.astype(np.float64) + 0.5, 255.0) / 255.0) ** 2 * DEPTH_MAX
        excess = speed - (np.minimum(np.sqrt(GRAV * d_up), VMAX) + _VEL_STEP)
        n_over = int((excess > 0).sum())
        if n_over:
            t["n_over"] += n_over
            k = int(np.argmax(excess))
            if excess[k] > t["worst"][0]:
                t["worst"] = (float(excess[k]), float(speed[k]), float(decode_depth(Rw[k])))
        dec = decode_depth(Rw).astype(np.float64)
        t["n_wet"] += int(Rw.size)
        t["dec_sum"] += float(dec.sum())
        t["dec_max"] = max(t["dec_max"], float(dec.max()))
        # flowpaths G is built from the TRUE depth * speed, and code 255 holds every depth from 7.97 m upward, so a
        # pixel at the depth clip has no upper bound on q below the format's own QMAX (code 255).
        q_hi = np.where(Rw == 255, np.uint8(255), encode_q(d_up * (speed + rounding)))      # true speed <= decoded + rounding
        q = qcode_max[sl]
        q[wet] = np.maximum(q[wet], q_hi)
        np.maximum(rmax[sl], R, out=rmax[sl])
        count[sl] += wet
    return t


def _scan_flowpaths(fp, min_code, a_ref, rmax, count, qcode_max, vs_hours):
    """Row-batched checks of flowpaths.rgba.gz; vs_hours also compares it with what the hour files imply."""
    h, w = fp.shape[:2]
    t = {"n_alpha": 0, "a_diff": 0, "n_zero": 0, "n_low": 0, "n_below": 0, "n_above": 0, "n_b": 0, "n_q": 0}
    for sl in _row_batches(h, w):
        px = fp[sl]
        R, G, B, A = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
        t["n_alpha"] += int(((A != 0) & (A != 255)).sum())
        if a_ref is not None:
            t["a_diff"] += int(((A == 255) != a_ref[sl]).sum())
        t["n_zero"] += int(((R == 0) & ((G != 0) | (B != 0))).sum())
        t["n_low"] += int(((R > 0) & (R < min_code)).sum())
        if vs_hours:
            Ri = R.astype(np.int16)
            rm = rmax[sl].astype(np.int16)
            t["n_below"] += int((Ri < rm).sum())
            t["n_above"] += int((Ri > rm + 1).sum())
            want_b = np.rint(255.0 * np.minimum(count[sl] / float(HOURS_TOTAL), 1.0))
            t["n_b"] += int(((np.abs(B.astype(np.float64) - want_b) > 1.0) & (R > 0)).sum())
            t["n_q"] += int((G.astype(np.int16) > qcode_max[sl].astype(np.int16) + 1).sum())
    return t


def validate_storm_dir(path):
    """
    Check a layers/flow/<storm> folder against FLOW_FORMAT.md using only its files. Returns a list of problem
    strings; empty means valid. Checked:
      no stray *.tmp files (what a killed write_rgba_gz / write_meta leaves behind);
      meta.json fields, types and internal consistency; meta.hours against the hNN.rgba.gz files present;
      each gz decompresses to exactly width*height*4 bytes; A only 0 / 255, A = 0 means no water, A identical in all
      files; dry cells (R = 0) have G = B = 128; no depth code below the wet threshold; speed <= min(sqrt(g*depth),
      vmax) plus one velocity quantisation step (depth taken at the upper half-step of its code, since the true
      depth may be that much deeper than the decoded one); hourly stats agree with the pixels (area and volume
      within 10 %, max depth not above the stats; the volume may only be lower in the pixels when the hour reports
      clipped_cells); flowpaths.rgba.gz: size, R = max over the hour files, B = hours wet, G <= the largest hourly
      depth*speed (except at pixels at the depth clip, R = 255: their true depth is unknown, and G is built from
      it), and R = 0 implies G = B = 0.
    Pixel data is scanned in row batches, so memory stays near one hour's rgba buffer even at 6000 x 6000.
    Never raises on bad data: a folder whose files are malformed in a way no check anticipated is reported with a
    "validator error" problem, because its validity could not be established.
    """
    problems = []
    try:
        _validate_storm_dir(str(path), problems.append)
    except (ArithmeticError, MemoryError, LookupError, TypeError, ValueError, AttributeError, OSError,
            RecursionError) as e:
        problems.append("validator error: %s: %s" % (type(e).__name__, e))
    return problems


def _validate_storm_dir(path, bad):
    if not os.path.isdir(path):
        bad("%s: not a directory" % path)
        return
    for name in sorted(os.listdir(path)):
        if name.endswith(".tmp"):
            bad("%s: stray temporary file left by an interrupted write (delete it)" % name)
    meta_path = os.path.join(path, "meta.json")
    if not os.path.isfile(meta_path):
        bad("meta.json: missing")
        return
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError, RecursionError) as e:
        bad("meta.json: unreadable (%s)" % e)
        return
    if not isinstance(meta, dict):
        bad("meta.json: top level must be an object")
        return
    _check_meta(meta, bad)

    w, h = meta.get("width"), meta.get("height")
    if not (_is_int(w) and _is_int(h) and 1 <= w <= MAX_GRID_DIM and 1 <= h <= MAX_GRID_DIM):
        # checked before any array is sized from them: width = height = 1e9 would ask for exabytes
        bad("pixel files not checked: width/height unusable (integers in 1..%d needed)" % MAX_GRID_DIM)
        return
    hours = meta.get("hours") if isinstance(meta.get("hours"), list) else []
    hours = [x for x in hours if _is_int(x)]
    on_disk = {}
    for name in os.listdir(path):
        m = _HOUR_FILE.match(name)
        if m:
            on_disk[int(m.group(1))] = name
    for hr in hours:
        if hr not in on_disk:
            bad("%s: listed in meta.hours but the file is missing" % hour_filename(hr))
    for hr in sorted(on_disk):
        if hr not in hours:
            bad("%s: file present but hour %d is not in meta.hours" % (on_disk[hr], hr))

    min_depth_code = int(encode_depth(WET_THRESHOLD))
    hourly_list = meta.get("hourly") if isinstance(meta.get("hourly"), list) else []
    hourly = {e["hour"]: e for e in hourly_list if _hourly_ok(e)}          # _check_meta reports the damaged entries
    cg = meta.get("cell_ground_m") if _is_num(meta.get("cell_ground_m")) else None
    rmax = np.zeros((h, w), np.uint8)
    count = np.zeros((h, w), np.uint8)
    qcode_max = np.zeros((h, w), np.uint8)
    a_ref = np.zeros((h, w), bool)
    have_ref = False
    n_read = 0
    for hr in hours:
        if hr not in on_disk:
            continue
        label = on_disk[hr]
        rgba = _read_checked(os.path.join(path, label), w, h, label, bad)
        if rgba is None:
            continue
        n_read += 1
        t = _scan_hour(rgba, min_depth_code, a_ref, not have_ref, rmax, count, qcode_max)
        have_ref = True
        del rgba
        if t["n_alpha"]:
            bad("%s: alpha channel has values other than 0 and 255 (%d pixels)" % (label, t["n_alpha"]))
        if t["a_diff"]:
            bad("%s: valid-domain (alpha) differs from the first hour file" % label)
        if t["n_low"]:
            bad("%s: %d pixels have a depth code below the wet threshold (%g m) but are not written dry"
                % (label, t["n_low"], WET_THRESHOLD))
        if t["n_out"]:
            bad("%s: %d wet pixels outside the valid domain (alpha 0)" % (label, t["n_out"]))
        if t["n_dry"]:
            bad("%s: %d dry pixels have a velocity (G, B must be 128)" % (label, t["n_dry"]))
        if t["n_over"]:
            bad("%s: %d pixels move faster than min(sqrt(g*depth), %g m/s) + one step; worst %.3f m/s at depth %.3f m"
                % (label, t["n_over"], VMAX, t["worst"][1], t["worst"][2]))
        e = hourly.get(hr)
        if t["n_wet"] and e is not None and cg:
            area_km2 = t["n_wet"] * cg * cg / 1e6
            vol = t["dec_sum"] * cg * cg
            slack = 3 * cg * cg
            if abs(area_km2 * 1e6 - e["wet_area_km2"] * 1e6) > 0.10 * max(area_km2, e["wet_area_km2"]) * 1e6 + slack:
                bad("%s: wet area %.4f km2 in the pixels vs %.4f km2 in meta.hourly" % (label, area_km2, e["wet_area_km2"]))
            vol_tol = 0.10 * max(vol, e["volume_m3"]) + slack * 0.1
            if e["clipped_cells"] > 0:
                # depth above 8 m is clipped in the pixels but not in the stats: the pixels can only hold less
                vol_bad = vol > e["volume_m3"] + vol_tol
            else:
                vol_bad = abs(vol - e["volume_m3"]) > vol_tol
            if vol_bad:
                bad("%s: volume %.0f m3 in the pixels vs %.0f m3 in meta.hourly" % (label, vol, e["volume_m3"]))
            if t["dec_max"] > e["max_depth_m"] * 1.02 + 0.04:
                bad("%s: deepest pixel %.3f m is above meta.hourly max_depth_m %.3f" % (label, t["dec_max"], e["max_depth_m"]))
        elif not t["n_wet"] and e is not None and e.get("wet_cells", 0) > 0:
            bad("%s: no wet pixels but meta.hourly says %s wet cells" % (label, e["wet_cells"]))

    fp_path = os.path.join(path, "flowpaths.rgba.gz")
    if not os.path.isfile(fp_path):
        bad("flowpaths.rgba.gz: missing")
        return
    fp = _read_checked(fp_path, w, h, "flowpaths.rgba.gz", bad)
    if fp is None:
        return
    t = _scan_flowpaths(fp, min_depth_code, a_ref if have_ref else None, rmax, count, qcode_max, n_read == len(hours))
    name = "flowpaths.rgba.gz"
    if t["n_alpha"]:
        bad("%s: alpha channel has values other than 0 and 255" % name)
    if t["a_diff"]:
        bad("%s: valid-domain (alpha) differs from the hour files" % name)
    if t["n_zero"]:
        bad("%s: %d pixels with max depth 0 have discharge or hours-wet set" % (name, t["n_zero"]))
    if t["n_low"]:
        bad("%s: %d pixels have a max-depth code below the wet threshold" % (name, t["n_low"]))
    if t["n_below"]:
        bad("%s: max depth is below an hour file at %d pixels" % (name, t["n_below"]))
    if t["n_above"]:
        bad("%s: max depth exceeds every hour file at %d pixels" % (name, t["n_above"]))
    if t["n_b"]:
        bad("%s: hours-wet (B) disagrees with the hour files at %d pixels" % (name, t["n_b"]))
    if t["n_q"]:
        bad("%s: mean unit discharge (G) exceeds the largest hourly value at %d pixels" % (name, t["n_q"]))
    return


def main(argv=None):
    """Command line: validate one or more storm folders; returns the exit status (0 ok, 1 problems, 2 usage)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) >= 2 and argv[0] == "validate":
        status = 0
        for d in argv[1:]:
            problems = validate_storm_dir(d)
            if problems:
                status = 1
                print("INVALID %s" % d)
                for p in problems:
                    print("  - " + p)
            else:
                print("OK %s" % d)
        return status
    print("usage: python flowgrid.py validate <storm_dir> [<storm_dir> ...]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
