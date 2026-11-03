r"""
build_flow_from_rasters.py - viewer flow grids ("bgflow", FLOW_FORMAT.md) from the solver-v2 hourly depth rasters.

Input  : a run folder with depth_<SSSSSS>s.tif (float32 metres, nodata -9999, EPSG:32616, hours 1..24 = 003600..086400 s,
         t=0 is ignored), optionally max_depth.tif and run_summary.json.   City runs are 2 m cells, county runs 5 m
         (10404 x 12043 = 125 M cells per raster).
Output : <out>/meta.json, hNN.rgba.gz, flowpaths.rgba.gz with "source": "rasters" (validated by flowgrid.validate_storm_dir).
         Depth comes from the rasters; the velocity is DERIVED (flowgrid.face_velocity: Manning on the simulated water-surface
         slope z + depth), it is not solver output, and meta.json says so.

Streaming: nothing the size of a full raster is ever held. For every hour the raster is read in row strips (default 512 rows
plus one halo row each side), the velocity is computed per strip at the solver's native resolution, and the strip is
block-averaged by k (flowgrid.block_reduce: mean depth, depth-weighted velocity) straight into a coarse UTM grid
(county: k = 4 -> 20 m, 2610 x 3010 cells; city: k = 3 -> 6 m). Only that coarse grid (3 float32 arrays) lives across strips.
Then: rotate (u, v) from UTM grid axes to true east/north, warp bilinearly onto the Web Mercator output grid (depth by mean,
velocity depth-weighted), limit the speed to min(vmax, sqrt(g h)), pack, write. Peak RAM per worker is about 0.5 GB at county
size (measured, see COUNTY_RUN.md), so --workers 4..8 fits an 8 GB machine / a Vista node.

Terrain (--dem) is read in the same strips and resampled to the solver grid (direct windowed read when the grids are
aligned, otherwise a bilinear WarpedVRT). Without --dem the surface is the water depth alone (flat bed) and the velocity is
only indicative: meta.source_note says so. Manning n: --mann raster (strips) or 0.035.

Output grid: default cell 25 m (mercator, about 20 m on the ground) when the solver domain is over 30 km wide (county), else
6.4 m (the city grid of build_flow_from_tiers.py). Cropped to the wet area of max_depth.tif (the whole raster when there is
none) plus 150 m, bounds snapped outward to multiples of the cell. One storm of the county is about 2.6k x 3k px = 31 MB raw
per hour (a few MB gzipped).

Resume: every hour is written to a work folder (<out parent>/.<out name>.building) together with hNN.json (its input key); a rerun
skips hours whose key matches (same raster size/mtime, cell, DEM, Manning, block size). A storm whose finished output already
has the same key is skipped altogether (--force rebuilds). The folder is swapped in only after validate_storm_dir finds
nothing. Do not run two conversions of the same output at once.

Usage
  python build_flow_from_rasters.py --run <folder with depth_*.tif> --out <layers/flow/<storm>> --storm 100yr
         [--label TEXT] [--cell-m M] [--dem FILE.tif [--dem-units feet|metres]] [--mann FILE.tif] [--hours 1-24]
         [--workers N] [--strip-rows N] [--force]
Numpy + rasterio only (no scipy).
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import flowgrid as fg  # noqa: E402

NOAA_DEPTH_IN = {"2yr": 3.54, "5yr": 4.32, "10yr": 4.97, "25yr": 5.89, "50yr": 6.65, "100yr": 7.45}
COUNTY_STORMS = ("2yr", "5yr", "10yr", "25yr", "50yr", "100yr", "ev1", "ev2", "ev3", "ev4", "ev5", "ev6")
DEM_UNITS = {"dem_bg.tif": "feet", "dem_bg_2m.tif": "metres", "dem_warren_5m.tif": "metres"}
CELL_CITY_M = 6.4                 # mercator metres (5.1 m on the ground), same as build_flow_from_tiers.py
CELL_COUNTY_M = 25.0              # mercator metres (about 20 m on the ground)
COUNTY_DOMAIN_M = 30000.0         # solver domain wider than this -> county default cell
STRIP_ROWS_DEFAULT = 512
DRY_FLOOR_M = 0.005               # m, solver depth below this is treated as dry (the face rule ignores it anyway)
METHOD_VERSION = 1
WORK_STALE_S = 2 * 3600
_DEPTH_FILE = re.compile(r"^depth_(\d+)s\.tif$")


def _log(msg):
    print(msg, flush=True)


def _peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 1048576.0
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return float("nan")


# =====================================================================================================
# small helpers
# =====================================================================================================

def parse_hours(spec):
    """'1-24', '5', '1,2,5-7' -> sorted list of hours 1..24."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError("bad hours %r (use e.g. 1-24 or 1,2,5-7)" % spec)
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if a > b or a < 1 or b > fg.HOURS_TOTAL:
            raise ValueError("hours must be within 1..%d and increasing (got %r)" % (fg.HOURS_TOTAL, part))
        out.update(range(a, b + 1))
    if not out:
        raise ValueError("no hours given")
    return sorted(out)


def find_hour_rasters(run_dir):
    """{hour: path} for depth_<s>s.tif with s a positive multiple of 3600 (t = 0 and odd times are ignored)."""
    out = {}
    for name in sorted(os.listdir(run_dir)):
        m = _DEPTH_FILE.match(name)
        if not m:
            continue
        sec = int(m.group(1))
        if sec > 0 and sec % 3600 == 0 and sec // 3600 <= fg.HOURS_TOTAL:
            out[sec // 3600] = os.path.join(run_dir, name)
    return out


def file_key(path):
    st = os.stat(path)
    return "%s:%d:%d" % (os.path.basename(path), st.st_size, st.st_mtime_ns)


def utm_extent_to_merc(transform, rows, cols):
    """Mercator bounds of a block of cells, rows / cols = (first, last) inclusive (corners and edge midpoints projected)."""
    r0, r1 = rows
    c0, c1 = cols
    cs = np.array([c0, 0.5 * (c0 + c1 + 1), c1 + 1.0])
    rs = np.array([r0, 0.5 * (r0 + r1 + 1), r1 + 1.0])
    cc, rr = np.meshgrid(cs, rs)
    e = transform.c + transform.a * cc + transform.b * rr
    n = transform.f + transform.d * cc + transform.e * rr
    lon, lat = fg.utm16n_to_lnglat(e, n)
    mx, my = fg.lnglat_to_merc(lon, lat)
    return float(mx.min()), float(my.min()), float(mx.max()), float(my.max())


def cell_ground_m(grid):
    x0, y0, x1, y1 = grid["bounds_merc"]
    lat_c = float(fg.merc_to_lnglat(0.5 * (x0 + x1), 0.5 * (y0 + y1))[1])
    return float(grid["cell_m"]) * math.cos(math.radians(lat_c))


# =====================================================================================================
# raster rows on the solver grid
# =====================================================================================================

class RowReader:
    """
    Rows [a, b) of a single-band GeoTIFF on the solver grid (float32, NaN where there is no data), read lazily.
    Aligned grids (same CRS and cell size, whole-cell offset) are read as plain windows; anything else goes through a
    bilinear WarpedVRT onto the solver grid. `scale` converts the unit (feet -> metres 0.3048).
    """

    def __init__(self, path, grid, scale=1.0):
        import rasterio
        from rasterio.windows import Window
        self.path = str(path)
        self.grid = grid
        self.scale = float(scale)
        self._Window = Window
        self.ds = rasterio.open(self.path)
        ds, tr = self.ds, grid["transform"]
        self.nodata = ds.nodata
        res_ok = abs(ds.transform.a - tr.a) < 1e-9 and abs(ds.transform.e - tr.e) < 1e-9 and ds.transform.b == 0 and ds.transform.d == 0
        same_crs = ds.crs is not None and ds.crs == grid["crs"]
        self.vrt = None
        if res_ok and same_crs:
            dc = (ds.transform.c - tr.c) / tr.a
            dr = (ds.transform.f - tr.f) / tr.e
            if abs(dc - round(dc)) < 1e-3 and abs(dr - round(dr)) < 1e-3:
                self.off_c, self.off_r = int(round(dc)), int(round(dr))     # raster cell (0,0) sits at grid cell (off_r, off_c)
                self.mode = "window"
                return
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        self.mode = "warp"
        self.vrt = WarpedVRT(ds, crs=grid["crs"], transform=tr, width=grid["width"], height=grid["height"],
                             resampling=Resampling.bilinear, src_nodata=ds.nodata if ds.nodata is not None else fg.NODATA,
                             nodata=fg.NODATA)

    def read_rows(self, a, b):
        W = self.grid["width"]
        out = np.full((b - a, W), np.nan, np.float32)
        if self.mode == "window":
            ds = self.ds
            r0, r1 = max(a - self.off_r, 0), min(b - self.off_r, ds.height)       # raster rows wanted
            c0, c1 = max(-self.off_c, 0), min(W - self.off_c, ds.width)
            if r1 > r0 and c1 > c0:
                raw = ds.read(1, window=self._Window(c0, r0, c1 - c0, r1 - r0))
                out[r0 + self.off_r - a:r1 + self.off_r - a, c0 + self.off_c:c1 + self.off_c] = raw
        else:
            out[:] = self.vrt.read(1, window=self._Window(0, a, W, b - a))
        bad = ~np.isfinite(out) | (out <= -9000.0)
        if self.nodata is not None and self.mode == "window":
            bad |= out == self.nodata
        if self.scale != 1.0:
            out *= np.float32(self.scale)
        out[bad] = np.nan
        return out

    def close(self):
        if self.vrt is not None:
            self.vrt.close()
        self.ds.close()


def dem_units(path, units):
    if units and units != "auto":
        if units not in ("feet", "metres"):
            raise ValueError("--dem-units must be feet or metres")
        return units
    u = DEM_UNITS.get(os.path.basename(str(path)).lower())
    if u is None:
        raise ValueError("elevation units of %s are unknown: pass --dem-units feet|metres (dem_bg.tif is FEET, the Vista "
                         "dem_bg_2m.tif / dem_warren_5m.tif are metres)" % path)
    return u


def solver_grid(path):
    """The solver grid of a depth raster: dict(transform, crs, width, height, dx, nodata)."""
    import rasterio
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 32616:
            raise ValueError("%s: CRS %s, expected EPSG:32616 (UTM 16N)" % (path, ds.crs))
        tr = ds.transform
        if abs(abs(tr.a) - abs(tr.e)) > 1e-9 or tr.b != 0 or tr.d != 0 or tr.e >= 0:
            raise ValueError("%s: not a north-up grid with square cells" % path)
        return {"transform": tr, "crs": ds.crs, "width": ds.width, "height": ds.height, "dx": float(tr.a),
                "nodata": ds.nodata}


def _check_same_grid(ds, grid, path):
    if (ds.width, ds.height) != (grid["width"], grid["height"]) or any(
            abs(x - y) > 1e-6 for x, y in zip(tuple(ds.transform)[:6], tuple(grid["transform"])[:6])):
        raise ValueError("%s is not on the same grid as the other depth rasters" % path)


def _open_depth(path):
    import rasterio
    return rasterio.open(path)


def _read_depth_rows(ds, a, b):
    from rasterio.windows import Window
    raw = ds.read(1, window=Window(0, a, ds.width, b - a))
    bad = ~np.isfinite(raw) | (raw <= -9000.0)
    if ds.nodata is not None:
        bad |= raw == ds.nodata
    d = raw.astype(np.float32, copy=True)
    d[bad] = np.nan
    np.maximum(d, np.float32(0.0), out=d)           # NaN stays NaN; tiny negative solver noise becomes 0
    return d


# =====================================================================================================
# planning: domain mask and output grid
# =====================================================================================================

def _strip_bounds(H, k, strip_rows):
    sr = max(k, (int(strip_rows) // k) * k)
    Hc = (H // k) * k
    return [(r0, min(r0 + sr, Hc)) for r0 in range(0, Hc, sr)]


def plan_grid(run_dir, rasters, sg, cell_m, k_override, strip_rows, log=_log):
    """
    Output grid, block size and domain mask. Reads the first raster for the valid domain and max_depth.tif (if present)
    for the wet bounding box. Returns dict(grid, k, coarse_transform, valid_dst, wet_bbox).
    """
    H, W, dx, tr = sg["height"], sg["width"], sg["dx"], sg["transform"]
    if cell_m is None:
        cell_m = CELL_COUNTY_M if max(H, W) * dx > COUNTY_DOMAIN_M else CELL_CITY_M
    cell_m = float(cell_m)
    # block size from the ground size of an output cell at the latitude of the domain centre
    ce = tr.c + tr.a * W / 2.0
    cn = tr.f + tr.e * H / 2.0
    lat_c = float(fg.utm16n_to_lnglat(np.array([ce]), np.array([cn]))[1][0])
    cg = cell_m * math.cos(math.radians(lat_c))
    k = int(k_override) if k_override else max(1, int(round(cg / dx)))

    first = rasters[min(rasters)] if rasters else None
    t0_path = os.path.join(run_dir, "depth_000000s.tif")
    valid_src = t0_path if os.path.isfile(t0_path) else first
    Hk, Wk = H // k, W // k
    if Hk < 1 or Wk < 1:
        raise ValueError("raster smaller than one %dx%d block" % (k, k))
    valid_k = np.zeros((Hk, Wk), bool)
    with _open_depth(valid_src) as ds:
        _check_same_grid(ds, sg, valid_src)
        for r0, r1 in _strip_bounds(H, k, strip_rows):
            d = _read_depth_rows(ds, r0, r1)
            ok = np.isfinite(d)[:, :Wk * k]
            valid_k[r0 // k:r1 // k] = ok.reshape((r1 - r0) // k, k, Wk, k).any(axis=(1, 3))
    # wet bounding box (native cells) from max_depth.tif
    md = os.path.join(run_dir, "max_depth.tif")
    bbox = None
    if os.path.isfile(md):
        rmin = cmin = 1 << 60
        rmax = cmax = -1
        with _open_depth(md) as ds:
            _check_same_grid(ds, sg, md)
            for r0 in range(0, H, 1024):
                r1 = min(r0 + 1024, H)
                d = _read_depth_rows(ds, r0, r1)
                wet = d >= fg.WET_THRESHOLD
                rows = np.flatnonzero(wet.any(axis=1))
                if rows.size:
                    cols = np.flatnonzero(wet.any(axis=0))
                    rmin, rmax = min(rmin, r0 + int(rows[0])), max(rmax, r0 + int(rows[-1]))
                    cmin, cmax = min(cmin, int(cols[0])), max(cmax, int(cols[-1]))
        if rmax >= 0:
            bbox = (rmin, rmax, cmin, cmax)
        else:
            raise RuntimeError("max_depth.tif has no cell wetter than %.2f m: nothing to show" % fg.WET_THRESHOLD)
    if bbox is None:
        bbox = (0, H - 1, 0, W - 1)
    bounds = utm_extent_to_merc(tr, (bbox[0], bbox[1]), (bbox[2], bbox[3]))
    grid = fg.grid_from_wet_bounds(bounds, cell_m)
    from rasterio.transform import Affine
    c_tr = tr @ Affine.scale(k, k)
    valid_dst = fg.warp_to_grid(valid_k.astype(np.float32), c_tr, sg["crs"], grid, resampling="nearest") > 0.5
    mb = grid["width"] * grid["height"] * 4 / 1e6
    log("grid %d x %d px, cell %.2f merc m (%.2f m ground), block k=%d (%.1f m), %.1f MB raw per hour%s" % (
        grid["width"], grid["height"], cell_m, cell_ground_m(grid), k, k * dx, mb,
        "  WARNING: over ~40 MB per hour, consider a larger --cell-m" if mb > 45 else ""))
    return {"grid": grid, "k": k, "coarse_transform": c_tr, "valid_dst": valid_dst, "wet_bbox": bbox}


# =====================================================================================================
# one hour
# =====================================================================================================

def hour_fields(path, sg, k, strip_rows, dem, mann, log_peak=False):
    """
    Stream one depth raster in strips -> coarse UTM arrays (depth mean, u, v in GRID axes, m/s) of H//k x W//k blocks,
    plus raster statistics at native resolution. dem / mann are RowReader or None.
    """
    H, W, dx = sg["height"], sg["width"], sg["dx"]
    Hk, Wk = H // k, W // k
    dk = np.zeros((Hk, Wk), np.float32)
    uk = np.zeros((Hk, Wk), np.float32)
    vk = np.zeros((Hk, Wk), np.float32)
    vol = 0.0
    nwet = 0
    dmax = 0.0
    with _open_depth(path) as ds:
        _check_same_grid(ds, sg, path)
        for r0, r1 in _strip_bounds(H, k, strip_rows):
            a, b = max(r0 - 1, 0), min(r1 + 1, H)
            d = _read_depth_rows(ds, a, b)
            cen = d[r0 - a:r1 - a]
            fin = np.isfinite(cen)
            vol += float(np.sum(cen, where=fin, dtype=np.float64))
            nwet += int(np.count_nonzero(cen >= fg.WET_THRESHOLD))
            dmax = max(dmax, float(np.max(cen, where=fin, initial=0.0)))
            np.putmask(d, d < DRY_FLOOR_M, 0.0)                 # thinner than this counts as dry (also the halo rows)
            if not (cen > 0).any():                             # nothing wet in this strip: zero depth, zero velocity
                continue
            z = dem.read_rows(a, b) if dem is not None else np.zeros(d.shape, np.float32)
            n = mann.read_rows(a, b) if mann is not None else fg.N_DEFAULT
            ug, vg = fg.face_velocity(d, z, n, dx)
            del z
            ok = np.isfinite(cen)
            db, ub, vb, _ = fg.block_reduce(cen, ug[r0 - a:r1 - a], vg[r0 - a:r1 - a], k, valid=ok)
            dk[r0 // k:r1 // k] = db
            uk[r0 // k:r1 // k] = ub
            vk[r0 // k:r1 // k] = vb
            del d, cen, ug, vg, ub, vb, db
    return dk, uk, vk, {"raster_volume_m3": vol * dx * dx, "raster_wet_cells": nwet, "raster_max_depth_m": dmax}


def warp_hour(grid, c_tr, crs, dk, uk, vk, gamma):
    """
    Coarse UTM fields -> mercator grid. Depth: bilinear mean (volume conserving). Velocity: rotated to true east/north,
    depth-weighted bilinear mean. Speed limited to min(vmax, sqrt(g h)). Returns float32 depth, u_east, v_north.
    """
    ue, vn = fg.rotate_to_true(uk, vk, gamma)
    d = fg.warp_to_grid(dk, c_tr, crs, grid, resampling="bilinear")
    num_u = fg.warp_to_grid(dk * ue, c_tr, crs, grid, resampling="bilinear")
    num_v = fg.warp_to_grid(dk * vn, c_tr, crs, grid, resampling="bilinear")
    wet = d > 1e-6
    den = np.where(wet, d, 1.0)
    u = np.where(wet, num_u / den, 0.0).astype(np.float32)
    v = np.where(wet, num_v / den, 0.0).astype(np.float32)
    lim = np.minimum(fg.VMAX, np.sqrt(fg.GRAV * np.maximum(d, 0.0)))
    sp = np.hypot(u, v)
    sc = np.where(sp > lim, lim / np.where(sp > 0, sp, 1.0), 1.0).astype(np.float32)
    return d.astype(np.float32), u * sc, v * sc


def _write_json(path, obj):
    tmp = "%s.%d.%s.tmp" % (path, os.getpid(), uuid.uuid4().hex[:6])
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def process_hour(cfg, hour):
    """Worker: one hour -> work/hNN.rgba.gz + work/hNN.json. Returns the hNN.json dict."""
    t0 = time.time()
    sg = cfg["sg"]
    grid = cfg["grid"]
    work = cfg["work"]
    dem = RowReader(cfg["dem"], sg, cfg["dem_scale"]) if cfg.get("dem") else None
    mann = RowReader(cfg["mann"], sg) if cfg.get("mann") else None
    try:
        dk, uk, vk, st = hour_fields(cfg["rasters"][hour], sg, cfg["k"], cfg["strip_rows"], dem, mann)
    finally:
        if dem is not None:
            dem.close()
        if mann is not None:
            mann.close()
    t1 = time.time()
    gamma = fg.convergence_field_utm16n(cfg["coarse_transform"], dk.shape[0], dk.shape[1])
    d, u, v = warp_hour(grid, cfg["coarse_transform"], sg["crs"], dk, uk, vk, gamma)
    del dk, uk, vk, gamma
    valid = np.load(os.path.join(work, "valid.npy"))
    entry = fg.hourly_entry(hour, d, u, v, cfg["cg"])
    rec = {"hour": hour, "key": cfg["hour_keys"][hour], "hourly": entry, "dry": entry["wet_cells"] == 0}
    rec.update(st)
    if not rec["dry"]:
        fg.write_rgba_gz(os.path.join(work, fg.hour_filename(hour)), fg.pack_rgba(d, u, v, valid))
    _write_json(os.path.join(work, "h%02d.json" % hour), rec)
    rec["seconds"] = time.time() - t0
    rec["read_seconds"] = t1 - t0
    rec["peak_mb"] = _peak_mb()
    gc.collect()
    return rec


# =====================================================================================================
# one storm
# =====================================================================================================

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            o = json.load(f)
        return o if isinstance(o, dict) else None
    except (OSError, ValueError):
        return None


def source_note(sg, k, have_dem, have_mann, dem_name):
    base = "depth rasters from solver v2 (%g m cells, block-averaged %dx to %g m), " % (sg["dx"], k, sg["dx"] * k)
    if have_dem:
        vel = ("velocity DERIVED by Manning from the simulated water-surface slope (terrain %s + depth), not solver output; "
               % dem_name)
    else:
        vel = ("velocity DERIVED by Manning from the slope of the water DEPTH alone (no terrain was given, so the bed is "
               "treated as flat: indicative only), not solver output; ")
    return base + vel + ("Manning n from raster" if have_mann else "Manning n = 0.035")


def build_storm(run_dir, out_dir, storm, *, label=None, cell_m=None, dem=None, dem_units_arg="auto", mann=None, hours=None,
                workers=1, strip_rows=STRIP_ROWS_DEFAULT, k=None, force=False, log=_log):
    """
    Convert one run folder. Returns dict(dir, hours, dry_hours, seconds, problems, grid, hourly, skipped).
    Raises on invalid input; leaves a previous valid out_dir untouched on any failure.
    """
    t_start = time.time()
    run_dir = os.path.abspath(run_dir)
    out_dir = os.path.abspath(out_dir)
    rasters_all = find_hour_rasters(run_dir)
    if not rasters_all:
        raise SystemExit("no depth_<seconds>s.tif with a whole-hour time found in %s" % run_dir)
    want = parse_hours(hours) if hours else list(range(1, fg.HOURS_TOTAL + 1))
    rasters = {h: p for h, p in rasters_all.items() if h in want}
    if not rasters:
        raise SystemExit("none of the requested hours %s exist in %s (found %s)" % (want, run_dir, sorted(rasters_all)))
    missing = [h for h in want if h not in rasters]
    if missing and hours is None:
        log("[%s] WARNING: hours %s have no raster" % (storm, missing))
    sg = solver_grid(rasters[min(rasters)])
    dem_scale = None
    if dem:
        units = dem_units(dem, dem_units_arg)
        dem_scale = fg.FEET_TO_M if units == "feet" else 1.0
    dem_key = file_key(dem) if dem else None
    mann_key = file_key(mann) if mann else None

    hour_keys = {h: file_key(p) for h, p in rasters.items()}
    params = {"method": METHOD_VERSION, "cell_m": cell_m, "k": k, "dem": dem_key, "dem_scale": dem_scale, "mann": mann_key,
              "max_depth": file_key(os.path.join(run_dir, "max_depth.tif")) if os.path.isfile(os.path.join(run_dir, "max_depth.tif")) else None}
    build_key = hashlib.sha1(json.dumps([params, sorted(hour_keys.items())], sort_keys=True).encode()).hexdigest()[:16]

    old = _read_json(os.path.join(out_dir, "meta.json")) if os.path.isdir(out_dir) else None
    if old is not None and not force and old.get("source") == "rasters" and (old.get("rasters_build") or {}).get("key") == build_key:
        log("[%s] %s is already built from these inputs (key %s); --force rebuilds" % (storm, out_dir, build_key))
        return {"dir": out_dir, "skipped_all": True, "problems": [], "seconds": 0.0}

    root_parent = os.path.dirname(out_dir)
    os.makedirs(root_parent, exist_ok=True)
    work = os.path.join(root_parent, ".%s.building" % os.path.basename(out_dir))
    if os.path.isdir(work) and (force or (_read_json(os.path.join(work, "plan.json")) or {}).get("build_key") != build_key):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    plan = _read_json(os.path.join(work, "plan.json"))
    if plan is None or plan.get("build_key") != build_key or not os.path.isfile(os.path.join(work, "valid.npy")):
        t0 = time.time()
        pl = plan_grid(run_dir, rasters, sg, cell_m, k, strip_rows, log=lambda m: log("[%s] %s" % (storm, m)))
        np.save(os.path.join(work, "valid.npy"), pl["valid_dst"])
        grid = pl["grid"]
        plan = {"build_key": build_key, "k": pl["k"], "grid": {kk: grid[kk] for kk in ("bounds_merc", "width", "height", "cell_m")}}
        _write_json(os.path.join(work, "plan.json"), plan)
        log("[%s] planned in %.1f s" % (storm, time.time() - t0))
    grid = dict(plan["grid"])
    from rasterio.transform import Affine
    x0, y0, x1, y1 = grid["bounds_merc"]
    grid["transform"] = Affine(grid["cell_m"], 0.0, x0, 0.0, -grid["cell_m"], y1)
    k = plan["k"]
    cg = cell_ground_m(grid)
    c_tr = sg["transform"] @ Affine.scale(k, k)

    cfg = {"sg": sg, "grid": grid, "work": work, "rasters": rasters, "hour_keys": hour_keys, "k": k, "strip_rows": strip_rows,
           "dem": dem, "dem_scale": dem_scale, "mann": mann, "cg": cg, "coarse_transform": c_tr}

    recs, todo = {}, []
    for h in sorted(rasters):
        r = _read_json(os.path.join(work, "h%02d.json" % h))
        if r and r.get("key") == hour_keys[h] and (r.get("dry") or os.path.isfile(os.path.join(work, fg.hour_filename(h)))):
            recs[h] = r
            log("[%s] h%02d skipped (already converted)" % (storm, h))
        else:
            todo.append(h)

    def report(rec):
        e = rec["hourly"]
        recs[rec["hour"]] = rec
        log("[%s] h%02d wet %9d cells %8.3f km2 max %.2f m %.2f m/s vol %.3g m3 (raster %.3g) | read %.1fs total %.1fs | peak %.0f MB%s" % (
            storm, rec["hour"], e["wet_cells"], e["wet_area_km2"], e["max_depth_m"], e["max_speed_ms"], e["volume_m3"],
            rec["raster_volume_m3"], rec["read_seconds"], rec["seconds"], rec["peak_mb"], "  DRY" if rec["dry"] else ""))

    try:
        if workers <= 1 or len(todo) <= 1:
            for h in todo:
                report(process_hour(cfg, h))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
                futs = [ex.submit(process_hour, cfg, h) for h in todo]
                for f in as_completed(futs):
                    report(f.result())

        # ---- flowpaths from the written pixels, meta, validation
        written = [h for h in sorted(recs) if not recs[h]["dry"]]
        dry = [h for h in sorted(recs) if recs[h]["dry"]]
        if not written:
            raise RuntimeError("%s: every hour is dry on the output grid, nothing to write" % storm)
        W_, H_ = grid["width"], grid["height"]
        rmax = np.zeros((H_, W_), np.uint8)
        q = np.zeros((H_, W_), np.float32)
        cnt = np.zeros((H_, W_), np.uint8)
        for h in written:
            rgba = fg.read_rgba_gz(os.path.join(work, fg.hour_filename(h)), W_, H_)
            R = rgba[..., 0]
            wetp = R > 0
            sp = np.hypot(fg.decode_vel(rgba[..., 1]), fg.decode_vel(rgba[..., 2]))
            np.maximum(rmax, R, out=rmax)
            q += np.where(wetp, fg.decode_depth(R) * sp, 0.0).astype(np.float32)
            cnt += wetp
            del rgba, R, sp
        valid = np.load(os.path.join(work, "valid.npy"))
        fp = fg.pack_flowpaths(fg.decode_depth(rmax), q / len(written), cnt, valid)
        fg.write_rgba_gz(os.path.join(work, "flowpaths.rgba.gz"), fp)
        del fp, rmax, q, cnt
        for stale in os.listdir(work):                  # hour files of an earlier run that are not part of this one
            m = re.match(r"^h(\d\d)\.rgba\.gz$", stale)
            if m and int(m.group(1)) not in written:
                os.remove(os.path.join(work, stale))

        run_summary = _read_json(os.path.join(run_dir, "run_summary.json"))
        depth_in = NOAA_DEPTH_IN.get(storm)
        if run_summary and isinstance(run_summary.get("total_rain_mm"), (int, float)):
            depth_in = round(run_summary["total_rain_mm"] / 25.4, 2)
        if label is None:
            m = re.fullmatch(r"(\d+)yr", storm)
            m2 = re.fullmatch(r"ev(\d+)", storm)
            label = ("%s-year storm (county)" % m.group(1)) if m else (
                "Historical event %s (county)" % m2.group(1)) if m2 else storm
        note = source_note(sg, k, bool(dem), bool(mann), os.path.basename(dem) if dem else None)
        meta = fg.build_meta(storm, label, "rasters", grid, written, [recs[h]["hourly"] for h in written], depth_in,
                             run_summary, note)
        meta["rasters_build"] = {
            "key": build_key,
            "method": "hourly depth rasters streamed in row strips; face-Manning velocity at solver resolution; block mean (k) to "
                      "a coarse UTM grid; rotation to true east/north; bilinear warp (depth mean, depth-weighted velocity) to "
                      "the mercator grid; speed limited to min(vmax, sqrt(g h))",
            "solver_cell_m": sg["dx"], "block_k": k, "coarse_cell_m": sg["dx"] * k,
            "dem": os.path.basename(dem) if dem else None, "dem_units": "feet" if dem_scale == fg.FEET_TO_M else ("metres" if dem else None),
            "manning": os.path.basename(mann) if mann else "uniform 0.035",
            "raster_volume_m3": {str(h): round(recs[h]["raster_volume_m3"], 1) for h in sorted(recs)},
            "volume_ratio": {str(h): round(recs[h]["hourly"]["volume_m3"] / recs[h]["raster_volume_m3"], 4)
                             for h in written if recs[h]["raster_volume_m3"] > 0},
            "dry_hours": dry,
            "inputs": {str(h): hour_keys[h] for h in sorted(hour_keys)},
        }
        fg.write_meta(os.path.join(work, "meta.json"), meta)
        problems = fg.validate_storm_dir(work)
    except BaseException:
        log("[%s] interrupted/failed: finished hours stay in %s for a resume" % (storm, work))
        raise

    if problems:
        log("[%s] NOT installed: validate_storm_dir found %d problems; build kept in %s" % (storm, len(problems), work))
        final = work
    else:
        for f in ("plan.json", "valid.npy") + tuple("h%02d.json" % h for h in range(1, 25)):
            try:
                os.remove(os.path.join(work, f))
            except OSError:
                pass
        aside = None
        if os.path.isdir(out_dir):
            aside = "%s.old.%d" % (work, os.getpid())
            fg._replace_retry(out_dir, aside)
        try:
            fg._replace_retry(work, out_dir)
        except BaseException:
            if aside:
                os.replace(aside, out_dir)
            raise
        if aside:
            shutil.rmtree(aside, ignore_errors=True)
        final = out_dir
    secs = time.time() - t_start
    mb = sum(os.path.getsize(os.path.join(final, f)) for f in os.listdir(final)) / 1048576.0
    log("[%s] done in %.0f s: %d hour files + flowpaths, %.1f MB, validate_storm_dir: %s" % (
        storm, secs, len(written), mb, "OK" if not problems else "%d PROBLEMS" % len(problems)))
    for p in problems:
        log("[%s]   - %s" % (storm, p))
    return {"dir": final, "hours": written, "dry_hours": dry, "seconds": secs, "problems": problems, "grid": grid,
            "hourly": [recs[h]["hourly"] for h in written], "records": recs, "mb": mb}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Solver-v2 hourly depth rasters -> viewer flow grids (source 'rasters').")
    ap.add_argument("--run", required=True, help="folder with depth_<seconds>s.tif (+ max_depth.tif, run_summary.json)")
    ap.add_argument("--out", required=True, help="output storm folder, e.g. layers/flow/100yr")
    ap.add_argument("--storm", required=True, help="storm name: 2yr 5yr 10yr 25yr 50yr 100yr ev1..ev6 ...")
    ap.add_argument("--label", help="label for meta.json (default from the storm name)")
    ap.add_argument("--cell-m", type=float, help="output cell in MERCATOR metres (ground = x0.80); default 25 for a domain over "
                                                "30 km (county), 6.4 otherwise (city)")
    ap.add_argument("--dem", help="terrain GeoTIFF (any grid; county: dem/dem_warren_5m.tif, city: dem_bg_2m.tif or dem_bg.tif)")
    ap.add_argument("--dem-units", default="auto", help="feet | metres | auto (by file name; dem_bg.tif = feet)")
    ap.add_argument("--mann", help="Manning n GeoTIFF (county: dem/mannings_n_county.tif); default uniform 0.035")
    ap.add_argument("--hours", help="e.g. 1-24, 5, 1,2,5-7 (default every hour found)")
    ap.add_argument("--workers", type=int, default=1, help="hours converted in parallel (each about 0.5 GB at county size)")
    ap.add_argument("--strip-rows", type=int, default=STRIP_ROWS_DEFAULT)
    ap.add_argument("--force", action="store_true", help="rebuild even if the output is already built from the same inputs")
    a = ap.parse_args(argv)
    try:
        res = build_storm(a.run, a.out, a.storm, label=a.label, cell_m=a.cell_m, dem=a.dem, dem_units_arg=a.dem_units, mann=a.mann,
                          hours=a.hours, workers=a.workers, strip_rows=a.strip_rows, force=a.force)
    except (ValueError, RuntimeError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    return 1 if res.get("problems") else 0


if __name__ == "__main__":
    sys.exit(main())
