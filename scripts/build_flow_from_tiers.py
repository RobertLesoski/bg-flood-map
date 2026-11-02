r"""
build_flow_from_tiers.py - local preview of the "bgflow" flow grids (FLOW_FORMAT.md) from the depth-TIER polygons.

The real builder works from the solver's depth rasters on Vista. The only flood data on this computer are the tier
polygons (v2_<storm>_depth_<SSSSSS>s.geojson, EPSG:32616, props tier shallow|moderate|deep|severe), so this script turns
them into the same files: layers/flow/<storm>/{meta.json, hNN.rgba.gz, flowpaths.rgba.gz} with source "tiers".

Method (per storm, per hour)
  1. Stream-parse the tier file (never json.load: a 100 MB line costs over 1.5 GB) and rasterise the polygons onto the
     DEM grid (5.088 m, cell-centre test, all_touched=False). Tier -> depth by midpoint: shallow 0.2, moderate 0.45,
     deep 0.9, severe 1.5 m. Water below 0.1 m is not in the tiers, so it is not represented. Water outside the DEM
     footprint (the solver domain reaches about 1.3 km further west and 3.2 km further north) has no terrain here
     and is dropped; with the 100yr storm that is about 17 % of the polygon area. The DEM holds FEET (0.3048 m each;
     min 391, max 797 = the Barren River level and the hills of Warren County, and scripts/dem_preprocess.py reads it as
     feet), so the terrain is converted to metres before any slope is taken.
  2. Depth by midpoint is a staircase (0.2 / 0.45 / 0.9 / 1.5) on top of a 5 m DEM that carries cm-scale noise, so the
     water surface eta = z + h is rough: the slope-based velocity then shows jets along every tier edge and whirls in
     every pond (measured below). The surface (never the bed) is therefore replaced by the SMOOTHEST surface that is still
     consistent with the tiers: eta minimises the sum of squared differences between neighbouring wet cells subject to
         z + (tier floor) <= eta <= z + (tier ceiling)       (shallow 0.1-0.3, moderate 0.3-0.6, deep 0.6-1.2, severe >1.2 m;
                                                              the open-ended severe tier is capped at 6 m)
     solved by projected red-black SOR on the graph of wet cells (about 60 sweeps; one sweep over 800 000 cells is
     50 ms; the energy has converged after about 50). The footprint is unchanged, wet cells stay wet (depth >= 0.05 m),
     and a pond becomes a flat surface (calm) while a street keeps its slope. Two details make it honest:
       - The tier footprint was made on the solver's 2 m terrain, this DEM has 5 m cells. On steep walls (sinkholes,
         banks: 3 m of relief per cell) the two disagree by up to half a cell, which makes the bands of neighbouring
         cells infeasible (a severe cell next to a moderate cell that is 3 m higher). The bands are therefore widened by
         the bed relief inside the 3x3 neighbourhood beyond RELIEF_FREE_M (0.6 m, i.e. about a 6 % slope); flat ground
         keeps the exact bands.
       - Dry cells touching the footprint carry a thin film of water in reality (the tiers start at 0.1 m). For the
         velocity only, they get depth = surface of the wet neighbours minus bed (clipped at 0 where the bed is higher),
         so a wet cell never reads a drop of its own depth into a "bone-dry" neighbour. A halo capped at 0.1 m (the first
         version) left (h - 0.1)/dx = 7 % of surface drop at a 0.45 m cell, and every pond at rest jetted outward from its
         rim at 1.2 m/s, whatever its size. These halo cells are never written.
     The volume is NOT forced back to the midpoint volume: the midpoint underestimates the open-ended severe tier (the 100yr
     hour 24 has 21.4 M m3 by midpoints inside the DEM against 31.3 M m3 stored in the solver's run_summary for the whole
     domain), and a multiplicative correction would tilt the flat ponds. The ratio smoothed/midpoint is logged per hour,
     stored in meta.tiers_build and warned about above VOLUME_GUARD.
  3. flowgrid.face_velocity with dx = DEM cell (Manning n from buildings 0.08, roads 0.016 buffered 4 m each side,
     buildings win, 0.035 elsewhere), rotate UTM grid axes to true east/north (grid convergence per cell), then limit the
     unit discharge h*|v| to QMAX (2 m2/s, the largest value flowpaths.rgba.gz can hold): in deep water Manning is
     ill-conditioned (3 m of water moves at 1.9 m/s on a 0.1 % slope) and the tiers fix the surface only to 0.1 m, so
     the speed there is a prior, not an inference. Between the two sits a slope DEAD BAND (--dead-band, 0.002): the
     surface slope S is only believed above 0.2 % (2 cm per 10 m, about what a tier band leaves over a pond), Manning
     is applied to max(S - 0.002, 0), i.e. the speed is scaled by sqrt(max(1 - 0.002/S, 0)). Without it the interior
     of an isolated pond (the surface is smooth but never exactly flat, and Manning multiplies a 1 cm residual by
     h^(2/3)/n) moved at a median 0.62 m/s and 22 % of it above 1 m/s (measured below); with it 0.00 m/s and 6.5 %.
  4. Warp onto the Web Mercator grid planned from the union of the wet bounds over all hours of the storm
     (cell_m = 6.4 = the DEM ground size, 5.1 m, in mercator metres). The wet footprint comes from a nearest-neighbour
     mask (so wet area is kept and no source cell is lost); depth and velocity are the bilinear averages over WET source
     cells only, so they do not fade to zero at the water's edge. Quantise, write hNN.rgba.gz, accumulate
     flowpaths from the decoded pixels, write meta.json.

What the numbers say (100yr, hour 13 = 046800s, 802 280 wet DEM cells, DEM in metres, measured with this module; the "rev"
column is the share of neighbouring wet cell pairs, both moving faster than 0.05 m/s, whose velocities point in opposite
directions; "ponds" is the share of the 3 151 isolated water bodies of 40-400 cells whose mean speed is above 1 m/s;
"outside" is the share of wet cells whose depth left its tier interval; unit-discharge cap on):
    surface                          rev     median m/s   p99 m/s   ponds>1m/s   outside   volume vs midpoint
    midpoints, no smoothing          28.6 %     1.05        2.22       75 %        0 %          1.00
    Gaussian of eta, sigma 1.5        0.9 %     0.47        2.33       27 %       11 %          0.99
    Gaussian of eta, sigma 3.0        0.5 %     0.00        2.14        9 %       16 %          1.00
    bands, exact (SOR)                4.0 %     0.00        2.30       16 %        0 %          1.07
    bands, relief-widened (SOR)       3.4 %     0.00        2.16        1 %       10 %          1.11     <- used
Gaussian smoothing of eta needs a large sigma (15 m, wider than a street) to calm the ponds (9 % of them are still above
1 m/s), and then it moves 16 % of the cells out of their tier; the band solution keeps the channels as coherent streams and
flattens the ponds, and its widened bands (steep walls only) are the 10 %. A final Gaussian polish of the solved surface
(--sigma, inside the bands) was measured at 0.7 / 1.0 / 1.5 / 2.0 cells: it raises the reversal share (3.4 % to 3.4 / 3.7 /
4.7 / 5.7 %) and lowers the pond interior above 1 m/s only from 6.5 % to 6.2 / 5.6 / 4.5 / 3.7 %, so the default is 0 (off).
The cap changes the p99 speed from 2.61 to 2.16 m/s and the ponds above 1 m/s from 2.2 to 1.1 %.
The remaining reversals sit on one-cell-wide gutters and isolated cells (2 m polygons sampled by 5 m cells), where
neighbouring cells are not hydraulically connected.
The dead band, same hour, on the DEM grid (surface slope over the wet cells, halo included: median 0.07 %, p90 0.59 %, so
74 % of the water is below the line):
    dead band                     pond interior median / p90 / >1 m/s    cells moving (>0.1 m/s)   shallow steep ground,
                                                                                                    moving cells pointing downhill
    0 (off)                              0.62 / 1.30 m/s / 22 %                  79 %                      71 %
    0.001                                0.00 / 1.07 m/s / 12 %                  40 %                      77 %
    0.002  <- used                       0.00 / 0.77 m/s /  6 %                  25 %                      80 %
    0.005                                0.00 / 0.00 m/s /  2 %                  12 %                      84 %
("pond interior" = cells of isolated water bodies of 40-400 cells that are more than 2.5 cells from the water's edge;
"shallow steep ground" = depth < 0.3 m on bed slope above 2 % measured over 10 m, direction = downhill of that bed.)
So the dead band trades coverage for honesty: velocity is only drawn where the tiers resolve a slope, and it points
downhill more often there. The Barren River and the other wide flat water read as at rest.
The halo, on the written grids (100yr hour 13: every number above is the DEM grid, these are output pixels). Water bodies of
1-3 pixels are the puddles of the first hours: with the halo capped at 0.1 m the 5yr hour 5 puddles had a median speed of
1.30 m/s, 98 % of them above 0.1 m/s and 69 % above 1 m/s; uncapped they read 0.00 m/s, 6 % and 1 % (5yr hour 6: 73 % to 0 %
above 1 m/s, 100yr hour 3: 69 % to 0 %). At 100yr hour 13 the share of wet pixels above 0.1 m/s fell from 41.8 to 27.9 %
(78.2 % with the dead band off) and above 1 m/s from 10.5 to 7.7 %; the pixels of wide water within 1.5 pixels of the dry edge
(local half-width of 6 pixels or more) are still 10 % above 1 m/s (12 % before), the interior of wide water 2 %.

Honest labelling: depth is quantised and smoothed (a pond's level is known to about +-0.1 m, deep water is calmer than
Manning would say, thin gutters are fragmentary), velocities are derived from a smoothed surface slope: indicative only.
meta.json carries source "tiers" and says so.

Usage
  python build_flow_from_tiers.py [--storms 5yr 10yr ... | all] [--app DIR] [--force] [--hours 3-24] [--sigma X]
                                  [--sweeps N] [--dead-band S] [--cell-m M] [--out DIR] [--cache DIR] [--crosscheck 8,13,20]
Resume: hours already written (meta.json lists them, the file is there, and the grid, the method parameters, the tier
file and dem_bg.tif / buildings.geojson / roads.geojson, compared by size and mtime, are unchanged) are skipped unless
--force; their pixels are re-read only to rebuild flowpaths. Hours of an earlier build that are not requested are kept, and
the grid is planned from the requested hours plus those, so `--hours 3-6` on a finished storm resumes instead of planning
another grid. The new folder is built in .<out>.<storm>.<pid>-<id>.building next to the output and swapped in only after
validate_storm_dir passes (two renames, the storm folder is missing for a moment): an interrupted build (Ctrl-C, full disk) or a
failed validation leaves the previous output as it was (a rerun redoes the hours of the interrupted run; hours of the previous
complete build are still skipped; a work folder that failed validation is kept and named in the log). Two builders of one storm
may run at once, the last one to finish stands; work folders of killed builds are removed by the next build of that storm after
2 h. Rasterised tier hours and the Manning classes are cached (--cache, default <temp>/bgflow_tier_cache, written atomically,
a damaged file is rebuilt) so the grid planning pass and a re-run do not parse the JSON again.
--crosscheck compares, for the given hours, the polygon area inside the DEM with the wet area of the DEM raster and of the
output grid.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import flowgrid as fg  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

APP_DEFAULT = "C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map"
STORMS = ("5yr", "10yr", "25yr", "50yr", "100yr")
NOAA_DEPTH_IN = {"2yr": 3.54, "5yr": 4.32, "10yr": 4.97, "25yr": 5.89, "50yr": 6.65, "100yr": 7.45}
TIERS = ("shallow", "moderate", "deep", "severe")          # raster code = index + 1, 0 = dry
TIER_MID = (0.2, 0.45, 0.9, 1.5)                           # m, depth assigned to each tier
TIER_LO = (0.1, 0.3, 0.6, 1.2)                             # m, tier_to_geojson bins: 0.1-0.3-0.6-1.2-inf
TIER_HI = (0.3, 0.6, 1.2, 6.0)                             # severe is open ended: 6 m is the cap of the smoothing
N_OPEN, N_BUILDINGS, N_ROADS = 0.035, 0.08, 0.016
ROAD_BUFFER_M = 4.0
DEM_Z_TO_M = fg.FEET_TO_M                                  # dem_bg.tif elevations are in feet (flowgrid.read_dem_m)
RELIEF_FREE_M = 0.6                                        # 3x3 bed relief up to this keeps the exact tier bands
SWEEPS_DEFAULT = 60                                        # projected SOR sweeps, see the module docstring
SOR_OMEGA = 1.85
SIGMA_DEFAULT = 0.0                                        # cells, optional Gaussian polish inside the bands
SLOPE_DEAD_BAND = 0.002                                    # |surface slope| the tiers cannot resolve (2 cm per 10 m)
MIN_DEPTH_M = 0.05                                         # wet cells stay at least this deep
HALO_MAX_M = 6.0                                           # velocity-only halo: wet neighbours' surface over the dry cell, same cap as the smoothing
VOLUME_GUARD = 0.25                                        # warn when |V_smooth / V_midpoint - 1| exceeds this
CELL_M_DEFAULT = 6.4                                       # mercator m, DEM ground size 5.088 m / cos(37 deg)
CACHE_VERSION = 1                                          # layout of the rasterised-tier and Manning caches
METHOD_VERSION = 2                                         # bump when the output changes for the same inputs (2: uncapped halo)

_HEAD = b'{"type":"FeatureCollection","features":['
_SEP = b',{"type":"Feature",'
_FEATURE_PREFIX = b'{"type":"Feature",'
_JSON_FALLBACK_MAX = 60 << 20                              # bytes: unknown file shapes are json.load-ed up to this size


# =====================================================================================================
# reading the tier files
# =====================================================================================================

def tier_file(app, storm, hour):
    sec = int(hour) * 3600
    return os.path.join(app, "BG FLOOD MAP VISTA SUPERCOMPUTER", "v2", storm, "v2_%s_depth_%06ds.geojson" % (storm, sec))


def parse_hours(spec):
    """'3-24', '5', '1,2,5-7' -> sorted list of hours 1..24."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError("bad hours %r (use e.g. 3-24 or 1,2,5-7)" % spec)
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if a > b or a < 1 or b > fg.HOURS_TOTAL:
            raise ValueError("hours must be within 1..%d and increasing (got %r)" % (fg.HOURS_TOTAL, part))
        out.update(range(a, b + 1))
    if not out:
        raise ValueError("no hours given")
    return sorted(out)


def iter_features(path, chunk=8 << 20):
    """
    Yield the parsed GeoJSON features of a tier file one at a time.

    Fast path: the exact shape tif_to_geojson.py writes, ONE line of compact JSON
    '{"type":"FeatureCollection","features":[{"type":"Feature",...},{"type":"Feature",...}]}'. Features are cut at the
    literal ',{"type":"Feature",', which cannot occur inside a feature (the strings in it are tier names and
    colours), so memory stays at about two chunks plus one feature. Any other file shape (pretty printed, extra
    members) falls back to json.load, which is refused above 60 MB because it would need over 1.5 GB at 100 MB.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = f.read(chunk)
        if not buf.startswith(_HEAD):
            if size > _JSON_FALLBACK_MAX:
                raise ValueError("%s: not the compact tier-file shape and too large (%d MB) to json.load"
                                 % (path, size >> 20))
            f.seek(0)
            for feat in json.load(f).get("features", []):
                yield feat
            return
        buf = buf[len(_HEAD):]
        while True:
            parts = buf.split(_SEP)
            more = f.read(chunk)
            if more:
                buf = parts.pop() + more                     # last part may end inside a separator: carry it over
            else:
                buf = b""
                last = parts[-1].rstrip()
                if not last.endswith(b"]}"):
                    raise ValueError("%s: truncated tier file (does not end with ']}')" % path)
                parts[-1] = last[:-2]
            for p in parts:
                if not p:
                    continue                                   # empty collection: '{...,"features":[]}'
                yield json.loads(p if p[:1] == b"{" else _FEATURE_PREFIX + p)
            if not more:
                return


def _polygons(geom):
    t = geom.get("type")
    if t == "Polygon":
        return [geom]
    if t == "MultiPolygon":
        return [{"type": "Polygon", "coordinates": c} for c in geom["coordinates"]]
    return []


def read_tier_raster(path, transform, shape, batch=20000, chunk=8 << 20):
    """
    Rasterise one tier file onto the grid (rasterio Affine `transform`, `shape` = (rows, cols)), all_touched=False.
    Returns (tier uint8 (rows, cols) with 0 dry and 1..4 = TIERS, info dict). Where polygons of different tiers
    overlap by a sliver (the files are simplified to 2 m) the deeper tier wins.
    """
    from rasterio.features import rasterize
    masks = [np.zeros(shape, np.uint8) for _ in TIERS]
    pending = [[] for _ in TIERS]
    info = {"features": 0, "skipped": 0, "by_tier": dict.fromkeys(TIERS, 0)}

    def flush():
        for k, geoms in enumerate(pending):
            if geoms:
                rasterize(geoms, out=masks[k], transform=transform, all_touched=False, default_value=1)
                geoms.clear()

    n_pending = 0
    for feat in iter_features(path, chunk=chunk):
        tier = (feat.get("properties") or {}).get("tier")
        polys = _polygons(feat.get("geometry") or {})
        if tier not in TIERS or not polys:
            info["skipped"] += 1
            continue
        k = TIERS.index(tier)
        for g in polys:
            pending[k].append((g, 1))
        info["features"] += 1
        info["by_tier"][tier] += 1
        n_pending += 1
        if n_pending >= batch:
            flush()
            n_pending = 0
    flush()
    out = np.zeros(shape, np.uint8)
    for k, m in enumerate(masks):
        out[m.astype(bool)] = k + 1
    return out, info


def _ring_area(ring):
    a = np.asarray(ring, dtype=np.float64)
    if len(a) < 4:
        return 0.0
    x = a[:, 0] - a[0, 0]
    y = a[:, 1] - a[0, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def source_area_km2(path, clip_bounds=None):
    """
    Independent of the rasteriser: area of the tier polygons by the shoelace formula (holes subtracted), per tier and
    in total, in km2. clip_bounds=(xmin, ymin, xmax, ymax) in the file's CRS also returns the part inside that box
    (shapely clip for the polygons that straddle it). Returns {"total": km2, "by_tier": {...}, "inside": km2 or None}.
    """
    by_tier = dict.fromkeys(TIERS, 0.0)
    inside = 0.0
    box = None
    if clip_bounds is not None:
        from shapely.geometry import box as shp_box
        box = shp_box(*clip_bounds)
        x0, y0, x1, y1 = clip_bounds
    for feat in iter_features(path):
        tier = (feat.get("properties") or {}).get("tier")
        for g in _polygons(feat.get("geometry") or {}):
            rings = g["coordinates"]
            area = abs(_ring_area(rings[0])) - sum(abs(_ring_area(r)) for r in rings[1:])
            if tier in by_tier:
                by_tier[tier] += area
            if box is not None:
                xs = [p[0] for p in rings[0]]
                ys = [p[1] for p in rings[0]]
                if min(xs) >= x0 and max(xs) <= x1 and min(ys) >= y0 and max(ys) <= y1:
                    inside += area
                elif max(xs) > x0 and min(xs) < x1 and max(ys) > y0 and min(ys) < y1:
                    from shapely.geometry import shape as shp_shape
                    inside += shp_shape(g).intersection(box).area
    return {"total": sum(by_tier.values()) / 1e6, "by_tier": {k: v / 1e6 for k, v in by_tier.items()},
            "inside": None if box is None else inside / 1e6}


# =====================================================================================================
# terrain and roughness
# =====================================================================================================

def load_dem(app):
    """
    The local DEM in METRES. Returns dict(z float32 (H,W) with NaN where the DEM has no data, valid bool (H,W),
    transform Affine, crs, shape, dx ground cell size in m, bounds (xmin, ymin, xmax, ymax) in UTM).
    """
    z, tr, crs = fg.read_dem_m(os.path.join(app, "dem_bg.tif"))              # feet -> metres, nodata -> NaN
    h, w = z.shape
    bounds = (tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f)
    return {"z": z, "valid": np.isfinite(z), "transform": tr, "crs": crs, "shape": z.shape, "dx": abs(tr.a), "bounds": bounds}


def _load_cache(path, key, *names):
    """The arrays `names` of an npz cache whose stored key is `key`; None for a missing, stale, truncated or garbage file."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh, np.load(fh) as z:   # our own handle: np.load leaks its file when the zip is damaged
            if str(z["key"]) != key:
                return None
            return [z[n] for n in names]
    except Exception:                                    # BadZipFile, EOFError, zlib.error, pickle errors, ...
        return None


def _save_cache(path, **arrays):
    """Write an npz cache atomically (temp file in the same folder + os.replace); a failed write leaves the old file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fg._atomic_write(path, lambda f: np.savez_compressed(f, **arrays))
    except OSError:                                      # a cache that cannot be written only costs time
        pass


def _file_key(path):
    st = os.stat(path)
    return "%d_%d" % (st.st_size, st.st_mtime_ns)


def _lonlat_rings_to_utm(rings):
    """list of (n,2) lon/lat arrays -> list of (n,2) UTM-16N arrays, one projection call for all points."""
    from rasterio.warp import transform
    if not rings:
        return []
    allc = np.concatenate(rings)
    xs, ys = transform("EPSG:4326", "EPSG:32616", allc[:, 0].tolist(), allc[:, 1].tolist())
    utm = np.column_stack([np.asarray(xs), np.asarray(ys)])
    out, i = [], 0
    for r in rings:
        out.append(utm[i:i + len(r)])
        i += len(r)
    return out


def manning_classes(dem, buildings_path, roads_path):
    """
    uint8 (H,W): 0 open, 1 road (centre line buffered by ROAD_BUFFER_M on each side), 2 building (wins over road),
    rasterised onto the DEM grid with the cell-centre test. Inputs are WGS84 GeoJSON.
    """
    from rasterio.features import rasterize
    from shapely.geometry import LineString, mapping
    tr, shape = dem["transform"], dem["shape"]
    x0, y0, x1, y1 = dem["bounds"]
    cls = np.zeros(shape, np.uint8)

    def inside(a):
        return a[:, 0].max() >= x0 and a[:, 0].min() <= x1 and a[:, 1].max() >= y0 and a[:, 1].min() <= y1

    if roads_path and os.path.isfile(roads_path):
        with open(roads_path, encoding="utf-8") as f:
            feats = json.load(f).get("features", [])
        lines = []
        for ft in feats:
            g = ft.get("geometry") or {}
            if g.get("type") == "LineString":
                lines.append(np.asarray(g["coordinates"], dtype=np.float64)[:, :2])
            elif g.get("type") == "MultiLineString":
                lines.extend(np.asarray(c, dtype=np.float64)[:, :2] for c in g["coordinates"])
        shapes = []
        for a in _lonlat_rings_to_utm([ln for ln in lines if len(ln) >= 2]):
            if inside(a):
                shapes.append((mapping(LineString(a).buffer(ROAD_BUFFER_M)), 1))
        if shapes:
            rasterize(shapes, out=cls, transform=tr, all_touched=False, default_value=1)
        del feats, lines, shapes
    if buildings_path and os.path.isfile(buildings_path):
        with open(buildings_path, encoding="utf-8") as f:
            feats = json.load(f).get("features", [])
        polys = []                                       # list of polygons, each a list of rings
        for ft in feats:
            g = ft.get("geometry") or {}
            if g.get("type") == "Polygon":
                polys.append([np.asarray(r, dtype=np.float64)[:, :2] for r in g["coordinates"]])
            elif g.get("type") == "MultiPolygon":
                polys.extend([np.asarray(r, dtype=np.float64)[:, :2] for r in p] for p in g["coordinates"])
        del feats
        flat = [r for p in polys for r in p]
        flat_utm = _lonlat_rings_to_utm(flat)
        shapes, i = [], 0
        for p in polys:
            rings = flat_utm[i:i + len(p)]
            i += len(p)
            if inside(rings[0]):
                shapes.append(({"type": "Polygon", "coordinates": [r.tolist() for r in rings]}, 2))
        if shapes:
            building = np.zeros(shape, np.uint8)
            rasterize(shapes, out=building, transform=tr, all_touched=False, default_value=2)
            cls[building == 2] = 2
        del polys, flat, flat_utm, shapes
    return cls


def manning_from_classes(cls):
    n = np.full(cls.shape, N_OPEN, np.float32)
    n[cls == 1] = N_ROADS
    n[cls == 2] = N_BUILDINGS
    return n


def load_manning(app, dem, cache_dir=None):
    """Manning n float32 (H,W) on the DEM grid; the class raster is cached (it costs about a minute to build)."""
    bpath = os.path.join(app, "layers", "buildings.geojson")
    rpath = os.path.join(app, "layers", "roads.geojson")
    key = "|".join(_file_key(p) if os.path.isfile(p) else "none" for p in (bpath, rpath))
    key += "|%s|%s|%d" % (dem["transform"].to_gdal(), dem["shape"], CACHE_VERSION)
    cache = os.path.join(cache_dir, "manning_classes.npz") if cache_dir else None
    hit = _load_cache(cache, key, "cls")
    if hit is not None:
        return manning_from_classes(hit[0])
    cls = manning_classes(dem, bpath, rpath)
    if cache:
        _save_cache(cache, cls=cls, key=np.array(key))
    return manning_from_classes(cls)


# =====================================================================================================
# water surface from the tiers
# =====================================================================================================

def _wet_graph(wet):
    """Cells of the footprint as a flat list with 4-neighbour indices (N = absent), counts and a red/black colour."""
    ys, xs = np.nonzero(wet)
    n = ys.size
    idx = np.full((wet.shape[0] + 2, wet.shape[1] + 2), n, np.int32)
    idx[ys + 1, xs + 1] = np.arange(n, dtype=np.int32)
    nb = np.stack([idx[ys, xs + 1], idx[ys + 2, xs + 1], idx[ys + 1, xs], idx[ys + 1, xs + 2]])
    cnt = (nb != n).sum(0)
    colour = ((ys + xs) & 1).astype(np.int8)
    return ys, xs, nb, cnt, colour


def _sor(eta0, lo, hi, nb, cnt, colour, sweeps, omega=SOR_OMEGA):
    """
    Projected red-black SOR for: minimise the sum over wet 4-neighbour pairs of (eta_i - eta_j)^2 with lo <= eta <= hi.
    Each update moves a cell toward the mean of its wet neighbours and clips to its bounds. Cells without a wet
    neighbour keep eta0. Returns eta (float64, same order as eta0).
    """
    n = eta0.size
    eta = np.zeros(n + 1)                         # eta[n] is the sentinel that absent neighbours point at (weight 0)
    eta[:n] = eta0
    groups = []
    for c in (0, 1):
        ids = np.flatnonzero((colour == c) & (cnt > 0))
        groups.append((ids, nb[:, ids].copy(), 1.0 / cnt[ids], lo[ids], hi[ids]))
    for _ in range(int(sweeps)):
        for ids, nbc, icnt, lo_c, hi_c in groups:
            mean = (eta[nbc[0]] + eta[nbc[1]] + eta[nbc[2]] + eta[nbc[3]]) * icnt
            old = eta[ids]
            eta[ids] = np.clip(old + omega * (mean - old), lo_c, hi_c)
    return eta[:n]


def _shifted_sum(a):
    """Sum of the four 4-neighbours of every cell (zero beyond the array)."""
    s = np.zeros_like(a)
    s[1:] += a[:-1]
    s[:-1] += a[1:]
    s[:, 1:] += a[:, :-1]
    s[:, :-1] += a[:, 1:]
    return s


def smooth_surface(tier, z, *, relief_free=RELIEF_FREE_M, sweeps=SWEEPS_DEFAULT, sigma=SIGMA_DEFAULT,
                   halo_max=HALO_MAX_M, min_depth=MIN_DEPTH_M):
    """
    Water depth from a tier raster: the smoothest surface consistent with the tier bands (module docstring).
    tier uint8 (H,W) 0 dry / 1..4; z float (H,W) bed in metres without NaN.
    sweeps=0 returns the plain tier midpoints. Returns dict:
      depth    float32 (H,W) depth on the footprint (tier > 0), 0 elsewhere, always >= min_depth on the footprint
      halo     float32 (H,W) depth (<= halo_max) of dry cells that touch the footprint, 0 elsewhere (velocity support only)
      eta      float64 (H,W) water surface relative to the lowest footprint bed (NaN outside the footprint)
      volume_ratio  sum(depth) / sum(midpoint depth)
    """
    mid = np.array((0.0,) + TIER_MID)
    lo_t = np.array((0.0,) + TIER_LO)
    hi_t = np.array((0.0,) + TIER_HI)
    wet = tier > 0
    out = {"depth": np.zeros(tier.shape, np.float32), "halo": np.zeros(tier.shape, np.float32),
           "eta": np.full(tier.shape, np.nan), "volume_ratio": 1.0}
    if not wet.any():
        return out
    ys, xs, nb, cnt, colour = _wet_graph(wet)
    t = tier[ys, xs]
    z64 = np.asarray(z, dtype=np.float64)
    zref = float(z64[ys, xs].min())
    zr = z64 - zref
    zc = zr[ys, xs]
    eta = zc + mid[t]
    h_mid_sum = float(mid[t].sum())
    if sweeps > 0:
        from scipy import ndimage as ndi
        dn = np.maximum(zr - ndi.minimum_filter(zr, 3, mode="nearest"), 0.0)[ys, xs]      # bed below this cell nearby
        up = np.maximum(ndi.maximum_filter(zr, 3, mode="nearest") - zr, 0.0)[ys, xs]      # bed above it
        half = 0.5 * relief_free
        lo = np.maximum(zc + min_depth, zc + lo_t[t] - np.maximum(dn - half, 0.0))
        hi = np.minimum(zc + hi_t[t] + np.maximum(up - half, 0.0), zc + 6.0)
        hi = np.maximum(hi, lo)
        eta = _sor(np.clip(eta, lo, hi), lo, hi, nb, cnt, colour, sweeps)
        if sigma > 0:
            from scipy import ndimage as ndi
            g = np.zeros(tier.shape)
            g[ys, xs] = eta
            num = ndi.gaussian_filter(g, sigma)
            den = ndi.gaussian_filter(wet.astype(np.float64), sigma)
            eta = np.clip(num[ys, xs] / np.maximum(den[ys, xs], 1e-9), lo, hi)
    depth = np.zeros(tier.shape, np.float64)
    depth[ys, xs] = eta - zc
    out["depth"] = depth.astype(np.float32)
    e_grid = np.full(tier.shape, np.nan)
    e_grid[ys, xs] = eta
    out["eta"] = e_grid
    # velocity-only halo: the surface of the wet neighbours continued over the dry cell, at most halo_max above its bed
    e0 = np.zeros(tier.shape)
    e0[ys, xs] = eta
    nsum = _shifted_sum(wet.astype(np.float64))
    ring = ~wet & (nsum > 0)
    ext = _shifted_sum(e0)
    halo = np.zeros(tier.shape)
    halo[ring] = np.clip(ext[ring] / nsum[ring] - zr[ring], 0.0, halo_max)
    out["halo"] = halo.astype(np.float32)
    out["volume_ratio"] = float(depth[ys, xs].sum() / h_mid_sum)
    return out


# =====================================================================================================
# velocity on the DEM grid
# =====================================================================================================

def surface_slope(h, z, dx, *, h_face_min=0.005, strip_rows=1024):
    """
    |grad(eta)| (float32, m/m) of the water surface eta = z + h per cell: the mean over the faces that carry flow (same
    test as flowgrid.face_velocity: water depth over the higher bed above h_face_min) of the face slope, per axis, then
    the length of the two-axis vector. A cell without a flowing face gets 0.
    """
    H, W = h.shape
    out = np.zeros((H, W), np.float32)
    for r0 in range(0, H, strip_rows):
        r1 = min(r0 + strip_rows, H)
        a0, a1 = max(r0 - 1, 0), min(r1 + 1, H)                 # one halo row so the strip's faces match the full grid
        zz = np.asarray(z[a0:a1], np.float64)
        eta = zz + np.asarray(h[a0:a1], np.float64)
        comp = []
        for ax in (0, 1):
            ea, eb = (eta[:-1], eta[1:]) if ax == 0 else (eta[:, :-1], eta[:, 1:])
            za, zb = (zz[:-1], zz[1:]) if ax == 0 else (zz[:, :-1], zz[:, 1:])
            flow = (np.maximum(ea, eb) - np.maximum(za, zb)) > h_face_min
            f = np.where(flow, (eb - ea) / dx, 0.0)
            num = np.zeros_like(eta)
            cnt = np.zeros_like(eta)
            if ax == 0:
                num[:-1] += f; cnt[:-1] += flow; num[1:] += f; cnt[1:] += flow
            else:
                num[:, :-1] += f; cnt[:, :-1] += flow; num[:, 1:] += f; cnt[:, 1:] += flow
            comp.append(num / np.maximum(cnt, 1))
        sl = np.hypot(comp[0], comp[1])
        out[r0:r1] = sl[r0 - a0:r1 - a0]
    return out


def velocity_fields(depth, halo, z, n, dx, transform, *, qmax=fg.QMAX, dead_band=SLOPE_DEAD_BAND):
    """
    True east / true north velocity (m/s, float32) on the window grid. depth is the footprint depth, halo the
    velocity-only support (see smooth_surface), z the bed (m), n Manning, dx the cell size (ground m), transform the
    window's UTM transform (for the grid convergence). The unit discharge h*|v| is limited to qmax; cells outside the
    footprint (depth 0) get zero velocity.
    dead_band: the tiers fix the surface to about 5 cm, so a slope S below `dead_band` is not measured. Manning is applied
    to the dead-banded slope max(S - dead_band, 0), i.e. the speed is scaled by sqrt(max(1 - dead_band/S, 0)): water that
    the tiers show flatter than that (ponds, the river, wide floodplain) is at rest, steeper water keeps nearly its speed
    (S = 10 x dead_band: -5 %). 0 switches it off.
    """
    h_vel = depth + halo
    ug, vg = fg.face_velocity_strips(h_vel, z, n, dx, strip_rows=1024)
    if dead_band > 0:
        s_cell = surface_slope(h_vel, z, dx)
        f = np.sqrt(np.maximum(1.0 - dead_band / np.maximum(s_cell, 1e-9), 0.0)).astype(np.float32)
        ug = ug * f
        vg = vg * f
        del s_cell, f
    gamma = fg.convergence_field_utm16n(transform, h_vel.shape[0], h_vel.shape[1])
    ue, vn = fg.rotate_to_true(ug, vg, gamma)
    sp = np.hypot(ue, vn)
    lim = qmax / np.maximum(depth, MIN_DEPTH_M)
    scale = np.where(sp > lim, lim / np.maximum(sp, 1e-9), 1.0).astype(np.float32)
    foot = depth > 0
    ue = np.where(foot, ue * scale, 0.0).astype(np.float32)
    vn = np.where(foot, vn * scale, 0.0).astype(np.float32)
    return ue, vn


def footprint_window(foot, pad=3):
    """(row slice, col slice) of the footprint's bounding box grown by pad cells, or None when it is empty."""
    rows = np.flatnonzero(foot.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(foot.any(axis=0))
    H, W = foot.shape
    return (slice(max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, H)),
            slice(max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, W)))


# =====================================================================================================
# the Mercator grid
# =====================================================================================================

def utm_extent_to_merc(transform, rows, cols):
    """
    Mercator bounds (xmin, ymin, xmax, ymax) of a block of DEM cells: rows, cols = (first, last) inclusive.
    The four corners and four edge midpoints of the block are projected, so the rotation between UTM grid and true
    north (up to 0.4 degrees here) is covered.
    """
    r0, r1 = rows
    c0, c1 = cols
    cs = np.array([c0, 0.5 * (c0 + c1 + 1), c1 + 1.0])
    rs = np.array([r0, 0.5 * (r0 + r1 + 1), r1 + 1.0])
    cc, rr = np.meshgrid(cs, rs)
    e = transform.c + transform.a * cc + transform.b * rr
    nn = transform.f + transform.d * cc + transform.e * rr
    lon, lat = fg.utm16n_to_lnglat(e, nn)
    mx, my = fg.lnglat_to_merc(lon, lat)
    return float(mx.min()), float(my.min()), float(mx.max()), float(my.max())


def cell_ground_m(grid):
    x0, y0, x1, y1 = grid["bounds_merc"]
    lat_c = float(fg.merc_to_lnglat(0.5 * (x0 + x1), 0.5 * (y0 + y1))[1])
    return float(grid["cell_m"]) * math.cos(math.radians(lat_c))


def sub_grid(grid, bounds, margin_px=2):
    """The part of `grid` covering mercator `bounds`, as another grid dict plus its (row0, col0) offset in the grid."""
    from rasterio.transform import Affine
    x0, y0, x1, y1 = grid["bounds_merc"]
    cell = grid["cell_m"]
    c0 = max(int(math.floor((bounds[0] - x0) / cell)) - margin_px, 0)
    c1 = min(int(math.ceil((bounds[2] - x0) / cell)) + margin_px, grid["width"])
    r0 = max(int(math.floor((y1 - bounds[3]) / cell)) - margin_px, 0)
    r1 = min(int(math.ceil((y1 - bounds[1]) / cell)) + margin_px, grid["height"])
    if c1 <= c0 or r1 <= r0:
        return None, (0, 0)
    sub = {"bounds_merc": [x0 + c0 * cell, y1 - r1 * cell, x0 + c1 * cell, y1 - r0 * cell], "width": c1 - c0,
           "height": r1 - r0, "cell_m": cell, "transform": Affine(cell, 0.0, x0 + c0 * cell, 0.0, -cell, y1 - r0 * cell)}
    return sub, (r0, c0)


def warp_hour(grid, win_transform, crs, foot, depth, ue, vn):
    """
    DEM-window fields -> the mercator grid. Footprint = nearest neighbour of the wet mask; depth and velocity =
    bilinear averages over wet source cells only (numerator and denominator are warped separately).
    Returns float32 (height, width) depth, u_east, v_north (zero where dry).
    """
    H, W = grid["height"], grid["width"]
    d_out = np.zeros((H, W), np.float32)
    u_out = np.zeros((H, W), np.float32)
    v_out = np.zeros((H, W), np.float32)
    r, c = foot.shape
    corners = utm_extent_to_merc(win_transform, (0, r - 1), (0, c - 1))
    sub, (r0, c0) = sub_grid(grid, corners)
    if sub is None:
        return d_out, u_out, v_out
    m = foot.astype(np.float32)
    near = fg.warp_to_grid(m, win_transform, crs, sub, resampling="nearest") > 0.5
    den = fg.warp_to_grid(m, win_transform, crs, sub, resampling="bilinear")
    ok = near & (den > 1e-3)
    h_, w_ = sub["height"], sub["width"]
    for src, dst in ((depth, d_out), (ue, u_out), (vn, v_out)):
        num = fg.warp_to_grid(src * m, win_transform, crs, sub, resampling="bilinear")
        dst[r0:r0 + h_, c0:c0 + w_] = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
    return d_out, u_out, v_out


# =====================================================================================================
# one storm
# =====================================================================================================

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


def _log(msg):
    print(msg, flush=True)


def tier_hour_cached(app, storm, hour, dem, cache_dir):
    """(tier raster uint8, info) for one hour; rasterised from the JSON once, then cached (compressed npz)."""
    path = tier_file(app, storm, hour)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    key = "%s|%s|%s|%d" % (_file_key(path), dem["transform"].to_gdal(), dem["shape"], CACHE_VERSION)
    cache = os.path.join(cache_dir, "%s_%02d.npz" % (storm, hour)) if cache_dir else None
    hit = _load_cache(cache, key, "tier", "info")
    if hit is not None:
        try:
            return hit[0], dict(json.loads(str(hit[1])), cached=True)
        except ValueError:
            pass
    t0 = time.time()
    tier, info = read_tier_raster(path, dem["transform"], dem["shape"])
    info["seconds"] = round(time.time() - t0, 1)
    if cache:
        _save_cache(cache, tier=tier, key=np.array(key), info=np.array(json.dumps(info)))
    return tier, info


def input_keys(app):
    """Size and mtime of the files that shape every hour (terrain, buildings, roads); 'none' when a file is absent."""
    paths = {"dem": os.path.join(app, "dem_bg.tif"), "buildings": os.path.join(app, "layers", "buildings.geojson"),
             "roads": os.path.join(app, "layers", "roads.geojson")}
    return {k: _file_key(p) if os.path.isfile(p) else "none" for k, p in paths.items()}


def _method_params(sigma, sweeps, cell_m, dead_band, inputs=None):
    p = {"relief_free_m": RELIEF_FREE_M, "sweeps": int(sweeps), "sigma_cells": float(sigma), "qmax_m2s": fg.QMAX,
         "slope_dead_band": float(dead_band), "tier_mid_m": list(TIER_MID), "dem_z_to_m": DEM_Z_TO_M,
         "halo_max_m": HALO_MAX_M, "cell_m": float(cell_m), "v": METHOD_VERSION}
    if inputs is not None:
        p["inputs"] = dict(inputs)
    return p


def source_note(storm, dead_band=SLOPE_DEAD_BAND):
    rest = ("Water whose smoothed surface is flatter than %g %% (ponds, the river, wide floodplain) is shown at rest because the "
            "tiers cannot resolve its slope. " % (100.0 * dead_band)) if dead_band > 0 else ""
    return ("Preview built locally from the solver's depth-TIER polygons (%s storm; tiers 0.1-0.3, 0.3-0.6, 0.6-1.2 and over "
            "1.2 m, nothing below 0.1 m) on the 5 m terrain, not from the depth rasters. Depth is the smoothest water surface "
            "consistent with the tiers, so a pond's level is known to about 0.1 m and the severe tier is open ended; water "
            "outside the local terrain (the north and west strips of the solver domain) is not shown. Velocity is derived "
            "from that smoothed surface slope with Manning (unit discharge limited to 2 m2/s), not solver output. %sIndicative "
            "only." % (storm, rest))


def _accumulate(acc, rgba):
    R = rgba[..., 0]
    wet = R > 0
    d = fg.decode_depth(R)
    sp = np.hypot(fg.decode_vel(rgba[..., 1]), fg.decode_vel(rgba[..., 2]))
    np.maximum(acc["rmax"], R, out=acc["rmax"])
    acc["q"] += np.where(wet, d * sp, 0.0).astype(np.float32)
    acc["count"] += wet
    acc["n"] += 1


def _read_meta(path):
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


WORK_STALE_S = 2 * 3600                                    # a work folder untouched this long belongs to a killed build


def _work_names(root, storm):
    """(work folder, stale folder) for one build: beside root, unique per call so two builders cannot share one."""
    tag = "%s.%s.%d-%s" % (os.path.basename(root), storm, os.getpid(), uuid.uuid4().hex[:8])
    base = os.path.dirname(root)
    return os.path.join(base, ".%s.building" % tag), os.path.join(base, ".%s.old" % tag)


def _clean_stale_work(root, storm):
    """Remove work folders of this storm that a killed build left behind (not touched for WORK_STALE_S)."""
    base = os.path.dirname(root)
    pat = re.compile(r"^\.%s\.%s\.\d+-[0-9a-f]{8}\.(building|old(\.\d+)?)$" % (re.escape(os.path.basename(root)), re.escape(storm)))
    try:
        names = os.listdir(base)
    except OSError:
        return
    for n in names:
        path = os.path.join(base, n)
        try:
            if pat.match(n) and os.path.isdir(path) and time.time() - os.stat(path).st_mtime > WORK_STALE_S:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


SWAP_TRIES = 6


def _swap_dir(work_dir, out_dir, stale_dir):
    """
    Put the finished work_dir at out_dir. The previous out_dir waits beside it until the new one is in place and comes back
    when the swap fails. Windows cannot replace a folder that is not empty, so the swap is two renames, and another builder
    of the same storm can swap in between them: the second rename then finds that builder's finished output at out_dir. It is
    moved aside under a fresh name (stale_dir, stale_dir.1, ...) and the swap is tried again, so the last build wins and
    neither fails. Reusing one name for the moved-aside folders would make every retry fail on the folder the first one left.
    """
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    aside = []                                           # folders moved out of the way, oldest first
    try:
        for attempt in range(SWAP_TRIES):
            name = stale_dir if attempt == 0 else "%s.%d" % (stale_dir, attempt)
            try:
                fg._replace_retry(out_dir, name)
                aside.append(name)
            except FileNotFoundError:                    # first build, or another builder is between its two renames
                pass
            if not os.path.exists(out_dir):              # (os.replace onto a folder that is not empty only costs the retries)
                try:
                    fg._replace_retry(work_dir, out_dir)
                    return
                except OSError:
                    if attempt == SWAP_TRIES - 1:
                        raise
            time.sleep(0.05)
        raise PermissionError("another builder kept putting its output at %s" % out_dir)
    except BaseException:
        if aside and not os.path.exists(out_dir):
            try:
                os.replace(aside[-1], out_dir)
                aside.pop()
            except OSError:
                pass
        raise
    finally:
        for p in aside:
            shutil.rmtree(p, ignore_errors=True)


def build_storm(app, storm, hours, dem, manning, *, sigma=SIGMA_DEFAULT, sweeps=SWEEPS_DEFAULT, dead_band=SLOPE_DEAD_BAND,
                force=False, cache_dir=None, cell_m=None, out_root=None, log=_log):
    """
    Build layers/flow/<storm> from the tier files of `hours` (list of 1..24). Returns a summary dict:
    storm, dir, hours (written), dry_hours (flagged, no file), skipped (resumed), seconds, problems (validate_storm_dir),
    grid, hourly (meta.hourly). out_root is the folder that holds <storm>/ (default <app>/layers/flow).
    The new folder is built beside out_root (.<root>.<storm>.<pid>-<id>.building, unique per call) and swapped in only when
    validate_storm_dir finds nothing, so an interrupted or failed build leaves the previous output untouched. Hours of an existing build that are not
    requested are kept when the grid and method match; the grid is planned from the requested hours and those, so a subset
    rerun keeps the grid of the full build.
    """
    t_start = time.time()
    root = os.path.abspath(out_root or os.path.join(app, "layers", "flow"))
    out_dir = os.path.join(root, storm)
    work_dir, stale_dir = _work_names(root, storm)
    _clean_stale_work(root, storm)
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "bgflow_tier_cache")
    z, valid = dem["z"], dem["valid"]
    tr, crs = dem["transform"], dem["crs"]

    # ---- existing output
    meta_path = os.path.join(out_dir, "meta.json")
    old = _read_meta(meta_path) if os.path.isfile(meta_path) else None
    if old is not None and old.get("source") != "tiers":
        raise SystemExit("%s: %s holds a %r build (full precision); refusing to overwrite it with a tier preview"
                         % (storm, out_dir, old.get("source")))
    existing = [f for f in os.listdir(out_dir) if re.match(r"^h\d\d\.rgba\.gz$", f)] if os.path.isdir(out_dir) else []
    disk_hours = sorted(int(f[1:3]) for f in existing) if old is not None else []
    disk_hours = [h for h in disk_hours if h not in hours and os.path.isfile(tier_file(app, storm, h))]
    plan_hours = sorted(set(hours) | set(disk_hours))

    # ---- pass 1: rasterise (cached) every hour once and collect the wet extent for the grid
    ext = None
    wet_count = {}
    for hour in plan_hours:
        t0 = time.time()
        tier, tinfo = tier_hour_cached(app, storm, hour, dem, cache_dir)
        foot = (tier > 0) & valid
        wet_count[hour] = int(foot.sum())
        log("[%s] h%02d rasterised (%s) %5.1fs, %d polygons, %d wet DEM cells | peak %.0f MB" % (
            storm, hour, "cache" if tinfo.get("cached") else "tier JSON, %d MB" % (os.path.getsize(tier_file(app, storm, hour)) >> 20),
            time.time() - t0, tinfo.get("features", 0), wet_count[hour], _peak_mb()))
        w = footprint_window(foot, pad=0)
        if w is not None:
            e = (w[0].start, w[0].stop - 1, w[1].start, w[1].stop - 1)
            ext = e if ext is None else (min(ext[0], e[0]), max(ext[1], e[1]), min(ext[2], e[2]), max(ext[3], e[3]))
        del tier, foot
    if ext is None:
        raise RuntimeError("%s: no wet cell inside the DEM in hours %s" % (storm, hours))
    bounds = utm_extent_to_merc(tr, (ext[0], ext[1]), (ext[2], ext[3]))
    cm = float(cell_m or CELL_M_DEFAULT)
    grid = fg.grid_from_wet_bounds(bounds, cm)
    cg = cell_ground_m(grid)
    log("[%s] grid %d x %d px, cell %.2f merc m (%.2f m ground), %d wet DEM cells in the largest hour" % (
        storm, grid["width"], grid["height"], cm, cg, max(wet_count.values())))
    params = _method_params(sigma, sweeps, cm, dead_band, input_keys(app))
    src_keys = {h: _file_key(tier_file(app, storm, h)) for h in plan_hours}

    # ---- resume only when the grid, the method and the input files are the same
    same = (old is not None and old.get("width") == grid["width"]
            and old.get("height") == grid["height"] and old.get("bounds_merc") == [float(b) for b in grid["bounds_merc"]]
            and (old.get("tiers_build") or {}).get("params") == params)
    build_hours = list(hours)
    if not same and (existing or old is not None):
        if not force:
            if old is None:
                raise SystemExit("%s: %s holds hour files but no readable meta.json (a build that was never finished, or a "
                                 "damaged folder), so nothing can be resumed; use --force to rebuild it" % (storm, out_dir))
            raise SystemExit("%s: existing output was built with another grid, method or input file; use --force to rebuild "
                             "it (%s)" % (storm, out_dir))
        build_hours = plan_hours                         # the old hour files are on another grid: rebuild what has a source
        old = None
    old_hourly = {e["hour"]: e for e in (old or {}).get("hourly", [])} if same else {}
    old_ratio = {int(k): v for k, v in ((old or {}).get("tiers_build") or {}).get("volume_ratio", {}).items()} if same else {}
    old_src = {int(k): v for k, v in ((old or {}).get("tiers_build") or {}).get("sources", {}).items()} if same else {}

    valid_dst = fg.warp_to_grid(valid.astype(np.float32), tr, crs, grid, resampling="nearest") > 0.5
    acc = {"rmax": np.zeros((grid["height"], grid["width"]), np.uint8), "q": np.zeros((grid["height"], grid["width"]), np.float32),
           "count": np.zeros((grid["height"], grid["width"]), np.uint8), "n": 0}
    written, dry, skipped, hourly, ratio, sources = [], [], [], {}, {}, {}
    os.makedirs(work_dir)
    try:
        # ---- pass 2: per hour
        wanted = set(build_hours)
        for hour in sorted(wanted | ({int(f[1:3]) for f in existing} if same else set())):
            hpath = os.path.join(out_dir, fg.hour_filename(hour))
            wpath = os.path.join(work_dir, fg.hour_filename(hour))
            t0 = time.time()
            kept = hour not in wanted                    # from an earlier run with the same grid
            if same and os.path.isfile(hpath) and hour in old_hourly and (
                    kept or (not force and old_src.get(hour) == src_keys.get(hour))):
                rgba = fg.read_rgba_gz(hpath, grid["width"], grid["height"])
                shutil.copy2(hpath, wpath)
                _accumulate(acc, rgba)
                written.append(hour)
                hourly[hour] = old_hourly[hour]
                if hour in old_ratio:
                    ratio[hour] = old_ratio[hour]
                if hour in old_src:
                    sources[hour] = old_src[hour]
                if not kept:
                    skipped.append(hour)
                    log("[%s] h%02d skipped (already written)" % (storm, hour))
                del rgba
                continue
            if kept:
                continue
            if wet_count[hour] == 0:
                dry.append(hour)
                sources[hour] = src_keys[hour]
                log("[%s] h%02d DRY: no wet cell inside the DEM, no file written" % (storm, hour))
                continue
            tier, _ = tier_hour_cached(app, storm, hour, dem, cache_dir)
            foot = (tier > 0) & valid
            tier = np.where(foot, tier, 0).astype(np.uint8)
            win = footprint_window(foot, pad=3)
            zw = z[win].astype(np.float64)
            if not np.isfinite(zw).all():                # no data cells (none in the local DEM): walls, flat fill
                zw = np.where(np.isfinite(zw), zw, np.nanmean(zw))
            tw = tier[win]
            w_tr = tr @ Affine.translation(win[1].start, win[0].start)
            t1 = time.time()
            surf = smooth_surface(tw, zw, sweeps=sweeps, sigma=sigma)
            t2 = time.time()
            ue, vn = velocity_fields(surf["depth"], surf["halo"], zw, manning[win], dem["dx"], w_tr, dead_band=dead_band)
            t3 = time.time()
            d_dst, u_dst, v_dst = warp_hour(grid, w_tr, crs, tw > 0, surf["depth"], ue, vn)
            t4 = time.time()
            rgba = fg.pack_rgba(d_dst, u_dst, v_dst, valid_dst)
            fg.write_rgba_gz(wpath, rgba)
            _accumulate(acc, rgba)
            written.append(hour)
            sources[hour] = src_keys[hour]
            hourly[hour] = fg.hourly_entry(hour, d_dst, u_dst, v_dst, cg)
            ratio[hour] = round(surf["volume_ratio"], 4)
            e = hourly[hour]
            flag = "  WARNING volume ratio outside +-%d %%" % int(VOLUME_GUARD * 100) if abs(surf["volume_ratio"] - 1) > VOLUME_GUARD else ""
            log("[%s] h%02d wet %8d cells %7.3f km2 max %.2f m %.2f m/s | surface %4.1fs vel %4.1fs warp %4.1fs total %5.1fs "
                "| vol x%.3f | peak %.0f MB%s" % (storm, hour, e["wet_cells"], e["wet_area_km2"], e["max_depth_m"], e["max_speed_ms"],
                                                 t2 - t1, t3 - t2, t4 - t3, time.time() - t0, surf["volume_ratio"], _peak_mb(), flag))
            del tier, foot, tw, zw, surf, ue, vn, d_dst, u_dst, v_dst, rgba
            gc.collect()

        # ---- flowpaths and meta
        for h in ((old or {}).get("tiers_build") or {}).get("dry_hours", []) if same else []:
            if h not in wanted and h not in dry:         # a dry hour of the earlier build that this run did not look at
                dry.append(h)
                if h in old_src:
                    sources[h] = old_src[h]
        dry.sort()
        written = sorted(set(written))
        if not written:
            raise RuntimeError("%s: every hour was dry, nothing to write" % storm)
        mean_q = acc["q"] / max(acc["n"], 1)
        rmax_m = fg.decode_depth(acc["rmax"])
        fp = fg.pack_flowpaths(rmax_m, mean_q, acc["count"], valid_dst)
        fg.write_rgba_gz(os.path.join(work_dir, "flowpaths.rgba.gz"), fp)
        del fp, acc
        run_summary = None
        rs_path = os.path.join(app, "BG FLOOD MAP VISTA SUPERCOMPUTER", "v2", storm, "run_summary.json")
        if os.path.isfile(rs_path):
            with open(rs_path, encoding="utf-8") as f:
                run_summary = json.load(f)
        n_year = int(re.match(r"(\d+)", storm).group(1))
        meta = fg.build_meta(storm, "%d-year storm" % n_year, "tiers", grid, written, [hourly[h] for h in written],
                             NOAA_DEPTH_IN.get(storm), run_summary, source_note(storm, dead_band))
        meta["tiers_build"] = {
            "params": params,
            "method": "smoothest water surface within the tier depth bands (projected SOR on the wet-cell graph), relief-widened "
                      "bands on steep walls, velocity-only halo, face-Manning velocity on the dead-banded slope, unit discharge capped at qmax",
            "tier_depth_m": dict(zip(TIERS, TIER_MID)),
            "dem": "dem_bg.tif (feet, converted to metres), water outside its footprint dropped",
            "volume_ratio": {str(h): ratio[h] for h in sorted(ratio)},
            "sources": {str(h): sources[h] for h in sorted(sources)},
            "dry_hours": dry,
        }
        fg.write_meta(os.path.join(work_dir, "meta.json"), meta)
        problems = fg.validate_storm_dir(work_dir)
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    if problems:
        log("[%s] NOT installed: validate_storm_dir found %d problems; the new build stays in %s, %s is unchanged" % (
            storm, len(problems), work_dir, out_dir))
        final_dir = work_dir
    else:
        try:
            _swap_dir(work_dir, out_dir, stale_dir)
        except OSError as e:
            raise RuntimeError("%s: the new build is complete and valid but could not replace %s (%s); stop whatever holds "
                               "its files open (the web server while it streams a grid) and swap %s into place by hand"
                               % (storm, out_dir, e, work_dir)) from e
        final_dir = out_dir
    secs = time.time() - t_start
    mb = sum(os.path.getsize(os.path.join(final_dir, f)) for f in os.listdir(final_dir)) / 1048576.0
    log("[%s] done in %.0f s: %d hour files + flowpaths, %.1f MB, validate_storm_dir: %s" % (
        storm, secs, len(written), mb, "OK" if not problems else "%d PROBLEMS" % len(problems)))
    for p in problems:
        log("[%s]   - %s" % (storm, p))
    return {"storm": storm, "dir": out_dir, "hours": written, "dry_hours": dry, "skipped": skipped, "seconds": secs,
            "problems": problems, "grid": grid, "hourly": [hourly[h] for h in written], "mb": mb}


# =====================================================================================================
# cross-check against the source polygons
# =====================================================================================================

def crosscheck_storm(app, storm, hours, dem, cache_dir=None, out_root=None, log=_log):
    """
    For each hour: polygon area inside the DEM (shoelace + clip), wet area of the DEM-grid raster, wet area of the output
    grid from meta/pixels, and the polygon area outside the DEM. Returns a list of dicts and logs the gaps.
    """
    out_dir = os.path.join(out_root or os.path.join(app, "layers", "flow"), storm)
    meta = _read_meta(os.path.join(out_dir, "meta.json"))
    if meta is None:
        raise RuntimeError("%s: no meta.json to check" % storm)
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "bgflow_tier_cache")
    rows = []
    for hour in hours:
        path = tier_file(app, storm, hour)
        src = source_area_km2(path, dem["bounds"])
        tier, _ = tier_hour_cached(app, storm, hour, dem, cache_dir)
        cells = int(((tier > 0) & dem["valid"]).sum())
        raster_km2 = cells * dem["dx"] ** 2 / 1e6
        e = next((x for x in meta["hourly"] if x["hour"] == hour), None)
        grid_km2 = e["wet_area_km2"] if e else 0.0
        inside = src["inside"]
        row = {"hour": hour, "polygons_km2": src["total"], "polygons_inside_dem_km2": inside,
               "outside_dem_km2": src["total"] - inside, "dem_raster_km2": raster_km2, "grid_km2": grid_km2,
               "gap_raster_pct": 100.0 * (raster_km2 / inside - 1.0) if inside > 0 else 0.0,
               "gap_grid_pct": 100.0 * (grid_km2 / inside - 1.0) if inside > 0 else 0.0}
        rows.append(row)
        log("[%s] h%02d polygons %.3f km2 (inside DEM %.3f, outside %.3f) | DEM raster %.3f (%+.1f %%) | output grid %.3f (%+.1f %%)" % (
            storm, hour, row["polygons_km2"], inside, row["outside_dem_km2"], raster_km2, row["gap_raster_pct"], grid_km2,
            row["gap_grid_pct"]))
    return rows


# =====================================================================================================
# command line
# =====================================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Preview flow grids from the depth-tier polygons (FLOW_FORMAT.md, source 'tiers').")
    ap.add_argument("--storms", nargs="+", default=["all"], help="5yr 10yr 25yr 50yr 100yr or all (default)")
    ap.add_argument("--app", default=APP_DEFAULT, help="folder the web server serves (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="rebuild hours that are already written")
    ap.add_argument("--hours", default="1-24", help="hours to build, e.g. 3-24 or 1,2,5-7 (default 1-24)")
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT, help="Gaussian polish of the solved surface in cells (default 0 = off)")
    ap.add_argument("--sweeps", type=int, default=SWEEPS_DEFAULT, help="SOR sweeps of the surface solve (0 = plain tier midpoints)")
    ap.add_argument("--dead-band", type=float, default=SLOPE_DEAD_BAND,
                    help="surface slope below which water counts as at rest (default %s, 0 = off)" % SLOPE_DEAD_BAND)
    ap.add_argument("--cell-m", type=float, default=None, help="output cell in mercator metres (default %s)" % CELL_M_DEFAULT)
    ap.add_argument("--out", default=None, help="folder that holds <storm>/ (default <app>/layers/flow)")
    ap.add_argument("--cache", default=None, help="folder for rasterised tier hours (default <temp>/bgflow_tier_cache)")
    ap.add_argument("--crosscheck", default=None, help="after the build compare polygon and grid areas for these hours, e.g. 8,13,20")
    args = ap.parse_args(argv)
    storms = list(STORMS) if args.storms == ["all"] else args.storms
    bad = [s for s in storms if s not in NOAA_DEPTH_IN]
    if bad:
        ap.error("unknown storm %s (known: %s)" % (", ".join(bad), ", ".join(STORMS)))
    try:
        hours = parse_hours(args.hours)
        check_hours = parse_hours(args.crosscheck) if args.crosscheck else []
    except ValueError as e:
        ap.error(str(e))
    app = args.app.replace("\\", "/")
    t0 = time.time()
    dem = load_dem(app)
    cache = args.cache or os.path.join(tempfile.gettempdir(), "bgflow_tier_cache")
    manning = load_manning(app, dem, cache)
    log0 = "DEM %dx%d cells of %.3f m, %.0f..%.0f m after feet->m; Manning: %.1f %% buildings, %.1f %% roads" % (
        dem["shape"][1], dem["shape"][0], dem["dx"], np.nanmin(dem["z"]), np.nanmax(dem["z"]),
        100.0 * float((manning == N_BUILDINGS).mean()), 100.0 * float((manning == N_ROADS).mean()))
    _log(log0)
    status = 0
    for storm in storms:
        res = build_storm(app, storm, hours, dem, manning, sigma=args.sigma, sweeps=args.sweeps, dead_band=args.dead_band,
                          force=args.force, cache_dir=cache, cell_m=args.cell_m, out_root=args.out)
        if res["problems"]:
            status = 1
        if check_hours:
            crosscheck_storm(app, storm, check_hours, dem, cache, args.out)
        gc.collect()
    _log("all done in %.0f s, peak memory %.0f MB" % (time.time() - t0, _peak_mb()))
    return status


if __name__ == "__main__":
    sys.exit(main())
