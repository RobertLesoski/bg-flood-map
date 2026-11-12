"""Compare a model flood-depth raster with the NASA OPERA DSWx-S1 water map.

Inputs (never modified):
  --opera-dir   folder with the OPERA WTR tiles (T16SEF, T16SEG; UTM 16N, 30 m, class codes per the OPERA product spec)
  --boundary    county boundary GeoJSON (FIPS 21227 polygon)
  --model       model max-depth GeoTIFF (EPSG:3857 or EPSG:32616; uint16 centimetres or float metres)

Outputs (new files only, under --out):
  <name>.json   inputs, library versions, thresholds, both OPERA water definitions, status
  <name>.csv    one row per threshold and water definition

Method
  * OPERA cells inside the county polygon (UTM 16N), overlap between tiles counted once.
  * Each cell is sampled at its centre from the model (nearest cell). Cells with model nodata or
    OPERA fill/layover (255, 251) are excluded.
  * Observed wet = OPERA open water (class 1) [definition A], or classes 1 and 3 [definition B].
  * Model wet = model depth >= threshold, for thresholds 0.05 to 1.0 m.
  * Scores: hit rate (POD), false-alarm ratio (FAR), critical success index (CSI).

Limits (stated in every output): OPERA is one overpass at 2025-04-06 23:40 UTC; the model is a 24 h
rain-window maximum; radar misses water under canopy and in urban areas, so observed wet is a lower bound.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import sys

import numpy as np
from PIL import Image, ImageDraw

R = 6378137.0                       # WGS84 semi-major axis (m)
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)
EP2 = E2 / (1.0 - E2)
K0 = 0.9996
ZONE_LON0 = -87.0                   # UTM zone 16N central meridian
OPERA_FILL, OPERA_LAYOVER, OPERA_WATER, OPERA_IVEG = 255, 251, 1, 3
THRESHOLDS_M = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
CSV_HEADER = ("definition,threshold_m,n_valid,hits,misses,false_alarms,correct_negatives,"
              "hit_rate,false_alarm_ratio,csi,observed_wet_km2,model_wet_km2")


# ---------------------------------------------------------------- projections
def utm_forward(lat, lon, lon0=ZONE_LON0):
    """Lat/lon (deg) to UTM easting/northing (m), WGS84, zone with central meridian lon0 (Snyder 1987)."""
    la, lo = np.radians(lat), np.radians(lon)
    l0 = math.radians(lon0)
    N = R / np.sqrt(1 - E2 * np.sin(la) ** 2)
    T = np.tan(la) ** 2
    C = EP2 * np.cos(la) ** 2
    A = (lo - l0) * np.cos(la)
    M = R * ((1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256) * la
             - (3 * E2 / 8 + 3 * E2 ** 2 / 32 + 45 * E2 ** 3 / 1024) * np.sin(2 * la)
             + (15 * E2 ** 2 / 256 + 45 * E2 ** 3 / 1024) * np.sin(4 * la)
             - (35 * E2 ** 3 / 3072) * np.sin(6 * la))
    x = K0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * EP2) * A ** 5 / 120) + 500000.0
    y = K0 * (M + N * np.tan(la) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                   + (61 - 58 * T + T * T + 600 * C - 330 * EP2) * A ** 6 / 720))
    return x, y


def utm_inverse(x, y, lon0=ZONE_LON0):
    """UTM easting/northing (m) to lat/lon (deg). Inverse of utm_forward (Snyder 1987)."""
    xx = np.asarray(x, float) - 500000.0
    M = np.asarray(y, float) / K0
    mu = M / (R * (1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    N1 = R / np.sqrt(1 - E2 * np.sin(phi1) ** 2)
    T1 = np.tan(phi1) ** 2
    C1 = EP2 * np.cos(phi1) ** 2
    R1 = R * (1 - E2) / (1 - E2 * np.sin(phi1) ** 2) ** 1.5
    D = xx / (N1 * K0)
    lat = phi1 - (N1 * np.tan(phi1) / R1) * (D ** 2 / 2
                                            - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * EP2) * D ** 4 / 24
                                            + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * EP2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = math.radians(lon0) + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
                                + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * EP2 + 24 * T1 ** 2) * D ** 5 / 120) / np.cos(phi1)
    return np.degrees(lat), np.degrees(lon)


def merc_forward(lat, lon):
    """EPSG:3857 (spherical Web Mercator) from lat/lon in degrees."""
    x = R * np.radians(np.asarray(lon, float))
    y = R * np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2))
    return x, y


# ---------------------------------------------------------------- rasters
GEOKEY_PROJECTED = 3072


def read_geotiff(path):
    """Read a north-up GeoTIFF with PIL. Returns dict(array, x0, y0, sx, sy, epsg)."""
    im = Image.open(path)
    tags = im.tag_v2
    if 33550 not in tags or 33922 not in tags:
        raise ValueError(f"{path}: no ModelPixelScale/ModelTiepoint tags; cannot georeference")
    sx, sy = float(tags[33550][0]), float(tags[33550][1])
    tp = tags[33922]
    x0, y0 = float(tp[3]), float(tp[4])
    epsg = None
    keys = tags.get(34735, ())
    for i in range(4, len(keys) - 3, 4):
        if keys[i] == GEOKEY_PROJECTED and keys[i + 1] == 0:
            epsg = int(keys[i + 3])
    return {"array": np.array(im), "x0": x0, "y0": y0, "sx": sx, "sy": sy, "epsg": epsg,
            "nodata": tags.get(42113), "path": os.path.abspath(path)}


def sample_nearest(arr, x0, y0, sx, sy, x, y):
    """Nearest-cell value at map coordinates (x, y); returns (values, inside_mask). Values are float."""
    h, w = arr.shape
    col = np.floor((x - x0) / sx).astype(np.int64)
    row = np.floor((y0 - y) / sy).astype(np.int64)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.full(np.shape(x), np.nan)
    out[inside] = arr[row[inside], col[inside]].astype(float)
    return out, inside


def points_in_polygon(px, py, xs, ys):
    """Even-odd (crossing-number) test, exact for points not on an edge. xs, ys: polygon vertices."""
    inside = np.zeros(px.shape, bool)
    n = len(xs)
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]
        if y1 == y2:
            continue
        cross = (y1 > py) != (y2 > py)
        xint = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        inside ^= cross & (px < xint)
    return inside


def county_mask_in_tile(poly_xy, x0, y0, sx, sy, shape):
    """Boolean mask of tile cells whose centre lies inside the polygon (UTM metres). Exact; no edge bias."""
    h, w = shape
    xs = np.array([p[0] for p in poly_xy], float)
    ys = np.array([p[1] for p in poly_xy], float)
    mask = np.zeros((h, w), bool)
    c0 = max(0, int(np.floor((xs.min() - x0) / sx)))
    c1 = min(w, int(np.ceil((xs.max() - x0) / sx)) + 1)
    r0 = max(0, int(np.floor((y0 - ys.max()) / sy)))
    r1 = min(h, int(np.ceil((y0 - ys.min()) / sy)) + 1)
    if c1 <= c0 or r1 <= r0:
        return mask
    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    px = x0 + (cc + 0.5) * sx
    py = y0 - (rr + 0.5) * sy
    mask[r0:r1, c0:c1] = points_in_polygon(px.ravel(), py.ravel(), xs, ys).reshape(px.shape)
    return mask


def boundary_ring_lonlat(path):
    with open(path, encoding="utf-8-sig") as fh:
        data = json.load(fh)
    feat = data["features"][0] if "features" in data else data
    geom = feat.get("geometry", feat)
    coords = geom["coordinates"]
    ring = coords[0] if geom["type"] == "Polygon" else coords[0][0]
    return [(float(lon), float(lat)) for lon, lat in ring]


# ---------------------------------------------------------------- scoring
def contingency(obs_wet, mod_wet, valid):
    """Counts (hits, misses, false_alarms, correct_negatives) over valid cells."""
    o, m = obs_wet[valid], mod_wet[valid]
    hits = int(np.sum(o & m))
    misses = int(np.sum(o & ~m))
    fas = int(np.sum(~o & m))
    cns = int(np.sum(~o & ~m))
    return hits, misses, fas, cns


def scores(hits, misses, fas):
    """POD (hit rate), FAR (false-alarm ratio) and CSI. NaN where the denominator is zero."""
    pod = hits / (hits + misses) if hits + misses else float("nan")
    far = fas / (hits + fas) if hits + fas else float("nan")
    csi = hits / (hits + misses + fas) if hits + misses + fas else float("nan")
    return pod, far, csi


def sweep(obs_wet, depth, valid, cell_km2, thresholds=THRESHOLDS_M):
    """Scores for each threshold. depth is metres (NaN where unknown)."""
    rows = []
    for t in thresholds:
        mod_wet = np.nan_to_num(depth, nan=-1.0) >= t
        h, mi, fa, cn = contingency(obs_wet, mod_wet, valid)
        pod, far, csi = scores(h, mi, fa)
        rows.append({"threshold_m": t, "n_valid": int(valid.sum()), "hits": h, "misses": mi,
                     "false_alarms": fa, "correct_negatives": cn,
                     "hit_rate": pod, "false_alarm_ratio": far, "csi": csi,
                     "observed_wet_km2": float((obs_wet & valid).sum()) * cell_km2,
                     "model_wet_km2": float((mod_wet & valid).sum()) * cell_km2})
    return rows


# ---------------------------------------------------------------- pipeline
def collect_opera_cells(tile_paths, poly_xy):
    """Return arrays over OPERA cells inside the county: easting, northing, class value, tile name.
    A tile's cells are skipped where they fall inside an earlier tile's extent (no double counting)."""
    tiles = [read_geotiff(p) for p in tile_paths]
    tiles.sort(key=lambda t: -t["y0"])          # northern tile first
    xs, ys, cls, names = [], [], [], []
    seen = []                                    # extents already counted: (xmin, xmax, ymin, ymax)
    for t in tiles:
        arr = t["array"]
        h, w = arr.shape
        xmin, xmax = t["x0"], t["x0"] + w * t["sx"]
        ymax, ymin = t["y0"], t["y0"] - h * t["sy"]
        mask = county_mask_in_tile(poly_xy, t["x0"], t["y0"], t["sx"], t["sy"], (h, w))
        rr, cc = np.nonzero(mask)
        ex = t["x0"] + (cc + 0.5) * t["sx"]
        ny = t["y0"] - (rr + 0.5) * t["sy"]
        keep = np.ones(ex.shape, bool)
        for (sxmin, sxmax, symin, symax) in seen:
            keep &= ~((ex >= sxmin) & (ex <= sxmax) & (ny >= symin) & (ny <= symax))
        xs.append(ex[keep]); ys.append(ny[keep])
        cls.append(arr[rr[keep], cc[keep]].astype(int))
        names.append(np.full(int(keep.sum()), os.path.basename(t["path"])))
        seen.append((xmin, xmax, ymin, ymax))
    if not xs:
        raise ValueError("no OPERA cells inside the county polygon")
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(cls), np.concatenate(names))


def model_depth_m(model, ex, ny, scale, nodata):
    """Model depth in metres at OPERA cell centres. Reprojects centres to the model's CRS."""
    lat, lon = utm_inverse(ex, ny)
    if model["epsg"] == 3857:
        mx, my = merc_forward(lat, lon)
    elif model["epsg"] == 32616:
        mx, my = utm_forward(lat, lon)
    else:
        raise ValueError(f"unsupported model CRS EPSG:{model['epsg']} (expected 3857 or 32616)")
    raw, inside = sample_nearest(model["array"], model["x0"], model["y0"], model["sx"], model["sy"], mx, my)
    depth = raw * scale
    if nodata is not None:
        depth[raw == nodata] = np.nan
    depth[~inside] = np.nan
    return depth


def run(args):
    poly = boundary_ring_lonlat(args.boundary)
    poly_xy = [utm_forward(np.array([la]), np.array([lo]))  for lo, la in poly]
    poly_xy = [(float(x[0]), float(y[0])) for x, y in poly_xy]
    tile_paths = sorted(os.path.join(args.opera_dir, f) for f in os.listdir(args.opera_dir)
                        if f.lower().endswith(".tif"))
    if not tile_paths:
        raise SystemExit(f"no .tif files in {args.opera_dir}")
    ex, ny, cls, tname = collect_opera_cells(tile_paths, poly_xy)
    model = read_geotiff(args.model)
    depth = model_depth_m(model, ex, ny, args.model_scale, args.model_nodata)
    cell_km2 = (30.0 * 30.0) / 1e6
    ok_opera = (cls != OPERA_FILL) & (cls != OPERA_LAYOVER)
    valid = ok_opera & np.isfinite(depth)
    water_a = cls == OPERA_WATER
    water_b = (cls == OPERA_WATER) | (cls == OPERA_IVEG)
    res = {
        "definition_A_open_water_only": sweep(water_a, depth, valid, cell_km2),
        "definition_B_open_water_plus_inundated_veg": sweep(water_b, depth, valid, cell_km2),
    }
    counts = {
        "opera_cells_in_county": int(cls.size),
        "opera_fill_or_layover": int((~ok_opera).sum()),
        "model_nodata_or_outside": int((ok_opera & ~np.isfinite(depth)).sum()),
        "valid_cells": int(valid.sum()),
        "valid_area_km2": float(valid.sum()) * cell_km2,
        "inundated_vegetation_km2_in_valid": float((valid & (cls == OPERA_IVEG)).sum()) * cell_km2,
    }
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"opera_vs_model_{stamp}")
    meta = {
        "script": os.path.relpath(os.path.abspath(__file__)),
        "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "platform": platform.platform(),
        "inputs": {"opera_tiles": [os.path.basename(p) for p in tile_paths],
                   "boundary": os.path.basename(args.boundary),
                   "model": model["path"], "model_epsg": model["epsg"],
                   "model_cell_m": [model["sx"], model["sy"]],
                   "model_scale_to_m": args.model_scale, "model_nodata": args.model_nodata},
        "counts": counts,
        "thresholds_m": list(THRESHOLDS_M),
        "results": res,
        "status": "computed",
        "limits": [
            "OPERA is a single overpass, 2025-04-06 23:40 UTC; the model is a 24 h rain-window maximum.",
            "Radar underestimates water under vegetation canopy and in urban areas; observed wet is a lower bound.",
            "Definition B adds inundated vegetation, which is an estimate, not open water.",
            "Only presence/absence is tested here; no depths are observed in these inputs.",
        ],
    }
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(base + ".csv", "w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER + "\n")
        for name, rows in res.items():
            for r in rows:
                fh.write(",".join(str(v) for v in [
                    name, r["threshold_m"], r["n_valid"], r["hits"], r["misses"], r["false_alarms"],
                    r["correct_negatives"], r["hit_rate"], r["false_alarm_ratio"], r["csi"],
                    r["observed_wet_km2"], r["model_wet_km2"]]) + "\n")
    print("wrote", base + ".json", "and .csv")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opera-dir", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-scale", type=float, default=0.01,
                    help="multiply model values by this to get metres (0.01 for uint16 cm; 1.0 for float m)")
    ap.add_argument("--model-nodata", type=float, default=65535,
                    help="model value treated as no data (default 65535, uint16 nodata)")
    ap.add_argument("--out", default=os.path.join("verification", "derived"))
    args = ap.parse_args(argv)
    if not os.path.exists(args.model):
        meta = {"status": "blocked", "reason": f"model raster not found: {args.model}",
                "needed": "ev4 max-depth GeoTIFF (layers/cog/county/ev4/max_depth.tif or equivalent)"}
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "opera_vs_model_BLOCKED.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print("BLOCKED:", meta["reason"], "->", path)
        return 2
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
