r"""
crosscheck_cogs.py - compare the downloaded real-depth COGs (layers/cog/county/<storm>) with the OTHER depth products built from the same solver rasters:
  1. the viewer flow grids   layers/flow_county/<storm>/hNN.rgba.gz (R = depth, sqrt-coded 8 bit, 25 m mercator cells = 4x4 block average of the 5 m solver cells)
  2. the address index       layers/county/address_index.json.gz   (per building: max over the footprint of the flow-grid depth, over hours 4..24; cm)
  3. the infrastructure file layers/county/infrastructure.geojson   (d_<storm>: same rule for the 98 facilities; cm)

    python scripts/crosscheck_cogs.py --app "<web app folder>" --storms "2yr 100yr ev4" --n 300 --out crosscheck_report.json

Only windowed reads are used (no full-raster loads). What differences are EXPECTED (not errors):
  * a grid cell is the MEAN of ~4x4 solver cells and is stored 8 bit sqrt coded (step 8*(2k+1)/255^2 m: 0.25 cm near 0, 1.2 cm at 1 m, 4.8 cm at 3 m);
    the COG keeps the 5 m cells, so a COG footprint MAX is >= the grid value almost always;
  * the index is the max over the hourly SNAPSHOTS 4..24 of the grid, the COG max_depth.tif is the maximum over every solver time step;
  * the footprint rule differs slightly (grid: cells touched; COG: all-touched 6.25 m pixels).
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flowgrid as fg  # noqa: E402

NODATA = 65535
R = 6378137.0


def merc(lng, lat):
    return R * np.radians(lng), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def hour_file(folder, h):
    import glob, re
    best = None
    for p in glob.glob(os.path.join(folder, "depth_*s.tif")):
        sec = int(re.search(r"depth_(\d+)s", p).group(1))
        if round(sec / 3600.0) == h and (best is None or abs(sec - h * 3600) < best[0]):
            best = (abs(sec - h * 3600), p)
    return best[1]


def stats(d):
    d = np.asarray(d, float)
    if not len(d):
        return None
    a = np.abs(d)
    return {"n": int(len(d)), "mean_diff_cm": round(float(d.mean()), 3), "mean_abs_cm": round(float(a.mean()), 3), "median_abs_cm": float(np.median(a)), "p99_abs_cm": round(float(np.percentile(a, 99)), 2),
            "max_abs_cm": float(a.max()), "within_5cm": round(float((a <= 5).mean()), 4), "share_cog_ge_grid": round(float((d >= 0).mean()), 4)}


def win_for(d, xa, xb, yb, yt):
    """pixel window (clipped) covering the mercator box"""
    ca, cb = int(np.floor((xa - d.transform.c) / d.transform.a)), int(np.ceil((xb - d.transform.c) / d.transform.a))
    ra, rb = int(np.floor((d.transform.f - yt) / d.transform.a)), int(np.ceil((d.transform.f - yb) / d.transform.a))
    ca, ra = max(ca, 0), max(ra, 0)
    cb, rb = min(cb, d.width), min(rb, d.height)
    if cb <= ca or rb <= ra:
        return None
    return Window(ca, ra, cb - ca, rb - ra)


def grid_vs_cog(app, storm, hour, n, rng):
    meta = json.load(open(os.path.join(app, "layers", "flow_county", storm, "meta.json")))
    w, h, cell = meta["width"], meta["height"], meta["cell_m"]
    x0, y0, x1, y1 = meta["bounds_merc"]
    raw = gzip.open(os.path.join(app, "layers", "flow_county", storm, "h%02d.rgba.gz" % hour)).read()
    rgba = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
    depth = fg.decode_depth(rgba[:, :, 0]).astype(float)                     # m
    cogp = hour_file(os.path.join(app, "layers", "cog", "county", storm), hour)
    wet = np.argwhere(depth > 0.02)
    dry = np.argwhere(depth <= 0.0)
    nw = n // 2 if len(wet) else 0
    pick = ([tuple(p) for p in wet[rng.integers(0, len(wet), nw)]] if nw else []) + [tuple(p) for p in dry[rng.integers(0, len(dry), n - nw)]]
    dm, dx = [], []
    with rasterio.open(cogp) as d:
        for r, c in pick:
            xa, xb = x0 + c * cell, x0 + (c + 1) * cell
            yt, yb = y1 - r * cell, y1 - (r + 1) * cell
            win = win_for(d, xa, xb, yb, yt)
            if win is None:
                continue
            a = d.read(1, window=win)
            xs = d.transform.c + (np.arange(win.col_off, win.col_off + win.width) + 0.5) * d.transform.a
            ys = d.transform.f - (np.arange(win.row_off, win.row_off + win.height) + 0.5) * d.transform.a
            inx = (xs >= xa) & (xs < xb)
            iny = (ys <= yt) & (ys > yb)
            v = a[np.ix_(iny, inx)].ravel()
            v = v[v != NODATA]
            if v.size < 8:
                continue
            g = depth[r, c] * 100
            dm.append(v.mean() - g)
            dx.append(v.max() - g)
    return {"storm": storm, "hour": hour, "cells": len(dm), "wet_cells_in_grid": int(len(wet)), "cog_mean_minus_grid": stats(dm), "cog_max_minus_grid": stats(dx)}


def building_depth(d, geom):
    """max COG depth (cm) over the footprint (all-touched), or None when outside the model"""
    if geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        polys = [geom["coordinates"]]
    gm = {"type": "MultiPolygon", "coordinates": [[[list(merc(x, y)) for x, y in ring] for ring in poly] for poly in polys]}
    allx = [p[0] for poly in gm["coordinates"] for ring in poly for p in ring]
    ally = [p[1] for poly in gm["coordinates"] for ring in poly for p in ring]
    win = win_for(d, min(allx), max(allx), min(ally), max(ally))
    if win is None:
        return None
    a = d.read(1, window=win)
    m = ~geometry_mask([gm], out_shape=a.shape, transform=d.window_transform(win), all_touched=True)
    v = a[m]
    v = v[v != NODATA]
    return int(v.max()) if v.size else None


def buildings_vs_cog(app, storm, n, rng, ai, byid):
    flood = ai["flood"].get(storm, [])
    cm = {}
    r = 0
    for i in range(0, len(flood) - 4, 5):
        r += flood[i]
        cm[r] = flood[i + 1]
    kinds = np.array(ai["kind"]); bids = np.array(ai["bid"])
    bidx = [i for i in range(ai["n"]) if kinds[i] == 0 and int(bids[i]) in byid]
    wet = [i for i in bidx if i in cm]
    dry = [i for i in bidx if i not in cm]
    nw = min(n // 2, len(wet)) if wet else 0
    pick = ([wet[j] for j in rng.choice(len(wet), nw, replace=False)] if nw else []) + [dry[j] for j in rng.choice(len(dry), n - nw, replace=False)]
    cogp = os.path.join(app, "layers", "cog", "county", storm, "max_depth.tif")
    rows = []
    with rasterio.open(cogp) as d:
        for i in pick:
            c = building_depth(d, byid[int(bids[i])]["geometry"])
            if c is None:
                continue
            rows.append((c, cm.get(i, 0)))
    rows = np.array(rows)
    diffs = rows[:, 0] - rows[:, 1]
    both_wet = rows[(rows[:, 0] >= 10) & (rows[:, 1] >= 10)]
    out = {"storm": storm, "buildings": int(len(rows)), "wet_in_index_total": len(wet), "sampled_wet_in_index": int(nw), "cog_max_minus_index": stats(diffs),
           "index_ge10_cog_lt10": int(((rows[:, 1] >= 10) & (rows[:, 0] < 10)).sum()), "index_lt10_cog_ge10": int(((rows[:, 1] < 10) & (rows[:, 0] >= 10)).sum()),
           "both_flooded": int(len(both_wet)), "flooded10_agreement_share": round(float(((rows[:, 0] >= 10) == (rows[:, 1] >= 10)).mean()), 4),
           "cog_below_index_by_more_than_5cm": int((diffs < -5).sum())}
    if len(both_wet):
        out["both_flooded_ratio_median"] = round(float(np.median(both_wet[:, 0] / both_wet[:, 1])), 3)
        out["both_flooded_diff"] = stats(both_wet[:, 0] - both_wet[:, 1])
    return out


def facility_groups(app):
    """the 98 facility groups exactly as build_infrastructure.py builds them (OSM outlines from the cached tagged pass; merged parts of one facility)"""
    import geodata as gd
    import osm_pbf
    import build_infrastructure as bi
    dom = gd.get_domain(app, "county")
    res = osm_pbf.tagged(app, dom.bbox, pbf=None, log=lambda *a: None)
    pois = [p for p in res["pois"] if gd.in_box(p["lng"], p["lat"], dom.bbox)]
    groups = bi.dedupe_pois(pois)
    return groups, bi


def geom_max(d, g):
    """max COG depth (cm) over a lng/lat shapely Point (its pixel) or polygon (all-touched pixels); None when outside the model"""
    from shapely.geometry import mapping
    if g.geom_type == "Point":
        X, Y = merc(g.x, g.y)
        ci, ri = int((X - d.transform.c) // d.transform.a), int((d.transform.f - Y) // d.transform.a)
        if not (0 <= ci < d.width and 0 <= ri < d.height):
            return None
        v = int(d.read(1, window=Window(ci, ri, 1, 1))[0, 0])
        return None if v == NODATA else v
    gm = mapping(g)
    polys = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
    mm = {"type": "MultiPolygon", "coordinates": [[[list(merc(x, y)) for x, y in ring] for ring in poly] for poly in polys]}
    allx = [p[0] for poly in mm["coordinates"] for ring in poly for p in ring]
    ally = [p[1] for poly in mm["coordinates"] for ring in poly for p in ring]
    win = win_for(d, min(allx), max(allx), min(ally), max(ally))
    if win is None:
        return None
    a = d.read(1, window=win)
    m = ~geometry_mask([mm], out_shape=a.shape, transform=d.window_transform(win), all_touched=True)
    v = a[m]
    v = v[v != NODATA]
    return int(v.max()) if v.size else None


def infra_vs_cog(app, storm, fc, groups, bi):
    """per facility: COG max over the SAME footprint the index used (all member outlines / points) vs d_<storm> of infrastructure.geojson"""
    cogp = os.path.join(app, "layers", "cog", "county", storm, "max_depth.tif")
    byname = {}
    for f in fc["features"]:
        byname[(f["properties"].get("osm_id"), f["properties"]["name"])] = f
    rows = []
    with rasterio.open(cogp) as d:
        for g in groups:
            main = g["main"]
            f = byname.get((main.get("osm_id", ""), main.get("name", "")))
            if f is None:
                continue
            vals = [geom_max(d, geom) for geom in bi.poi_geoms(g["members"])]
            vals = [v for v in vals if v is not None]
            rows.append((max(vals) if vals else None, f["properties"].get("d_" + storm, 0), f["properties"]["name"], f["properties"]["type"], len(g["members"])))
    ok = [r for r in rows if r[0] is not None]
    cm_ = np.array([r[0] for r in ok]); idx = np.array([r[1] for r in ok])
    worst = sorted(ok, key=lambda r: -abs(r[0] - r[1]))[:3]
    return {"storm": storm, "facilities_matched": len(rows), "outside_model": len(rows) - len(ok), "cog_footprint_max_minus_index": stats(cm_ - idx), "index_wet(>0)": int((idx > 0).sum()),
            "cog_wet(>0)": int((cm_ > 0).sum()), "index_ge10": int((idx >= 10).sum()), "cog_ge10": int((cm_ >= 10).sum()), "flooded10_agree_share": round(float(((cm_ >= 10) == (idx >= 10)).mean()), 4),
            "worst": [{"name": r[2], "type": r[3], "cog_cm": r[0], "index_cm": r[1], "members": r[4]} for r in worst]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--storms", default="100yr ev4 2yr")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="crosscheck_report.json")
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    ai = json.load(gzip.open(os.path.join(a.app, "layers", "county", "address_index.json.gz")))
    bl = json.load(gzip.open(os.path.join(a.app, "layers", "county", "buildings.geojson.gz")))
    byid = {int(f["properties"]["osm_id"]): f for f in bl["features"] if f["properties"].get("osm_id") is not None}
    fc = json.load(open(os.path.join(a.app, "layers", "county", "infrastructure.geojson")))
    groups, bi = facility_groups(a.app)
    rep = {"flow_grid": [], "address_index": [], "infrastructure": []}
    for s in a.storms.split():
        for h in (12, 24):
            r = grid_vs_cog(a.app, s, h, a.n, rng); rep["flow_grid"].append(r); print(json.dumps(r), flush=True)
        r = buildings_vs_cog(a.app, s, a.n, rng, ai, byid); rep["address_index"].append(r); print(json.dumps(r), flush=True)
        r = infra_vs_cog(a.app, s, fc, groups, bi); rep["infrastructure"].append(r); print(json.dumps(r), flush=True)
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
