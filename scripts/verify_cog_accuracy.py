r"""
verify_cog_accuracy.py - independent accuracy audit of the county COGs against the SOLVER rasters (run on Vista, read-only, windowed point reads only).

    python verify_cog_accuracy.py --src runs/county_v2 --cog county_cog --storms "2yr 100yr" --out acc_county_cog.json [--pick-from county_cog] [--n-max 660] [--n-hour 60]

Per storm:
  * structure of every file (EPSG 3857, uint16, tiled 512, overviews, nodata 65535, same size/bounds across the 25 hourly files + max_depth.tif, file count, COG layout tag)
  * stratified random points (picked on a factor-8 overview of the COG, labelled afterwards from the full-resolution value): depth strata (cm: deep >=120,
    mid 30-119, low 10-29, trace 1-9, dry 0), 'edge' (a 7x7 window around the pixel contains nodata = within 3 px of the domain boundary), 'tileb' (pixel is the first/last
    column or row of a 512 px tile), 'outside' (COG nodata)
  * at each point the COG pixel centre (EPSG:3857) is transformed with PROJ to the source CRS (UTM 16N) and compared with the SOURCE raster (float32 m -> cm):
        nearest   = the single source cell that contains the position
        bilinear  = bilinear interpolation of the four surrounding cell centres (what the converter asked GDAL for), only when all four are valid
        hull      = COG value lies within [min,max] of those four cells (+-1 cm): a convex-combination property that must hold for any bilinear resample
  * errors are reported as max / mean / 99th percentile of the ABSOLUTE difference in cm, overall and per stratum
  * max_depth.tif consistency: COG max vs the maximum of the 25 hourly COGs at the same pixels, and the same on the source rasters
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import zlib

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

NODATA = 65535
TILE = 512
STRATA = ["deep(>=1.2m)", "mid(0.3-1.19m)", "low(0.1-0.29m)", "trace(0.01-0.09m)", "dry(0)"]


def label(v):
    if v == NODATA:
        return "nodata"
    if v >= 120:
        return STRATA[0]
    if v >= 30:
        return STRATA[1]
    if v >= 10:
        return STRATA[2]
    if v >= 1:
        return STRATA[3]
    return STRATA[4]


def structure(path):
    with rasterio.open(path) as d:
        t = d.tags(ns="IMAGE_STRUCTURE")
        return {"size": [d.width, d.height], "epsg": d.crs.to_epsg() if d.crs else None, "dtype": d.dtypes[0], "tiled": bool(d.profile.get("tiled")),
                "block": list(d.block_shapes[0]), "overviews": d.overviews(1), "nodata": d.nodata, "res": [d.res[0], d.res[1]],
                "bounds": [round(v, 4) for v in d.bounds], "compress": str(d.compression), "layout": t.get("LAYOUT"), "predictor": t.get("PREDICTOR"),
                "bytes": os.path.getsize(path)}


def pick_candidates(d, n, rng, spec):
    """(col,row) full-res pixel candidates from a factor-8 overview read; spec = {class: fraction}."""
    f = 8
    oh, ow = int(np.ceil(d.height / f)), int(np.ceil(d.width / f))
    ov = d.read(1, out_shape=(oh, ow), resampling=rasterio.enums.Resampling.nearest)
    valid = ov != NODATA
    pad = np.pad(valid, 1, constant_values=False)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    cls = {"deep": valid & (ov >= 100), "mid": valid & (ov >= 25) & (ov < 125), "low": valid & (ov >= 8) & (ov < 35), "trace": valid & (ov >= 1) & (ov < 12),
           "dry": valid & (ov == 0), "edge": valid & ~inner, "outside": ~valid & (np.pad(valid, 1)[:-2, 1:-1] | np.pad(valid, 1)[2:, 1:-1] | np.pad(valid, 1)[1:-1, :-2] | np.pad(valid, 1)[1:-1, 2:]),
           "tileb": valid}
    pts = []
    for k, frac in spec.items():
        arr = np.argwhere(cls[k])
        if not len(arr):
            continue
        take = max(1, int(round(n * frac)))
        for r, c in arr[rng.integers(0, len(arr), take)]:
            col = int(min(d.width - 1, c * f + rng.integers(0, f)))
            row = int(min(d.height - 1, r * f + rng.integers(0, f)))
            if k == "edge":
                col = int(min(d.width - 1, max(0, col + rng.integers(-4, 5))))
                row = int(min(d.height - 1, max(0, row + rng.integers(-4, 5))))
            if k == "tileb":                                           # snap to the first / last pixel column or row of a tile
                if rng.integers(0, 2):
                    col = int(min(d.width - 1, max(0, int(round(col / TILE)) * TILE + int(rng.integers(-1, 1)))))
                else:
                    row = int(min(d.height - 1, max(0, int(round(row / TILE)) * TILE + int(rng.integers(-1, 1)))))
            pts.append((k, col, row))
    return pts


SPEC_MAX = {"deep": 0.15, "mid": 0.15, "low": 0.14, "trace": 0.10, "dry": 0.08, "edge": 0.12, "outside": 0.04, "tileb": 0.22}


def stats(v):
    sv = np.asarray(v, float)
    v = np.abs(sv)
    if not len(v):
        return None
    return {"n": int(len(v)), "max": float(v.max()), "mean": round(float(v.mean()), 4), "p99": float(np.percentile(v, 99)), "within_1cm": round(float((v <= 1).mean()), 4),
            "signed_mean": round(float(sv.mean()), 4), "share_cog_lower": round(float((sv < -1).mean()), 4), "share_cog_higher": round(float((sv > 1).mean()), 4)}


def compare(cog_path, src_path, pts):
    """returns list of per-point records"""
    recs = []
    with rasterio.open(cog_path) as d, rasterio.open(src_path) as s:
        tr = Transformer.from_crs(d.crs, s.crs, always_xy=True)
        xs, ys = zip(*[d.xy(row, col) for _, col, row in pts])
        us, vs = tr.transform(list(xs), list(ys))
        order = sorted(range(len(pts)), key=lambda i: (pts[i][2] // TILE, pts[i][1] // TILE))
        out = [None] * len(pts)
        for i in order:
            kind, col, row = pts[i]
            w = d.read(1, window=Window(col - 3, row - 3, 7, 7), boundless=True, fill_value=NODATA)
            cv = int(w[3, 3])
            rec = {"k": kind, "col": col, "row": row, "cog": cv, "lab": label(cv), "edge": bool((w == NODATA).any() and cv != NODATA),
                   "tileb": bool(col % TILE in (0, TILE - 1) or row % TILE in (0, TILE - 1))}
            fc, fr = ~s.transform * (us[i], vs[i])
            ci, ri = int(np.floor(fc)), int(np.floor(fr))
            if not (0 <= ci < s.width and 0 <= ri < s.height):
                rec["src"] = "outside"
                out[i] = rec
                continue
            a = s.read(1, window=Window(ci - 1, ri - 1, 3, 3), boundless=True, fill_value=-9999.0).astype(float)
            bad = ~np.isfinite(a) | (a <= -9990)
            near = a[1, 1]
            if bad[1, 1]:
                rec["src"] = "nodata"
                out[i] = rec
                continue
            rec["nearest"] = int(round(max(near, 0) * 100))
            c0, r0 = int(np.floor(fc - 0.5)), int(np.floor(fr - 0.5))
            oc, orr = c0 - (ci - 1), r0 - (ri - 1)
            q = a[orr:orr + 2, oc:oc + 2]
            qb = bad[orr:orr + 2, oc:oc + 2]
            rec["src"] = "valid"
            if qb.any():
                rec["partial"] = True                                  # a neighbour is nodata: no exact bilinear expectation
                vq = q[~qb]
                rec["hull"] = bool(cv != NODATA and (np.floor(vq.min() * 100) - 1 <= cv <= np.ceil(max(vq.max(), 0) * 100) + 1))
            else:
                wx, wy = fc - 0.5 - c0, fr - 0.5 - r0
                bil = q[0, 0] * (1 - wx) * (1 - wy) + q[0, 1] * wx * (1 - wy) + q[1, 0] * (1 - wx) * wy + q[1, 1] * wx * wy
                rec["bilinear"] = int(round(max(float(bil), 0) * 100))
                rec["hull"] = bool(cv != NODATA and (np.floor(max(q.min(), 0) * 100) - 1 <= cv <= np.ceil(max(q.max(), 0) * 100) + 1))
                rec["range4"] = int(round((q.max() - q.min()) * 100))
            out[i] = rec
    return out


def summarize(rows):
    cv = [r for r in rows if r["cog"] != NODATA]
    both = [r for r in cv if r.get("nearest") is not None]
    bil = [r for r in cv if r.get("bilinear") is not None]
    res = {"points": len(rows),
           "nearest_err": stats([r["cog"] - r["nearest"] for r in both]),
           "bilinear_err": stats([r["cog"] - r["bilinear"] for r in bil]),
           "hull_ok": {"n": sum(1 for r in cv if "hull" in r), "violations": sum(1 for r in cv if "hull" in r and not r["hull"])},
           "cog_nodata_src_valid": sum(1 for r in rows if r["cog"] == NODATA and r.get("src") == "valid"),
           "cog_valid_src_nodata": sum(1 for r in rows if r["cog"] != NODATA and r.get("src") in ("nodata", "outside")),
           "both_nodata": sum(1 for r in rows if r["cog"] == NODATA and r.get("src") in ("nodata", "outside")),
           "partial_neighbour_points": sum(1 for r in rows if r.get("partial")),
           "by_depth_stratum": {}, "by_flag": {}}
    for lab in STRATA:
        sub = [r for r in both if r["lab"] == lab]
        subb = [r for r in bil if r["lab"] == lab]
        res["by_depth_stratum"][lab] = {"n_points": sum(1 for r in rows if r["lab"] == lab), "nearest_err": stats([r["cog"] - r["nearest"] for r in sub]),
                                         "bilinear_err": stats([r["cog"] - r["bilinear"] for r in subb]),
                                         "median_local_range_cm": float(np.median([r["range4"] for r in subb])) if subb else None}
    for flag in ("edge", "tileb"):
        sub = [r for r in both if r[flag]]
        subb = [r for r in bil if r[flag]]
        res["by_flag"][flag] = {"n_points": sum(1 for r in rows if r[flag]), "nearest_err": stats([r["cog"] - r["nearest"] for r in sub]), "bilinear_err": stats([r["cog"] - r["bilinear"] for r in subb])}
    worst = sorted(bil, key=lambda r: -abs(r["cog"] - r["bilinear"]))[:3]
    res["worst_bilinear"] = [{k: r.get(k) for k in ("k", "col", "row", "cog", "bilinear", "nearest", "range4")} for r in worst]
    wn = sorted(both, key=lambda r: -abs(r["cog"] - r["nearest"]))[:3]
    res["worst_nearest"] = [{k: r.get(k) for k in ("k", "col", "row", "cog", "bilinear", "nearest", "range4")} for r in wn]
    return res


def maxconsistency(storm_cog, storm_src, hours, rows, rng, nmax=300):
    """COG max vs max of hourly COGs, and source max vs max of hourly source rasters, at the same pixels (valid points only)."""
    rows = [r for r in rows if r["cog"] != NODATA and r.get("src") == "valid"]
    if len(rows) > nmax:
        rows = [rows[i] for i in rng.choice(len(rows), nmax, replace=False)]
    out = {"n": len(rows)}
    dcs = [rasterio.open(os.path.join(storm_cog, h)) for h in hours]
    dss = [rasterio.open(os.path.join(storm_src, h)) for h in hours]
    ms = rasterio.open(os.path.join(storm_src, "max_depth.tif"))
    mc = rasterio.open(os.path.join(storm_cog, "max_depth.tif"))
    tr = Transformer.from_crs(mc.crs, ms.crs, always_xy=True)
    dcog, dsrc, viol = [], [], 0
    for r in sorted(rows, key=lambda r: (r["row"] // TILE, r["col"] // TILE)):
        col, row = r["col"], r["row"]
        hmax = max(int(dc.read(1, window=Window(col, row, 1, 1))[0, 0]) for dc in dcs)
        hmax = 0 if hmax == NODATA else hmax
        diff = r["cog"] - (0 if hmax == NODATA else hmax)
        dcog.append(diff)
        x, y = mc.xy(row, col)
        u, v = tr.transform(x, y)
        fc, fr = ~ms.transform * (u, v)
        ci, ri = int(np.floor(fc)), int(np.floor(fr))
        if 0 <= ci < ms.width and 0 <= ri < ms.height:
            sm = float(ms.read(1, window=Window(ci, ri, 1, 1))[0, 0])
            sh = max(float(ds.read(1, window=Window(ci, ri, 1, 1))[0, 0]) for ds in dss)
            if sm > -9990:
                dsrc.append(round((sm - max(sh, 0)) * 100))
    for x in dcs + dss + [ms, mc]:
        x.close()
    a = np.array(dcog)
    out["cog_max_minus_max_of_hourly_cog_cm"] = {"min": int(a.min()), "mean": round(float(a.mean()), 3), "p99": float(np.percentile(a, 99)), "max": int(a.max()),
                                                   "n_below_minus1": int((a < -1).sum()), "n_above_1": int((a > 1).sum())}
    if dsrc:
        b = np.array(dsrc)
        out["source_max_minus_max_of_hourly_source_cm"] = {"n": len(b), "min": int(b.min()), "mean": round(float(b.mean()), 3), "p99": float(np.percentile(b, 99)), "max": int(b.max()),
                                                           "n_negative_below_minus1": int((b < -1).sum()), "n_gt_1cm": int((b > 1).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cog", required=True)
    ap.add_argument("--pick-from", default=None, help="COG folder whose pixels define the sample (so two COG sets are compared on identical points)")
    ap.add_argument("--storms", required=True)
    ap.add_argument("--n-max", type=int, default=660)
    ap.add_argument("--n-hour", type=int, default=60)
    ap.add_argument("--out", default="acc.json")
    a = ap.parse_args()
    pick = a.pick_from or a.cog
    report = {"cog": a.cog, "src": a.src, "pick_from": pick}
    for storm in a.storms.split():
        t0 = time.time()
        sc, ss, sp = os.path.join(a.cog, storm), os.path.join(a.src, storm), os.path.join(pick, storm)
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(sc, "depth_*s.tif")))
        allf = os.listdir(sc)
        st = {"hourly_files": len(names), "max_depth": os.path.isfile(os.path.join(sc, "max_depth.tif")), "n_files_in_folder": len(allf),
              "extra_files": sorted(f for f in allf if not (f.startswith("depth_") or f in ("max_depth.tif", "cog_meta.json")))}
        structs = {n: structure(os.path.join(sc, n)) for n in names + ["max_depth.tif"]}
        ref = structs["max_depth.tif"]
        key = lambda s: (s["size"], s["epsg"], s["dtype"], s["tiled"], tuple(s["block"]), tuple(s["overviews"]), s["nodata"], tuple(s["bounds"]))
        st["structure_ref"] = ref
        st["all_files_same_geometry"] = all(key(v) == key(ref) for v in structs.values())
        st["all_files_ok"] = all(v["epsg"] == 3857 and v["dtype"] == "uint16" and v["tiled"] and v["block"] == [512, 512] and v["overviews"] and v["nodata"] == NODATA for v in structs.values())
        st["layouts"] = sorted({str(v["layout"]) for v in structs.values()})
        st["bytes_total"] = sum(v["bytes"] for v in structs.values())
        rng = np.random.default_rng(5000 + zlib.crc32(storm.encode()) % 1000)
        # max_depth
        with rasterio.open(os.path.join(sp, "max_depth.tif")) as d:
            pts = pick_candidates(d, a.n_max, rng, SPEC_MAX)
        rows = compare(os.path.join(sc, "max_depth.tif"), os.path.join(ss, "max_depth.tif"), pts)
        st["max_depth"] = summarize(rows)
        # hourly
        hrows = []
        per_hour = {}
        spec_h = dict(SPEC_MAX)
        for n in names:
            with rasterio.open(os.path.join(sp, n)) as d:
                hp = pick_candidates(d, a.n_hour, rng, spec_h)
            r = compare(os.path.join(sc, n), os.path.join(ss, n), hp)
            hrows += r
            sm = summarize(r)
            per_hour[n] = {"points": sm["points"], "nearest_max": (sm["nearest_err"] or {}).get("max"), "bilinear_max": (sm["bilinear_err"] or {}).get("max"),
                           "hull_violations": sm["hull_ok"]["violations"]}
        st["hourly_all"] = summarize(hrows)
        st["per_hour"] = per_hour
        st["max_vs_hourly"] = maxconsistency(sc, ss, names, rows, rng)
        st["seconds"] = round(time.time() - t0, 1)
        report[storm] = st
        b = st["max_depth"]["bilinear_err"] or {}
        nn = st["max_depth"]["nearest_err"] or {}
        hb, hn = st["hourly_all"]["bilinear_err"] or {}, st["hourly_all"]["nearest_err"] or {}
        print("%-5s files=%d ok=%s same=%s | MAX pts=%d bil max=%s mean=%s p99=%s | nearest max=%s mean=%s p99=%s | HOURLY pts=%d bil max=%s p99=%s nearest max=%s p99=%s | hull viol max/hourly %d/%d | %ss" % (
            storm, st["hourly_files"], st["all_files_ok"], st["all_files_same_geometry"], st["max_depth"]["points"], b.get("max"), b.get("mean"), b.get("p99"), nn.get("max"), nn.get("mean"), nn.get("p99"),
            st["hourly_all"]["points"], hb.get("max"), hb.get("p99"), hn.get("max"), hn.get("p99"), st["max_depth"]["hull_ok"]["violations"], st["hourly_all"]["hull_ok"]["violations"], st["seconds"]), flush=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
