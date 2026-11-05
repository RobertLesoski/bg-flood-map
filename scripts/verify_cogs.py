r"""
verify_cogs.py - check COGs made by make_cogs.py against their SOURCE rasters (run on Vista, where both live).

    python verify_cogs.py --src runs/county_v2 --cog county_cog --storms "2yr 100yr ev4" --files "max 12 24" --n 240 --out verify_report.json

For each storm and file it checks the structure (EPSG:3857, uint16, tiled 512, overviews, nodata 65535, bounds inside / covering the source) and then picks
>= n random points of five kinds, taken from the COG itself: deep (>= 100 cm), shallow (1..99 cm), dry (0), points within 3 px of the edge of the modelled
area, and points just outside the modelled area. At each point the COG pixel is read (nearest, exactly what the viewer reads) and compared with the SOURCE
raster at the same ground position:
  expected_bilinear = bilinear interpolation of the source cells (centres) at the pixel centre in UTM, clamped at 0, x100, rounded  <- what WarpedVRT(bilinear) computes
  expected_nearest  = the single source cell containing that position, x100, rounded                                          <- the "cell value" a user would expect
The exact-method error (COG - expected_bilinear) should be <= 1 cm (rounding of the stored value plus GDAL's fixed-point weights). The nearest error is the
real resampling difference: bounded by the local depth gradient over one source cell. Both are reported (max, mean abs, share within 1 cm).
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


def hour_file(folder, h):
    best = None
    for p in glob.glob(os.path.join(folder, "depth_*s.tif")):
        sec = int(re.search(r"depth_(\d+)s", p).group(1))
        if round(sec / 3600.0) == h and (best is None or abs(sec - h * 3600) < best[0]):
            best = (abs(sec - h * 3600), p)
    return best[1] if best else None


def structure(path, src_path):
    out = {}
    with rasterio.open(path) as d, rasterio.open(src_path) as s:
        out["size"] = [d.width, d.height]
        out["epsg"] = d.crs.to_epsg()
        out["dtype"] = d.dtypes[0]
        out["tiled"] = bool(d.profile.get("tiled"))
        out["block"] = list(d.block_shapes[0])
        out["overviews"] = d.overviews(1)
        out["nodata"] = d.nodata
        out["res"] = d.transform.a
        out["bytes"] = os.path.getsize(path)
        tr = Transformer.from_crs(s.crs, "EPSG:3857", always_xy=True)
        sb = s.bounds
        xs, ys = tr.transform([sb.left, sb.right, sb.left, sb.right], [sb.bottom, sb.bottom, sb.top, sb.top])
        out["bounds_cog"] = list(d.bounds)
        out["bounds_src_in_3857"] = [min(xs), min(ys), max(xs), max(ys)]
        slack = 2 * d.transform.a
        out["covers_source"] = bool(d.bounds.left <= min(xs) + slack and d.bounds.right >= max(xs) - slack and d.bounds.bottom <= min(ys) + slack and d.bounds.top >= max(ys) - slack)
        out["ok"] = bool(out["epsg"] == 3857 and out["dtype"] == "uint16" and out["tiled"] and out["block"] == [512, 512] and out["overviews"] and out["nodata"] == NODATA)
    return out


def pick_points(d, n, rng):
    """(col,row) full-resolution pixels of the COG: deep / shallow / dry / edge / outside."""
    f = 8
    oh, ow = int(np.ceil(d.height / f)), int(np.ceil(d.width / f))
    ov = d.read(1, out_shape=(oh, ow), resampling=rasterio.enums.Resampling.nearest)
    valid = ov != NODATA
    cls = {"deep": np.argwhere(valid & (ov >= 100)), "shallow": np.argwhere(valid & (ov >= 1) & (ov < 100)), "dry": np.argwhere(valid & (ov == 0)),
           "outside": np.argwhere(~valid)}
    # edge of the modelled area: valid overview cells with an invalid neighbour
    pad = np.pad(valid, 1, constant_values=False)
    nb = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    cls["edge"] = np.argwhere(valid & ~nb)
    pts = []
    for k, arr in cls.items():
        if not len(arr):
            continue
        take = n // 5 + (n % 5 if k == "dry" else 0)
        idx = rng.integers(0, len(arr), take)
        for r, c in arr[idx]:
            col = int(min(d.width - 1, c * f + rng.integers(0, f)))
            row = int(min(d.height - 1, r * f + rng.integers(0, f)))
            if k == "edge":                                   # also probe single full-res pixels right at the boundary
                col = int(min(d.width - 1, max(0, c * f + rng.integers(-3, f + 3))))
            pts.append((k, col, row))
    return pts


def compare(cog_path, src_path, n, seed):
    rng = np.random.default_rng(seed)
    res = {"n": 0}
    rows = []
    with rasterio.open(cog_path) as d, rasterio.open(src_path) as s:
        tr = Transformer.from_crs("EPSG:3857", s.crs, always_xy=True)
        snd = s.nodata if s.nodata is not None else -9999.0
        full = d.read(1) if d.width * d.height < 2.5e8 else None
        t0 = time.time()
        pts = pick_points(d, n, rng)
        for kind, col, row in pts:
            cv = int(full[row, col]) if full is not None else int(d.read(1, window=Window(col, row, 1, 1))[0, 0])
            x, y = d.xy(row, col)                              # pixel centre, EPSG:3857
            e, nn = tr.transform(x, y)
            fc, fr = ~s.transform * (e, nn)                    # fractional source pixel (corner-based): cell i spans [i, i+1)
            ci, ri = int(np.floor(fc)), int(np.floor(fr))
            rec = {"kind": kind, "col": col, "row": row, "cog": cv}
            if not (0 <= ci < s.width and 0 <= ri < s.height):
                rec.update(src="outside", nearest=None, bilinear=None)
                rows.append(rec)
                continue
            # 2x2 neighbourhood around the position (cell centres at i + 0.5)
            c0, r0 = int(np.floor(fc - 0.5)), int(np.floor(fr - 0.5))
            win = Window(max(c0, 0), max(r0, 0), 2, 2)
            a = s.read(1, window=win, boundless=False)
            nearest = float(s.read(1, window=Window(ci, ri, 1, 1))[0, 0])
            rec["nearest_m"] = nearest
            if nearest <= -9990 or (snd is not None and nearest == snd):
                rec.update(src="nodata", nearest=None, bilinear=None)
                rows.append(rec)
                continue
            rec["nearest"] = int(round(max(nearest, 0) * 100))
            if c0 < 0 or r0 < 0 or c0 + 1 >= s.width or r0 + 1 >= s.height or a.shape != (2, 2) or (a <= -9990).any():
                rec.update(src="valid-edge", bilinear=None)       # a neighbour is nodata/outside: bilinear is not exact there
                rows.append(rec)
                continue
            wx, wy = fc - 0.5 - c0, fr - 0.5 - r0
            bil = (a[0, 0] * (1 - wx) * (1 - wy) + a[0, 1] * wx * (1 - wy) + a[1, 0] * (1 - wx) * wy + a[1, 1] * wx * wy)
            rec["bilinear"] = int(round(max(float(bil), 0) * 100))
            rec["src"] = "valid"
            rows.append(rec)
    def stats(vals):
        v = np.array(vals, float)
        if not len(v):
            return None
        return {"n": int(len(v)), "max_abs": float(np.abs(v).max()), "mean_abs": float(np.abs(v).mean()), "within_1cm": float((np.abs(v) <= 1).mean()), "within_2cm": float((np.abs(v) <= 2).mean())}
    valid = [r for r in rows if r["src"] == "valid" and r["cog"] != NODATA]
    out = {"points": len(rows), "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in ("deep", "shallow", "dry", "edge", "outside")}}
    out["bilinear_err"] = stats([r["cog"] - r["bilinear"] for r in valid])
    out["nearest_err"] = stats([r["cog"] - r["nearest"] for r in rows if r["src"] in ("valid", "valid-edge") and r["cog"] != NODATA])
    out["cog_nodata_but_src_valid"] = sum(1 for r in rows if r["cog"] == NODATA and r["src"] in ("valid", "valid-edge"))
    out["cog_value_but_src_nodata_or_outside"] = sum(1 for r in rows if r["cog"] != NODATA and r["src"] in ("nodata", "outside"))
    out["src_outside_or_nodata_and_cog_nodata"] = sum(1 for r in rows if r["cog"] == NODATA and r["src"] in ("nodata", "outside"))
    out["valid_edge_points"] = sum(1 for r in rows if r["src"] == "valid-edge")
    big = sorted(valid, key=lambda r: -abs(r["cog"] - r["bilinear"]))[:3]
    out["worst_bilinear"] = [{k: r[k] for k in ("kind", "col", "row", "cog", "bilinear", "nearest")} for r in big]
    bign = sorted([r for r in rows if r["src"] in ("valid", "valid-edge") and r["cog"] != NODATA], key=lambda r: -abs(r["cog"] - r["nearest"]))[:3]
    out["worst_nearest"] = [{k: r[k] for k in ("kind", "col", "row", "cog", "bilinear", "nearest")} for r in bign]
    out["classes_in_dry_points_cog_nonzero"] = sum(1 for r in rows if r["kind"] == "dry" and r["cog"] not in (0, NODATA))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cog", required=True)
    ap.add_argument("--storms", required=True)
    ap.add_argument("--files", default="max 12 24")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default="verify_report.json")
    a = ap.parse_args()
    report = {}
    for storm in a.storms.split():
        report[storm] = {}
        for f in a.files.split():
            t0 = time.time()
            if f == "max":
                name, sp, cp = "max_depth.tif", os.path.join(a.src, storm, "max_depth.tif"), os.path.join(a.cog, storm, "max_depth.tif")
            else:
                sp = hour_file(os.path.join(a.src, storm), int(f))
                cp = hour_file(os.path.join(a.cog, storm), int(f))
                name = os.path.basename(cp) if cp else "hour%s" % f
            if not (sp and cp and os.path.isfile(cp)):
                report[storm][f] = {"error": "missing file"}
                continue
            r = {"file": name, "structure": structure(cp, sp)}
            r["compare"] = compare(cp, sp, a.n, 1000 + zlib.crc32((storm + f).encode()) % 1000)
            r["seconds"] = round(time.time() - t0, 1)
            report[storm][f] = r
            b = r["compare"]["bilinear_err"] or {}
            nn = r["compare"]["nearest_err"] or {}
            print("%-5s %-5s struct_ok=%s covers=%s pts=%d | bilinear max=%s mean=%.3f in1cm=%.3f | nearest max=%s mean=%.3f | nodata-mismatch %d/%d | %ss" % (
                storm, f, r["structure"]["ok"], r["structure"]["covers_source"], r["compare"]["points"], b.get("max_abs"), b.get("mean_abs", -1), b.get("within_1cm", -1),
                nn.get("max_abs"), nn.get("mean_abs", -1), r["compare"]["cog_nodata_but_src_valid"], r["compare"]["cog_value_but_src_nodata_or_outside"], r["seconds"]), flush=True)
            json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
