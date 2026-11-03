r"""
make_cogs.py - depth rasters -> Cloud-Optimized GeoTIFFs a browser can range-read with geotiff.js (REAL-depth click-to-query).

    python make_cogs.py --src $SCRATCH/bgflood/runs/county_v2/100yr --dst $SCRATCH/bgflood/county_cog/100yr [--res-m 6.25] [--force]

Input : max_depth.tif (optional; computed from the hourly files when absent) and depth_<SSSSSS>s.tif (float32 metres, UTM 16N, nodata -9999).
Output: same file names, EPSG:3857, uint16 DEPTH IN CENTIMETRES (0..65534), nodata 65535 (outside the model domain), 512x512 tiles, DEFLATE with
        predictor 2, overviews (average); plus cog_meta.json (units, bounds, hours, sizes). Hour N = depth_{N*3600:06d}s.tif.
EPSG:3857 so the browser needs no projection library: x = R*lng_rad, y = R*ln(tan(pi/4+lat/2)), pixel = ((x-x0)/res, (y0-y)/res).
Default 6.25 mercator m = about 5 m on the ground at 37 N (the solver's 5 m cells). Query the FULL-resolution level (image 0); overviews are averages.
Resumable: a finished output is skipped (source size+mtime recorded in cog_meta.json) unless --force. Uses the GDAL COG driver when present,
else a tiled GTiff with overviews. Only rasterio + numpy. Check a result:  python make_cogs.py --verify <file.tif>
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rcopy
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window

NODATA = 65535
BLOCK = 512


def convert(src_path, dst_path, res_m=6.25):
    """float32 metres (any CRS) -> COG uint16 cm in EPSG:3857. Returns (width, height)."""
    with rasterio.open(src_path) as s:
        tr, w, h = calculate_default_transform(s.crs, "EPSG:3857", s.width, s.height, *s.bounds, resolution=res_m)
        vrt = WarpedVRT(s, crs="EPSG:3857", transform=tr, width=w, height=h, resampling=Resampling.bilinear,
                        src_nodata=s.nodata if s.nodata is not None else -9999.0, nodata=-9999.0)
        fd, tmp = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(os.path.abspath(dst_path)))
        os.close(fd)
        prof = dict(driver="GTiff", width=w, height=h, count=1, dtype="uint16", crs="EPSG:3857", transform=tr, nodata=NODATA,
                    tiled=True, blockxsize=BLOCK, blockysize=BLOCK, compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
        with rasterio.open(tmp, "w", **prof) as d:
            for r0 in range(0, h, BLOCK):
                for c0 in range(0, w, BLOCK * 4):
                    win = Window(c0, r0, min(BLOCK * 4, w - c0), min(BLOCK, h - r0))
                    a = vrt.read(1, window=win)
                    out = np.where(a <= -9990, NODATA, np.clip(np.rint(np.maximum(a, 0) * 100), 0, 65534)).astype(np.uint16)
                    d.write(out, 1, window=win)
        vrt.close()
    try:
        write_cog(tmp, dst_path)
    finally:
        os.unlink(tmp)
    return w, h


def write_cog(tmp, dst):
    try:
        rcopy(tmp, dst, driver="COG", COMPRESS="DEFLATE", PREDICTOR="2", BLOCKSIZE=BLOCK, OVERVIEW_RESAMPLING="AVERAGE",
              NUM_THREADS="ALL_CPUS", BIGTIFF="IF_SAFER")
        return "COG"
    except Exception:                                    # no COG driver: tiled GTiff + overviews
        with rasterio.open(tmp, "r+") as d:
            f = [x for x in (2, 4, 8, 16, 32, 64) if max(d.width, d.height) // x >= 256]
            if f:
                d.build_overviews(f, Resampling.average)
        rcopy(tmp, dst, driver="GTiff", tiled=True, blockxsize=BLOCK, blockysize=BLOCK, compress="deflate", predictor=2,
              copy_src_overviews=True, BIGTIFF="IF_SAFER")
        return "GTiff+overviews"


def verify(path):
    with rasterio.open(path) as d:
        ov = d.overviews(1)
        ok = d.crs.to_epsg() == 3857 and d.dtypes[0] == "uint16" and d.profile.get("tiled") and d.nodata == NODATA
        print("%s: %dx%d %s epsg=%s tiled=%s block=%s overviews=%s %s" % (
            os.path.basename(path), d.width, d.height, d.dtypes[0], d.crs.to_epsg(), d.profile.get("tiled"), d.block_shapes[0], ov,
            "OK" if ok else "PROBLEM"))
        return bool(ok)


def run(src, dst, res_m=6.25, force=False, log=print):
    os.makedirs(dst, exist_ok=True)
    mp = os.path.join(dst, "cog_meta.json")
    meta = json.load(open(mp)) if os.path.isfile(mp) else {"files": {}}
    hours = sorted(glob.glob(os.path.join(src, "depth_*s.tif")))
    mx = os.path.join(src, "max_depth.tif")
    jobs = ([("max_depth.tif", mx)] if os.path.isfile(mx) else []) + [(os.path.basename(p), p) for p in hours]
    t0 = time.perf_counter()
    for name, p in jobs:
        out = os.path.join(dst, name)
        sig = [os.path.getsize(p), int(os.path.getmtime(p)), res_m]
        if not force and os.path.isfile(out) and meta["files"].get(name, {}).get("src") == sig:
            log("  skip %s (done)" % name)
            continue
        w, h = convert(p, out, res_m)
        meta["files"][name] = {"src": sig, "width": w, "height": h, "bytes": os.path.getsize(out)}
        json.dump(meta, open(mp, "w"))
        log("  %s -> %.1f MB (%dx%d)" % (name, os.path.getsize(out) / 1e6, w, h))
    if not os.path.isfile(mx) and hours and "max_depth.tif" not in meta["files"]:
        log("  max_depth.tif missing in the source: building it from the hourly rasters")
        acc = None
        for p in hours:
            with rasterio.open(p) as s:
                a = s.read(1)
            acc = a if acc is None else np.maximum(acc, a)
        with rasterio.open(hours[0]) as s:
            prof = s.profile
        tmpsrc = os.path.join(dst, "_max_src.tif")
        with rasterio.open(tmpsrc, "w", **prof) as o:
            o.write(acc, 1)
        convert(tmpsrc, os.path.join(dst, "max_depth.tif"), res_m)
        os.unlink(tmpsrc)
        meta["files"]["max_depth.tif"] = {"src": "computed", "bytes": os.path.getsize(os.path.join(dst, "max_depth.tif"))}
    first = next((os.path.join(dst, n) for n in meta["files"] if os.path.isfile(os.path.join(dst, n))), None)
    if first:
        with rasterio.open(first) as d:
            b = d.bounds
            meta.update({"crs": "EPSG:3857", "units": "cm", "dtype": "uint16", "nodata": NODATA, "res_m": res_m,
                         "bounds_merc": [b.left, b.bottom, b.right, b.top], "width": d.width, "height": d.height,
                         "hours": {str(int(re.search(r"depth_(\d+)s", n).group(1)) // 3600): n
                                   for n in meta["files"] if n.startswith("depth_")},
                         "note": "depth_cm = value (65535 = outside the model domain); read image 0 (full resolution) for queries"})
    json.dump(meta, open(mp, "w"), indent=1)
    log("  total %.1f MB in %.0fs" % (sum(f.get("bytes", 0) for f in meta["files"].values()) / 1e6, time.perf_counter() - t0))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="depth rasters -> COGs (see the module docstring)")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--res-m", type=float, default=6.25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.verify) else 1)
    if not (a.src and a.dst):
        ap.error("--src and --dst are required")
    run(a.src, a.dst, a.res_m, a.force)
