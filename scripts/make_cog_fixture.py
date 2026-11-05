r"""
make_cog_fixture.py - synthetic Cloud-Optimized GeoTIFFs with KNOWN values, for the COG reader tests (pytest + the browser harness).

    python scripts/make_cog_fixture.py --out "<web app folder>/cog/test/fixture"

Writes
  a_direct.tif     EPSG:3857 uint16 COG written straight from an analytic pattern (1300 x 1100 px: edge tiles are partial, three overview
                   levels), with a dry block (0), a nodata block (65535) and values that make predictor 2 wrap (the pattern jumps by > 32768).
  b_pipeline.tif   made by make_cogs.convert from a UTM 16N float32 raster whose depth is a LINEAR ramp (m) of the UTM easting/northing, so the
                   bilinear resampling reproduces the analytic value exactly (to the 1 cm rounding), plus a nodata strip and a dry block.
  expected.json    Python-side truth: for hundreds of points the value of every level read back with rasterio (GDAL's decoder, independent of
                   the JavaScript reader), lng/lat -> mercator from rasterio.warp.transform (PROJ), analytic expectations for b_pipeline,
                   and footprint maxima (rasterio.features.geometry_mask, all_touched False and True).
Deterministic (fixed seeds), small (< 1 MB).
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_cogs  # noqa: E402

W, H = 1300, 1100
RES = 6.25
X0, Y0 = -9640000.0, 4450000.0                     # top-left of a_direct (mercator), near Warren County
NODATA = 65535


def pattern_a():
    c = np.arange(W, dtype=np.int64)[None, :]
    r = np.arange(H, dtype=np.int64)[:, None]
    v = (c * 7 + r * 13 + (c * r) % 101 + (c // 64) * 9000) % 60000
    v = v.astype(np.uint16)
    v[200:260, 300:420] = 0                              # dry block
    v[700:760, 900:1010] = NODATA                        # no-data block
    v[0, :] = 5; v[:, 0] = 7; v[H - 1, :] = 11; v[:, W - 1] = 13   # edge marks
    return v


def write_a(path):
    import tempfile
    tr = from_origin(X0, Y0, RES, RES)
    fd, tmp = tempfile.mkstemp(suffix=".tif"); os.close(fd)
    with rasterio.open(tmp, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint16", crs="EPSG:3857", transform=tr, nodata=NODATA) as d:
        d.write(pattern_a(), 1)
    try:
        how = make_cogs.write_cog(tmp, path)
    finally:
        os.unlink(tmp)
    return how


# b_pipeline: UTM 16N float32, 5 m cells; depth(m) = 0.2 + 0.0004 * (E - E0) + 0.0002 * (N0 - N)  (0.2 .. ~2.3 m), strip on the left = -9999
UE0, UN0, UN_W, UN_H = 560000.0, 4090000.0, 700, 600


def depth_b_utm(e, n):
    return 0.2 + 0.0004 * (e - UE0) + 0.0002 * (UN0 - n)


def write_b(path, src_tif):
    cols = UE0 + (np.arange(UN_W) + 0.5) * 5.0
    rows = UN0 - (np.arange(UN_H) + 0.5) * 5.0
    a = depth_b_utm(cols[None, :], rows[:, None]).astype(np.float32)
    a[:, :6] = -9999
    a[100:140, 200:260] = 0.0
    with rasterio.open(src_tif, "w", driver="GTiff", width=UN_W, height=UN_H, count=1, dtype="float32", crs="EPSG:32616",
                       transform=Affine(5, 0, UE0, 0, -5, UN0), nodata=-9999.0) as d:
        d.write(a, 1)
    make_cogs.convert(src_tif, path, res_m=RES)
    return a


def level_arrays(path):
    """Every level of a COG as a full array, via rasterio/GDAL."""
    out = []
    with rasterio.open(path) as d:
        out.append(d.read(1))
        for f in d.overviews(1):
            out.append(d.read(1, out_shape=(int(np.ceil(d.height / f)), int(np.ceil(d.width / f))), resampling=rasterio.enums.Resampling.nearest))
    return out


def native_overviews(path):
    """Overview IFDs read natively (GDAL 'OVERVIEW_LEVEL' open option), the exact bytes stored in the file."""
    out = []
    with rasterio.open(path) as d:
        n = len(d.overviews(1))
        out.append(d.read(1))
    for i in range(n):
        with rasterio.open(path, OVERVIEW_LEVEL=i) as d:
            out.append(d.read(1))
    return out


def points_for(path, rng, n_random, special):
    """Pixel/mercator sample points for one file: random + tile boundaries + edges + outside."""
    with rasterio.open(path) as d:
        tr, w, h = d.transform, d.width, d.height
    pts = []
    for _ in range(n_random):
        pts.append((rng.uniform(0, w), rng.uniform(0, h)))
    for c in special:
        for r in special:
            if c < w and r < h:
                pts.append((c + 0.5, r + 0.5))
    for c, r in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (0.0001, 0.0001), (w - 0.0001, h - 0.0001)]:
        pts.append((c + (0.5 if c == int(c) else 0), r + (0.5 if r == int(r) else 0)))
    # exact pixel edges (floor rule) and points outside the file
    pts += [(512.0, 512.0), (1024.0, 512.0), (511.999999, 511.999999), (w / 2, h / 2)]
    pts += [(-0.5, 5.5), (w + 0.5, 5.5), (5.5, -0.5), (5.5, h + 0.5), (-100, -100)]
    out = []
    for c, r in pts:
        x, y = tr * (c, r)
        out.append((float(x), float(y)))
    return out


def merc_to_lnglat(x, y):
    lng, lat = warp_transform("EPSG:3857", "EPSG:4326", [x], [y])
    return lng[0], lat[0]


def build(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261104)
    pa, pb = os.path.join(out, "a_direct.tif"), os.path.join(out, "b_pipeline.tif")
    how = write_a(pa)
    src_b = os.path.join(out, "_b_src_utm.tif")
    ab = write_b(pb, src_b)
    exp = {"driver": how, "res": RES, "nodata": NODATA, "files": {}}
    special = [0, 1, 255, 256, 510, 511, 512, 513, 1023, 1024, 1025, 1100, 1299]
    for name, path in (("a_direct.tif", pa), ("b_pipeline.tif", pb)):
        with rasterio.open(path) as d:
            info = {"width": d.width, "height": d.height, "origin": [d.transform.c, d.transform.f], "res": d.transform.a,
                    "overviews": d.overviews(1), "blockshapes": d.block_shapes[0], "epsg": d.crs.to_epsg()}
        levels = native_overviews(path)
        info["level_sizes"] = [[a.shape[1], a.shape[0]] for a in levels]
        pts = points_for(path, rng, 150, special)
        recs = []
        with rasterio.open(path) as d:
            tr = d.transform
        for (x, y) in pts:
            lng, lat = merc_to_lnglat(x, y)
            rec = {"x": x, "y": y, "lng": lng, "lat": lat, "v": []}
            for li, arr in enumerate(levels):
                sx, sy = info["width"] / arr.shape[1], info["height"] / arr.shape[0]
                c = int(np.floor((x - tr.c) / (tr.a * sx))); r = int(np.floor((tr.f - y) / (tr.a * sy)))
                rec["v"].append(int(arr[r, c]) if 0 <= c < arr.shape[1] and 0 <= r < arr.shape[0] else None)
            recs.append(rec)
        info["points"] = recs
        exp["files"][name] = info
    # b_pipeline analytic expectation (cm) for points well inside the raster: depth = ramp of the UTM position of the point
    utm_pts = []
    pb_info = exp["files"]["b_pipeline.tif"]
    for rec in pb_info["points"]:
        e, n = warp_transform("EPSG:3857", "EPSG:32616", [rec["x"]], [rec["y"]])
        e, n = e[0], n[0]
        rec["utm"] = [e, n]
        inside = UE0 + 6 * 5 + 20 < e < UE0 + UN_W * 5 - 20 and UN0 - UN_H * 5 + 20 < n < UN0 - 20
        in_dry = (UE0 + 200 * 5 - 15 < e < UE0 + 260 * 5 + 15) and (UN0 - 140 * 5 - 15 < n < UN0 - 100 * 5 + 15)
        rec["analytic_cm"] = int(round(max(depth_b_utm(e, n), 0) * 100)) if inside and not in_dry else None
    # footprints (polygons in lng/lat) on a_direct: max over pixels, centre-inside and all-touched, from rasterio
    with rasterio.open(pa) as d:
        a0, tra = d.read(1), d.transform
    polys = []
    for _ in range(60):
        cx, cy = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        wd, ht = rng.uniform(0.6, 9), rng.uniform(0.6, 9)
        ang = rng.uniform(0, np.pi)
        ring = [(cx + dx * np.cos(ang) - dy * np.sin(ang), cy + dx * np.sin(ang) + dy * np.cos(ang)) for dx, dy in ((-wd, -ht), (wd, -ht), (wd, ht), (-wd, ht))]
        merc = [tra * p for p in ring]
        ll = [merc_to_lnglat(x, y) for x, y in merc]
        ll.append(ll[0])
        geom = {"type": "Polygon", "coordinates": [[list(p) for p in ll]]}
        res = {}
        for key, at in (("center", False), ("touched", True)):
            m = ~geometry_mask([{"type": "Polygon", "coordinates": [[list(p) for p in merc] + [list(merc[0])]]}], out_shape=a0.shape, transform=tra, all_touched=at)
            vals = a0[m]
            valid = vals[vals != NODATA]
            res[key] = {"n": int(vals.size), "max": int(valid.max()) if valid.size else None}
        polys.append({"ring": geom["coordinates"][0], "center": res["center"], "touched": res["touched"]})
    # a polygon fully in the nodata block, one fully outside the file
    for ring_px in ([(905, 705), (1000, 705), (1000, 750), (905, 750)], [(-60, -60), (-30, -60), (-30, -30), (-60, -30)]):
        merc = [tra * p for p in ring_px]
        ll = [merc_to_lnglat(x, y) for x, y in merc]; ll.append(ll[0])
        polys.append({"ring": [list(p) for p in ll], "center": None, "touched": None, "special": True})
    exp["footprints"] = polys
    # lng/lat -> mercator truth (PROJ) at 200 random places incl. high latitudes
    ll_pts = []
    for _ in range(200):
        lng, lat = rng.uniform(-180, 180), rng.uniform(-85, 85)
        x, y = warp_transform("EPSG:4326", "EPSG:3857", [lng], [lat])
        ll_pts.append({"lng": lng, "lat": lat, "x": x[0], "y": y[0]})
    for lng, lat in ((-86.4, 36.97), (-86.7, 36.65), (-86.13, 37.2)):
        x, y = warp_transform("EPSG:4326", "EPSG:3857", [lng], [lat]); ll_pts.append({"lng": lng, "lat": lat, "x": x[0], "y": y[0]})
    exp["lnglat"] = ll_pts
    # XYZ tile colours: expected tier index grid for two tiles of a_direct at an exact-pixel zoom (z=20 -> 0.149 m px, so use z=17: 1.19 m px)
    os.unlink(src_b)
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump(exp, f)
    return exp


def build_viewer_demo(out_base, storms=("25yr",)):
    """SYNTHETIC DEMO rasters (not real model output!) laid out like the real thing: <out_base>/county/<storm>/{max_depth,depth_*s}.tif + cog_meta.json,
    so the viewer can be tested end to end with  viewer.html?cogbase=./cog/test/demo  (never copy this into layers/cog)."""
    import tempfile
    cx_m, cy_m = warp_transform("EPSG:4326", "EPSG:32616", [-86.4385], [36.9930])
    e0, n0 = cx_m[0] - 3000.0, cy_m[0] + 2500.0
    nx, ny = 1200, 1000
    ee = e0 + (np.arange(nx) + 0.5) * 5.0
    nn = n0 - (np.arange(ny) + 0.5) * 5.0
    E, N = np.meshgrid(ee, nn)
    blobs = [(0.30, 0.40, 350, 1.8), (0.62, 0.55, 500, 0.9), (0.50, 0.80, 250, 0.45), (0.85, 0.25, 180, 0.25)]    # x frac, y frac, radius m, peak m
    base = np.zeros((ny, nx), np.float32)
    for fx, fy, rad, peak in blobs:
        d2 = (E - (e0 + fx * nx * 5.0)) ** 2 + (N - (n0 - fy * ny * 5.0)) ** 2
        base += (peak * np.exp(-d2 / (2 * (rad / 2.2) ** 2))).astype(np.float32)
    for storm in storms:
        src = tempfile.mkdtemp(prefix="demo_src_")
        mx = np.zeros_like(base)
        for h in range(0, 25):
            a = base * min(1.0, max(0.0, (h - 2) / 14.0)) if h < 16 else base * (1.0 - 0.012 * (h - 16))
            a = a.astype(np.float32)
            a[a < 0.004] = 0
            a[:, :8] = -9999                                            # a no-data strip at the west edge
            mx = np.maximum(mx, a)
            sec = h * 3600 + {0: 15, 1: 4, 2: 0, 3: -2}.get(h, 0)          # real solver names are not exact: 000015s, 003604s, 007200s, 010798s
            with rasterio.open(os.path.join(src, "depth_%06ds.tif" % max(sec, 0)), "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32",
                               crs="EPSG:32616", transform=Affine(5, 0, e0, 0, -5, n0), nodata=-9999.0) as d:
                d.write(a, 1)
        with rasterio.open(os.path.join(src, "max_depth.tif"), "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32",
                           crs="EPSG:32616", transform=Affine(5, 0, e0, 0, -5, n0), nodata=-9999.0) as d:
            d.write(mx, 1)
        make_cogs.run(src, os.path.join(out_base, "county", storm), res_m=RES)
    return out_base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--viewer-demo", help="write SYNTHETIC demo COGs for ?cogbase= testing into this folder instead")
    a = ap.parse_args()
    if a.viewer_demo:
        build_viewer_demo(a.viewer_demo)
        sys.exit(0)
    if not a.out:
        ap.error("--out or --viewer-demo is required")
    e = build(a.out)
    print("wrote", a.out, "driver", e["driver"], {k: (v["width"], v["height"], v["overviews"]) for k, v in e["files"].items()})
