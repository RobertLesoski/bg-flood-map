"""SYNTHETIC county demo data, so the county view can be tried before the real Vista results arrive.

    python -m bgv --make-county-demo <folder>
    python -m bgv --domain county --county-demo <folder>          # real data in the data root always takes precedence

Writes into <folder> (never into the data root):
    layers/dem_warren_demo.tif                  synthetic terrain on the real county extent: UTM 16N, METRES, 20 m cells,
                                                rolling hills, a main river with three creeks in incised valleys, karst sinkholes
    layers/flow_county/100yr/, 10yr/            two storms of 24 hourly depth grids built by the REAL converter
                                                scripts/build_flow_from_rasters.py from fabricated hourly depth rasters on the
                                                solver grid (the same code path the Vista rasters take), so the files have exactly
                                                the shape of the real ones (about 2630 x 3050 cells of 25 m mercator / 20 m ground)
    README_DEMO.txt
Every meta.json carries `"synthetic": true`: the viewer then shows a DEMO DATA badge on every screen and stamps screenshots.
The water is a bathtub model (height above the nearest channel against a stage that follows a flood hydrograph, plus ponding in
the sinkholes): it has the right geometry and a plausible timing but it is NOT a simulation of anything.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

COUNTY_UTM_BOX = (524851.0, 4056081.0, 576871.0, 4116296.0)     # xmin, ymin, xmax, ymax, EPSG:32616 (the Vista county grid)
DEMO_CELL_M = 20.0
DEM_FILE = "dem_warren_demo.tif"
# (storm, label, peak stage of the main river above its bed (m), hour of the peak)
DEMO_STORMS = (("100yr", "100-year storm (county, DEMO DATA)", 7.0, 15.0), ("10yr", "10-year storm (county, DEMO DATA)", 3.2, 13.0))

# river / creeks as polylines in fractions of the box (u east, v north), loosely where the real ones run
CHANNELS = (
    # (points, incision m, valley half width m, stage scale vs the main river)
    ([(1.0, 0.30), (0.82, 0.38), (0.68, 0.55), (0.58, 0.66), (0.50, 0.70), (0.42, 0.66), (0.30, 0.72), (0.12, 0.80),
      (0.0, 0.86)], 14.0, 520.0, 1.0),
    ([(0.55, 0.0), (0.56, 0.25), (0.52, 0.45), (0.50, 0.68)], 7.0, 300.0, 0.55),
    ([(0.30, 1.0), (0.33, 0.88), (0.36, 0.74)], 6.0, 260.0, 0.5),
    ([(0.85, 1.0), (0.78, 0.80), (0.70, 0.60)], 6.0, 280.0, 0.5),
)


def _find_scripts() -> Path:
    from .. import config
    cands = [config.source_root().parent / "scripts"]
    try:
        cands.append(config.data_root() / "scripts")
    except Exception:
        pass
    for c in cands:
        if (c / "build_flow_from_rasters.py").is_file() and (c / "flowgrid.py").is_file():
            return c
    raise FileNotFoundError("scripts/build_flow_from_rasters.py not found (looked in "
                            + ", ".join(str(c) for c in cands) + "): the demo flow grids are built with it")


# ------------------------------------------------------------------------------------------------ terrain
def _chaikin(pts: np.ndarray, n: int = 3) -> np.ndarray:
    for _ in range(n):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.concatenate([pts[:1], np.stack([q, r], 1).reshape(-1, 2), pts[-1:]])
    return pts


def _channel_pixels(pts_frac, width: int, height: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Row / col of the pixels of a meandering polyline given in box fractions (v north)."""
    p = np.array(pts_frac, np.float64)
    p = _chaikin(p, 3)
    t = np.linspace(0, 1, p.shape[0])
    wig = 0.012 * np.sin(t * rng.uniform(14, 22) + rng.uniform(0, 6.28)) + 0.006 * np.sin(t * rng.uniform(40, 60))
    p[:, 0] += wig * np.gradient(p[:, 1]) * 20.0
    p[:, 1] -= wig * np.gradient(p[:, 0]) * 20.0
    col = p[:, 0] * (width - 1)
    row = (1.0 - p[:, 1]) * (height - 1)
    seg = np.hypot(np.diff(col), np.diff(row))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(cum[-1] * 2), 2)
    s = np.linspace(0, cum[-1], n)
    return np.rint(np.interp(s, cum, row)).astype(np.int64), np.rint(np.interp(s, cum, col)).astype(np.int64)


def synth_terrain(width: int, height: int, cell_m: float = DEMO_CELL_M, seed: int = 7, channels=CHANNELS):
    """(z float32[h, w] metres, hand float32[h, w] height above the nearest channel's bed, stage float32[h, w] stage scale of
    that channel, sinks list[(row, col, radius_px, depth_m, bottom_m)]).  Row 0 = north, like a GeoTIFF.

    The surface is built FROM the drainage: every cell is its nearest channel's bed (falling downstream) plus a valley profile
    that rises with the distance to the channel plus rolling hills that fade in away from it, so water always has somewhere
    to go and the height above the nearest channel (HAND) is meaningful over the whole county."""
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    cs = max(int(round(100.0 / cell_m)), 1)
    h5, w5 = -(-height // cs) + 4, -(-width // cs) + 4
    big = ndimage.gaussian_filter(rng.standard_normal((h5, w5)), sigma=(max(1500.0 / 100.0, 1.0),) * 2)
    big = big / max(float(big.std()), 1e-9)
    hills = ndimage.zoom(big, cs, order=3)[:height, :width].astype(np.float32) * 9.0
    hills += ndimage.gaussian_filter(rng.standard_normal((height, width)).astype(np.float32), sigma=2.0) * 2.0
    del big
    # ---- channels: pixels, bed elevation (main river falls downstream, creeks fall toward their confluence)
    mask = np.zeros((height, width), bool)
    bed = np.zeros((height, width), np.float32)
    inc = np.zeros((height, width), np.float32)
    wid = np.ones((height, width), np.float32)
    stg = np.zeros((height, width), np.float32)
    main_r = main_c = main_bed = None
    for k, (pts, incision, half_w, stage) in enumerate(channels):
        r, c = _channel_pixels(pts, width, height, rng)
        ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        r, c = r[ok], c[ok]
        n = r.size
        s_frac = np.arange(n, dtype=np.float32) / max(n - 1, 1)
        if k == 0:
            b = 150.0 + 28.0 * (1.0 - s_frac)                          # upstream (east) high, downstream (west) low
            main_r, main_c, main_bed = r, c, b
        else:                                                         # the creek ends where it meets the main river
            d2 = (main_r[None, :] - r[-1]) ** 2 + (main_c[None, :] - c[-1]) ** 2
            conf = int(np.argmin(d2.min(axis=0))) if main_r is not None else 0
            length_m = float(n) * cell_m * 0.5
            b = main_bed[conf] + 0.0022 * length_m * (1.0 - s_frac)
        mask[r, c] = True
        bed[r, c] = b
        inc[r, c] = incision
        wid[r, c] = half_w
        stg[r, c] = stage
    dist, idx = ndimage.distance_transform_edt(~mask, return_distances=True, return_indices=True)
    iy, ix = idx[0], idx[1]
    del idx
    d_m = dist.astype(np.float32) * np.float32(cell_m)
    del dist
    bed_n, inc_n, wid_n, stg_n = bed[iy, ix], inc[iy, ix], wid[iy, ix], stg[iy, ix]
    del iy, ix, bed, inc, wid, stg, mask
    valley = inc_n * (1.0 - np.exp(-(d_m / (wid_n * 1.6)) ** 2)) + 42.0 * (1.0 - np.exp(-d_m / 6000.0))
    fade = np.clip(d_m / 3000.0, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    z = bed_n + valley + hills * fade
    z = np.maximum(z, bed_n + 0.0015 * d_m + 0.3)                       # never below the bed of the nearest channel: no lakes
    del valley, fade, hills, d_m, inc_n, wid_n
    # ---- karst sinkholes: flat-floored bowls
    sinks = []
    for _ in range(max(int(round(40 * (width * height) / (2600.0 * 3000.0))), 3)):
        r0, c0 = int(rng.integers(0, height)), int(rng.integers(0, width))
        rad = float(rng.uniform(80.0, 260.0)) / cell_m
        dep = float(rng.uniform(2.5, 7.0))
        a, b = max(int(r0 - 3 * rad), 0), min(int(r0 + 3 * rad) + 1, height)
        e, f = max(int(c0 - 3 * rad), 0), min(int(c0 + 3 * rad) + 1, width)
        if b <= a or f <= e:
            continue
        rr, cc = np.mgrid[a:b, e:f]
        g = np.exp(-1.2 * ((rr - r0) ** 2 + (cc - c0) ** 2) / (rad * rad)).astype(np.float32)
        blk = z[a:b, e:f]
        ring = (np.hypot(rr - r0, cc - c0) > 1.6 * rad)
        z_ref = float(blk[ring].mean()) if ring.any() else float(blk.mean())
        z[a:b, e:f] = blk * (1.0 - g) + np.float32(z_ref - dep) * g        # rim blended into the slope
        sinks.append((r0, c0, rad, dep, z_ref - dep))
    hand = np.maximum(z - bed_n, 0.0).astype(np.float32)
    return z.astype(np.float32), hand, stg_n.astype(np.float32), sinks


def hydrograph(t: float, t_peak: float) -> float:
    """Flood stage fraction (1 at the peak hour): rises quickly, recedes slowly."""
    if t <= 0:
        return 0.0
    x = t / t_peak
    return float((x * math.exp(1.0 - x)) ** 2.5)


def depth_field(z, hand, stage, sinks, t: float, peak_m: float, t_peak: float, cell_m: float = DEMO_CELL_M) -> np.ndarray:
    """Water depth (m) at hour t: river / creek flooding (stage over the height above the nearest channel) + ponding that
    fills the sinkholes with the rain.  Tiny depths are dropped (dry)."""
    h = hydrograph(t, t_peak)
    d = np.maximum(peak_m * h * stage - hand, 0.0)
    fill = min(1.0, 0.25 + 0.85 * (t / 24.0)) * (peak_m / 7.0) ** 0.5
    for r0, c0, rad, dep, bottom in sinks:
        a, b = max(int(r0 - 3 * rad), 0), min(int(r0 + 3 * rad) + 1, d.shape[0])
        e, f = max(int(c0 - 3 * rad), 0), min(int(c0 + 3 * rad) + 1, d.shape[1])
        level = bottom + dep * fill * 0.9 * min(h * 1.5 + 0.15, 1.0)
        rr, cc = np.mgrid[a:b, e:f]
        bowl = np.exp(-1.2 * ((rr - r0) ** 2 + (cc - c0) ** 2) / (rad * rad)) > 0.35
        pond = np.minimum(np.maximum(level - z[a:b, e:f], 0.0), dep) * bowl
        blk = d[a:b, e:f]
        np.maximum(blk, pond, out=blk)
    d[d < 0.005] = 0.0
    return d.astype(np.float32)


# ------------------------------------------------------------------------------------------------ files
def _write_tif(path, arr, transform, nodata=-9999.0):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                       crs="EPSG:32616", transform=transform, nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3, zlevel=4) as ds:
        ds.write(np.ascontiguousarray(arr, np.float32), 1)


def make_county_demo(folder, *, utm_box=COUNTY_UTM_BOX, cell_m: float = DEMO_CELL_M, storms=DEMO_STORMS, hours=None,
                     workers: int = 3, seed: int = 7, log=print, keep_work: bool = False) -> dict:
    """Write the synthetic county terrain + storms into `folder` (see the module docstring).  Returns {dir, dem, storms,
    seconds, grid}.  `utm_box` / `hours` / `cell_m` let the tests build a tiny version."""
    from rasterio.transform import Affine
    t_all = time.perf_counter()
    folder = Path(folder)
    layers = folder / "layers"
    layers.mkdir(parents=True, exist_ok=True)
    scripts = _find_scripts()
    sys.path.insert(0, str(scripts))
    try:
        import build_flow_from_rasters as br
        import flowgrid as fg
    finally:
        sys.path.remove(str(scripts))
    xmin, ymin, xmax, ymax = utm_box
    width = int(math.ceil((xmax - xmin) / cell_m))
    height = int(math.ceil((ymax - ymin) / cell_m))
    tr = Affine(cell_m, 0.0, xmin, 0.0, -cell_m, ymax)
    log(f"[demo] synthetic terrain {width} x {height} cells of {cell_m:g} m (UTM 16N, metres) ...")
    t0 = time.perf_counter()
    z, hand, stage, sinks = synth_terrain(width, height, cell_m, seed)
    dem = layers / DEM_FILE
    _write_tif(dem, z, tr)
    log(f"[demo] terrain: {float(z.min()):.0f} .. {float(z.max()):.0f} m, {len(sinks)} sinkholes, wrote {dem} "
        f"({dem.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s)")
    work = folder / ".work"
    hours = list(range(1, 25)) if hours is None else [int(h) for h in hours]
    out_storms = []
    flow = layers / "flow_county"
    for name, label, peak_m, t_peak in storms:
        run = work / name
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        t1 = time.perf_counter()
        wet_any = np.zeros((height, width), np.float32)
        _write_tif(run / "depth_000000s.tif", np.zeros((height, width), np.float32), tr)
        for h in hours:
            d = depth_field(z, hand, stage, sinks, float(h), peak_m, t_peak, cell_m)
            np.maximum(wet_any, d, out=wet_any)
            _write_tif(run / f"depth_{h * 3600:06d}s.tif", d, tr)
        wet_any[:3, :3] = np.maximum(wet_any[:3, :3], 0.05)               # the wet bounding box spans the whole domain
        wet_any[:3, -3:] = np.maximum(wet_any[:3, -3:], 0.05)
        wet_any[-3:, :3] = np.maximum(wet_any[-3:, :3], 0.05)
        wet_any[-3:, -3:] = np.maximum(wet_any[-3:, -3:], 0.05)
        _write_tif(run / "max_depth.tif", wet_any, tr)
        (run / "run_summary.json").write_text(json.dumps(
            {"storm": name, "solver": "synthetic demo (bathtub), NOT a simulation", "total_rain_mm": 25.4 * (7.45 if name == "100yr" else 4.97)}))
        log(f"[demo] {name}: {len(hours)} synthetic depth rasters in {time.perf_counter() - t1:.0f} s, converting ...")
        out = flow / name
        res = br.build_storm(str(run), str(out), name, label=label, cell_m=25.0, dem=str(dem), dem_units_arg="metres",
                             workers=workers, hours=",".join(str(h) for h in hours), force=True, log=lambda m: log("[demo]   " + m))
        if res.get("problems"):
            raise RuntimeError(f"demo storm {name}: validation problems: {res['problems'][:3]}")
        mp = out / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["synthetic"] = True
        meta["label"] = label
        meta["source_note"] = ("DEMO DATA - SYNTHETIC. The terrain and the water were fabricated by `python -m bgv --make-county-demo` "
                               "(a bathtub model with a flood hydrograph) to try the county view before the real Vista results arrive; "
                               "nothing here is a simulation of Warren County. " + str(meta.get("source_note", "")))
        fg.write_meta(str(mp), meta)
        out_storms.append(name)
        shutil.rmtree(run, ignore_errors=True)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    (folder / "README_DEMO.txt").write_text(
        "SYNTHETIC DEMO DATA for the Warren County view of the BG Flood Viewer.\n"
        "Written by `python -m bgv --make-county-demo`. The terrain and the flooding are fabricated; they have the real\n"
        "grid, extent and file formats but are NOT results of any simulation.\n"
        "View:  python -m bgv --domain county --county-demo <this folder>\n"
        "Real county data in <data root>\\layers\\flow_county and <data root>\\layers\\dem_warren_5m.tif always takes precedence.\n",
        encoding="utf-8")
    meta = json.loads((flow / out_storms[0] / "meta.json").read_text(encoding="utf-8"))
    return {"dir": str(folder), "dem": str(dem), "storms": out_storms, "seconds": time.perf_counter() - t_all,
            "grid": (meta["width"], meta["height"], meta["cell_m"], meta["cell_ground_m"])}
