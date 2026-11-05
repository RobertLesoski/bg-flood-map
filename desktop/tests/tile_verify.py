"""Spot check of the native tile reader against rasterio (GDAL) on the real COGs, and of tile depth against the flow-grid depth.

    python desktop/tests/tile_verify.py [--storms 2yr 100yr ev4] [--n 250] [--hour 12] [--json out.json]

For every storm: N random lng/lat points inside the county tile grid (about two thirds uniform, one third drawn from wet cells so that
the comparison is not mostly zeros); lng/lat -> mercator through PROJ (rasterio.warp.transform, independent of the reader's own
spherical formula); the reader's value must be IDENTICAL to rasterio's at level 0 (peak and the chosen hour).  Then the same points
are read from the flow grid (flow_county, 20 m cells, sqrt-coded) to report how far the two depth sources are apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")


def run(storm: str, n: int, hour: int, rng, root=ROOT) -> dict:
    import rasterio
    from rasterio.warp import transform as warp
    from bgv.data import cog as cogmod
    from bgv.data import flow as flowdata
    from bgv.data.tiles import TileService
    from bgv.geo import GridFrame

    svc = TileService(root)
    ts = svc.tileset(storm)
    assert ts is not None, f"no tiles for {storm}"
    peak = ts.cog("peak")
    xmin, ymin, xmax, ymax = peak.bounds
    # wet cells of the peak raster (overview 2 is enough to find wet areas), then jitter inside the cell
    lv = 2
    win = peak.read_window(lv, 0, 0, peak.levels[lv].width, peak.levels[lv].height)
    wet = np.argwhere((win > 20) & (win != 65535))
    nw = n // 3
    pick = wet[rng.integers(0, len(wet), nw)] if len(wet) else np.zeros((0, 2), int)
    rx, ry = peak.level_res(lv)
    xw = peak.x0 + (pick[:, 1] + rng.random(nw)) * rx
    yw = peak.y0 - (pick[:, 0] + rng.random(nw)) * ry
    xu = rng.uniform(xmin, xmax, n - nw)
    yu = rng.uniform(ymin, ymax, n - nw)
    xs, ys = np.concatenate([xw, xu]), np.concatenate([yw, yu])
    lng, lat = warp("EPSG:3857", "EPSG:4326", list(xs), list(ys))
    lng, lat = np.array(lng), np.array(lat)
    x_proj, y_proj = warp("EPSG:4326", "EPSG:3857", list(lng), list(lat))        # PROJ's own lng/lat -> mercator

    out = {"storm": storm, "n": int(n), "hour": hour}
    for which, name in (("peak", "max_depth.tif"), (hour, ts.hours.get(hour))):
        mism = 0
        vals = []
        with rasterio.open(ts.folder / name) as d:
            ref = np.array([v[0] for v in d.sample(list(zip(x_proj, y_proj)))], dtype=np.int64)
        got = np.array([svc.depth_lnglat(storm, a, b, which) for a, b in zip(lng, lat)], dtype=object)
        for g, r in zip(got, ref):
            if r == 65535:
                ok = g.status == "nodata"
            else:
                ok = g.status in ("depth", "dry") and int(round(g.cm)) == int(r)
            mism += 0 if ok else 1
            vals.append(-1 if g.cm is None else g.cm)
        out[f"{which}_mismatch"] = int(mism)
        out[f"{which}_file"] = name
        out[f"{which}_wet_points"] = int(((ref > 0) & (ref != 65535)).sum())
        out[f"{which}_nodata_points"] = int((ref == 65535).sum())
        out[f"{which}_max_cm"] = int(ref[ref != 65535].max()) if (ref != 65535).any() else 0
        out[f"_tile_{which}"] = np.array(vals)
    # flow grid (county)
    st = flowdata.Storm.open(root, storm, "flow_county")
    fr = st.frame
    X, Y = np.array([fr.lnglat_to_world(a, b) for a, b in zip(lng, lat)]).T
    def grid_at(h):
        r = [st.sample_t(float(h), x, y) for x, y in zip(X, Y)]
        return np.array([q["depth"] if q["valid"] else np.nan for q in r])
    grid_h = grid_at(hour)
    gp = np.zeros(n)
    for h in st.hours:
        gp = np.fmax(gp, grid_at(h))
    for which, g in ((hour, grid_h), ("peak", gp)):
        t = out[f"_tile_{which}"] / 100.0
        ok = np.isfinite(g) & (t >= 0)
        both_wet = ok & (t >= 0.1) & (g >= 0.1)
        out[f"grid_vs_tile_{which}"] = {
            "points_compared": int(ok.sum()),
            "tile_wet_grid_dry": int((ok & (t >= 0.1) & (g < 0.1)).sum()),
            "grid_wet_tile_dry": int((ok & (g >= 0.1) & (t < 0.1)).sum()),
            "both_wet": int(both_wet.sum()),
            "mean_abs_diff_m_both_wet": float(np.abs(t[both_wet] - g[both_wet]).mean()) if both_wet.any() else None,
            "median_ratio_grid_over_tile_both_wet": float(np.median(g[both_wet] / t[both_wet])) if both_wet.any() else None,
            "mean_abs_diff_m_all": float(np.abs(t[ok] - g[ok]).mean()) if ok.any() else None,
            "max_abs_diff_m": float(np.abs(t[ok] - g[ok]).max()) if ok.any() else None,
        }
    st.close()
    for k in [k for k in out if k.startswith("_tile_")]:
        del out[k]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storms", nargs="+", default=["2yr", "100yr", "ev4"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    res = [run(s, a.n, a.hour, rng, Path(a.root)) for s in a.storms]
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    bad = sum(r["peak_mismatch"] + r[f"{a.hour}_mismatch"] for r in res)
    print("TOTAL MISMATCHES vs rasterio:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
