"""Tile-derived vs flow-grid-derived results on the real county data.

    python -m bgv.analysis.verify [--points 150] [--seed 1] [--out docs/ANALYSIS_TILE_VS_GRID.md]

Prints a markdown report (and writes it with --out):
  1. depth at random points (>= 100 per case, half of them where either source is wet) for several storms / hours: tile bilinear depth
     minus flow-grid bilinear depth (mean, mean |diff|, RMS, p50 / p95 / max |diff|, agreement of wet / dry);
  2. the domain-wide numbers of the same hour computed both ways: flooded buildings, wet road length, wet area, max depth
     (tiles: full-resolution band pass of the analysis timeline; flow grid: data/stats.py, the sidebar's numbers).
It reads the data folder read-only.  Nothing here is used by the app.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

CASES = (("100yr", 12), ("100yr", 24), ("25yr", 12), ("2yr", 18), ("ev4", 18), ("ev6", 12))
STAT_CASES = (("100yr", 6), ("100yr", 12), ("100yr", 24), ("25yr", 12), ("2yr", 18))


def point_comparison(src, storm: str, hour: int, n: int, rng) -> dict:
    from . import geodesy
    lng = rng.uniform(-86.62, -86.22, 40 * n)
    lat = rng.uniform(36.76, 37.2, 40 * n)
    dt = src.tiles.depth_many(lng, lat, storm, float(hour))
    dg = src.grid.depth_many(lng, lat, storm, float(hour))
    wet = np.flatnonzero((dt > 0) | (dg > 0))
    dry = np.flatnonzero((dt == 0) & (dg == 0))
    pick = np.concatenate([rng.choice(wet, min(n // 2 + n % 2, wet.size), replace=False),
                           rng.choice(dry, min(n // 2, dry.size), replace=False)])
    t, g = dt[pick], dg[pick]
    w = (t > 0) | (g > 0)
    d = t - g
    ad = np.abs(d)
    out = {"storm": storm, "hour": hour, "n": int(pick.size), "n_wet": int(w.sum()),
           "both_wet": int(((t > 0) & (g > 0)).sum()), "only_tiles": int(((t > 0) & (g == 0)).sum()),
           "only_grid": int(((t == 0) & (g > 0)).sum()), "max_tile": float(t.max()), "max_grid": float(g.max())}
    if w.any():
        dw = d[w]
        out.update(mean=float(dw.mean()), mean_abs=float(np.abs(dw).mean()), rms=float(np.sqrt((dw ** 2).mean())),
                   p50=float(np.percentile(np.abs(dw), 50)), p95=float(np.percentile(np.abs(dw), 95)), max_abs=float(np.abs(dw).max()))
    return out


def domain_numbers(src, dom, frame, B, R, storm: str, hour: int) -> dict:
    from ..data import stats
    from . import timeline
    ts = src.tiles.storm(storm)
    r = ts.raster(hour)
    bpts, rpts = timeline.prepare_points(r, frame, B, R)
    tile = timeline.tile_hour_stats(r, frame, bpts, rpts)
    st = src.grid.storm_obj(storm)
    g = None if st.is_zero(hour) else st.load_hour(hour, cache=False)
    bs = stats.evaluate_buildings(g, B.index)
    rs = stats.evaluate_roads(g, R)
    gs = stats.summarize_grid(g, frame)
    return {"storm": storm, "hour": hour, "tile": tile,
            "grid": {"flooded": float(bs.flooded_count), "road_km": rs.flooded_length_m / 1000.0, "area_km2": gs.wet_area_km2,
                     "max_depth_m": gs.max_depth_m, "volume_m3": gs.volume_m3}}


def report(points: int = 150, seed: int = 1) -> str:
    from .. import config, domain as dm
    from ..data import buildings as bmod, roads as rmod
    from .depth_source import make_depth_source
    root = config.data_root()
    dom = dm.resolve("county", root, {})
    src = make_depth_source(root=root, domain=dom)
    if src.tiles is None:
        raise SystemExit("no county tiles found")
    rng = np.random.default_rng(seed)
    lines = ["# Tile-derived vs flow-grid-derived depth (Warren County)", "",
             f"Tiles: `{src.tiles.dir}` (6.25 m, EPSG:3857, uint16 cm), label: *{src.label('100yr')}*.",
             "Flow grid: `layers/flow_county/<storm>` (25 m Mercator cells, about 20 m on the ground; the builder block-averages the solver depth).",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m bgv.analysis.verify --points {points} --seed {seed}`.", "",
             "## 1. Depth at random points (bilinear on both sides; wet = depth >= 0.02 m)", "",
             "| storm | hour | points | wet in either | both wet | tiles only | grid only | mean (tile - grid) m | mean abs m | RMS m | p50 abs | p95 abs | max abs | max tile | max grid |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = 0
    for storm, hour in CASES:
        c = point_comparison(src, storm, hour, points, rng)
        tot += c["n"]
        lines.append(f"| {storm} | {hour} | {c['n']} | {c['n_wet']} | {c['both_wet']} | {c['only_tiles']} | {c['only_grid']} | "
                     f"{c.get('mean', float('nan')):+.3f} | {c.get('mean_abs', float('nan')):.3f} | {c.get('rms', float('nan')):.3f} | "
                     f"{c.get('p50', float('nan')):.3f} | {c.get('p95', float('nan')):.3f} | {c.get('max_abs', float('nan')):.2f} | "
                     f"{c['max_tile']:.2f} | {c['max_grid']:.2f} |")
    lines += ["", f"{tot} points in total; the statistics are over the wet-in-either points of each row.", "",
              "## 2. Domain-wide numbers of the same hour", ""]
    frame = dom.open_storm("100yr").frame
    bp, rp, _ = dom.scene_paths(frame)
    B, R = bmod.load_buildings(root, frame, geojson=bp), rmod.load_roads(root, frame, geojson=rp)
    lines += [f"Scene: {B.n:,} OSM buildings, {R.total_length_m / 1000:,.0f} km of roads.", "",
              "| storm | hour | flooded buildings tiles / grid | wet roads km tiles / grid | wet area km2 tiles / grid | max depth m tiles / grid | volume million m3 tiles / grid |",
              "|---|---|---|---|---|---|---|"]
    for storm, hour in STAT_CASES:
        d = domain_numbers(src, dom, frame, B, R, storm, hour)
        t, g = d["tile"], d["grid"]
        lines.append(f"| {storm} | {hour} | {t['flooded']:,.0f} / {g['flooded']:,.0f} | {t['road_km']:.1f} / {g['road_km']:.1f} | "
                     f"{t['area_km2']:.1f} / {g['area_km2']:.1f} | {t['max_depth_m']:.2f} / {g['max_depth_m']:.2f} | "
                     f"{t['volume_m3'] / 1e6:.1f} / {g['volume_m3'] / 1e6:.1f} |")
    lines += ["", "Tile rule: flooded building = depth >= 0.1 m at any footprint corner or the centroid (nearest 6.25 m cell); wet road = 10 m "
              "road samples with depth >= 0.1 m; area = cells >= 0.02 m. Flow-grid rule: data/stats.py (all-touched pixels, same thresholds)."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    text = report(a.points, a.seed)
    sys.stdout.buffer.write(text.encode("utf-8"))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
