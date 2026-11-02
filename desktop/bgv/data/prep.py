"""Build every data cache in one go.

    python -m bgv.data.prep [--all] [--data <root>] [--storm 25yr] [--rebuild]

PUBLIC API
    default_frame(root, storm=None) -> GridFrame        flow-grid frame read from layers/flow/<storm>/meta.json
                                                        (default storm: 25yr, else the first storm folder with a meta.json)
    ensure_all(root, frame=None, progress=None, *, include_terrain=False, rebuild=False) -> dict
        Builds / loads the caches (desktop/cache/*.npz) and returns {"frame", "buildings", "roads"[, "terrain"]}.
        Safe to call from a worker thread (no GUI, no GPU). progress(msg:str, frac:float 0..1) is called as it goes.
        include_terrain also warms bgv.data.terrain.load_terrain (cache) when that module is present.
        warm_stats (default True) also compiles / loads the numba kernels of bgv.data.stats (stats.warmup()).
    --all  : also builds the terrain cache. Without it only the buildings and roads caches are built.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..geo import GridFrame


def default_frame(root, storm: str | None = None) -> GridFrame:
    flow = Path(root) / "layers" / "flow"
    cands = [storm] if storm else ["25yr"]
    cands += sorted(p.name for p in flow.iterdir() if p.is_dir())
    for s in cands:
        mp = flow / s / "meta.json"
        if mp.is_file():
            with open(mp, "r", encoding="utf-8") as fh:
                return GridFrame.from_meta(json.load(fh))
    raise FileNotFoundError(f"no meta.json under {flow}")


def ensure_all(root, frame: GridFrame | None = None, progress=None, *, include_terrain: bool = False,
               rebuild: bool = False, warm_stats: bool = True) -> dict:
    from . import buildings, roads

    def rep(msg, frac):
        if progress is not None:
            try:
                progress(msg, frac)
            except Exception:
                pass

    if frame is None:
        frame = default_frame(root)
    out = {"frame": frame}
    use_cache = not rebuild
    # fraction of the whole job each step owns
    plan = [("buildings", 0.35), ("roads", 0.15), ("stats", 0.10 if warm_stats else 0.0),
            ("terrain", 0.40 if include_terrain else 0.0)]
    total = sum(w for _, w in plan)
    lo = {}
    acc = 0.0
    for name, w in plan:
        lo[name] = (acc / total, (acc + w) / total)
        acc += w

    def sub(name):
        a, b = lo[name]
        return lambda msg, f: rep(msg, a + (b - a) * (f if f is not None else 0.0))

    out["buildings"] = buildings.load_buildings(root, frame, progress=sub("buildings"), use_cache=use_cache)
    rep("buildings ready", lo["buildings"][1])
    out["roads"] = roads.load_roads(root, frame, progress=sub("roads"), use_cache=use_cache)
    rep("roads ready", lo["roads"][1])
    if warm_stats:
        from . import stats
        rep("compiling statistics kernels", lo["stats"][0])
        out["warmup_s"] = stats.warmup()
        rep("statistics kernels ready", lo["stats"][1])
    if include_terrain:
        try:
            from . import terrain
            rep("terrain: loading / warping DEM", lo["terrain"][0])
            out["terrain"] = terrain.load_terrain(root, frame, use_cache=use_cache)
            rep("terrain ready", 1.0)
        except ImportError:
            rep("terrain module not available, skipped", 1.0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bgv.data.prep", description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="also build the terrain cache")
    ap.add_argument("--data", default=None, help="data root (folder containing layers/flow)")
    ap.add_argument("--storm", default=None, help="storm whose meta.json defines the grid (default 25yr)")
    ap.add_argument("--rebuild", action="store_true", help="ignore existing caches")
    a = ap.parse_args(argv)
    from .. import config
    if a.data:
        config.set_data_root(a.data)
    root = config.data_root()
    frame = default_frame(root, a.storm)
    print(f"data root: {root}\ngrid: {frame.width}x{frame.height} cell {frame.cell_m} m ({frame.cell_ground_m} m ground)")
    t0 = time.perf_counter()

    def cb(msg, frac):
        print(f"[{100 * (frac or 0):5.1f}%] {msg}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    res = ensure_all(root, frame, cb, include_terrain=a.all, rebuild=a.rebuild)
    b, r = res["buildings"], res["roads"]
    print(f"buildings: {b.n} ({int(b.in_grid.sum())} on the grid), mesh {b.mesh.vertex_count} verts / "
          f"{b.mesh.triangle_count} tris, pixel index {b.index.pixels.size} px  {b.timings}")
    print(f"roads: {r.n_roads} roads, {r.n_segments} segments, {r.n_samples} samples, "
          f"{r.total_length_m / 1000:.1f} km  {r.timings}")
    print(f"done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
