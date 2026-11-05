"""Per-building footprint maximum of the solver tiles vs rasterio (all_touched mask) on the real county buildings.

    python desktop/tests/tile_verify_buildings.py [--storm 100yr] [--n 300] [--which peak|12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")


def main(argv=None):
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from rasterio.windows import Window, from_bounds
    from bgv import config
    from bgv.data import buildings as bmod, cog as cogmod, tiles
    from bgv.geo import GridFrame
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm", default="100yr")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--which", default="peak")
    ap.add_argument("--geojson", default=None)
    a = ap.parse_args(argv)
    gj = Path(a.geojson) if a.geojson else next((config.cache_dir() / "osm").glob("*/buildings.geojson"))
    feats = json.loads(gj.read_text(encoding="utf-8"))["features"]
    which = "peak" if a.which == "peak" else int(a.which)
    svc = tiles.TileService(ROOT)
    ts = svc.tileset(a.storm)
    c = ts.cog(which)
    W, H, res = c.width, c.height, c.res
    frame = GridFrame(W, H, res, res, c.x0, c.y0 - H * res, c.x0 + W * res, c.y0)
    idx = bmod.buildings_from_features(feats, frame, index_only=True)
    got = tiles.footprint_max(c, idx)
    rng = np.random.default_rng(7)
    ids = rng.choice(len(feats), a.n, replace=False)
    bad = 0
    wet = 0
    with rasterio.open(ts.folder / ts.file_for(which)) as d:
        for i in ids:
            g = feats[i]["geometry"]
            gm = transform_geom("EPSG:4326", "EPSG:3857", g)
            geoms = [gm]
            win = from_bounds(*rasterio.features.bounds(gm), transform=d.transform).round_offsets().round_lengths()
            win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2)
            arr = d.read(1, window=win, boundless=True, fill_value=65535)
            tr = d.window_transform(win)
            m = ~geometry_mask(geoms, out_shape=arr.shape, transform=tr, all_touched=True)
            v = arr[m & (arr != 65535)]
            exp = int(v.max()) if v.size else -1
            wet += exp >= 10
            if exp != int(got[i]):
                bad += 1
                print("MISMATCH", i, exp, int(got[i]))
    print(f"{a.storm} {a.which}: {a.n} random buildings (of {len(feats)}), {wet} flooded (>=10 cm), mismatches vs rasterio all_touched: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
