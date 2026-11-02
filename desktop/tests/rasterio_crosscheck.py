"""Cross-check of the all-touched pixel index against rasterio.features.rasterize(all_touched=True) for EVERY building.

Run standalone:  python tests/rasterio_crosscheck.py [storm] [hour]      (prints the report)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("tests", 1)[0])


def crosscheck(root, frame, buildings, grid_rgba, progress=None):
    import rasterio.features
    from affine import Affine
    from bgv.geo import R_MERC

    with open(root / "layers" / "buildings.geojson", "r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    W, H = frame.width, frame.height
    idx = buildings.index
    R = grid_rgba[..., 0].reshape(-1)
    A = grid_rgba[..., 3].reshape(-1)
    diff_pix = []
    diff_depth = []
    only_mine = 0
    only_rio = 0
    t0 = time.perf_counter()
    for i, f in enumerate(feats):
        ring0 = np.asarray(f["geometry"]["coordinates"][0], dtype=np.float64)
        rings = [np.asarray(r, dtype=np.float64) for r in f["geometry"]["coordinates"]]
        pr = []
        for r in rings:
            x = R_MERC * np.radians(r[:, 0])
            y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(r[:, 1]) / 2))
            pr.append(np.stack([(x - frame.xmin) / frame.cell_m, (frame.ymax - y) / frame.cell_m], axis=1))
        allp = np.concatenate(pr)
        c0 = max(int(np.floor(allp[:, 0].min())), 0)
        c1 = min(int(np.floor(allp[:, 0].max())), W - 1)
        r0 = max(int(np.floor(allp[:, 1].min())), 0)
        r1 = min(int(np.floor(allp[:, 1].max())), H - 1)
        mine = idx.pixels_of(i)
        if c0 > c1 or r0 > r1:
            ref = np.zeros(0, dtype=np.int64)
        else:
            geom = {"type": "Polygon", "coordinates": [p.tolist() for p in pr]}
            m = rasterio.features.rasterize([(geom, 1)], out_shape=(r1 - r0 + 1, c1 - c0 + 1),
                                            transform=Affine(1, 0, c0, 0, 1, r0), all_touched=True, fill=0,
                                            dtype="uint8")
            rr, cc = np.nonzero(m)
            ref = (rr + r0).astype(np.int64) * W + (cc + c0)
        sm = set(mine.tolist())
        sr = set(ref.tolist())
        if sm != sr:
            diff_pix.append(i)
            only_mine += len(sm - sr)
            only_rio += len(sr - sm)
        # depth code differences
        def maxcode(px):
            if len(px) == 0:
                return 0
            px = np.asarray(sorted(px), dtype=np.int64)
            v = np.where(A[px] != 0, R[px], 0)
            return int(v.max())
        if maxcode(sm) != maxcode(sr):
            diff_depth.append(i)
        if progress and i % 5000 == 0:
            progress(i)
    return {"n": len(feats), "pixel_set_differs": diff_pix, "pixels_only_mine": only_mine, "pixels_only_rasterio": only_rio,
            "depth_code_differs": diff_depth, "seconds": time.perf_counter() - t0}


def crosscheck_shapely(root, frame, buildings):
    """Exact reference: pixel owned iff the closed pixel square intersects the polygon (shapely). Returns mismatching ids."""
    import shapely
    from bgv.geo import R_MERC

    with open(root / "layers" / "buildings.geojson", "r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    W, H = frame.width, frame.height
    bad = []
    for i, f in enumerate(feats):
        pr = []
        for r in f["geometry"]["coordinates"]:
            r = np.asarray(r, dtype=np.float64)
            x = R_MERC * np.radians(r[:, 0])
            y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(r[:, 1]) / 2))
            pr.append(np.stack([(x - frame.xmin) / frame.cell_m, (frame.ymax - y) / frame.cell_m], axis=1))
        poly = shapely.Polygon(pr[0], pr[1:])
        allp = pr[0]
        c0, c1 = max(int(np.floor(allp[:, 0].min())), 0), min(int(np.floor(allp[:, 0].max())), W - 1)
        r0, r1 = max(int(np.floor(allp[:, 1].min())), 0), min(int(np.floor(allp[:, 1].max())), H - 1)
        mine = buildings.index.pixels_of(i)
        if c0 > c1 or r0 > r1:
            ref = np.zeros(0, dtype=np.int64)
        else:
            cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
            cc, rr = cc.reshape(-1), rr.reshape(-1)
            boxes = shapely.box(cc, rr, cc + 1, rr + 1)
            hit = shapely.intersects(poly, boxes)
            ref = (rr[hit] * W + cc[hit]).astype(np.int64)
        if not np.array_equal(np.sort(ref), np.sort(mine.astype(np.int64))):
            bad.append(i)
    return bad


if __name__ == "__main__":
    from data_helpers import real_root, real_frame, real_buildings, real_hour
    storm = sys.argv[1] if len(sys.argv) > 1 else "25yr"
    hour = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    res = crosscheck(real_root(), real_frame(), real_buildings(), real_hour(storm, hour))
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()})
    print("pixel-set mismatches (ids):", res["pixel_set_differs"][:50])
    print("depth-code mismatches (ids):", res["depth_code_differs"][:50])
    from data_helpers import real_root as _r
    t0 = time.perf_counter()
    bad = crosscheck_shapely(real_root(), real_frame(), real_buildings())
    print(f"shapely exact reference: {len(bad)} buildings differ ({time.perf_counter() - t0:.1f}s)", bad[:50])
