r"""
make_real_expected.py - ground truth for cog/test/real-check.html: random lng/lat points (deep, shallow, dry, near the edge, outside) and the value rasterio/PROJ reads
from the REAL downloaded COGs (layers/cog/county/<storm>), so the browser reader can be compared with an independent decoder on production files.

    python scripts/make_real_expected.py --app "<web app folder>" --storms "2yr 100yr ev4" --n 80 --out "<web app folder>/cog/test/real_expected.json"
"""
import argparse, glob, json, os, re
import numpy as np
import rasterio
from rasterio.warp import transform as wt

NODATA = 65535


def hour_file(folder, h):
    best = None
    for p in glob.glob(os.path.join(folder, "depth_*s.tif")):
        sec = int(re.search(r"depth_(\d+)s", p).group(1))
        if round(sec / 3600.0) == h and (best is None or abs(sec - h * 3600) < best[0]):
            best = (abs(sec - h * 3600), p)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True); ap.add_argument("--storms", default="2yr 100yr ev4"); ap.add_argument("--n", type=int, default=80); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(99)
    out = {}
    for s in a.storms.split():
        folder = os.path.join(a.app, "layers", "cog", "county", s)
        out[s] = {}
        for which in ("max", 12):
            p = os.path.join(folder, "max_depth.tif") if which == "max" else hour_file(folder, which)
            with rasterio.open(p) as d:
                f = 8
                ov = d.read(1, out_shape=(d.height // f, d.width // f))
                valid = ov != NODATA
                groups = {"deep": np.argwhere(valid & (ov >= 100)), "shallow": np.argwhere(valid & (ov >= 1) & (ov < 100)), "dry": np.argwhere(valid & (ov == 0)), "outside": np.argwhere(~valid)}
                full = d.read(1)
                pts = []
                for kind, arr in groups.items():
                    if not len(arr):
                        continue
                    for r, c in arr[rng.integers(0, len(arr), a.n // 4)]:
                        col, row = c * f + rng.uniform(0, f), r * f + rng.uniform(0, f)          # a random point inside the 8 px block (not a pixel centre)
                        x, y = d.transform * (col, row)
                        lng, lat = wt("EPSG:3857", "EPSG:4326", [x], [y])
                        ci, ri = int(np.floor(col)), int(np.floor(row))
                        pts.append({"kind": kind, "lng": lng[0], "lat": lat[0], "v": int(full[ri, ci])})
                for _ in range(6):                                                            # beyond the file edges
                    x = d.bounds.left - rng.uniform(10, 500); y = d.bounds.top - rng.uniform(0, 1e4)
                    lng, lat = wt("EPSG:3857", "EPSG:4326", [x], [y]); pts.append({"kind": "beyond", "lng": lng[0], "lat": lat[0], "v": None})
                out[s][str(which)] = {"file": os.path.basename(p), "points": pts}
    json.dump(out, open(a.out, "w"))
    print({s: {k: len(v["points"]) for k, v in d.items()} for s, d in out.items()})


if __name__ == "__main__":
    main()
