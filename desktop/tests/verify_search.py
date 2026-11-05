"""Independent verification of the search features against the real data (a tool, not a pytest test; the quick version of the
same checks is in test_search.py).  Everything the app computes is recomputed here with DIFFERENT code:

    python desktop/tests/verify_search.py [--n 220] [--seed 7] [--data <folder>]

 1. address index, both domains: random addresses (and flooded ones) - the app's StormRow numbers vs a line-by-line decode of the raw
    gzip JSON (the builder's `flood_of` rules), address text, coordinates, typeahead round trip.
 2. county depth tiles: random building addresses x all storms - the app's footprint maximum (shapely boxes) vs rasterio.mask
    (GDAL all_touched) on the polygon reprojected by rasterio.warp.transform_geom; lots/points vs rasterio index + read.
 3. all county facilities x all storms - the app's point samples vs rasterio index + read at coordinates from rasterio.warp.
 4. the comparison between tiles and index (how often they disagree) - reported, not asserted.
 5. road lists: every way of every county storm file has geometry; the flooded-km totals of the file header vs the rows.
Exit code 0 when 1-3 and 5 have no unexplained mismatch.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bgv.search import addrindex as A          # noqa: E402
from bgv.search import assess as asm           # noqa: E402
from bgv.search import facilities as F         # noqa: E402
from bgv.search import roadrisk as RR          # noqa: E402
from bgv.search import tiles as T              # noqa: E402

DEFAULT = r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map"


def check_index(root, domain, n, rnd):
    path = A.index_path(root, domain)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    idx = A.AddressIndex.load(path)
    ref = {}
    for s in raw["storms"]:
        fl, out, a = raw["flood"].get(s["key"], []), {}, 0
        for t in range(0, len(fl), 5):
            a += fl[t]
            out[a] = tuple(fl[t + 1:t + 5])
        ref[s["key"]] = out
    pick = rnd.sample(range(idx.n), n)
    wet = [i for i in range(idx.n) if idx.depth_cm[idx.storm_keys[-1]][i] > 0]
    pick += rnd.sample(wet, min(n // 2, len(wet)))
    starts = raw["streets"]["start"]
    bad = 0
    for i in pick:
        si = bisect.bisect_right(starts, i) - 1
        rec = idx.record(i)
        ok = rec.address == (raw["num"][i] + " " + raw["streets"]["disp"][si]).strip()
        ok &= abs(rec.lng - (raw["origin"][0] + raw["x"][i] / raw["scale"])) < 1e-9 and abs(rec.lat - (raw["origin"][1] + raw["y"][i] / raw["scale"])) < 1e-9
        for row in idx.storm_rows(i):
            ok &= (row.depth_cm, row.tier, row.first_hour, row.hours) == ref[row.key].get(i, (0, 0, 0, 0))
        if rec.num:
            ok &= any(h.kind == "address" and h.label == rec.address for h in idx.search(rec.address, 60))
        bad += 0 if ok else 1
    print(f"[1] index {domain}: {len(pick)} addresses ({len(wet)} flooded in the last storm available), mismatches: {bad}")
    return bad, idx


def load_county_buildings(root):
    """The cached OSM county scene the viewer uses (osm_id -> polygon coordinates), read with plain json."""
    cands = sorted((Path(__file__).resolve().parents[1] / "cache" / "osm").glob("*/buildings.geojson"))
    if not cands:
        cands = sorted((Path(r"C:\Users\Minec\Documents\BG_3D_Map\desktop\cache\osm")).glob("*/buildings.geojson"))
    if not cands:
        return None
    doc = json.loads(cands[-1].read_text(encoding="utf-8"))
    return {int(f["properties"]["osm_id"]): f["geometry"] for f in doc["features"] if (f.get("properties") or {}).get("osm_id") is not None}


def check_tiles(root, idx, n, rnd):
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import transform, transform_geom
    ts = T.TileSet.discover(root, "county")
    if ts is None:
        print("[2] county tiles: not found - skipped")
        return 0, None
    geoms = load_county_buildings(root)
    fp = asm.Footprints({k: (g["coordinates"] if g["type"] == "Polygon" else g["coordinates"]) for k, g in geoms.items()}, "verify") if geoms else None
    bld = [i for i in range(idx.n) if idx.kind[i] == 0 and idx.bid[i] in (geoms or {})]
    lots = [i for i in range(idx.n) if idx.kind[i] == 1]
    pick_b = rnd.sample(bld, n)
    pick_l = rnd.sample(lots, min(30, len(lots)))
    storms = ts.storms
    t0 = time.perf_counter()
    mism_fp = mism_pt = 0
    maxdiff = 0
    cells = 0
    dsets = {s: rasterio.open(ts.paths[s]) for s in storms}
    try:
        for i in pick_b:
            rec = idx.record(i)
            g = geoms[rec.bid]
            geom3857 = transform_geom("EPSG:4326", "EPSG:3857", g)
            poly = g["coordinates"]
            for s in storms:
                got, ncell = ts.footprint_max(s, poly)
                try:
                    out, _ = mask(dsets[s], [geom3857], crop=True, all_touched=True, nodata=65535, filled=True)
                    v = out[0][out[0] != 65535]
                    exp = int(v.max()) if v.size else -1
                except ValueError:
                    exp = -1
                cells += 1
                if got != exp:
                    mism_fp += 1
                    maxdiff = max(maxdiff, abs(got - exp))
        for i in pick_l:
            rec = idx.record(i)
            (x,), (y,) = transform("EPSG:4326", "EPSG:3857", [rec.lng], [rec.lat])
            for s in storms:
                got = int(ts.sample_points(s, [rec.lng], [rec.lat])[0])
                ds = dsets[s]
                r, c = ds.index(x, y)
                exp = -1
                if 0 <= r < ds.height and 0 <= c < ds.width:
                    v = int(ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
                    exp = -1 if v == 65535 else v
                if got != exp:
                    mism_pt += 1
    finally:
        for d in dsets.values():
            d.close()
    print(f"[2] tiles: {len(pick_b)} building footprints x {len(storms)} storms = {cells} footprint maxima, mismatches vs rasterio.mask(all_touched): "
          f"{mism_fp} (max diff {maxdiff} cm); {len(pick_l)} lots x {len(storms)} = {len(pick_l) * len(storms)} point samples, mismatches: {mism_pt}  [{time.perf_counter() - t0:.0f}s]")
    # tiles vs index agreement on the same addresses
    dis = tot = tier_diff = 0
    deltas = []
    for i in pick_b:
        a = asm.assess(idx, i, ts, fp)
        for r in a.rows:
            if r.comparison is not None:
                tot += 1
                dis += r.comparison.disagree
                tier_diff += r.comparison.tier_differs
                deltas.append(r.comparison.delta_cm)
    d = np.array(deltas)
    print(f"[4] tiles vs index on {tot} (address, storm) pairs: material disagreement (flooded state flips or >= 10 cm apart) {dis} ({100.0 * dis / max(tot, 1):.1f} %), "
          f"tier differs {tier_diff} ({100.0 * tier_diff / max(tot, 1):.1f} %); tile - index: median {np.median(d):+.0f} cm, mean {d.mean():+.1f} cm, "
          f"p5 {np.percentile(d, 5):+.0f}, p95 {np.percentile(d, 95):+.0f}")
    return mism_fp + mism_pt, ts


def check_facilities(root, ts):
    import rasterio
    from rasterio.warp import transform
    if ts is None:
        return 0
    fs = F.FacilitySet.load(F.facilities_path(root, "county"))
    fs.attach_tiles(ts)
    xs, ys = transform("EPSG:4326", "EPSG:3857", list(fs.lng), list(fs.lat))
    bad = 0
    n = 0
    for s in ts.storms:
        with rasterio.open(ts.paths[s]) as ds:
            for k in range(fs.n):
                r, c = ds.index(xs[k], ys[k])
                exp = -1
                if 0 <= r < ds.height and 0 <= c < ds.width:
                    v = int(ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
                    exp = -1 if v == 65535 else v
                n += 1
                bad += int(fs.d_tiles[s][k] != exp)
    print(f"[3] facilities: {fs.n} x {len(ts.storms)} storms = {n} point samples, mismatches vs rasterio: {bad}")
    for s in ("2yr", "100yr", "ev4"):
        a, b = fs.d_index[s], fs.d_tiles[s]
        dis = sum(T.compare(int(x), int(y)).disagree for x, y in zip(a, b) if y >= 0)
        print(f"    {s}: flooded by index {int((a >= 10).sum())}, by tiles {int((b >= 10).sum())}, material disagreement {dis} of {int((b >= 0).sum())}; "
              f"no tile data at {int((b < 0).sum())}")
    return bad


def check_roads(root):
    bad = 0
    g = RR.RoadGeometry.load(root)
    for dom in ("city", "county"):
        folder = Path(root) / "layers" / "roads_flooded" / dom
        for f in sorted(folder.glob("*.json")):
            if f.name == "index.json":
                continue
            doc = json.loads(f.read_text(encoding="utf-8"))
            rows = doc["rows"]
            miss = sum(1 for r in rows if g.coords(r[0]) is None)
            fk = sum(r[4] * r[6] / 100.0 for r in rows) / 1000.0
            tot_ok = abs(fk - doc["flooded_km"]) <= max(0.5, 0.02 * doc["flooded_km"])
            if miss or not tot_ok:
                bad += 1
                print(f"    {dom}/{f.name}: ways without geometry {miss}; flooded km rows {fk:.1f} vs header {doc['flooded_km']}")
    print(f"[5] road lists: all files checked, files with a problem: {bad}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", default=DEFAULT)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    total = 0
    b1, _ = check_index(a.data, "city", a.n, rnd)
    b2, idx = check_index(a.data, "county", a.n, rnd)
    total += b1 + b2
    b3, ts = check_tiles(a.data, idx, a.n, rnd)
    total += b3
    total += check_facilities(a.data, ts)
    total += check_roads(a.data)
    print("RESULT:", "PASS" if total == 0 else f"{total} unexplained mismatches")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
