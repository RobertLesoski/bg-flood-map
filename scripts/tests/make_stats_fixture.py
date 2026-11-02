"""Fixtures and INDEPENDENT expected values for fluid/test/stats-harness.js (fluid-stats.js).

  python make_stats_fixture.py build    [--storms 5yr 25yr] [--hours 6 13 24] [--polys 3000]
  python make_stats_fixture.py explain  <mismatches.json>     classify pixel differences with exact geometry
  python make_stats_fixture.py clean                          delete the fixture folder

Everything is computed without any code of fluid-stats.js:
  * lng/lat -> Web Mercator with PROJ (rasterio.warp.transform), not the JS formula;
  * building pixel sets: ONE rasterio.features.rasterize(all_touched=True) per building on a window of the real storm grid
    (rasterizing all buildings into one label raster would let neighbours steal each other's pixels);
  * per-building depth from the decoded R channel of the real hNN.rgba.gz (A = 0 pixels ignored);
  * roads: the lines are sampled every 1 ground metre per SEGMENT (the JS module samples every 10 m per LINE), ground metres =
    mercator metres * cos(latitude) with latitude from atan(exp(y/R)) (the JS module uses 1/cosh(y/R));
  * rand_polys.json: random polygons (stars, holes, slivers, tiny, edge-crossing) with the GDAL all_touched pixel sets.
Fixtures go to <APP>/fluid/test/fixture/ (served by serve.py) and are deleted again by `clean`.
"""
import argparse
import gzip
import json
import math
import os
import shutil
import sys

import numpy as np
import rasterio.features
import rasterio.warp
from affine import Affine
from shapely.geometry import Point, Polygon, box

APP = r"C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map"
EARTH_R = 6378137.0
MIN_DEPTH_M = 0.1
TIER_BINS = [0.1, 0.3, 0.6, 1.2]
ROAD_STEP_M = 1.0

# the hand-made grid of stats-harness.js (used for the random polygon fixture)
HAND = dict(width=24, height=24, cell_m=10.0, xmin=-9625000.0, ymax=4435240.0)


def fixture_dir(app):
    return os.path.join(app, "fluid", "test", "fixture")


def to_merc(xs, ys):
    """lng/lat arrays -> mercator arrays through PROJ."""
    X, Y = rasterio.warp.transform("EPSG:4326", "EPSG:3857", list(map(float, xs)), list(map(float, ys)))
    return np.asarray(X), np.asarray(Y)


def from_merc(x, y):
    lng = np.degrees(np.asarray(x) / EARTH_R)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / EARTH_R)) - np.pi / 2)
    return lng, lat


def decode_depth(R):
    return (R.astype(np.float64) / 255.0) ** 2 * 8.0


def decode_vel(G):
    return (G.astype(np.float64) - 128.0) / 127.0 * 4.0


def storm_dir(app, storm):
    return os.path.join(app, "layers", "flow", storm)


def check_readable(app, storm):
    d = storm_dir(app, storm)
    if os.path.isdir(os.path.join(app, "layers", ".flow.%s.building" % storm)):
        raise SystemExit("storm %s is being rebuilt, not reading it" % storm)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        raise SystemExit("no meta.json for %s" % storm)
    return json.load(open(os.path.join(d, "meta.json")))


def load_grid(app, storm, hour, W, H):
    raw = gzip.open(os.path.join(storm_dir(app, storm), "h%02d.rgba.gz" % hour), "rb").read()
    if len(raw) != W * H * 4:
        raise SystemExit("h%02d of %s has %d bytes, expected %d" % (hour, storm, len(raw), W * H * 4))
    return np.frombuffer(raw, np.uint8).reshape(H * W, 4)


# ------------------------------------------------------------------ buildings

def merc_polygons(features):
    """Per feature: list of (shell, holes) in mercator, from ONE bulk PROJ call."""
    xs, ys, spans = [], [], []
    for f in features:
        g = f["geometry"]
        parts = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        rec = []
        for rings in parts:
            rr = []
            for ring in rings:
                a = len(xs)
                xs.extend(p[0] for p in ring)
                ys.extend(p[1] for p in ring)
                rr.append((a, len(xs)))
            rec.append(rr)
        spans.append(rec)
    X, Y = to_merc(xs, ys)
    out = []
    for rec in spans:
        polys = []
        for rr in rec:
            rings = [np.column_stack([X[a:b], Y[a:b]]) for a, b in rr]
            polys.append((rings[0], rings[1:]))
        out.append(polys)
    return out


def rasterize_touched(polys, grid):
    """Pixel indices (ascending) all_touched by one building (list of shapely-style polygons), clipped to the grid."""
    W, H, cell, xmin, ymax = grid["width"], grid["height"], grid["cell_m"], grid["xmin"], grid["ymax"]
    allx = np.concatenate([p[0][:, 0] for p in polys])
    ally = np.concatenate([p[0][:, 1] for p in polys])
    c0 = max(int(math.floor((allx.min() - xmin) / cell)) - 1, 0)
    c1 = min(int(math.floor((allx.max() - xmin) / cell)) + 1, W - 1)
    r0 = max(int(math.floor((ymax - ally.max()) / cell)) - 1, 0)
    r1 = min(int(math.floor((ymax - ally.min()) / cell)) + 1, H - 1)
    if c0 > c1 or r0 > r1:
        return np.zeros(0, np.int32)
    aff = Affine(cell, 0, xmin + c0 * cell, 0, -cell, ymax - r0 * cell)
    shapes = [(Polygon(shell, holes), 1) for shell, holes in polys]
    img = rasterio.features.rasterize(shapes, out_shape=(r1 - r0 + 1, c1 - c0 + 1), transform=aff, all_touched=True, fill=0, dtype="uint8")
    rr, cc = np.nonzero(img)
    idx = (rr + r0).astype(np.int64) * W + (cc + c0)
    return np.sort(idx).astype(np.int32)


def csr(sets):
    offsets = np.zeros(len(sets) + 1, np.int32)
    for i, s in enumerate(sets):
        offsets[i + 1] = offsets[i] + len(s)
    pixels = np.concatenate(sets).astype(np.int32) if len(sets) else np.zeros(0, np.int32)
    return offsets, pixels


def building_hour(grid, offsets, pixels):
    """Per building: max R code, mean depth, speed at the first deepest pixel, over A != 0 pixels."""
    n = len(offsets) - 1
    maxR = np.zeros(n, np.int32)
    mean = np.zeros(n)
    speed = np.zeros(n)
    R = grid[:, 0]
    A = grid[:, 3]
    for i in range(n):
        s, e = offsets[i], offsets[i + 1]
        if s == e:
            continue
        p = pixels[s:e]
        ok = A[p] != 0
        if not ok.any():
            continue
        p = p[ok]
        r = R[p]
        mean[i] = decode_depth(r).mean()
        k = int(np.argmax(r))
        maxR[i] = r[k]
        if r[k] > 0:
            speed[i] = math.hypot(decode_vel(grid[p[k], 1]), decode_vel(grid[p[k], 2]))
    return maxR, mean, speed


def tier_counts(maxR):
    d = decode_depth(maxR)
    fl = d >= MIN_DEPTH_M
    out = {"flooded": int(fl.sum())}
    for name, lo, hi in zip(["shallow", "moderate", "deep", "severe"], TIER_BINS, TIER_BINS[1:] + [1e9]):
        out[name] = int(((d >= lo) & (d < hi)).sum())
    return out


# ---------------------------------------------------------------------- roads

def road_points(features, grid):
    """1 m ground samples per segment. Returns fid, weight, flat pixel (-1 outside), per-feature ground length."""
    xs, ys, rec = [], [], []
    for fi, f in enumerate(features):
        g = f.get("geometry")
        if not g:
            continue
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"] if g["type"] == "MultiLineString" else []
        for ln in lines:
            a = len(xs)
            xs.extend(p[0] for p in ln)
            ys.extend(p[1] for p in ln)
            rec.append((fi, a, len(xs)))
    X, Y = to_merc(xs, ys)
    fid, px, py, w = [], [], [], []
    for fi, a, b in rec:
        if b - a < 2:
            continue
        x0, y0, x1, y1 = X[a:b - 1], Y[a:b - 1], X[a + 1:b], Y[a + 1:b]
        lat = 2 * np.arctan(np.exp((y0 + y1) / 2 / EARTH_R)) - np.pi / 2
        glen = np.hypot(x1 - x0, y1 - y0) * np.cos(lat)
        n = np.where(glen > 0, np.maximum(1, np.ceil(glen / ROAD_STEP_M)), 0).astype(int)
        for k in np.nonzero(n)[0]:
            t = (np.arange(n[k]) + 0.5) / n[k]
            px.append(x0[k] + t * (x1[k] - x0[k]))
            py.append(y0[k] + t * (y1[k] - y0[k]))
            w.append(np.full(n[k], glen[k] / n[k]))
            fid.append(np.full(n[k], fi))
    fid = np.concatenate(fid)
    px = np.concatenate(px)
    py = np.concatenate(py)
    w = np.concatenate(w)
    col = np.floor((px - grid["xmin"]) / grid["cell_m"]).astype(np.int64)
    row = np.floor((grid["ymax"] - py) / grid["cell_m"]).astype(np.int64)
    inside = (col >= 0) & (col < grid["width"]) & (row >= 0) & (row < grid["height"])
    pix = np.where(inside, row * grid["width"] + col, -1)
    length = np.bincount(fid, weights=w, minlength=len(features))
    return fid, w, pix, length


def roads_hour(grid_px, fid, w, pix, length):
    ok = pix >= 0
    flooded = np.zeros(len(pix), bool)
    p = pix[ok]
    flooded[ok] = (grid_px[p, 3] != 0) & (decode_depth(grid_px[p, 0]) >= MIN_DEPTH_M)
    per = np.bincount(fid[flooded], weights=w[flooded], minlength=len(length))
    pct = np.where(length > 0, 100.0 * per / np.where(length > 0, length, 1), 0.0)
    tot = float(length.sum())
    return {"total_length_m": tot, "flooded_length_m": float(per.sum()), "pct": 100.0 * float(per.sum()) / tot, "per_feature_pct": [round(float(v), 4) for v in pct]}


# ------------------------------------------------------------- random polygons

def star(rng, cx, cy, rad, n):
    a0 = rng.random() * 2 * np.pi
    ang = a0 + (np.arange(n) + rng.random(n) * 0.5) / n * 2 * np.pi
    r = rad * (0.4 + 0.6 * rng.random(n))
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


def point_in(poly, x, y):
    return Polygon(poly).contains(Point(x, y))


def random_polys(n, seed=7):
    """Polygons in HAND grid coordinates: list of rings (first = shell). Mix of classes."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        kind = t % 6
        cx, cy = -3 + rng.random() * 30, -3 + rng.random() * 30
        if kind == 0:       # small stars
            rings = [star(rng, cx, cy, math.exp(math.log(0.1) + rng.random() * math.log(10)), 3 + int(rng.integers(0, 6)))]
        elif kind == 1:     # big stars, sometimes with a hole
            rad = math.exp(math.log(3) + rng.random() * math.log(40))
            rings = [star(rng, cx, cy, rad, 3 + int(rng.integers(0, 8)))]
            if rng.random() < 0.7:
                h = star(rng, cx, cy, rad * 0.25, 3 + int(rng.integers(0, 4)))
                if all(point_in(rings[0], x, y) for x, y in h):
                    rings.append(h)
        elif kind == 2:     # thin slivers (width 0.01 .. 0.2 px)
            a = rng.random() * 2 * np.pi
            L = 1 + rng.random() * 20
            wd = 0.01 + rng.random() * 0.19
            d = np.array([math.cos(a), math.sin(a)])
            nrm = np.array([-d[1], d[0]])
            p, q = np.array([cx, cy]), np.array([cx, cy]) + d * L
            rings = [np.array([p + nrm * wd / 2, q + nrm * wd / 2, q - nrm * wd / 2, p - nrm * wd / 2])]
        elif kind == 3:     # building-like rotated rectangles, 4..25 m side (0.4..2.5 px)
            a = rng.random() * np.pi
            w, h = 0.4 + rng.random() * 2.1, 0.4 + rng.random() * 2.1
            c, s = math.cos(a), math.sin(a)
            corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            rings = [corners @ np.array([[c, s], [-s, c]]) + [cx, cy]]
        elif kind == 4:     # tiny, far below a pixel
            rings = [star(rng, cx, cy, 0.02 + rng.random() * 0.1, 3 + int(rng.integers(0, 4)))]
        else:               # huge, covering most of the grid
            rings = [star(rng, 12, 12, 15 + rng.random() * 20, 4 + int(rng.integers(0, 8)))]
        if rng.random() < 0.5:
            rings = [r[::-1] for r in rings]
        out.append(rings)
    return out


def build_rand(app, n):
    g = dict(HAND, xmin=HAND["xmin"], ymax=HAND["ymax"])
    polys_grid = random_polys(n)
    ll, expect = [], []
    for rings in polys_grid:
        rl = []
        for r in rings:
            lng, lat = from_merc(g["xmin"] + r[:, 0] * g["cell_m"], g["ymax"] - r[:, 1] * g["cell_m"])
            rl.append([[float(a), float(b)] for a, b in zip(lng, lat)] + [[float(lng[0]), float(lat[0])]])
        ll.append(rl)
    # expected from the lng/lat the JS side will read, through PROJ
    sets = []
    for rl in ll:
        xs = [p[0] for r in rl for p in r]
        ys = [p[1] for r in rl for p in r]
        X, Y = to_merc(xs, ys)
        o, rings = 0, []
        for r in rl:
            rings.append(np.column_stack([X[o:o + len(r)], Y[o:o + len(r)]]))
            o += len(r)
        sets.append(rasterize_touched([(rings[0], rings[1:])], g))
    off, pix = csr(sets)
    return {"grid": g, "polys": ll, "offsets": off.tolist(), "pixels": pix.tolist()}


# ----------------------------------------------------------------------- main

def cmd_build(a):
    out = fixture_dir(a.app)
    os.makedirs(out, exist_ok=True)
    metas = {s: check_readable(a.app, s) for s in a.storms}
    m0 = metas[a.storms[0]]
    grid = dict(width=m0["width"], height=m0["height"], cell_m=m0["cell_m"], xmin=m0["bounds_merc"][0], ymax=m0["bounds_merc"][3])
    for s, m in metas.items():
        assert (m["width"], m["height"], m["cell_m"], m["bounds_merc"]) == (m0["width"], m0["height"], m0["cell_m"], m0["bounds_merc"]), "storms differ in grid"
    info = {"grid": grid, "built": {s: m.get("built") for s, m in metas.items()}, "hours": a.hours, "minDepthM": MIN_DEPTH_M}

    feats = json.load(open(os.path.join(a.app, "layers", "buildings.geojson"), encoding="utf-8"))["features"]
    print("buildings", len(feats))
    polys = merc_polygons(feats)
    sets = [rasterize_touched(p, grid) for p in polys]
    off, pix = csr(sets)
    info["nBuildings"] = len(feats)
    info["buildingPixels"] = int(len(pix))
    np.concatenate([[len(feats), len(pix)], off, pix]).astype("<i4").tofile(os.path.join(out, "bpx.bin"))
    print("pixel sets", len(pix), "pixels, empty:", int(sum(1 for s in sets if len(s) == 0)))

    rf = json.load(open(os.path.join(a.app, "layers", "roads.geojson"), encoding="utf-8"))["features"]
    fid, w, rpix, length = road_points(rf, grid)
    info["nRoads"] = len(rf)
    info["roadTotalLengthM"] = float(length.sum())
    print("roads", len(rf), "features", round(float(length.sum())), "m, samples", len(w))

    for s in a.storms:
        for h in a.hours:
            g = load_grid(a.app, s, h, grid["width"], grid["height"])
            maxR, mean, speed = building_hour(g, off, pix)
            rec = {"maxR": maxR.tolist(), "mean": [round(float(v), 7) for v in mean], "speed": [round(float(v), 7) for v in speed], "tiers": tier_counts(maxR)}
            json.dump(rec, open(os.path.join(out, "bld_%s_h%02d.json" % (s, h)), "w"))
            rr = roads_hour(g, fid, w, rpix, length)
            json.dump(rr, open(os.path.join(out, "roads_%s_h%02d.json" % (s, h)), "w"))
            print(s, h, "flooded buildings", rec["tiers"]["flooded"], "roads %.3f %%" % rr["pct"])
            del g
    for s in a.storms:   # a rebuild during the run would make the expected values stale
        again = check_readable(a.app, s)
        if again.get("built") != info["built"][s]:
            raise SystemExit("storm %s was rebuilt while the fixture was made" % s)

    rp = build_rand(a.app, a.polys)
    json.dump(rp, open(os.path.join(out, "rand_polys.json"), "w"))
    info["randPolys"] = len(rp["polys"])
    print("random polygons", len(rp["polys"]), "pixels", len(rp["pixels"]))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print("fixtures in", out)


def cmd_explain(a):
    """Mismatch records {kind: 'building'|'rand', id, jsOnly: [pixel], gdalOnly: [pixel]} -> exact-geometry classification."""
    recs = json.load(open(a.file))
    out = fixture_dir(a.app)
    info = json.load(open(os.path.join(out, "info.json")))
    grids = {"building": info["grid"], "rand": dict(HAND)}
    feats = None
    rp = None
    classes = {}
    for r in recs:
        g = grids[r["kind"]]
        if r["kind"] == "building":
            if feats is None:
                feats = json.load(open(os.path.join(a.app, "layers", "buildings.geojson"), encoding="utf-8"))["features"]
            polys = merc_polygons([feats[r["id"]]])[0]
        else:
            if rp is None:
                rp = json.load(open(os.path.join(out, "rand_polys.json")))
            rl = rp["polys"][r["id"]]
            xs = [p[0] for ring in rl for p in ring]
            ys = [p[1] for ring in rl for p in ring]
            X, Y = to_merc(xs, ys)
            o, rings = 0, []
            for ring in rl:
                rings.append(np.column_stack([X[o:o + len(ring)], Y[o:o + len(ring)]]))
                o += len(ring)
            polys = [(rings[0], rings[1:])]
        cell = g["cell_m"]
        # polygon in grid pixel coordinates (row down)
        gp = []
        for shell, holes in polys:
            conv = lambda a_: np.column_stack([(a_[:, 0] - g["xmin"]) / cell, (g["ymax"] - a_[:, 1]) / cell])
            gp.append(Polygon(conv(shell), [conv(h) for h in holes]))
        from shapely.ops import unary_union
        poly = unary_union([p.buffer(0) for p in gp])
        for who, pxs in (("js_only", r["jsOnly"]), ("gdal_only", r["gdalOnly"])):
            for p in pxs:
                c, rw = p % g["width"], p // g["width"]
                sq = box(c, rw, c + 1, rw + 1)
                area = poly.intersection(sq).area
                dist = poly.distance(sq)
                if area > 1e-9:
                    cls = "polygon overlaps the pixel with area %s px2" % ("> 1e-3" if area > 1e-3 else "<= 1e-3")
                elif dist < 1e-6:
                    cls = "polygon only touches the pixel square (distance %.1e px, overlap area 0)" % dist
                else:
                    cls = "polygon does NOT reach the pixel (distance %.3g px)" % dist
                key = (who, cls)
                classes.setdefault(key, []).append((r["kind"], r["id"], p, round(area, 8), round(dist, 8)))
    for (who, cls), items in sorted(classes.items()):
        print("%-9s %5d  %s" % (who, len(items), cls))
        for it in items[:3]:
            print("            e.g.", it)


def cmd_clean(a):
    d = fixture_dir(a.app)
    if os.path.isdir(d):
        shutil.rmtree(d)
        print("deleted", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default=APP)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--storms", nargs="+", default=["5yr", "25yr"])
    b.add_argument("--hours", nargs="+", type=int, default=[6, 13, 24])
    b.add_argument("--polys", type=int, default=3000)
    b.set_defaults(fn=cmd_build)
    e = sub.add_parser("explain")
    e.add_argument("file")
    e.set_defaults(fn=cmd_explain)
    c = sub.add_parser("clean")
    c.set_defaults(fn=cmd_clean)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
