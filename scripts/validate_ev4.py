#!/usr/bin/env python
"""Validate model event ev4 (April 2025 rain block) against observed flood evidence.

Reproduce:  python scripts/validate_ev4.py [--app "<app folder>"] [--out docs]
Reads   <app>/layers/validation/ev4_observations.json  (hand-collected, see docs/VALIDATION_EV4.md)
        <app>/layers/cog/county/ev4/max_depth.tif       (EPSG:3857, uint16 cm, 65535 = nodata)
        <app>/layers/county/roads.geojson.gz, layers/county/address_index.json.gz,
        <app>/layers/roads_flooded/county/ev4.json, raw USGS files
Writes  <app>/layers/validation/ev4_scores.json and <out>/VALIDATION_EV4_map.png
Never calls the network. Pure functions (merc, sample_line, depth_at, road_stats, wilson,
contingency, auc) are unit tested in scripts/tests/test_validate_ev4.py.
"""
import argparse
import gzip
import json
import math
import os
import random

import numpy as np

R = 6378137.0
THRESHOLDS = (0.10, 0.30)
NODATA = 65535
# City box used for the control set / building counts (lon_min, lat_min, lon_max, lat_max).
CITY_BBOX = (-86.56, 36.90, -86.33, 37.07)
# Model rain by hour (inches, hour 0 = 2025-04-03 13:00 UTC), from STORM_DATA_SOURCES.md event 4.
MODEL_RAIN_IN = [0.19, 0.06, 0.36, 0.12, 0.43, 0.40, 0.03, 0.12, 0.02, 0.16, 0.07, 0.05, 0.01, 0.00,
                 0.01, 0.00, 0.00, 0.52, 0.09, 0.00, 0.10, 0.99, 1.07, 0.17]


# ------------------------------------------------------------------ pure helpers
def merc(lon, lat):
    """Spherical Web Mercator (EPSG:3857) from degrees; arrays or scalars."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def sample_line(coords, step=6.25):
    """Points every `step` (same units as coords) along a polyline [(x, y), ...], including both ends.
    Returns (xs, ys, seg_len_total)."""
    pts = np.asarray(coords, float)
    if len(pts) == 1:
        return pts[:, 0], pts[:, 1], 0.0
    d = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total == 0:
        return pts[:1, 0], pts[:1, 1], 0.0
    n = max(2, int(math.ceil(total / step)) + 1)
    s = np.linspace(0, total, n)
    return np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1]), total


def depth_at(arr, origin_x, origin_y, px, xs, ys, nodata=NODATA, win=0):
    """Depth in metres at mercator points from a uint16-cm array (north-up, origin = top-left corner).
    win = radius in pixels: the maximum of the (2 win + 1)^2 neighbourhood is used. NaN where the
    point (or all of the neighbourhood) is outside the array or nodata."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    col = np.floor((xs - origin_x) / px).astype(int)
    row = np.floor((origin_y - ys) / px).astype(int)
    h, w = arr.shape
    out = np.full(xs.shape, np.nan)
    for i in range(xs.size):
        r0, r1 = max(row.flat[i] - win, 0), min(row.flat[i] + win + 1, h)
        c0, c1 = max(col.flat[i] - win, 0), min(col.flat[i] + win + 1, w)
        if r0 >= r1 or c0 >= c1:
            continue
        blk = arr[r0:r1, c0:c1]
        blk = blk[blk != nodata]
        if blk.size:
            out.flat[i] = blk.max() / 100.0
    return out


def road_stats(vals, thresholds=THRESHOLDS):
    """max depth and fraction of samples (= fraction of length) at/above each threshold; NaN ignored."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return dict(n=0, max=None, **{"frac_%g" % t: None for t in thresholds})
    d = dict(n=int(v.size), max=float(v.max()))
    for t in thresholds:
        d["frac_%g" % t] = float((v >= t).mean())
    return d


def flagged(stats, t, min_frac=0.0):
    """Model 'flags flooded' at threshold t: any sample >= t (min_frac = 0) or at least min_frac of length."""
    if not stats or stats.get("max") is None:
        return False
    if min_frac <= 0:
        return stats["max"] >= t
    return stats["frac_%g" % t] >= min_frac


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n; (nan, nan) if n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def contingency(pred, truth):
    """2x2 counts from boolean lists: hits, misses, false_alarms, correct_negatives."""
    h = m = fa = cn = 0
    for p, t in zip(pred, truth):
        if t and p:
            h += 1
        elif t and not p:
            m += 1
        elif p:
            fa += 1
        else:
            cn += 1
    return dict(hits=h, misses=m, false_alarms=fa, correct_negatives=cn)


def auc(pos, neg):
    """P(random observed value > random control value), ties count 1/2 (Mann-Whitney)."""
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


# ------------------------------------------------------------------ data access
class Depth:
    def __init__(self, path):
        import rasterio
        with rasterio.open(path) as ds:
            self.arr = ds.read(1)
            self.ox, self.oy = ds.transform.c, ds.transform.f
            self.px = ds.transform.a
            self.crs = str(ds.crs)
            self.nodata = ds.nodata
        assert "3857" in self.crs and self.nodata == NODATA, (self.crs, self.nodata)

    def at(self, xs, ys, win=0):
        return depth_at(self.arr, self.ox, self.oy, self.px, xs, ys, win=win)

    def line(self, lonlat):
        x, y = merc([c[0] for c in lonlat], [c[1] for c in lonlat])
        xs, ys, tot = sample_line(list(zip(x, y)), self.px)
        return self.at(xs, ys), tot


def norm(s):
    return " ".join(s.lower().replace("-", " ").split())


def load_roads(app):
    d = json.load(gzip.open(os.path.join(app, "layers", "county", "roads.geojson.gz")))
    ways = {}
    for f in d["features"]:
        nm = f["properties"].get("name") or ""
        if not nm:
            continue
        g = f["geometry"]
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        ways.setdefault(norm(nm), []).append(dict(osm_id=f["properties"]["osm_id"], name=nm, lines=lines,
                                                  hw=f["properties"].get("highway")))
    return ways


def in_bbox(lonlat, bbox=CITY_BBOX):
    xs = [c[0] for c in lonlat]
    ys = [c[1] for c in lonlat]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]


def score_named_road(depth, ways, restrict_bbox):
    """Pool samples from all ways with the same name (optionally only those with midpoint in the city
    box). Returns stats dict plus metadata."""
    vals, length, n_ways, ids = [], 0.0, 0, []
    for w in ways:
        for ln in w["lines"]:
            if restrict_bbox and not in_bbox(ln):
                continue
            v, tot = depth.line(ln)
            vals.append(v)
            length += tot
            n_ways += 1
            ids.append(w["osm_id"])
    if not vals:
        return None
    st = road_stats(np.concatenate(vals))
    st.update(length_m=round(length, 1), n_ways=n_ways, osm_ids=ids)
    return st


def model_roads_flag(app, ids):
    """Flooded share of length from roads_flooded/ev4.json (built from the 20 m flow grid)."""
    p = os.path.join(app, "layers", "roads_flooded", "county", "ev4.json")
    d = json.load(open(p))
    rows = {r[0]: r for r in d["rows"]}
    tot = fl = 0.0
    found = 0
    for i in ids:
        r = rows.get(i)
        if r:
            found += 1
            tot += r[4]
            fl += r[4] * r[6] / 100.0
    return dict(rows_found=found, flooded_share=(fl / tot if tot else None))


def usgs_summary(path):
    d = json.load(open(path))
    out = {}
    for ts in d["value"]["timeSeries"]:
        name = "stage_ft" if "Gage" in ts["variable"]["variableName"] else "cfs"
        v = [(x["dateTime"], float(x["value"])) for x in ts["values"][0]["value"] if x["value"] not in ("", "-999999")]
        pk = max(v, key=lambda a: a[1])
        out[name] = dict(first=v[0], peak=pk, last=v[-1])
        if name == "stage_ft":
            base = v[0][1]
            rise = next(t for t, s in v if s >= base + 5)
            out["rise_5ft_above_start"] = rise
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default=r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs"))
    ap.add_argument("--n-control", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    app = a.app
    vdir = os.path.join(app, "layers", "validation")
    obs = json.load(open(os.path.join(vdir, "ev4_observations.json")))["observations"]
    depth = Depth(os.path.join(app, "layers", "cog", "county", "ev4", "max_depth.tif"))
    ways = load_roads(app)
    res = dict(thresholds=list(THRESHOLDS), city_bbox=CITY_BBOX)

    # (a) observed roads
    rows = []
    obs_names = set()
    for o in obs:
        if o["type"] != "road_closed":
            continue
        row = dict(id=o["id"], name=o["name"], model_name=o.get("model_name"), date=o["date"], scope=o.get("scope"),
                   match_quality=o.get("match_quality", "ok"), url=o["url"])
        mn = o.get("model_name")
        if mn and norm(mn) in ways:
            st = score_named_road(depth, ways[norm(mn)], restrict_bbox=(o.get("scope") == "city_corridor"))
            if st:
                row.update(st)
                row["roads_flooded"] = model_roads_flag(app, st["osm_ids"])
                obs_names.add(norm(mn))
        rows.append(row)
    res["observed_roads"] = rows
    scored = [r for r in rows if r.get("n")]
    good = [r for r in scored if r["match_quality"] == "ok"]

    # (b) control: random named roads in the city box, not observed (truth UNKNOWN)
    rnd = random.Random(a.seed)
    cand = []
    for nm, ws in ways.items():
        if nm in obs_names:
            continue
        lines = [ln for w in ws for ln in w["lines"] if in_bbox(ln)]
        if lines and ws[0]["hw"] not in ("motorway", "motorway_link", "trunk"):
            cand.append(nm)
    cand.sort()
    pick = rnd.sample(cand, min(a.n_control, len(cand)))
    ctrl = []
    for nm in pick:
        st = score_named_road(depth, ways[nm], restrict_bbox=True)
        if st and st["n"] >= 3:
            st["name"] = ways[nm][0]["name"]
            ctrl.append(st)
    res["control_roads"] = dict(n_candidates=len(cand), seed=a.seed, rows=ctrl)

    summ = {}
    for t in THRESHOLDS:
        for label, mf in (("any", 0.0), ("20pct", 0.2)):
            o_flags = [flagged(r, t, mf) for r in good]
            c_flags = [flagged(r, t, mf) for r in ctrl]
            k, n = sum(o_flags), len(o_flags)
            kc, nc = sum(c_flags), len(c_flags)
            summ["t%g_%s" % (t, label)] = dict(
                observed_hits=k, observed_n=n, hit_rate=k / n if n else None, hit_ci95=wilson(k, n),
                control_flagged=kc, control_n=nc, control_rate=kc / nc if nc else None, control_ci95=wilson(kc, nc))
    summ["auc_max_depth"] = auc([r["max"] for r in good], [r["max"] for r in ctrl])
    summ["median_max_observed"] = float(np.median([r["max"] for r in good])) if good else None
    summ["median_max_control"] = float(np.median([r["max"] for r in ctrl])) if ctrl else None
    # length sensitivity: long roads have more chances to cross some deep cell, so repeat for roads < 2000 m
    sh_o = [r for r in good if r["length_m"] < 2000]
    sh_c = [r for r in ctrl if r["length_m"] < 2000]
    sens = dict(observed_n=len(sh_o), control_n=len(sh_c), auc=auc([r["max"] for r in sh_o], [r["max"] for r in sh_c]))
    for t in THRESHOLDS:
        for label, mf in (("any", 0.0), ("20pct", 0.2)):
            sens["t%g_%s" % (t, label)] = dict(
                observed=sum(flagged(r, t, mf) for r in sh_o), control=sum(flagged(r, t, mf) for r in sh_c))
    summ["short_roads_lt2000m"] = sens
    res["road_summary"] = summ

    # points (LSR) and weak not-flooded controls
    pts = []
    for o in obs:
        if o["type"] == "flooded_property" and o["lat"] is not None:
            x, y = merc(o["lng"], o["lat"])
            pts.append(dict(id=o["id"], name=o["name"], date=o["date"], lat=o["lat"], lng=o["lng"],
                            depth_point=float(depth.at([x], [y])[0]),
                            max_within_100m=float(depth.at([x], [y], win=16)[0])))
    res["points"] = pts
    # random point baseline in the city box
    bx = []
    for _ in range(300):
        lon = rnd.uniform(CITY_BBOX[0], CITY_BBOX[2])
        lat = rnd.uniform(CITY_BBOX[1], CITY_BBOX[3])
        x, y = merc(lon, lat)
        bx.append(float(depth.at([x], [y], win=16)[0]))
    bxv = np.array([v for v in bx if not np.isnan(v)])
    res["random_points_100m_max"] = dict(n=int(bxv.size), frac_ge_0_10=float((bxv >= .1).mean()),
                                         frac_ge_0_30=float((bxv >= .3).mean()))

    # (c) gauges
    g = {}
    for sid in ("03314500", "03314000"):
        p = os.path.join(vdir, "raw", "usgs_%s.json" % sid)
        if os.path.exists(p):
            g[sid] = usgs_summary(p)
    res["gauges"] = g
    res["model_rain_in"] = MODEL_RAIN_IN
    res["model_rain_total_in"] = round(sum(MODEL_RAIN_IN), 2)

    # (d) buildings: records of kind 0 in the city box, max depth within 3x3 pixels of the point
    ai = json.load(gzip.open(os.path.join(app, "layers", "county", "address_index.json.gz")))
    ox, oy = ai["origin"]
    sc = ai["scale"]
    xs = np.array(ai["x"]) / sc + ox
    ys = np.array(ai["y"]) / sc + oy
    kind = np.array(ai["kind"])
    m = (kind == 0) & (xs >= CITY_BBOX[0]) & (xs <= CITY_BBOX[2]) & (ys >= CITY_BBOX[1]) & (ys <= CITY_BBOX[3])
    mx, my = merc(xs[m], ys[m])
    bd = depth.at(mx, my, win=1)
    bd = bd[~np.isnan(bd)]
    res["buildings_city_box"] = dict(n_records=int(m.sum()), n_with_depth=int(bd.size),
                                     ge_0_10=int((bd >= .1).sum()), ge_0_30=int((bd >= .3).sum()),
                                     ge_1_0=int((bd >= 1.0).sum()), observed_damaged_structures=196)

    with open(os.path.join(vdir, "ev4_scores.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    make_map(depth, res, ways, os.path.join(a.out, "VALIDATION_EV4_map.png"), obs_names)
    print(json.dumps({k: res[k] for k in ("road_summary", "random_points_100m_max", "buildings_city_box")}, indent=1, default=float))


def make_map(depth, res, ways, path, obs_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bb = CITY_BBOX
    x0, y0 = merc(bb[0], bb[3])
    x1, y1 = merc(bb[2], bb[1])
    c0, r0 = int((x0 - depth.ox) / depth.px), int((depth.oy - y0) / depth.px)
    c1, r1 = int((x1 - depth.ox) / depth.px), int((depth.oy - y1) / depth.px)
    sub = depth.arr[r0:r1, c0:c1][::2, ::2].astype(float) / 100.0
    sub[depth.arr[r0:r1, c0:c1][::2, ::2] == NODATA] = np.nan
    fig, ax = plt.subplots(figsize=(10, 9), dpi=130)
    ext = (x0, x1, y1, y0)
    ax.imshow(np.where(sub < 0.1, np.nan, np.minimum(sub, 2.0)), extent=ext, cmap="Blues", vmin=0.1, vmax=2.0)
    for nm in obs_names:
        for w in ways[nm]:
            for ln in w["lines"]:
                if in_bbox(ln) or True:
                    x, y = merc([c[0] for c in ln], [c[1] for c in ln])
                    ax.plot(x, y, color="#c0392b", lw=1.6)
    for p in res["points"]:
        x, y = merc(p["lng"], p["lat"])
        ax.plot(x, y, "o", ms=8, mfc="#f39c12", mec="k")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title("ev4 model max depth (blue, >= 0.10 m) vs observed flooded roads (red) and NWS reports (orange)\n"
                 "Observed roads are drawn whole (named road, not just the reported segment)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)


if __name__ == "__main__":
    main()
