r"""
build_infrastructure.py - critical facilities + flooded-road lists from OpenStreetMap and the flow grids.

    python scripts/build_infrastructure.py                  # city   -> layers/infrastructure.geojson + layers/roads_flooded/city/<storm>.json
    python scripts/build_infrastructure.py --domain county  # county -> layers/county/infrastructure.geojson + layers/roads_flooded/county/<storm>.json
    python scripts/build_infrastructure.py --domain both --hifld layers/hifld_warren.geojson   # also merge the HIFLD points (fetch_hifld.py)

infrastructure.geojson: Point features, properties
    name, type (hospital clinic fire_station police school shelter assembly_point community_centre townhall ems), source (osm | hifld),
    osm_id ('n123' node / 'w456' way), lng, lat,
    d_<storm>      max flood depth in cm at the facility (flow-grid maximum over all hours; a node = its pixel, a way/campus = max over
                   the outline's pixels, all-touched; duplicates of one named facility within 200 m are merged: max over all parts)
    flooded_in     comma list of the storms with depth >= 0.1 m ("10yr,25yr,50yr,100yr"), '' when none
    first_flooded  smallest design storm (2yr..100yr) that floods it, or null    ("flooded in 25yr" -> first_flooded == "25yr")
    max_cm         deepest d_<storm> over all storms
roads_flooded/<domain>/<storm>.json: OSM driveable road ways that have water at >= 0.02 m on at least one 10 m sample, per storm:
    columns [id, name, hw, evac, len_m, wet_pct, flooded_pct, max_cm]; hw = index into "classes"; evac = 0 none, 1 tagged in OSM
    (evacuation tags), 2 major class (motorway/trunk/primary, treated as an evacuation corridor); wet_pct = % of the way's length
    wet (>= 0.02 m), flooded_pct = % >= 0.1 m (the apps' flooded-road rule). Sampling = flowquery.py (10 m, midpoints).
Needs the OSM extract (kentucky-latest.osm.pbf) and the flow grids of the domain; no network.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import numpy as np
import shapely
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flowquery as fq      # noqa: E402
import geodata as gd        # noqa: E402
import osm_pbf              # noqa: E402

MERGE_RADIUS_M = 200.0
HIFLD_RADIUS_M = 150.0
MAJOR_CLASSES = ("motorway", "motorway_link", "trunk", "primary", "primary_link")
CLASSES = ("motorway", "motorway_link", "trunk", "primary", "primary_link", "secondary", "tertiary", "unclassified", "residential", "other")
DESIGN = ("2yr", "5yr", "10yr", "25yr", "50yr", "100yr")


def _m_per_deg(lat):
    return 111320.0 * np.cos(np.radians(lat)), 110540.0


def dedupe_pois(pois, radius_m=MERGE_RADIUS_M):
    """Merge parts of one named facility (type + name equal, within radius_m): keep the part with the largest outline as the
    representative, list the others as members. Unnamed facilities are never merged. -> list of dict(main=poi, members=[poi...])."""
    groups = []
    by_key = {}
    for p in pois:
        if not p.get("name"):
            groups.append({"main": p, "members": [p]})
            continue
        k = (p["type"], p["name"].strip().lower())
        placed = False
        for g in by_key.get(k, []):
            kx, ky = _m_per_deg(g["main"]["lat"])
            if np.hypot((p["lng"] - g["main"]["lng"]) * kx, (p["lat"] - g["main"]["lat"]) * ky) <= radius_m:
                g["members"].append(p)
                if _area(p) > _area(g["main"]):
                    g["main"] = p
                placed = True
                break
        if not placed:
            g = {"main": p, "members": [p]}
            by_key.setdefault(k, []).append(g)
            groups.append(g)
    return groups


def _area(p):
    r = p.get("ring")
    return Polygon(r).area if r and len(r) >= 4 else 0.0


def poi_geoms(members):
    """(polygons or None, points) of the members for pixel lookup."""
    out = []
    for p in members:
        r = p.get("ring")
        if r and len(r) >= 4:
            g = Polygon(r)
            out.append(g if g.is_valid else shapely.make_valid(g))
        else:
            out.append(shapely.Point(p["lng"], p["lat"]))
    return out


def depths_for_groups(groups, storms, log=print):
    """-> {storm: int cm array over groups} (max over the members' footprints, flow-grid max over all hours)."""
    members_flat, owner = [], []
    for gi, g in enumerate(groups):
        for m in g["members"]:
            members_flat.append(m)
            owner.append(gi)
    owner = np.array(owner, dtype=np.int64)
    geoms = np.array(poi_geoms(members_flat), dtype=object) if members_flat else np.array([], dtype=object)
    is_pt = np.array([isinstance(g, shapely.Point) for g in geoms], dtype=bool) if len(geoms) else np.zeros(0, bool)
    out = {}
    cache = {}
    for s in storms:
        k = s.grid.key()
        if k not in cache:
            cache.clear()
            idx = fq.building_pixels(geoms, s.grid)          # a Point owns the pixel it lies in (all-touched, boundary ties aside)
            cache[k] = idx
        idx = cache[k]
        R, A = s.read_max_valid()
        code = fq.max_code(R, A, idx)
        cm = np.zeros(len(groups), dtype=np.int64)
        if code.size:
            np.maximum.at(cm, owner, np.rint(fq.code_depth(code) * 100).astype(np.int64))
        out[s.name] = cm
        if log:
            log("  %-6s %d facilities with water, %d flooded (>= 10 cm)" % (s.name, int((cm > 0).sum()), int((cm >= 10).sum())))
    return out


def build_features(groups, depth_cm, storms, source="osm"):
    names = [s.name for s in storms]
    feats = []
    for gi, g in enumerate(groups):
        p = g["main"]
        pr = {"name": p.get("name", ""), "type": p["type"], "source": source, "osm_id": p.get("osm_id", ""),
              "lng": round(p["lng"], 6), "lat": round(p["lat"], 6)}
        if p.get("operator") or (p.get("tags") or {}).get("operator"):
            pr["operator"] = p.get("operator") or p["tags"]["operator"]
        flooded = []
        mx = 0
        for nm in names:
            c = int(depth_cm[nm][gi])
            pr["d_" + nm] = c
            mx = max(mx, c)
            if c >= int(fq.FLOOD_MIN_M * 100):
                flooded.append(nm)
        pr["flooded_in"] = ",".join(flooded)
        pr["first_flooded"] = next((d for d in DESIGN if d in flooded), None)
        pr["max_cm"] = mx
        if g.get("members") and len(g["members"]) > 1:
            pr["parts"] = len(g["members"])
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [pr["lng"], pr["lat"]]}, "properties": pr})
    return feats


def merge_extra_points(osm_groups, extra, radius_m=HIFLD_RADIUS_M):
    """Add extra points {name,type,lng,lat,source,...} that have no OSM facility of the same type within radius_m.
    Returns (groups, added, skipped)."""
    groups = list(osm_groups)
    added = skipped = 0
    for e in extra:
        dup = False
        for g in osm_groups:
            p = g["main"]
            if p["type"] != e["type"]:
                continue
            kx, ky = _m_per_deg(p["lat"])
            if np.hypot((e["lng"] - p["lng"]) * kx, (e["lat"] - p["lat"]) * ky) <= radius_m:
                dup = True
                break
        if dup:
            skipped += 1
            continue
        e = dict(e, ring=None, osm_id=e.get("osm_id", ""), tags={})
        groups.append({"main": e, "members": [e], "source": e.get("source", "hifld")})
        added += 1
    return groups, added, skipped


def read_extra(path):
    """GeoJSON points (fetch_hifld.py output) -> list of dict(name,type,lng,lat,source)."""
    fc = gd.read_json_any(path)
    out = []
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        pr = f.get("properties") or {}
        out.append({"name": pr.get("name", ""), "type": pr.get("type", ""), "lng": float(g["coordinates"][0]),
                    "lat": float(g["coordinates"][1]), "source": pr.get("source", "hifld"), "operator": pr.get("operator", ""),
                    "osm_id": pr.get("id", "")})
    return [e for e in out if e["type"]]


# ------------------------------------------------------------------------------------------------ roads
def roads_flooded(roads, storms, dom, out_dir, log=print):
    sel = [r for r in roads if gd.in_box(np.array([c[0] for c in r["coords"]]), np.array([c[1] for c in r["coords"]]), dom.bbox).any()]
    lines = [r["coords"] for r in sel]
    cls_idx = np.array([CLASSES.index(r["highway"]) if r["highway"] in CLASSES else len(CLASSES) - 1 for r in sel])
    evac = np.array([1 if r["evac"] else (2 if r["highway"] in MAJOR_CLASSES else 0) for r in sel])
    os.makedirs(out_dir, exist_ok=True)
    cache = {}
    index = []
    for s in storms:
        t0 = time.perf_counter()
        k = s.grid.key()
        if k not in cache:
            cache.clear()
            cache[k] = fq.road_samples(lines, s.grid)
        rs = cache[k]
        R, A = s.read_max_valid()
        wet, fl, mx = fq.road_stats(R, A, rs)
        keep = np.flatnonzero(mx > 0)
        order = keep[np.lexsort((np.array([sel[i]["osm_id"] for i in keep]), -mx[keep].astype(np.int64)))] if keep.size else keep
        rows = []
        for i in order:
            r = sel[i]
            rows.append([r["osm_id"], r["name"] or r.get("ref", ""), int(cls_idx[i]), int(evac[i]), int(round(rs.length_m[i])),
                         int(round(wet[i])), int(round(fl[i])), int(round(fq.code_depth(mx[i]) * 100))])
        tot_km = float(rs.length_m.sum()) / 1000.0
        wet_km = float((rs.length_m * wet / 100.0).sum()) / 1000.0
        fl_km = float((rs.length_m * fl / 100.0).sum()) / 1000.0
        doc = {"format": "bgroads", "version": 1, "domain": dom.key, "storm": s.name, "label": s.label, "source": s.source,
               "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "rules": "OSM driveable ways sampled every 10 m (midpoints); wet >= 0.02 m, flooded >= 0.1 m; depth = flow-grid maximum over all hours",
               "classes": list(CLASSES), "columns": ["id", "name", "hw", "evac", "len_m", "wet_pct", "flooded_pct", "max_cm"],
               "evac_codes": {"0": "none", "1": "tagged evacuation route in OSM", "2": "major class (motorway, trunk, primary)"},
               "n_roads": len(sel), "total_km": round(tot_km, 1), "wet_km": round(wet_km, 2), "flooded_km": round(fl_km, 2),
               "n_rows": len(rows), "rows": rows}
        path = os.path.join(out_dir, s.name + ".json")
        size = gd.write_json(path, doc)
        index.append({"storm": s.name, "file": s.name + ".json", "rows": len(rows), "wet_km": round(wet_km, 2),
                      "flooded_km": round(fl_km, 2), "bytes": size})
        if log:
            log("  %-6s %5d ways with water of %d (%.1f km wet, %.1f km >= 10 cm of %.0f km)  %.0f KB  [%.1fs]"
                % (s.name, len(rows), len(sel), wet_km, fl_km, tot_km, size / 1024, time.perf_counter() - t0))
    gd.write_json(os.path.join(out_dir, "index.json"), {"domain": dom.key, "storms": index, "n_roads": len(sel)})
    return index


# ------------------------------------------------------------------------------------------------ driver
def build_domain(app, key, hifld=None, do_roads=True, pbf=None, log=print):
    t0 = time.perf_counter()
    dom = gd.get_domain(app, key)
    log("== infrastructure: %s ==" % dom.label)
    storms = fq.load_storms(dom.flow_dir, log=log)
    if not storms:
        raise SystemExit("no complete storms in %s" % dom.flow_dir)
    log("  storms: %s" % " ".join(s.name for s in storms))
    county = gd.get_domain(app, "county")
    res = osm_pbf.tagged(app, county.bbox, pbf=pbf, log=log)
    pois = [p for p in res["pois"] if gd.in_box(p["lng"], p["lat"], dom.bbox)]
    groups = dedupe_pois(pois)
    log("  %d OSM facilities (%d after merging parts of one facility): %s" % (
        len(pois), len(groups), ", ".join("%s %d" % (t, sum(1 for g in groups if g["main"]["type"] == t))
                                           for t in sorted({g["main"]["type"] for g in groups}))))
    n_extra = 0
    if hifld and os.path.isfile(hifld):
        extra = [e for e in read_extra(hifld) if gd.in_box(e["lng"], e["lat"], dom.bbox)]
        groups, added, skipped = merge_extra_points(groups, extra)
        n_extra = added
        log("  HIFLD: %d added, %d already in OSM" % (added, skipped))
    depth = depths_for_groups(groups, storms, log=log)
    feats = []
    osm_feats = build_features([g for g in groups if g.get("source") != "hifld"], {k: np.array([v[i] for i, g in enumerate(groups) if g.get("source") != "hifld"], dtype=np.int64) for k, v in depth.items()}, storms)
    hi = [g for g in groups if g.get("source") == "hifld"]
    hi_feats = build_features(hi, {k: np.array([v[i] for i, g in enumerate(groups) if g.get("source") == "hifld"], dtype=np.int64) for k, v in depth.items()}, storms, source="hifld") if hi else []
    feats = osm_feats + hi_feats
    out_path = dom.layer("infrastructure.geojson")
    doc = {"type": "FeatureCollection", "name": "infrastructure",
           "metadata": {"domain": dom.key, "storms": [s.name for s in storms], "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                        "note": "d_<storm> = max flood depth in cm at the facility; flooded = >= 10 cm. OSM facilities; HIFLD points only when fetch_hifld.py was run."},
           "features": feats}
    size = gd.write_json(out_path, doc)
    log("  wrote %s: %d facilities (%d HIFLD), %.0f KB" % (out_path, len(feats), n_extra, size / 1024))
    info = {"facilities": len(feats), "bytes": size}
    if do_roads:
        log("  flooded roads ...")
        info["roads"] = roads_flooded(res["roads"], storms, dom, os.path.join(app, "layers", "roads_flooded", key), log=log)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    log("  done in %.0fs" % info["seconds"])
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Critical facilities with flood depths + flooded-road lists (see the module docstring).")
    ap.add_argument("--domain", choices=("city", "county", "both"), default="city")
    ap.add_argument("--app", default=gd.APP_DEFAULT, help="folder the web server serves (default %(default)s)")
    ap.add_argument("--hifld", default=None, help="GeoJSON written by fetch_hifld.py to merge (default: <app>/layers/hifld_warren.geojson when it exists)")
    ap.add_argument("--no-roads", action="store_true")
    ap.add_argument("--pbf", default=None)
    a = ap.parse_args(argv)
    hifld = a.hifld or os.path.join(a.app, "layers", "hifld_warren.geojson")
    for key in (("city", "county") if a.domain == "both" else (a.domain,)):
        build_domain(a.app, key, hifld, not a.no_roads, a.pbf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
