r"""
build_utilities.py - utilities and infrastructure layer (water, power) from OpenStreetMap + flood exposure from the county COG tiles.

    python scripts/build_utilities.py                       # -> <app>/layers/utilities.geojson.gz
    python scripts/build_utilities.py --storms ev6 100yr    # a subset of storms (tests / quick looks)

Source: kentucky-latest.osm.pbf (the desktop viewer's pure-python reader, bgv/data/osmpbf.py, through scripts/osm_pbf.py; nodes and
ways are read here with a copy of the tagged-pass logic, the desktop's cache is untouched).  RELATIONS ARE NOT READ (a plant or
reservoir mapped only as a multipolygon relation is missing).  OpenStreetMap completeness varies: absent here = not mapped (or only
as a relation).

Tags -> type:
  man_made=water_tower                       water_tower        man_made=water_works             water_works
  man_made=storage_tank + content=water      water_tank         man_made=wastewater_plant        wastewater_plant
  man_made=reservoir_covered                 water_tank         man_made=pumping_station         pumping_station
  waterway=dam | weir                        dam | weir         power=substation                 substation
  power=plant|generator, source hydro        hydro_plant        power=transformer                transformer (only when <= 400 in the box)
  power=plant, other source                  power_plant        power=line|minor_line|cable      power_line | minor_line | power_cable (lines)
  power=generator, not solar                 generator (optional sub-type: one unit, e.g. a backup diesel set)
  power=generator, solar                     DROPPED (one panel row of a farm, or a rooftop array; counted in metadata.dropped)
  (Before this rule all 244 "power plants" in the county box were solar: 241 power=generator ground panel rows
  dropped now; the 3 left are power=plant solar farms.)  Dams: only 3 dam/weir ways are mapped in OSM inside the county box
  (Shanty Hollow Lake Dam, Franklin Dam (weir), one unnamed); dams as closed ways/areas ARE read, relations are not read
  (whether a dam relation exists in the box is unknown).  No NID (National Inventory of Dams) list exists in the data folder, so
  the layer under-represents dams; add NID points if a copy is obtained.
  man_made=pipeline                          pipeline (lines)
Power towers / poles are not features.

Output (FeatureCollection, features inside the county box; `in_city` marks those inside the city model box):
  metadata.storms  storm names = order of every per-storm array
  point features   properties: type, name, operator, source (plant / generator source tag), osm_id ('n123' / 'w456'), lng, lat, in_city,
                   d = max depth cm per storm (point = its pixel; area / short way = max over its footprint pixels),
                   first = first design storm (2yr..100yr) with d >= 10 cm, or null;  geometry Point
  line features    properties: type, voltage, name, operator, osm_id, in_city, d = max depth along the line per storm (cm),
                   f10 / f30 = % of the line's length at >= 10 / >= 30 cm per storm, first;  geometry LineString
Depth = layers/cog/county/<storm>/max_depth.tif (uint16 cm, 65535 = no data, EPSG:3857, 6.25 m pixels), the convention of
build_infrastructure.py (flooded >= 10 cm).  No data / outside the tiles counts as dry.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geodata as gd        # noqa: E402
import osm_pbf              # noqa: E402

DESIGN = ("2yr", "5yr", "10yr", "25yr", "50yr", "100yr")
NODATA = 65535
FLOOD_CM, DEEP_CM = 10, 30
TRANSFORMER_MAX = 400
LINE_TYPES = ("power_line", "minor_line", "power_cable", "pipeline")
KEYS = (b"man_made", b"power", b"waterway")
SAMPLE_M = 3.0                      # mercator units between samples along a line (a pixel is 6.25)


def _hydro(t):
    return "hydro" in (t.get("plant:source", "") + ";" + t.get("generator:source", "")).lower()


def classify(t):
    """OSM tags -> utility type or None."""
    mm, pw, ww = t.get("man_made"), t.get("power"), t.get("waterway")
    if mm == "water_tower":
        return "water_tower"
    if (mm == "storage_tank" and t.get("content") == "water") or mm == "reservoir_covered":
        return "water_tank"
    if mm in ("water_works", "wastewater_plant", "pumping_station"):
        return mm
    if ww in ("dam", "weir"):
        return ww
    if pw == "plant":
        return "hydro_plant" if _hydro(t) else "power_plant"
    if pw == "generator":
        # a generator is one unit (often one row of panels inside a solar farm, or a rooftop array), not a plant.
        # Solar panels are dropped (counted in metadata.dropped); other generators are an optional sub-type.
        if (t.get("generator:source") or "").lower() == "solar" or t.get("generator:type") == "solar_photovoltaic_panel":
            return "solar_generator"
        return "hydro_plant" if _hydro(t) else "generator"
    if pw == "substation":
        return "substation"
    if pw == "transformer":
        return "transformer"
    if pw == "line":
        return "power_line"
    if pw == "minor_line":
        return "minor_line"
    if pw == "cable":
        return "power_cable"
    if mm == "pipeline":
        return "pipeline"
    return None


def _props(t, typ):
    p = {"type": typ, "name": t.get("name", "") or "", "operator": t.get("operator", "") or ""}
    src = t.get("plant:source") or t.get("generator:source")
    if src:
        p["source"] = src
    if typ == "solar_generator" and t.get("location"):
        p["location"] = t["location"]
    if t.get("voltage"):
        p["voltage"] = t["voltage"]
    return p


# ------------------------------------------------------------------------------------------------ OSM pass
def read_utilities(pbf, bbox, margin_deg=0.05, max_blobs=None, log=print):
    """One pass over the PBF -> list of dict(osm_id, props, kind 'point'|'line'|'area', coords [(lng, lat)...]) inside bbox."""
    op = osm_pbf.desktop_reader()
    w, s, e, n = (float(x) for x in bbox)
    mw, ms, me, mn = w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg
    id_l, lon_l, lat_l = [], [], []
    node_ids = node_lon = node_lat = None
    seen_way = False
    feats = []
    with open(pbf, "rb") as fh:
        for nb, (btype, raw) in enumerate(op.iter_blobs(fh)):
            if btype != "OSMData":
                continue
            if max_blobs is not None and nb > max_blobs:
                break
            blk = op._parse_block(raw)
            want, key_ids = {}, []
            for i, sb in enumerate(blk.strings):
                if sb in KEYS:
                    want[i] = sb
                    key_ids.append(i)
            key_ids = np.array(key_ids, dtype=np.int64)
            for gs, ge in blk.groups:
                dense, has_way = None, False
                for f, wt, v in op._fields(raw, gs, ge):
                    if f == 2 and wt == 2:
                        dense = v
                    elif f == 3 and wt == 2:
                        has_way = True
                        break
                if dense is not None:
                    if seen_way:
                        raise ValueError("this PBF is not sorted (a node block follows way blocks)")
                    r = op._dense_nodes(raw, dense[0], dense[1], blk)
                    if r is not None:
                        ids, lon, lat = r
                        m = (lon >= mw) & (lon <= me) & (lat >= ms) & (lat <= mn)
                        if m.any():
                            id_l.append(ids[m]); lon_l.append(lon[m]); lat_l.append(lat[m])
                        for nid, nlon, nlat, tags in osm_pbf._node_tags(raw, dense[0], dense[1], blk, ids, lon, lat, key_ids, (w, s, e, n)):
                            typ = classify(tags)
                            if typ and typ not in LINE_TYPES:
                                feats.append({"osm_id": "n%d" % nid, "props": _props(tags, typ), "kind": "point", "coords": [(nlon, nlat)]})
                if has_way:
                    if not seen_way:
                        seen_way = True
                        if id_l:
                            node_ids = np.concatenate(id_l); node_lon = np.concatenate(lon_l); node_lat = np.concatenate(lat_l)
                            o = np.argsort(node_ids, kind="stable")
                            node_ids, node_lon, node_lat = node_ids[o], node_lon[o], node_lat[o]
                        else:
                            node_ids = np.zeros(0, np.int64); node_lon = node_lat = np.zeros(0)
                        id_l = lon_l = lat_l = None
                    if not want:
                        continue
                    ways, _ = op._ways(raw, gs, ge, blk, want)
                    sel = [(wy, classify(wy.tags)) for wy in ways]
                    sel = [(wy, t) for wy, t in sel if t]
                    if not sel:
                        continue
                    coords = osm_pbf._way_coords(raw, [x[0] for x in sel], node_ids, node_lon, node_lat)
                    for (wy, typ), (lon, lat, ok) in zip(sel, coords):
                        if ok.sum() < 1:
                            continue
                        pts = np.stack([lon[ok], lat[ok]], axis=1)
                        inside = (pts[:, 0] >= w) & (pts[:, 0] <= e) & (pts[:, 1] >= s) & (pts[:, 1] <= n)
                        if not inside.any():
                            continue
                        closed = bool(ok.all() and len(ok) >= 4 and lon[0] == lon[-1] and lat[0] == lat[-1])
                        kind = "line" if (typ in LINE_TYPES or not closed) else "area"
                        if kind == "line" and len(pts) < 2:
                            kind = "point"
                        feats.append({"osm_id": "w%d" % wy.wid, "props": _props(wy.tags, typ), "kind": kind,
                                      "coords": [tuple(p) for p in np.round(pts, 6).tolist()]})
    n_tr = sum(1 for f in feats if f["props"]["type"] == "transformer")
    if n_tr > TRANSFORMER_MAX:
        feats = [f for f in feats if f["props"]["type"] != "transformer"]
        if log:
            log("  %d transformers in the box: more than %d, left out" % (n_tr, TRANSFORMER_MAX))
    return feats


# ------------------------------------------------------------------------------------------------ geometry -> pixels
R_EARTH = 6378137.0


def to_merc(lng, lat):
    lng, lat = np.asarray(lng, float), np.asarray(lat, float)
    return np.radians(lng) * R_EARTH, np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85, 85)) / 2)) * R_EARTH


class Raster:
    """Geometry of a max_depth.tif: flat pixel index of mercator coordinates."""

    def __init__(self, x0, y0, res, width, height):
        self.x0, self.y0, self.res, self.width, self.height = x0, y0, res, width, height

    def flat(self, x, y):
        c = np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        r = np.floor((self.y0 - np.asarray(y)) / self.res).astype(np.int64)
        ok = (c >= 0) & (c < self.width) & (r >= 0) & (r < self.height)
        return np.where(ok, r * self.width + c, -1)


def feature_samples(f, rs: Raster):
    """Flat pixel indices (-1 outside) for one feature: point = its pixel; line = a sample every SAMPLE_M along it;
    area = every pixel centre inside + the outline vertices."""
    xs, ys = to_merc([c[0] for c in f["coords"]], [c[1] for c in f["coords"]])
    if f["kind"] == "point" or len(xs) == 1:
        return rs.flat(xs[:1], ys[:1])
    if f["kind"] == "line":
        seg = np.hypot(np.diff(xs), np.diff(ys))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(np.ceil(cum[-1] / SAMPLE_M)) + 1)
        d = np.linspace(0.0, cum[-1], n)
        return rs.flat(np.interp(d, cum, xs), np.interp(d, cum, ys))
    import shapely
    from shapely.geometry import Polygon
    poly = Polygon(np.stack([xs, ys], axis=1))
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    x0, y0, x1, y1 = poly.bounds
    gx = np.arange(np.floor(x0 / rs.res) * rs.res + rs.res / 2, x1 + rs.res, rs.res)
    gy = np.arange(np.floor(y0 / rs.res) * rs.res + rs.res / 2, y1 + rs.res, rs.res)
    if gx.size * gy.size > 4_000_000:                    # a huge reservoir: a coarser grid is plenty for a maximum
        k = int(np.ceil(np.sqrt(gx.size * gy.size / 4_000_000)))
        gx, gy = gx[::k], gy[::k]
    GX, GY = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(poly, GX.ravel(), GY.ravel())
    return rs.flat(np.concatenate([GX.ravel()[inside], xs]), np.concatenate([GY.ravel()[inside], ys]))


def exposure(feats, rs: Raster, depth_arrays, storms):
    """-> (d [n, S] cm, f10 [n, S] %, f30 [n, S] %).  depth_arrays: {storm: flat uint16 array or callable returning one}."""
    idx_list = [feature_samples(f, rs) for f in feats]
    lens = np.array([len(i) for i in idx_list], dtype=np.int64)
    idx = np.concatenate(idx_list) if idx_list else np.zeros(0, np.int64)
    owner = np.repeat(np.arange(len(feats)), lens)
    n, S = len(feats), len(storms)
    d = np.zeros((n, S), np.int64)
    f10 = np.zeros((n, S), np.int64)
    f30 = np.zeros((n, S), np.int64)
    for si, s in enumerate(storms):
        arr = depth_arrays[s]() if callable(depth_arrays[s]) else depth_arrays[s]
        v = np.where(idx >= 0, arr[np.maximum(idx, 0)], NODATA).astype(np.int64)
        v = np.where(v == NODATA, 0, v)
        mx = np.zeros(n, np.int64)
        np.maximum.at(mx, owner, v)
        c10 = np.bincount(owner, weights=(v >= FLOOD_CM).astype(float), minlength=n)
        c30 = np.bincount(owner, weights=(v >= DEEP_CM).astype(float), minlength=n)
        d[:, si] = mx
        f10[:, si] = np.rint(100.0 * c10 / np.maximum(lens, 1))
        f30[:, si] = np.rint(100.0 * c30 / np.maximum(lens, 1))
    return d, f10, f30


def first_flooded(drow, storms):
    for nm in DESIGN:
        if nm in storms and drow[storms.index(nm)] >= FLOOD_CM:
            return nm
    return None


# ------------------------------------------------------------------------------------------------ driver
def _cog_depth_loader(path):
    def load():
        import rasterio
        with rasterio.open(path) as r:
            return r.read(1).ravel()
    return load


def raster_of(path):
    import rasterio
    with rasterio.open(path) as r:
        t = r.transform
        assert r.crs.to_epsg() == 3857 and abs(t.a) == abs(t.e), "max_depth.tif must be EPSG:3857 with square pixels"
        return Raster(t.c, t.f, t.a, r.width, r.height)


def available_storms(cog_dir):
    names = [d for d in os.listdir(cog_dir) if os.path.isfile(os.path.join(cog_dir, d, "max_depth.tif"))]
    return sorted(names, key=lambda s: (0, DESIGN.index(s)) if s in DESIGN else (1, s))


def build(app, storms=None, pbf=None, cog_dir=None, out=None, bbox=None, city_box=None, max_blobs=None, log=print):
    t0 = time.perf_counter()
    cog_dir = cog_dir or os.path.join(app, "layers", "cog", "county")
    storms = list(storms) if storms else available_storms(cog_dir)
    bbox = bbox or gd.get_domain(app, "county").bbox
    city_box = city_box or gd.get_domain(app, "city").bbox
    pbf = pbf or os.path.join(app, "kentucky-latest.osm.pbf")
    log("== utilities: %s, storms %s ==" % (os.path.basename(pbf), " ".join(storms)))
    feats = read_utilities(pbf, bbox, max_blobs=max_blobs, log=log)
    seen, uniq = set(), []
    for f in feats:
        if f["osm_id"] not in seen:
            seen.add(f["osm_id"]); uniq.append(f)
    dropped = {}
    for f in uniq:
        if f["props"]["type"] == "solar_generator":
            k = "solar_generator_" + ("roof" if f["props"].get("location") == "roof" else "ground")
            dropped[k] = dropped.get(k, 0) + 1
    feats = [f for f in uniq if f["props"]["type"] != "solar_generator"]
    log("  %d features from OSM (dropped solar panel generators: %s)" % (len(feats), dropped))
    rs = raster_of(os.path.join(cog_dir, storms[0], "max_depth.tif"))
    arrays = {s: _cog_depth_loader(os.path.join(cog_dir, s, "max_depth.tif")) for s in storms}
    d, f10, f30 = exposure(feats, rs, arrays, storms)
    out_feats, counts = [], {}
    for i, f in enumerate(feats):
        p = dict(f["props"])
        lng = float(np.mean([c[0] for c in f["coords"]]))
        lat = float(np.mean([c[1] for c in f["coords"]]))
        p["osm_id"] = f["osm_id"]
        p["in_city"] = bool(city_box[0] <= lng <= city_box[2] and city_box[1] <= lat <= city_box[3])
        p["d"] = [int(v) for v in d[i]]
        p["first"] = first_flooded(d[i], storms)
        if f["kind"] == "line":
            p["f10"], p["f30"] = [int(v) for v in f10[i]], [int(v) for v in f30[i]]
            geom = {"type": "LineString", "coordinates": [list(c) for c in f["coords"]]}
        else:
            p["lng"], p["lat"] = round(lng, 6), round(lat, 6)
            geom = {"type": "Point", "coordinates": [p["lng"], p["lat"]]}
        counts[p["type"]] = counts.get(p["type"], 0) + 1
        out_feats.append({"type": "Feature", "geometry": geom, "properties": p})
    doc = {"type": "FeatureCollection", "name": "utilities",
           "metadata": {"storms": storms, "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                        "counts": counts, "dropped": dropped, "flooded_cm": FLOOD_CM, "deep_cm": DEEP_CM,
                        "note": "OpenStreetMap utilities (relations not read; completeness varies; solar panel generators dropped, see dropped; only dams mapped in OSM, no NID list). d = max depth cm per storm from the "
                                "COG tiles (6.25 m), f10 / f30 = % of line length at >= 10 / >= 30 cm. Verification pending."},
           "features": out_feats}
    out = out or os.path.join(app, "layers", "utilities.geojson.gz")
    size, raw = gd.write_json_gz(out, doc)
    log("  wrote %s: %d features, %.0f KB (%.0f KB raw)" % (out, len(out_feats), size / 1024, raw / 1024))
    for t in sorted(counts):
        k = [i for i, x in enumerate(out_feats) if x["properties"]["type"] == t]
        nfl = sum(1 for i in k if first_flooded(d[i], storms) is not None)
        log("    %-16s %5d  (flooded >= 10 cm in a design storm: %d)" % (t, counts[t], nfl))
    log("  done in %.0fs" % (time.perf_counter() - t0))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Utilities layer from OSM + COG flood exposure (see the module docstring).")
    ap.add_argument("--app", default=gd.APP_DEFAULT)
    ap.add_argument("--storms", nargs="*", default=None)
    ap.add_argument("--pbf", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    build(a.app, a.storms, a.pbf, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
