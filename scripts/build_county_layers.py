r"""
build_county_layers.py - Warren County footprints and roads for the WEB viewer: layers/county/buildings.geojson.gz + roads.geojson.gz

    python scripts/build_county_layers.py [--app APP] [--pbf FILE] [--road-tol-m 1.5] [--max-gz-mb 40]

Source: the OpenStreetMap county scene of the desktop viewer (desktop cache reused when it matches, else one pass over the PBF; osm_pbf.py).
ALL OSM building ways in the county box are written (multipolygon relations are not read by the reader). Building properties follow
layers/buildings.geojson: name, building, amenity, ADDRESS, height_m, LAND_USE, sale_price (+ osm_id, PVA_PARCEL, prop_value_usd) where the
all_parcels.geojson parcels (city area only) match; ADDRESS = parcel site address, else OSM addr tags, else the desktop's 'near <street>'
label. Roads: highway (class), name, osm_id, length (m), simplified with --road-tol-m. Coordinates are rounded to 6 decimals.
"""
import argparse, os, sys, time
import numpy as np
import shapely
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import geodata as gd, osm_pbf


def build(app, pbf=None, road_tol_m=1.5, max_gz_mb=40.0, log=print):
    t0 = time.perf_counter()
    dom = gd.get_domain(app, "county")
    feats, roads = osm_pbf.county_scene(app, dom.bbox, pbf, log)
    parcels = gd.load_parcels(app, log=log)
    geoms = np.array([shapely.geometry.shape(f["geometry"]) for f in feats], dtype=object)
    pj = gd.join_points_to_parcels(shapely.point_on_surface(geoms), parcels) if len(geoms) else np.zeros(0, int)
    out = []
    for f, pi in zip(feats, pj):
        p = dict(f["properties"])
        pi = int(pi)
        if pi >= 0:
            if parcels.address[pi]:
                p["ADDRESS"] = parcels.address[pi]
            p["LAND_USE"] = parcels.land_use[pi]
            p["PVA_PARCEL"] = parcels.pva[pi]
            if parcels.sale_price[pi]:
                p["sale_price"] = parcels.sale_price[pi]
            if parcels.value[pi]:
                p["prop_value_usd"] = int(parcels.value[pi])
        out.append({"type": "Feature", "properties": p, "geometry": f["geometry"]})
    sz_b, raw_b = gd.write_json_gz(gd.Domain.layer(dom, "buildings.geojson.gz"), {"type": "FeatureCollection", "features": out})
    log("  buildings: %d (%d joined to a parcel) -> %.1f MB gz (%.1f MB json)" % (len(out), int((pj >= 0).sum()), sz_b / 1e6, raw_b / 1e6))
    tol = road_tol_m / 111320.0
    rf = []
    for f in roads:
        g = shapely.geometry.shape(f["geometry"])
        s = g.simplify(tol) if tol > 0 else g
        c = np.round(np.asarray(s.coords), 6).tolist()
        p = f["properties"]
        rf.append({"type": "Feature", "properties": {"highway": p["highway"], "name": p.get("name", ""), "osm_id": p["osm_id"],
                   "length": round(g.length * 111320.0 * 0.8, 0)}, "geometry": {"type": "LineString", "coordinates": c}})
    sz_r, raw_r = gd.write_json_gz(gd.Domain.layer(dom, "roads.geojson.gz"), {"type": "FeatureCollection", "features": rf})
    log("  roads: %d -> %.1f MB gz" % (len(rf), sz_r / 1e6))
    tot = (sz_b + sz_r) / 1e6
    if tot > max_gz_mb:
        log("  WARNING: %.1f MB gz exceeds %.0f MB - raise --road-tol-m" % (tot, max_gz_mb))
    log("  done in %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--app", default=gd.APP_DEFAULT); ap.add_argument("--pbf"); ap.add_argument("--road-tol-m", type=float, default=1.5)
    ap.add_argument("--max-gz-mb", type=float, default=40.0)
    a = ap.parse_args(); build(a.app, a.pbf, a.road_tol_m, a.max_gz_mb)
