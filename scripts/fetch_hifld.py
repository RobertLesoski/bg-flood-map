r"""
fetch_hifld.py - USER-RUN, NOT RUN BY ANY BUILD. Downloads HIFLD hospitals, fire stations and EMS stations for Warren County KY and merges
them into the facility layer.

    python scripts/fetch_hifld.py [--app APP] [--no-merge] [--hospitals URL] [--fire URL] [--ems URL]

What it does
  1. Queries three ArcGIS FeatureServer layers (county FIPS 21227 / state KY, WGS84 points) with plain urllib (no extra packages):
        hospitals, fire stations, EMS stations.
  2. Normalises them to {name, type (hospital|fire_station|ems), lng, lat, source "hifld", operator} and writes
        <app>/layers/hifld_warren.geojson     (raw normalised download, kept so a rebuild can merge again)
  3. Unless --no-merge: runs build_infrastructure for the county and the city, which merge those points (a HIFLD point is skipped when an
     OSM facility of the same type is within 150 m) and compute their per-storm flood depths.
CAVEAT: the default service URLs below are the HIFLD Open Data ones as last known. HIFLD Open has been reorganised by DHS and mirrors move, so
a URL may be dead: pass the current FeatureServer layer URL with --hospitals / --fire / --ems (any layer with point geometry and a
name field; the field names tried are listed in NAME_FIELDS). Nothing is written when a layer fails; the error says which URL.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE = "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services"
LAYERS = {
    "hospital": BASE + "/Hospitals_1/FeatureServer/0",
    "fire_station": BASE + "/Fire_Stations/FeatureServer/0",
    "ems": BASE + "/EMS_Stations/FeatureServer/0",
}
NAME_FIELDS = ("NAME", "Name", "name", "FACILITY_NAME")
OPERATOR_FIELDS = ("OWNER", "TYPE", "OPERATOR")
WARREN_WHERE = ["COUNTYFIPS='21227'", "FIPS='21227'", "COUNTY='WARREN' AND STATE='KY'", "COUNTY='WARREN'"]
BBOX = (-86.80, 36.60, -86.05, 37.25)          # Warren County plus margin, used when no county attribute works


def query(url, where=None, bbox=None):
    p = {"f": "geojson", "outFields": "*", "outSR": "4326", "returnGeometry": "true"}
    if where:
        p["where"] = where
    else:
        p["where"] = "1=1"
        p.update({"geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
                  "spatialRel": "esriSpatialRelIntersects"})
    req = urllib.request.Request(url + "/query?" + urllib.parse.urlencode(p), headers={"User-Agent": "bg-flood-map/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_layer(url):
    last = None
    for w in WARREN_WHERE:
        try:
            d = query(url, where=w)
            if d.get("features"):
                return d["features"]
        except Exception as e:                      # noqa: BLE001
            last = e
    try:
        return query(url, bbox=BBOX).get("features", [])
    except Exception as e:                          # noqa: BLE001
        raise RuntimeError("HIFLD layer failed: %s (%s)" % (url, e if last is None else last))


def normalise(features, typ):
    out = []
    for f in features:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        pr = f.get("properties") or {}
        name = next((str(pr[k]).strip().title() for k in NAME_FIELDS if pr.get(k)), "")
        op = next((str(pr[k]).strip() for k in OPERATOR_FIELDS if pr.get(k)), "")
        lng, lat = g["coordinates"][:2]
        if BBOX[0] <= lng <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
            out.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lng, 6), round(lat, 6)]},
                        "properties": {"name": name, "type": typ, "source": "hifld", "operator": op}})
    return out


def main(argv=None):
    import geodata as gd
    ap = argparse.ArgumentParser(description="Download HIFLD facilities for Warren County and merge (see the module docstring).")
    ap.add_argument("--app", default=gd.APP_DEFAULT)
    ap.add_argument("--hospitals", default=LAYERS["hospital"])
    ap.add_argument("--fire", default=LAYERS["fire_station"])
    ap.add_argument("--ems", default=LAYERS["ems"])
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args(argv)
    feats = []
    for typ, url in (("hospital", a.hospitals), ("fire_station", a.fire), ("ems", a.ems)):
        got = normalise(fetch_layer(url), typ)
        print("%-13s %3d points from %s" % (typ, len(got), url))
        feats += got
    out = os.path.join(a.app, "layers", "hifld_warren.geojson")
    gd.write_json(out, {"type": "FeatureCollection", "name": "hifld_warren", "features": feats})
    print("wrote", out)
    if not a.no_merge:
        import build_infrastructure as bi
        for key in ("county", "city"):
            bi.build_domain(a.app, key, out, do_roads=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
