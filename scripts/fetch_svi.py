r"""
fetch_svi.py - USER-RUN, NOT RUN BY ANY BUILD. CDC/ATSDR Social Vulnerability Index for Warren County KY census tracts -> layers/svi.geojson

    python scripts/fetch_svi.py [--app APP] [--svi-csv FILE] [--tracts-geojson FILE] [--svi-url URL] [--tracts-url URL]

Two downloads (or two local files, for an offline run: --svi-csv / --tracts-geojson):
  * SVI table for Kentucky tracts (CDC/ATSDR, default the 2022 state CSV on svi.cdc.gov), filtered to county FIPS 21227 (Warren)
  * tract polygons from the Census TIGERweb REST service (GeoJSON, state 21 county 227)
joined on the 11-digit tract FIPS. Output properties: FIPS, LOCATION, E_TOTPOP, RPL_THEMES (overall percentile 0-1, higher = more vulnerable),
RPL_THEME1 (socioeconomic), RPL_THEME2 (household characteristics), RPL_THEME3 (racial and ethnic minority status), RPL_THEME4 (housing type and
transportation). SVI value -999 (no data) becomes null. Percentiles are ranked against all US tracts.
CAVEAT: the default URLs are the ones last known and were not tested here (no network access when this was written); if one fails pass the
current URL or a downloaded file. The SVI is a vulnerability ranking from census data, not a flood model.
"""
import argparse
import csv
import io
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COUNTY_FIPS = "21227"
SVI_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states/Kentucky.csv"
TRACT_URL = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query?"
             + urllib.parse.urlencode({"where": "STATE='21' AND COUNTY='227'", "outFields": "GEOID,NAME", "outSR": "4326", "f": "geojson"}))
FIELDS = ("E_TOTPOP", "RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "bg-flood-map/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f <= -990 else f


def read_svi(text, county=COUNTY_FIPS):
    """CSV text -> {tract FIPS: props} for one county."""
    out = {}
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        fips = str(row.get("FIPS") or row.get("STCNTY", "") + row.get("TRACTCE", "")).strip().zfill(11)
        if not fips.startswith(county):
            continue
        p = {"FIPS": fips, "LOCATION": row.get("LOCATION", "")}
        for k in FIELDS:
            p[k] = _num(row.get(k))
        out[fips] = p
    return out


def join(tracts_fc, svi):
    feats = []
    for f in tracts_fc.get("features", []):
        pr = f.get("properties") or {}
        fips = str(pr.get("GEOID") or pr.get("FIPS") or "").strip()
        if fips in svi:
            feats.append({"type": "Feature", "geometry": f["geometry"], "properties": svi[fips]})
    return {"type": "FeatureCollection", "name": "svi", "features": feats}


def main(argv=None):
    import geodata as gd
    ap = argparse.ArgumentParser(description="CDC/ATSDR SVI for Warren County tracts (see the module docstring).")
    ap.add_argument("--app", default=gd.APP_DEFAULT)
    ap.add_argument("--svi-csv")
    ap.add_argument("--tracts-geojson")
    ap.add_argument("--svi-url", default=SVI_URL)
    ap.add_argument("--tracts-url", default=TRACT_URL)
    a = ap.parse_args(argv)
    text = open(a.svi_csv, encoding="utf-8-sig").read() if a.svi_csv else _get(a.svi_url).decode("utf-8-sig")
    tracts = json.load(open(a.tracts_geojson, encoding="utf-8")) if a.tracts_geojson else json.loads(_get(a.tracts_url))
    svi = read_svi(text)
    fc = join(tracts, svi)
    print("%d SVI tracts in Warren County, %d joined to polygons" % (len(svi), len(fc["features"])))
    if not fc["features"]:
        print("nothing joined - check that both sources use 11-digit tract FIPS (21227xxxxxx)")
        return 1
    out = os.path.join(a.app, "layers", "svi.geojson")
    gd.write_json(out, fc)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
