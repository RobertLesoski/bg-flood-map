r"""
fetch_census.py - USER-RUN downloader for the public Census data used by build_demographic_exposure.py.

    python scripts/fetch_census.py [--app APP] [--year 2023] [--out FOLDER]

Downloads (public, no key needed for this volume):
  * American Community Survey 5-year estimates, block-group level, Warren County KY (state 21, county 227) from
    https://api.census.gov/data/<year>/acs/acs5   (tries --year, then the previous year if the API answers 404/204)
      B01003_001  total population            B01001_002 / _026  male / female total
      B03002_001,003-009,012  race / Hispanic origin    B19013_001  median household income (past 12 months, inflation adjusted)
      B25003_001-003  housing tenure (all / owner / renter occupied)         every variable with its margin of error (M)
  * TIGER/Line block-group polygons for Kentucky: https://www2.census.gov/geo/tiger/TIGER<year>/BG/tl_<year>_21_bg.zip

Raw files are saved untouched in a NEW folder <app>/layers/census/acs<year>_warren_bg/ together with census_manifest.json
(source URL, UTC date, size, sha256 of every file).  The downloaded files are DATA: nothing in them is ever executed; the builder reads them
with json / zipfile / rasterio only.  Stops with a clear message (and writes nothing half-way) when a download fails; no number is ever invented.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

APP_DEFAULT = "C:/Users/Minec/Desktop/DSOC AI/Bowling Green Flood Map"
STATE, COUNTY = "21", "227"
TABLE_VARS = (["B01003_001", "B01001_002", "B01001_026"]
              + ["B03002_%03d" % i for i in (1, 3, 4, 5, 6, 7, 8, 9, 12)]
              + ["B19013_001", "B25003_001", "B25003_002", "B25003_003"])
GET = ["NAME"] + [v + s for v in TABLE_VARS for s in ("E", "M")]


def _get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "bg-flood-map-census/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def acs_url(year):
    return ("https://api.census.gov/data/%d/acs/acs5?get=%s&for=block%%20group:*&in=state:%s%%20county:%s"
            % (year, ",".join(GET), STATE, COUNTY))


def tiger_url(year):
    return "https://www2.census.gov/geo/tiger/TIGER%d/BG/tl_%d_%s_bg.zip" % (year, year, STATE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default=APP_DEFAULT)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    manifest = {"fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "files": []}
    got_year = None
    acs = None
    for y in (a.year, a.year - 1):
        url = acs_url(y)
        try:
            acs = _get(url)
            json.loads(acs)
            got_year = y
            manifest["acs_url"] = url
            break
        except (urllib.error.URLError, ValueError) as e:
            print("ACS %d failed: %s" % (y, e))
            acs = None
    if acs is None:
        print("STOP: the ACS download failed for every year tried; nothing written. Run this script again with network access.")
        return 2
    out = a.out or os.path.join(a.app, "layers", "census", "acs%d_warren_bg" % got_year)
    if os.path.exists(out) and os.listdir(out):
        print("STOP: %s already exists and is not empty (never overwritten)." % out)
        return 3
    try:
        zurl = tiger_url(got_year)
        z = _get(zurl, timeout=600)
    except urllib.error.URLError as e:
        print("STOP: TIGER download failed (%s); nothing written." % e)
        return 2
    os.makedirs(out)
    for name, url, blob in (("acs5_blockgroups.json", manifest["acs_url"], acs), ("tl_%d_21_bg.zip" % got_year, zurl, z)):
        with open(os.path.join(out, name), "wb") as f:
            f.write(blob)
        manifest["files"].append({"file": name, "url": url, "bytes": len(blob), "sha256": sha256(blob)})
        print("saved %s  %d bytes  sha256 %s" % (name, len(blob), sha256(blob)))
    manifest["acs_year"] = got_year
    with open(os.path.join(out, "census_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("done ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
