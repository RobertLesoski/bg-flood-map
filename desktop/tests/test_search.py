"""Tests for bgv.search (address index, depth tiles, facilities, roads, report, help content) on small synthetic files, plus
`real` tests that compare the app's numbers with INDEPENDENT code on the real data (skipped when the data folder is absent)."""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

from bgv.search import addrindex as A
from bgv.search import assess as asm
from bgv.search import facilities as F
from bgv.search import helpdoc as H
from bgv.search import report as R
from bgv.search import roadrisk as RR
from bgv.search import tiles as T

DATA = Path(r"C:\Users\Minec\Desktop\DSOC AI\Bowling Green Flood Map")
real = pytest.mark.skipif(not (DATA / "layers" / "address_index.json.gz").is_file(), reason="real data folder not available")


# ------------------------------------------------------------------------------------------------ synthetic index
def make_doc(sharded=False):
    streets = [("ELM ST", "Elm Street"), ("MAIN ST", "Main Street"), ("MAIN ST W", "West Main Street"), ("W 10TH AVE", "West 10th Avenue")]
    recs = []        # (street index, num, kind)
    for si, nums in enumerate((["100", "102", "1000"], ["1", "10", "100", "1000A", "~"], ["5"], ["12", "14"])):
        for n in nums:
            recs.append((si, "" if n == "~" else n, 1 if n == "~" else 0))
    n = len(recs)
    start, last = [], -1
    for i, (si, _, _) in enumerate(recs):
        if si != last:
            start.append(i)
            last = si
    start.append(n)
    # flood: record 3 ("1" Main St): 25yr 45 cm tier 2 first 7 for 9 h; record 5 ("100" Main): 25yr 130 cm; 5yr 12 cm
    def enc(items):
        flat, prev = [], 0
        for idx, cm, tier, fh, hh in items:
            flat += [idx - prev, cm, tier, fh, hh]
            prev = idx
        return flat
    doc = {"format": "bgaddr", "version": 1, "domain": "city", "label": "Test", "built": "2026-01-01", "n": n, "origin": [-86.5, 36.9],
           "scale": 100000, "bbox": [0, 0, 0, 0], "use_legend": ["", "R"],
           "storms": [{"key": "5yr", "label": "5-year storm", "source": "tiers", "grid": {"w": 10, "h": 10, "cell_m": 6.4}},
                      {"key": "25yr", "label": "25-year storm", "source": "tiers", "grid": {"w": 10, "h": 10, "cell_m": 6.4}}],
           "x": [i * 10 for i in range(n)], "y": [i * 5 for i in range(n)], "num": [r[1] for r in recs], "use": [1] * n,
           "val": [1000 * i for i in range(n)], "parcel": ["P%d" % i for i in range(n)], "bid": [i if r[2] == 0 else -1 for i, r in enumerate(recs)],
           "kind": [r[2] for r in recs], "nb": [1] * n,
           "streets": {"key": [s[0] for s in streets], "disp": [s[1] for s in streets], "start": start},
           "flood": {"5yr": enc([(5, 12, 1, 9, 2)]), "25yr": enc([(3, 45, 2, 7, 9), (5, 130, 4, 5, 15)])}}
    return doc, recs


def write_gz(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f)


def test_index_decode_and_search(tmp_path):
    doc, recs = make_doc()
    p = tmp_path / "address_index.json.gz"
    write_gz(p, doc)
    idx = A.AddressIndex.load(p)
    assert idx.n == len(recs) and idx.storm_keys == ["5yr", "25yr"]
    assert idx.depth_cm["25yr"][3] == 45 and idx.tier["25yr"][3] == 2 and idx.first["25yr"][3] == 7 and idx.hours["25yr"][3] == 9
    assert idx.depth_cm["25yr"][5] == 130 and idx.depth_cm["5yr"][5] == 12 and idx.depth_cm["5yr"][3] == 0
    assert not idx.wet["25yr"][4] and idx.wet["25yr"][5]
    rows = idx.storm_rows(5)
    assert [r.flooded for r in rows] == [True, True] and rows[1].tier == 4
    # search: number + street prefix, abbreviation of the completed words, exact number first
    h = idx.search("100 main s")
    assert [x.label for x in h][:1] == ["100 Main Street"]
    assert idx.search("100 main street")[0].label == "100 Main Street"
    assert idx.search("12 w 10th")[0].label == "12 West 10th Avenue"
    assert idx.search("12 west 10th avenue")[0].label == "12 West 10th Avenue"
    assert [x.kind for x in idx.search("main")] == ["street", "street"]            # two streets match -> street rows
    assert idx.search("elm")[0].kind == "address"                                   # one street -> its addresses
    assert idx.search("1000")[0].label.startswith("1000 ")                          # number only; exact first
    assert idx.search("100 Main Street, Bowling Green, KY")[0].label == "100 Main Street"
    assert idx.search("zzzzz") == [] and idx.search("") == []
    assert idx.record(3).address == "1 Main Street" and idx.by_bid(3) == 3 and idx.by_bid(999) == -1
    r = idx.record(3)
    assert r.lng == pytest.approx(-86.5 + 30 / 1e5) and r.lat == pytest.approx(36.9 + 15 / 1e5)
    assert idx.nearest(r.lng, r.lat, 5.0) == 3


def test_index_sharded_equals_single(tmp_path):
    doc, recs = make_doc()
    single = tmp_path / "a" / "address_index.json.gz"
    write_gz(single, doc)
    # build a sharded layout exactly like the builder: manifest (no records) + shards by first character of the street key
    sh = tmp_path / "b"
    keys = doc["streets"]["key"]
    start = doc["streets"]["start"]
    fc = [k[:1] for k in keys]
    shards, i = [], 0
    while i < len(keys):
        j = i
        while j < len(keys) and fc[j] == fc[i]:
            j += 1
        lo, hi = start[i], start[j]
        blk = {"shard": fc[i], "rec_lo": lo, "rec_hi": hi}
        for k in ("x", "y", "num", "use", "val", "parcel", "bid", "kind", "nb"):
            blk[k] = doc[k][lo:hi]
        blk["flood"] = {}
        for nm, f in doc["flood"].items():
            out, a = [], 0
            for t in range(0, len(f), 5):
                a += f[t]
                if lo <= a < hi:
                    out.append(a)
                    out += f[t + 1:t + 5]
            # re-delta within the shard relative to rec_lo
            flat, prev = [], lo
            for t in range(0, len(out), 5):
                flat += [out[t] - prev] + out[t + 1:t + 5]
                prev = out[t]
            blk["flood"][nm] = flat
        write_gz(sh / "address" / ("%s.json.gz" % fc[i]), blk)
        shards.append({"shard": fc[i], "file": "address/%s.json.gz" % fc[i], "street_lo": i, "street_hi": j, "rec_lo": lo, "rec_hi": hi})
        i = j
    man = {k: v for k, v in doc.items() if k not in ("x", "y", "num", "use", "val", "parcel", "bid", "kind", "nb", "flood")}
    man.update(sharded=True, shards=shards)
    write_gz(sh / "address_index.json.gz", man)
    a, b = A.AddressIndex.load(single), A.AddressIndex.load(sh / "address_index.json.gz")
    assert a.num == b.num and np.array_equal(a.bid, b.bid)
    for k in a.storm_keys:
        assert np.array_equal(a.depth_cm[k], b.depth_cm[k]) and np.array_equal(a.first[k], b.first[k])


def test_index_missing_or_bad_gives_readable_error(tmp_path):
    with pytest.raises(A.IndexError_) as e:
        A.AddressIndex.load(tmp_path / "nope.json.gz")
    assert "not found" in str(e.value) and "build_address_index" in str(e.value)
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(b"not gzip")
    with pytest.raises(A.IndexError_):
        A.AddressIndex.load(bad)
    wrong = tmp_path / "wrong.json.gz"
    write_gz(wrong, {"format": "other"})
    with pytest.raises(A.IndexError_) as e:
        A.AddressIndex.load(wrong)
    assert "not an address index" in str(e.value)


def test_index_path_and_tier_edges():
    assert A.index_path("R", "city").as_posix().endswith("R/layers/address_index.json.gz")
    assert A.index_path("R", "county").as_posix().endswith("R/layers/county/address_index.json.gz")
    assert [A.tier_of_cm(v) for v in (0, 9, 10, 29, 30, 59, 60, 119, 120, 800)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


# ------------------------------------------------------------------------------------------------ tiles
def merc(lng, lat):
    return T.merc(lng, lat)


def make_tile(path, left, top, res=6.25, w=60, h=50, fill=0, patch=None, nodata_box=None):
    import rasterio
    from rasterio.transform import from_origin
    arr = np.full((h, w), fill, np.uint16)
    for (r0, r1, c0, c1, v) in (patch or []):
        arr[r0:r1, c0:c1] = v
    if nodata_box:
        r0, r1, c0, c1 = nodata_box
        arr[r0:r1, c0:c1] = 65535
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint16", crs="EPSG:3857",
                       transform=from_origin(left, top, res, res), nodata=65535, tiled=True, blockxsize=16, blockysize=16) as ds:
        ds.write(arr, 1)
    return arr


LNG0, LAT0 = -86.44, 36.99


def tile_root(tmp_path, patch=None, nodata_box=None, storms=("100yr", "25yr")):
    x0, y0 = merc(LNG0, LAT0)
    left, top = float(x0) - 60 * 6.25 / 2, float(y0) + 50 * 6.25 / 2
    arrs = {}
    for k, s in enumerate(storms):
        arrs[s] = make_tile(tmp_path / "layers" / "cog" / "county" / s / "max_depth.tif", left, top, patch=patch if k == 0 else [(10, 20, 10, 20, 7)],
                            nodata_box=nodata_box)
    return arrs, left, top


def test_tiles_discover_and_labels(tmp_path):
    arrs, _, _ = tile_root(tmp_path)
    ts = T.TileSet.discover(tmp_path, "county")
    assert ts is not None and sorted(ts.storms) == ["100yr", "25yr"]
    assert ts.label == "solver depth (tiles, 6.25 m; verification pending)" and T.INDEX_LABEL == "index (grid)"
    assert T.TileSet.discover(tmp_path, "city") is None                    # the city has no tiles
    assert T.TileSet.discover(tmp_path / "empty", "county") is None
    ts2 = T.TileSet.discover(tmp_path, "county", {"tiles_verified": True})
    assert ts2.label == "solver depth (tiles, 6.25 m)"
    # configurable folder (settings and environment)
    alt = tmp_path / "alt"
    (alt / "county").mkdir(parents=True)
    import shutil
    shutil.copytree(tmp_path / "layers" / "cog" / "county" / "100yr", alt / "county" / "100yr")
    ts3 = T.TileSet.discover(tmp_path / "nothing", "county", {"tiles_dir": str(alt)})
    assert ts3 is not None and ts3.storms == ["100yr"]
    ts.close(), ts2.close(), ts3.close()


def test_tiles_resolution_is_read_from_the_raster(tmp_path):
    x0, y0 = merc(LNG0, LAT0)
    make_tile(tmp_path / "layers" / "cog" / "county" / "100yr" / "max_depth.tif", float(x0) - 100, float(y0) + 100, res=2.0, w=100, h=100, fill=33)
    ts = T.TileSet.discover(tmp_path, "county")
    assert ts.res_m == 2.0 and "2 m" in ts.label
    assert ts.sample_points("100yr", [LNG0], [LAT0])[0] == 33
    ts.close()


def test_tiles_point_sampling_matches_rasterio(tmp_path):
    import rasterio
    from rasterio.warp import transform
    arrs, left, top = tile_root(tmp_path, patch=[(20, 30, 20, 30, 55), (22, 24, 22, 24, 200)], nodata_box=(0, 5, 0, 5))
    ts = T.TileSet.discover(tmp_path, "county")
    rng = np.random.default_rng(3)
    xs = left + rng.uniform(-30, 60 * 6.25 + 30, 300)
    ys = top - rng.uniform(-30, 50 * 6.25 + 30, 300)
    xs[:3], ys[:3] = left + np.array([23.0, 25.5, 2.5]) * 6.25, top - np.array([23.0, 25.5, 2.5]) * 6.25     # on 200 / 55 / nodata
    lngs, lats = transform("EPSG:3857", "EPSG:4326", list(xs), list(ys))
    got = ts.sample_points("100yr", lngs, lats)
    with rasterio.open(ts.paths["100yr"]) as ds:
        for k in range(300):
            try:
                r, c = ds.index(xs[k], ys[k])
            except Exception:
                r, c = -1, -1
            ok = 0 <= r < ds.height and 0 <= c < ds.width
            exp = int(arrs["100yr"][r, c]) if ok else -1
            exp = -1 if exp == 65535 else exp
            assert got[k] == exp, (k, got[k], exp)
    assert (got == 200).any() and (got == -1).any() and (got == 55).any()
    assert ts.sample_points("nonexistent", [LNG0], [LAT0])[0] == -1
    ts.close()


def square_lnglat(cx, cy, half_m):
    from rasterio.warp import transform
    xs, ys = transform("EPSG:3857", "EPSG:4326", [cx - half_m, cx + half_m, cx + half_m, cx - half_m, cx - half_m],
                       [cy - half_m, cy - half_m, cy + half_m, cy + half_m, cy - half_m])
    return [[[float(a), float(b)] for a, b in zip(xs, ys)]]


def test_tiles_footprint_max_all_touched(tmp_path):
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import transform_geom
    arrs, left, top = tile_root(tmp_path, patch=[(20, 22, 20, 22, 90), (30, 31, 30, 31, 140), (40, 50, 40, 55, 65535)])
    ts = T.TileSet.discover(tmp_path, "county")
    rng = np.random.default_rng(11)
    mism = 0
    for k in range(60):
        cx = left + rng.uniform(20, 60 * 6.25 - 20)
        cy = top - rng.uniform(20, 50 * 6.25 - 20)
        half = rng.uniform(3, 14)
        ring = square_lnglat(cx, cy, half)
        got, cells = ts.footprint_max("100yr", ring)
        geom = transform_geom("EPSG:4326", "EPSG:3857", {"type": "Polygon", "coordinates": ring})
        with rasterio.open(ts.paths["100yr"]) as ds:
            try:
                out, _ = mask(ds, [geom], crop=True, all_touched=True, nodata=65535, filled=True)
                v = out[0][out[0] != 65535]
                exp = int(v.max()) if v.size else -1
            except ValueError:
                exp = -1
        if got != exp:
            mism += 1
    assert mism <= 1, "footprint maximum differs from rasterio.mask(all_touched) in %d of 60 squares" % mism
    # a footprint on the patch sees the patch value; a footprint fully outside sees nothing
    ring = square_lnglat(left + 30.5 * 6.25, top - 30.5 * 6.25, 2.0)
    assert ts.footprint_max("100yr", ring)[0] == 140
    far = square_lnglat(left - 5000, top + 5000, 5.0)
    assert ts.footprint_max("100yr", far) == (-1, 0)
    # multipolygon form works too
    assert ts.footprint_max("100yr", [ring])[0] == 140
    ts.close()


def test_tiles_neighbourhood_max(tmp_path):
    arrs, left, top = tile_root(tmp_path, patch=[(25, 26, 30, 31, 210)])
    ts = T.TileSet.discover(tmp_path, "county")
    from rasterio.warp import transform
    cx, cy = left + 30.5 * 6.25, top - 25.5 * 6.25
    (lng,), (lat,) = transform("EPSG:3857", "EPSG:4326", [cx + 12.0], [cy])
    assert ts.sample_points("100yr", [lng], [lat])[0] == 0
    assert ts.neighbourhood_max("100yr", [lng], [lat], 30.0)[0] == 210
    assert ts.neighbourhood_max("100yr", [lng], [lat], 1.0)[0] == 0
    ts.close()


def test_compare_flags_material_disagreement_only():
    c = T.compare(57, 63)                       # +6 across the 60 cm tier edge: the tier differs, NOT a material disagreement
    assert c.tier_differs is True and c.disagree is False and T.compare(52, 58).tier_differs is False
    assert c.delta_cm == 6
    assert T.compare(52, 63).disagree is True and T.compare(5, 14).disagree is True      # flooded state flips / 10+ cm
    assert T.compare(120, 12).disagree is True and T.compare(None, 12).disagree is False and T.compare(30, None).delta_cm is None
    assert T.compare(52, -1).tile_cm is None


# ------------------------------------------------------------------------------------------------ assess
def test_assess_uses_tiles_and_keeps_both(tmp_path):
    doc, recs = make_doc()
    # place record 5 on the tile
    x0, y0 = merc(LNG0, LAT0)
    doc["origin"] = [LNG0, LAT0]
    doc["x"] = [0] * len(recs)
    doc["y"] = [0] * len(recs)
    doc["storms"] = [dict(s, key=k) for s, k in zip(doc["storms"], ("25yr", "100yr"))]
    doc["flood"] = {"25yr": doc["flood"]["5yr"], "100yr": doc["flood"]["25yr"]}
    write_gz(tmp_path / "idx.json.gz", doc)
    idx = A.AddressIndex.load(tmp_path / "idx.json.gz")
    tile_root(tmp_path, patch=[(24, 26, 29, 31, 80)])
    ts = T.TileSet.discover(tmp_path, "county")
    a = asm.assess(idx, 5, ts, None)
    r100, r25 = a.row("100yr"), a.row("25yr")
    assert a.method == "point" and a.tile_label == ts.label
    assert r100.index.depth_cm == 130 and r100.tile_cm == 80 and r100.shown_cm == 80 and r100.shown_source == ts.label
    assert r100.comparison.delta_cm == -50 and r100.comparison.disagree
    assert r25.tile_cm == 0 and r25.shown_cm == 0 and r25.comparison.disagree      # dry in the tile, 12 cm in the index: flooded state flips
    # without tiles the index is the headline
    b = asm.assess(idx, 5, None, None)
    assert b.method is None and b.row("100yr").shown_cm == 130 and b.row("100yr").shown_source == "index (grid)"
    assert asm.short_label("Historical event 6 (county)") == "Event 6" and asm.short_label("100-year storm (county)") == "100-year storm"
    ts.close()


# ------------------------------------------------------------------------------------------------ facilities
def facility_file(tmp_path, with_tiles_point=None):
    feats = []
    pts = [("Hospital A", "hospital", LNG0, LAT0, {"2yr": 0, "5yr": 40, "10yr": 60}),
           ("School B", "school", LNG0 + 0.01, LAT0, {"2yr": 0, "5yr": 0, "10yr": 0}),
           ("", "police", LNG0 + 0.02, LAT0, {"2yr": 15, "5yr": 15, "10yr": 20})]
    for name, t, lng, lat, d in pts:
        pr = {"name": name, "type": t, "source": "osm", "osm_id": "n1", "lng": lng, "lat": lat}
        pr.update({"d_" + k: v for k, v in d.items()})
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lng, lat]}, "properties": pr})
    p = tmp_path / "layers" / "county" / "infrastructure.geojson"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "FeatureCollection", "metadata": {"storms": ["10yr", "2yr", "5yr", "ev1"]}, "features": feats}), encoding="utf-8")
    return p


def test_facilities_load_order_and_first_flooded(tmp_path):
    fs = F.FacilitySet.load(facility_file(tmp_path))
    assert fs.n == 3 and fs.storms == ["2yr", "5yr", "10yr", "ev1"]
    assert fs.first_flooded(0) == "5yr" and fs.first_flooded(1) is None and fs.first_flooded(2) == "2yr"
    assert fs.counts("5yr") == (2, 3) and fs.source("5yr") == "index (grid)"
    info = fs.info(0, "10yr")
    assert info["max_cm"] == 60 and info["max_storm"] == "10yr" and info["current"]["effective_cm"] == 60
    assert info["name"] == "Hospital A" and fs.info(2, "2yr")["name"] == "(unnamed)"
    assert F.style("hospital")[2] == "cross" and F.style("weird")[0] == "Other facility"


def test_facilities_tiles_override_and_compare(tmp_path):
    fs = F.FacilitySet.load(facility_file(tmp_path))
    x0, y0 = merc(LNG0, LAT0)
    tile_root(tmp_path, storms=("5yr", "10yr"))                      # all zeros except a 7 cm patch in the first storm array
    ts = T.TileSet.discover(tmp_path, "county")
    fs.attach_tiles(ts)
    assert fs.has_tiles("5yr") and not fs.has_tiles("2yr")
    # Hospital A sits at the tile centre (row 25, col 30): 5yr tile has fill 0 there -> tile depth 0 overrides index 40
    assert fs.depth_cm("5yr")[0] == 0 and fs.depth_cm("2yr")[2] == 15       # storm without tiles keeps the index
    assert fs.source("5yr") == ts.label and fs.source("2yr") == "index (grid)"
    inf = fs.info(0, "5yr")
    p5 = next(p for p in inf["per_storm"] if p["storm"] == "5yr")
    assert p5["tile_cm"] == 0 and p5["index_cm"] == 40 and p5["comparison"].disagree and p5["comparison"].delta_cm == -40
    assert next(p for p in inf["per_storm"] if p["storm"] == "2yr")["tile_cm"] is None
    ts.close()


def test_facilities_missing_file_message(tmp_path):
    with pytest.raises(F.FacilityError) as e:
        F.FacilitySet.load(tmp_path / "nope.geojson")
    assert "not found" in str(e.value)
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(F.FacilityError):
        F.FacilitySet.load(bad)


# ------------------------------------------------------------------------------------------------ roads
def road_doc():
    rows = [[11, "Alpha Road", 8, 0, 1000, 80, 80, 150], [12, "", 9, 0, 200, 100, 100, 90], [13, "Beta Street", 3, 2, 500, 40, 20, 30],
            [14, "Gamma Pike", 3, 1, 3000, 10, 5, 300], [15, "alpha Lane", 8, 0, 100, 50, 50, 15]]
    return {"format": "bgroads", "version": 1, "domain": "county", "storm": "100yr", "label": "100-year storm (county)", "source": "rasters",
            "classes": ["motorway", "motorway_link", "trunk", "primary", "primary_link", "secondary", "tertiary", "unclassified", "residential", "other"],
            "columns": ["id", "name", "hw", "evac", "len_m", "wet_pct", "flooded_pct", "max_cm"], "n_roads": 99, "total_km": 50.0,
            "flooded_km": 1.2, "rows": rows}


def test_road_table_sort_filter(tmp_path):
    p = tmp_path / "layers" / "roads_flooded" / "county" / "100yr.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(road_doc()), encoding="utf-8")
    t = RR.RoadTable.load(tmp_path, "county", "100yr")
    ids = lambda v: [int(t.id[i]) for i in v]               # noqa: E731
    assert ids(t.view("flooded_pct")) == [12, 11, 15, 13, 14]
    assert ids(t.view("flooded_pct", ascending=True)) == [14, 13, 15, 11, 12]
    assert ids(t.view("max_cm")) == [14, 11, 12, 13, 15]
    assert ids(t.view("len_m")) == [14, 11, 13, 12, 15]
    assert ids(t.view("flooded_m")) == [11, 12, 14, 13, 15]         # 800, 200, 150, 100, 50 m
    assert ids(t.view("evac"))[:2] == [13, 14]                       # evac code 2 (major) before 1 (tagged), then the rest
    assert ids(t.view("name", ascending=True)) == [15, 11, 13, 14, 12]            # alpha lane, alpha road, beta, gamma, unnamed last
    assert sorted(ids(t.view(evac_only=True))) == [13, 14]
    assert sorted(ids(t.view(min_flooded_pct=50))) == [11, 12, 15]
    assert sorted(ids(t.view(text="alpha"))) == [11, 15] and sorted(ids(t.view(text="residential"))) == [11, 15] and ids(t.view(text="other")) == [12]
    assert t.name(1) == "(unnamed other)" and t.class_name(0) == "residential" and t.evac_text(3).startswith("evacuation route")
    assert t.flooded_m[0] == pytest.approx(800.0)


def test_road_table_missing_and_geometry(tmp_path):
    with pytest.raises(RR.RoadsError) as e:
        RR.RoadTable.load(tmp_path, "county", "100yr")
    assert "100yr" in str(e.value) and "missing" in str(e.value)
    with pytest.raises(RR.RoadsError):
        RR.RoadGeometry.load(tmp_path)
    gz = tmp_path / "layers" / "county" / "roads.geojson.gz"
    gz.parent.mkdir(parents=True)
    doc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"osm_id": 11}, "geometry": {"type": "LineString", "coordinates": [[-86.5, 36.9], [-86.4, 37.0]]}},
        {"type": "Feature", "properties": {"osm_id": 12}, "geometry": {"type": "MultiLineString", "coordinates": [[[-86.3, 36.8], [-86.2, 36.7]], [[-86.2, 36.7], [-86.1, 36.6]]]}}]}
    write_gz(gz, doc)
    g = RR.RoadGeometry.load(tmp_path)
    assert g.bbox(11) == (-86.5, 36.9, -86.4, 37.0) and g.bbox(12) == (-86.3, 36.6, -86.1, 36.8) and g.coords(99) is None


# ------------------------------------------------------------------------------------------------ report
def sample_assessment(tmp_path, tiles=True):
    doc, recs = make_doc()
    write_gz(tmp_path / "i.json.gz", doc)
    idx = A.AddressIndex.load(tmp_path / "i.json.gz")
    return asm.assess(idx, 5, None, None), idx


def test_report_html_content(tmp_path):
    a, idx = sample_assessment(tmp_path)
    d = R.from_assessment(a, "Test <Town>", current_storm="25yr")
    html = R.build_html(d)
    assert "not been validated against observed floods" in html.lower()
    for needle in ("generated", "Depth by storm", "index (grid)", "Uncertainty",
                   "colour-blind safe", "25-year storm", "&lt;Town&gt;", "YlGnBu", "svg"):
        assert needle in html, needle
    assert "<Town>" not in html                              # escaped
    assert re.search(r"20\d\d-\d\d-\d\d \d\d:\d\d", html)      # date generated
    assert "&gt;= 120 cm" in html                               # tiers-source severe: no false precision
    for c in R.TIER_COLORS[1:]:
        assert c in html
    # colour-blind safety: the four tier colours are strictly ordered by luminance
    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4          # noqa: E731
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    ls = [lum(c) for c in R.TIER_COLORS]
    assert all(x > y for x, y in zip(ls, ls[1:]))


def test_report_write_with_png_and_tiles_columns(tmp_path):
    a, idx = sample_assessment(tmp_path)
    for r in a.rows:                                          # pretend tiles exist: both numbers and a flagged difference
        r.tile_cm = r.index.depth_cm + 25
        r.comparison = T.compare(r.index.depth_cm, r.tile_cm)
        r.shown_cm, r.shown_tier, r.shown_source = r.tile_cm, T.tier_of_cm(r.tile_cm), "solver depth (tiles, 6.25 m; verification pending)"
    a.tile_label = "solver depth (tiles, 6.25 m; verification pending)"
    d = R.from_assessment(a, "County")
    rgb = (np.random.default_rng(1).integers(0, 255, (40, 60, 3))).astype(np.uint8)
    hp, pp = R.write_report(tmp_path / "out", d, rgb)
    assert hp.is_file() and pp.is_file() and pp.read_bytes()[:4] == b"\x89PNG"
    html = hp.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html and "verification pending" in html and "+25 cm" in html and "Tiles" in html
    assert "src=\"http" not in html and "<link" not in html and "<script" not in html        # self-contained


def test_report_live_fallback_says_current_storm_only():
    d = R.from_live("5 Elm St", -86.4, 37.0, "Bowling Green", "25yr", "25-year storm", 45, 2, True, "viewer snapshot", 12)
    html = R.build_html(d)
    assert "only the storm shown on screen" in html and "~45 cm" in html


# ------------------------------------------------------------------------------------------------ help content
def test_markdown_parser_and_sources(tmp_path):
    md = "# Title\n\nSome **bold** text with `code` and [a link](http://x).\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nx = 1\n```\n---\n"
    bl = H.parse_markdown(md)
    kinds = [b.kind for b in bl]
    assert kinds == ["h1", "p", "bullet", "bullet", "table", "code", "hr"]
    assert bl[1].text == "Some bold text with code and a link." and bl[4].rows == [["a", "b"], ["1", "2"]]
    blocks, path, msg = H.storm_sources(tmp_path)
    assert blocks is None and "not documented yet" in msg and path.name == "STORM_DATA_SOURCES.md"
    (tmp_path / "STORM_DATA_SOURCES.md").write_text("# Sources\n\n## 100yr\n\n- NOAA Atlas 14\n", encoding="utf-8")
    blocks, path, msg = H.storm_sources(tmp_path)
    assert msg == "" and [b.kind for b in blocks] == ["h1", "h2", "bullet"]


def test_help_search_and_data_files(tmp_path):
    tops = H.topics()
    assert H.search(tops, "") == []
    hit = H.search(tops, "facilities ring")
    assert hit and hit[0][0].key in ("start", "facilities")
    assert any(g == "Search and reports" for g, k, d in H.search_shortcuts("ctrl+f"))
    assert H.search(tops, "qqqqzzzz") == []
    rows = H.data_files(tmp_path, "county", {})
    labels = [r[0] for r in rows]
    assert "Address index" in labels and "Depth tiles (folder)" in labels and all(r[2] is False for r in rows[:5])
    keys = {k for g, items in H.SHORTCUTS for k, d in items}
    assert {"Ctrl+F  or  /", "M", "L", "Ctrl+P"} <= keys


# ------------------------------------------------------------------------------------------------ real data, independent checks
def reference_flood_of(idx_doc, storm):
    """The builder's reference decoder (scripts/build_address_index.flood_of), re-implemented line by line here."""
    f = idx_doc["flood"].get(storm, [])
    out, a = {}, 0
    for t in range(0, len(f), 5):
        a += f[t]
        out[a] = (f[t + 1], f[t + 2], f[t + 3], f[t + 4])
    return out


@real
@pytest.mark.parametrize("domain", ["city", "county"])
def test_real_index_lookup_equals_independent_decode(domain):
    path = A.index_path(DATA, domain)
    if not path.is_file():
        pytest.skip("index missing")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    idx = A.AddressIndex.load(path)
    rnd = random.Random(20260404)
    pick = rnd.sample(range(idx.n), 250)
    # make sure flooded records are covered too
    wet = [i for i in range(idx.n) if idx.depth_cm[idx.storm_keys[-1]][i] > 0]
    pick += rnd.sample(wet, min(100, len(wet)))
    ref = {s["key"]: reference_flood_of(raw, s["key"]) for s in raw["storms"]}
    starts = raw["streets"]["start"]
    import bisect
    for i in pick:
        rec = idx.record(i)
        si = bisect.bisect_right(starts, i) - 1
        assert rec.address == (raw["num"][i] + " " + raw["streets"]["disp"][si]).strip()
        assert rec.lng == pytest.approx(raw["origin"][0] + raw["x"][i] / raw["scale"], abs=1e-9)
        assert rec.lat == pytest.approx(raw["origin"][1] + raw["y"][i] / raw["scale"], abs=1e-9)
        for row in idx.storm_rows(i):
            cm, tier, first, hrs = ref[row.key].get(i, (0, 0, 0, 0))
            assert (row.depth_cm, row.tier, row.first_hour, row.hours) == (cm, tier, first, hrs), (domain, i, row.key)
            assert row.wet == (i in ref[row.key])
    # typeahead finds every sampled address by its own text
    for i in pick[:60]:
        r = idx.record(i)
        if not r.num:
            continue
        hits = idx.search(r.address, limit=40)
        assert any(h.kind == "address" and h.label == r.address for h in hits), r.address


@real
def test_real_tiles_and_facilities_equal_independent_rasterio():
    import rasterio
    from rasterio.warp import transform
    ts = T.TileSet.discover(DATA, "county")
    if ts is None:
        pytest.skip("no depth tiles")
    fs = F.FacilitySet.load(F.facilities_path(DATA, "county"))
    assert fs.n == 98
    storms = ["2yr", "100yr", "ev4"]
    fs.attach_tiles(ts, storms)
    xs, ys = transform("EPSG:4326", "EPSG:3857", list(fs.lng), list(fs.lat))
    for s in storms:
        with rasterio.open(ts.paths[s]) as ds:
            for k in range(fs.n):
                try:
                    r, c = ds.index(xs[k], ys[k])
                    ok = 0 <= r < ds.height and 0 <= c < ds.width
                except Exception:
                    ok = False
                exp = -1
                if ok:
                    v = int(ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
                    exp = -1 if v == 65535 else v
                assert fs.d_tiles[s][k] == exp, (s, fs.items[k].name, fs.d_tiles[s][k], exp)
    ts.close()
