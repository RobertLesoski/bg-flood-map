r"""
build_address_index.py - one searchable, phone-sized file: "is MY address flooded, how deep, when, for how long" per storm.

    python scripts/build_address_index.py                    # city  -> <app>/layers/address_index.json.gz
    python scripts/build_address_index.py --domain county    # county-> <app>/layers/county/address_index.json.gz (sharded when > 8 MB)
    python scripts/build_address_index.py --domain both

Format and search algorithm: ADDRESS_INDEX.md (read it before writing a client).

Records (one per addressable place):
  * kind 0 BUILDING: all footprints that share one parcel + one address collapse into one record (located at the largest
    footprint; depth = max over every footprint of the record). Address = the parcel's site address (all_parcels.geojson),
    else the footprint's own ADDRESS / OSM addr tags ("near <street>" labels are NOT addresses and are dropped).
  * kind 1 LOT: a parcel with a real address (house number > 0) and no footprint in layers/buildings.geojson (county: OSM):
    its depth is the water at the lot's representative point.
Per storm (flow grids of the domain, only complete storms are used): max depth over the footprint (cm), tier, first hour
flooded (>= 0.1 m), hours flooded - the statistics rules of the apps (flowquery.py header).
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flowquery as fq      # noqa: E402
import geodata as gd        # noqa: E402

FORMAT = "bgaddr"
VERSION = 1
COORD_SCALE = 100000            # 1e-5 degree (about 1 m)
SINGLE_FILE_MAX_MB = 8.0

# ------------------------------------------------------------------------------------------------ address text
ABBREV = {
    "STREET": "ST", "ROAD": "RD", "DRIVE": "DR", "AVENUE": "AVE", "BOULEVARD": "BLVD", "LANE": "LN", "COURT": "CT",
    "CIRCLE": "CIR", "PARKWAY": "PKWY", "HIGHWAY": "HWY", "TERRACE": "TER", "PLACE": "PL", "TRAIL": "TRL",
    "SQUARE": "SQ", "EXPRESSWAY": "EXPY", "ALLEY": "ALY", "BYPASS": "BYP", "CROSSING": "XING", "POINT": "PT",
    "HEIGHTS": "HTS", "RIDGE": "RDG", "VALLEY": "VLY", "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}
UNIT_RE = re.compile(r"\s+(?:APT|APARTMENT|UNIT|STE|SUITE|BLDG|BUILDING|LOT|RM|ROOM|FL|FLOOR|SPACE|SPC|TRLR|TRAILER)\b.*$|\s+#.*$")
NUM_RE = re.compile(r"^(\d+[A-Z]?)(?:\s*-\s*\d+[A-Z]?)?(?:\s+\d/\d)?\s+(.*)$")
_UP_TOKENS = {"US", "KY", "SR", "CR", "NE", "NW", "SE", "SW", "PO", "II", "III", "IV", "I"}


def clean_text(s):
    s = str(s or "").upper().replace("&", " AND ")
    s = re.sub(r"[.'`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_address(raw):
    """Raw address -> (house number token, street text); '' for a missing part. Multi-number strings such as
    '401,403,405,1101, 1103 PEARL STREET, BEAUTY AVENUE' give ('401', 'PEARL STREET'). Unit designators are cut off."""
    s = clean_text(raw)
    if not s or s.startswith("NEAR "):
        return "", ""
    first_num = ""
    for p in (x.strip() for x in s.split(",")):
        if not p:
            continue
        if re.fullmatch(r"\d+[A-Z]?", p):
            first_num = first_num or p
            continue
        m = NUM_RE.match(p)
        if m:
            return first_num or m.group(1), UNIT_RE.sub("", m.group(2)).strip()
        return first_num, UNIT_RE.sub("", p).strip()
    return first_num, ""


def normalize_street(street, partial=False):
    """Upper case, punctuation removed, suffixes and directions abbreviated (STREET -> ST, WEST -> W ...).
    partial=True leaves the LAST token as typed (the user is still typing it)."""
    toks = re.sub(r"[^A-Z0-9 ]", " ", clean_text(street)).split()
    out = []
    for i, t in enumerate(toks):
        if partial and i == len(toks) - 1:
            out.append(t)
        else:
            out.append(ABBREV.get(t, t))
    return " ".join(out)


def normalize_address(raw):
    """'592 Interstate Drive' -> '592 INTERSTATE DR' (the form every key in the index is built from)."""
    num, street = parse_address(raw)
    st = normalize_street(street)
    return (num + " " + st).strip()


def display_street(street):
    """Readable street name: 'WEST VILLA DRIVE' -> 'West Villa Drive', '10TH AVENUE' -> '10th Avenue', 'US 31W BYPASS'."""
    out = []
    for t in re.sub(r"[^A-Z0-9 /-]", " ", clean_text(street)).split():
        if t in _UP_TOKENS:
            out.append(t)
        elif re.fullmatch(r"\d+(ST|ND|RD|TH)", t):
            out.append(t[:-2] + t[-2:].lower())
        elif re.fullmatch(r"\d+[A-Z]", t):
            out.append(t)
        elif t.startswith("MC") and len(t) > 3:
            out.append("Mc" + t[2:].capitalize())
        else:
            out.append(t.capitalize())
    return " ".join(out)


def _num_key(num):
    m = re.match(r"(\d+)(.*)", num or "")
    return (int(m.group(1)), m.group(2)) if m else (-1, num or "")


# ------------------------------------------------------------------------------------------------ records
class Records:
    """Columnar table of the addressable places (unsorted until sort())."""

    def __init__(self):
        self.cols = {k: [] for k in ("lng", "lat", "num", "street", "street_raw", "use", "val", "parcel", "bid", "kind", "nb")}
        self.members = []          # per record: list of building positions in the BuildingSet (empty for a lot)
        self.lot_pt = []           # per record: (lng, lat) of the lot point or None

    def add(self, lng, lat, num, street, street_raw, use, val, parcel, bid, kind, members, lot_pt=None):
        c = self.cols
        for k, v in (("lng", lng), ("lat", lat), ("num", num), ("street", street), ("street_raw", street_raw), ("use", use),
                     ("val", val), ("parcel", parcel), ("bid", bid), ("kind", kind), ("nb", len(members))):
            c[k].append(v)
        self.members.append(members)
        self.lot_pt.append(lot_pt)

    def __len__(self):
        return len(self.cols["lng"])

    def sort(self):
        c = self.cols
        order = sorted(range(len(self)), key=lambda i: (c["street"][i], _num_key(c["num"][i]), c["parcel"][i], c["bid"][i]))
        for k in c:
            c[k] = [c[k][i] for i in order]
        self.members = [self.members[i] for i in order]
        self.lot_pt = [self.lot_pt[i] for i in order]
        return order


def _good(num, street):
    return bool(street)


def make_records(bset, parcels, domain, include_lots=True, log=print):
    """BuildingSet + ParcelSet -> Records (unsorted). See the module docstring for the rules."""
    nb = len(bset.geoms)
    rep = shapely.point_on_surface(bset.geoms) if nb else np.array([], dtype=object)
    blng = shapely.get_x(rep) if nb else np.zeros(0)
    blat = shapely.get_y(rep) if nb else np.zeros(0)
    barea = shapely.area(bset.geoms) if nb else np.zeros(0)
    pj = gd.join_points_to_parcels(rep, parcels) if nb else np.zeros(0, np.int64)
    inbox = gd.in_box(blng, blat, domain.bbox) & np.isfinite(blng)
    groups = {}
    n_noaddr = n_out = 0
    for i in range(nb):
        if not inbox[i]:
            n_out += 1
            continue
        pi = int(pj[i])
        cand = []
        if pi >= 0:
            cand.append(parcels.address[pi])
        cand.append(bset.address[i])
        pick = None
        for a in cand:                                   # a real house number wins over the placeholder number 0
            num, st = parse_address(a)
            if st and num not in ("", "0"):
                pick = (num, st)
                break
        if pick is None:
            for a in cand:
                num, st = parse_address(a)
                if st:
                    pick = (num, st)
                    break
        if pick is None:
            n_noaddr += 1
            continue
        num, st = pick
        pva = parcels.pva[pi] if pi >= 0 else ""
        key = (pva, normalize_street(st), num) if pva else ("b%s" % bset.ids[i], normalize_street(st), num)
        g = groups.setdefault(key, {"m": [], "num": num, "st": st, "pi": pi})
        g["m"].append(i)
    rec = Records()
    pvas_with_building = set()
    for (pva, stn, num), g in groups.items():
        m = g["m"]
        main = max(m, key=lambda j: barea[j])
        pi = g["pi"]
        use = (parcels.land_use[pi] if pi >= 0 else "") or bset.land_use[main]
        val = int(round(parcels.value[pi])) if pi >= 0 else 0
        if pva:
            pvas_with_building.add(pva)
        rec.add(float(blng[main]), float(blat[main]), num, stn, g["st"], use, val, pva, bset.ids[main], 0, m)
    n_lots = 0
    if include_lots and parcels.n:
        rp = shapely.point_on_surface(parcels.geoms)
        plng, plat = shapely.get_x(rp), shapely.get_y(rp)
        seen = set()
        for pi in range(parcels.n):
            pva = parcels.pva[pi]
            if pva in pvas_with_building or (pva and pva in seen):
                continue
            num, st = parse_address(parcels.address[pi])
            if not st or num in ("", "0") or not np.isfinite(plng[pi]) or not gd.in_box(plng[pi], plat[pi], domain.bbox):
                continue
            if pva:
                seen.add(pva)
            rec.add(float(plng[pi]), float(plat[pi]), num, normalize_street(st), st, parcels.land_use[pi],
                    int(round(parcels.value[pi])), pva, -1, 1, [], (float(plng[pi]), float(plat[pi])))
            n_lots += 1
    if log:
        log("  records: %d buildings/addresses, %d lots; dropped %d footprints outside the domain, %d without an address"
            % (len(rec) - n_lots, n_lots, n_out, n_noaddr))
    rec.stats = {"footprints": nb, "outside_domain": int(n_out), "no_address": int(n_noaddr), "lots": n_lots}
    return rec


# ------------------------------------------------------------------------------------------------ flood per storm
def compute_flood(rec, bset, storms, log=print, hours_all=range(1, 25)):
    """-> {storm: dict(idx, cm, tier, first, hours)} (arrays over the records with any water, ascending record index)."""
    n = len(rec)
    out = {}
    lot_ids = [i for i in range(n) if rec.cols["kind"][i] == 1]
    bld_pos = np.array([j for i in range(n) for j in rec.members[i]], dtype=np.int64)
    bld_rec = np.array([i for i in range(n) for _ in rec.members[i]], dtype=np.int64)
    starts = np.zeros(n, dtype=np.int64)               # first slot of record i inside bld_pos (kind 0 only)
    cnt = np.array([len(rec.members[i]) for i in range(n)], dtype=np.int64)
    starts[1:] = np.cumsum(cnt)[:-1]
    has_b = cnt > 0
    cache = {}
    for s in storms:
        t0 = time.perf_counter()
        gk = s.grid.key()
        if gk not in cache:
            geoms = bset.geoms[bld_pos] if bld_pos.size else np.array([], dtype=object)
            bi = fq.building_pixels(geoms, s.grid)
            lots = np.array(lot_ids, dtype=np.int64)
            li = fq.index_from_points(np.array([rec.lot_pt[i][0] for i in lot_ids]), np.array([rec.lot_pt[i][1] for i in lot_ids]),
                                      s.grid) if lot_ids else fq.PixelIndex(np.zeros(1, np.int64), np.zeros(0, np.int64))
            cache.clear()
            cache[gk] = (bi, li, lots)
        bi, li, lots = cache[gk]
        mx = np.zeros(n, np.uint8)
        first = np.zeros(n, np.int16)
        nh = np.zeros(n, np.int16)

        def rec_codes(R, A):
            code = np.zeros(n, np.uint8)
            if bld_pos.size:
                bc = fq.max_code(R, A, bi)                               # per footprint slot
                code[has_b] = np.maximum.reduceat(bc, starts[has_b]) if bc.size else 0
            if lots.size:
                code[lots] = fq.max_code(R, A, li)
            return code

        for h in s.hours:
            if h not in hours_all:
                continue
            R, A = s.read_hour(h)
            c = rec_codes(R, A)
            mx = np.maximum(mx, c)
            fl = c >= fq.FLOOD_CODE
            nh += fl
            newly = fl & (first == 0)
            first[newly] = h
        wet = np.flatnonzero(mx > 0)
        d = fq.code_depth(mx[wet])
        out[s.name] = dict(idx=wet, cm=np.rint(d * 100).astype(np.int64), tier=fq.tier_of(d).astype(np.int64),
                           first=first[wet].astype(np.int64), hours=nh[wet].astype(np.int64),
                           n_flooded=int((mx >= fq.FLOOD_CODE).sum()))
        if log:
            log("  %-6s %-6s %d records with water, %d flooded (>= 0.1 m), deepest %.2f m   [%.1fs]"
                % (s.name, s.source, wet.size, out[s.name]["n_flooded"], float(d.max()) if d.size else 0.0,
                   time.perf_counter() - t0))
    return out


# ------------------------------------------------------------------------------------------------ encoding
COLUMNS = {
    "x": "longitude: lng = origin[0] + x / scale (integer, 1e-5 degree = about 1 m)",
    "y": "latitude: lat = origin[1] + y / scale",
    "num": "house number token as text ('' when the address has none), e.g. '592', '1500A'",
    "use": "index into use_legend (parcel LAND_USE code; '' = unknown)",
    "val": "assessed value in whole USD (parcel prop_value_usd), 0 = unknown",
    "parcel": "PVA_PARCEL id ('' = footprint outside every parcel)",
    "bid": "city: 0-based feature index in layers/buildings.geojson of the record's main footprint; county: OSM way id; -1 for a lot",
    "kind": "0 = building/address record, 1 = lot (no footprint, depth at the lot point)",
    "nb": "number of footprints folded into the record (0 for a lot)",
    "streets.key": "normalised street, sorted ascending; records are sorted by (street, house number)",
    "streets.disp": "display street name",
    "streets.start": "index of the first record of every street, plus n at the end (length = streets + 1)",
    "flood.<storm>": "flat int array, stride 5: [record index delta, depth cm, tier 0-4, first hour flooded (0 never), hours flooded] per record "
                     "with water; index delta is relative to the previous entry (first one relative to the file's first record)",
}


def _street_table(rec):
    keys, disp, start = [], [], []
    best = {}
    for i, k in enumerate(rec.cols["street"]):
        raw = rec.cols["street_raw"][i]
        if k not in best or len(raw) > len(best[k]):
            best[k] = raw
        if not keys or keys[-1] != k:
            keys.append(k)
            start.append(i)
    start.append(len(rec))
    return keys, [display_street(best[k]) for k in keys], start


def encode_block(rec, lo, hi, origin, flood, use_legend):
    c = rec.cols
    sl = slice(lo, hi)
    uidx = {u: i for i, u in enumerate(use_legend)}
    blk = {
        "x": [int(round((v - origin[0]) * COORD_SCALE)) for v in c["lng"][sl]],
        "y": [int(round((v - origin[1]) * COORD_SCALE)) for v in c["lat"][sl]],
        "num": c["num"][sl],
        "use": [uidx[u] for u in c["use"][sl]],
        "val": c["val"][sl],
        "parcel": c["parcel"][sl],
        "bid": c["bid"][sl],
        "kind": c["kind"][sl],
        "nb": c["nb"][sl],
    }
    fl = {}
    for name, f in flood.items():
        sel = (f["idx"] >= lo) & (f["idx"] < hi)
        idx = f["idx"][sel]
        flat = np.empty(idx.size * 5, dtype=np.int64)
        if idx.size:
            d = np.diff(idx, prepend=lo)
            flat[0::5] = d
            flat[1::5] = f["cm"][sel]
            flat[2::5] = f["tier"][sel]
            flat[3::5] = f["first"][sel]
            flat[4::5] = f["hours"][sel]
        fl[name] = flat.tolist()
    blk["flood"] = fl
    return blk


def build_payload(domain, rec, flood, storms, origin_extra=None):
    c = rec.cols
    n = len(rec)
    west = float(min(c["lng"])) if n else domain.bbox[0]
    south = float(min(c["lat"])) if n else domain.bbox[1]
    origin = [round(west, 5), round(south, 5)]
    use_legend = sorted(set(c["use"]), key=lambda u: (u == "", u))
    keys, disp, start = _street_table(rec)
    head = {
        "format": FORMAT, "version": VERSION, "domain": domain.key, "label": domain.label,
        "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n": n, "origin": origin, "scale": COORD_SCALE,
        "bbox": [round(v, 5) for v in domain.bbox],
        "use_legend": use_legend,
        "streets": {"key": keys, "disp": disp, "start": start},
        "storms": [{"key": s.name, "label": s.label, "source": s.source, "depth_in": s.meta.get("storm_depth_in"),
                    "hours": s.hours, "grid": {"w": s.grid.width, "h": s.grid.height, "cell_m": s.grid.cell_m},
                    "records_with_water": int(flood[s.name]["idx"].size), "records_flooded": flood[s.name]["n_flooded"]}
                   for s in storms],
        "tiers": {"edges_m": list(fq.TIER_EDGES_M), "names": list(fq.TIER_NAMES), "flooded_min_m": fq.FLOOD_MIN_M},
        "columns": COLUMNS,
        "rules": "depth = max over the footprint pixels (all-touched) of the flow grid; flooded >= 0.1 m; tier 0 = below 0.1 m, "
                 "1 shallow, 2 moderate, 3 deep, 4 severe (>= 1.2 m); hours counted over the hourly grids (hour N = end of hour N). "
                 "Records without a flood entry for a storm are dry there. 'tiers' source grids are an approximation (FLOW_FORMAT.md).",
        "stats": getattr(rec, "stats", {}),
    }
    return head, origin, use_legend


def write_index(domain, rec, flood, storms, out_path, max_single_mb=SINGLE_FILE_MAX_MB, log=print):
    """Write the single file (or, when it would exceed max_single_mb gz, a manifest + street-prefix shards). Returns info dict."""
    head, origin, use_legend = build_payload(domain, rec, flood, storms)
    n = len(rec)
    full = dict(head)
    full.update(encode_block(rec, 0, n, origin, flood, use_legend))
    size, raw = gd.write_json_gz(out_path, full)
    info = {"path": out_path, "gz_bytes": size, "json_bytes": raw, "sharded": False, "records": n}
    if size <= max_single_mb * 1e6:
        _clear_shards(out_path)
        return info
    # ---- sharded layout: manifest at out_path (street table + shard list, no records), shards in <dir>/address/<shard>.json.gz
    shard_dir = os.path.join(os.path.dirname(out_path), "address")
    os.makedirs(shard_dir, exist_ok=True)
    keys = head["streets"]["key"]
    start = head["streets"]["start"]
    first_char = [(k[:1] if k[:1].isalnum() else "_") for k in keys]
    shards, files = [], []
    i = 0
    while i < len(keys):
        j = i
        while j < len(keys) and first_char[j] == first_char[i]:
            j += 1
        lo, hi = start[i], start[j]
        blk = {"format": FORMAT, "version": VERSION, "domain": domain.key, "shard": first_char[i], "origin": origin,
               "scale": COORD_SCALE, "rec_lo": lo, "rec_hi": hi}
        blk.update(encode_block(rec, lo, hi, origin, flood, use_legend))
        fname = "%s.json.gz" % first_char[i]
        sz, _ = gd.write_json_gz(os.path.join(shard_dir, fname), blk)
        shards.append({"shard": first_char[i], "file": "address/" + fname, "street_lo": i, "street_hi": j,
                       "rec_lo": lo, "rec_hi": hi, "gz_bytes": sz})
        files.append(fname)
        i = j
    man = dict(head)
    man.update({"sharded": True, "shards": shards, "columns": COLUMNS})
    gd.write_json_gz(out_path, man)
    for f in os.listdir(shard_dir):
        if f.endswith(".json.gz") and f not in files:
            os.unlink(os.path.join(shard_dir, f))
    info.update(sharded=True, shards=len(shards), gz_bytes=sum(s["gz_bytes"] for s in shards) + os.path.getsize(out_path))
    return info


def _clear_shards(out_path):
    d = os.path.join(os.path.dirname(out_path), "address")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json.gz"):
                os.unlink(os.path.join(d, f))


# ------------------------------------------------------------------------------------------------ reference client
def load_index(path):
    """Load a single-file index or a sharded one (all shards merged) into one dict of arrays - reference for clients/tests."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        idx = json.load(f)
    if idx.get("sharded"):
        base = os.path.dirname(path)
        parts = []
        for sh in idx["shards"]:
            with gzip.open(os.path.join(base, sh["file"]), "rt", encoding="utf-8") as f:
                parts.append(json.load(f))
        for k in ("x", "y", "num", "use", "val", "parcel", "bid", "kind", "nb"):
            idx[k] = [v for p in parts for v in p[k]]
        names = {s["key"] for s in idx["storms"]}
        idx["flood"] = {}
        for nm in names:
            flat, prev_abs = [], None
            for p in parts:                                   # re-delta over the whole file
                f = p["flood"].get(nm, [])
                a = p["rec_lo"]
                for t in range(0, len(f), 5):
                    a += f[t]
                    flat += [a - (prev_abs if prev_abs is not None else 0), f[t + 1], f[t + 2], f[t + 3], f[t + 4]]
                    prev_abs = a
            idx["flood"][nm] = flat
    return idx


def flood_of(idx, storm):
    """-> {record index: (cm, tier, first, hours)} of one storm."""
    f = idx["flood"].get(storm, [])
    out, a = {}, 0
    for t in range(0, len(f), 5):
        a += f[t]
        out[a] = (f[t + 1], f[t + 2], f[t + 3], f[t + 4])
    return out


def record(idx, i):
    key = idx["streets"]["key"]
    s = bisect.bisect_right(idx["streets"]["start"], i) - 1
    lng = idx["origin"][0] + idx["x"][i] / idx["scale"]
    lat = idx["origin"][1] + idx["y"][i] / idx["scale"]
    return {"i": i, "address": (idx["num"][i] + " " + idx["streets"]["disp"][s]).strip(), "key": (idx["num"][i] + " " + key[s]).strip(),
            "lng": lng, "lat": lat, "use": idx["use_legend"][idx["use"][i]], "value": idx["val"][i], "parcel": idx["parcel"][i],
            "bid": idx["bid"][i], "kind": idx["kind"][i]}


def search(idx, query, limit=10):
    """Typeahead: '592 interstate d' -> records; 'interstate' -> first records of matching streets; '592' -> number match.
    Street names are matched by prefix on the normalised key (completed words are abbreviated, see ADDRESS_INDEX.md)."""
    q = clean_text(query)
    keys = idx["streets"]["key"]
    start = idx["streets"]["start"]
    if not q:
        return []
    m = re.match(r"^(\d+[A-Z]?)\s+(.*)$", q)
    num_q, street_q = (m.group(1), m.group(2)) if m else (("", q) if not re.fullmatch(r"\d+[A-Z]?", q) else (q, ""))
    out = []
    if street_q:
        trailing_space = street_q != street_q.rstrip() or query.endswith(" ")
        v1 = normalize_street(street_q, partial=not trailing_space)
        v2 = normalize_street(street_q, partial=False)
        seen = set()
        for pref in (v1, v2):
            if pref in seen:
                continue
            seen.add(pref)
            si = bisect.bisect_left(keys, pref)
            while si < len(keys) and keys[si].startswith(pref):
                for i in range(start[si], start[si + 1]):
                    if idx["num"][i].startswith(num_q):
                        out.append(record(idx, i))
                        if len(out) >= limit:
                            return out
                si += 1
        if not out and len(street_q) >= 3:                    # fallback: a word of the street starts with the text
            words = street_q.split()
            for si, k in enumerate(keys):
                if all(any(w2.startswith(w) for w2 in k.split()) for w in words):
                    for i in range(start[si], start[si + 1]):
                        if idx["num"][i].startswith(num_q):
                            out.append(record(idx, i))
                            if len(out) >= limit:
                                return out
    else:                                                     # number only: linear scan
        for i, nm in enumerate(idx["num"]):
            if nm.startswith(num_q):
                out.append(record(idx, i))
                if len(out) >= limit:
                    break
    return out


# ------------------------------------------------------------------------------------------------ driver
def build_domain(app, key, storms_filter=None, include_lots=True, max_single_mb=SINGLE_FILE_MAX_MB, pbf=None, out=None, log=print):
    t0 = time.perf_counter()
    dom = gd.get_domain(app, key)
    log("== address index: %s (%s) ==" % (dom.label, dom.flow_dir))
    storms = fq.load_storms(dom.flow_dir, storms_filter, log=log)
    if not storms:
        raise SystemExit("no complete storms in %s" % dom.flow_dir)
    log("  storms: %s" % " ".join(s.name for s in storms))
    log("  reading parcels ...")
    parcels = gd.load_parcels(app, log=log)
    if key == "city":
        bset = gd.load_city_buildings(app, log=log)
    else:
        import osm_pbf
        bset = osm_pbf.load_county_buildings(app, dom.bbox, pbf=pbf, log=log)
    log("  %d footprints, %d parcels" % (len(bset.geoms), parcels.n))
    rec = make_records(bset, parcels, dom, include_lots=include_lots, log=log)
    rec.sort()
    log("  flood statistics per storm ...")
    flood = compute_flood(rec, bset, storms, log=log)
    out_path = out or dom.layer("address_index.json.gz")
    info = write_index(dom, rec, flood, storms, out_path, max_single_mb=max_single_mb, log=log)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info["storms"] = [s.name for s in storms]
    log("  wrote %s: %d records, %.2f MB gz (%.2f MB json)%s in %.0fs" % (
        out_path, info["records"], info["gz_bytes"] / 1e6, info["json_bytes"] / 1e6,
        ", SHARDED into %d files" % info["shards"] if info["sharded"] else "", info["seconds"]))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the address search index (ADDRESS_INDEX.md).")
    ap.add_argument("--domain", choices=("city", "county", "both"), default="city")
    ap.add_argument("--app", default=gd.APP_DEFAULT, help="folder the web server serves (default %(default)s)")
    ap.add_argument("--storms", nargs="+", default=None, help="only these storms (default: every complete storm folder)")
    ap.add_argument("--no-lots", action="store_true", help="skip parcels without a footprint")
    ap.add_argument("--max-single-mb", type=float, default=SINGLE_FILE_MAX_MB, help="shard the output above this gz size")
    ap.add_argument("--pbf", default=None, help="OSM extract for the county (default <app>/kentucky-latest.osm.pbf)")
    ap.add_argument("--out", default=None, help="output file (single domain only)")
    a = ap.parse_args(argv)
    for key in (("city", "county") if a.domain == "both" else (a.domain,)):
        build_domain(a.app, key, a.storms, not a.no_lots, a.max_single_mb, a.pbf, a.out if a.domain != "both" else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
