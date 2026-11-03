"""A small OpenStreetMap PBF reader (zlib + hand-decoded protobuf, numpy) that extracts building footprints and roads
inside a lng/lat box.  Pure python + numpy: no osmium / pyrosm / geopandas needed.

    feats = read_pbf("kentucky-latest.osm.pbf", bbox=(west, south, east, north), progress=cb)
    feats.buildings   list[GeoJSON Feature]  Polygon, props height_m / ADDRESS / name / building / amenity / osm_id
    feats.roads       list[GeoJSON Feature]  LineString, props highway (one of bgv.data.roads.ROAD_CLASSES) / name / osm_id
    feats.stats       dict (blobs, nodes kept, ways seen, seconds ...)

What it reads: `OSMData` blobs (raw or zlib), `PrimitiveBlock`s with dense nodes and ways.  Closed ways tagged `building=*`
(except `no`) become polygons; ways tagged `highway=*` of a driveable class become lines.  Multipolygon relations are NOT
read (a few large complex buildings are missing), nodes with tags are ignored.  The file must be sorted the usual way (all
nodes before the first way), which every Geofabrik / planet extract is; an unsorted file raises a clear error.

How it stays fast: dense-node and way-reference arrays are packed varints, decoded with one vectorised numpy pass per
block (`decode_varints`); only nodes inside the box (plus a margin) are kept (about 1.5 M of the 30 M of Kentucky); the
tag test of a way happens before its node references are touched.  Kentucky (150 MB) takes about a minute, once: the
caller caches the result (bgv.data.countyscene).
"""
from __future__ import annotations

import re
import struct
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

VERSION = "osm3"
NODE_MARGIN_DEG = 0.02            # nodes this far outside the box are kept so ways crossing the box edge stay complete

# highway=* values that are driveable and how they map onto bgv.data.roads.ROAD_CLASSES
HIGHWAY_CLASS = {
    "motorway": "motorway", "motorway_link": "motorway_link", "trunk": "trunk", "trunk_link": "motorway_link",
    "primary": "primary", "primary_link": "primary_link", "secondary": "secondary", "secondary_link": "primary_link",
    "tertiary": "tertiary", "tertiary_link": "primary_link", "unclassified": "unclassified", "residential": "residential",
    "living_street": "residential", "road": "unclassified", "service": "other",
}

# default heights (m) by building=* value when the way has no height / building:levels
DEFAULT_HEIGHT = {
    "garage": 3.0, "garages": 3.0, "carport": 3.0, "shed": 2.5, "hut": 2.5, "cabin": 4.0, "roof": 3.5, "greenhouse": 3.5,
    "barn": 7.0, "farm_auxiliary": 5.0, "stable": 4.5, "static_caravan": 3.0, "service": 4.0, "kiosk": 3.0,
    "house": 6.0, "residential": 6.0, "detached": 6.0, "semidetached_house": 6.0, "bungalow": 4.5, "terrace": 7.0,
    "apartments": 12.0, "dormitory": 12.0, "hotel": 14.0, "hospital": 14.0, "school": 7.0, "university": 10.0,
    "college": 9.0, "church": 9.0, "chapel": 7.0, "cathedral": 20.0, "civic": 8.0, "public": 8.0, "government": 9.0,
    "commercial": 7.0, "retail": 6.0, "supermarket": 7.0, "office": 10.0, "industrial": 9.0, "warehouse": 8.0,
    "manufacture": 9.0, "stadium": 18.0, "parking": 6.0, "train_station": 8.0, "hangar": 10.0, "silo": 12.0,
}
LEVEL_HEIGHT_M = 3.2
# service=* values that are parking aisles / driveways: real asphalt, but 19 000 of them add nothing at county scale
SKIP_SERVICE = frozenset({"parking_aisle", "driveway", "drive-through", "drive_through", "emergency_access"})
NEAR_ROAD_M = 90.0                # a building without an address is labelled "near <street>" when a named road is this close


# ------------------------------------------------------------------------------------------------ protobuf pieces
def decode_varints(buf: np.ndarray) -> np.ndarray:
    """All varints of a byte array of complete varints -> uint64[n] (vectorised, any length up to 10 bytes)."""
    buf = np.asarray(buf, np.uint8)
    if buf.size == 0:
        return np.zeros(0, np.uint64)
    ends = np.flatnonzero(buf < 128)
    if ends.size == 0:
        raise ValueError("truncated varint array")
    starts = np.empty(ends.size, np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lens = ends - starts + 1
    if int(lens.max()) == 1:                       # the common case for small deltas
        return buf[ends].astype(np.uint64)
    pos = np.arange(buf.size, dtype=np.int64) - np.repeat(starts, lens)
    part = (buf & 0x7F).astype(np.uint64) << (pos.astype(np.uint64) * np.uint64(7))
    return np.add.reduceat(part, starts)


def unzigzag(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, np.uint64)
    return (u >> np.uint64(1)).astype(np.int64) ^ -((u & np.uint64(1)).astype(np.int64))


def _varint(buf, i: int):
    """(value, next index) of the varint at buf[i]."""
    b = buf[i]
    if b < 128:
        return b, i + 1
    v = b & 0x7F
    shift = 7
    i += 1
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        if b < 128:
            return v, i
        shift += 7


def _packed_uints(buf, s: int, e: int) -> list:
    out = []
    while s < e:
        v, s = _varint(buf, s)
        out.append(v)
    return out


def _fields(buf, s: int, e: int):
    """Yield (field number, wire type, value) of the message at buf[s:e]; value = int (varint / fixed) or (start, end)."""
    while s < e:
        key, s = _varint(buf, s)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, s = _varint(buf, s)
            yield f, 0, v
        elif wt == 2:
            n, s = _varint(buf, s)
            yield f, 2, (s, s + n)
            s += n
        elif wt == 1:
            yield f, 1, int.from_bytes(buf[s:s + 8], "little")
            s += 8
        elif wt == 5:
            yield f, 5, int.from_bytes(buf[s:s + 4], "little")
            s += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")


def iter_blobs(fh):
    """(type, payload bytes) of every file block: 4-byte big-endian header length, BlobHeader, Blob."""
    while True:
        h = fh.read(4)
        if not h:
            return
        if len(h) < 4:
            raise ValueError("truncated PBF (block header length)")
        hl = struct.unpack(">I", h)[0]
        hdr = fh.read(hl)
        btype, size = "", 0
        for f, wt, v in _fields(hdr, 0, len(hdr)):
            if f == 1 and wt == 2:
                btype = hdr[v[0]:v[1]].decode("ascii", "replace")
            elif f == 3 and wt == 0:
                size = v
        blob = fh.read(size)
        if len(blob) < size:
            raise ValueError("truncated PBF (blob)")
        raw = None
        for f, wt, v in _fields(blob, 0, len(blob)):
            if f == 1 and wt == 2:
                raw = bytes(blob[v[0]:v[1]])
            elif f == 3 and wt == 2:
                raw = zlib.decompress(blob[v[0]:v[1]])
            elif f in (4, 5, 6, 7) and wt == 2:
                raise ValueError("PBF blob uses an unsupported compression (only raw and zlib are read)")
        yield btype, (raw if raw is not None else b"")


# ------------------------------------------------------------------------------------------------ blocks
@dataclass
class _Block:
    strings: list
    groups: list                # [(start, end)]
    gran: int = 100
    lat_off: int = 0
    lon_off: int = 0


def _parse_block(buf: bytes) -> _Block:
    strings, groups = [], []
    gran, lat_off, lon_off = 100, 0, 0
    for f, wt, v in _fields(buf, 0, len(buf)):
        if f == 1 and wt == 2:
            for f2, wt2, v2 in _fields(buf, v[0], v[1]):
                if f2 == 1 and wt2 == 2:
                    strings.append(buf[v2[0]:v2[1]])
        elif f == 2 and wt == 2:
            groups.append(v)
        elif f == 17 and wt == 0:
            gran = v
        elif f == 19 and wt == 0:
            lat_off = v if v < (1 << 63) else v - (1 << 64)
        elif f == 20 and wt == 0:
            lon_off = v if v < (1 << 63) else v - (1 << 64)
    return _Block(strings, groups, gran, lat_off, lon_off)


def _dense_nodes(buf, s, e, blk: _Block):
    """(ids int64, lon deg, lat deg) of a DenseNodes message."""
    sl = {}
    for f, wt, v in _fields(buf, s, e):
        if f in (1, 8, 9) and wt == 2:
            sl[f] = v
    if 1 not in sl or 8 not in sl or 9 not in sl:
        return None
    arr = np.frombuffer(buf, np.uint8)
    ids = np.cumsum(unzigzag(decode_varints(arr[sl[1][0]:sl[1][1]])))
    lat = np.cumsum(unzigzag(decode_varints(arr[sl[8][0]:sl[8][1]])))
    lon = np.cumsum(unzigzag(decode_varints(arr[sl[9][0]:sl[9][1]])))
    g = blk.gran * 1e-9
    return ids, (blk.lon_off + lon) * g, (blk.lat_off + lat) * g


@dataclass
class _RawWay:
    wid: int
    tags: dict
    refs: tuple                 # (start, end) of the packed refs inside the block buffer


def _ways(buf, s, e, blk: _Block, want_keys: dict):
    """Ways of one PrimitiveGroup that carry a building / highway tag -> list[_RawWay]; the cheap key test runs before
    anything else is decoded."""
    strings = blk.strings
    out = []
    n_all = 0
    for f, wt, v in _fields(buf, s, e):
        if f != 3 or wt != 2:
            continue
        n_all += 1
        wid, keys, vals, refs = 0, None, None, None
        for f2, wt2, v2 in _fields(buf, v[0], v[1]):
            if f2 == 1:
                wid = v2
            elif f2 == 2:
                keys = v2
            elif f2 == 3:
                vals = v2
            elif f2 == 8:
                refs = v2
        if keys is None or refs is None or vals is None or refs[1] <= refs[0]:
            continue
        ks = _packed_uints(buf, keys[0], keys[1])
        hit = False
        for k in ks:
            if k in want_keys:
                hit = True
                break
        if not hit:
            continue
        vs = _packed_uints(buf, vals[0], vals[1])
        tags = {}
        for k, val in zip(ks, vs):
            try:
                tags[strings[k].decode("utf-8", "replace")] = strings[val].decode("utf-8", "replace")
            except IndexError:
                pass
        out.append(_RawWay(wid, tags, refs))
    return out, n_all


# ------------------------------------------------------------------------------------------------ tag helpers
_NUM = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*(m|meter|meters|metre|metres|ft|feet|')?\s*$", re.I)


def parse_height(v) -> float:
    """OSM height / est_height text ("12", "12.5 m", "40 ft", "40'") -> metres, NaN when unusable."""
    if v is None:
        return float("nan")
    m = _NUM.match(str(v))
    if not m:
        return float("nan")
    x = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "m").lower()
    if unit in ("ft", "feet", "'"):
        x *= 0.3048
    return x if 0.0 < x < 600.0 else float("nan")


def building_height(tags: dict) -> float:
    """height / building:levels / a default by building type (always > 0)."""
    h = parse_height(tags.get("height"))
    if np.isfinite(h):
        return h
    h = parse_height(tags.get("est_height"))
    if np.isfinite(h):
        return h
    try:
        lv = float(str(tags.get("building:levels", "")).replace(",", "."))
        if 0 < lv < 200:
            return lv * LEVEL_HEIGHT_M + 1.0              # + roof
    except ValueError:
        pass
    return DEFAULT_HEIGHT.get(str(tags.get("building", "yes")).lower(), 6.0)


def address_of(tags: dict) -> str:
    num = tags.get("addr:housenumber", "").strip()
    street = tags.get("addr:street", "").strip()
    if num and street:
        return f"{num} {street}"
    if street:
        return street
    if num:
        return num
    return ""


# ------------------------------------------------------------------------------------------------ the reader
@dataclass
class OsmFeatures:
    buildings: list = field(default_factory=list)
    roads: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _rounded(a: np.ndarray) -> list:
    return np.round(a, 7).tolist()


def read_pbf(path, bbox, *, progress=None, margin_deg: float = NODE_MARGIN_DEG, max_blobs: int | None = None) -> OsmFeatures:
    """Buildings and roads of the PBF inside bbox = (west, south, east, north) degrees.  progress(msg, fraction)."""
    import os
    west, south, east, north = (float(x) for x in bbox)
    mw, ms, me, mn = west - margin_deg, south - margin_deg, east + margin_deg, north + margin_deg
    t0 = time.perf_counter()
    size = os.path.getsize(path)
    want = {}
    id_l, lon_l, lat_l = [], [], []
    node_ids = node_lon = node_lat = None
    seen_way = False
    res = OsmFeatures()
    n_blobs = n_nodes = n_kept = n_ways = 0
    t_prog = 0.0

    def rep(frac):
        nonlocal t_prog
        now = time.perf_counter()
        if progress is not None and now - t_prog > 0.25:
            t_prog = now
            try:
                progress(f"reading OpenStreetMap data: {len(res.buildings):,} buildings, {len(res.roads):,} roads", frac)
            except Exception:
                pass

    with open(path, "rb") as fh:
        for btype, raw in iter_blobs(fh):
            if btype != "OSMData":
                continue
            n_blobs += 1
            if max_blobs is not None and n_blobs > max_blobs:
                break
            blk = _parse_block(raw)
            want = {}
            for i, sb in enumerate(blk.strings):
                if sb == b"building" or sb == b"highway":
                    want[i] = sb
            for gs, ge in blk.groups:
                dense = None
                has_way = False
                for f, wt, v in _fields(raw, gs, ge):
                    if f == 2 and wt == 2:
                        dense = v
                    elif f == 3 and wt == 2:
                        has_way = True
                        break
                if dense is not None:
                    if seen_way:
                        raise ValueError("this PBF is not sorted (a node block follows way blocks): run `osmium sort` first")
                    r = _dense_nodes(raw, dense[0], dense[1], blk)
                    if r is not None:
                        ids, lon, lat = r
                        n_nodes += ids.size
                        m = (lon >= mw) & (lon <= me) & (lat >= ms) & (lat <= mn)
                        if m.any():
                            id_l.append(ids[m])
                            lon_l.append(lon[m])
                            lat_l.append(lat[m])
                            n_kept += int(m.sum())
                if has_way:
                    if not seen_way:
                        seen_way = True
                        if id_l:
                            node_ids = np.concatenate(id_l)
                            node_lon = np.concatenate(lon_l)
                            node_lat = np.concatenate(lat_l)
                            o = np.argsort(node_ids, kind="stable")
                            node_ids, node_lon, node_lat = node_ids[o], node_lon[o], node_lat[o]
                        else:
                            node_ids = np.zeros(0, np.int64)
                            node_lon = node_lat = np.zeros(0)
                        id_l = lon_l = lat_l = None
                    if want:
                        ways, n_all = _ways(raw, gs, ge, blk, want)
                        n_ways += n_all
                        if ways:
                            _resolve(raw, ways, node_ids, node_lon, node_lat, (west, south, east, north), res)
            rep(min(fh.tell() / max(size, 1), 0.999))
    res.stats = {"blobs": n_blobs, "nodes_seen": n_nodes, "nodes_kept": n_kept, "ways_seen": n_ways,
                 "buildings": len(res.buildings), "roads": len(res.roads), "seconds": time.perf_counter() - t0,
                 "version": VERSION}
    if progress is not None:
        try:
            progress(f"OpenStreetMap: {len(res.buildings):,} buildings, {len(res.roads):,} roads", 1.0)
        except Exception:
            pass
    return res


def _resolve(raw, ways, node_ids, node_lon, node_lat, bbox, res: OsmFeatures):
    """Decode the node references of all tagged ways of a group at once, look the coordinates up, build the features."""
    west, south, east, north = bbox
    arr = np.frombuffer(raw, np.uint8)
    segs = [arr[w.refs[0]:w.refs[1]] for w in ways]
    lens = np.array([s.size for s in segs], np.int64)
    flat = np.concatenate(segs) if len(segs) > 1 else segs[0]
    term = flat < 128
    cs = np.cumsum(term)
    byte_end = np.cumsum(lens)
    byte_start = byte_end - lens
    n_end = cs[byte_end - 1]
    n_start = np.where(byte_start > 0, cs[np.maximum(byte_start - 1, 0)], 0)
    counts = (n_end - n_start).astype(np.int64)
    vals = unzigzag(decode_varints(flat))
    ids = np.cumsum(vals)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    base = np.where(first > 0, ids[np.maximum(first - 1, 0)], 0)
    ids = ids - np.repeat(base, counts)
    if node_ids.size:
        pos = np.searchsorted(node_ids, ids)
        pos[pos >= node_ids.size] = node_ids.size - 1
        ok = node_ids[pos] == ids
        lon = np.where(ok, node_lon[pos], np.nan)
        lat = np.where(ok, node_lat[pos], np.nan)
    else:
        ok = np.zeros(ids.size, bool)
        lon = lat = np.full(ids.size, np.nan)
    inside = ok & (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    # per way: number of nodes inside, number resolved
    cnt_in = np.add.reduceat(inside.astype(np.int64), first) if ids.size else np.zeros(len(ways), np.int64)
    cnt_ok = np.add.reduceat(ok.astype(np.int64), first) if ids.size else np.zeros(len(ways), np.int64)
    ends = first + counts
    for k, w in enumerate(ways):
        n = int(counts[k])
        if n < 2 or cnt_in[k] == 0:
            continue
        a, b = int(first[k]), int(ends[k])
        tags = w.tags
        bld = tags.get("building")
        if bld is not None and bld != "no" and n >= 4 and ids[a] == ids[b - 1] and cnt_ok[k] == n:
            ring = np.stack([lon[a:b], lat[a:b]], axis=1)
            if cnt_in[k] * 2 < n:                   # keep a footprint when most of it lies inside (its centre does)
                continue
            props = {"height_m": round(float(building_height(tags)), 2), "ADDRESS": address_of(tags),
                     "name": tags.get("name", ""), "building": bld, "amenity": tags.get("amenity", ""),
                     "osm_id": int(w.wid)}
            res.buildings.append({"type": "Feature", "properties": props,
                                  "geometry": {"type": "Polygon", "coordinates": [_rounded(ring)]}})
            continue
        hw = tags.get("highway")
        cls = HIGHWAY_CLASS.get(hw) if hw is not None else None
        if cls is not None and tags.get("area") != "yes" and tags.get("service") not in SKIP_SERVICE:
            good = ok[a:b]
            if good.sum() < 2:
                continue
            pts = np.stack([lon[a:b][good], lat[a:b][good]], axis=1)
            res.roads.append({"type": "Feature", "properties": {"highway": cls, "name": tags.get("name", ""),
                                                                "osm_id": int(w.wid), "osm_highway": hw},
                              "geometry": {"type": "LineString", "coordinates": _rounded(pts)}})


# ------------------------------------------------------------------------------------------------ labels
def label_near_roads(res: OsmFeatures, max_m: float = NEAR_ROAD_M) -> int:
    """Buildings without an OSM address get ADDRESS = "near <street name>" (nearest named road within `max_m`), so the
    at-risk list and its search box are useful where houses carry no address.  Returns the number of labels set."""
    from scipy.spatial import cKDTree
    pts, names = [], []
    name_ids: dict = {}
    for f in res.roads:
        nm = f["properties"].get("name")
        if not nm:
            continue
        c = np.asarray(f["geometry"]["coordinates"], np.float64)
        if c.shape[0] < 2:
            continue
        # densify to ~12 m so the nearest vertex is close to the nearest point of the line
        lat0 = float(c[:, 1].mean())
        kx = 111320.0 * np.cos(np.radians(lat0))
        seg = np.hypot(np.diff(c[:, 0]) * kx, np.diff(c[:, 1]) * 110540.0)
        n = np.maximum(np.ceil(seg / 12.0).astype(int), 1)
        xs = np.concatenate([np.linspace(c[i, 0], c[i + 1, 0], n[i], endpoint=False) for i in range(c.shape[0] - 1)] + [c[-1:, 0]])
        ys = np.concatenate([np.linspace(c[i, 1], c[i + 1, 1], n[i], endpoint=False) for i in range(c.shape[0] - 1)] + [c[-1:, 1]])
        pts.append(np.stack([xs, ys], axis=1))
        names.append(np.full(xs.size, name_ids.setdefault(nm, len(name_ids)), np.int32))
    if not pts or not res.buildings:
        return 0
    P = np.concatenate(pts)
    N = np.concatenate(names)
    inv = [None] * len(name_ids)
    for nm, i in name_ids.items():
        inv[i] = nm
    lat0 = float(P[:, 1].mean())
    kx = 111320.0 * np.cos(np.radians(lat0))
    tree = cKDTree(np.stack([P[:, 0] * kx, P[:, 1] * 110540.0], axis=1))
    todo = [i for i, f in enumerate(res.buildings) if not f["properties"].get("ADDRESS")]
    if not todo:
        return 0
    C = np.array([np.mean(np.asarray(res.buildings[i]["geometry"]["coordinates"][0], np.float64), axis=0) for i in todo])
    d, k = tree.query(np.stack([C[:, 0] * kx, C[:, 1] * 110540.0], axis=1), distance_upper_bound=max_m)
    n_set = 0
    for j, i in enumerate(todo):
        if np.isfinite(d[j]):
            res.buildings[i]["properties"]["ADDRESS"] = "near " + inv[int(N[int(k[j])])]
            n_set += 1
    return n_set
