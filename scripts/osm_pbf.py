r"""
osm_pbf.py - OpenStreetMap data for the county builders, WITHOUT osmium / geopandas.

It reuses the pure-python PBF reader of the desktop viewer (desktop/bgv/data/osmpbf.py, loaded read-only by file path, so the
`bgv` package and its GPU dependencies are never imported):

  county_scene(app, bbox, pbf)        -> (building features, road features) exactly as the desktop viewer cuts them
                                         (desktop cache folder is reused when one matches the PBF + box; otherwise the desktop
                                         reader runs once, about a minute, and the result is cached in <app>/cache/bgscripts)
  load_county_buildings(app, bbox)    -> geodata.BuildingSet (ids = OSM way ids)
  read_tagged(pbf, bbox)              -> {"pois": [...], "roads": [...]}  one more pass over the PBF for what the desktop reader drops:
        pois   : nodes and ways tagged amenity / emergency / healthcare / social_facility (hospital, clinic, fire_station, police,
                 school, shelter, assembly_point, community_centre, townhall, ems) with lng/lat and, for ways, the outline ring
        roads  : driveable highway ways with ALL their tags that matter here (name, ref, class, evacuation-route tags) and geometry
  classify(tags) -> infrastructure type or None

Multipolygon relations are not read (a campus mapped only as a relation is missing).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DESKTOP_OSMPBF = os.path.join(REPO, "desktop", "bgv", "data", "osmpbf.py")
_op = None


def desktop_reader():
    """The desktop viewer's osmpbf module (imported from its file; needs only numpy)."""
    global _op
    if _op is None:
        if not os.path.isfile(DESKTOP_OSMPBF):
            raise FileNotFoundError("desktop/bgv/data/osmpbf.py not found at %s" % DESKTOP_OSMPBF)
        spec = importlib.util.spec_from_file_location("bgv_osmpbf_for_scripts", DESKTOP_OSMPBF)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m                       # dataclasses looks the module up by name
        spec.loader.exec_module(m)
        for need in ("iter_blobs", "_parse_block", "_fields", "_dense_nodes", "_ways", "decode_varints", "unzigzag", "read_pbf",
                     "HIGHWAY_CLASS", "SKIP_SERVICE", "VERSION"):
            if not hasattr(m, need):
                raise RuntimeError("desktop osmpbf.py no longer has %r: update scripts/osm_pbf.py" % need)
        _op = m
    return _op


def bbox_rounded(bbox, pad=0.01):
    """The desktop's box rule (countyscene.frame_bbox): pad and round OUTWARD to 0.01 degree."""
    import math
    w, s, e, n = bbox
    r = lambda v, f: round(f(v * 100.0) / 100.0, 2)                    # noqa: E731
    return (r(w - pad, math.floor), r(s - pad, math.floor), r(e + pad, math.ceil), r(n + pad, math.ceil))


# ------------------------------------------------------------------------------------------------ desktop scene (buildings, roads)
def _desktop_cache_roots(app):
    roots = []
    if os.environ.get("BGV_CACHE"):
        roots.append(os.path.join(os.environ["BGV_CACHE"], "osm"))
    roots += [os.path.join(REPO, "desktop", "cache", "osm"), os.path.join(app, "desktop", "cache", "osm")]
    return roots


def _find_desktop_scene(app, pbf, bbox):
    size = os.path.getsize(pbf)
    for root in _desktop_cache_roots(app):
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            try:
                with open(os.path.join(p, "info.json"), encoding="utf-8") as f:
                    info = json.load(f)
                b = info["bbox"]
                if (info.get("pbf_size") == size and b[0] <= bbox[0] and b[1] <= bbox[1] and b[2] >= bbox[2] and b[3] >= bbox[3]
                        and os.path.isfile(os.path.join(p, "buildings.geojson")) and os.path.isfile(os.path.join(p, "roads.geojson"))):
                    return p, info
            except (OSError, ValueError, KeyError):
                continue
    return None, None


def county_scene(app, bbox, pbf=None, log=print):
    """-> (buildings, roads) GeoJSON feature lists of the county box, from the desktop cache or one desktop-reader pass."""
    pbf = pbf or os.path.join(app, "kentucky-latest.osm.pbf")
    if not os.path.isfile(pbf):
        raise FileNotFoundError("OpenStreetMap extract not found: %s" % pbf)
    box = bbox_rounded(bbox)
    folder, info = _find_desktop_scene(app, pbf, box)
    if folder:
        if log:
            log("  OSM scene: reusing the desktop cache %s" % folder)
    else:
        from geodata import cache_dir
        st = os.stat(pbf)
        key = hashlib.sha1(("%s|%s|%s|%r" % (desktop_reader().VERSION, st.st_size, st.st_mtime_ns, box)).encode()).hexdigest()[:14]
        folder = os.path.join(cache_dir(app), "osm_scene_" + key)
        if not (os.path.isfile(os.path.join(folder, "buildings.geojson")) and os.path.isfile(os.path.join(folder, "roads.geojson"))):
            op = desktop_reader()
            if log:
                log("  OSM scene: reading %s (about a minute) ..." % pbf)
            t0 = time.perf_counter()
            res = op.read_pbf(pbf, box)
            nl = op.label_near_roads(res)
            os.makedirs(folder, exist_ok=True)
            for nm, feats in (("buildings", res.buildings), ("roads", res.roads)):
                tmp = os.path.join(folder, nm + ".geojson.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"type": "FeatureCollection", "features": feats}, f, separators=(",", ":"), ensure_ascii=False)
                os.replace(tmp, os.path.join(folder, nm + ".geojson"))
            with open(os.path.join(folder, "info.json"), "w", encoding="utf-8") as f:
                json.dump(dict(res.stats, bbox=list(box), pbf_size=st.st_size, labelled_near_road=nl,
                               built_s=time.perf_counter() - t0), f)
            if log:
                log("  OSM scene: %d buildings, %d roads in %.0fs" % (len(res.buildings), len(res.roads), time.perf_counter() - t0))
    with open(os.path.join(folder, "buildings.geojson"), encoding="utf-8") as f:
        b = json.load(f)["features"]
    with open(os.path.join(folder, "roads.geojson"), encoding="utf-8") as f:
        r = json.load(f)["features"]
    return b, r


def load_county_buildings(app, bbox, pbf=None, log=print):
    """County footprints as a geodata.BuildingSet (ids = OSM way ids, address '' when the way has none; the desktop's
    'near <street>' labels are not addresses and are removed)."""
    import shapely
    from shapely.geometry import shape
    import geodata as gd
    feats, _ = county_scene(app, bbox, pbf, log)
    geoms, ids, addr, lu, props = [], [], [], [], []
    for f in feats:
        pr = f.get("properties") or {}
        try:
            g = shape(f["geometry"])
            if not g.is_valid:
                g = shapely.make_valid(g)
        except Exception:                                      # noqa: BLE001
            continue
        a = str(pr.get("ADDRESS") or "").strip()
        geoms.append(g)
        ids.append(int(pr.get("osm_id", -1)))
        addr.append("" if a.lower().startswith("near ") else a)
        lu.append("")
        props.append(pr)
    return gd.BuildingSet(ids, np.array(geoms, dtype=object), addr, lu, props, "county")


# ------------------------------------------------------------------------------------------------ tagged pass
POI_KEYS = (b"amenity", b"emergency", b"healthcare", b"social_facility")
SHELTER_TYPES = {"emergency", "disaster", "storm_shelter", "tornado_shelter", "bunker"}


def classify(tags):
    """OSM tags -> infrastructure type (hospital, clinic, fire_station, police, school, shelter, assembly_point,
    community_centre, townhall, ems) or None."""
    a = tags.get("amenity")
    e = tags.get("emergency")
    h = tags.get("healthcare")
    if a == "hospital" or h == "hospital":
        return "hospital"
    if a == "clinic" or h == "clinic":
        return "clinic"
    if a == "fire_station" or e == "fire_station":
        return "fire_station"
    if a == "police":
        return "police"
    if a == "school":
        return "school"
    if e == "assembly_point":
        return "assembly_point"
    if e == "ambulance_station":
        return "ems"
    if e == "shelter" or tags.get("social_facility") == "shelter" or (a == "shelter" and tags.get("shelter_type") in SHELTER_TYPES):
        return "shelter"
    if a == "community_centre":
        return "community_centre"
    if a == "townhall":
        return "townhall"
    return None


def evac_flag(tags):
    for k, v in tags.items():
        if "evacuation" in k and str(v).lower() not in ("no", "false", ""):
            return True
    return tags.get("emergency") == "evacuation_route"


def _node_tags(raw, s, e, blk, ids, lon, lat, key_ids, box):
    """Tagged dense nodes whose tag KEY is one of key_ids -> list of (id, lon, lat, tags) inside box."""
    op = desktop_reader()
    kvr = None
    for f, wt, v in op._fields(raw, s, e):
        if f == 10 and wt == 2:
            kvr = v
    if kvr is None or kvr[1] <= kvr[0] or not len(key_ids):
        return []
    kv = op.decode_varints(np.frombuffer(raw, np.uint8)[kvr[0]:kvr[1]]).astype(np.int64)
    z = kv == 0
    zi = np.flatnonzero(z)
    if zi.size == 0:
        return []
    node_of = np.cumsum(z) - z
    first = np.concatenate([[0], zi[:-1] + 1])
    pos = np.arange(kv.size) - first[np.minimum(node_of, first.size - 1)]
    hit = (~z) & (pos % 2 == 0) & np.isin(kv, key_ids)
    out = []
    w, so, ea, no = box
    for n in np.unique(node_of[hit]):
        if n >= ids.size or not (w <= lon[n] <= ea and so <= lat[n] <= no):
            continue
        seg = kv[first[n]:zi[n]]
        tags = {}
        for k in range(0, seg.size - 1, 2):
            try:
                tags[blk.strings[seg[k]].decode("utf-8", "replace")] = blk.strings[seg[k + 1]].decode("utf-8", "replace")
            except IndexError:
                pass
        out.append((int(ids[n]), float(lon[n]), float(lat[n]), tags))
    return out


def _way_coords(raw, ways, node_ids, node_lon, node_lat):
    """Node coordinates of the ways (vectorised like the desktop reader) -> list of (lon[], lat[], ok[])."""
    op = desktop_reader()
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
    vals = op.unzigzag(op.decode_varints(flat))
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
    out = []
    for k in range(len(ways)):
        a, b = int(first[k]), int(first[k] + counts[k])
        out.append((lon[a:b], lat[a:b], ok[a:b]))
    return out


def read_tagged(pbf, bbox, progress=None, margin_deg=0.02, max_blobs=None):
    """One pass over the PBF -> {"pois": [...], "roads": [...], "stats": {...}} inside bbox = (west, south, east, north)."""
    op = desktop_reader()
    w, s, e, n = (float(x) for x in bbox)
    mw, ms, me, mn = w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg
    t0 = time.perf_counter()
    size = os.path.getsize(pbf)
    id_l, lon_l, lat_l = [], [], []
    node_ids = node_lon = node_lat = None
    seen_way = False
    pois, roads = [], []
    n_blobs = 0
    t_prog = 0.0
    with open(pbf, "rb") as fh:
        for btype, raw in op.iter_blobs(fh):
            if btype != "OSMData":
                continue
            n_blobs += 1
            if max_blobs is not None and n_blobs > max_blobs:
                break
            blk = op._parse_block(raw)
            want = {}
            key_ids = []
            for i, sb in enumerate(blk.strings):
                if sb == b"highway" or sb in POI_KEYS or b"evacuation" in sb:
                    want[i] = sb
                if sb in POI_KEYS:
                    key_ids.append(i)
            key_ids = np.array(key_ids, dtype=np.int64)
            for gs, ge in blk.groups:
                dense = None
                has_way = False
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
                        for nid, nlon, nlat, tags in _node_tags(raw, dense[0], dense[1], blk, ids, lon, lat, key_ids, (w, s, e, n)):
                            t = classify(tags)
                            if t:
                                pois.append({"osm_id": "n%d" % nid, "type": t, "name": tags.get("name", ""), "lng": nlon,
                                             "lat": nlat, "ring": None, "tags": {k: tags[k] for k in ("amenity", "emergency", "operator") if k in tags}})
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
                    if want:
                        ways, _ = op._ways(raw, gs, ge, blk, want)
                        sel = []
                        for wy in ways:
                            t = wy.tags
                            hw = t.get("highway")
                            if hw is not None and hw in op.HIGHWAY_CLASS and t.get("area") != "yes" and t.get("service") not in op.SKIP_SERVICE:
                                sel.append((wy, "road"))
                            elif classify(t):
                                sel.append((wy, "poi"))
                        if sel:
                            coords = _way_coords(raw, [x[0] for x in sel], node_ids, node_lon, node_lat)
                            for (wy, kind), (lon, lat, ok) in zip(sel, coords):
                                t = wy.tags
                                if kind == "road":
                                    good = ok
                                    if good.sum() < 2:
                                        continue
                                    pts = np.stack([lon[good], lat[good]], axis=1)
                                    if not ((pts[:, 0] >= w) & (pts[:, 0] <= e) & (pts[:, 1] >= s) & (pts[:, 1] <= n)).any():
                                        continue
                                    roads.append({"osm_id": int(wy.wid), "name": t.get("name", ""), "ref": t.get("ref", ""),
                                                  "highway": op.HIGHWAY_CLASS[t["highway"]], "osm_highway": t["highway"],
                                                  "evac": evac_flag(t), "coords": np.round(pts, 6).tolist()})
                                else:
                                    if ok.sum() < 1:
                                        continue
                                    pts = np.stack([lon[ok], lat[ok]], axis=1)
                                    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                                    if not (w <= cx <= e and s <= cy <= n):
                                        continue
                                    closed = ok.all() and len(ok) >= 4
                                    pois.append({"osm_id": "w%d" % wy.wid, "type": classify(t), "name": t.get("name", ""),
                                                 "lng": cx, "lat": cy, "ring": np.round(pts, 6).tolist() if closed else None,
                                                 "tags": {k: t[k] for k in ("amenity", "emergency", "operator", "building") if k in t}})
            now = time.perf_counter()
            if progress is not None and now - t_prog > 0.5:
                t_prog = now
                progress("OSM tagged pass: %d POIs, %d roads" % (len(pois), len(roads)), min(fh.tell() / max(size, 1), 0.999))
    return {"pois": pois, "roads": roads, "stats": {"blobs": n_blobs, "pois": len(pois), "roads": len(roads),
                                                    "seconds": round(time.perf_counter() - t0, 1), "bbox": list(bbox)}}


def tagged(app, bbox, pbf=None, log=print):
    """read_tagged with a pickle cache in <app>/cache/bgscripts (keyed by PBF size + mtime, box, reader version)."""
    from geodata import cache_dir
    pbf = pbf or os.path.join(app, "kentucky-latest.osm.pbf")
    if not os.path.isfile(pbf):
        raise FileNotFoundError("OpenStreetMap extract not found: %s" % pbf)
    st = os.stat(pbf)
    key = hashlib.sha1(("tag3|%s|%s|%s|%r" % (desktop_reader().VERSION, st.st_size, st.st_mtime_ns, tuple(round(v, 4) for v in bbox))).encode()).hexdigest()[:14]
    path = os.path.join(cache_dir(app), "osm_tagged_%s.pkl" % key)
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                res = pickle.load(f)
            if log:
                log("  OSM tagged pass: cached (%d POIs, %d roads)" % (len(res["pois"]), len(res["roads"])))
            return res
        except Exception:                                      # noqa: BLE001
            pass
    if log:
        log("  OSM tagged pass: reading %s (about a minute) ..." % pbf)

    def prog(msg, frac):
        if log and int(frac * 100) % 20 == 0:
            pass

    res = read_tagged(pbf, bbox, progress=prog)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f, protocol=4)
    os.replace(path + ".tmp", path)
    if log:
        log("  OSM tagged pass: %d POIs, %d roads in %.0fs" % (len(res["pois"]), len(res["roads"]), res["stats"]["seconds"]))
    return res
