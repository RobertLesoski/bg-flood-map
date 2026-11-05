"""Flooded-roads routing: the shortest route between two points over the road network that avoids road pieces whose solver depth
is at or above a threshold at a given hour, compared with the normal (dry-weather) shortest route.

Graph: preferably the OSM PBF (exact junctions by node id, bridge / tunnel tags); fallback the county road GeoJSON (its
simplification dropped most junctions: 8 620 disconnected pieces, so routes there are often "not connected").  One
undirected edge per consecutive vertex pair, length in metres (local equirectangular, < 0.1 % error over the county).
Depth of an edge: the maximum of the native solver tiles sampled every `SAMPLE_M` metres along it (tile cells are 6.25 m).

Honesty (ROUTE_NOTE): one-way rules, turn restrictions, speed and closures are ignored; road classes are not filtered (the
county file labels 60 % of ways 'other'); a sample with no tile data (outside the modelled area) counts as passable; edges tagged bridge / tunnel in OSM are never blocked
(the tile samples the water under the deck), their approaches are.
"""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_M = 6.0
R_EARTH = 6371008.8
ROUTE_NOTE = ("solver depth on the road centre line; ignores one-way rules, turn restrictions and closures; OSM bridges / tunnels "
              "never blocked; no-data counts as passable; model not validated")


@dataclass
class RoadGraph:
    lng: np.ndarray            # float64[n] node coordinates
    lat: np.ndarray
    x: np.ndarray              # float64[n] local metres (equirectangular about lat0)
    y: np.ndarray
    ea: np.ndarray             # int32[m] edge endpoints
    eb: np.ndarray
    elen: np.ndarray           # float64[m] metres
    s_edge: np.ndarray         # int32[k] edge of each depth sample
    s_lng: np.ndarray          # float64[k]
    s_lat: np.ndarray
    lat0: float
    bridge: np.ndarray = None   # bool[m] OSM bridge / tunnel: depth under it does not block

    @property
    def n(self):
        return int(self.lng.size)

    @property
    def m(self):
        return int(self.ea.size)


def _xy(lng, lat, lat0):
    k = np.pi / 180.0 * R_EARTH
    return np.asarray(lng) * k * np.cos(np.radians(lat0)), np.asarray(lat) * k


def graph_from_features(features, sample_m: float = SAMPLE_M) -> RoadGraph:
    lines = []
    for f in features:
        g = (f or {}).get("geometry") or {}
        if g.get("type") == "LineString":
            lines.append(g.get("coordinates") or [])
        elif g.get("type") == "MultiLineString":
            lines.extend(g.get("coordinates") or [])
    pts, segs, i0 = [], [], 0
    for c in lines:
        if len(c) < 2:
            continue
        a = np.asarray(c, np.float64)[:, :2]
        pts.append(a)
        idx = np.arange(i0, i0 + len(a))
        i0 += len(a)
        segs.append(np.stack([idx[:-1], idx[1:]], 1))
    if not pts:
        raise ValueError("no LineString roads")
    allp = np.concatenate(pts)
    q = np.round(allp * 1e7).astype(np.int64) + (1 << 31)               # 1e-7 deg ~ 1 cm; one int64 key per vertex
    uk, inv = np.unique((q[:, 0] << 32) | q[:, 1], return_inverse=True)
    inv = inv.reshape(-1)
    lng, lat = ((uk >> 32) - (1 << 31)) / 1e7, ((uk & 0xFFFFFFFF) - (1 << 31)) / 1e7
    s = np.concatenate(segs)
    ea, eb = inv[s[:, 0]], inv[s[:, 1]]
    keep = ea != eb
    ea, eb = ea[keep], eb[keep]
    return _finish(lng, lat, ea[keep], eb[keep], np.zeros(int(keep.sum()), bool), sample_m)


def _finish(lng, lat, ea, eb, br, sample_m=SAMPLE_M) -> RoadGraph:
    n = len(lng)
    lo, hi = np.minimum(ea, eb).astype(np.int64), np.maximum(ea, eb).astype(np.int64)
    pk, first = np.unique(lo * n + hi, return_index=True)               # drop duplicate edges (overlapping ways)
    ea, eb, br = (pk // n).astype(np.int32), (pk % n).astype(np.int32), np.asarray(br, bool)[first]
    lat0 = float(np.mean(lat))
    x, y = _xy(lng, lat, lat0)
    elen = np.hypot(x[eb] - x[ea], y[eb] - y[ea])
    nseg = np.maximum(1, np.ceil(elen / sample_m).astype(np.int64))    # depth samples: both ends + every sample_m
    cnt = nseg + 1
    s_edge = np.repeat(np.arange(ea.size, dtype=np.int32), cnt)
    starts = np.cumsum(cnt) - cnt
    frac = (np.arange(s_edge.size) - np.repeat(starts, cnt)) / np.repeat(nseg, cnt)
    s_lng = lng[ea][s_edge] + (lng[eb][s_edge] - lng[ea][s_edge]) * frac
    s_lat = lat[ea][s_edge] + (lat[eb][s_edge] - lat[ea][s_edge]) * frac
    return RoadGraph(np.asarray(lng, float), np.asarray(lat, float), x, y, ea, eb, elen, s_edge, s_lng, s_lat, lat0, br)


def _osm_ways(pbf, bbox, progress=None):
    """Driveable OSM ways of the PBF inside bbox -> list of (node_ids int64[k], lon[k], lat[k], bridge_or_tunnel bool).
    Uses the node IDs (not coordinates) so junctions are exact; the county GeoJSON lost them in simplification."""
    from . import osmpbf as op
    west, south, east, north = (float(v) for v in bbox)
    mg = op.NODE_MARGIN_DEG
    out, idl, lonl, latl = [], [], [], []
    node_ids = node_lon = node_lat = None
    with open(pbf, "rb") as fh:
        for btype, raw in op.iter_blobs(fh):
            if btype != "OSMData":
                continue
            blk = op._parse_block(raw)
            want = {i: sb for i, sb in enumerate(blk.strings) if sb == b"highway"}
            for gs, ge in blk.groups:
                dense, has_way = None, False
                for f, wt, v in op._fields(raw, gs, ge):
                    if f == 2 and wt == 2:
                        dense = v
                    elif f == 3 and wt == 2:
                        has_way = True
                        break
                if dense is not None and node_ids is None:
                    r = op._dense_nodes(raw, dense[0], dense[1], blk)
                    if r is not None:
                        ids, lon, lat = r
                        m = (lon >= west - mg) & (lon <= east + mg) & (lat >= south - mg) & (lat <= north + mg)
                        idl.append(ids[m]); lonl.append(lon[m]); latl.append(lat[m])
                if has_way:
                    if node_ids is None:
                        node_ids, node_lon, node_lat = np.concatenate(idl), np.concatenate(lonl), np.concatenate(latl)
                        o = np.argsort(node_ids, kind="stable")
                        node_ids, node_lon, node_lat = node_ids[o], node_lon[o], node_lat[o]
                    if not want:
                        continue
                    ways, _ = op._ways(raw, gs, ge, blk, want)
                    arr = np.frombuffer(raw, np.uint8)
                    for w in ways:
                        t = w.tags
                        if op.HIGHWAY_CLASS.get(t.get("highway")) is None or t.get("area") == "yes" or t.get("service") in op.SKIP_SERVICE:
                            continue
                        refs = np.cumsum(op.unzigzag(op.decode_varints(arr[w.refs[0]:w.refs[1]])))
                        pos = np.clip(np.searchsorted(node_ids, refs), 0, max(node_ids.size - 1, 0))
                        ok = node_ids[pos] == refs if node_ids.size else np.zeros(refs.size, bool)
                        if ok.sum() < 2:
                            continue
                        lo, la = node_lon[pos][ok], node_lat[pos][ok]
                        if not ((lo >= west) & (lo <= east) & (la >= south) & (la <= north)).any():
                            continue
                        bt = t.get("bridge", "no") not in ("no", "") or t.get("tunnel", "no") not in ("no", "")
                        out.append((refs[ok], lo, la, bt))
            if progress is not None:
                progress(fh.tell() / max(1, __import__("os").path.getsize(pbf)))
    return out


def graph_from_osm(ways, sample_m: float = SAMPLE_M) -> RoadGraph:
    allid = np.concatenate([w[0] for w in ways])
    alllo = np.concatenate([w[1] for w in ways])
    allla = np.concatenate([w[2] for w in ways])
    uid, first, inv = np.unique(allid, return_index=True, return_inverse=True)
    lng, lat = alllo[first], allla[first]
    ea, eb, br = [], [], []
    i0 = 0
    for ids, lo, la, bt in ways:
        k = len(ids)
        idx = inv[i0:i0 + k]
        i0 += k
        ea.append(idx[:-1]); eb.append(idx[1:]); br.append(np.full(k - 1, bt))
    ea, eb, br = np.concatenate(ea), np.concatenate(eb), np.concatenate(br)
    keep = ea != eb
    return _finish(lng, lat, ea[keep], eb[keep], br[keep], sample_m)


def load_osm_graph(pbf, bbox, cache_dir=None, progress=None) -> RoadGraph:
    """Graph from the OSM PBF (exact junctions + bridge / tunnel tags), cached as .npz next to the other caches."""
    p = Path(pbf)
    cp = None
    if cache_dir is not None:
        st = p.stat()
        tag = f"route1_{st.st_size}_{int(st.st_mtime)}_" + "_".join(f"{v:.3f}" for v in bbox)
        cp = Path(cache_dir) / f"{tag}.npz"
        if cp.is_file():
            try:
                z = np.load(cp)
                return _finish(z["lng"], z["lat"], z["ea"], z["eb"], z["br"], SAMPLE_M)
            except Exception:
                pass
    g = graph_from_osm(_osm_ways(p, bbox, progress))
    if cp is not None:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cp, lng=g.lng, lat=g.lat, ea=g.ea, eb=g.eb, br=g.bridge)
        except OSError:
            pass
    return g


def load_graph(path) -> RoadGraph:
    p = Path(path)
    raw = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")
    with raw as fh:
        return graph_from_features(json.load(fh).get("features", []))


def edge_depth(g: RoadGraph, sample_depth_m: np.ndarray) -> np.ndarray:
    """float64[m] max depth per edge from per-sample depths (NaN samples ignored; NaN when every sample is NaN)."""
    v = np.where(np.isnan(sample_depth_m), -1.0, sample_depth_m)
    starts = np.flatnonzero(np.r_[True, g.s_edge[1:] != g.s_edge[:-1]])
    mx = np.maximum.reduceat(v, starts)
    out = np.full(g.m, np.nan)
    out[g.s_edge[starts]] = np.where(mx < 0, np.nan, mx)
    return out


def effective_depth(g: RoadGraph, depth_m: np.ndarray) -> np.ndarray:
    """Edge depth that counts against a route: NaN -> 0, bridges / tunnels -> 0."""
    d = np.nan_to_num(np.asarray(depth_m, float), nan=0.0)
    return np.where(g.bridge, 0.0, d) if g.bridge is not None else d


def blocked_edges(g: RoadGraph, depth_m: np.ndarray, threshold_m: float) -> np.ndarray:
    return effective_depth(g, depth_m) >= threshold_m


def nearest_node(g: RoadGraph, lng: float, lat: float) -> tuple[int, float]:
    x, y = _xy(lng, lat, g.lat0)
    d2 = (g.x - x) ** 2 + (g.y - y) ** 2
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))


@dataclass
class Route:
    nodes: list            # node ids from start to end ([] when unreachable)
    length_m: float        # inf when unreachable

    @property
    def found(self):
        return bool(self.nodes)


def shortest(g: RoadGraph, a: int, b: int, blocked: np.ndarray | None = None) -> Route:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    keep = np.ones(g.m, bool) if blocked is None else ~blocked
    ea, eb, w = g.ea[keep], g.eb[keep], np.maximum(g.elen[keep], 1e-3)
    M = csr_matrix((np.r_[w, w], (np.r_[ea, eb], np.r_[eb, ea])), shape=(g.n, g.n))
    dist, pred = dijkstra(M, directed=True, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        return Route([], float("inf"))
    path, i = [b], b
    while i != a:
        i = int(pred[i])
        if i < 0:
            return Route([], float("inf"))
        path.append(i)
    return Route(path[::-1], float(dist[b]))


def path_length(g: RoadGraph, nodes) -> float:
    n = np.asarray(nodes)
    return float(np.hypot(np.diff(g.x[n]), np.diff(g.y[n])).sum()) if n.size > 1 else 0.0


def path_max_depth(g: RoadGraph, nodes, depth_m: np.ndarray) -> float:
    """Largest edge depth along a node path (NaN-free; 0 when none)."""
    if len(nodes) < 2:
        return 0.0
    lo = np.minimum(nodes[:-1], nodes[1:])
    hi = np.maximum(nodes[:-1], nodes[1:])
    order = np.lexsort((g.eb, g.ea))
    keys = g.ea[order].astype(np.int64) * g.n + g.eb[order]
    pos = np.searchsorted(keys, np.asarray(lo, np.int64) * g.n + hi)
    e = order[np.clip(pos, 0, order.size - 1)]
    return float(np.nanmax(np.r_[0.0, np.nan_to_num(depth_m[e], nan=0.0)]))


class DepthCache:
    """Edge depths per (storm, hour) from a TileService (depth_profile).  Thread-safe; keeps the last few."""

    def __init__(self, graph: RoadGraph, service, keep: int = 6):
        self.g, self.svc, self.keep = graph, service, keep
        self._d: dict = {}
        self._lock = threading.Lock()

    def get(self, storm: str, hour) -> np.ndarray | None:
        key = (storm, hour)
        with self._lock:
            if key in self._d:
                return self._d[key]
        if self.svc.tileset(storm) is None:
            return None
        s = self.svc.depth_profile(storm, self.g.s_lng, self.g.s_lat, hour)
        e = edge_depth(self.g, s)
        with self._lock:
            self._d[key] = e
            while len(self._d) > self.keep:
                self._d.pop(next(iter(self._d)))
        return e


@dataclass
class RouteResult:
    normal: Route
    dry: Route
    a_snap_m: float
    b_snap_m: float
    normal_max_depth_m: float
    dry_max_depth_m: float
    blocked_count: int
    threshold_m: float
    storm: str
    hour: int
    label: str

    def summary(self) -> str:
        if not self.normal.found:
            return "The two points are not connected by the road network."
        n_km = self.normal.length_m / 1000.0
        if not self.dry.found:
            return (f"No dry route: every connection crosses water >= {self.threshold_m * 100:.0f} cm "
                    f"(normal route {n_km:.2f} km, deepest {self.normal_max_depth_m * 100:.0f} cm).")
        d_km = self.dry.length_m / 1000.0
        extra = self.dry.length_m - self.normal.length_m
        if extra < 1.0:
            return f"The normal route ({n_km:.2f} km) stays dry (deepest {self.normal_max_depth_m * 100:.0f} cm)."
        return (f"Dry route {d_km:.2f} km vs normal {n_km:.2f} km (+{extra / 1000.0:.2f} km, +{100.0 * extra / max(self.normal.length_m, 1):.0f} %); "
                f"normal route crosses up to {self.normal_max_depth_m * 100:.0f} cm.")


def plan(g: RoadGraph, cache: DepthCache, a_lnglat, b_lnglat, storm: str, hour: int, threshold_m: float, label: str = "") -> RouteResult:
    a, da = nearest_node(g, *a_lnglat)
    b, db = nearest_node(g, *b_lnglat)
    depth = cache.get(storm, hour)
    if depth is None:
        depth = np.full(g.m, np.nan)
    eff = effective_depth(g, depth)
    blk = eff >= threshold_m
    normal = shortest(g, a, b)
    dry = shortest(g, a, b, blk)
    return RouteResult(normal, dry, da, db, path_max_depth(g, normal.nodes, eff) if normal.found else 0.0,
                       path_max_depth(g, dry.nodes, eff) if dry.found else 0.0, int(blk.sum()), float(threshold_m), storm,
                       int(hour), label)
