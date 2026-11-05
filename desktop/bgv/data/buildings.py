"""Building footprints -> per-building table, extruded world-frame mesh and a per-building pixel index on the flow grid.

PUBLIC API (pure CPU, no wgpu; numba is imported only when a cache miss forces a build)
------------------------------------------------------------------------------------
    load_buildings(root, frame, *, progress=None, use_cache=True, skirt_m=0.0) -> Buildings
        root   : data root (Path/str) containing layers/buildings.geojson
        frame  : bgv.geo.GridFrame of the flow grid (identical for every storm)
        progress(msg:str, frac:float) optional callback (called from the calling thread).
        First build ~10-20 s, cached load ~0.1 s (cache: desktop/cache/buildings_*.npz, keyed by size+mtime+frame).

    attach_base_z(index, terrain, fill=None) -> float32[n]
        index   : PixelIndex (or Buildings, its .index is used)
        terrain : float32[H, W] terrain metres on the flow grid (NaN = no DEM); minimum over the building's pixels,
                  NaN-safe; buildings with no pixel / only NaN pixels get `fill` (default NaN).

Coordinate frame (see bgv.geo): WORLD metres, +X east, +Y north, +Z up, origin = grid centre. Mesh Z is RELATIVE TO THE
BUILDING BASE (0 .. height_m): the renderer adds base_z[building_id] (from attach_base_z + terrain exaggeration).

class Buildings (all arrays indexed by building number i = position in buildings.geojson; n == len(features), buildings
with unusable geometry keep their slot: area 0, centroid NaN, no pixels, no mesh):
    n                : int
    id               : int32[n]      feature index (== arange(n))
    cx, cy           : float32[n]    footprint centroid, world metres (area weighted, holes subtracted)
    area_m2          : float32[n]    footprint area in ground m2 (exterior minus holes)
    height_m         : float32[n]    building height: props.height_m, 6.0 when missing/zero/NaN, clamped to [2, 60]
    sale_price       : float32[n]    NaN when unknown
    address, land_use, name, building_type, amenity : numpy unicode arrays <U..>[n] ('' when missing); use e.g.
                       b.address[i] (np.str_) or str(b.address[i]).
    in_grid          : bool[n]       building owns at least one flow-grid pixel
    index            : PixelIndex    ALL-TOUCHED pixel index (CSR):  offsets int32[n+1], pixels int32[nnz] (pixel = row*W+col,
                       row 0 = north, ascending inside a building), width, height. index.pixels_of(i) -> int32 view.
    mesh             : BuildingMesh
    frame_key        : str

class BuildingMesh (one indexed triangle list, draw with draw_indexed(len(indices))):
    positions        : float32[V, 3]  world X, Y and Z-above-base (0..height)
    normals          : float32[V, 3]  roof (0,0,1); walls horizontal outward unit normals (flat shaded, 4 verts per wall quad)
    building_id      : uint32[V]      building number per vertex (index into every Buildings array)
    indices          : uint32[3*T]    counter-clockwise when seen from outside (back-face culling is safe); no bottom faces
    vertex_count / triangle_count properties.
    Holes are real holes in the roof (earcut) and get inner walls. Ring orientation is normalised (exterior CCW, holes CW).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geo import GridFrame, R_MERC
from . import cachelib

VERSION = "bld5"
DEFAULT_HEIGHT_M = 6.0
MIN_HEIGHT_M = 2.0
MAX_HEIGHT_M = 60.0
MIN_RING_AREA_PX2 = 1e-7          # same degenerate-footprint rule as fluid-stats.js


# ------------------------------------------------------------------------------------------------ data classes
@dataclass
class PixelIndex:
    offsets: np.ndarray            # int32[n+1]
    pixels: np.ndarray             # int32[nnz]
    width: int
    height: int

    @property
    def n(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def pixels_of(self, i: int) -> np.ndarray:
        return self.pixels[self.offsets[i]:self.offsets[i + 1]]

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)


@dataclass
class BuildingMesh:
    positions: np.ndarray          # float32[V,3]
    normals: np.ndarray            # float32[V,3]
    building_id: np.ndarray        # uint32[V]
    indices: np.ndarray            # uint32[3T]

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)


@dataclass
class Buildings:
    n: int
    id: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area_m2: np.ndarray
    height_m: np.ndarray
    sale_price: np.ndarray
    address: np.ndarray
    land_use: np.ndarray
    name: np.ndarray
    building_type: np.ndarray
    amenity: np.ndarray
    in_grid: np.ndarray
    index: PixelIndex
    mesh: BuildingMesh
    frame_key: str = ""
    timings: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ helpers
def _report(progress, msg, frac=None):
    if progress is not None:
        try:
            progress(msg, frac)
        except Exception:
            pass


def _geojson_path(root, geojson=None) -> Path:
    """`geojson`: another footprint file (the county domain: <cache>/osm/<key>/buildings.geojson); default the city's."""
    return Path(geojson) if geojson is not None else Path(root) / "layers" / "buildings.geojson"


def _num(v, default=float("nan")) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _s(v) -> str:
    return "" if v is None else str(v)


def _parse(features):
    """-> per-building props + flat raw ring structure (lng/lat, ring_off, part_ring_off, bpart_off)."""
    n = len(features)
    pts = []
    ring_len = []
    part_rings = []                 # rings per part
    bpart_count = np.zeros(n, dtype=np.int64)
    height = np.full(n, np.nan)
    price = np.full(n, np.nan)
    address, land_use, name, btype, amenity = [], [], [], [], []
    for i, f in enumerate(features):
        pr = (f or {}).get("properties") or {}
        height[i] = _num(pr.get("height_m"))
        price[i] = _num(pr.get("sale_price"))
        address.append(_s(pr.get("ADDRESS")))
        land_use.append(_s(pr.get("LAND_USE")))
        name.append(_s(pr.get("name")))
        btype.append(_s(pr.get("building")))
        amenity.append(_s(pr.get("amenity")))
        g = (f or {}).get("geometry") or {}
        t = g.get("type")
        c = g.get("coordinates")
        if t == "Polygon" and isinstance(c, list):
            polys = [c]
        elif t == "MultiPolygon" and isinstance(c, list):
            polys = c
        else:
            continue
        for rings in polys:
            if not isinstance(rings, list) or not rings:
                continue
            nr = 0
            for ring in rings:
                if not isinstance(ring, list):
                    continue
                pts.extend(ring)
                ring_len.append(len(ring))
                nr += 1
            if nr:
                part_rings.append(nr)
                bpart_count[i] += 1
    try:
        arr = np.array(pts, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError
    except ValueError:
        arr = np.array([(p[0], p[1]) for p in pts], dtype=np.float64)
    lng = np.ascontiguousarray(arr[:, 0])
    lat = np.ascontiguousarray(arr[:, 1])
    ring_off = np.concatenate([[0], np.cumsum(ring_len, dtype=np.int64)])
    part_ring_off = np.concatenate([[0], np.cumsum(part_rings, dtype=np.int64)])
    bpart_off = np.concatenate([[0], np.cumsum(bpart_count)])
    props = dict(height=height, price=price, address=address, land_use=land_use, name=name, btype=btype, amenity=amenity)
    return props, lng, lat, ring_off, part_ring_off, bpart_off


def _earcut_roofs(X, Y, ring_off, part_ring_off):
    """Triangulate every part; returns int64 indices into the compact vertex arrays (3 per triangle)."""
    import mapbox_earcut as ec
    XY = np.ascontiguousarray(np.stack([X, Y], axis=1))
    out = []
    P = part_ring_off.shape[0] - 1
    for p in range(P):
        r0 = part_ring_off[p]
        r1 = part_ring_off[p + 1]
        vs = ring_off[r0]
        ve = ring_off[r1]
        if r1 - r0 == 1 and ve - vs == 3:
            out.append(np.array([vs, vs + 1, vs + 2], dtype=np.int64))
            continue
        ends = (ring_off[r0 + 1:r1 + 1] - vs).astype(np.uint32)
        idx = ec.triangulate_float64(XY[vs:ve], ends)
        if idx.size:
            out.append(idx.astype(np.int64) + vs)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# ------------------------------------------------------------------------------------------------ build
def build_buildings(root, frame: GridFrame, *, progress=None, skirt_m: float = 0.0, geojson=None) -> Buildings:
    """Uncached build from <root>/layers/buildings.geojson (or `geojson`)."""
    _report(progress, "buildings: reading geojson", 0.0)
    t0 = time.perf_counter()
    with open(_geojson_path(root, geojson), "r", encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    b = buildings_from_features(features, frame, progress=progress, skirt_m=skirt_m)
    b.timings["read_s"] = time.perf_counter() - t0 - b.timings["total_s"]
    return b


def buildings_from_features(features, frame: GridFrame, *, progress=None, skirt_m: float = 0.0, index_only: bool = False):
    """Build from GeoJSON Feature dicts (Polygon / MultiPolygon, lng/lat). Used by build_buildings and the unit tests.
    index_only=True stops after the all-touched pixel index and returns just that PixelIndex (any grid frame, e.g. the
    solver-tile grid: see bgv.data.tiles)."""
    from . import _raster
    t_all = time.perf_counter()
    timings = {}
    n = len(features)
    t0 = time.perf_counter()
    props, lng, lat, ring_off0, part_ring_off0, bpart_off0 = _parse(features)
    timings["parse_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _report(progress, "buildings: projecting", 0.15)
    xm = R_MERC * np.radians(lng)
    ym = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    gx = (xm - frame.xmin) / frame.cell_m
    gy = (frame.ymax - ym) / frame.cell_m
    k = frame.k
    X = (xm - frame.xc) * k
    Y = (ym - frame.yc) * k
    sel, ring_off1 = _raster.clean_rings(gx, gy, ring_off0)
    gx, gy, X, Y = gx[sel], gy[sel], X[sel], Y[sel]
    a_px, a_w, cxr, cyr = _raster.ring_metrics(gx, gy, X, Y, ring_off1)

    R0 = ring_off1.shape[0] - 1
    P0 = part_ring_off0.shape[0] - 1
    ring_n = np.diff(ring_off1)
    ring_ok = (ring_n >= 3) & (np.abs(a_px) > MIN_RING_AREA_PX2)
    rings_per_part0 = np.diff(part_ring_off0)
    ring_part = np.repeat(np.arange(P0), rings_per_part0)
    first_ring = part_ring_off0[:-1]
    part_ok = np.zeros(P0, dtype=bool)
    has = rings_per_part0 > 0
    part_ok[has] = ring_ok[first_ring[has]]
    ring_keep = ring_ok & part_ok[ring_part]
    is_ext0 = np.arange(R0) == first_ring[ring_part]

    # compaction ----------------------------------------------------------------------------
    vkeep = np.repeat(ring_keep, ring_n)
    gx, gy, X, Y = gx[vkeep], gy[vkeep], X[vkeep], Y[vkeep]
    ring_off = np.concatenate([[0], np.cumsum(ring_n[ring_keep])]).astype(np.int64)
    rings_per_part = np.bincount(ring_part[ring_keep], minlength=P0)[part_ok]
    part_ring_off = np.concatenate([[0], np.cumsum(rings_per_part)]).astype(np.int64)
    part_building0 = np.repeat(np.arange(n), np.diff(bpart_off0))
    parts_per_b = np.bincount(part_building0[part_ok], minlength=n)
    bpart_off = np.concatenate([[0], np.cumsum(parts_per_b)]).astype(np.int64)
    a_w = a_w[ring_keep]
    a_px = a_px[ring_keep]
    cxr, cyr = cxr[ring_keep], cyr[ring_keep]
    is_ext = is_ext0[ring_keep]
    ring_bld = part_building0[ring_part[ring_keep]]
    part_bld = part_building0[part_ok]
    R = ring_off.shape[0] - 1
    P = part_ring_off.shape[0] - 1

    # per-building aggregates -----------------------------------------------------------------
    w = np.abs(a_w) * np.where(is_ext, 1.0, -1.0)
    wsum = np.bincount(ring_bld, weights=w, minlength=n)
    wx = np.bincount(ring_bld, weights=w * cxr, minlength=n)
    wy = np.bincount(ring_bld, weights=w * cyr, minlength=n)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    good = wsum > 0
    cx[good] = wx[good] / wsum[good]
    cy[good] = wy[good] / wsum[good]
    has_geom = parts_per_b > 0
    fb = ~good & has_geom                     # holes swallow everything (invalid input): first vertex
    if fb.any():
        first_vert_of_b = ring_off[part_ring_off[bpart_off[:-1][fb]]]
        cx[fb] = X[first_vert_of_b]
        cy[fb] = Y[first_vert_of_b]
    area = np.where(good, wsum, 0.0)
    h = props["height"].copy()
    h[~np.isfinite(h) | (h <= 0)] = DEFAULT_HEIGHT_M
    h = np.clip(h, MIN_HEIGHT_M, MAX_HEIGHT_M)

    # pixel index -----------------------------------------------------------------------------
    _report(progress, "buildings: pixel index (all-touched)", 0.3)
    box = _raster.building_bboxes(gx, gy, ring_off, part_ring_off, bpart_off, frame.width, frame.height)
    bwid = np.maximum(box[:, 1] - box[:, 0] + 1, 0)
    bhei = np.maximum(box[:, 3] - box[:, 2] + 1, 0)
    barea = bwid * bhei
    cap = int(barea.sum()) + 1
    max_mask = int(barea.max()) + 1 if n else 1
    vcount_b = np.bincount(ring_bld, weights=np.diff(ring_off), minlength=n)
    max_verts = int(vcount_b.max()) if n else 1
    out_pix = np.empty(cap, dtype=np.int32)
    offs64 = _raster.rasterize_all_touched(gx, gy, ring_off, part_ring_off, bpart_off, box, frame.width, out_pix,
                                           max_mask, max_verts)
    pixels = out_pix[:offs64[-1]].copy()
    del out_pix
    index = PixelIndex(offs64.astype(np.int32), pixels, frame.width, frame.height)
    in_grid = np.diff(offs64) > 0
    timings["index_s"] = time.perf_counter() - t0
    if index_only:
        return index

    # mesh ------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    _report(progress, "buildings: extruded mesh", 0.55)
    flip = (is_ext & (a_w < 0)) | (~is_ext & (a_w > 0))
    if flip.any():
        ring_of_v = np.repeat(np.arange(R), np.diff(ring_off))
        j = np.arange(X.shape[0]) - ring_off[ring_of_v]
        m = np.diff(ring_off)[ring_of_v]
        newj = np.where(flip[ring_of_v], m - 1 - j, j)
        perm = ring_off[ring_of_v] + newj
        X, Y, gx, gy = X[perm], Y[perm], gx[perm], gy[perm]
    V0 = X.shape[0]
    ring_of_v = np.repeat(np.arange(R), np.diff(ring_off))
    bld_of_v = ring_bld[ring_of_v]
    hv = h[bld_of_v]

    # roofs
    roof_pos = np.empty((V0, 3), dtype=np.float32)
    roof_pos[:, 0] = X
    roof_pos[:, 1] = Y
    roof_pos[:, 2] = hv
    roof_nrm = np.zeros((V0, 3), dtype=np.float32)
    roof_nrm[:, 2] = 1.0
    roof_bid = bld_of_v.astype(np.uint32)
    tri = _earcut_roofs(X, Y, ring_off, part_ring_off).reshape(-1, 3)
    if tri.shape[0]:
        ax, ay = X[tri[:, 0]], Y[tri[:, 0]]
        bx, by = X[tri[:, 1]], Y[tri[:, 1]]
        cx_, cy_ = X[tri[:, 2]], Y[tri[:, 2]]
        cross = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)
        neg = cross < 0
        tri[neg] = tri[neg][:, [0, 2, 1]]
        # drop slivers: collinear footprint vertices give triangles whose height is below 2 mm (their winding is
        # not even well defined once positions are rounded to float32)
        longest = np.maximum.reduce([np.hypot(bx - ax, by - ay), np.hypot(cx_ - bx, cy_ - by), np.hypot(ax - cx_, ay - cy_)])
        tri = tri[np.abs(cross) > 2e-3 * np.maximum(longest, 1e-9)]
    roof_idx = tri.reshape(-1).astype(np.uint32)

    # walls
    nxt = np.arange(V0) + 1
    nxt[ring_off[1:] - 1] = ring_off[:-1]
    dx = X[nxt] - X
    dy = Y[nxt] - Y
    L = np.hypot(dx, dy)
    ek = np.flatnonzero(L > 1e-4)
    E = ek.shape[0]
    p0 = ek
    p1 = nxt[ek]
    nxv = (dy[ek] / L[ek]).astype(np.float32)
    nyv = (-dx[ek] / L[ek]).astype(np.float32)
    wh = h[bld_of_v[ek]].astype(np.float32)
    zb = np.float32(-skirt_m)
    wpos = np.empty((E, 4, 3), dtype=np.float32)
    wpos[:, 0, 0] = X[p0]; wpos[:, 0, 1] = Y[p0]; wpos[:, 0, 2] = zb
    wpos[:, 1, 0] = X[p1]; wpos[:, 1, 1] = Y[p1]; wpos[:, 1, 2] = zb
    wpos[:, 2, 0] = X[p1]; wpos[:, 2, 1] = Y[p1]; wpos[:, 2, 2] = wh
    wpos[:, 3, 0] = X[p0]; wpos[:, 3, 1] = Y[p0]; wpos[:, 3, 2] = wh
    wnrm = np.zeros((E, 4, 3), dtype=np.float32)
    wnrm[:, :, 0] = nxv[:, None]
    wnrm[:, :, 1] = nyv[:, None]
    wbid = np.repeat(bld_of_v[ek].astype(np.uint32), 4)
    base = (V0 + 4 * np.arange(E, dtype=np.uint32))[:, None]
    wtri = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)[None, :] + base
    positions = np.concatenate([roof_pos, wpos.reshape(-1, 3)])
    normals = np.concatenate([roof_nrm, wnrm.reshape(-1, 3)])
    bid = np.concatenate([roof_bid, wbid])
    indices = np.concatenate([roof_idx, wtri.reshape(-1)]).astype(np.uint32)
    mesh = BuildingMesh(positions, normals, bid, indices)
    timings["mesh_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_all
    _report(progress, "buildings: done", 1.0)

    return Buildings(
        n=n, id=np.arange(n, dtype=np.int32), cx=cx.astype(np.float32), cy=cy.astype(np.float32),
        area_m2=area.astype(np.float32), height_m=h.astype(np.float32), sale_price=props["price"].astype(np.float32),
        address=np.array(props["address"], dtype=np.str_), land_use=np.array(props["land_use"], dtype=np.str_),
        name=np.array(props["name"], dtype=np.str_), building_type=np.array(props["btype"], dtype=np.str_),
        amenity=np.array(props["amenity"], dtype=np.str_), in_grid=in_grid, index=index, mesh=mesh,
        frame_key=cachelib.frame_key(frame), timings=timings)


# ------------------------------------------------------------------------------------------------ cache
_FIELDS = ("id", "cx", "cy", "area_m2", "height_m", "sale_price", "address", "land_use", "name", "building_type", "amenity",
           "in_grid")


def _to_arrays(b: Buildings) -> dict:
    d = {f: getattr(b, f) for f in _FIELDS}
    d.update(idx_offsets=b.index.offsets, idx_pixels=b.index.pixels, m_pos=b.mesh.positions, m_nrm=b.mesh.normals,
             m_bid=b.mesh.building_id, m_idx=b.mesh.indices)
    return d


def _from_arrays(a: dict, meta: dict, frame: GridFrame) -> Buildings:
    index = PixelIndex(a["idx_offsets"], a["idx_pixels"], frame.width, frame.height)
    mesh = BuildingMesh(a["m_pos"], a["m_nrm"], a["m_bid"], a["m_idx"])
    kw = {f: a[f] for f in _FIELDS}
    return Buildings(n=int(meta["n"]), index=index, mesh=mesh, frame_key=cachelib.frame_key(frame),
                     timings=dict(meta.get("timings", {})), **kw)


def _cache_file(root, frame, skirt_m, geojson=None):
    return cachelib.cache_path("buildings", frame, [_geojson_path(root, geojson)], VERSION, extra=f"skirt={skirt_m!r}")


def load_buildings(root, frame: GridFrame, *, progress=None, use_cache: bool = True, skirt_m: float = 0.0,
                   geojson=None) -> Buildings:
    cf = _cache_file(root, frame, skirt_m, geojson)
    if use_cache:
        t0 = time.perf_counter()
        hit = cachelib.load_npz(cf)
        if hit is not None:
            try:
                b = _from_arrays(hit[0], hit[1], frame)
                b.timings = dict(b.timings, load_s=time.perf_counter() - t0, cached=True)
                return b
            except Exception:
                pass
    b = build_buildings(root, frame, progress=progress, skirt_m=skirt_m, geojson=geojson)
    if use_cache:
        t0 = time.perf_counter()
        try:
            cachelib.save_npz(cf, _to_arrays(b), {"n": b.n, "timings": b.timings, "version": VERSION})
        except Exception:
            pass
        b.timings["save_s"] = time.perf_counter() - t0
    return b


# ------------------------------------------------------------------------------------------------ terrain
def attach_base_z(index, terrain, fill=None) -> np.ndarray:
    """Minimum terrain elevation (metres) over each building's pixels -> float32[n]. NaN-safe, pure."""
    idx = getattr(index, "index", index)
    t = np.asarray(terrain, dtype=np.float32)
    if t.size != idx.width * idx.height:
        raise ValueError(f"terrain has {t.size} cells, the index was built for {idx.width}x{idx.height}")
    t = t.reshape(-1)
    n = idx.n
    out = np.full(n, np.nan, dtype=np.float32)
    if idx.pixels.size:
        vals = t[idx.pixels]
        vals = np.where(np.isfinite(vals), vals, np.float32(np.inf))
        cnt = np.diff(idx.offsets)
        nz = cnt > 0
        starts = idx.offsets[:-1][nz].astype(np.int64)
        mins = np.minimum.reduceat(vals, starts)
        mins[~np.isfinite(mins)] = np.nan
        out[nz] = mins
    if fill is not None:
        out[~np.isfinite(out)] = np.float32(fill)
    return out
