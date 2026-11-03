"""Scene data shared by BuildingsLayer, RoadsLayer, the stats hub and the picking code: ONE background load.

    scene = get_scene(app)            # idempotent: one SceneData per (app, grid frame); starts loading on first call
    scene.state                       # 'loading' | 'ready' | 'error'          (read from any thread)
    scene.message, scene.frac         # progress text / 0..1 for the loading indicator
    scene.buildings, scene.roads      # bgv.data Buildings / RoadSet           (None until ready)
    scene.base_z                      # float32[n]  metres, min terrain height over the building's pixels (never NaN)
    scene.mesh                        # BuildingGpuMesh   (vertex bytes + tile-sorted indices + tile boxes)
    scene.road_mesh                   # RoadGpuMesh
    scene.pick                        # PickData (kd-trees, roof triangles) - built in the same worker thread

Everything is built in ONE worker thread (cached data load ~0.1-1.5 s, mesh packing ~0.3 s), never on the render thread.
After `state == 'ready'` all attributes are immutable (readers on any thread need no lock).

Packed vertex formats (little endian, tightly packed):
    buildings 20 B : pos f32x3 (world X, Y, Z above the building base)  | normal snorm8x4 (x, y, z, 0)  | building id u32
    roads     20 B : pos f32x2 (world X, Y) | across f32 (-1 right edge .. +1 left edge) | along m f32 | road class u32
Index buffers are u32 and sorted by spatial tile (TILE_M ground metres) so the layers can frustum-cull whole tiles with
a few `draw_indexed(first_index=...)` calls; inside a tile the triangles of one building are contiguous.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

TILE_M = 400.0
BLD_VERTEX_DTYPE = np.dtype([("pos", "<f4", 3), ("nrm", "i1", 4), ("bid", "<u4")])
ROAD_VERTEX_DTYPE = np.dtype([("pos", "<f4", 2), ("across", "<f4"), ("along", "<f4"), ("cls", "<u4")])
assert BLD_VERTEX_DTYPE.itemsize == 20 and ROAD_VERTEX_DTYPE.itemsize == 20


@dataclass
class Tiles:
    """Spatial tiles of an index buffer: tile k owns indices [first[k], first[k]+count[k]); boxes in ground metres."""
    first: np.ndarray          # int64[T]  first index
    count: np.ndarray          # int64[T]  index count (multiple of 3)
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    zlo: np.ndarray            # min un-exaggerated ground height (m) of the tile's objects
    zhi: np.ndarray            # max un-exaggerated ground height
    hmax: np.ndarray           # tallest object above its base (m); 0 for roads

    @property
    def n(self) -> int:
        return int(self.first.shape[0])


@dataclass
class BuildingGpuMesh:
    vertices: np.ndarray       # structured BLD_VERTEX_DTYPE[V]
    indices: np.ndarray        # uint32[3T], tile-sorted, building-contiguous
    tiles: Tiles
    bld_first: np.ndarray      # int64[n]  first index of building i (-1 when it has no mesh)
    bld_count: np.ndarray      # int64[n]  index count of building i

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)


@dataclass
class RoadGpuMesh:
    vertices: np.ndarray       # structured ROAD_VERTEX_DTYPE[V]
    indices: np.ndarray        # uint32[3T], tile-sorted
    tiles: Tiles


@dataclass
class PickData:
    """CPU structures for selection (see bgv.gpu.selection)."""
    bld_tree: object                # scipy cKDTree over building centroids (only buildings with geometry)
    bld_ids: np.ndarray             # int32[k]  building id of each tree point
    bld_radius: np.ndarray          # float32[n] bounding radius of the footprint around the centroid (m), 0 = no mesh
    max_radius: float
    roof_tri: np.ndarray            # float32[T, 6]  (ax, ay, bx, by, cx, cy) roof triangles sorted by building
    roof_off: np.ndarray            # int64[n+1]     CSR into roof_tri
    seg_tree: object                # cKDTree over road segment midpoints
    seg_half: np.ndarray            # float32[S] half width (m)
    seg_len_max: float


def footprint_relief(index, terrain) -> np.ndarray:
    """float32[n]: max - min of the terrain under each building's pixels (0 when it has none)."""
    idx = getattr(index, "index", index)
    t = np.asarray(terrain, np.float32).reshape(-1)
    out = np.zeros(idx.n, np.float32)
    if idx.pixels.size:
        v = t[idx.pixels]
        cnt = np.diff(idx.offsets)
        nz = cnt > 0
        starts = idx.offsets[:-1][nz].astype(np.int64)
        out[nz] = np.maximum.reduceat(v, starts) - np.minimum.reduceat(v, starts)
    return out


def shadow_ok(b, relief) -> np.ndarray:
    """bool[n]: buildings that may cast a ground shadow.  A building that is mostly buried in a slope (big footprint on uneven
    ground) would paint its whole footprint dark through the terrain, so it casts none."""
    return relief < np.maximum(1.0, 0.6 * b.height_m)


# ------------------------------------------------------------------------------------------------ building mesh
def _tile_index(x, y, x0, y0, nx):
    return (np.floor((y - y0) / TILE_M).astype(np.int64) * nx + np.floor((x - x0) / TILE_M).astype(np.int64))


def build_building_mesh(b, base_z: np.ndarray) -> BuildingGpuMesh:
    m = b.mesh
    V = m.vertex_count
    n = b.n
    vb = np.empty(V, BLD_VERTEX_DTYPE)
    vb["pos"] = m.positions
    nr = np.zeros((V, 4), np.int8)
    nr[:, :3] = np.clip(np.rint(m.normals * 127.0), -127, 127).astype(np.int8)
    vb["nrm"] = nr
    vb["bid"] = m.building_id
    tri = m.indices.reshape(-1, 3)
    tb = m.building_id[tri[:, 0]].astype(np.int64)
    keep = np.asarray(b.in_grid)[tb]                  # buildings outside the model grid have no terrain under them: not drawn
    tri, tb = tri[keep], tb[keep]
    cx = np.nan_to_num(b.cx.astype(np.float64), nan=0.0)
    cy = np.nan_to_num(b.cy.astype(np.float64), nan=0.0)
    x0 = float(np.floor(cx.min() / TILE_M) * TILE_M) if n else 0.0
    y0 = float(np.floor(cy.min() / TILE_M) * TILE_M) if n else 0.0
    nx = int((cx.max() - x0) // TILE_M) + 2 if n else 1
    tile_of_b = _tile_index(cx, cy, x0, y0, nx)
    tt = tile_of_b[tb]
    order = np.argsort(tt * (n + 1) + tb, kind="stable")
    tri = tri[order]
    tb = tb[order]
    tt = tt[order]
    indices = np.ascontiguousarray(tri.reshape(-1).astype(np.uint32))
    # per building ranges
    bld_first = np.full(n, -1, np.int64)
    bld_count = np.zeros(n, np.int64)
    if tb.size:
        ub, first, cnt = np.unique(tb, return_index=True, return_counts=True)
        bld_first[ub] = first * 3
        bld_count[ub] = cnt * 3
    # tiles
    ut, tfirst, tcnt = np.unique(tt, return_index=True, return_counts=True)
    T = ut.size
    tile_id_of_b = np.searchsorted(ut, tile_of_b)
    has = bld_count > 0
    # box per tile from its buildings' extents (centroid +- radius) and vertices
    bx0 = np.full(n, np.inf); bx1 = np.full(n, -np.inf); by0 = np.full(n, np.inf); by1 = np.full(n, -np.inf)
    px = m.positions[:, 0].astype(np.float64)
    py = m.positions[:, 1].astype(np.float64)
    vord = np.argsort(m.building_id, kind="stable")
    sb = m.building_id[vord].astype(np.int64)
    ub2, vfirst = np.unique(sb, return_index=True)
    bx0[ub2] = np.minimum.reduceat(px[vord], vfirst)
    bx1[ub2] = np.maximum.reduceat(px[vord], vfirst)
    by0[ub2] = np.minimum.reduceat(py[vord], vfirst)
    by1[ub2] = np.maximum.reduceat(py[vord], vfirst)
    sel = np.flatnonzero(has)
    tb_tile = np.searchsorted(ut, tile_of_b[sel])
    bz = np.asarray(base_z, np.float64)[sel]
    hh = b.height_m.astype(np.float64)[sel]

    def agg(vals, op, init):
        out = np.full(T, init, np.float64)
        op.at(out, tb_tile, vals)
        return out

    tiles = Tiles(first=tfirst.astype(np.int64) * 3, count=tcnt.astype(np.int64) * 3,
                  x0=agg(bx0[sel], np.minimum, np.inf), x1=agg(bx1[sel], np.maximum, -np.inf),
                  y0=agg(by0[sel], np.minimum, np.inf), y1=agg(by1[sel], np.maximum, -np.inf),
                  zlo=agg(bz, np.minimum, np.inf), zhi=agg(bz, np.maximum, -np.inf),
                  hmax=agg(hh, np.maximum, 0.0))
    return BuildingGpuMesh(vb, indices, tiles, bld_first, bld_count)


# ------------------------------------------------------------------------------------------------ road mesh
def build_road_mesh(roads, heights_sampler=None, extent=None) -> RoadGpuMesh:
    """extent = (size_x, size_y) ground metres of the grid: ribbon segments whose centre lies outside are not drawn."""
    rb = roads.ribbon
    V = rb.vertex_count
    vb = np.empty(V, ROAD_VERTEX_DTYPE)
    vb["pos"] = rb.positions[:, :2]
    vb["across"] = rb.across
    vb["along"] = rb.along_m
    vb["cls"] = rb.road_class
    S = V // 4
    if S == 0:
        z = np.zeros(0, np.float64)
        return RoadGpuMesh(vb, np.zeros(0, np.uint32), Tiles(*(z.astype(np.int64),) * 2, *(z,) * 7))
    pos = rb.positions.reshape(S, 4, 3)
    mx = pos[:, :, 0].mean(axis=1).astype(np.float64)
    my = pos[:, :, 1].mean(axis=1).astype(np.float64)
    inside = np.ones(S, bool) if extent is None else (np.abs(mx) <= extent[0] * 0.5 - 1.0) & (np.abs(my) <= extent[1] * 0.5 - 1.0)
    if not inside.any():
        z = np.zeros(0, np.float64)
        return RoadGpuMesh(vb, np.zeros(0, np.uint32), Tiles(*(z.astype(np.int64),) * 2, *(z,) * 7))
    keep_seg = np.flatnonzero(inside)
    pos, mx, my = pos[keep_seg], mx[keep_seg], my[keep_seg]
    S = keep_seg.size
    x0 = float(np.floor(mx.min() / TILE_M) * TILE_M)
    y0 = float(np.floor(my.min() / TILE_M) * TILE_M)
    nx = int((mx.max() - x0) // TILE_M) + 2
    tile = _tile_index(mx, my, x0, y0, nx)
    order = np.argsort(tile, kind="stable")
    base = (4 * keep_seg[order].astype(np.uint32))[:, None]
    idx = (np.array([0, 2, 3, 0, 3, 1], np.uint32)[None, :] + base).reshape(-1)
    st = tile[order]
    ut, tfirst, tcnt = np.unique(st, return_index=True, return_counts=True)
    T = ut.size
    sx = pos[order][:, :, 0].astype(np.float64)
    sy = pos[order][:, :, 1].astype(np.float64)
    seg_t = np.searchsorted(ut, st)

    def agg(vals, op, init):
        out = np.full(T, init, np.float64)
        op.at(out, seg_t, vals)
        return out

    zlo = np.zeros(T)
    zhi = np.zeros(T)
    if heights_sampler is not None:
        # tile ground range from the segment midpoints (+- a margin added by the caller)
        zg = heights_sampler.sample(mx[order], my[order])
        zlo = agg(zg, np.minimum, np.inf)
        zhi = agg(zg, np.maximum, -np.inf)
    tiles = Tiles(first=tfirst.astype(np.int64) * 6, count=tcnt.astype(np.int64) * 6,
                  x0=agg(sx.min(axis=1), np.minimum, np.inf), x1=agg(sx.max(axis=1), np.maximum, -np.inf),
                  y0=agg(sy.min(axis=1), np.minimum, np.inf), y1=agg(sy.max(axis=1), np.maximum, -np.inf),
                  zlo=zlo, zhi=zhi, hmax=np.zeros(T))
    return RoadGpuMesh(vb, np.ascontiguousarray(idx), tiles)


# ------------------------------------------------------------------------------------------------ pick structures
def build_pick(b, roads) -> PickData:
    from scipy.spatial import cKDTree
    n = b.n
    m = b.mesh
    good = np.flatnonzero(np.isfinite(b.cx) & np.isfinite(b.cy) & (b.area_m2 > 0) & np.asarray(b.in_grid))
    tree = cKDTree(np.stack([b.cx[good], b.cy[good]], axis=1).astype(np.float64)) if good.size else None
    # bounding radius around the centroid
    px = m.positions[:, 0].astype(np.float64)
    py = m.positions[:, 1].astype(np.float64)
    bid = m.building_id.astype(np.int64)
    d = np.hypot(px - np.nan_to_num(b.cx[bid]), py - np.nan_to_num(b.cy[bid]))
    rad = np.zeros(n, np.float64)
    np.maximum.at(rad, bid, d)
    # roof triangles = triangles whose three vertices have normal z == 1
    tri = m.indices.reshape(-1, 3)
    roof = (m.normals[tri[:, 0], 2] > 0.5) & (m.normals[tri[:, 1], 2] > 0.5) & (m.normals[tri[:, 2], 2] > 0.5)
    tri = tri[roof]
    tb = bid[tri[:, 0]]
    o = np.argsort(tb, kind="stable")
    tri, tb = tri[o], tb[o]
    rt = np.empty((tri.shape[0], 6), np.float32)
    for k in range(3):
        rt[:, 2 * k] = m.positions[tri[:, k], 0]
        rt[:, 2 * k + 1] = m.positions[tri[:, k], 1]
    off = np.zeros(n + 1, np.int64)
    np.add.at(off, tb + 1, 1)
    off = np.cumsum(off)
    # road segments
    S = roads.n_segments
    if S:
        mx = 0.5 * (roads.seg_x0 + roads.seg_x1).astype(np.float64)
        my = 0.5 * (roads.seg_y0 + roads.seg_y1).astype(np.float64)
        stree = cKDTree(np.stack([mx, my], axis=1))
        seg_len = np.hypot(roads.seg_x1 - roads.seg_x0, roads.seg_y1 - roads.seg_y0)
        slm = float(seg_len.max())
    else:
        stree, slm = None, 0.0
    return PickData(tree, good.astype(np.int32), rad.astype(np.float32), float(rad.max()) if n else 0.0, rt, off, stree,
                    (0.5 * roads.seg_width_m).astype(np.float32), slm)


# ------------------------------------------------------------------------------------------------ the loader
class SceneData:
    def __init__(self, root, frame, heights: np.ndarray, sampler=None, domain=None):
        """`domain`: a bgv.domain.Domain; it says where the building / road GeoJSON files come from (the city's files, or
        the county's, cut from the OpenStreetMap extract on first use).  None = the city's files under `root`."""
        self.root = root
        self.frame = frame
        self.heights = heights
        self.sampler = sampler
        self.domain = domain
        self.osm_info: dict = {}
        self._lo, self._hi = 0.0, 1.0
        self.state = "loading"
        self.message = "starting"
        self.frac = 0.0
        self.error: str | None = None
        self.buildings = None
        self.roads = None
        self.base_z: np.ndarray | None = None
        self.shadow_ok: np.ndarray | None = None
        self.mesh: BuildingGpuMesh | None = None
        self.road_mesh: RoadGpuMesh | None = None
        self.pick: PickData | None = None
        self.timings: dict = {}
        self._thread = threading.Thread(target=self._run, name="bgv-scene-load", daemon=True)
        self._thread.start()

    def _progress(self, msg, frac):
        self.message = str(msg)
        if frac is not None:
            self.frac = min(0.9, 0.9 * (self._lo + (self._hi - self._lo) * float(frac)))

    def _run(self):
        from .bgthread import lower_priority
        lower_priority()
        t0 = time.perf_counter()
        try:
            from ..data import buildings as bmod
            from ..data import prep
            bpath = rpath = None
            if self.domain is not None and self.domain.key != "city":
                self._lo, self._hi = 0.0, 0.5            # reading the OpenStreetMap extract (first run only)
                bpath, rpath, self.osm_info = self.domain.scene_paths(self.frame, self._progress)
                self._lo, self._hi = 0.5, 1.0
            res = prep.ensure_all(self.root, self.frame, self._progress, buildings_path=bpath, roads_path=rpath)
            b, r = res["buildings"], res["roads"]
            self.timings["data_s"] = time.perf_counter() - t0
            self.message = "preparing meshes"
            t1 = time.perf_counter()
            base = bmod.attach_base_z(b.index, self.heights)
            bad = ~np.isfinite(base)
            if bad.any():
                ok = bad & np.isfinite(b.cx) & np.isfinite(b.cy)
                if self.sampler is not None and ok.any():
                    base[ok] = self.sampler.sample(b.cx[ok], b.cy[ok]).astype(np.float32)
                base[~np.isfinite(base)] = np.float32(np.nanmedian(self.heights))
            self.frac = 0.92
            mesh = build_building_mesh(b, base)
            self.frac = 0.95
            rmesh = build_road_mesh(r, self.sampler, self.frame.size_m)
            self.frac = 0.98
            pick = build_pick(b, r)
            self.timings["mesh_s"] = time.perf_counter() - t1
            self.buildings, self.roads, self.base_z = b, r, base
            self.shadow_ok = shadow_ok(b, footprint_relief(b.index, self.heights))
            self.mesh, self.road_mesh, self.pick = mesh, rmesh, pick
            self.frac = 1.0
            self.message = "ready"
            self.state = "ready"
        except Exception as e:                       # report through the sidebar, never raise into the app
            import traceback
            self.error = f"{type(e).__name__}: {e}"
            self.message = self.error
            self.state = "error"
            self._tb = traceback.format_exc()
        self.timings["total_s"] = time.perf_counter() - t0

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def wait(self, timeout: float = 120.0) -> bool:
        self._thread.join(timeout)
        return self.state == "ready"


def get_scene(app) -> SceneData:
    """The shared SceneData of this app (created and started on first use; replaced when the grid changes)."""
    sc = getattr(app, "_scene_data", None)
    frame = app.terrain.frame
    if sc is None or sc.frame != frame or sc.heights is not app.terrain.heights:
        sc = SceneData(app.root, frame, app.terrain.heights, app.terrain.sampler, getattr(app, "domain", None))
        app._scene_data = sc
    return sc
