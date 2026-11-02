"""Road network -> render segments, ~10 m statistics samples and a ribbon mesh, all in the WORLD frame.

PUBLIC API (pure CPU, numpy only)
---------------------------------
    load_roads(root, frame, *, progress=None, use_cache=True, spacing_m=10.0) -> RoadSet
        root  : data root containing layers/roads.geojson;  frame : bgv.geo.GridFrame of the flow grid.
        Cache: desktop/cache/roads_*.npz (size+mtime+frame keyed). Build ~0.1 s, cached load a few ms.

    ROAD_CLASSES : tuple of highway names, class id = index (last entry 'other' catches unknown values)
    ROAD_WIDTH_M : float32[len(ROAD_CLASSES)] carriageway width in metres used for the ribbon (motorway 20 ... residential 6.5)

class RoadSet  (world frame: +X east, +Y north, metres; Z is NOT stored - drape on the terrain in the shader)
    n_roads                      : int            number of features in roads.geojson (road id = feature index)
    road_class  uint8[n_roads]   class id (ROAD_CLASSES)
    road_length_m float64[n_roads]               ground length (sum over the feature's lines)
    road_off    int32[n_roads+1] CSR: statistics samples of road r are  sample_*[road_off[r]:road_off[r+1]]
    sample_pixel  int32[Ns]      flow-grid pixel (row*W+col) of every sample, -1 outside the grid
    sample_weight_m float32[Ns]  ground metres the sample represents (line length / number of samples, ~10 m)
    sample_x, sample_y float32[Ns] world position of the sample (midpoint of its part)
    total_length_m, in_grid_length_m : float
    -- render segments (the exact polyline, cut at every sample boundary so each segment belongs to ONE sample) --
    seg_x0, seg_y0, seg_x1, seg_y1 float32[S]
    seg_class   uint8[S]         class id
    seg_width_m float32[S]       ROAD_WIDTH_M[class]
    seg_road    int32[S]         road id
    seg_sample  int32[S]         index into the sample_* arrays (look the flood state up there)
    ribbon      : RoadRibbon (see below), built with width scale 1.0

class RoadRibbon:  one triangle list; 4 vertices per segment, segments that meet end to start share their joint edge (mitered, limit 2x)
    positions float32[V,3] (X, Y, 0), across float32[V] (-1 right edge .. +1 left edge), along_m float32[V] (distance along
    the line, metres), sample_id uint32[V] (-> sample_* arrays and per-sample flood state), road_class uint32[V],
    indices uint32[3T] counter-clockwise seen from above.
    make_ribbon(roads, width_scale=1.0) -> RoadRibbon   (rebuilds with scaled widths, ~10 ms)

Sampling rule is identical to fluid/fluid-stats.js buildRoadIndex: a line of ground length L gets n = ceil(L/spacing)
samples at the midpoints of n equal parts, each carrying L/n metres. (Ground length here = world-frame length, which uses
one cos(latitude) for the whole grid instead of the JS per-segment cosine: relative difference < 0.15 %.)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geo import GridFrame, R_MERC
from . import cachelib

VERSION = "rd3"

ROAD_CLASSES = ("motorway", "motorway_link", "trunk", "primary", "primary_link", "secondary", "tertiary",
                "unclassified", "residential", "other")
ROAD_WIDTH_M = np.array([20.0, 8.0, 14.0, 12.0, 8.0, 10.0, 9.0, 7.0, 6.5, 6.5], dtype=np.float32)
_CLASS_ID = {c: i for i, c in enumerate(ROAD_CLASSES)}


@dataclass
class RoadRibbon:
    positions: np.ndarray
    across: np.ndarray
    along_m: np.ndarray
    sample_id: np.ndarray
    road_class: np.ndarray
    indices: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0] // 3)


@dataclass
class RoadSet:
    n_roads: int
    road_class: np.ndarray
    road_length_m: np.ndarray
    road_off: np.ndarray
    sample_pixel: np.ndarray
    sample_weight_m: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    total_length_m: float
    in_grid_length_m: float
    seg_x0: np.ndarray
    seg_y0: np.ndarray
    seg_x1: np.ndarray
    seg_y1: np.ndarray
    seg_class: np.ndarray
    seg_width_m: np.ndarray
    seg_line: np.ndarray       # line id per segment (info)
    seg_s0: np.ndarray             # internal: distance along line at the segment start
    seg_road: np.ndarray
    seg_sample: np.ndarray
    ribbon: RoadRibbon
    spacing_m: float = 10.0
    frame_key: str = ""
    timings: dict = None
    grid_w: int = 0              # flow grid size the sample pixels refer to
    grid_h: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.sample_pixel.shape[0])

    @property
    def n_segments(self) -> int:
        return int(self.seg_x0.shape[0])


def _report(progress, msg, frac=None):
    if progress is not None:
        try:
            progress(msg, frac)
        except Exception:
            pass


def _geojson_path(root) -> Path:
    return Path(root) / "layers" / "roads.geojson"


# ------------------------------------------------------------------------------------------------ ribbon
def make_ribbon(roads: RoadSet, width_scale: float = 1.0) -> RoadRibbon:
    S = roads.n_segments
    if S == 0:
        z = np.zeros(0, dtype=np.float32)
        return RoadRibbon(np.zeros((0, 3), np.float32), z, z, np.zeros(0, np.uint32), np.zeros(0, np.uint32),
                          np.zeros(0, np.uint32))
    x0 = roads.seg_x0.astype(np.float64)
    y0 = roads.seg_y0.astype(np.float64)
    x1 = roads.seg_x1.astype(np.float64)
    y1 = roads.seg_y1.astype(np.float64)
    dx = x1 - x0
    dy = y1 - y0
    L = np.hypot(dx, dy)
    L[L == 0] = 1.0
    nx = -dy / L                      # left normal
    ny = dx / L
    hw = 0.5 * roads.seg_width_m.astype(np.float64) * width_scale
    # a joint exists where the next segment of the same road starts where this one ends (also across the separate
    # 2-vertex lines of the MultiLineStrings in roads.geojson)
    same_next = np.zeros(S, dtype=bool)
    same_next[:-1] = ((roads.seg_road[:-1] == roads.seg_road[1:]) &
                      (np.hypot(x1[:-1] - x0[1:], y1[:-1] - y0[1:]) < 0.1))
    same_prev = np.zeros(S, dtype=bool)
    same_prev[1:] = same_next[:-1]

    def miter(ax, ay, bx, by):
        mx, my = ax + bx, ay + by
        ml = np.hypot(mx, my)
        ok = ml > 1e-6
        mx = np.where(ok, mx / np.where(ok, ml, 1), bx)
        my = np.where(ok, my / np.where(ok, ml, 1), by)
        d = np.maximum(mx * bx + my * by, 0.5)
        return mx / d, my / d

    # start offsets (join with the previous segment), end offsets (join with the next)
    px = np.roll(nx, 1)
    py = np.roll(ny, 1)
    mx0, my0 = miter(px, py, nx, ny)
    ox0 = np.where(same_prev, mx0, nx)
    oy0 = np.where(same_prev, my0, ny)
    qx = np.roll(nx, -1)
    qy = np.roll(ny, -1)
    mx1, my1 = miter(nx, ny, qx, qy)
    ox1 = np.where(same_next, mx1, nx)
    oy1 = np.where(same_next, my1, ny)

    pos = np.zeros((S, 4, 3), dtype=np.float32)
    pos[:, 0, 0] = x0 - ox0 * hw; pos[:, 0, 1] = y0 - oy0 * hw
    pos[:, 1, 0] = x0 + ox0 * hw; pos[:, 1, 1] = y0 + oy0 * hw
    pos[:, 2, 0] = x1 - ox1 * hw; pos[:, 2, 1] = y1 - oy1 * hw
    pos[:, 3, 0] = x1 + ox1 * hw; pos[:, 3, 1] = y1 + oy1 * hw
    across = np.tile(np.array([-1, 1, -1, 1], dtype=np.float32), (S, 1))
    s0 = roads.seg_s0.astype(np.float64)
    along = np.stack([s0, s0, s0 + L, s0 + L], axis=1).astype(np.float32)
    sid = np.repeat(roads.seg_sample.astype(np.uint32), 4)
    cls = np.repeat(roads.seg_class.astype(np.uint32), 4)
    base = (4 * np.arange(S, dtype=np.uint32))[:, None]
    idx = (np.array([0, 2, 3, 0, 3, 1], dtype=np.uint32)[None, :] + base).reshape(-1)
    return RoadRibbon(pos.reshape(-1, 3), across.reshape(-1), along.reshape(-1), sid, cls, idx)


# ------------------------------------------------------------------------------------------------ build
def build_roads(root, frame: GridFrame, *, progress=None, spacing_m: float = 10.0) -> RoadSet:
    """Uncached build from <root>/layers/roads.geojson."""
    _report(progress, "roads: reading geojson", 0.0)
    with open(_geojson_path(root), "r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    return roads_from_features(feats, frame, progress=progress, spacing_m=spacing_m)


def roads_from_features(feats, frame: GridFrame, *, progress=None, spacing_m: float = 10.0) -> RoadSet:
    """Build from GeoJSON Feature dicts (LineString / MultiLineString, lng/lat). Used by build_roads and the unit tests."""
    t_all = time.perf_counter()
    n = len(feats)
    road_class = np.full(n, _CLASS_ID["other"], dtype=np.uint8)
    lines = []                       # (road id, ndarray[m,2] lng/lat)
    for i, f in enumerate(feats):
        pr = (f or {}).get("properties") or {}
        road_class[i] = _CLASS_ID.get(str(pr.get("highway")), _CLASS_ID["other"])
        g = (f or {}).get("geometry") or {}
        c = g.get("coordinates")
        if g.get("type") == "LineString" and isinstance(c, list):
            cl = [c]
        elif g.get("type") == "MultiLineString" and isinstance(c, list):
            cl = c
        else:
            continue
        for ln in cl:
            if not isinstance(ln, list):
                continue
            try:
                a = np.array([(p[0], p[1]) for p in ln], dtype=np.float64)
            except (TypeError, IndexError, ValueError):
                continue
            a = a[np.isfinite(a).all(axis=1)]
            if a.shape[0] >= 2:
                lines.append((i, a))

    k = frame.k
    W, H = frame.width, frame.height
    sx_m, sy_m = frame.size_m
    seg_cols = {key: [] for key in ("x0", "y0", "x1", "y1", "road", "sample", "line", "s0")}
    smp_x, smp_y, smp_w, smp_pix = [], [], [], []
    sample_road = []
    road_len = np.zeros(n, dtype=np.float64)
    n_samples = 0
    for line_id, (rid, ll) in enumerate(lines):
        xm = R_MERC * np.radians(ll[:, 0])
        ym = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(ll[:, 1]) / 2))
        X = (xm - frame.xc) * k
        Y = (ym - frame.yc) * k
        keep = np.r_[True, (np.diff(X) != 0) | (np.diff(Y) != 0)]
        X, Y = X[keep], Y[keep]
        m = X.shape[0]
        if m < 2:
            continue
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
        L = float(cum[-1])
        if not L > 0:
            continue
        cnt = max(1, int(math.ceil(L / spacing_m - 1e-9)))
        step = L / cnt

        def point_at(b, side):
            kk = np.clip(np.searchsorted(cum, b, side=side) - 1, 0, m - 2)
            d = cum[kk + 1] - cum[kk]
            t = np.where(d > 0, (b - cum[kk]) / np.where(d > 0, d, 1), 0.0)
            t = np.clip(t, 0.0, 1.0)
            return X[kk] + (X[kk + 1] - X[kk]) * t, Y[kk] + (Y[kk + 1] - Y[kk]) * t

        pos = (np.arange(cnt) + 0.5) * step
        px, py = point_at(pos, "left")
        col = np.floor((px / sx_m + 0.5) * W).astype(np.int64)
        row = np.floor((0.5 - py / sy_m) * H).astype(np.int64)
        inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        smp_pix.append(np.where(inside, row * W + col, -1).astype(np.int32))
        smp_x.append(px)
        smp_y.append(py)
        smp_w.append(np.full(cnt, step))
        sample_road.append(np.full(cnt, rid, dtype=np.int32))
        road_len[rid] += L

        bp = np.unique(np.round(np.concatenate([cum, np.arange(1, cnt) * step]), 6))
        bp = np.clip(bp, 0.0, L)
        bx, by = point_at(bp, "right")
        seg_len = np.diff(bp)
        ok = seg_len > 1e-6
        mid = 0.5 * (bp[:-1] + bp[1:])
        part = np.clip(np.floor(mid / step).astype(np.int64), 0, cnt - 1)
        seg_cols["x0"].append(bx[:-1][ok]); seg_cols["y0"].append(by[:-1][ok])
        seg_cols["x1"].append(bx[1:][ok]); seg_cols["y1"].append(by[1:][ok])
        seg_cols["road"].append(np.full(int(ok.sum()), rid, dtype=np.int32))
        seg_cols["sample"].append((n_samples + part[ok]).astype(np.int32))
        seg_cols["line"].append(np.full(int(ok.sum()), line_id, dtype=np.int32))
        seg_cols["s0"].append(bp[:-1][ok])
        n_samples += cnt

    def cat(lst, dt):
        return np.concatenate(lst).astype(dt) if lst else np.zeros(0, dtype=dt)

    sample_pixel = cat(smp_pix, np.int32)
    sample_w = cat(smp_w, np.float32)
    sample_road_a = cat(sample_road, np.int32)
    counts = np.bincount(sample_road_a, minlength=n)
    road_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    seg_road = cat(seg_cols["road"], np.int32)
    seg_class = road_class[seg_road] if seg_road.size else np.zeros(0, np.uint8)
    roads = RoadSet(
        n_roads=n, road_class=road_class, road_length_m=road_len, road_off=road_off, sample_pixel=sample_pixel,
        sample_weight_m=sample_w, sample_x=cat(smp_x, np.float32), sample_y=cat(smp_y, np.float32),
        total_length_m=float(road_len.sum()),
        in_grid_length_m=float(sample_w[sample_pixel >= 0].astype(np.float64).sum()),
        seg_x0=cat(seg_cols["x0"], np.float32), seg_y0=cat(seg_cols["y0"], np.float32),
        seg_x1=cat(seg_cols["x1"], np.float32), seg_y1=cat(seg_cols["y1"], np.float32),
        seg_class=seg_class.astype(np.uint8), seg_width_m=ROAD_WIDTH_M[seg_class],
        seg_line=cat(seg_cols["line"], np.int32), seg_s0=cat(seg_cols["s0"], np.float32), seg_road=seg_road,
        seg_sample=cat(seg_cols["sample"], np.int32), ribbon=None, spacing_m=spacing_m,
        frame_key=cachelib.frame_key(frame), timings={}, grid_w=frame.width, grid_h=frame.height)
    roads.ribbon = make_ribbon(roads)
    roads.timings = {"total_s": time.perf_counter() - t_all}
    _report(progress, "roads: done", 1.0)
    return roads


# ------------------------------------------------------------------------------------------------ cache
_FIELDS = ("road_class", "road_length_m", "road_off", "sample_pixel", "sample_weight_m", "sample_x", "sample_y",
           "seg_x0", "seg_y0", "seg_x1", "seg_y1", "seg_class", "seg_width_m", "seg_line", "seg_s0", "seg_road",
           "seg_sample")
_RIB = ("positions", "across", "along_m", "sample_id", "road_class", "indices")


def _cache_file(root, frame, spacing_m):
    return cachelib.cache_path("roads", frame, [_geojson_path(root)], VERSION, extra=f"sp={spacing_m!r}")


def load_roads(root, frame: GridFrame, *, progress=None, use_cache: bool = True, spacing_m: float = 10.0) -> RoadSet:
    cf = _cache_file(root, frame, spacing_m)
    if use_cache:
        t0 = time.perf_counter()
        hit = cachelib.load_npz(cf)
        if hit is not None:
            try:
                a, meta = hit
                rib = RoadRibbon(**{f: a["rib_" + f] for f in _RIB})
                r = RoadSet(n_roads=int(meta["n_roads"]), total_length_m=float(meta["total_length_m"]),
                            in_grid_length_m=float(meta["in_grid_length_m"]), ribbon=rib, spacing_m=spacing_m,
                            frame_key=cachelib.frame_key(frame), timings={"load_s": time.perf_counter() - t0, "cached": True},
                            grid_w=frame.width, grid_h=frame.height,
                            **{f: a[f] for f in _FIELDS})
                return r
            except Exception:
                pass
    r = build_roads(root, frame, progress=progress, spacing_m=spacing_m)
    if use_cache:
        try:
            arrays = {f: getattr(r, f) for f in _FIELDS}
            arrays.update({"rib_" + f: getattr(r.ribbon, f) for f in _RIB})
            cachelib.save_npz(cf, arrays, {"n_roads": r.n_roads, "total_length_m": r.total_length_m,
                                           "in_grid_length_m": r.in_grid_length_m, "version": VERSION})
        except Exception:
            pass
    return r
