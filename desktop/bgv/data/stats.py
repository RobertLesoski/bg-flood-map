"""Flood statistics from decoded flow-grid pixels (CPU, numba). Same semantics as fluid/fluid-stats.js.

PUBLIC API
----------
A "grid" is the uint8[H, W, 4] (or flat uint8[H*W*4]) RGBA array of one hour (R = sqrt depth code, G/B = velocity codes,
A = valid-domain flag; FLOW_FORMAT.md) - any C-contiguous array of that size works, it is only viewed, never copied.
`None` stands for the shared all-dry zero grid (also accepted: an object with attribute isZero / is_zero / isZeroGrid
set True): every function then takes a fast path and reads no pixel.

Two-hour mix (what the renderer draws at continuous time t):  pass rgba_b and mix in (0,1): codes are interpolated linearly
(R in code space, velocity as decoded per-hour m/s) and the depth is decoded AFTER the interpolation, like the shaders.
mix <= 0 -> only `rgba`; mix >= 1 -> only `rgba_b`; a None grid in the blend reads as dry (so
evaluate_buildings(A, idx, None, 0.5) is "half way from A to dry").

    evaluate_buildings(rgba, index, rgba_b=None, mix=0.0, *, min_depth_m=0.1) -> BuildingStats
        index: PixelIndex or Buildings (bgv.data.buildings).  Building depth = MAX depth over its pixels (all-touched),
        pixels with A == 0 are ignored, flooded = depth >= min_depth_m. Evaluate cost ~1-3 ms for 20k buildings.
    BuildingStats:
        depth_m       float32[n]   max depth over the building's pixels (0 = dry / outside grid)
        speed_ms      float32[n]   speed at the deepest pixel (first in row-major order among equal codes), 0 when dry
        tier          uint8[n]     0 = below 0.1 m, 1 shallow [0.1,0.3), 2 moderate [0.3,0.6), 3 deep [0.6,1.2), 4 severe >= 1.2
        flooded       bool[n]      depth_m >= min_depth_m
        flooded_count int          flooded.sum()
        tier_counts   int64[4]     counts of flooded buildings per tier (shallow, moderate, deep, severe); .tier_dict() -> dict
        below_tier    int          flooded buildings with depth < 0.1 m (only non-zero when min_depth_m < 0.1)
        order         int32[k]     ids of the flooded buildings, deepest first, ties by ascending id  (k == flooded_count)
        min_depth_m   float
    evaluate_roads(rgba, roads, rgba_b=None, mix=0.0, *, min_depth_m=0.1) -> RoadStats
        roads: RoadSet (bgv.data.roads). A sample is flooded when its pixel has A != 0 and depth >= min_depth_m.
    RoadStats:
        flooded_pct       float   100 * flooded length / total_length_m (the whole network, also the part outside the grid)
        flooded_length_m  float
        total_length_m    float
        flooded_roads     int     roads with at least one flooded sample
        per_road_pct      float32[n_roads]   0..100 (0 for zero-length roads)
        sample_flooded    uint8[Ns]          1 where the sample is flooded (look up with RoadSet.seg_sample / ribbon.sample_id)
    summarize_grid(rgba, frame, *, wet_threshold_m=0.02) -> GridSummary    (single hour; frame: bgv.geo.GridFrame)
        wet_cells, wet_area_km2 (= wet_cells * cell_ground_m^2), max_depth_m, mean_depth_m (over wet cells), max_speed_ms
        (over wet cells), volume_m3 (= sum depth * cell_ground_m^2). Cells with A == 0 are ignored. ~5 ms (multi-threaded).
    decode_depth(codes) -> float64 metres; decode_velocity(codes) -> float64 m/s (code 128 = 0);
    DEPTH_MAX_M = 8.0, VMAX_MS = 4.0, WET_THRESHOLD_M = 0.02, TIER_EDGES_M, TIER_NAMES.

Depth law: depth_m = (R/255)^2 * 8, velocity = (code-128)/127 * 4 (u east, v north).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

try:        # start numba's threading layer NOW: once rasterio / GDAL DLLs are loaded, numba's tbb backend can fail to
    from numba.np.ufunc import parallel as _numba_parallel     # load (harmless: it falls back to omp / workqueue, but slower)
    _numba_parallel._launch_threads()
    import os as _os
    import numba as _numba
    # the statistics kernels run in the background: leave the render thread a core (8 logical -> 3 worker threads)
    _numba.set_num_threads(max(1, min(_numba.get_num_threads(), (_os.cpu_count() or 4) // 2 - 1)))
except Exception:                                              # pragma: no cover
    pass

DEPTH_MAX_M = 8.0
VMAX_MS = 4.0
WET_THRESHOLD_M = 0.02
TIER_EDGES_M = (0.1, 0.3, 0.6, 1.2)
TIER_NAMES = ("shallow", "moderate", "deep", "severe")

_CODES = np.arange(256, dtype=np.float64)
_DEPTH64 = (_CODES / 255.0) ** 2 * DEPTH_MAX_M
_VEL64 = (_CODES - 128.0) / 127.0 * VMAX_MS
_TIER_EDGES = np.array(TIER_EDGES_M, dtype=np.float64)


def decode_depth(codes):
    return (np.asarray(codes, dtype=np.float64) / 255.0) ** 2 * DEPTH_MAX_M


def decode_velocity(codes):
    return (np.asarray(codes, dtype=np.float64) - 128.0) / 127.0 * VMAX_MS


def min_code_for(min_depth_m: float) -> int:
    """Smallest depth code >= 1 whose decoded depth is >= min_depth_m (256 when none)."""
    ok = np.flatnonzero(_DEPTH64[1:] >= min_depth_m)
    return int(ok[0]) + 1 if ok.size else 256


# ------------------------------------------------------------------------------------------------ input handling
_DUMMY = np.zeros(1, dtype=np.uint32)
# numba's parallel kernels are launched one at a time (the 'workqueue' threading layer is not re-entrant; tbb is, but a lock is free)
_PAR_LOCK = threading.Lock()


def _is_zero(g) -> bool:
    if g is None:
        return True
    for a in ("isZero", "is_zero", "isZeroGrid"):
        if getattr(g, a, False) is True:
            return True
    return False


def _u32(g, w: int, h: int):
    """uint8 RGBA grid -> flat uint32 view (little endian: R = v & 255, G = v >> 8 & 255, B = v >> 16 & 255, A = v >> 24)."""
    if g.dtype == np.uint32 and g.size == w * h:
        return g.reshape(-1)
    if g.dtype != np.uint8 or g.size != w * h * 4:
        raise ValueError(f"grid must be uint8 with {w * h * 4} values ({w}x{h}x4), got {g.dtype} size {g.size}")
    if not g.flags.c_contiguous:
        g = np.ascontiguousarray(g)
    return g.reshape(-1).view(np.uint32)


def _prep(rgba, rgba_b, mix):
    """-> (mode, a, b, mix) with mode 'zero' | 'a' | 'mix'; None / zero grids are dropped (they read as dry)."""
    za, zb = _is_zero(rgba), _is_zero(rgba_b)
    if mix <= 0.0:
        return ("zero", None, None, 0.0) if za else ("a", rgba, None, 0.0)
    if mix >= 1.0:
        return ("zero", None, None, 0.0) if zb else ("a", rgba_b, None, 0.0)
    if za and zb:
        return "zero", None, None, 0.0
    return "mix", (None if za else rgba), (None if zb else rgba_b), float(mix)


def _tier_of(depth64: np.ndarray) -> np.ndarray:
    return np.searchsorted(_TIER_EDGES, depth64, side="right").astype(np.uint8)


# ------------------------------------------------------------------------------------------------ buildings
@njit(cache=True, parallel=True, nogil=True)
def _bld_pure(g, offsets, pixels, vel, out_code, out_speed):
    n = offsets.shape[0] - 1
    for i in prange(n):
        best = 0
        bp = -1
        for k in range(offsets[i], offsets[i + 1]):
            p = pixels[k]
            v = g[p]
            if (v >> 24) == 0:
                continue
            r = v & 255
            if r > best:
                best = r
                bp = p
        out_code[i] = best
        if bp >= 0:
            v = g[bp]
            u = vel[(v >> 8) & 255]
            w = vel[(v >> 16) & 255]
            out_speed[i] = np.sqrt(u * u + w * w)
        else:
            out_speed[i] = 0.0


@njit(cache=True, parallel=True, nogil=True)
def _bld_mix(ga, gb, has_a, has_b, mix, offsets, pixels, vel, out_code, out_speed):
    n = offsets.shape[0] - 1
    ma = 1.0 - mix
    for i in prange(n):
        best = -1.0
        bu = 0.0
        bv = 0.0
        for k in range(offsets[i], offsets[i + 1]):
            p = pixels[k]
            ra = 0
            rb = 0
            valid = False
            ua = 0.0
            va = 0.0
            ub = 0.0
            vb = 0.0
            if has_a:
                v = ga[p]
                if (v >> 24) != 0:
                    valid = True
                ra = v & 255
                if ra > 0:
                    ua = vel[(v >> 8) & 255]
                    va = vel[(v >> 16) & 255]
            if has_b:
                v = gb[p]
                if (v >> 24) != 0:
                    valid = True
                rb = v & 255
                if rb > 0:
                    ub = vel[(v >> 8) & 255]
                    vb = vel[(v >> 16) & 255]
            if not valid:
                continue
            rm = ma * ra + mix * rb
            if rm > best:
                best = rm
                bu = ma * ua + mix * ub
                bv = ma * va + mix * vb
        if best < 0.0:
            best = 0.0
        out_code[i] = best
        out_speed[i] = np.sqrt(bu * bu + bv * bv) if best > 0.0 else 0.0


@dataclass
class BuildingStats:
    depth_m: np.ndarray
    speed_ms: np.ndarray
    tier: np.ndarray
    flooded: np.ndarray
    flooded_count: int
    tier_counts: np.ndarray
    below_tier: int
    order: np.ndarray
    min_depth_m: float

    def tier_dict(self) -> dict:
        return {n: int(c) for n, c in zip(TIER_NAMES, self.tier_counts)}


def _bld_result(depth64, speed, min_depth_m) -> BuildingStats:
    n = depth64.shape[0]
    flooded = depth64 >= min_depth_m
    flooded &= depth64 > 0
    tier = _tier_of(depth64)
    tier_counts = np.bincount(tier[flooded], minlength=5)[1:5].astype(np.int64)
    fl_idx = np.flatnonzero(flooded)
    depth32 = depth64.astype(np.float32)
    if fl_idx.size:
        order = fl_idx[np.argsort(-depth64[fl_idx], kind="stable")].astype(np.int32)
    else:
        order = np.zeros(0, dtype=np.int32)
    return BuildingStats(depth32, speed, tier, flooded, int(fl_idx.size), tier_counts,
                         int(fl_idx.size - tier_counts.sum()), order, float(min_depth_m))


def evaluate_buildings(rgba, index, rgba_b=None, mix: float = 0.0, *, min_depth_m: float = 0.1) -> BuildingStats:
    idx = getattr(index, "index", index)
    n = idx.n
    mode, a, b, mx = _prep(rgba, rgba_b, mix)
    if mode == "zero":
        return _bld_result(np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float32), min_depth_m)
    speed = np.empty(n, dtype=np.float32)
    if mode == "a":
        code = np.empty(n, dtype=np.uint8)
        with _PAR_LOCK:
            _bld_pure(_u32(a, idx.width, idx.height), idx.offsets, idx.pixels, _VEL64, code, speed)
        depth64 = _DEPTH64[code]
    else:
        code = np.empty(n, dtype=np.float64)
        ga = _u32(a, idx.width, idx.height) if a is not None else _DUMMY
        gb = _u32(b, idx.width, idx.height) if b is not None else _DUMMY
        with _PAR_LOCK:
            _bld_mix(ga, gb, a is not None, b is not None, mx, idx.offsets, idx.pixels, _VEL64, code, speed)
        depth64 = (code / 255.0) ** 2 * DEPTH_MAX_M
    return _bld_result(depth64, speed, min_depth_m)


# ------------------------------------------------------------------------------------------------ roads
@njit(cache=True, nogil=True)
def _roads_pure(g, pix, wgt, road_off, road_len, min_code, flag, pct):
    total = 0.0
    nf = 0
    for f in range(road_off.shape[0] - 1):
        fl = 0.0
        for s in range(road_off[f], road_off[f + 1]):
            p = pix[s]
            if p < 0:
                continue
            v = g[p]
            if (v >> 24) != 0 and (v & 255) >= min_code:
                flag[s] = 1
                fl += wgt[s]
        if fl > 0.0:
            nf += 1
            total += fl
            q = 100.0 * fl / road_len[f]
            pct[f] = q if q < 100.0 else 100.0
    return total, nf


@njit(cache=True, nogil=True)
def _roads_mix(ga, gb, has_a, has_b, mix, pix, wgt, road_off, road_len, min_depth, flag, pct):
    total = 0.0
    nf = 0
    ma = 1.0 - mix
    for f in range(road_off.shape[0] - 1):
        fl = 0.0
        for s in range(road_off[f], road_off[f + 1]):
            p = pix[s]
            if p < 0:
                continue
            valid = False
            ra = 0
            rb = 0
            if has_a:
                v = ga[p]
                if (v >> 24) != 0:
                    valid = True
                ra = v & 255
            if has_b:
                v = gb[p]
                if (v >> 24) != 0:
                    valid = True
                rb = v & 255
            if not valid:
                continue
            rm = ma * ra + mix * rb
            d = (rm / 255.0) * (rm / 255.0) * 8.0
            if d >= min_depth and rm > 0.0:
                flag[s] = 1
                fl += wgt[s]
        if fl > 0.0:
            nf += 1
            total += fl
            q = 100.0 * fl / road_len[f]
            pct[f] = q if q < 100.0 else 100.0
    return total, nf


@dataclass
class RoadStats:
    flooded_pct: float
    flooded_length_m: float
    total_length_m: float
    flooded_roads: int
    per_road_pct: np.ndarray
    sample_flooded: np.ndarray


def evaluate_roads(rgba, roads, rgba_b=None, mix: float = 0.0, *, min_depth_m: float = 0.1) -> RoadStats:
    nr = roads.n_roads
    ns = roads.n_samples
    pct = np.zeros(nr, dtype=np.float32)
    flag = np.zeros(ns, dtype=np.uint8)
    total = float(roads.total_length_m)
    mode, a, b, mx = _prep(rgba, rgba_b, mix)
    if mode == "zero" or ns == 0:
        return RoadStats(0.0, 0.0, total, 0, pct, flag)
    length = np.maximum(roads.road_length_m, 1e-12)
    wf = roads.sample_weight_m
    gw, gh = roads.grid_w, roads.grid_h
    if mode == "a":
        fl, nf = _roads_pure(_u32(a, gw, gh), roads.sample_pixel, wf, roads.road_off, length, min_code_for(min_depth_m),
                             flag, pct)
    else:
        ga = _u32(a, gw, gh) if a is not None else _DUMMY
        gb = _u32(b, gw, gh) if b is not None else _DUMMY
        fl, nf = _roads_mix(ga, gb, a is not None, b is not None, mx, roads.sample_pixel, wf, roads.road_off, length,
                            float(min_depth_m), flag, pct)
    pct[roads.road_length_m <= 0] = 0.0
    return RoadStats(min(100.0, 100.0 * fl / total) if total > 0 else 0.0, float(fl), total, int(nf), pct, flag)


# ------------------------------------------------------------------------------------------------ grid summary
@njit(cache=True, parallel=True, nogil=True)
def _summ(g, wet_code, vel, nblocks):
    n = g.shape[0]
    hist = np.zeros((nblocks, 256), dtype=np.int64)
    ms = np.zeros(nblocks)
    for b in prange(nblocks):
        s = b * n // nblocks
        e = (b + 1) * n // nblocks
        mx = 0.0
        for p in range(s, e):
            v = g[p]
            r = v & 255
            if r < wet_code or (v >> 24) == 0:
                continue
            hist[b, r] += 1
            u = vel[(v >> 8) & 255]
            w = vel[(v >> 16) & 255]
            s2 = u * u + w * w
            if s2 > mx:
                mx = s2
        ms[b] = mx
    return hist, ms


@dataclass
class GridSummary:
    wet_cells: int
    wet_area_km2: float
    max_depth_m: float
    mean_depth_m: float
    max_speed_ms: float
    volume_m3: float


def summarize_grid(rgba, frame, *, wet_threshold_m: float = WET_THRESHOLD_M) -> GridSummary:
    if _is_zero(rgba):
        return GridSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    g = _u32(rgba, frame.width, frame.height)
    wet_code = min_code_for(wet_threshold_m)
    with _PAR_LOCK:
        hist, ms = _summ(g, wet_code, _VEL64, 32)
    h = hist.sum(axis=0)
    wet = int(h.sum())
    cg2 = frame.cell_ground_m ** 2
    depth_sum = float((h * _DEPTH64).sum())
    nz = np.flatnonzero(h)
    return GridSummary(wet, wet * cg2 / 1e6, float(_DEPTH64[nz[-1]]) if nz.size else 0.0,
                       depth_sum / wet if wet else 0.0, float(np.sqrt(ms.max())) if wet else 0.0, depth_sum * cg2)


def warmup() -> float:
    """Compile / load every numba kernel of this module on tiny inputs (about 0.5 s with a warm numba cache, several
    seconds on the very first run of a new install). Call once from a worker thread at startup. Returns seconds."""
    import time
    from types import SimpleNamespace
    t0 = time.perf_counter()
    idx = SimpleNamespace(n=2, offsets=np.array([0, 1, 2], dtype=np.int32), pixels=np.array([0, 1], dtype=np.int32),
                          width=2, height=1)
    g = np.zeros((1, 2, 4), dtype=np.uint8)
    g[..., 3] = 255
    g[0, 1, 0] = 100
    evaluate_buildings(g, idx)
    evaluate_buildings(g, idx, g, 0.5)
    rd = SimpleNamespace(n_roads=1, n_samples=1, road_off=np.array([0, 1], dtype=np.int32),
                         sample_pixel=np.array([1], dtype=np.int32), sample_weight_m=np.array([10.0], dtype=np.float32),
                         road_length_m=np.array([10.0]), total_length_m=10.0, grid_w=2, grid_h=1)
    evaluate_roads(g, rd)
    evaluate_roads(g, rd, g, 0.5)
    summarize_grid(g, SimpleNamespace(width=2, height=1, cell_ground_m=1.0))
    return time.perf_counter() - t0
