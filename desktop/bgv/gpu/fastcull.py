"""numba kernels for the per-frame culling of the terrain chunk grid and of the building / road tile boxes.

Why: on this laptop a numpy version of the same maths costs 0.7 ms per frame (about 60 small array operations, each of
them 3x slower when the core has just woken from the idle wait of a vsync-paced frame), the kernels below cost ~20 us
in one call.  `available()` is False when numba cannot be imported; the callers then use their numpy implementations
(which are also the reference the tests compare the kernels against).

terrain_kernel   chunk LOD from the squared distance to the chunk box, frustum test (box centre / extent form), snap word
                 (how much coarser each neighbour is), sorted instance lists for terrain and water (group by LOD, front to
                 back inside a group, same order as `TerrainMesh.update_reference`).
tile_visible     frustum test of N boxes (centre / extent form).
tile_runs        merged [first_index, index_count) draw ranges of the visible tiles (gaps merged when more than max_draws).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _OK = True
except Exception:            # pragma: no cover - numba is installed on the target machine
    _OK = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


def available() -> bool:
    return _OK


@njit(cache=True, nogil=True)
def terrain_kernel(xmin, xmax, ymin, ymax, z0, z1, cx, cy, cz, ex, ey, ez, planes, e0, e1, e2, thr, nx, ny, wet,
                   use_wet, max_inst, lod, word, vis, d2, data_t, cnt_t, data_w, cnt_w):
    n = nx * ny
    for i in range(n):
        dx = max(max(xmin[i] - e0, e0 - xmax[i]), 0.0)
        dy = max(max(ymin[i] - e1, e1 - ymax[i]), 0.0)
        dz = max(max(z0[i] - e2, e2 - z1[i]), 0.0)
        dd = dx * dx + dy * dy + dz * dz
        d2[i] = dd
        l = 0
        for k in range(thr.shape[0]):
            if dd >= thr[k]:
                l += 1
        lod[i] = l
        v = True
        for p in range(5):
            s = (planes[p, 0] * cx[i] + planes[p, 1] * cy[i] + planes[p, 2] * cz[i] + planes[p, 3]
                 + abs(planes[p, 0]) * ex[i] + abs(planes[p, 1]) * ey[i] + abs(planes[p, 2]) * ez[i])
            if s < 0.0:
                v = False
                break
        vis[i] = v
    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            l = lod[c]
            w = max(lod[j * nx + max(i - 1, 0)] - l, 0)
            w |= max(lod[j * nx + min(i + 1, nx - 1)] - l, 0) << 4
            w |= max(lod[max(j - 1, 0) * nx + i] - l, 0) << 8
            w |= max(lod[min(j + 1, ny - 1) * nx + i] - l, 0) << 12
            word[c] = w
    nv = 0
    for i in range(n):
        if vis[i]:
            nv += 1
    keys = np.empty(nv, np.float64)
    idx = np.empty(nv, np.int64)
    k = 0
    for i in range(n):
        if vis[i]:
            keys[k] = lod[i] * 1.0e12 + d2[i]
            idx[k] = i
            k += 1
    order = np.argsort(keys, kind="mergesort")
    for l in range(cnt_t.shape[0]):
        cnt_t[l] = 0
        cnt_w[l] = 0
    kt = 0
    kw = 0
    for q in range(nv):
        i = idx[order[q]]
        if kt < max_inst:
            data_t[kt, 0] = i % nx
            data_t[kt, 1] = i // nx
            data_t[kt, 2] = lod[i]
            data_t[kt, 3] = word[i]
            cnt_t[lod[i]] += 1
            kt += 1
        if use_wet and wet[i] and kw < max_inst:
            data_w[kw, 0] = i % nx
            data_w[kw, 1] = i // nx
            data_w[kw, 2] = lod[i]
            data_w[kw, 3] = word[i]
            cnt_w[lod[i]] += 1
            kw += 1
    return kt, kw


@njit(cache=True, nogil=True)
def tile_visible(ctr, ext, planes, out):
    n = ctr.shape[0]
    for i in range(n):
        v = True
        for p in range(5):
            s = (planes[p, 0] * ctr[i, 0] + planes[p, 1] * ctr[i, 1] + planes[p, 2] * ctr[i, 2] + planes[p, 3]
                 + abs(planes[p, 0]) * ext[i, 0] + abs(planes[p, 1]) * ext[i, 1] + abs(planes[p, 2]) * ext[i, 2])
            if s < 0.0:
                v = False
                break
        out[i] = v


@njit(cache=True, nogil=True)
def tile_runs(vis, tfirst, tcount, max_draws, out_first, out_count):
    """Merged draw ranges of the visible tiles -> number of ranges written to out_first / out_count."""
    n = vis.shape[0]
    lo = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    m = 0
    i = 0
    while i < n:
        if vis[i]:
            j = i
            while j + 1 < n and vis[j + 1]:
                j += 1
            lo[m] = i
            hi[m] = j
            m += 1
            i = j + 1
        else:
            i += 1
    if m == 0:
        return 0
    keep = np.ones(max(m - 1, 1), np.bool_)          # keep[g]: gap g (between run g and g+1) stays a gap
    if m > max_draws:
        gap = np.empty(m - 1, np.int64)
        for g in range(m - 1):
            gap[g] = tfirst[lo[g + 1]] - (tfirst[hi[g]] + tcount[hi[g]])
        order = np.argsort(gap, kind="mergesort")
        for q in range(m - max_draws):
            keep[order[q]] = False
    r = 0
    start = lo[0]
    for g in range(m - 1):
        if keep[g]:
            out_first[r] = tfirst[start]
            out_count[r] = tfirst[hi[g]] + tcount[hi[g]] - tfirst[start]
            r += 1
            start = lo[g + 1]
    out_first[r] = tfirst[start]
    out_count[r] = tfirst[hi[m - 1]] + tcount[hi[m - 1]] - tfirst[start]
    r += 1
    return r


def warmup() -> None:
    """Pay numba's one-time start-up (target registries, BLAS probe -> scipy.linalg import, ~1 s) and load the cached kernels
    off the render thread; without it the first terrain.update of the real window stalled for about a second."""
    if not _OK:
        return
    try:
        from numba.np.linalg import ensure_blas
        ensure_blas()
    except Exception:
        pass
    planes = np.zeros((6, 4), np.float32)
    for dt in (np.float32, np.float64):
        ctr = np.zeros((2, 3), dt)
        out = np.zeros(2, np.bool_)
        tile_visible(ctr, ctr, planes.astype(dt), out)
    tile_runs(np.ones(2, np.bool_), np.zeros(2, np.int64), np.ones(2, np.int64), 48, np.zeros(4, np.int64),
              np.zeros(4, np.int64))
