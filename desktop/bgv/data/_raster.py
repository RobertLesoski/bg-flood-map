"""numba kernels for footprint processing (internal; imported lazily by buildings.py so a cache hit never imports numba).

Port of the polygon rasteriser of `fluid/fluid-stats.js` (ALL-TOUCHED rule, GDAL all_touched=True semantics):
a footprint owns every grid pixel whose centre lies inside it (even-odd over the rings of a part, so holes are respected)
plus every pixel a ring edge passes through.  Pixel (col,row) is the half-open square [col,col+1) x [row,row+1) in grid
coordinates gx = (x_merc - xmin)/cell_m, gy = (ymax - y_merc)/cell_m.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def clean_rings(gx, gy, ring_off):
    """Drop consecutive duplicate vertices and the closing duplicate of every ring.
    Returns (sel int64[M] indices of the kept vertices, new_ring_off int64[R+1])."""
    nr = ring_off.shape[0] - 1
    sel = np.empty(gx.shape[0], dtype=np.int64)
    new_off = np.zeros(nr + 1, dtype=np.int64)
    m = 0
    for r in range(nr):
        start = m
        for j in range(ring_off[r], ring_off[r + 1]):
            if m > start and gx[j] == gx[sel[m - 1]] and gy[j] == gy[sel[m - 1]]:
                continue
            sel[m] = j
            m += 1
        if m - start > 1 and gx[sel[start]] == gx[sel[m - 1]] and gy[sel[start]] == gy[sel[m - 1]]:
            m -= 1
        new_off[r + 1] = m
    return sel[:m], new_off


@njit(cache=True)
def ring_metrics(gx, gy, X, Y, ring_off):
    """Per ring: signed pixel area (gx, gy), signed world area, world centroid (area-weighted, relative to first vertex)."""
    nr = ring_off.shape[0] - 1
    a_px = np.zeros(nr)
    a_w = np.zeros(nr)
    cxw = np.zeros(nr)
    cyw = np.zeros(nr)
    for r in range(nr):
        s = ring_off[r]
        e = ring_off[r + 1]
        n = e - s
        if n < 3:
            continue
        ox = gx[s]
        oy = gy[s]
        a = 0.0
        for j in range(n):
            k = j + 1 if j + 1 < n else 0
            a += (gx[s + j] - ox) * (gy[s + k] - oy) - (gx[s + k] - ox) * (gy[s + j] - oy)
        a_px[r] = a / 2.0
        ox = X[s]
        oy = Y[s]
        a = 0.0
        cx = 0.0
        cy = 0.0
        for j in range(n):
            k = j + 1 if j + 1 < n else 0
            x0 = X[s + j] - ox
            y0 = Y[s + j] - oy
            x1 = X[s + k] - ox
            y1 = Y[s + k] - oy
            cr = x0 * y1 - x1 * y0
            a += cr
            cx += (x0 + x1) * cr
            cy += (y0 + y1) * cr
        a *= 0.5
        a_w[r] = a
        if a != 0.0:
            cxw[r] = cx / (6.0 * a) + ox
            cyw[r] = cy / (6.0 * a) + oy
        else:
            cxw[r] = ox
            cyw[r] = oy
    return a_px, a_w, cxw, cyw


@njit(cache=True)
def building_bboxes(gx, gy, ring_off, part_ring_off, bpart_off, W, H):
    """Per building clipped pixel bbox (c0, c1, r0, r1); c0 > c1 marks an empty / fully outside building."""
    n = bpart_off.shape[0] - 1
    box = np.empty((n, 4), dtype=np.int64)
    for i in range(n):
        minx = 1e300
        maxx = -1e300
        miny = 1e300
        maxy = -1e300
        any_v = False
        for p in range(bpart_off[i], bpart_off[i + 1]):
            for r in range(part_ring_off[p], part_ring_off[p + 1]):
                for j in range(ring_off[r], ring_off[r + 1]):
                    any_v = True
                    x = gx[j]
                    y = gy[j]
                    if x < minx:
                        minx = x
                    if x > maxx:
                        maxx = x
                    if y < miny:
                        miny = y
                    if y > maxy:
                        maxy = y
        if not any_v:
            box[i, 0] = 1
            box[i, 1] = 0
            box[i, 2] = 1
            box[i, 3] = 0
            continue
        c0 = max(int(np.floor(minx)), 0)
        c1 = min(int(np.floor(maxx)), W - 1)
        r0 = max(int(np.floor(miny)), 0)
        r1 = min(int(np.floor(maxy)), H - 1)
        box[i, 0] = c0
        box[i, 1] = c1
        box[i, 2] = r0
        box[i, 3] = r1
    return box


@njit(cache=True)
def _mark_edge(mask, bw, mc0, mc1, mr0, mr1, x1, y1, x2, y2):
    if y1 > y2:
        t = x1
        x1 = x2
        x2 = t
        t = y1
        y1 = y2
        y2 = t
    r0 = int(np.floor(y1))
    r1 = int(np.floor(y2))
    if r0 < mr0:
        r0 = mr0
    if r1 > mr1:
        r1 = mr1
    dy = y2 - y1
    slope = (x2 - x1) / dy if dy > 0 else 0.0
    for r in range(r0, r1 + 1):
        if dy == 0:
            xa = x1
            xb = x2
        else:
            ya = y1 if y1 > r else float(r)
            yb = y2 if y2 < r + 1 else float(r + 1)
            xa = x1 if ya == y1 else x1 + (ya - y1) * slope
            xb = x2 if yb == y2 else x1 + (yb - y1) * slope
        if xa > xb:
            t = xa
            xa = xb
            xb = t
        c0 = int(np.floor(xa))
        c1 = int(np.floor(xb))
        if c0 < mc0:
            c0 = mc0
        if c1 > mc1:
            c1 = mc1
        base = (r - mr0) * bw - mc0
        for c in range(c0, c1 + 1):
            mask[base + c] = 1


@njit(cache=True)
def _fill_part(mask, bw, mc0, mc1, mr0, mr1, gx, gy, ring_off, r_start, r_end, xs):
    for r in range(mr0, mr1 + 1):
        yc = r + 0.5
        nx = 0
        for q in range(r_start, r_end):
            s = ring_off[q]
            e = ring_off[q + 1]
            for j in range(s, e):
                k = j + 1 if j + 1 < e else s
                y1 = gy[j]
                y2 = gy[k]
                if (y1 <= yc) != (y2 <= yc):
                    x1 = gx[j]
                    xs[nx] = x1 + (yc - y1) * (gx[k] - x1) / (y2 - y1)
                    nx += 1
        for i in range(1, nx):
            v = xs[i]
            j = i - 1
            while j >= 0 and xs[j] > v:
                xs[j + 1] = xs[j]
                j -= 1
            xs[j + 1] = v
        base = (r - mr0) * bw - mc0
        i = 0
        while i + 1 < nx:
            c0 = int(np.ceil(xs[i] - 0.5))
            c1 = int(np.ceil(xs[i + 1] - 0.5)) - 1
            if c0 < mc0:
                c0 = mc0
            if c1 > mc1:
                c1 = mc1
            for c in range(c0, c1 + 1):
                mask[base + c] = 1
            i += 2


@njit(cache=True)
def rasterize_all_touched(gx, gy, ring_off, part_ring_off, bpart_off, box, W, out_pix, max_mask, max_verts):
    """Fills out_pix (int32) with the pixel indices of every building (ascending, row-major) and returns offsets int64[n+1]."""
    n = bpart_off.shape[0] - 1
    offs = np.zeros(n + 1, dtype=np.int64)
    mask = np.zeros(max_mask, dtype=np.uint8)
    xs = np.empty(max_verts + 2, dtype=np.float64)
    np_ = 0
    for i in range(n):
        offs[i] = np_
        mc0 = box[i, 0]
        mc1 = box[i, 1]
        mr0 = box[i, 2]
        mr1 = box[i, 3]
        if mc0 > mc1 or mr0 > mr1:
            continue
        bw = mc1 - mc0 + 1
        size = bw * (mr1 - mr0 + 1)
        mask[:size] = 0
        for p in range(bpart_off[i], bpart_off[i + 1]):
            for r in range(part_ring_off[p], part_ring_off[p + 1]):
                s = ring_off[r]
                e = ring_off[r + 1]
                for j in range(s, e):
                    k = j + 1 if j + 1 < e else s
                    _mark_edge(mask, bw, mc0, mc1, mr0, mr1, gx[j], gy[j], gx[k], gy[k])
            _fill_part(mask, bw, mc0, mc1, mr0, mr1, gx, gy, ring_off, part_ring_off[p], part_ring_off[p + 1], xs)
        for r in range(mr0, mr1 + 1):
            base = (r - mr0) * bw - mc0
            for c in range(mc0, mc1 + 1):
                if mask[base + c]:
                    out_pix[np_] = r * W + c
                    np_ += 1
    offs[n] = np_
    return offs
