"""OpenStreetMap-style basemap for the terrain: XYZ tile maths, polite tile fetching, progressive composite.  Pure CPU
(numpy + PIL + urllib), no wgpu, thread safe, fully usable offline.

WHY NO REPROJECTION: the flow grid is EPSG:3857 (Web Mercator) and so are XYZ tiles, so a tile maps to the grid by a
plain scale + offset on each axis.  The composite is an RGBA8 image that covers EXACTLY the grid extent
(`frame.xmin..xmax`, `frame.ymin..ymax`), row 0 = north, so its UV is the flow grid's UV (`u = X/size_x + 0.5`,
`v = 0.5 - Y/size_y`, see bgv.geo): `terrain uv -> basemap uv` is the identity.

PUBLIC API
    tile maths     tile_span_m(z), merc_to_tilef(mx, my, z) -> (fx, fy), tile_bounds_merc(z, x, y), lnglat_to_tile(lng, lat, z),
                   tile_range(frame, z) -> (x0, x1, y0, y1) inclusive, tiles_for_frame(frame, z), choose_zoom(frame, long_px),
                   composite_dims(frame, long_px) -> (w, h), meters_per_px(z, lat_deg=0.0), composite_vram_mb(w, h)
    TileSource     where tiles come from: url template (default OSM), disk cache under desktop/cache/tiles/<z>/<x>/<y>.png
                   (forever), optional read-only local folder (same z/x/y.png layout), polite rate limiter, retries + backoff,
                   honours HTTP errors.  `network=False` never touches the network.
    BasemapComposite(frame, long_px=4096, zoom=None)
                   .rgba uint8[h, w, 4] (alpha 255), .mips list of arrays (CPU box filter in linear light), .band_gen per
                   64-row band generation counters (the GPU layer uploads bands whose generation changed),
                   .put_tile(z, x, y, tile, getter), .fill_fallback() (procedural map-like paper), .coverage
    BasemapLoader(composite, source, threads=4)
                   .plan() -> Plan (tile counts, cached / missing, approx MB, needs_confirm), .start(download=False,
                   confirmed=False) (background thread; download=False only reads the cache / local folder), .cancel(),
                   .status() -> LoadStatus copy (phase, message, done, total, ...), progress callback(LoadStatus)
    ATTRIBUTION = "(c) OpenStreetMap contributors"

OSM TILE USAGE POLICY (https://operations.osmfoundation.org/policies/tiles/): valid User-Agent identifying the app, low
rate (max 4 connections here, global minimum interval between request starts, per-thread delay), cache every tile and never
fetch it twice, no bulk download: the loader only ever requests tiles that cover the grid footprint (plan() / hard limit
MAX_TILES_HARD), zoom capped at MAX_ZOOM, and a download needs an explicit call (the GPU layer wires it to a button).
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .. import config
from ..geo import GridFrame, R_MERC

ORIGIN = math.pi * R_MERC                 # Web Mercator half world extent, 20037508.34 m
WORLD_M = 2.0 * ORIGIN
TILE_PX = 256
DEFAULT_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "DSOC-BG-FloodMap-desktop/0.1 (robert.lesoski138@topper.wku.edu)"
ATTRIBUTION = "(c) OpenStreetMap contributors"
MAX_ZOOM = 16                             # never request finer tiles (policy: no heavy use; 2.39 m/px at the equator)
CONFIRM_TILES = 1500                      # more missing tiles than this -> the caller must confirm
MAX_TILES_HARD = 4000                     # absolute refusal limit
APPROX_KB_PER_TILE = 20.0                 # typical OSM raster tile (urban, z12-16) for the "approx MB" estimate
BAND_ROWS = 64                            # GPU upload granularity (rows)
CHUNK_ROWS = 64                           # rows per lock acquisition / GIL yield while compositing a big (up-sampled) tile
FALLBACK_RGB = (236, 233, 226)            # procedural fallback: neutral map paper
PAD = 2                                   # source pixels borrowed from neighbour tiles while resampling

SIZE_PRESETS = {"standard": 4096, "high": 6144, "ultra": 8192}   # long side of the composite in pixels


# ====================================================================================================== tile maths
def tile_span_m(z: int) -> float:
    """Mercator metres covered by one tile edge at zoom z."""
    return WORLD_M / (1 << z)


def meters_per_px(z: int, lat_deg: float = 0.0) -> float:
    """Ground metres per tile pixel (2.387 at z16 on the equator, x cos(lat) on the ground)."""
    return tile_span_m(z) / TILE_PX * math.cos(math.radians(lat_deg))


def merc_to_tilef(mx: float, my: float, z: int):
    """Fractional tile coordinates (x east, y SOUTH: tile row 0 is the north edge of the world map)."""
    n = 1 << z
    return (mx + ORIGIN) / WORLD_M * n, (ORIGIN - my) / WORLD_M * n


def tile_bounds_merc(z: int, x: int, y: int):
    """(xmin, ymin, xmax, ymax) in EPSG:3857 metres of tile z/x/y."""
    s = tile_span_m(z)
    return (-ORIGIN + x * s, ORIGIN - (y + 1) * s, -ORIGIN + (x + 1) * s, ORIGIN - y * s)


def lnglat_to_tile(lng: float, lat: float, z: int):
    """Slippy-map tile (x, y) containing lng/lat - the textbook formula, independent of the mercator metre code."""
    n = 1 << z
    x = int(math.floor((lng + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_range(frame: GridFrame, z: int):
    """Inclusive tile index range (x0, x1, y0, y1) that covers the grid footprint at zoom z."""
    n = 1 << z
    fx0, fy0 = merc_to_tilef(frame.xmin, frame.ymax, z)          # north-west corner
    fx1, fy1 = merc_to_tilef(frame.xmax, frame.ymin, z)          # south-east corner
    eps = 1e-9
    x0, x1 = int(math.floor(fx0 + eps)), int(math.ceil(fx1 - eps)) - 1
    y0, y1 = int(math.floor(fy0 + eps)), int(math.ceil(fy1 - eps)) - 1
    return max(x0, 0), min(max(x1, x0), n - 1), max(y0, 0), min(max(y1, y0), n - 1)


def tiles_for_frame(frame: GridFrame, z: int, focus_merc=None) -> list:
    """[(x, y)] covering the grid at zoom z, nearest to `focus_merc` (default the grid centre) first."""
    x0, x1, y0, y1 = tile_range(frame, z)
    fx, fy = merc_to_tilef(*(focus_merc or (frame.xc, frame.yc)), z)
    out = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    out.sort(key=lambda t: (t[0] + 0.5 - fx) ** 2 + (t[1] + 0.5 - fy) ** 2)
    return out


def _extent_long(frame: GridFrame) -> float:
    return max(frame.xmax - frame.xmin, frame.ymax - frame.ymin)


def choose_zoom(frame: GridFrame, long_px: int, max_zoom: int = MAX_ZOOM, min_zoom: int = 0) -> int:
    """Smallest zoom whose NATIVE size of the grid footprint is >= `long_px` on the long side, so the composite is made by
    down-sampling only (never up-sampling).  4096 -> z15 (4512 px native), 6144 / 8192 -> z16 (9024 px native)."""
    need = long_px * WORLD_M / (TILE_PX * _extent_long(frame))
    z = int(math.ceil(math.log2(need) - 1e-9))
    return max(min_zoom, min(z, max_zoom))


def composite_dims(frame: GridFrame, long_px: int):
    """(width, height) of the composite: `long_px` along the longer grid side, the other side in proportion (multiple of
    64 so row bands and mips line up).  It covers exactly the grid extent, so uv is shared with the flow grid."""
    w_m, h_m = frame.xmax - frame.xmin, frame.ymax - frame.ymin
    if h_m >= w_m:
        h, w = int(long_px), int(round(long_px * w_m / h_m / 64.0)) * 64
    else:
        w, h = int(long_px), int(round(long_px * h_m / w_m / 64.0)) * 64
    return max(w, 64), max(h, 64)


def composite_bytes(w: int, h: int, mips: bool = True) -> int:
    total, a, b = 0, w, h
    while True:
        total += a * b * 4
        if not mips or (a == 1 and b == 1):
            break
        a, b = max(a // 2, 1), max(b // 2, 1)
    return total


def composite_vram_mb(w: int, h: int, mips: bool = True) -> float:
    return composite_bytes(w, h, mips) / 1048576.0


def max_long_px_for_vram(frame: GridFrame, budget_mb: float, step: int = 256) -> int:
    """Largest long side (multiple of `step`) whose composite incl. mips fits `budget_mb` of VRAM."""
    lo = 512
    p = lo
    while True:
        q = p + step
        w, h = composite_dims(frame, q)
        if composite_vram_mb(w, h) > budget_mb:
            return p
        p = q
        if p > 16384:
            return p


def levels_for(zf: int) -> list:
    """Zoom levels loaded low -> high: an overview first (instant context), a middle one, then the final zoom."""
    return sorted({max(zf - 4, 4), max(zf - 2, 4), zf})


# ====================================================================================================== resampling
def _axis_weights(f0: float, f1: float, ndst: int, t: int):
    """1-D resampling weights for the destination pixels owned by tile column/row `t`.

    Destination axis: `ndst` pixels cover the fractional tile range [f0, f1) (in tile units).  A destination pixel is
    owned by the tile that contains its centre.  Returns (i0, i1, W) with W[i - i0, k] over the padded source pixels
    k = 0 .. 256 + 2*PAD - 1 (mosaic pixel index t*256 - PAD + k).  Down-sampling (>= 1 source px per dest px) uses an
    exact box filter, up-sampling bilinear."""
    span = f1 - f0
    i0 = int(math.ceil((t - f0) / span * ndst - 0.5))
    i1 = int(math.ceil((t + 1 - f0) / span * ndst - 0.5))
    i0, i1 = max(i0, 0), min(i1, ndst)
    if i1 <= i0:
        return i0, i0, None
    s = span * TILE_PX / ndst                                       # source px per destination px
    i = np.arange(i0, i1, dtype=np.float64)
    c = (f0 + (i + 0.5) / ndst * span) * TILE_PX                     # footprint centre, mosaic px
    P = TILE_PX + 2 * PAD
    left = (t * TILE_PX - PAD) + np.arange(P, dtype=np.float64)      # left edge of each padded source pixel
    if s >= 1.0:
        lo, hi = (c - 0.5 * s)[:, None], (c + 0.5 * s)[:, None]
        W = np.clip(np.minimum(hi, left[None, :] + 1.0) - np.maximum(lo, left[None, :]), 0.0, None)
    else:
        W = np.clip(1.0 - np.abs(left[None, :] + 0.5 - c[:, None]), 0.0, None)
    sums = W.sum(axis=1, keepdims=True)
    W = W / np.where(sums > 0, sums, 1.0)
    return i0, i1, W.astype(np.float32)


def _band(W: np.ndarray):
    """Dense weights (n, P) -> banded form (idx (n, K) int, w (n, K) float32); every row's non-zeros are contiguous."""
    nz = W > 0
    start = np.argmax(nz, axis=1)
    last = W.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1)
    K = int((last - start).max()) + 1
    idx = np.minimum(start[:, None] + np.arange(K)[None, :], W.shape[1] - 1)
    w = np.take_along_axis(W, idx, axis=1).astype(np.float32)
    w[idx != (start[:, None] + np.arange(K)[None, :])] = 0.0        # clipped duplicates carry no weight
    return idx, w


def _srgb_luts():
    v = np.arange(256, dtype=np.float32) / 255.0
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)
    x = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    enc = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return lin, np.clip(enc * 255.0 + 0.5, 0, 255).astype(np.uint8)


_LIN, _ENC = _srgb_luts()


def box_mip(src: np.ndarray, dst: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Update the rectangle [x0:x1, y0:y1] of `dst` (the next mip level) from `src` with a 2x2 box filter done in LINEAR light
    (colour channels) - what a hardware mip chain of an sRGB texture does.  Dimensions follow wgpu (floor, min 1)."""
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, dw), min(y1, dh)
    if x1 <= x0 or y1 <= y0:
        return
    h, w = y1 - y0, x1 - x0
    if 2 * y1 <= sh and 2 * x1 <= sw:                       # fast path: whole 2x2 blocks inside the source
        blk = src[2 * y0:2 * y1, 2 * x0:2 * x1]
        L = _LIN[blk[..., :3]]
        lin = L[0::2, 0::2] + L[0::2, 1::2]
        lin += L[1::2, 0::2]
        lin += L[1::2, 1::2]
        A = blk[..., 3].astype(np.uint16)
        alpha = A[0::2, 0::2] + A[0::2, 1::2] + A[1::2, 0::2] + A[1::2, 1::2]
    else:                                                    # border of an odd-sized level (or 1-pixel high / wide levels)
        yi, xi = np.arange(y0, y1) * 2, np.arange(x0, x1) * 2
        yj, xj = np.minimum(yi + 1, sh - 1), np.minimum(xi + 1, sw - 1)
        lin = 0.0
        alpha = 0
        for yy in (yi, yj):
            for xx in (xi, xj):
                t = src[yy[:, None], xx[None, :]]
                lin = lin + _LIN[t[..., :3]]
                alpha = alpha + t[..., 3].astype(np.uint32)
    idx = np.clip(lin * (0.25 * 4095.0) + 0.5, 0, 4095).astype(np.int32)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = _ENC[idx]
    out[..., 3] = ((alpha + 2) >> 2).astype(np.uint8)
    dst[y0:y1, x0:x1] = out


# ====================================================================================================== composite
class BasemapComposite:
    """RGBA8 composite of tiles in the flow-grid frame (row 0 north) with a CPU mip chain and dirty-band tracking."""

    def __init__(self, frame: GridFrame, long_px: int = 4096, zoom: int | None = None):
        self.frame = frame
        self.long_px = int(long_px)
        self.width, self.height = composite_dims(frame, long_px)
        self.zoom = int(zoom) if zoom is not None else choose_zoom(frame, long_px)
        self.lock = threading.RLock()
        self.rgba = np.zeros((self.height, self.width, 4), np.uint8)
        self.mips = [self.rgba]
        w, h = self.width, self.height
        while w > 1 or h > 1:
            w, h = max(w // 2, 1), max(h // 2, 1)
            self.mips.append(np.zeros((h, w, 4), np.uint8))
        self.n_bands = (self.height + BAND_ROWS - 1) // BAND_ROWS
        self.band_gen = np.zeros(self.n_bands, np.int64)
        self.gen = 0
        self.filled = False                                  # fallback / any content written yet
        self.fallback_gen = 0                                # generation that means "fallback colour everywhere"
        self.tiles_done: dict[int, set] = {}                 # zoom -> {(x, y)} rendered
        self.gaps: dict[tuple, frozenset] = {}               # (z, x, y) -> neighbour tiles that were missing when it rendered
        self.n_final = len(tiles_for_frame(frame, self.zoom))
        # tile-coordinate range of the composite at each zoom is computed on demand (see _extent_tiles)

    # ---------------------------------------------------------------------------------------- bookkeeping
    @property
    def n_levels(self) -> int:
        return len(self.mips)

    @property
    def coverage(self) -> float:
        """Fraction of the FINAL-zoom tiles already in the composite (0..1)."""
        return len(self.tiles_done.get(self.zoom, ())) / max(self.n_final, 1)

    @property
    def has_tiles(self) -> bool:
        return any(len(s) for s in self.tiles_done.values())

    def vram_mb(self) -> float:
        return composite_vram_mb(self.width, self.height)

    def _mark(self, y0: int, y1: int):
        b0, b1 = y0 // BAND_ROWS, (y1 - 1) // BAND_ROWS
        self.gen += 1
        self.band_gen[b0:b1 + 1] = self.gen

    def mark_all_dirty(self):
        with self.lock:
            self.gen += 1
            self.band_gen[:] = self.gen

    def _update_mips(self, x0, y0, x1, y1):
        for k in range(1, len(self.mips)):
            x0, y0, x1, y1 = x0 >> 1, y0 >> 1, (x1 + 1) >> 1, (y1 + 1) >> 1
            box_mip(self.mips[k - 1], self.mips[k], x0, y0, x1, y1)

    def rebuild_mips(self):
        """Recompute every mip level from level 0 (use after writing `rgba` directly)."""
        with self.lock:
            for k in range(1, len(self.mips)):
                h = self.mips[k].shape[0]
                for r in range(0, h, 128):
                    box_mip(self.mips[k - 1], self.mips[k], 0, r, self.mips[k].shape[1], min(r + 128, h))
            self.filled = True
            self.mark_all_dirty()

    # ---------------------------------------------------------------------------------------- fallback
    def fill_fallback(self):
        """Neutral map-paper tone in every level (flat colour, so no mip maths is needed and it costs ~tens of ms).  Shown while
        no real tile is available and under tiles that have not arrived yet.  Bands are NOT marked dirty one by one: the GPU
        side clears its texture to the same colour (BasemapTexture.adopt_fallback) instead of uploading 75-300 MB of paper;
        `fallback_gen` is the generation that stands for "fallback everywhere", later tile writes get higher generations."""
        px = np.array([*FALLBACK_RGB, 255], np.uint8)
        for m in self.mips:
            for r in range(0, m.shape[0], 256):
                with self.lock:
                    m[r:r + 256] = px
        with self.lock:
            self.filled = True
            self.gen += 1
            self.band_gen[:] = self.gen
            self.fallback_gen = self.gen

    # ---------------------------------------------------------------------------------------- tiles
    def put_tile(self, z: int, x: int, y: int, tile: np.ndarray, getter=None) -> bool:
        """Resample tile z/x/y (uint8 [256, 256, 3|4]) into the composite pixels it owns.  `getter(z, x, y) -> tile | None`
        supplies neighbour tiles for seamless filtering (missing neighbours are edge-replicated).  Returns False when the
        tile does not intersect the composite.  The write happens in row chunks (the lock is only held per chunk, so a
        frame thread that uploads never waits long)."""
        n = 1 << z
        fr = self.frame
        fx0, fy0 = merc_to_tilef(fr.xmin, fr.ymax, z)
        fx1, fy1 = merc_to_tilef(fr.xmax, fr.ymin, z)
        i0, i1, Wx = _axis_weights(fx0, fx1, self.width, x)
        j0, j1, Wy = _axis_weights(fy0, fy1, self.height, y)
        if Wx is None or Wy is None:
            return False
        tile = np.asarray(tile)[..., :3]
        P = TILE_PX + 2 * PAD
        pad = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
        gap = []
        if getter is not None:
            def seg(d):
                if d < 0:
                    return slice(0, PAD), slice(TILE_PX - PAD, TILE_PX)
                if d == 0:
                    return slice(PAD, PAD + TILE_PX), slice(0, TILE_PX)
                return slice(PAD + TILE_PX, P), slice(0, PAD)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx_, ny_ = x + dx, y + dy
                    if nx_ < 0 or ny_ < 0 or nx_ >= n or ny_ >= n:
                        continue
                    nb = getter(z, nx_, ny_)
                    if nb is None:
                        gap.append((nx_, ny_))
                        continue
                    (dsx, ssx), (dsy, ssy) = seg(dx), seg(dy)
                    pad[dsy, dsx] = np.asarray(nb)[ssy, ssx, :3]
        self_gap = frozenset(gap)
        padf = pad.astype(np.float32)
        ix, wx = _band(Wx)                                           # (nx, K) source indices / weights per dest column
        iy, wy = _band(Wy)
        a = j0
        chunk = CHUNK_ROWS if (j1 - j0) > 5 * CHUNK_ROWS else (j1 - j0)      # small tiles: one pass; big up-sampled ones in slices
        while a < j1:
            b = min(a + chunk, j1)
            r0, r1 = a - j0, b - j0
            tmp = np.take(padf, iy[r0:r1, 0], axis=0)
            tmp *= wy[r0:r1, 0, None, None]
            for k in range(1, iy.shape[1]):
                g = np.take(padf, iy[r0:r1, k], axis=0)
                g *= wy[r0:r1, k, None, None]
                tmp += g                                                        # (rows, P, 3)
            out = np.take(tmp, ix[:, 0], axis=1)
            out *= wx[None, :, 0, None]
            for k in range(1, ix.shape[1]):
                g = np.take(tmp, ix[:, k], axis=1)
                g *= wx[None, :, k, None]
                out += g                                                        # (rows, nx, 3)
            u8 = np.clip(out + 0.5, 0, 255).astype(np.uint8)
            with self.lock:
                self.rgba[a:b, i0:i1, :3] = u8
                self.rgba[a:b, i0:i1, 3] = 255
                self._update_mips(i0, a, i1, b)
                self._mark(a, b)
            a = b
            if b < j1:
                time.sleep(0.0002)                                   # let the render thread have the GIL between slices
        with self.lock:
            self.tiles_done.setdefault(z, set()).add((x, y))
            if self_gap:
                self.gaps[(z, x, y)] = self_gap
            else:
                self.gaps.pop((z, x, y), None)
        return True

    def tiles_with_gaps(self, z: int) -> list:
        """[(x, y, missing_neighbours)] rendered at zoom z while some neighbour tile was not yet available."""
        with self.lock:
            return [(x, y, g) for (zz, x, y), g in self.gaps.items() if zz == z]

    # ---------------------------------------------------------------------------------------- export (debug / tests)
    def to_image(self, max_side: int | None = None):
        from PIL import Image
        with self.lock:
            im = Image.fromarray(self.rgba[..., :3].copy())
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(int(im.width * s), 1), max(int(im.height * s), 1)), Image.LANCZOS)
        return im


# ====================================================================================================== tile source
class Blocked(Exception):
    """The server refused us (HTTP 401/403/...): stop all requests."""


class RateLimiter:
    """Global minimum interval between request STARTS across all threads (polite client)."""

    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self, cancel: threading.Event | None = None):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next)
            self._next = at + self.min_interval
        delay = at - now
        if delay > 0:
            if cancel is not None:
                cancel.wait(delay)
            else:
                time.sleep(delay)


def decode_tile(data: bytes) -> np.ndarray:
    """PNG/JPEG bytes -> uint8 [256, 256, 3].  Retina (512) or odd sizes are box-resampled to 256."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    im.load()
    im = im.convert("RGB")
    if im.size != (TILE_PX, TILE_PX):
        im = im.resize((TILE_PX, TILE_PX), Image.BOX)
    return np.asarray(im, dtype=np.uint8)


class TileSource:
    """Tile provider.  Order: read-only `local_dir` (z/x/y.png), the disk cache, then (only when asked) the network."""

    def __init__(self, url_template: str | None = DEFAULT_URL, cache_root=None, local_dir=None,
                 user_agent: str = USER_AGENT, timeout: float = 12.0, retries: int = 3, backoff: float = 1.0,
                 delay: float = 0.30, min_interval: float = 0.15, network: bool = True, max_bytes: int = 2_000_000):
        self.url_template = url_template or None
        self.local_dir = Path(local_dir) if local_dir else None
        if cache_root is None:
            if self.url_template in (None, DEFAULT_URL):
                name = "tiles"
            else:
                name = "tiles_" + hashlib.sha1(self.url_template.encode()).hexdigest()[:8]
            cache_root = config.cache_dir() / name
        self.cache_root = Path(cache_root)
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.delay = float(delay)
        self.network = bool(network) and bool(self.url_template)
        self.max_bytes = int(max_bytes)
        self.limiter = RateLimiter(min_interval)
        self.requests = 0                                    # network requests issued (tests / reporting)

    # ---------------------------------------------------------------------------------------- disk
    def cache_path(self, z: int, x: int, y: int) -> Path:
        return self.cache_root / str(z) / str(x) / f"{y}.png"

    def _paths(self, z, x, y):
        if self.local_dir is not None:
            yield self.local_dir / str(z) / str(x) / f"{y}.png"
        yield self.cache_path(z, x, y)

    def has_cached(self, z: int, x: int, y: int) -> bool:
        return any(p.is_file() for p in self._paths(z, x, y))

    def read_cached(self, z: int, x: int, y: int) -> bytes | None:
        for p in self._paths(z, x, y):
            try:
                return p.read_bytes()
            except OSError:
                continue
        return None

    def drop_cached(self, z, x, y):
        try:
            self.cache_path(z, x, y).unlink()
        except OSError:
            pass

    def _store(self, z, x, y, data: bytes):
        p = self.cache_path(z, x, y)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    # ---------------------------------------------------------------------------------------- network
    def url(self, z, x, y) -> str:
        return self.url_template.format(z=z, x=x, y=y)

    def fetch(self, z: int, x: int, y: int, cancel: threading.Event | None = None):
        """Download one tile (rate limited, retried) into the disk cache.  Returns (bytes | None, status) with status
        'net' | 'missing' (404/410) | 'error' | 'offline' | 'cancelled'.  Raises Blocked on 401/403/451."""
        if not self.network:
            return None, "offline"
        last = "error"
        for attempt in range(self.retries + 1):
            if cancel is not None and cancel.is_set():
                return None, "cancelled"
            if attempt:
                wait = self.backoff * (2 ** (attempt - 1)) * (0.75 + 0.5 * random.random())
                wait = min(wait, 60.0)
                if cancel is not None:
                    if cancel.wait(wait):
                        return None, "cancelled"
                else:
                    time.sleep(wait)
            self.limiter.wait(cancel)
            req = urllib.request.Request(self.url(z, x, y), headers={"User-Agent": self.user_agent,
                                                                     "Accept": "image/png,image/*;q=0.8"})
            self.requests += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read(self.max_bytes + 1)
                if len(data) > self.max_bytes:
                    last = "error"
                    continue
                try:
                    decode_tile(data)                       # never cache garbage (captive portals, HTML error pages)
                except Exception:
                    last = "error"
                    continue
                self._store(z, x, y, data)
                if self.delay > 0:
                    (cancel.wait(self.delay) if cancel is not None else time.sleep(self.delay))
                return data, "net"
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    return None, "missing"
                if e.code in (401, 403, 451):
                    raise Blocked(f"tile server refused the request (HTTP {e.code})")
                last = "error"
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    try:
                        extra = min(float(ra), 60.0) if ra else 0.0
                    except ValueError:
                        extra = 0.0
                    if extra:
                        (cancel.wait(extra) if cancel is not None else time.sleep(extra))
                    continue
                return None, "error"                         # other 4xx: retrying will not help
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                last = "error"
                continue
        return None, last


# ====================================================================================================== loader
@dataclass
class Plan:
    zoom: int
    levels: list                         # [(z, [(x, y), ...])] low -> high zoom
    total: int
    cached: int
    missing: int
    approx_mb: float                     # download size estimate of the missing tiles
    composite_px: tuple
    vram_mb: float
    needs_confirm: bool
    final_tiles: int


@dataclass
class LoadStatus:
    phase: str = "idle"                  # idle | reading | downloading | done | cancelled | unavailable | error
    message: str = ""
    level: int = 0                       # zoom being processed
    done: int = 0                        # tiles processed (rendered or skipped)
    total: int = 0
    from_cache: int = 0
    downloaded: int = 0
    absent: int = 0                      # tiles the server does not have (404) / not cached while reading offline
    failed: int = 0
    bytes_down: int = 0
    t0: float = 0.0
    t_end: float = 0.0
    downloading: bool = False
    stamp: int = 0                       # increments on every update (cheap change detection)

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def running(self) -> bool:
        return self.phase in ("reading", "downloading")


class ConfirmationRequired(Exception):
    def __init__(self, n: int):
        super().__init__(f"{n} tiles needed (> {CONFIRM_TILES}): confirmation required")
        self.n = n


class BasemapLoader:
    def __init__(self, composite: BasemapComposite, source: TileSource, threads: int = 4, progress=None,
                 lru_tiles: int = 256, duty: float = 1.0):
        self.comp = composite
        self.src = source
        self.threads = max(1, int(threads))
        self.duty = min(max(float(duty), 0.05), 1.0)          # compositing CPU duty cycle (the GPU layer uses 0.5)
        self.progress = progress
        self._lock = threading.RLock()
        self._status = LoadStatus()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._lru: OrderedDict = OrderedDict()
        self._lru_cap = int(lru_tiles)
        self._consec_fail = 0
        self._abort_msg: str | None = None
        self._compose_lock = threading.Lock()                # one thread composites at a time (others download / decode)

    # ---------------------------------------------------------------------------------------- plan
    def plan(self, levels: list | None = None) -> Plan:
        c = self.comp
        zs = levels or levels_for(c.zoom)
        lv, total, cached = [], 0, 0
        for z in zs:
            tiles = tiles_for_frame(c.frame, z)
            lv.append((z, tiles))
            total += len(tiles)
            cached += sum(1 for (x, y) in tiles if self.src.has_cached(z, x, y))
        missing = total - cached
        return Plan(zoom=c.zoom, levels=lv, total=total, cached=cached, missing=missing,
                    approx_mb=missing * APPROX_KB_PER_TILE / 1024.0, composite_px=(c.width, c.height),
                    vram_mb=c.vram_mb(), needs_confirm=missing > CONFIRM_TILES, final_tiles=len(lv[-1][1]))

    # ---------------------------------------------------------------------------------------- status
    def status(self) -> LoadStatus:
        with self._lock:
            return replace(self._status)

    def _set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self._status, k, v)
            self._status.stamp += 1
            st = replace(self._status)
        if self.progress is not None:
            try:
                self.progress(st)
            except Exception:
                pass

    def _bump(self, **inc):
        with self._lock:
            for k, v in inc.items():
                setattr(self._status, k, getattr(self._status, k) + v)
            self._status.stamp += 1
            st = replace(self._status)
        if self.progress is not None:
            try:
                self.progress(st)
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------------------------------------- control
    def start(self, download: bool = False, confirmed: bool = False, levels: list | None = None, plan: Plan | None = None):
        """Begin loading in a background thread.  download=False reads only the cache / local folder (no network).
        download=True fetches the missing tiles of the plan; raises ConfirmationRequired when more than CONFIRM_TILES
        tiles are missing and `confirmed` is False, ValueError beyond MAX_TILES_HARD.  No-op while running."""
        if self.running:
            return
        plan = plan or self.plan(levels)
        if download:
            if plan.missing > MAX_TILES_HARD:
                raise ValueError(f"refusing to download {plan.missing} tiles (limit {MAX_TILES_HARD})")
            if plan.needs_confirm and not confirmed:
                raise ConfirmationRequired(plan.missing)
        if not download and plan.levels and all(self.src.has_cached(plan.zoom, x, y) for x, y in plan.levels[-1][1]):
            # the final zoom is complete on disk: the overview levels would only be painted over (and cost seconds)
            plan = replace(plan, levels=plan.levels[-1:], total=len(plan.levels[-1][1]),
                           cached=len(plan.levels[-1][1]), missing=0)
        self._cancel.clear()
        self._consec_fail = 0
        self._abort_msg = None
        self._set(phase="downloading" if download else "reading", message="", level=0, done=0, total=plan.total,
                  from_cache=0, downloaded=0, absent=0, failed=0, bytes_down=0, t0=time.time(), t_end=0.0,
                  downloading=bool(download))
        self._thread = threading.Thread(target=self._run, args=(plan, download), name="basemap-loader", daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel.set()

    def join(self, timeout: float | None = None):
        if self._thread is not None:
            self._thread.join(timeout)

    # ---------------------------------------------------------------------------------------- worker
    def _lru_get(self, key):
        with self._lock:
            a = self._lru.get(key)
            if a is not None:
                self._lru.move_to_end(key)
            return a

    def _lru_put(self, key, arr):
        with self._lock:
            self._lru[key] = arr
            self._lru.move_to_end(key)
            while len(self._lru) > self._lru_cap:
                self._lru.popitem(last=False)

    def _neighbour(self, z, x, y):
        a = self._lru_get((z, x, y))
        if a is not None:
            return a
        data = self.src.read_cached(z, x, y)
        if data is None:
            return None
        try:
            a = decode_tile(data)
        except Exception:
            return None
        self._lru_put((z, x, y), a)
        return a

    def _tile_task(self, z, x, y, download: bool):
        if self._cancel.is_set() or self._abort_msg:
            return
        data, how = self.src.read_cached(z, x, y), "cache"
        if data is None and download:
            try:
                data, how = self.src.fetch(z, x, y, self._cancel)
            except Blocked as e:
                self._abort_msg = str(e)
                self._cancel.set()
                return
            if how == "net":
                self._consec_fail = 0
                self._bump(downloaded=1, bytes_down=len(data))
            elif how == "missing":
                self._consec_fail = 0
                self._bump(absent=1)
                return
            elif how == "error":
                self._consec_fail += 1
                self._bump(failed=1)
                if self._consec_fail >= 8:
                    self._abort_msg = "network unavailable (8 requests in a row failed)"
                    self._cancel.set()
                return
            else:                                            # cancelled / offline
                return
        elif data is None:
            self._bump(absent=1)
            return
        else:
            self._bump(from_cache=1)
        try:
            arr = decode_tile(data)
        except Exception:
            if how == "cache":
                self.src.drop_cached(z, x, y)                # damaged file: forget it so a download can replace it
            self._bump(failed=1)
            return
        self._lru_put((z, x, y), arr)
        with self._compose_lock:
            t0 = time.perf_counter()
            self.comp.put_tile(z, x, y, arr, self._neighbour)
            self._throttle(time.perf_counter() - t0)

    def _throttle(self, spent: float):
        """Keep the compositing duty cycle at `duty` (default 50 %) so the render thread is not starved of the GIL / CPU
        while tiles stream in.  Called inside the compose lock, so all workers share the budget."""
        if self.duty < 1.0 and spent > 0:
            time.sleep(spent * (1.0 / self.duty - 1.0))

    def _refine(self, z: int):
        """Tiles that rendered before a neighbour arrived were edge-replicated at that border: redo them now so the
        result does not depend on download timing (a seam pixel is at most 2 source px wide, but it is exact now)."""
        for x, y, gap in self.comp.tiles_with_gaps(z):
            if self._cancel.is_set():
                return
            if not any(self._neighbour(z, gx, gy) is not None for gx, gy in gap):
                continue
            arr = self._neighbour(z, x, y)
            if arr is not None:
                with self._compose_lock:
                    t0 = time.perf_counter()
                    self.comp.put_tile(z, x, y, arr, self._neighbour)
                    self._throttle(time.perf_counter() - t0)

    def _run(self, plan: Plan, download: bool):
        try:
            if not self.comp.filled:
                self.comp.fill_fallback()
            with ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="basemap-tile") as ex:
                for z, tiles in plan.levels:
                    if self._cancel.is_set():
                        break
                    self._set(level=z)
                    futs = [ex.submit(self._tile_task, z, x, y, download) for (x, y) in tiles]
                    for f in futs:
                        try:
                            f.result()
                        except Exception:                    # one broken tile must not stop the rest
                            self._bump(failed=1)
                        self._bump(done=1)
                    self._refine(z)
        except Exception as e:                               # pragma: no cover - defensive
            self._set(phase="error", message=f"{type(e).__name__}: {e}", t_end=time.time())
            return
        st = self.status()
        if self._abort_msg:
            self._set(phase="unavailable", message=self._abort_msg, t_end=time.time())
        elif self._cancel.is_set():
            self._set(phase="cancelled", message="cancelled", t_end=time.time())
        else:
            have = self.comp.has_tiles
            if not have:
                msg = ("no map tiles cached for this area" if not download else "no tiles could be downloaded")
                self._set(phase="unavailable", message=msg, t_end=time.time())
            else:
                self._set(phase="done", message="", t_end=time.time())
        self._lru.clear()
        del st
