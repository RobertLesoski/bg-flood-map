"""Read-only windowed raster access with a shared block cache (rasterio; used for the depth tiles and the DEM).

    r = BlockRaster(path, scale=0.01, nodata=65535)          # opened lazily, never written
    v = r.sample(x, y)                                       # bilinear at points in the raster's CRS, NaN = no data / outside
    r.read_rows(r0, r1)                                      # a band of rows at full resolution (float32, NaN = no data)

Only the file blocks that the requested points fall in are read (one `ds.read(window=block)` each, a few ms), and decoded blocks
live in ONE process-wide LRU (`BLOCK_CACHE`, default 192 MB) shared by every raster, so a profile along a line, the 25 hours x 12
storms of a point's time series and a repeated query cost almost nothing after the first read.  A raster is a north-up grid (no
rotation), values are `raw * scale`, `nodata` raw values become NaN.

Thread safety: every dataset has its own lock (GDAL datasets are not thread-safe); the cache has one more.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np


class BlockCache:
    """LRU of decoded blocks bounded by bytes."""

    def __init__(self, max_bytes: int = 192 << 20):
        self.max_bytes = int(max_bytes)
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
                self.hits += 1
            return v

    def put(self, key, arr: np.ndarray):
        with self._lock:
            if key in self._d:
                return
            self.misses += 1
            self._d[key] = arr
            self._bytes += arr.nbytes
            while self._bytes > self.max_bytes and len(self._d) > 1:
                _, old = self._d.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self):
        with self._lock:
            self._d.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self):
        return len(self._d)


BLOCK_CACHE = BlockCache()
_COGS: dict = {}                                                # path -> bgv.data.cog.Cog or None (not a native-readable COG)
_OPEN: "OrderedDict[str, object]" = OrderedDict()           # path -> rasterio dataset (LRU, so 300 hourly files never exhaust handles)
_OPEN_LOCK = threading.Lock()
MAX_OPEN = 96


def _open(path: str):
    import rasterio
    with _OPEN_LOCK:
        ds = _OPEN.get(path)
        if ds is not None:
            _OPEN.move_to_end(path)
            return ds
        ds = rasterio.open(path)
        _OPEN[path] = ds
        while len(_OPEN) > MAX_OPEN:
            _, old = _OPEN.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return ds


def close_all():
    with _OPEN_LOCK:
        for ds in _OPEN.values():
            try:
                ds.close()
            except Exception:
                pass
        _OPEN.clear()
    BLOCK_CACHE.clear()


_LOCKS: dict = {}


def _lock_for(path: str):
    with _OPEN_LOCK:
        lk = _LOCKS.get(path)
        if lk is None:
            lk = _LOCKS[path] = threading.RLock()
        return lk


class BlockRaster:
    def __init__(self, path, scale: float = 1.0, nodata=None, cache: BlockCache | None = None, block: int | None = None):
        self.path = str(path)
        self.scale = float(scale)
        self._nodata_override = nodata
        self.cache = cache if cache is not None else BLOCK_CACHE      # (an empty BlockCache is falsy: len() == 0)
        self._info = None
        self._block_override = block

    # ---------------------------------------------------------------------------------------- metadata
    def info(self) -> dict:
        if self._info is None:
            with _lock_for(self.path):
                ds = _open(self.path)
                t = ds.transform
                if abs(t.b) > 1e-12 or abs(t.d) > 1e-12 or t.e >= 0:
                    raise ValueError(f"{self.path}: only north-up rasters are supported")
                bh, bw = ds.block_shapes[0]
                if self._block_override:
                    bh = bw = self._block_override
                elif bw >= ds.width and bh <= 16:                # striped file: read a band of rows per block (a few MB)
                    bh = max(bh, 64)
                nodata = self._nodata_override if self._nodata_override is not None else ds.nodata
                self._info = dict(width=ds.width, height=ds.height, left=t.c, top=t.f, resx=t.a, resy=-t.e,
                                  crs=ds.crs, dtype=ds.dtypes[0], nodata=nodata, bh=int(bh), bw=int(min(bw, ds.width)),
                                  bounds=tuple(ds.bounds))
        return self._info

    @property
    def crs(self):
        return self.info()["crs"]

    # ---------------------------------------------------------------------------------------- blocks
    def _block(self, br: int, bc: int) -> np.ndarray:
        i = self.info()
        key = (self.path, br, bc)
        a = self.cache.get(key)
        if a is not None:
            return a
        from rasterio.windows import Window
        r0, c0 = br * i["bh"], bc * i["bw"]
        h, w = min(i["bh"], i["height"] - r0), min(i["bw"], i["width"] - c0)
        raw = None
        if self.path.lower().endswith(".tif"):
            try:                                                      # COG tiles: native reader + the ONE shared tile LRU (cog.TILE_CACHE)
                from ..data import cog as cogmod
                cg = _COGS.get(self.path)
                if cg is None and self.path not in _COGS:
                    try:
                        cg = cogmod.open_cog(self.path)
                    except Exception:
                        cg = None
                    _COGS[self.path] = cg
                if cg is not None and len(cg.levels) and (cg.width, cg.height) == (i["width"], i["height"]):
                    raw = cg.read_window(0, c0, r0, c0 + w, r0 + h)
            except Exception:
                raw = None
        if raw is None:
            with _lock_for(self.path):
                ds = _open(self.path)
                raw = ds.read(1, window=Window(c0, r0, w, h))
        a = self._decode(raw)
        self.cache.put(key, a)
        return a

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        nd = self.info()["nodata"]
        # scale in float64, THEN round to float32: float32(10) * float32(0.01) = 0.09999999 would put a 10 cm cell below the 0.1 m
        # flooded threshold (the whole-array test caught 96 buildings lost that way)
        out = (raw.astype(np.float64) * self.scale).astype(np.float32) if self.scale != 1.0 else raw.astype(np.float32)
        if nd is not None:
            out[raw == nd] = np.nan
        return out

    def _gather(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Decoded values at integer (row, col) arrays (all inside the raster)."""
        i = self.info()
        bh, bw = i["bh"], i["bw"]
        br, bc = rows // bh, cols // bw
        nbc = -(-i["width"] // bw)
        bid = br * nbc + bc
        out = np.empty(rows.shape, np.float32)
        order = np.argsort(bid, kind="stable")
        sb = bid[order]
        cuts = np.flatnonzero(np.diff(sb)) + 1
        for grp in np.split(order, cuts):
            if grp.size == 0:
                continue
            b = int(bid[grp[0]])
            blk = self._block(b // nbc, b % nbc)
            out[grp] = blk[rows[grp] - (b // nbc) * bh, cols[grp] - (b % nbc) * bw]
        return out

    # ---------------------------------------------------------------------------------------- sampling
    def pixel_coords(self, x, y):
        i = self.info()
        return (np.asarray(x, np.float64) - i["left"]) / i["resx"], (i["top"] - np.asarray(y, np.float64)) / i["resy"]

    def inside(self, x, y, margin_px: float = 0.5) -> np.ndarray:
        i = self.info()
        gx, gy = self.pixel_coords(x, y)
        return (gx >= -margin_px) & (gx <= i["width"] + margin_px) & (gy >= -margin_px) & (gy <= i["height"] + margin_px)

    def sample(self, x, y, method: str = "bilinear") -> np.ndarray:
        """Values at points (x, y) in the raster's CRS: bilinear between the four nearest pixel centres (weights renormalised
        over the valid ones: a no-data neighbour is skipped, all four no-data -> NaN), or 'nearest'.  Points outside the raster
        (by more than half a pixel) are NaN."""
        i = self.info()
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        shape = x.shape
        gx, gy = self.pixel_coords(x.reshape(-1), y.reshape(-1))
        W, H = i["width"], i["height"]
        ok = (gx >= 0.0) & (gx <= W) & (gy >= 0.0) & (gy <= H)
        out = np.full(gx.shape, np.nan, np.float64)
        if not ok.any():
            return out.reshape(shape)
        g_x, g_y = gx[ok], gy[ok]
        if method == "nearest":
            c = np.clip(np.floor(g_x).astype(np.int64), 0, W - 1)
            r = np.clip(np.floor(g_y).astype(np.int64), 0, H - 1)
            out[ok] = self._gather(r, c)
            return out.reshape(shape)
        fx, fy = g_x - 0.5, g_y - 0.5
        x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        tx, ty = fx - x0, fy - y0
        xa, xb = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
        ya, yb = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
        # one gather for the four corners (shared block lookups)
        rr = np.concatenate([ya, ya, yb, yb])
        cc = np.concatenate([xa, xb, xa, xb])
        v = self._gather(rr, cc).astype(np.float64).reshape(4, -1)
        valid = np.isfinite(v)
        w = np.where(valid, w, 0.0)
        ws = w.sum(axis=0)
        num = np.where(valid, v, 0.0) * w
        res = np.full(ws.shape, np.nan)
        good = ws > 1e-12
        res[good] = num.sum(axis=0)[good] / ws[good]
        out[ok] = res
        return out.reshape(shape)

    # ---------------------------------------------------------------------------------------- bands
    def read_rows(self, r0: int, r1: int, out_shape=None, resampling=None) -> np.ndarray:
        """Rows [r0, r1) at full resolution (float32, NaN = no data, scaled), bypassing the block cache (bulk reads)."""
        from rasterio.windows import Window
        i = self.info()
        r1 = min(r1, i["height"])
        with _lock_for(self.path):
            ds = _open(self.path)
            raw = ds.read(1, window=Window(0, r0, i["width"], r1 - r0))
        return self._decode(raw)

    def read_overview(self, factor: int) -> np.ndarray:
        """The whole raster decimated by `factor` (uses the file's overviews when they exist): float32, NaN = no data."""
        import rasterio
        i = self.info()
        shape = (-(-i["height"] // factor), -(-i["width"] // factor))
        with _lock_for(self.path):
            ds = _open(self.path)
            raw = ds.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.average)
        return self._decode(raw)
