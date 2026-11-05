"""Minimal Cloud-Optimized GeoTIFF reader for the solver depth tiles (pure python + numpy + zlib, no GDAL / rasterio).

Reads the files written by `scripts/make_cogs.py`: EPSG:3857, ONE band, uint16 depth in CENTIMETRES (65535 = outside the model),
512 px tiles, DEFLATE (+ horizontal predictor 2), average overviews as extra IFDs.  Classic TIFF and BigTIFF, little or big endian.
Same conventions as the web viewer's `cog/cog-depth.js`.

    cog = open_cog(path)                                  # parses the IFDs only (a few KB), no pixel is read
    cog.sample_merc(x, y, level=0) -> (status, raw)       # status: "value" | "nodata" | "outside"
    cog.sample_many(xs, ys, level=0) -> (raw u16[n], status u8[n])    # status 0 value, 1 nodata, 2 outside; vectorised per tile
    cog.read_window(level, c0, r0, c1, r1, fill=None) -> u16[h, w]
    cog.level_for(merc_per_px) -> int
    lnglat_to_merc / merc_to_lnglat (scalars and arrays)  spherical Web Mercator, R = 6378137 (the file's CRS)

Decoded tiles live in ONE process-wide LRU (`TILE_CACHE`, capped by bytes, thread safe, newest last).  A tile is read with a single
seek + read under a per-file lock, inflated and un-predicted outside every lock.

Depth semantics (be exact when labelling): max_depth.tif = largest depth the solver reached at the cell over the whole run;
depth_<N*3600>s.tif = depth at the END of simulated hour N.  Values are the solver's float metres resampled bilinearly from the 5 m
UTM grid to the 6.25 m mercator grid (about 5 m on the ground) and rounded to whole centimetres.
"""
from __future__ import annotations

import os
import struct
import threading
import zlib
from collections import OrderedDict

import numpy as np

R_EARTH = 6378137.0
MAX_LAT = 85.0511287798066
NODATA = 65535
M2FT = 3.28084

ST_VALUE, ST_NODATA, ST_OUTSIDE = 0, 1, 2


class CogError(Exception):
    pass


def lnglat_to_merc(lng, lat):
    lat = np.clip(lat, -MAX_LAT, MAX_LAT)
    return R_EARTH * np.radians(lng), R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def merc_to_lnglat(x, y):
    return np.degrees(x / R_EARTH), np.degrees(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2)


# ------------------------------------------------------------------------------------------------ tile cache
class TileCache:
    """Byte-capped LRU of decoded tiles: key -> ndarray.  Thread safe."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.hits = self.misses = 0
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            a = self._d.get(key)
            if a is not None:
                self._d.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return a

    def put(self, key, arr):
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= old.nbytes
            self._d[key] = arr
            self.bytes += arr.nbytes
            while self.bytes > self.max_bytes and len(self._d) > 1:
                k, v = self._d.popitem(last=False)
                self.bytes -= v.nbytes

    def clear(self):
        with self._lock:
            self._d.clear()
            self.bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return {"tiles": len(self._d), "mb": self.bytes / 1048576.0, "max_mb": self.max_bytes / 1048576.0,
                    "hits": self.hits, "misses": self.misses}

    def resize(self, max_bytes: int):
        with self._lock:
            self.max_bytes = int(max_bytes)
        self.put(("_", 0), np.zeros(0, np.uint8))        # trims


TILE_CACHE = TileCache()


# ------------------------------------------------------------------------------------------------ TIFF parsing
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 2: "c", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I", 16: "Q", 17: "q", 18: "Q"}
T_NEW_SUBFILE, T_WIDTH, T_HEIGHT, T_BITS, T_COMPRESSION, T_SAMPLES = 254, 256, 257, 258, 259, 277
T_PREDICTOR, T_TILE_W, T_TILE_H, T_TILE_OFFSETS, T_TILE_COUNTS, T_SAMPLE_FORMAT = 317, 322, 323, 324, 325, 339
T_PIXEL_SCALE, T_TIEPOINT, T_TRANSFORM, T_GEOKEYS, T_NODATA = 33550, 33922, 34264, 34735, 42113


class _Level:
    __slots__ = ("index", "width", "height", "tw", "th", "tx", "ty", "offsets", "counts", "comp", "pred", "dtype", "bits")


def _values(raw: bytes, ftype: int, count: int, endian: str):
    if ftype == 2:
        return raw[:count].decode("latin1").rstrip("\0")
    if ftype in (5, 10):
        v = struct.unpack(f"{endian}{2 * count}{'I' if ftype == 5 else 'i'}", raw[:8 * count])
        return [v[i] / v[i + 1] for i in range(0, len(v), 2)]
    return list(struct.unpack(f"{endian}{count}{_TYPE_FMT[ftype]}", raw[:_TYPE_SIZE[ftype] * count]))


def _array(raw: bytes, ftype: int, count: int, endian: str):
    dt = np.dtype({4: "u4", 16: "u8", 3: "u2"}[ftype]).newbyteorder(endian)
    return np.frombuffer(raw, dtype=dt, count=count).astype(np.int64)


class Cog:
    """One opened COG file (a reader; tiles come from the shared `TILE_CACHE`)."""

    def __init__(self, path, f, levels, geo, size, endian):
        self.path = str(path)
        self._f = f
        self._flock = threading.Lock()
        self.levels = levels
        self.size = size
        self.endian = endian
        self.nodata = geo["nodata"]
        self.x0, self.y0, self.res = geo["x0"], geo["y0"], geo["res"]
        self.width, self.height = levels[0].width, levels[0].height
        self.bounds = (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)   # xmin ymin xmax ymax
        self._id = (os.path.abspath(self.path), size, os.stat(self.path).st_mtime_ns)
        self.tiles_read = 0

    # ----------------------------------------------------------------------------- geometry
    def level_res(self, i: int):
        L = self.levels[i]
        return self.res * self.width / L.width, self.res * self.height / L.height

    def level_for(self, merc_per_px: float) -> int:
        best = 0
        for i in range(len(self.levels)):
            if self.level_res(i)[0] <= merc_per_px * 1.0000001:
                best = i
        return best

    def pixel_at(self, x, y, level=0):
        rx, ry = self.level_res(level)
        return (np.asarray(x) - self.x0) / rx, (self.y0 - np.asarray(y)) / ry

    def merc_of_pixel_centre(self, col, row, level=0):
        rx, ry = self.level_res(level)
        return self.x0 + (np.asarray(col) + 0.5) * rx, self.y0 - (np.asarray(row) + 0.5) * ry

    # ----------------------------------------------------------------------------- tiles
    def tile(self, level: int, tx: int, ty: int) -> np.ndarray:
        key = (self._id, level, tx, ty)
        a = TILE_CACHE.get(key)
        if a is not None:
            return a
        a = self._decode(self.levels[level], tx, ty)
        TILE_CACHE.put(key, a)
        return a

    def _decode(self, L: _Level, tx: int, ty: int) -> np.ndarray:
        idx = ty * L.tx + tx
        off, cnt = int(L.offsets[idx]), int(L.counts[idx])
        n = L.tw * L.th
        if cnt == 0:                                           # sparse tile: no data at all
            a = np.empty((L.th, L.tw), L.dtype)
            a.fill(self.nodata if self.nodata is not None else 0)
            a.flags.writeable = False
            return a
        with self._flock:
            self._f.seek(off)
            raw = self._f.read(cnt)
        if len(raw) != cnt:
            raise CogError(f"short read in {self.path} (tile {tx},{ty})")
        buf = zlib.decompress(raw) if L.comp in (8, 32946) else raw
        if len(buf) != n * (L.bits // 8):
            raise CogError(f"tile {tx},{ty} of {self.path} decoded to {len(buf)} bytes, expected {n * L.bits // 8}")
        dt = np.dtype(L.dtype).newbyteorder(self.endian)
        a = np.frombuffer(buf, dtype=dt).reshape(L.th, L.tw)
        if dt.byteorder not in ("=", "|") and dt != np.dtype(L.dtype):
            a = a.astype(L.dtype)
        if L.pred == 2:
            a = np.cumsum(a, axis=1, dtype=L.dtype)           # horizontal differencing: modular sum (uint16 wraps like the encoder)
        else:
            a = a.copy()
        a.flags.writeable = False
        self.tiles_read += 1
        return a

    # ----------------------------------------------------------------------------- sampling
    def sample_merc(self, x: float, y: float, level: int = 0):
        L = self.levels[level]
        cf, rf = self.pixel_at(x, y, level)
        c, r = int(np.floor(cf)), int(np.floor(rf))
        if not (0 <= c < L.width and 0 <= r < L.height):
            return "outside", None
        raw = int(self.tile(level, c // L.tw, r // L.th)[r % L.th, c % L.tw])
        return ("nodata" if self.nodata is not None and raw == self.nodata else "value"), raw

    def sample_lnglat(self, lng: float, lat: float, level: int = 0):
        x, y = lnglat_to_merc(lng, lat)
        return self.sample_merc(float(x), float(y), level)

    def sample_many(self, xs, ys, level: int = 0):
        """Vectorised sampling at mercator points.  -> (raw uint16[n], status uint8[n]: 0 value, 1 nodata, 2 outside)."""
        L = self.levels[level]
        cf, rf = self.pixel_at(np.asarray(xs, np.float64), np.asarray(ys, np.float64), level)
        c, r = np.floor(cf).astype(np.int64), np.floor(rf).astype(np.int64)
        inside = (c >= 0) & (r >= 0) & (c < L.width) & (r < L.height)
        return self.sample_pixels(c, r, level, inside)

    def sample_pixels(self, c, r, level=0, inside=None):
        L = self.levels[level]
        c, r = np.asarray(c, np.int64), np.asarray(r, np.int64)
        if inside is None:
            inside = (c >= 0) & (r >= 0) & (c < L.width) & (r < L.height)
        raw = np.zeros(c.shape, np.uint16)
        status = np.full(c.shape, ST_OUTSIDE, np.uint8)
        ii = np.flatnonzero(inside)
        if ii.size:
            tid = (r[ii] // L.th) * L.tx + (c[ii] // L.tw)
            order = np.argsort(tid, kind="stable")
            ii, tid = ii[order], tid[order]
            cuts = np.flatnonzero(np.diff(tid)) + 1
            for grp in np.split(np.arange(ii.size), cuts):
                k = ii[grp]
                t = int(tid[grp[0]])
                tile = self.tile(level, t % L.tx, t // L.tx)
                raw[k] = tile[r[k] % L.th, c[k] % L.tw]
            status[ii] = ST_VALUE
            if self.nodata is not None:
                status[ii[raw[ii] == self.nodata]] = ST_NODATA
        return raw, status

    def read_window(self, level: int, c0: int, r0: int, c1: int, r1: int, fill=None) -> np.ndarray:
        """Pixels [c0, c1) x [r0, r1) of one level (clipped to the image; outside cells stay `fill`, default nodata)."""
        L = self.levels[level]
        fv = (self.nodata if self.nodata is not None else 0) if fill is None else fill
        out = np.full((max(r1 - r0, 0), max(c1 - c0, 0)), fv, L.dtype)
        a0, b0, a1, b1 = max(c0, 0), max(r0, 0), min(c1, L.width), min(r1, L.height)
        if a1 <= a0 or b1 <= b0:
            return out
        for ty in range(b0 // L.th, (b1 - 1) // L.th + 1):
            for tx in range(a0 // L.tw, (a1 - 1) // L.tw + 1):
                t = self.tile(level, tx, ty)
                ca, cb = max(a0, tx * L.tw), min(a1, (tx + 1) * L.tw)
                ra, rb = max(b0, ty * L.th), min(b1, (ty + 1) * L.th)
                out[ra - r0:rb - r0, ca - c0:cb - c0] = t[ra - ty * L.th:rb - ty * L.th, ca - tx * L.tw:cb - tx * L.tw]
        return out

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ------------------------------------------------------------------------------------------------ open
def _read_ifd(f, off, endian, big):
    cs, es, ns = (8, 20, 8) if big else (2, 12, 4)
    f.seek(off)
    head = f.read(cs)
    n = struct.unpack(endian + ("Q" if big else "H"), head)[0]
    if n > 4096:
        raise CogError("implausible IFD entry count")
    body = f.read(n * es + ns)
    if len(body) < n * es + ns:
        raise CogError("truncated IFD")
    tags = {}
    inl = 8 if big else 4
    for i in range(n):
        p = i * es
        tag, ftype = struct.unpack(endian + "HH", body[p:p + 4])
        count = struct.unpack(endian + ("Q" if big else "I"), body[p + 4:p + 4 + (8 if big else 4)])[0]
        sz = _TYPE_SIZE.get(ftype)
        if sz is None:
            continue
        vp = p + (12 if big else 8)
        total = sz * count
        if total <= inl:
            raw = body[vp:vp + total]
        else:
            o = struct.unpack(endian + ("Q" if big else "I"), body[vp:vp + inl])[0]
            f.seek(o)
            raw = f.read(total)
        if tag in (T_TILE_OFFSETS, T_TILE_COUNTS) and ftype in (3, 4, 16):
            tags[tag] = _array(raw, ftype, count, endian)
        else:
            tags[tag] = _values(raw, ftype, count, endian)
    nxt = struct.unpack(endian + ("Q" if big else "I"), body[n * es:n * es + ns])[0]
    return tags, nxt


def open_cog(path) -> Cog:
    path = os.fspath(path)
    f = open(path, "rb")
    try:
        size = os.fstat(f.fileno()).st_size
        head = f.read(16)
        if len(head) < 8 or head[:2] not in (b"II", b"MM"):
            raise CogError(f"{path} is not a TIFF file")
        endian = "<" if head[:2] == b"II" else ">"
        magic = struct.unpack(endian + "H", head[2:4])[0]
        if magic == 42:
            big, nxt = False, struct.unpack(endian + "I", head[4:8])[0]
        elif magic == 43:
            big, nxt = True, struct.unpack(endian + "Q", head[8:16])[0]
        else:
            raise CogError(f"{path}: bad TIFF magic {magic}")
        ifds = []
        while nxt and len(ifds) < 40:
            tags, nxt2 = _read_ifd(f, nxt, endian, big)
            ifds.append(tags)
            nxt = nxt2
        if not ifds:
            raise CogError(f"{path}: no image")
        levels = [_make_level(t, path) for t in ifds]
        levels.sort(key=lambda L: -L.width)
        for i, L in enumerate(levels):
            L.index = i
        t0 = next((t for t in ifds if not ((t.get(T_NEW_SUBFILE) or [0])[0] & 1)), ifds[0])
        if levels[0].width != t0[T_WIDTH][0]:
            raise CogError(f"{path}: the full-resolution image is not the first one")
        scale, tie, tr = t0.get(T_PIXEL_SCALE), t0.get(T_TIEPOINT), t0.get(T_TRANSFORM)
        if scale and tie and len(tie) >= 6:
            if abs(scale[0] - scale[1]) > 1e-9 * scale[0]:
                raise CogError("non-square pixels are not supported")
            res, x0, y0 = scale[0], tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1]
        elif tr and len(tr) >= 16:
            if tr[1] != 0 or tr[4] != 0 or abs(tr[0] + tr[5]) > 1e-9 * abs(tr[0]):
                raise CogError("rotated / non-square rasters are not supported")
            res, x0, y0 = tr[0], tr[3], tr[7]
        else:
            raise CogError(f"{path}: no georeferencing")
        epsg, rtype = None, 1
        gk = t0.get(T_GEOKEYS)
        if gk and len(gk) >= 4:
            for i in range(1, gk[3] + 1):
                kid, loc, v = gk[4 * i], gk[4 * i + 1], gk[4 * i + 3]
                if loc == 0:
                    if kid == 3072 or (kid == 2048 and epsg is None):
                        epsg = v
                    if kid == 1025:
                        rtype = v
        if epsg not in (3857, 3785, 900913):
            raise CogError(f"{path}: not EPSG:3857 (found {epsg})")
        if rtype == 2:                                          # pixel-is-point: shift half a pixel
            x0 -= res / 2
            y0 += res / 2
        nd = t0.get(T_NODATA)
        try:
            nodata = int(float(nd)) if nd is not None else None
        except (TypeError, ValueError):
            nodata = None
        return Cog(path, f, levels, {"nodata": nodata, "x0": x0, "y0": y0, "res": res}, size, endian)
    except Exception:
        f.close()
        raise


def _make_level(t, path) -> _Level:
    g = lambda k, d=None: (t[k] if k in t else d)               # noqa: E731
    if T_TILE_W not in t:
        raise CogError(f"{path}: the image is not tiled (not a COG)")
    if g(T_SAMPLES, [1])[0] != 1:
        raise CogError("only single-band images are supported")
    L = _Level()
    L.width, L.height = g(T_WIDTH)[0], g(T_HEIGHT)[0]
    L.bits = g(T_BITS, [1])[0]
    fmt = g(T_SAMPLE_FORMAT, [1])[0]
    L.comp, L.pred = g(T_COMPRESSION, [1])[0], g(T_PREDICTOR, [1])[0]
    if L.comp not in (1, 8, 32946):
        raise CogError(f"unsupported compression {L.comp} (only none and DEFLATE)")
    if L.pred not in (1, 2):
        raise CogError(f"unsupported predictor {L.pred}")
    L.dtype = {(1, 8): np.uint8, (1, 16): np.uint16, (1, 32): np.uint32, (2, 16): np.int16, (2, 32): np.int32,
               (3, 32): np.float32}.get((fmt, L.bits))
    if L.dtype is None:
        raise CogError(f"unsupported sample type ({L.bits} bits, format {fmt})")
    if L.pred == 2 and fmt == 3:
        raise CogError("predictor 2 on floating point data is not supported")
    L.tw, L.th = g(T_TILE_W)[0], g(T_TILE_H)[0]
    L.tx, L.ty = -(-L.width // L.tw), -(-L.height // L.th)
    L.offsets, L.counts = g(T_TILE_OFFSETS), g(T_TILE_COUNTS)
    if len(L.offsets) < L.tx * L.ty or len(L.counts) < L.tx * L.ty:
        raise CogError("tile offset table too short")
    L.index = 0
    return L


# ------------------------------------------------------------------------------------------------ depth values
def depth_fields(raw: int | None, status: str):
    """(status, cm, m, ft) of one raw sample: 'depth' | 'dry' | 'nodata'."""
    if status != "value" or raw is None:
        return "nodata", None, None, None
    return ("dry" if raw == 0 else "depth"), float(raw), raw / 100.0, raw / 100.0 * M2FT


def hour_files(meta: dict) -> dict:
    """{hour: file name}: hour N = the file whose save time (seconds in its name) is nearest N * 3600 s (names are not exact:
    depth_003604s.tif is hour 1, depth_000015s.tif is hour 0)."""
    import re
    best: dict = {}
    for name in (meta.get("files") or {}):
        m = re.fullmatch(r"depth_(\d+)s\.tif", name)
        if not m:
            continue
        sec = int(m.group(1))
        h = int(round(sec / 3600.0))
        d = abs(sec - h * 3600)
        if h not in best or d < best[h][0]:
            best[h] = (d, name)
    return {h: v[1] for h, v in sorted(best.items())}
