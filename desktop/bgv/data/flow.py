"""Flow-grid storms (FLOW_FORMAT.md) - pure CPU, no wgpu.  Same semantics as fluid/fluid-data.js.

    storm = Storm.open(data_root, "25yr")
    grid  = storm.load_hour(12)            # uint8[H, W, 4]  (R depth code, G u code, B v code, A valid)
    storm.prefetch([12, 13, 14])           # background gunzip into the small LRU
    d, u, v = decode_depth(R), decode_vel(G)
    storm.sample_t(12.4, X, Y)             # world-metre probe, bilinear in code space, blend after decode

Hours that are not in `meta.hours` (and hour 0, and anything > 24) are DRY: `load_hour` returns one shared read-only
all-zero grid and `is_zero(h)` is True.  A velocity code is only meaningful where R > 0: the all-zero grid has
G = B = 0 which would otherwise decode to -4 m/s, so every consumer reads velocity as 0 wherever R = 0.
"""
from __future__ import annotations

import gzip
import json
import struct
import threading
import zlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..geo import GridFrame

FORMAT = "bgflow"
VERSION = 1
MAX_HOUR = 24
MAX_GRID_SIDE = 8192
MAX_GRID_PIXELS = 36_000_000
DEPTH_MAX_M = 8.0
VMAX_MS = 4.0
ZERO_CODE = 128
QMAX_M2S = 2.0
WET_THRESHOLD_M = 0.02
LRU_HOURS = 4
STORM_ORDER = ["2yr", "5yr", "10yr", "25yr", "50yr", "100yr"]


class FlowFormatError(Exception):
    """A flow file or meta.json violates FLOW_FORMAT.md."""


class StormNotFoundError(FileNotFoundError):
    """No such storm folder (or no meta.json in it)."""


# --------------------------------------------------------------------------------------------------- decode laws

def decode_depth(R, depth_max: float = DEPTH_MAX_M):
    """depth_m = (R/255)^2 * depth_max (R: code or array of codes)."""
    t = np.asarray(R, dtype=np.float32) / 255.0
    out = t * t * np.float32(depth_max)
    return float(out) if out.ndim == 0 else out


def encode_depth(depth_m, depth_max: float = DEPTH_MAX_M):
    d = np.clip(np.asarray(depth_m, dtype=np.float64) / depth_max, 0.0, 1.0)
    return np.rint(255.0 * np.sqrt(d)).astype(np.uint8)


def decode_vel(code, vmax: float = VMAX_MS):
    """u/v in m/s from a G/B code: (code - 128) / 127 * vmax.  Caller must treat the code as 0 where R == 0."""
    out = (np.asarray(code, dtype=np.float32) - ZERO_CODE) / 127.0 * np.float32(vmax)
    return float(out) if out.ndim == 0 else out


def decode_q(G, qmax: float = QMAX_M2S):
    """flowpaths G -> mean unit discharge m2/s."""
    t = np.asarray(G, dtype=np.float32) / 255.0
    return t * t * np.float32(qmax)


def hour_pair(t: float, max_hour: int = MAX_HOUR):
    """Continuous time t (hours) -> (h0, h1, mix): the two integer hour snapshots to blend.  t is clamped to
    [0, max_hour]; at the upper end both are max_hour with mix 0."""
    t = min(max(float(t), 0.0), float(max_hour))
    h0 = int(np.floor(t))
    if h0 >= max_hour:
        return max_hour, max_hour, 0.0
    return h0, h0 + 1, t - h0


# -------------------------------------------------------------------------------------------------------- meta

def validate_meta(meta, where: str = "meta.json") -> dict:
    if not isinstance(meta, dict):
        raise FlowFormatError(f"{where}: expected a JSON object")
    if meta.get("format") != FORMAT:
        raise FlowFormatError(f"{where}: format is {meta.get('format')!r}, expected {FORMAT!r}")
    if meta.get("version") != VERSION:
        raise FlowFormatError(f"{where}: unsupported version {meta.get('version')!r}")
    w, h = meta.get("width"), meta.get("height")
    for v in (w, h):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise FlowFormatError(f"{where}: width/height must be positive integers (got {w!r} x {h!r})")
    if w > MAX_GRID_SIDE or h > MAX_GRID_SIDE or w * h > MAX_GRID_PIXELS:
        raise FlowFormatError(f"{where}: grid {w} x {h} exceeds the supported size")
    for k in ("cell_m", "cell_ground_m"):
        v = meta.get(k)
        if not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            raise FlowFormatError(f"{where}: {k} must be a positive number (got {v!r})")
    b = meta.get("bounds_merc")
    if (not isinstance(b, list) or len(b) != 4
            or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in b)):
        raise FlowFormatError(f"{where}: bounds_merc must be [xmin, ymin, xmax, ymax]")
    if not (b[0] < b[2] and b[1] < b[3]):
        raise FlowFormatError(f"{where}: bounds_merc is empty or inverted")
    cm = meta["cell_m"]
    if round((b[2] - b[0]) / cm) != w or round((b[3] - b[1]) / cm) != h:
        raise FlowFormatError(f"{where}: bounds_merc does not span width x height cells of cell_m")
    hrs = meta.get("hours")
    if not isinstance(hrs, list) or not all(isinstance(x, int) and 1 <= x <= MAX_HOUR for x in hrs):
        raise FlowFormatError(f"{where}: hours must be a list of integers 1..24")
    de = meta.get("depth_encoding")
    if de is not None and (not isinstance(de, dict) or de.get("type") != "sqrt" or not (de.get("max_m", 0) > 0)):
        raise FlowFormatError(f"{where}: depth_encoding must be sqrt with max_m > 0")
    ve = meta.get("vel_encoding")
    if ve is not None and (not isinstance(ve, dict) or not (ve.get("vmax_ms", 0) > 0)
                           or ve.get("zero_code", ZERO_CODE) != ZERO_CODE):
        raise FlowFormatError(f"{where}: vel_encoding needs vmax_ms > 0 and zero_code 128")
    return meta


def available_storms(root, subdir: str = "flow") -> list[str]:
    """Storm folder names under <root>/layers/<subdir> (city: flow, county: flow_county) that have a meta.json, ordered
    2yr..100yr, then the others by name (ev1..ev6)."""
    base = Path(root) / "layers" / subdir
    try:
        if not base.is_dir():
            return []
        found = [p.name for p in base.iterdir() if p.is_dir() and (p / "meta.json").is_file()]
    except OSError:
        return []
    key = lambda s: (STORM_ORDER.index(s) if s in STORM_ORDER else len(STORM_ORDER), _natural(s))  # noqa: E731
    return sorted(found, key=key)


def _natural(s: str):
    """'ev10' after 'ev2'."""
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def storm_label(root, name: str, subdir: str = "flow") -> str:
    """The `label` of a storm's meta.json ('100-year storm (county)'), else the storm folder name."""
    try:
        meta = json.loads((Path(root) / "layers" / subdir / name / "meta.json").read_text(encoding="utf-8"))
        lab = meta.get("label")
        if isinstance(lab, str) and lab.strip():
            return lab.strip()
    except (OSError, ValueError):
        pass
    return name


# ------------------------------------------------------------------------------------------------ gzip loading

def gunzip_exact(path, expected: int) -> bytes:
    """Gunzip a file that must decode to exactly `expected` bytes.  The gzip trailer ISIZE is checked first, so a
    truncated / wrong-size file fails before any big allocation.  Releases the GIL (zlib), so threads scale."""
    raw = Path(path).read_bytes()
    if len(raw) < 18 or raw[:2] != b"\x1f\x8b":
        raise FlowFormatError(f"{path}: not a gzip file")
    isize = struct.unpack("<I", raw[-4:])[0]
    if isize != (expected & 0xFFFFFFFF):
        raise FlowFormatError(f"{path}: gzip says {isize} bytes, expected {expected}")
    try:
        d = zlib.decompressobj(wbits=31)
        out = d.decompress(raw, expected + 1)
        if not d.eof and len(out) <= expected:
            out += d.flush()
    except zlib.error as e:
        raise FlowFormatError(f"{path}: corrupt gzip data ({e})") from None
    if len(out) != expected or d.unconsumed_tail:
        raise FlowFormatError(f"{path}: decoded {len(out)} bytes, expected {expected}")
    return out


def gunzip_into(path, out: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """`gunzip_exact` into a caller-owned uint8 array of exactly the expected size (no 40 MB allocation: the output is
    produced in `chunk`-sized pieces, so a streaming loader reuses the same few buffers and never page-faults a fresh
    40 MB block while the render thread runs).  Same checks and errors as `gunzip_exact`.  Returns `out`."""
    expected = int(out.nbytes)
    raw = Path(path).read_bytes()
    if len(raw) < 18 or raw[:2] != b"\x1f\x8b":
        raise FlowFormatError(f"{path}: not a gzip file")
    isize = struct.unpack("<I", raw[-4:])[0]
    if isize != (expected & 0xFFFFFFFF):
        raise FlowFormatError(f"{path}: gzip says {isize} bytes, expected {expected}")
    flat = out.reshape(-1)
    d = zlib.decompressobj(wbits=31)
    pos = 0
    data = raw
    try:
        while True:
            piece = d.decompress(data, chunk)
            n = len(piece)
            if n:
                if pos + n > expected:
                    raise FlowFormatError(f"{path}: decoded more than the expected {expected} bytes")
                flat[pos:pos + n] = np.frombuffer(piece, np.uint8)
                pos += n
            data = d.unconsumed_tail
            if d.eof or (not data and not n):
                break
        if not d.eof:
            tail = d.flush()
            if tail:
                if pos + len(tail) > expected:
                    raise FlowFormatError(f"{path}: decoded more than the expected {expected} bytes")
                flat[pos:pos + len(tail)] = np.frombuffer(tail, np.uint8)
                pos += len(tail)
    except zlib.error as e:
        raise FlowFormatError(f"{path}: corrupt gzip data ({e})") from None
    if pos != expected or d.unconsumed_tail:
        raise FlowFormatError(f"{path}: decoded {pos} bytes, expected {expected}")
    return out


def block_max(arr2d: np.ndarray, block: int) -> np.ndarray:
    """Max over block x block tiles of a 2-D array (edge tiles clipped).  Result shape ceil(H/b) x ceil(W/b).
    Works on strided views (e.g. rgba[..., 0]) without copying (np.maximum.reduceat releases the GIL)."""
    H, W = arr2d.shape
    cols = np.arange(0, W, block)
    rows = np.arange(0, H, block)
    return np.maximum.reduceat(np.maximum.reduceat(arr2d, cols, axis=1), rows, axis=0)


# ---------------------------------------------------------------------------------------------------- Storm

class Storm:
    """One storm folder.  Thread safe for load_hour / prefetch / sample."""

    def __init__(self, root, name: str, meta: dict, subdir: str = "flow"):
        self.root = Path(root)
        self.name = name
        self.subdir = subdir
        self.dir = self.root / "layers" / subdir / name
        self.meta = meta
        self.frame = GridFrame.from_meta(meta)
        self.width, self.height = self.frame.width, self.frame.height
        self.hours = sorted(set(int(h) for h in meta["hours"]))
        self._hour_set = frozenset(self.hours)
        de, ve = meta.get("depth_encoding") or {}, meta.get("vel_encoding") or {}
        self.depth_max = float(de.get("max_m", DEPTH_MAX_M))
        self.vmax = float(ve.get("vmax_ms", VMAX_MS))
        self.qmax = float(meta.get("qmax_m2s", QMAX_M2S))
        self.wet_threshold = float(meta.get("wet_threshold_m", WET_THRESHOLD_M))
        self.source = str(meta.get("source", "rasters"))
        self.label = str(meta.get("label", name))
        self.synthetic = bool(meta.get("synthetic", False))          # demo data (bgv.data.demo): the UI says so
        self.nbytes = self.width * self.height * 4
        self._zero = None
        self._lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._futures: dict[int, Future] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._fp = None

    # ------------------------------------------------------------------ open
    @staticmethod
    def open(root, storm: str, subdir: str = "flow") -> "Storm":
        p = Path(root) / "layers" / subdir / storm / "meta.json"
        if not p.is_file():
            raise StormNotFoundError(f"storm {storm!r} not found: {p} does not exist")
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise FlowFormatError(f"{p}: not valid JSON ({e})") from None
        validate_meta(meta, str(p))
        return Storm(root, storm, meta, subdir)

    # ------------------------------------------------------------------ hours
    def has_hour(self, h: int) -> bool:
        return int(h) in self._hour_set

    def is_zero(self, h: int) -> bool:
        """True when hour h is dry (not in meta.hours): load_hour returns the shared zero grid."""
        return int(h) not in self._hour_set

    @property
    def zero_grid(self) -> np.ndarray:
        if self._zero is None:
            z = np.zeros((self.height, self.width, 4), np.uint8)
            z.flags.writeable = False
            self._zero = z
        return self._zero

    def hour_path(self, h: int) -> Path:
        return self.dir / f"h{int(h):02d}.rgba.gz"

    def _read_grid(self, path: Path) -> np.ndarray:
        buf = gunzip_exact(path, self.nbytes)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 4)
        return arr

    def read_hour_into(self, h: int, out: np.ndarray) -> np.ndarray:
        """Decode hour h into `out` (uint8[H, W, 4], caller owned) without caching; dry hours fill it with zeros."""
        h = int(h)
        if h not in self._hour_set:
            out[...] = 0
            return out
        gunzip_into(self.hour_path(h), out)
        return out

    def load_hour(self, h: int, cache: bool = True) -> np.ndarray:
        """uint8[H, W, 4], read-only.  Dry hours share one zero grid (see is_zero)."""
        h = int(h)
        if h not in self._hour_set:
            return self.zero_grid
        with self._lock:
            hit = self._lru.get(h)
            if hit is not None:
                self._lru.move_to_end(h)
                return hit
            fut = self._futures.get(h)
        if fut is not None:
            return fut.result()
        arr = self._read_grid(self.hour_path(h))
        if cache:
            self._remember(h, arr)
        return arr

    def _remember(self, h: int, arr: np.ndarray):
        with self._lock:
            self._lru[h] = arr
            self._lru.move_to_end(h)
            while len(self._lru) > LRU_HOURS:
                self._lru.popitem(last=False)

    def cached_hours(self) -> list[int]:
        with self._lock:
            return list(self._lru.keys())

    def prefetch(self, hours, workers: int = 2) -> list[Future]:
        """Decode up to LRU_HOURS hours in a background pool.  Returns one Future per hour that exists."""
        want = []
        for h in hours:
            h = int(h)
            if h in self._hour_set and h not in want:
                want.append(h)
            if len(want) >= LRU_HOURS:
                break
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bgv-prefetch")
        out = []
        for h in want:
            with self._lock:
                if h in self._lru:
                    self._lru.move_to_end(h)
                    f = Future()
                    f.set_result(self._lru[h])
                    out.append(f)
                    continue
                f = self._futures.get(h)
                if f is None:
                    f = self._pool.submit(self._prefetch_job, h)
                    self._futures[h] = f
            out.append(f)
        return out

    def _prefetch_job(self, h: int):
        try:
            arr = self._read_grid(self.hour_path(h))
            self._remember(h, arr)
            return arr
        finally:
            with self._lock:
                self._futures.pop(h, None)

    def load_flowpaths(self) -> np.ndarray:
        """uint8[H, W, 4]: R = max depth over all hours, G = mean unit discharge code, B = hours wet / 24, A valid."""
        if self._fp is None:
            self._fp = self._read_grid(self.dir / "flowpaths.rgba.gz")
        return self._fp

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        with self._lock:
            self._lru.clear()
            self._futures.clear()
        self._fp = None

    # ------------------------------------------------------------------ decode helpers
    def decode_depth(self, R):
        return decode_depth(R, self.depth_max)

    def decode_vel(self, code):
        return decode_vel(code, self.vmax)

    # ------------------------------------------------------------------ CPU probe
    def sample(self, rgba_a, rgba_b, mix: float, x_merc: float, y_merc: float):
        """JS `sample`: bilinear in code space at a Mercator position, each hour decoded, then blended
        value = (1-mix)*A + mix*B.  Either grid may be None (dry).  Returns dict(depth,u,v,speed,valid); `valid` is
        False outside the grid and where the interpolated A < 0.5 (outside the model domain)."""
        return sample_grids(rgba_a, rgba_b, mix, x_merc, y_merc, self.meta)

    def sample_world(self, rgba_a, rgba_b, mix: float, X: float, Y: float):
        x, y = self.frame.world_to_merc(X, Y)
        return self.sample(rgba_a, rgba_b, mix, x, y)

    def sample_t(self, t: float, X: float, Y: float):
        """Probe at world metres X,Y at continuous hour t (loads the two hours if needed)."""
        h0, h1, mix = hour_pair(t)
        a = self.load_hour(h0)
        b = self.load_hour(h1) if mix > 0 else None
        return self.sample_world(a, b, mix, X, Y)


def sample_grids(rgba_a, rgba_b, mix: float, x_merc: float, y_merc: float, meta: dict) -> dict:
    out = {"depth": 0.0, "u": 0.0, "v": 0.0, "speed": 0.0, "valid": False}
    W, H, cell = int(meta["width"]), int(meta["height"]), float(meta["cell_m"])
    b = meta["bounds_merc"]
    gx = (x_merc - b[0]) / cell
    gy = (b[3] - y_merc) / cell
    if not (0.0 <= gx <= W and 0.0 <= gy <= H):
        return out
    fx, fy = gx - 0.5, gy - 0.5
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    tx, ty = fx - x0, fy - y0
    xa, xb = max(x0, 0), min(x0 + 1, W - 1)
    ya, yb = max(y0, 0), min(y0 + 1, H - 1)
    w00, w10, w01, w11 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty
    de, ve = meta.get("depth_encoding") or {}, meta.get("vel_encoding") or {}
    dmax, vmax = float(de.get("max_m", DEPTH_MAX_M)), float(ve.get("vmax_ms", VMAX_MS))
    m = 0.0 if not mix > 0 else (1.0 if mix >= 1 else float(mix))

    def one(arr):
        if arr is None:
            return 0.0, 0.0, 0.0, 0.0
        c = (arr[ya, xa].astype(np.float64) * w00 + arr[ya, xb].astype(np.float64) * w10
             + arr[yb, xa].astype(np.float64) * w01 + arr[yb, xb].astype(np.float64) * w11)
        t = c[0] / 255.0
        d = t * t * dmax
        u = (c[1] - ZERO_CODE) / 127.0 * vmax if c[0] > 0 else 0.0
        v = (c[2] - ZERO_CODE) / 127.0 * vmax if c[0] > 0 else 0.0
        return d, u, v, c[3]

    dA, uA, vA, aA = one(rgba_a)
    dB, uB, vB, aB = one(rgba_b)
    if max(aA, aB) < 127.5:
        return out
    out["valid"] = True
    out["depth"] = dA * (1 - m) + dB * m
    out["u"] = uA * (1 - m) + uB * m
    out["v"] = vA * (1 - m) + vB * m
    out["speed"] = float(np.hypot(out["u"], out["v"]))
    return out
