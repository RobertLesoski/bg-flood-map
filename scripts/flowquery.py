r"""
flowquery.py - ask the flow grids (FLOW_FORMAT.md) "how deep is the water at this building / point / road?".

Shared by build_address_index.py and build_infrastructure.py.  The rules are the ones of the apps
(desktop/bgv/data/stats.py, fluid/fluid-stats.js) so the numbers agree:

  * depth code R -> depth_m = (R/255)^2 * 8;  pixels with A == 0 (outside the model domain) are ignored
  * a building owns every grid pixel its footprint touches (ALL-TOUCHED); its depth = MAX over those pixels
  * flooded = depth >= 0.1 m;  tiers 0 below 0.1 | 1 shallow [0.1,0.3) | 2 moderate [0.3,0.6) | 3 deep [0.6,1.2) | 4 severe >= 1.2
  * roads: a line of ground length L gets n = ceil(L/10 m) samples at the midpoints of n equal parts (L/n metres each);
    a sample is WET when its pixel depth >= 0.02 m (and "flooded" at >= 0.1 m)

numpy + shapely (2.x) + json only; rasterio is NOT needed here.

API
  load_storms(flow_dir, names=None, log=print) -> list[Storm]      complete storms only (meta + every hour file + sane gzip size)
  Storm.read_hour(h) -> (R, A) uint8[H*W];  Storm.read_max() -> R uint8[H*W] (flowpaths, else max over hours)
  building_pixels(geoms_lnglat, grid) -> PixelIndex                 all-touched CSR index (shapely, vectorised)
  point_pixels(lng, lat, grid) -> int64[n]                          pixel of each point, -1 outside the grid
  max_code(R, A, index) -> uint8[n]                                 max code over the valid pixels of every item
  road_samples(lines_lnglat, grid, spacing_m=10) -> RoadSamples
  road_stats(R, A, samples) -> (wet_pct, flooded_pct, max_code)
  code_depth(code) -> metres;  tier_of(depth_m) -> 0..4
"""
from __future__ import annotations

import gzip
import json
import math
import os
import struct
from dataclasses import dataclass, field

import numpy as np
import shapely

MERC_R = 6378137.0
DEPTH_MAX_M = 8.0
WET_MIN_M = 0.02
FLOOD_MIN_M = 0.1
TIER_EDGES_M = (0.1, 0.3, 0.6, 1.2)
TIER_NAMES = ("shallow", "moderate", "deep", "severe")
DEPTH_TABLE = (np.arange(256, dtype=np.float64) / 255.0) ** 2 * DEPTH_MAX_M      # metres for every R code
_EDGES = np.array(TIER_EDGES_M)


def min_code_for(depth_m):
    """Smallest code >= 1 whose decoded depth is >= depth_m (256 when none) - same as stats.min_code_for."""
    ok = np.flatnonzero(DEPTH_TABLE[1:] >= depth_m)
    return int(ok[0]) + 1 if ok.size else 256


WET_CODE = min_code_for(WET_MIN_M)         # 13
FLOOD_CODE = min_code_for(FLOOD_MIN_M)     # 29


def code_depth(code):
    return DEPTH_TABLE[np.asarray(code, dtype=np.int64)]


def tier_of(depth_m):
    """0 below 0.1 m, 1..4 = shallow, moderate, deep, severe (works on arrays)."""
    return np.searchsorted(_EDGES, np.asarray(depth_m, dtype=np.float64), side="right").astype(np.uint8)


def lnglat_to_merc(lng, lat):
    lng = np.asarray(lng, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    return MERC_R * np.radians(lng), MERC_R * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))


def merc_to_lnglat(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.degrees(x / MERC_R), np.degrees(2.0 * np.arctan(np.exp(y / MERC_R)) - np.pi / 2.0)


# ------------------------------------------------------------------------------------------------ grids and storms
@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    cell_m: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @staticmethod
    def from_meta(meta):
        b = meta["bounds_merc"]
        return Grid(int(meta["width"]), int(meta["height"]), float(meta["cell_m"]), float(b[0]), float(b[1]),
                    float(b[2]), float(b[3]))

    def key(self):
        return (self.width, self.height, self.cell_m, self.xmin, self.ymin, self.xmax, self.ymax)

    def to_grid_xy(self, lng, lat):
        """lng/lat -> (gx, gy) grid coordinates (pixel (col,row) = [col,col+1) x [row,row+1), row 0 = north)."""
        x, y = lnglat_to_merc(lng, lat)
        return (x - self.xmin) / self.cell_m, (self.ymax - y) / self.cell_m

    def lnglat_bounds(self):
        w, s = merc_to_lnglat(self.xmin, self.ymin)
        e, n = merc_to_lnglat(self.xmax, self.ymax)
        return float(w), float(s), float(e), float(n)


@dataclass
class Storm:
    name: str
    dir: str
    meta: dict
    grid: Grid
    hours: list
    problems: list = field(default_factory=list)

    @property
    def source(self):
        return self.meta.get("source", "")

    @property
    def label(self):
        return self.meta.get("label", self.name)

    def hour_path(self, h):
        return os.path.join(self.dir, "h%02d.rgba.gz" % h)

    def _read(self, path):
        n = self.grid.width * self.grid.height * 4
        with gzip.open(path, "rb") as f:
            buf = f.read()
        if len(buf) != n:
            raise ValueError("%s: decompressed %d bytes, expected %d" % (path, len(buf), n))
        a = np.frombuffer(buf, np.uint8).reshape(-1, 4)
        return a[:, 0].copy(), a[:, 3].copy()

    def read_hour(self, h):
        """(R, A) flat uint8 arrays of hour h; a missing hour is dry (R = 0) with A of the first existing hour."""
        return self._read(self.hour_path(h))

    def read_max(self):
        """Maximum-depth code per pixel over all hours: flowpaths.rgba.gz R when present, else the max of the hour files."""
        fp = os.path.join(self.dir, "flowpaths.rgba.gz")
        if os.path.isfile(fp):
            return self._read(fp)[0]
        r = None
        for h in self.hours:
            rr = self.read_hour(h)[0]
            r = rr if r is None else np.maximum(r, rr)
        return r if r is not None else np.zeros(self.grid.width * self.grid.height, np.uint8)

    def read_max_valid(self):
        """(R max-depth codes, A valid flags) from flowpaths.rgba.gz, else from the hour files."""
        fp = os.path.join(self.dir, "flowpaths.rgba.gz")
        if os.path.isfile(fp):
            return self._read(fp)
        return self.read_max(), self.read_valid()

    def read_valid(self):
        """A channel (255 inside the model domain) - identical in every file of a storm."""
        fp = os.path.join(self.dir, "flowpaths.rgba.gz")
        if os.path.isfile(fp):
            return self._read(fp)[1]
        return self.read_hour(self.hours[0])[1]


def _gz_isize(path):
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return struct.unpack("<I", f.read(4))[0]


def check_storm_dir(path):
    """Light completeness test (not the full flowgrid.validate): meta.json parses, every listed hour file exists and its gzip
    trailer says width*height*4 bytes (catches half-downloaded files). Returns (meta or None, problems)."""
    bad = []
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        w, h = int(meta["width"]), int(meta["height"])
        hours = [int(x) for x in meta["hours"]]
        Grid.from_meta(meta)
    except Exception as e:                                        # noqa: BLE001
        return None, ["meta.json: %s" % e]
    want = w * h * 4
    for hh in hours:
        p = os.path.join(path, "h%02d.rgba.gz" % hh)
        try:
            if _gz_isize(p) != want:
                bad.append("%s: gzip size does not match width*height*4" % os.path.basename(p))
        except OSError:
            bad.append("%s: missing or unreadable" % os.path.basename(p))
    return meta, bad


def load_storms(flow_dir, names=None, log=print):
    """All complete storm folders of `flow_dir` (sorted: design storms by return period, then the rest)."""
    out = []
    if not os.path.isdir(flow_dir):
        return out
    cand = sorted(d for d in os.listdir(flow_dir) if os.path.isdir(os.path.join(flow_dir, d)) and not d.startswith("."))
    if names:
        cand = [d for d in cand if d in set(names)]
    for d in cand:
        meta, bad = check_storm_dir(os.path.join(flow_dir, d))
        if meta is None or bad:
            if log:
                log("  skip storm %s: %s" % (d, "; ".join((bad or ["no meta"])[:2])))
            continue
        out.append(Storm(d, os.path.join(flow_dir, d), meta, Grid.from_meta(meta), sorted(int(x) for x in meta["hours"])))
    out.sort(key=storm_sort_key)
    return out


def storm_sort_key(s):
    n = s.name if isinstance(s, Storm) else str(s)
    if n.endswith("yr") and n[:-2].isdigit():
        return (0, int(n[:-2]), n)
    if n.startswith("ev") and n[2:].isdigit():
        return (1, int(n[2:]), n)
    return (2, 0, n)


# ------------------------------------------------------------------------------------------------ pixel indices
@dataclass
class PixelIndex:
    offsets: np.ndarray       # int64[n+1]
    pixels: np.ndarray        # int64[nnz]   row*W + col

    @property
    def n(self):
        return int(self.offsets.shape[0] - 1)


def _to_grid_geoms(geoms, grid):
    def f(c):
        gx, gy = grid.to_grid_xy(c[:, 0], c[:, 1])
        return np.stack([gx, gy], axis=1)
    return shapely.transform(np.asarray(geoms, dtype=object), f)


def building_pixels(geoms, grid):
    """ALL-TOUCHED pixel index of lng/lat polygons on `grid`: item i owns every pixel its footprint intersects
    (GDAL all_touched semantics: a pixel whose area meets the polygon). Pixels outside the grid are dropped."""
    n = len(geoms)
    offsets = np.zeros(n + 1, dtype=np.int64)
    if n == 0:
        return PixelIndex(offsets, np.zeros(0, np.int64))
    gg = _to_grid_geoms(geoms, grid)
    empty = shapely.is_empty(gg)
    b = np.where(empty[:, None], np.nan, shapely.bounds(gg))
    ok = np.isfinite(b).all(axis=1)
    c0 = np.full(n, 0, np.int64); c1 = np.full(n, -1, np.int64); r0 = np.full(n, 0, np.int64); r1 = np.full(n, -1, np.int64)
    c0[ok] = np.maximum(np.floor(b[ok, 0]), 0).astype(np.int64)
    c1[ok] = np.minimum(np.floor(b[ok, 2]), grid.width - 1).astype(np.int64)
    r0[ok] = np.maximum(np.floor(b[ok, 1]), 0).astype(np.int64)
    r1[ok] = np.minimum(np.floor(b[ok, 3]), grid.height - 1).astype(np.int64)
    nc = np.maximum(c1 - c0 + 1, 0)
    nr = np.maximum(r1 - r0 + 1, 0)
    cnt = nc * nr
    total = int(cnt.sum())
    if total == 0:
        return PixelIndex(offsets, np.zeros(0, np.int64))
    owner = np.repeat(np.arange(n), cnt)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    k = np.arange(total, dtype=np.int64) - np.repeat(start, cnt)
    ncr = np.repeat(np.maximum(nc, 1), cnt)
    col = np.repeat(c0, cnt) + k % ncr
    row = np.repeat(r0, cnt) + k // ncr
    shapely.prepare(gg)
    hit = np.zeros(total, dtype=bool)
    step = 2_000_000
    for s in range(0, total, step):
        e = min(total, s + step)
        boxes = shapely.box(col[s:e], row[s:e], col[s:e] + 1.0, row[s:e] + 1.0)
        hit[s:e] = shapely.intersects(gg[owner[s:e]], boxes)
    owner, col, row = owner[hit], col[hit], row[hit]
    offsets[1:] = np.cumsum(np.bincount(owner, minlength=n))
    return PixelIndex(offsets, row * grid.width + col)       # owner is already ascending, pixels ascending within an owner


def point_pixels(lng, lat, grid):
    gx, gy = grid.to_grid_xy(lng, lat)
    c = np.floor(gx).astype(np.int64)
    r = np.floor(gy).astype(np.int64)
    inside = (c >= 0) & (c < grid.width) & (r >= 0) & (r < grid.height)
    return np.where(inside, r * grid.width + c, -1)


def index_from_points(lng, lat, grid):
    """PixelIndex with one pixel per point (none when the point is outside the grid)."""
    px = point_pixels(lng, lat, grid)
    ok = px >= 0
    offsets = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    return PixelIndex(offsets, px[ok])


def max_code(R, A, index):
    """Max R code over the pixels of every item, ignoring pixels with A == 0 (-> uint8[n], 0 = dry / outside)."""
    n = index.n
    out = np.zeros(n, np.uint8)
    if index.pixels.size == 0:
        return out
    v = np.where(A[index.pixels] != 0, R[index.pixels], 0).astype(np.uint8)
    cnt = np.diff(index.offsets)
    nz = np.flatnonzero(cnt > 0)
    out[nz] = np.maximum.reduceat(v, index.offsets[:-1][nz])
    return out


# ------------------------------------------------------------------------------------------------ roads
@dataclass
class RoadSamples:
    offsets: np.ndarray       # int64[n+1] samples of road i = [offsets[i], offsets[i+1])
    pixel: np.ndarray         # int64[S]   -1 outside the grid
    weight_m: np.ndarray      # float64[S]
    length_m: np.ndarray      # float64[n] ground length of every road


def line_parts(geom):
    t = geom.get("type")
    c = geom.get("coordinates")
    if t == "LineString":
        return [c]
    if t == "MultiLineString":
        return list(c)
    return []


def road_samples(lines, grid, spacing_m=10.0):
    """`lines`: list of coordinate lists [[lng,lat],...] (one LineString each) or lists of such parts (MultiLineString).
    Sampling is the app rule: n = ceil(L/spacing) midpoints of n equal parts, each worth L/n metres.
    Ground length of a segment = mercator length * cos(latitude of the segment)."""
    pix, wts, counts, lens = [], [], [], []
    for ln in lines:
        parts = ln if (ln and isinstance(ln[0][0], (list, tuple))) else [ln]
        tot = 0.0
        cnt = 0
        for part in parts:
            a = np.asarray(part, dtype=np.float64)
            if a.shape[0] < 2:
                continue
            x, y = lnglat_to_merc(a[:, 0], a[:, 1])
            coslat = np.cos(np.radians(0.5 * (a[1:, 1] + a[:-1, 1])))
            seg = np.hypot(np.diff(x), np.diff(y)) * coslat
            L = float(seg.sum())
            tot += L
            if L <= 0:
                continue
            n = max(1, int(math.ceil(L / spacing_m)))
            s = (np.arange(n) + 0.5) * (L / n)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, seg.size - 1)
            t = np.where(seg[i] > 0, (s - cum[i]) / np.where(seg[i] > 0, seg[i], 1.0), 0.0)
            lng, lat = merc_to_lnglat(x[i] + t * (x[i + 1] - x[i]), y[i] + t * (y[i + 1] - y[i]))
            pix.append(point_pixels(lng, lat, grid))
            wts.append(np.full(n, L / n))
            cnt += n
        lens.append(tot)
        counts.append(cnt)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return RoadSamples(offsets, np.concatenate(pix) if pix else np.zeros(0, np.int64),
                       np.concatenate(wts) if wts else np.zeros(0), np.array(lens))


def road_stats(R, A, rs):
    """Per road: (% of length wet >= 0.02 m, % of length flooded >= 0.1 m, max code over its in-grid samples)."""
    n = rs.offsets.shape[0] - 1
    wet = np.zeros(n)
    fl = np.zeros(n)
    mx = np.zeros(n, np.uint8)
    S = rs.pixel.size
    if S == 0:
        return wet, fl, mx
    inside = rs.pixel >= 0
    px = np.where(inside, rs.pixel, 0)
    code = np.where(inside & (A[px] != 0), R[px], 0).astype(np.uint8)
    owner = np.repeat(np.arange(n), np.diff(rs.offsets))
    w = rs.weight_m
    wet = np.bincount(owner, weights=w * (code >= WET_CODE), minlength=n)
    fl = np.bincount(owner, weights=w * (code >= FLOOD_CODE), minlength=n)
    cnt = np.diff(rs.offsets)
    nz = np.flatnonzero(cnt > 0)
    mx[nz] = np.maximum.reduceat(code, rs.offsets[:-1][nz])
    L = np.maximum(rs.length_m, 1e-9)
    return np.minimum(100.0 * wet / L, 100.0), np.minimum(100.0 * fl / L, 100.0), mx
