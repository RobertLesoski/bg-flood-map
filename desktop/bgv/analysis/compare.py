"""Storm compare: depth grids of two (storm, hour) states on the flow frame, their difference, statistics and the sign-aware palette.

    prov = DepthGridProvider(depth_source, frame)
    a = prov.grid("25yr", 12)  ;  b = prov.grid("100yr", 12)        float32[H, W] metres on the FLOW grid (0 = dry / outside)
    st = compare_grids(a, b, frame)                                  CompareStats (areas, volume, max added depth ...)
    rb = building_flood(source, "25yr", 12.0, scene) ...             flooded-building / wet-road numbers of one state
    delta_pack(a, b) -> float16[H, W, 2]                             (B - A, max(A, B)) for the GPU difference layer

Grids: tiles (county) are read through the file's overviews at the factor that matches the flow cell (6.25 m x 4 = 25 m), averaged,
then bilinearly resampled onto the flow grid; flow-grid storms are decoded from their R channel.  Either way the picture and the
area / volume numbers are at flow-grid resolution; the building and road numbers sample the same depth adapter at the footprints /
road samples (full tile resolution) so they match the timeline.

Palette (sign aware, colour-blind safe orange / blue; the WGSL of the layer is generated from these constants):
    B deeper than A  (added depth)    pale yellow -> orange -> red -> dark maroon      delta > 0
    B shallower than A (less depth)   pale cyan -> blue -> deep blue -> navy            delta < 0
    |delta| < `FLAT_M`                neutral grey (wet in both, depth unchanged within 2 cm)
The ramp is symmetric: +-`range_m` (default 2 m) is the darkest end of both sides.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..data import flow as flowdata
from . import geodesy

WET_M = 0.02
FLOOD_M = 0.1
FLAT_M = 0.02

# sRGB stops at |delta| / range = 0, 0.12, 0.4, 0.75, 1.0
STOPS_X = (0.0, 0.12, 0.4, 0.75, 1.0)
POS_RGB = ((0.99, 0.91, 0.62), (0.98, 0.72, 0.28), (0.93, 0.42, 0.10), (0.72, 0.12, 0.10), (0.36, 0.02, 0.12))
NEG_RGB = ((0.72, 0.92, 0.96), (0.46, 0.76, 0.92), (0.20, 0.50, 0.85), (0.08, 0.27, 0.62), (0.03, 0.10, 0.34))
FLAT_RGB = (0.66, 0.68, 0.74)


def diverging_srgb(delta: float, range_m: float = 2.0) -> tuple:
    """Colour (r, g, b) in sRGB 0..1 of a depth difference (metres); the legend and the tests use this, the shader the same stops."""
    if abs(delta) < FLAT_M:
        return FLAT_RGB
    x = min(abs(delta) / max(range_m, 1e-6), 1.0)
    stops = POS_RGB if delta > 0 else NEG_RGB
    c = _lin(stops[0])                                    # same sequential smoothstep mix, in linear light, as the shader
    for i in range(1, len(STOPS_X)):
        t = min(max((x - STOPS_X[i - 1]) / (STOPS_X[i] - STOPS_X[i - 1]), 0.0), 1.0)
        t = t * t * (3.0 - 2.0 * t)
        s = _lin(stops[i])
        c = tuple(c[k] + (s[k] - c[k]) * t for k in range(3))
    return tuple(_enc(v) for v in c)


def _lin(c):
    return tuple((v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in c)


def _enc(v):
    v = min(max(v, 0.0), 1.0)
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def wgsl_flat() -> str:
    """WGSL literal of the neutral grey (linear)."""
    return "vec3<f32>(%.5f, %.5f, %.5f)" % _lin(FLAT_RGB)


def wgsl_ramp() -> str:
    """WGSL `fn delta_color(x: f32, pos: bool) -> vec3<f32>` (x = |delta| / range in 0..1, linear colour) built from the stops."""
    def side(var, stops):
        lin = [_lin(c) for c in stops]
        s = "  var %s = vec3<f32>(%.5f, %.5f, %.5f);\n" % ((var,) + lin[0])
        for i in range(1, len(STOPS_X)):
            s += "  %s = mix(%s, vec3<f32>(%.5f, %.5f, %.5f), smoothstep(%.3f, %.3f, x));\n" % (
                (var, var) + lin[i] + (STOPS_X[i - 1], STOPS_X[i]))
        return s
    return ("fn delta_color(x: f32, pos: bool) -> vec3<f32> {\n" + side("cp", POS_RGB) + side("cn", NEG_RGB)
            + "  return select(cn, cp, pos);\n}\n")


# ---------------------------------------------------------------------------------------------------- grids
class DepthGridProvider:
    """Depth (m) of any (storm, integer hour) on the flow frame, cached as uint16 centimetres (LRU)."""

    def __init__(self, source, frame, cache: int = 8):
        self.src = source
        self.frame = frame
        self._lru: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._max = cache
        self._lock = threading.Lock()
        self._maps = {}

    def key(self, storm: str, hour: int):
        return (storm, int(hour), self.src.source_key(storm))

    def grid(self, storm: str, hour: int) -> np.ndarray:
        hour = int(min(max(hour, 0), 24))
        k = self.key(storm, hour)
        with self._lock:
            cm = self._lru.get(k)
            if cm is not None:
                self._lru.move_to_end(k)
        if cm is None:
            d = self._compute(storm, hour)
            cm = np.minimum(np.rint(d * 100.0), 65534).astype(np.uint16)
            with self._lock:
                self._lru[k] = cm
                while len(self._lru) > self._max:
                    self._lru.popitem(last=False)
        return np.multiply(cm, np.float32(0.01), dtype=np.float32)   # one float32 allocation, not two

    def _compute(self, storm: str, hour: int) -> np.ndarray:
        if self.src.uses_tiles(storm):
            return self._from_tiles(storm, hour)
        st = self.src.grid.storm_obj(storm)
        if st.frame != self.frame:
            raise ValueError(f"storm {storm} is on another grid than the view: no difference layer for it")
        if st.is_zero(hour):
            return np.zeros((self.frame.height, self.frame.width), np.float32)
        a = st.load_hour(hour)
        t = a[..., 0].astype(np.float32) / 255.0
        d = t * t * np.float32(st.depth_max)
        d[(a[..., 3] == 0) | (d < st.wet_threshold)] = 0.0
        return d

    def _tile_map(self, raster, factor: int):
        """Overview-pixel coordinates (float32 [2, H, W]) of every flow cell centre (cached per tile geometry)."""
        i = raster.info()
        key = (i["left"], i["top"], i["resx"], factor, i["width"], i["height"])
        m = self._maps.get(key)
        if m is None:
            f = self.frame
            cols = (np.arange(f.width) + 0.5) * f.cell_m + f.xmin
            rows = f.ymax - (np.arange(f.height) + 0.5) * f.cell_m
            gx = ((cols - i["left"]) / (i["resx"] * factor) - 0.5).astype(np.float32)
            gy = ((i["top"] - rows) / (i["resy"] * factor) - 0.5).astype(np.float32)
            m = (gx, gy)
            self._maps = {key: m}                         # one geometry at a time
        return m

    def _from_tiles(self, storm: str, hour: int) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        ts = self.src.tiles.storm(storm)
        r = ts.raster(hour)
        if r is None:
            return np.zeros((self.frame.height, self.frame.width), np.float32)
        i = r.info()
        factor = max(1, int(round(self.frame.cell_m / i["resx"])))
        ov = np.nan_to_num(r.read_overview(factor), nan=0.0)
        gx, gy = self._tile_map(r, factor)
        # separable bilinear resample (rows then columns): cheaper and exact for a regular grid
        d = _bilinear_resample(ov, gx, gy)
        d[d < WET_M] = 0.0
        return d


def _bilinear_resample(a: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """a[h, w] sampled at pixel-centre coordinates (gx[W'], gy[H']) -> [H', W'] bilinear; out-of-range -> 0 (outside the tile)."""
    h, w = a.shape
    x0 = np.floor(gx).astype(np.int64)
    tx = (gx - x0).astype(np.float32)
    y0 = np.floor(gy).astype(np.int64)
    ty = (gy - y0).astype(np.float32)
    okx = (gx >= -0.5) & (gx <= w - 0.5)
    oky = (gy >= -0.5) & (gy <= h - 0.5)
    xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    top = a[ya][:, xa] * (1 - tx)[None, :] + a[ya][:, xb] * tx[None, :]
    bot = a[yb][:, xa] * (1 - tx)[None, :] + a[yb][:, xb] * tx[None, :]
    out = top * (1 - ty)[:, None] + bot * ty[:, None]
    out[~oky, :] = 0.0
    out[:, ~okx] = 0.0
    return out.astype(np.float32)


def delta_pack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float16[H, W, 2]: (B - A, max(A, B)) - the rg16float texture of the difference layer."""
    out = np.empty(a.shape + (2,), np.float16)
    out[..., 0] = (b - a)
    out[..., 1] = np.maximum(a, b)
    return out


# ---------------------------------------------------------------------------------------------------- statistics
@dataclass
class CompareStats:
    a_area_km2: float
    b_area_km2: float
    new_wet_km2: float                # wet (>= 2 cm) in B, dry in A
    lost_wet_km2: float               # wet in A, dry in B
    deeper_km2: float                 # B deeper than A by > 2 cm (includes newly wet)
    shallower_km2: float
    max_added_m: float
    max_less_m: float
    mean_delta_m: float               # over the cells wet in either state
    volume_delta_m3: float
    a_volume_m3: float
    b_volume_m3: float

    def as_lines(self) -> list[str]:
        return [f"Flooded area: A {self.a_area_km2:.2f} km2, B {self.b_area_km2:.2f} km2",
                f"Newly wet in B: {self.new_wet_km2:.2f} km2   dry again in B: {self.lost_wet_km2:.2f} km2",
                f"Deeper in B: {self.deeper_km2:.2f} km2   shallower in B: {self.shallower_km2:.2f} km2",
                f"Max added depth {self.max_added_m:.2f} m   max reduction {self.max_less_m:.2f} m",
                f"Volume change {self.volume_delta_m3 / 1e6:+.2f} million m3"]


def compare_grids(a: np.ndarray, b: np.ndarray, frame) -> CompareStats:
    cg2 = frame.cell_ground_m ** 2
    wa, wb = a >= WET_M, b >= WET_M
    d = b - a
    either = wa | wb
    km2 = lambda n: float(n) * cg2 / 1e6                   # noqa: E731
    mean = float(d[either].mean()) if either.any() else 0.0
    return CompareStats(km2(wa.sum()), km2(wb.sum()), km2((wb & ~wa).sum()), km2((wa & ~wb).sum()),
                        km2((d > FLAT_M).sum()), km2((d < -FLAT_M).sum()), float(max(d.max(), 0.0)), float(max(-d.min(), 0.0)),
                        mean, float(d.sum(dtype=np.float64)) * cg2, float(a.sum(dtype=np.float64)) * cg2,
                        float(b.sum(dtype=np.float64)) * cg2)


def tile_compare_stats(ra, rb, frame, band_rows: int = 512) -> CompareStats:
    """The same statistics as `compare_grids`, but at the native tile resolution: two hourly tile rasters of the SAME geometry read
    band by band (no resampling; cell area = (res x k(row))^2 as in timeline.row_cell_area_m2).  About 2 s per raster."""
    from .timeline import row_cell_area_m2
    ia, ib = ra.info(), rb.info()
    for k in ("width", "height", "left", "top", "resx", "resy"):
        if abs(ia[k] - ib[k]) > 1e-6:
            raise ValueError("the two tile rasters are on different grids")
    H = ia["height"]
    acc = dict(a=0.0, b=0.0, new=0.0, lost=0.0, deeper=0.0, shallower=0.0, vol=0.0, va=0.0, vb=0.0, dsum=0.0, wsum=0.0)
    mx = mn = 0.0
    for r0 in range(0, H, band_rows):
        r1 = min(r0 + band_rows, H)
        a = np.nan_to_num(ra.read_rows(r0, r1), nan=0.0)
        b = np.nan_to_num(rb.read_rows(r0, r1), nan=0.0)
        if not (a.any() or b.any()):
            continue
        ca = row_cell_area_m2(ra, frame, r0, r1)
        wa, wb = a >= WET_M, b >= WET_M
        d = b - a
        either = wa | wb
        rows = lambda m: float((m.sum(axis=1) * ca).sum())            # noqa: E731
        acc["a"] += rows(wa)
        acc["b"] += rows(wb)
        acc["new"] += rows(wb & ~wa)
        acc["lost"] += rows(wa & ~wb)
        acc["deeper"] += rows(d > FLAT_M)
        acc["shallower"] += rows(d < -FLAT_M)
        acc["vol"] += float((d.sum(axis=1, dtype=np.float64) * ca).sum())
        acc["va"] += float((a.sum(axis=1, dtype=np.float64) * ca).sum())
        acc["vb"] += float((b.sum(axis=1, dtype=np.float64) * ca).sum())
        acc["dsum"] += float((np.where(either, d, 0.0).sum(axis=1, dtype=np.float64) * ca).sum())
        acc["wsum"] += rows(either)
        mx = max(mx, float(d.max()))
        mn = min(mn, float(d.min()))
    return CompareStats(acc["a"] / 1e6, acc["b"] / 1e6, acc["new"] / 1e6, acc["lost"] / 1e6, acc["deeper"] / 1e6,
                        acc["shallower"] / 1e6, max(mx, 0.0), max(-mn, 0.0), acc["dsum"] / acc["wsum"] if acc["wsum"] else 0.0,
                        acc["vol"], acc["va"], acc["vb"])


@dataclass
class StateStats:
    storm: str
    hour: float
    flooded: int
    n_buildings: int
    road_km: float
    label: str
    flooded_ids: np.ndarray            # building ids flooded (>= 0.1 m)


class ScenePoints:
    """Building footprint samples and road samples as longitude / latitude, built once per scene."""

    def __init__(self, buildings, roads, frame):
        from .timeline import building_points
        X, Y, ids = building_points(buildings)
        self.b_lng_t, self.b_lat_t = geodesy.world_to_lnglat(frame, X, Y)
        self.b_ids_t = ids.astype(np.int64)
        pix = np.asarray(buildings.index.pixels, np.int64)
        cnt = np.diff(np.asarray(buildings.index.offsets, np.int64))
        ids_g = np.repeat(np.arange(buildings.n, dtype=np.int64), cnt)
        cols, rows = pix % frame.width, pix // frame.width
        xm = frame.xmin + (cols + 0.5) * frame.cell_m
        ym = frame.ymax - (rows + 0.5) * frame.cell_m
        self.b_lng_g, self.b_lat_g = geodesy.merc_to_lnglat(xm, ym)
        self.b_ids_g = ids_g
        self.n = int(buildings.n)
        self.r_lng, self.r_lat = geodesy.world_to_lnglat(frame, np.asarray(roads.sample_x, np.float64),
                                                         np.asarray(roads.sample_y, np.float64))
        self.r_w = np.asarray(roads.sample_weight_m, np.float64)
        self.road_total_km = float(roads.total_length_m) / 1000.0


def state_stats(source, storm: str, hour: float, pts: ScenePoints) -> StateStats:
    """Flooded buildings (depth >= 0.1 m under the statistics rule of the backend) and wet road length of one (storm, hour)."""
    backend = source.source_key(storm)
    if backend == "tiles":
        lng, lat, ids = pts.b_lng_t, pts.b_lat_t, pts.b_ids_t
    else:
        lng, lat, ids = pts.b_lng_g, pts.b_lat_g, pts.b_ids_g
    d = source.depth_many(lng, lat, storm, hour, method="nearest")
    bmax = np.zeros(pts.n, np.float64)
    np.maximum.at(bmax, ids, d)
    fl = bmax >= FLOOD_M
    dr = source.depth_many(pts.r_lng, pts.r_lat, storm, hour, method="nearest")
    road_km = float(pts.r_w[dr >= FLOOD_M].sum()) / 1000.0
    return StateStats(storm, float(hour), int(fl.sum()), pts.n, road_km, source.label(storm), np.flatnonzero(fl))


class CompareJob:
    """Background statistics of the current compare pair (one thread, newest request wins)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._req = None
        self._stop = False
        self.result = None            # dict | None
        self.key = None
        self.busy = False
        self.error = ""
        self._th = threading.Thread(target=self._run, name="bgv-compare", daemon=True)
        self._th.start()

    def request(self, key, fn):
        with self._cv:
            if key == self.key and (self.result is not None or self.busy):
                return
            self.key = key
            self.result = None
            self._req = (key, fn)
            self._cv.notify()

    def _run(self):
        try:
            from ..gpu.bgthread import lower_priority
            lower_priority()
        except Exception:
            pass
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                (key, fn), self._req = self._req, None
                self.busy = True
            try:
                res = fn()
                with self._cv:
                    if key == self.key:
                        self.result = res
                        self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"[:200]
            finally:
                self.busy = False

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
